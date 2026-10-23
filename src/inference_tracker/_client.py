"""Collects records in memory and sends them to the ingest endpoint in batches.

    inference_tracker.init(url=..., key=..., app="myapp")   # at startup, or before each request: see init()
    inference_tracker.record(record)                         # the wrappers call this; never blocks, never raises
    inference_tracker.flush()                                # send what's waiting now

Two ways to run:
- background=True (default): a thread sends every few seconds, or as soon as a batch is full.
  For processes that keep running between requests (our fal apps, servers).
- background=False: no thread; records go out only on flush(). For runtimes that freeze the
  process once the response is sent (Cloud Functions, Cloud Run's default, Lambda), where a
  background thread can't be relied on. Use context(..., flush=True) to flush at the end of the work.

flush() runs in the app's request, so it is short: one try per batch, for at most flush_timeout
seconds (1 by default), and no waiting on a busy server (429, Retry-After, backoff): what isn't
through stays queued for the next flush. After a flush that didn't get everything through, flushes
are skipped for flush_pause seconds (60 by default), so a slow or down server costs one request a
second, not every request. The background thread and shutdown() keep retrying with growing waits.

Failures never reach the app. Every record ends up either delivered or in the log:
- A busy or broken server (429, 5xx, no answer) is retried with growing waits (or, in flush(), on
  the next flush). A batch that still isn't through when time runs out stays at the front of the
  queue for the next try, and is written once to the "inference_tracker" log as JSON lines
  (`inference_tracker.unsent {...}`).
- A refused batch (any other 4xx) won't be accepted on a retry: it is logged and dropped.
- When the queue is full (the server has been down a long time), the oldest record is logged and dropped.
Every batch tells the server how many records weren't delivered since the last one that got
through (`lost`). Each record has its own id, so a record sent twice is stored once.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from . import _transport as transport
from ._record import Record
from ._version import __version__

logger = logging.getLogger("inference_tracker")

_SDK = {"name": "inference-tracker-python", "version": __version__}


class Client:
    def __init__(
        self,
        url: str,
        key: str,
        app: str,
        environment: str | None = None,
        *,
        background: bool = True,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue: int = 10_000,
        retry_for: float = 60.0,
        request_timeout: float = 10.0,
        first_retry_delay: float = 1.0,
        flush_timeout: float = 1.0,
        flush_pause: float = 60.0,
        post: Callable[..., transport.Result] = transport.post,
    ):
        self.url, self.key, self.app, self.environment = url, key, app, environment
        self.batch_size, self.flush_interval, self.max_queue = batch_size, flush_interval, max_queue
        self.retry_for, self.request_timeout, self.first_retry_delay = retry_for, request_timeout, first_retry_delay
        self.flush_timeout, self.flush_pause = flush_timeout, flush_pause
        self._post = post
        self._paused_until = 0.0  # time.monotonic() before which flush() doesn't try: the last one failed
        self._queue: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()  # guards the queue and the counter
        self.lost = 0  # records not delivered since the last batch that got through (reported, then reset)
        self._logged: set[str] = set()  # ids already written to the log, so a retried batch isn't logged twice
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if background:
            self._thread = threading.Thread(target=self._run, name="inference_tracker", daemon=True)
            self._thread.start()

    # --- adding -------------------------------------------------------------------------

    def record(self, record: Record) -> None:
        try:
            wire = record.to_wire()
            with self._lock:
                overflow = [self._queue.popleft()] if len(self._queue) >= self.max_queue else []  # oldest first
                self._queue.append(wire)
                full = len(self._queue) >= self.batch_size
            self._drop(overflow)
            if full:
                self._wake.set()
        except Exception:
            logger.exception("inference_tracker: could not queue a record")

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    # --- sending ------------------------------------------------------------------------

    def flush(self, timeout: float | None = None) -> bool:
        """Sends what's waiting, in the app's time: one try per batch, for at most `timeout` seconds
        (flush_timeout when None), never waiting on a busy server. True when nothing is left. After a
        flush that didn't get everything through, flushes are skipped (False) for flush_pause seconds:
        the records stay queued for the next one."""
        try:
            if self.paused():
                return self.pending() == 0
            deadline = time.monotonic() + (self.flush_timeout if timeout is None else timeout)
            done = self._drain(deadline=deadline, mode="flush")
            if not done:
                with self._lock:
                    self._paused_until = time.monotonic() + self.flush_pause
            return done
        except Exception:
            logger.exception("inference_tracker: flush failed")
            return False

    def paused(self) -> bool:
        """Whether flush() skips sending for now, after one that didn't get everything through."""
        with self._lock:
            return time.monotonic() < self._paused_until

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stops the thread and sends what's left, retrying for up to `timeout` seconds."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._drain(deadline=time.monotonic() + timeout, mode="shutdown")
        except Exception:
            logger.exception("inference_tracker: sending at shutdown failed")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.flush_interval)
            self._wake.clear()
            if self._stop.is_set():
                return  # shutdown() sends the rest
            try:
                self._drain(deadline=time.monotonic() + self.retry_for, mode="background")
            except Exception:
                logger.exception("inference_tracker: background send failed")

    def _drain(self, deadline: float, mode: str) -> bool:
        """mode: "flush" (one try per batch, no waiting), "background" (retries, stops on shutdown) or
        "shutdown" (retries until the deadline)."""
        while True:
            if _left(deadline) <= 0:
                return self.pending() == 0  # out of time: what's left stays queued
            batch = self._take()
            if not batch:
                return True
            if not self._send(batch, deadline, mode):
                # Not sent in time: log it (once) and keep it at the front for the next try.
                self._log_unsent(batch)
                self._put_back(batch)
                return False

    def _send(self, batch: list[dict[str, Any]], deadline: float, mode: str) -> bool:
        """True when the batch is done with (sent, or refused for good); False when time ran out (or, in a
        flush, when the first try didn't get through)."""
        delay = self.first_retry_delay
        while True:
            with self._lock:
                lost = self.lost
            body = {"sdk": _SDK, "app": self.app, "environment": self.environment, "lost": lost, "records": batch}
            result = self._post(self.url, self.key, body, min(self.request_timeout, _left(deadline)))
            if result.outcome == "sent":
                with self._lock:
                    self.lost -= lost
                    self._logged.difference_update(w["id"] for w in batch)
                    self._paused_until = 0.0  # the server is back: flushes try again
                return True
            if result.outcome == "rejected":
                logger.warning(
                    "inference_tracker: ingest refused a batch of %d records (HTTP %s)", len(batch), result.status
                )
                self._log_unsent(batch)
                with self._lock:
                    self.lost += len(batch)
                    self._logged.difference_update(w["id"] for w in batch)
                return True
            if mode == "flush":
                return False  # in the app's request: never wait on a busy server, the next flush tries again
            wait = result.retry_after if result.retry_after is not None else delay * random.uniform(1.0, 1.5)
            if wait > _left(deadline):
                return False
            if mode == "background":
                if self._stop.wait(wait):  # shutting down: hand the batch back to shutdown's flush
                    return False
            else:
                time.sleep(wait)
            delay = min(delay * 2, 30.0)

    # --- queue helpers ------------------------------------------------------------------

    def _take(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._queue.popleft() for _ in range(min(self.batch_size, len(self._queue)))]

    def _put_back(self, batch: list[dict[str, Any]]) -> None:
        with self._lock:
            self._queue.extendleft(reversed(batch))
            overflow = [self._queue.popleft() for _ in range(max(0, len(self._queue) - self.max_queue))]
        self._drop(overflow)

    def _drop(self, wires: list[dict[str, Any]]) -> None:
        """Records leaving memory without being sent: logged so they can be recovered, and counted."""
        if wires:
            self._log_unsent(wires)
            with self._lock:
                self.lost += len(wires)
                self._logged.difference_update(w["id"] for w in wires)

    def _log_unsent(self, wires: list[dict[str, Any]]) -> None:
        with self._lock:
            new = [w for w in wires if w["id"] not in self._logged]
            self._logged.update(w["id"] for w in new)
        for wire in new:
            logger.warning("inference_tracker.unsent %s", json.dumps(wire, separators=(",", ":")))


def _left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


# --- the process-wide client ---------------------------------------------------------------

_client: Client | None = None
_NOT_YET = object()
_settings: Any = _NOT_YET  # what the last init() was given: a client's values, or None for "nothing to send to"
_client_lock = threading.Lock()
# Called by flush() and shutdown() with their deadline, before sending: for wrappers whose provider SDK
# records a call on its own thread a moment after the call returns (litellm), so it is in this flush.
_before_flush: list[Callable[[float], None]] = []


def init(
    url: str | None = None,
    key: str | None = None,
    app: str | None = None,
    environment: str | None = None,
    **options: Any,
) -> Client | None:
    """Starts sending. Values left out come from INFERENCE_TRACKER_URL, INFERENCE_TRACKER_KEY, INFERENCE_TRACKER_APP and
    INFERENCE_TRACKER_ENVIRONMENT. Without a url, key and app nothing is sent and every call stays a no-op.

    Safe to call from several threads at once, and cheap to call again: with the same values it returns the
    client already running, so an app can call it at the start of every request instead of keeping a flag of
    its own. Other values replace the client (the old one sends what it has, without holding up the others)."""
    global _client, _settings
    url = url or os.environ.get("INFERENCE_TRACKER_URL")
    key = key or os.environ.get("INFERENCE_TRACKER_KEY")
    app = app or os.environ.get("INFERENCE_TRACKER_APP")
    environment = environment or os.environ.get("INFERENCE_TRACKER_ENVIRONMENT")
    settings = (url, key, app, environment, options) if url and key and app else None
    with _client_lock:
        if settings == _settings and (settings is None or _client is not None):
            return _client  # the same values again: nothing to do (and, without values, said once already)
        old = _client
        try:
            _client = Client(url, key, app, environment, **options) if settings is not None else None  # type: ignore[arg-type]
        except Exception:
            logger.exception("inference_tracker: could not start; records won't be sent")
            _client = None
        _settings = settings if _client is not None else None
        new = _client
    if settings is None:
        logger.warning("inference_tracker: no url, key or app set; records won't be sent")
    if old is not None:
        old.shutdown()  # outside the lock: calls that come meanwhile already see the new client
    return new


def active() -> bool:
    """Whether records are being kept: the wrappers do nothing (read no answer, parse no body) when not."""
    return _client is not None


def record(r: Record) -> None:
    client = _client
    if client is not None:
        client.record(r)


def flush(timeout: float | None = None) -> bool:
    """Sends what's waiting, briefly: see Client.flush. `timeout` defaults to init's flush_timeout (1 second)."""
    client = _client
    if client is None:
        return True
    timeout = client.flush_timeout if timeout is None else timeout
    if not client.paused():
        started = time.monotonic()
        _wait_for_late_records(started + timeout / 2)  # half of the time at most, the rest is for sending
        timeout = max(timeout / 2, timeout - (time.monotonic() - started))
    return client.flush(timeout)


def shutdown(timeout: float = 5.0) -> None:
    client = _client
    if client is not None:
        _wait_for_late_records(time.monotonic() + timeout)
        client.shutdown(timeout)


def _wait_for_late_records(deadline: float) -> None:
    for hook in list(_before_flush):
        try:
            hook(deadline)
        except Exception:
            logger.exception("inference_tracker: waiting for records before a flush failed")


atexit.register(shutdown)
