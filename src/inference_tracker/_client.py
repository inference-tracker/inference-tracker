"""Collects records in memory and sends them to the ingest endpoint in batches.

    inference_tracker.init(url=..., key=..., app="myapp")   # once per process
    inference_tracker.record(record)                         # the wrappers call this; never blocks, never raises
    inference_tracker.flush()                                # send what's waiting now

Two ways to run:
- background=True (default): a thread sends every few seconds, or as soon as a batch is full.
  For processes that keep running between requests (our fal apps, servers).
- background=False: no thread; records go out only on flush(). For runtimes that freeze the
  process once the response is sent (Cloud Functions, Cloud Run's default, Lambda), where a
  background thread can't be relied on. Use context(..., flush=True) to flush at the end of the work.

Failures never reach the app. Every record ends up either delivered or in the log:
- A busy or broken server (429, 5xx, no answer) is retried with growing waits. A batch that still
  isn't through when time runs out stays at the front of the queue for the next try, and is
  written once to the "inference_tracker" log as JSON lines (`inference_tracker.unsent {...}`).
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
        post: Callable[..., transport.Result] = transport.post,
    ):
        self.url, self.key, self.app, self.environment = url, key, app, environment
        self.batch_size, self.flush_interval, self.max_queue = batch_size, flush_interval, max_queue
        self.retry_for, self.request_timeout, self.first_retry_delay = retry_for, request_timeout, first_retry_delay
        self._post = post
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

    def flush(self, timeout: float = 5.0) -> bool:
        """Sends everything waiting, giving up after `timeout` seconds. True when nothing is left."""
        try:
            return self._drain(deadline=time.monotonic() + timeout, background=False)
        except Exception:
            logger.exception("inference_tracker: flush failed")
            return False

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.flush(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.flush_interval)
            self._wake.clear()
            if self._stop.is_set():
                return  # shutdown() sends the rest
            try:
                self._drain(deadline=time.monotonic() + self.retry_for, background=True)
            except Exception:
                logger.exception("inference_tracker: background send failed")

    def _drain(self, deadline: float, background: bool) -> bool:
        while True:
            batch = self._take()
            if not batch:
                return True
            if not self._send(batch, deadline, background):
                # Not sent in time: log it (once) and keep it at the front for the next try.
                self._log_unsent(batch)
                self._put_back(batch)
                return False

    def _send(self, batch: list[dict[str, Any]], deadline: float, background: bool) -> bool:
        """True when the batch is done with (sent, or refused for good); False when time ran out."""
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
            wait = result.retry_after if result.retry_after is not None else delay * random.uniform(1.0, 1.5)
            if wait > _left(deadline):
                return False
            if background:
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
_client_lock = threading.Lock()


def init(
    url: str | None = None,
    key: str | None = None,
    app: str | None = None,
    environment: str | None = None,
    **options: Any,
) -> Client | None:
    """Starts sending. Values left out come from INFERENCE_TRACKER_URL, INFERENCE_TRACKER_KEY, INFERENCE_TRACKER_APP and
    INFERENCE_TRACKER_ENVIRONMENT. Without a url, key and app nothing is sent and every call stays a no-op."""
    global _client
    url = url or os.environ.get("INFERENCE_TRACKER_URL")
    key = key or os.environ.get("INFERENCE_TRACKER_KEY")
    app = app or os.environ.get("INFERENCE_TRACKER_APP")
    environment = environment or os.environ.get("INFERENCE_TRACKER_ENVIRONMENT")
    with _client_lock:
        old, _client = _client, None
        if old is not None:
            old.shutdown()
        if not (url and key and app):
            logger.warning("inference_tracker: no url, key or app set; records won't be sent")
            return None
        _client = Client(url, key, app, environment, **options)
        return _client


def record(r: Record) -> None:
    client = _client
    if client is not None:
        client.record(r)


def flush(timeout: float = 5.0) -> bool:
    client = _client
    return client.flush(timeout) if client is not None else True


def shutdown(timeout: float = 5.0) -> None:
    client = _client
    if client is not None:
        client.shutdown(timeout)


atexit.register(shutdown)
