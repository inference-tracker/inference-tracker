"""Sees the answers an HTTP client gets, without changing them: for apps that call a provider over plain HTTP.

    _http.watch(session, watcher)   # an httpx.Client, httpx.AsyncClient or requests.Session

The watcher's hooks are added to the session itself (httpx's event_hooks, requests' hooks), so the app keeps
its own session object, of its own type, and every request made with it is seen: including those made by
code the app doesn't own that uses the same session. A request to a host the watcher doesn't accept is left
alone: nothing is read, timed or kept.

The body of an answer is read only when the watcher asks for it (wants_body), the way httpx and requests let
a hook read it: what the app reads afterwards is the same bytes. An error while reading is the call's own
(the connection broke) and reaches the app as it would have; an error in the watcher is logged, never raised.

A request that gets no answer at all (a timeout, a refused connection, the app cancelling it) raises in the
session's send(), which no hook sees: send() is wrapped on the session object itself, so the watcher's
failed() hears of it, and the error is raised as it was.

Without an active client (init() not called, or without a key) the hooks return at once: nothing is read.
"""

from __future__ import annotations

import contextvars
import inspect
import json
import logging
import threading
import time
import weakref
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from . import _client
from ._record import error_name

logger = logging.getLogger("inference_tracker")


class Exchange:
    """One request and the answer to it, as a hook sees them. For a request that got no answer (see failed()),
    `response` is None: status 0, no headers, and `error` the exception."""

    def __init__(
        self,
        request: Any,
        response: Any,
        url: str,
        started: tuple[float, float],
        elapsed: float,
        error: BaseException | None = None,
    ):
        parts = urlsplit(url)
        self.method = str(request.method).upper()
        self.host = (parts.hostname or "").lower()
        self.path = parts.path
        self.query = parse_qs(parts.query)
        self.request_headers: Mapping[str, str] = request.headers  # case-insensitive in httpx and requests
        self.status: int = response.status_code if response is not None else 0
        self.headers: Mapping[str, str] = response.headers if response is not None else {}
        self.started_at, self.clock = started  # unix seconds and time.monotonic() when the request was sent
        self.duration_s = round(elapsed, 3)  # until the answer's headers came (or the error)
        self.body: Any = None  # the answer's JSON, when the watcher asked for it
        self.error = error  # why there is no answer
        self._request = request

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def request_json(self) -> Any:
        """The request's JSON body, None when it has none (or isn't at hand, like a streamed upload)."""
        try:
            content = getattr(self._request, "content", None)  # httpx
            if content is None:
                content = getattr(self._request, "body", None)  # requests
            return json.loads(content) if isinstance(content, (bytes, str)) and content else None
        except Exception:
            return None


class Watcher(Protocol):
    name: str  # one watcher of each name per session: watching twice adds nothing

    def accepts(self, host: str) -> bool: ...

    def wants_body(self, exchange: Exchange) -> bool: ...

    def seen(self, exchange: Exchange) -> None: ...

    def failed(self, exchange: Exchange) -> None: ...  # the request raised: exchange.error, no answer


def watch(session: Any, watcher: Watcher) -> bool:
    """Adds the watcher's hooks to the session. False when it isn't a session this knows how to watch."""
    try:
        if isinstance(getattr(session, "event_hooks", None), Mapping):
            watched = _watch_httpx(session, watcher)
        elif isinstance(getattr(session, "hooks", None), dict) and hasattr(session, "mount"):
            watched = _watch_requests(session, watcher)
        else:
            return False
        _watch_failures(session, watcher)
        return watched
    except Exception:
        logger.exception("inference_tracker: could not watch an HTTP client")
    return False


_FAILURE_WATCHERS = "_inference_tracker_failure_watchers"
# Inside a watched send(): requests sends each redirect with send() again, and only the outer one reports.
_sending: contextvars.ContextVar[bool] = contextvars.ContextVar("inference_tracker_sending", default=False)


def _watch_failures(session: Any, watcher: Watcher) -> None:
    """Wraps the session's send() (once, for every watcher of the session) to see requests that raise."""
    watchers: list[Watcher] | None = getattr(session, _FAILURE_WATCHERS, None)
    if watchers is not None:
        if all(w.name != watcher.name for w in watchers):
            watchers.append(watcher)
        return
    watchers = [watcher]
    send = session.send
    is_async = inspect.iscoroutinefunction(send)

    def failed(request: Any, started: tuple[float, float], error: BaseException) -> None:
        try:
            if not _client.active():
                return
            url = str(request.url)
            host = (urlsplit(url).hostname or "").lower()
            for w in list(watchers):
                if w.accepts(host):
                    ex = Exchange(request, None, url, started, time.monotonic() - started[1], error)
                    try:
                        w.failed(ex)
                    except Exception:
                        logger.exception("inference_tracker: tracking a failed HTTP call failed")
        except Exception:
            logger.exception("inference_tracker: could not read a failed HTTP request")

    if is_async:

        async def send_async(request: Any, *args: Any, **kwargs: Any) -> Any:
            started = (time.time(), time.monotonic())
            token = _sending.set(True)
            try:
                return await send(request, *args, **kwargs)
            except BaseException as e:
                if token.old_value is not True:
                    failed(request, started, e)
                raise
            finally:
                _sending.reset(token)

        session.send = send_async
    else:

        def send_sync(request: Any, *args: Any, **kwargs: Any) -> Any:
            started = (time.time(), time.monotonic())
            token = _sending.set(True)
            try:
                return send(request, *args, **kwargs)
            except BaseException as e:
                if token.old_value is not True:
                    failed(request, started, e)
                raise
            finally:
                _sending.reset(token)

        session.send = send_sync
    setattr(session, _FAILURE_WATCHERS, watchers)


def error_type(ex: Exchange) -> str | None:
    """A failed request's error type: the exception's class name. None for an answered one."""
    return error_name(ex.error) if ex.error is not None else None


def _watch_httpx(session: Any, watcher: Watcher) -> bool:
    hooks = session.event_hooks
    if any(getattr(h, "_inference_tracker", None) == watcher.name for h in hooks.get("response", [])):
        return True
    is_async = hasattr(session, "aclose")  # httpx.AsyncClient
    sent: weakref.WeakKeyDictionary[Any, tuple[float, float]] = weakref.WeakKeyDictionary()
    lock = threading.Lock()

    def on_request(request: Any) -> None:
        try:
            if _client.active() and watcher.accepts((request.url.host or "").lower()):
                with lock:
                    sent[request] = (time.time(), time.monotonic())
        except Exception:
            logger.exception("inference_tracker: could not time an HTTP request")

    def exchange(response: Any) -> tuple[Exchange | None, bool]:
        """The exchange, and whether the watcher wants the answer's body."""
        try:
            request = response.request
            if not _client.active() or not watcher.accepts((request.url.host or "").lower()):
                return None, False
            with lock:
                started = sent.pop(request, None)
            now = time.monotonic()
            started = started or (time.time(), now)
            ex = Exchange(request, response, str(request.url), started, now - started[1])
            return ex, watcher.wants_body(ex)
        except Exception:
            logger.exception("inference_tracker: could not read an HTTP answer")
            return None, False

    def on_response(response: Any) -> None:
        ex, wanted = exchange(response)
        if ex is not None:
            if wanted:
                response.read()
                ex.body = _json(response.content)
            _seen(watcher, ex)

    async def on_request_async(request: Any) -> None:
        on_request(request)

    async def on_response_async(response: Any) -> None:
        ex, wanted = exchange(response)
        if ex is not None:
            if wanted:
                await response.aread()
                ex.body = _json(response.content)
            _seen(watcher, ex)

    request_hook, response_hook = (on_request_async, on_response_async) if is_async else (on_request, on_response)
    response_hook._inference_tracker = watcher.name  # type: ignore[attr-defined]
    session.event_hooks = {
        **hooks,
        "request": [*hooks.get("request", []), request_hook],
        "response": [*hooks.get("response", []), response_hook],
    }
    return True


def _watch_requests(session: Any, watcher: Watcher) -> bool:
    hooks = session.hooks.setdefault("response", [])
    if not isinstance(hooks, list):  # requests allows a single callable here
        hooks = session.hooks["response"] = [hooks]
    if any(getattr(h, "_inference_tracker", None) == watcher.name for h in hooks):
        return True

    def on_response(response: Any, *args: Any, **kwargs: Any) -> None:
        try:
            request = response.request
            if not _client.active() or not watcher.accepts((urlsplit(request.url).hostname or "").lower()):
                return None
            elapsed = response.elapsed.total_seconds()  # requests times until the headers came
            ex = Exchange(request, response, request.url, (time.time() - elapsed, time.monotonic() - elapsed), elapsed)
            wanted = watcher.wants_body(ex)
        except Exception:
            logger.exception("inference_tracker: could not read an HTTP answer")
            return None
        if wanted:
            ex.body = _json(response.content)
        _seen(watcher, ex)
        return None  # a hook that returns something replaces the response

    on_response._inference_tracker = watcher.name  # type: ignore[attr-defined]
    hooks.append(on_response)
    return True


def _seen(watcher: Watcher, ex: Exchange) -> None:
    try:
        watcher.seen(ex)
    except Exception:
        logger.exception("inference_tracker: tracking an HTTP call failed")


def _json(content: Any) -> Any:
    try:
        return json.loads(content) if content else None
    except Exception:
        return None


def header_number(headers: Mapping[str, str], name: str) -> float | None:
    try:
        value = headers.get(name)
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
