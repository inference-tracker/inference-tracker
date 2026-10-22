"""Tracks calls made with fal's Python client (the fal_client package), sync or async.

    import fal_client
    import inference_tracker

    client = inference_tracker.fal.track(fal_client.AsyncClient(key=...))
    handle = await client.submit("fal-ai/kling-video/v2.6/pro/motion-control", arguments=args)
    result = await handle.get()

The tracked client has every method of the one it wraps; what isn't listed below (upload, status,
cancel, stream, realtime...) passes through unchanged. The call itself is never changed. Tracking never
raises: a problem in it is logged and the call goes on as if nothing happened.

What is recorded, one record for each part of a call this process sees:

- submit(): a "start" record once fal has answered with the request id. When submit itself fails,
  one "call" record with status "error" instead.
- The handle submit() returns: a "finish" record when the result is fetched with get(), when it is
  fetched with client.result(application, request_id) for a request this process submitted, or when
  the request is cancelled. status() and iter_events() don't finish a call (only the result says
  whether it worked), but the metrics fal reports with the completed status go on the finish record.
- subscribe(): a "start" record when fal has queued the request, a "finish" record when it ends.
- run(): one "call" record, with the request id from fal's x-fal-request-id response header.
- webhook(body): the "finish" record of a call whose result fal sent to a webhook.

The server joins start and finish on provider + request id. A start without a finish is a call this
process never saw the end of: the result went to a webhook that doesn't call webhook(), or the app
gave up waiting.

Apps that call fal's HTTP API themselves (httpx, requests, or their own methods on a fal_client subclass)
use track_http(client) instead: the same records, read from the requests and answers. See track_http().

Settings: of the arguments, only those named in `settings` are copied, the ones that change the price
(resolution, duration, number of images...). Never the whole input: prompts and media URLs stay out.

Our own fal apps make calls of their own. In the app's endpoint, inside(self) makes the request the app
is serving the parent of every tracked call in the block.

Workflows: submitted with the header WORKFLOW_ENVELOPE_HEADER set to "true" (the app adds it; this module
never does), a fal workflow's result comes back as {output, nodes}, where nodes holds the endpoint and
request id of each step that ran. Each step then gets a "call" record with the run as its parent, from
webhook(body) and from a result fetched here with get(), client.result() or subscribe(). On a failed run
(seen only by webhook(): a failed result raises in fal_client), the step named by node_id failed and the
others that ran worked. A step that fell back to another model also gets a record for the failed attempt.
"""

from __future__ import annotations

import contextvars
import inspect
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from typing import Any

from . import _client, _http
from ._context import Context, context, current
from ._record import Record, error_name

logger = logging.getLogger("inference_tracker")

PROVIDER = "fal"
REQUEST_ID_HEADER = "x-fal-request-id"
# Sent as "true" on submit, it makes a workflow's result {output, nodes}. The app sends it, never this module.
WORKFLOW_ENVELOPE_HEADER = "X-Fal-Workflow-Envelope"

# Arguments copied onto the record because they change the price. Pass settings=[...] to track() for others.
SETTINGS = (
    "resolution",
    "aspect_ratio",
    "duration",
    "num_images",
    "image_size",
    "num_frames",
    "generate_audio",
    "enable_audio",
    "quality",
)

# Requests this process submitted and hasn't seen the end of are remembered by request id, so that
# client.result(application, request_id) can finish them. Beyond this many, the oldest is forgotten.
MAX_OPEN = 1000


def track(client: Any, *, settings: Iterable[str] = SETTINGS) -> Any:
    """Wraps a fal_client.SyncClient or fal_client.AsyncClient. See the module docstring."""
    tracker = _Tracker(settings)
    is_async = inspect.iscoroutinefunction(getattr(client, "submit", None))
    return _AsyncClient(client, tracker) if is_async else _Client(client, tracker)


@contextmanager
def inside(app: Any) -> Generator[Context, None, None]:
    """In one of our own fal apps: every tracked call in the block gets the fal request the app is serving
    as its parent. `app` is the fal.App, so in an endpoint:

        with inference_tracker.fal.inside(self):
            return await self._inference(input)

    The request id is read from app.current_request (as in fal 1.72). When there is none (the app runs
    outside fal, or an older fal), the block runs as if inside() weren't there."""
    request_id = _current_request_id(app)
    if request_id is None:
        logger.debug("inference_tracker: no current fal request, so calls in this block get no parent")
        yield current()
        return
    with context(parent_provider=PROVIDER, parent_request_id=request_id) as ctx:
        yield ctx


def _current_request_id(app: Any) -> str | None:
    try:
        request_id = getattr(getattr(app, "current_request", None), "request_id", None)
    except Exception:
        return None
    return request_id if isinstance(request_id, str) and request_id else None


def webhook(body: Any, *, model: str | None = None) -> Record | None:
    """Records the "finish" of a call whose result fal sent to a webhook. `body` is the webhook's JSON
    body (request_id, status "OK" or "ERROR", payload...). The application isn't in it; pass `model`
    when you know it, otherwise the server takes it from the start record. When the call is a workflow
    run with its steps in the payload, each step gets a record too (see the module docstring). Call it
    inside the user's context when the user is known. Returns the finish record, or None when the body
    has no request id."""
    try:
        request_id = body.get("request_id") if isinstance(body, Mapping) else None
        if not isinstance(request_id, str) or not request_id:
            return None
        ok = body.get("status") == "OK"
        record = Record.new(
            PROVIDER,
            model,
            phase="finish",
            request_id=request_id,
            status="ok" if ok else "error",
            error_type=None if ok else _webhook_error_type(body),
        )
        _client.record(record)
    except Exception:
        logger.exception("inference_tracker: could not record a fal webhook")
        return None
    _safe(_record_steps, body.get("payload"), request_id, error_type=record.error_type)
    return record


def _webhook_error_type(body: Mapping[str, Any]) -> str:
    """fal puts the failure in a `detail` list, [{type, msg, error_code, loc}]: under payload for an app,
    deeper for a workflow (payload.error.body.detail). The code or type is taken, never the message."""
    return _detail_code(body.get("payload")) or "error"


def _detail_code(data: Any) -> str | None:
    first = _first_detail(data)
    if first is not None:
        for key in ("error_code", "type"):
            if isinstance(first.get(key), str) and first[key]:
                return first[key]
    return None


def _first_detail(data: Any, depth: int = 0) -> Mapping[str, Any] | None:
    """The first entry of the first `detail` list found in `data`, looking only through dicts. The depth
    is capped so that a dict that holds itself can't loop."""
    if not isinstance(data, Mapping) or depth > 20:
        return None
    details = data.get("detail")
    if isinstance(details, list) and details and isinstance(details[0], Mapping):
        return details[0]
    for value in data.values():
        if (found := _first_detail(value, depth + 1)) is not None:
            return found
    return None


# --- workflow steps --------------------------------------------------------------------------------


def _record_steps(result: Any, run_id: Any, *, error_type: str | None = None, **fields: Any) -> None:
    """One "call" record for each step in a workflow result's `nodes`, with the run as parent. error_type
    is given for a failed run: it goes on the step named by node_id. `fields` (user, task) go on every record."""
    nodes = result.get("nodes") if isinstance(result, Mapping) else None
    if not isinstance(nodes, Mapping) or not isinstance(run_id, str) or not run_id:
        return
    if error_type is None and "output" not in result:
        return  # a successful run's result has both; anything else is an app's own output
    failed_node = result.get("node_id") if error_type else None
    for key, node in nodes.items():
        if not isinstance(node, Mapping):
            continue
        fallback = node.get("fallback")
        if isinstance(fallback, Mapping):  # the step's first model failed and another one did the step
            reason = fallback.get("reason")
            failed = reason if isinstance(reason, str) and reason else "error"
            _record_step(fallback.get("failed_endpoint"), fallback.get("failed_request_id"), run_id, failed, fields)
        step_error = error_type if key == failed_node else None
        _record_step(node.get("endpoint"), node.get("request_id"), run_id, step_error, fields)


def _record_step(endpoint: Any, request_id: Any, run_id: str, error_type: str | None, fields: dict) -> None:
    """A step without a request id can't be matched to the bill: it isn't recorded. The endpoint is kept as
    fal gives it, case included."""
    if not isinstance(request_id, str) or not request_id:
        return
    _client.record(
        Record.new(
            PROVIDER,
            endpoint if isinstance(endpoint, str) else None,
            request_id=request_id,
            status="error" if error_type else "ok",
            error_type=error_type,
            parent_provider=PROVIDER,
            parent_request_id=run_id,
            **fields,
        )
    )


# --- one call ----------------------------------------------------------------------------------


class _Call:
    """The record of one call while this process follows it."""

    def __init__(
        self,
        application: str,
        arguments: Mapping[str, Any],
        settings: tuple[str, ...],
        started: tuple[float, float] | None = None,  # (unix seconds, time.monotonic()) when it was sent; now if None
    ):
        started_at, self._clock = started or (time.time(), time.monotonic())
        self.record = Record.new(
            PROVIDER,
            application,
            phase="start",
            started_at=started_at,
            settings={k: arguments[k] for k in settings if k in arguments},
        )
        self.units: dict[str, float] = {}  # fal's metrics (inference_seconds) and billing headers
        self.envelope = False  # submitted over plain HTTP asking for a workflow's {output, nodes}
        self.done = False

    def submitted(self, request_id: str) -> None:
        self.record.request_id = request_id
        _client.record(self.record)

    def saw(self, status: Any) -> None:
        """A queue status answer. The completed one carries fal's metrics."""
        if type(status).__name__ == "Completed":
            self.saw_metrics(getattr(status, "metrics", None))

    def saw_metrics(self, metrics: Any) -> None:
        seconds = metrics.get("inference_time") if isinstance(metrics, Mapping) else None
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            self.units["inference_seconds"] = float(seconds)

    def finish(self, error_type: str | None, result: Any = None) -> None:
        """`result` is what fal returned, for the steps of a workflow run."""
        if self.done:
            return
        self.done = True
        start = self.record
        _client.record(
            Record.new(
                PROVIDER,
                start.model,
                phase="finish",
                request_id=start.request_id,
                status="error" if error_type else "ok",
                error_type=error_type,
                duration_s=self._elapsed(),
                units=self._units(),
                user_id=start.user_id,
                task_id=start.task_id,
                parent_provider=start.parent_provider,
                parent_request_id=start.parent_request_id,
            )
        )
        if result is not None:
            _record_steps(result, start.request_id, user_id=start.user_id, task_id=start.task_id)

    def completed(self, request_id: str | None, error_type: str | None) -> None:
        """The whole call seen here (run(), or a submit that failed): one "call" record."""
        if self.done:
            return
        self.done = True
        r = self.record
        r.phase, r.request_id = "call", request_id
        r.status, r.error_type = ("error" if error_type else "ok"), error_type
        r.duration_s, r.units = self._elapsed(), self._units()
        _client.record(r)

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._clock, 3)

    def _units(self) -> dict[str, float]:
        return dict(self.units)


class _Tracker:
    """Shared by the sync and async wrappers: starts calls and remembers the open ones."""

    def __init__(self, settings: Iterable[str]):
        self.settings = tuple(settings)
        self._open: OrderedDict[str, _Call] = OrderedDict()
        self._lock = threading.Lock()

    def start(
        self,
        application: str,
        arguments: Any,
        *,
        settings: Iterable[str] | None = None,
        started: tuple[float, float] | None = None,
    ) -> _Call | None:
        """Before the call: its record."""
        try:
            arguments = arguments if isinstance(arguments, Mapping) else {}
            return _Call(application, arguments, self.settings if settings is None else tuple(settings), started)
        except Exception:
            logger.exception("inference_tracker: could not start tracking a fal call")
            return None

    def submitted(self, call: _Call | None, request_id: str, *, keep: bool) -> None:
        if call is None:
            return
        call.submitted(request_id)
        if keep:
            with self._lock:
                self._open[request_id] = call
                while len(self._open) > MAX_OPEN:
                    self._open.popitem(last=False)

    def take(self, request_id: Any) -> _Call | None:
        with self._lock:
            return self._open.pop(request_id, None)

    def peek(self, request_id: Any) -> _Call | None:
        with self._lock:
            return self._open.get(request_id)

    def finish(self, call: _Call | None, error_type: str | None, result: Any = None) -> None:
        if call is not None:
            self.take(call.record.request_id)
            call.finish(error_type, result)


# --- the wrappers ----------------------------------------------------------------------------------


class _Handle:
    """Stands in for fal's request handle: same attributes and methods, and the finish gets recorded."""

    def __init__(self, handle: Any, call: _Call | None, tracker: _Tracker):
        self._handle, self._call, self._tracker = handle, call, tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def status(self, **kwargs: Any) -> Any:
        status = self._handle.status(**kwargs)
        self._saw(status)
        return status

    def iter_events(self, **kwargs: Any) -> Any:
        for status in self._handle.iter_events(**kwargs):
            self._saw(status)
            yield status

    def get(self, **kwargs: Any) -> Any:
        try:
            result = self._handle.get(**kwargs)
        except BaseException as e:
            _safe(self._tracker.finish, self._call, _error_type(e))
            raise
        _safe(self._tracker.finish, self._call, None, result)
        return result

    def cancel(self) -> None:
        self._handle.cancel()
        _safe(self._tracker.finish, self._call, "cancelled")

    def _saw(self, status: Any) -> None:
        if self._call is not None:
            _safe(self._call.saw, status)


class _AsyncHandle(_Handle):
    async def status(self, **kwargs: Any) -> Any:
        status = await self._handle.status(**kwargs)
        self._saw(status)
        return status

    async def iter_events(self, **kwargs: Any) -> Any:
        async for status in self._handle.iter_events(**kwargs):
            self._saw(status)
            yield status

    async def get(self, **kwargs: Any) -> Any:
        try:
            result = await self._handle.get(**kwargs)
        except BaseException as e:
            _safe(self._tracker.finish, self._call, _error_type(e))
            raise
        _safe(self._tracker.finish, self._call, None, result)
        return result

    async def cancel(self) -> None:
        await self._handle.cancel()
        _safe(self._tracker.finish, self._call, "cancelled")


class _Client:
    """Stands in for fal_client.SyncClient."""

    _handle_class: type[_Handle] = _Handle

    def __init__(self, client: Any, tracker: _Tracker):
        self._client, self._tracker = client, tracker
        self._watching = False  # whether the response hook that reads request ids for run() is in place

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def submit(self, application: str, arguments: Any, **kwargs: Any) -> Any:
        call = self._tracker.start(application, arguments)
        try:
            handle = self._client.submit(application, arguments, **kwargs)
        except BaseException as e:
            self._failed(call, e)
            raise
        # A result that goes to a webhook is finished by webhook(), not by this process.
        self._submitted(call, handle, keep=kwargs.get("webhook_url") is None)
        return self._handle_class(handle, call, self._tracker)

    def subscribe(self, application: str, arguments: Any, **kwargs: Any) -> Any:
        call = self._tracker.start(application, arguments)
        kwargs["on_enqueue"] = self._on_enqueue(call, kwargs.get("on_enqueue"))
        try:
            result = self._client.subscribe(application, arguments, **kwargs)
        except BaseException as e:
            self._failed(call, e)
            raise
        _safe(self._tracker.finish, call, None, result)
        return result

    def run(self, application: str, arguments: Any, **kwargs: Any) -> Any:
        call = self._tracker.start(application, arguments)
        if not self._watching:
            self._watching = True
            _watch_request_ids(getattr(self._client, "_client", None), is_async=False)
        token = _seen_request_id.set(None)
        try:
            result = self._client.run(application, arguments, **kwargs)
        except BaseException as e:
            self._ran(call, token, _error_type(e))
            raise
        self._ran(call, token, None)
        return result

    def result(self, application: str, request_id: str) -> Any:
        call = self._tracker.take(request_id)
        try:
            result = self._client.result(application, request_id)
        except BaseException as e:
            _safe(self._tracker.finish, call, _error_type(e))
            raise
        _safe(self._tracker.finish, call, None, result)
        return result

    def get_handle(self, application: str, request_id: str) -> Any:
        handle = self._client.get_handle(application, request_id)
        return self._handle_class(handle, self._tracker.take(request_id), self._tracker)

    def cancel(self, application: str, request_id: str) -> Any:
        result = self._client.cancel(application, request_id)
        _safe(self._tracker.finish, self._tracker.take(request_id), "cancelled")
        return result

    # --- shared steps ---

    def _submitted(self, call: _Call | None, handle: Any, *, keep: bool) -> None:
        _safe(self._tracker.submitted, call, getattr(handle, "request_id", None), keep=keep)

    def _on_enqueue(self, call: _Call | None, user_callback: Callable[[str], None] | None) -> Callable[[str], None]:
        def on_enqueue(request_id: str) -> None:
            _safe(self._tracker.submitted, call, request_id, keep=False)
            if user_callback is not None:
                user_callback(request_id)

        return on_enqueue

    def _failed(self, call: _Call | None, error: BaseException) -> None:
        """subscribe() or submit() raised: a call that never got a request id is one failed "call"
        record; one that did is finished with the error."""
        if call is None:
            return
        if call.record.request_id is None:
            _safe(call.completed, None, _error_type(error))
        else:
            _safe(self._tracker.finish, call, _error_type(error))

    def _ran(self, call: _Call | None, token: contextvars.Token, error_type: str | None) -> None:
        request_id = _seen_request_id.get()
        _seen_request_id.reset(token)
        if call is not None:
            _safe(call.completed, request_id, error_type)


class _AsyncClient(_Client):
    """Stands in for fal_client.AsyncClient."""

    _handle_class = _AsyncHandle

    async def submit(self, application: str, arguments: Any, **kwargs: Any) -> Any:
        call = self._tracker.start(application, arguments)
        try:
            handle = await self._client.submit(application, arguments, **kwargs)
        except BaseException as e:
            self._failed(call, e)
            raise
        self._submitted(call, handle, keep=kwargs.get("webhook_url") is None)
        return self._handle_class(handle, call, self._tracker)

    async def subscribe(self, application: str, arguments: Any, **kwargs: Any) -> Any:
        call = self._tracker.start(application, arguments)
        kwargs["on_enqueue"] = self._on_enqueue(call, kwargs.get("on_enqueue"))
        try:
            result = await self._client.subscribe(application, arguments, **kwargs)
        except BaseException as e:
            self._failed(call, e)
            raise
        _safe(self._tracker.finish, call, None, result)
        return result

    async def run(self, application: str, arguments: Any, **kwargs: Any) -> Any:
        call = self._tracker.start(application, arguments)
        if not self._watching:
            self._watching = True
            http = getattr(self._client, "_client", None)
            if inspect.isawaitable(http):  # fal's async client builds its session on first use
                try:
                    http = await http
                except Exception:
                    http = None
            _watch_request_ids(http, is_async=True)
        token = _seen_request_id.set(None)
        try:
            result = await self._client.run(application, arguments, **kwargs)
        except BaseException as e:
            self._ran(call, token, _error_type(e))
            raise
        self._ran(call, token, None)
        return result

    async def result(self, application: str, request_id: str) -> Any:
        call = self._tracker.take(request_id)
        try:
            result = await self._client.result(application, request_id)
        except BaseException as e:
            _safe(self._tracker.finish, call, _error_type(e))
            raise
        _safe(self._tracker.finish, call, None, result)
        return result

    async def get_handle(self, application: str, request_id: str) -> Any:
        handle = await self._client.get_handle(application, request_id)
        return self._handle_class(handle, self._tracker.take(request_id), self._tracker)

    async def cancel(self, application: str, request_id: str) -> Any:
        result = await self._client.cancel(application, request_id)
        _safe(self._tracker.finish, self._tracker.take(request_id), "cancelled")
        return result


# --- over plain HTTP ---------------------------------------------------------------------------------

QUEUE_HOSTS = frozenset({"queue.fal.run", "queue.falrun.com"})  # with the backup domain fal_client falls back to
RUN_HOSTS = frozenset({"fal.run", "falrun.com"})
ERROR_TYPE_HEADER = "x-fal-error-type"
BILLABLE_UNITS_HEADER = "x-fal-billable-units"  # fal's own billing count, when an answer carries it
# .../requests/<id>, .../requests/<id>/status, .../requests/<id>/cancel: the queue's URLs for one request
_REQUEST_PATH = re.compile(r"^/(?P<app>.+?)/requests/(?P<id>[^/]+)(?P<rest>/.*)?$")

# Requests submitted over plain HTTP and not finished yet, shared by every watched session: the result is
# often fetched with another session than the one that submitted.
_http_calls = _Tracker(SETTINGS)


def track_http(client: Any, *, settings: Iterable[str] = SETTINGS) -> Any:
    """For apps that call fal's HTTP API themselves, without fal_client's methods. Watches the client's
    answers from fal's hosts and returns the same client, of the same type:

        http = inference_tracker.fal.track_http(httpx.Client())     # or httpx.AsyncClient, requests.Session

    - POST https://queue.fal.run/<endpoint>: a "start" record with the request id from the answer
      (not kept open when the URL has ?fal_webhook=...: webhook() records that finish).
    - GET .../requests/<id>: the "finish" of a request submitted here, ok or with fal's error code, with
      the inference time from a completed status seen before it and fal's x-fal-billable-units header
      when the answer has it. An error answer that doesn't say what failed (no fal code, not a 5xx: the
      request may still be running) leaves the request open. Submitted with the workflow envelope
      header, the result's steps are recorded as with track().
    - PUT .../requests/<id>/cancel, accepted: the finish, "cancelled".
    - POST https://fal.run/<endpoint>: one "call" record with the request id from the x-fal-request-id
      header.

    Everything else (other hosts, status streams, uploads) passes through unseen. A failed answer's error
    type is fal's code (x-fal-error-type, or the body's error_type or detail), else "http_<status>". A 5xx
    without fal's request id header came from fal's gateway, not the app, and isn't recorded (fal_client
    retries those). A request that gets no answer at all isn't recorded either.

    A fal_client.SyncClient, or a subclass whose own methods call fal over HTTP, can be passed as it is:
    its session is watched, so its SDK methods are recorded the same way. Use either track() or
    track_http() on one client, not both, or calls are recorded twice. For a fal_client.AsyncClient (or a
    subclass), whose session is made on first use, pass the session: track_http(await client._client)."""
    watcher = _HttpWatcher(settings)
    try:
        if _http.watch(client, watcher):
            return client
        if inspect.iscoroutinefunction(getattr(client, "submit", None)):
            logger.warning("inference_tracker: for a fal_client.AsyncClient, pass its session: await client._client")
            return client
        if not _http.watch(getattr(client, "_client", None), watcher):  # fal_client.SyncClient's session
            logger.warning("inference_tracker: can't watch a %s for fal calls", type(client).__name__)
    except Exception:
        logger.exception("inference_tracker: could not watch a client for fal calls")
    return client


class _HttpWatcher:
    name = "fal"

    def __init__(self, settings: Iterable[str]):
        self.settings = tuple(settings)

    def accepts(self, host: str) -> bool:
        return host in QUEUE_HOSTS or host in RUN_HOSTS

    def wants_body(self, ex: _http.Exchange) -> bool:
        kind, request_id, _ = _route(ex)
        if kind == "submit":
            return True  # the request id is in it; a failed one says why
        if kind == "run":
            return not ex.ok or _asks_envelope(ex)
        call = _http_calls.peek(request_id)
        if kind == "status":
            return call is not None and ex.ok
        if kind == "result":
            return call is not None and (not ex.ok or call.envelope)
        return False

    def seen(self, ex: _http.Exchange) -> None:
        kind, request_id, model = _route(ex)
        if kind == "submit":
            self._submitted(ex, model)
        elif kind == "run":
            self._ran(ex, model)
        elif kind == "status":
            call = _http_calls.peek(request_id)
            if call is not None and isinstance(ex.body, Mapping) and ex.body.get("status") == "COMPLETED":
                call.saw_metrics(ex.body.get("metrics"))
        elif kind == "result":
            self._result(ex, _http_calls.peek(request_id))
        elif kind == "cancel" and ex.ok:
            _http_calls.finish(_http_calls.peek(request_id), "cancelled")

    def _start(self, ex: _http.Exchange, model: str) -> _Call | None:
        arguments = ex.request_json() if self.settings else None  # read only for the settings copied from it
        return _http_calls.start(model, arguments, settings=self.settings, started=(ex.started_at, ex.clock))

    def _submitted(self, ex: _http.Exchange, model: str) -> None:
        if not ex.ok:
            if _from_fal(ex) and (call := self._start(ex, model)) is not None:
                call.completed(ex.headers.get(REQUEST_ID_HEADER), _http_error_type(ex) or f"http_{ex.status}")
            return
        request_id = ex.body.get("request_id") if isinstance(ex.body, Mapping) else None
        if not isinstance(request_id, str) or not request_id:
            return
        call = self._start(ex, model)
        if call is not None:
            call.envelope = _asks_envelope(ex)
            _http_calls.submitted(call, request_id, keep="fal_webhook" not in ex.query)

    def _ran(self, ex: _http.Exchange, model: str) -> None:
        if not ex.ok and not _from_fal(ex):
            return
        call = self._start(ex, model)
        if call is None:
            return
        _billable_units(ex, call)
        request_id = ex.headers.get(REQUEST_ID_HEADER)
        call.completed(request_id, None if ex.ok else _http_error_type(ex) or f"http_{ex.status}")
        if ex.ok and _asks_envelope(ex):
            _safe(_record_steps, ex.body, request_id)

    def _result(self, ex: _http.Exchange, call: _Call | None) -> None:
        if call is None:
            return  # not submitted here (or already finished): nothing to finish
        error_type = None if ex.ok else _http_error_type(ex)
        if not ex.ok and error_type is None:
            if ex.status < 500:
                return  # doesn't say the request failed: it may still be running
            error_type = f"http_{ex.status}"
        _billable_units(ex, call)
        _http_calls.finish(call, error_type, ex.body if ex.ok and call.envelope else None)


def _route(ex: _http.Exchange) -> tuple[str | None, str | None, str]:
    """What a request to fal's hosts was: ("submit" | "run" | "status" | "result" | "cancel" | None, the
    request id in the URL, the endpoint in the URL)."""
    if ex.host in RUN_HOSTS:
        return "run", None, ex.path.strip("/")
    match = _REQUEST_PATH.match(ex.path)
    if match is None:
        return ("submit" if ex.method == "POST" else None), None, ex.path.strip("/")
    rest = (match["rest"] or "").rstrip("/")
    kind = {("GET", ""): "result", ("GET", "/status"): "status", ("PUT", "/cancel"): "cancel"}.get((ex.method, rest))
    return kind, match["id"], match["app"]


def _asks_envelope(ex: _http.Exchange) -> bool:
    return str(ex.request_headers.get(WORKFLOW_ENVELOPE_HEADER, "")).lower() == "true"


def _from_fal(ex: _http.Exchange) -> bool:
    """False for a 5xx that fal's gateway sent (no request id): the request never reached the app."""
    return ex.status < 500 or REQUEST_ID_HEADER in ex.headers


def _http_error_type(ex: _http.Exchange) -> str | None:
    """fal's code for a failed answer: the x-fal-error-type header, or the body's error_type or detail.
    Never the message. None when the answer has none."""
    code = ex.headers.get(ERROR_TYPE_HEADER)
    if isinstance(code, str) and code:
        return code
    body = ex.body
    if isinstance(body, Mapping):
        if isinstance(body.get("error_type"), str) and body["error_type"]:
            return body["error_type"]
        return _detail_code(body)
    return None


def _billable_units(ex: _http.Exchange, call: _Call) -> None:
    units = _http.header_number(ex.headers, BILLABLE_UNITS_HEADER)
    if units is not None:
        call.units["billable_units"] = units


# --- helpers ----------------------------------------------------------------------------------

# The request id fal's last answer carried, for run(), whose result has no id in it. Set by a response
# hook on the client's httpx session; per thread or task, like the context.
_seen_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "inference_tracker_fal_request", default=None
)


def _watch_request_ids(http: Any, is_async: bool) -> None:
    """Adds a response hook to fal's httpx session (the client's `_client`) that keeps the request id
    header. Without a session that takes hooks, run() records without a request id."""
    hooks = getattr(http, "event_hooks", None)
    if not isinstance(hooks, Mapping):
        return

    def remember(response: Any) -> None:
        _seen_request_id.set(response.headers.get(REQUEST_ID_HEADER))

    async def remember_async(response: Any) -> None:
        remember(response)

    try:
        http.event_hooks = {**hooks, "response": [*hooks.get("response", []), remember_async if is_async else remember]}
    except Exception:
        logger.debug("inference_tracker: could not watch fal's responses for request ids", exc_info=True)


def _error_type(error: BaseException) -> str:
    """fal's error code when its exception carries one, else the exception's class name. Never a message."""
    code = getattr(error, "error_type", None)
    return code if isinstance(code, str) and code else error_name(error)


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("inference_tracker: tracking a fal call failed")
        return None
