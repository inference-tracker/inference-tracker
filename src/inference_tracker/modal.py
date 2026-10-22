"""Tracks calls to our own Modal apps, made with the modal package or over HTTP to their web endpoints.

    import inference_tracker

    generate = inference_tracker.modal.function("video-app", "generate")   # modal.Function.from_name, tracked
    result = generate.remote(args)                       # or await generate.remote.aio(args)
    call = generate.spawn(args); ...; result = call.get()

    http = inference_tracker.modal.track_http(httpx.Client(), apps=["video-app"])   # web endpoints
    http.post("https://acme--video-app-generate.modal.run/", json=args)

Modal bills machine time (GPU, CPU, memory) by app, not by call. Each record says which app and function
ran, when and for how long, so the server can split an app's bill over the calls that ran on it. The model
is "<app>/<function>" (a class method: "<app>/<Class>.<method>"), the Modal function call id ("fc-...") is
the request id. Nothing of the arguments or the result is recorded. The call itself is never changed, and
tracking never raises: a problem in it is logged and the call goes on as if nothing happened.

What is recorded:

- remote() / remote.aio(): one "call" record with the time it took and whether it raised (the exception's
  class name). Without a request id: the modal package makes the function call id inside remote() and
  doesn't hand it back. When the id matters (the function makes tracked calls of its own, which name it
  as their parent), use spawn() and get(), which give it.
- spawn() / spawn.aio(): a "start" record with the FunctionCall's id. get() / get.aio() on what it returns
  is the "finish" (a get(timeout=...) that times out while the call still runs isn't); cancel() finishes
  it as "cancelled". A FunctionCall fetched again by id (FunctionCall.from_id) in another process isn't
  tracked: the start has no finish then.
- HTTP to a web endpoint (track_http): one "call" record per answer, with the id from Modal's
  modal-function-call-id header. Long requests that Modal redirects (303) until the result is ready are
  one record, from the first request to the final answer.

map(), starmap(), for_each(), spawn_map() and remote_gen() pass through untracked.

Code running inside a Modal function that makes AI calls of its own: inside() makes the function call it is
serving the parent of every tracked call in the block.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import _client, _http
from ._context import Context, context, current
from ._record import Record, error_name

logger = logging.getLogger("inference_tracker")

PROVIDER = "modal"
CALL_ID_HEADER = "modal-function-call-id"
CALL_ID_PARAM = "__modal_function_call_id"  # in the URL Modal redirects a long web request to
HOST_SUFFIX = ".modal.run"


def function(app_name: str, name: str, **kwargs: Any) -> Any:
    """modal.Function.from_name(app_name, name, **kwargs), tracked as "<app_name>/<name>"."""
    import modal

    return track(modal.Function.from_name(app_name, name, **kwargs), f"{app_name}/{name}")


def track(fn: Any, model: str) -> Any:
    """Wraps a modal.Function (or a method of a modal.Cls instance) so that remote() and spawn() are
    recorded. `model` is "<app>/<function>", as Modal names them:

        fn = modal.Function.from_name("video-app", "generate")
        generate = inference_tracker.modal.track(fn, "video-app/generate")

        upscaler = modal.Cls.from_name("video-app", "Upscaler")()
        upscale = inference_tracker.modal.track(upscaler.run, "video-app/Upscaler.run")

    Everything else is the function's own. with_options(gpu=...) gives a tracked function too, with the
    GPU noted (it changes the price)."""
    return _Function(fn, model, {})


@contextmanager
def inside(call_id: str | None = None) -> Generator[Context, None, None]:
    """In a Modal function: every tracked call in the block gets the function call being served as its parent.

        @app.function()
        def generate(args):
            with inference_tracker.modal.inside():
                ...

    The id is modal.current_function_call_id(), or `call_id` when given. When there is none (the code runs
    outside Modal), the block runs as if inside() weren't there."""
    call_id = call_id or _current_call_id()
    if not call_id:
        logger.debug("inference_tracker: no current Modal function call, so calls in this block get no parent")
        yield current()
        return
    with context(parent_provider=PROVIDER, parent_request_id=call_id) as ctx:
        yield ctx


def _current_call_id() -> str | None:
    try:
        import modal

        call_id = modal.current_function_call_id()
    except Exception:
        return None
    return call_id if isinstance(call_id, str) and call_id else None


# --- one call ----------------------------------------------------------------------------------


class _Call:
    """The record of one call while this process follows it."""

    def __init__(self, model: str | None, settings: dict[str, Any], started: tuple[float, float] | None = None):
        started_at, self._clock = started or (time.time(), time.monotonic())
        self.record = Record.new(PROVIDER, model, phase="start", started_at=started_at, settings=dict(settings))
        self.done = False

    def spawned(self, call_id: Any) -> None:
        self.record.request_id = call_id if isinstance(call_id, str) and call_id else None
        _client.record(self.record)

    def finish(self, error_type: str | None) -> None:
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
                user_id=start.user_id,
                task_id=start.task_id,
                parent_provider=start.parent_provider,
                parent_request_id=start.parent_request_id,
            )
        )

    def completed(self, request_id: str | None, error_type: str | None) -> None:
        """The whole call seen here: one "call" record."""
        if self.done:
            return
        self.done = True
        r = self.record
        r.phase, r.request_id = "call", request_id
        r.status, r.error_type = ("error" if error_type else "ok"), error_type
        r.duration_s = self._elapsed()
        _client.record(r)

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._clock, 3)


def _start(model: str, settings: dict[str, Any]) -> _Call | None:
    try:
        return _Call(model, settings)
    except Exception:
        logger.exception("inference_tracker: could not start tracking a Modal call")
        return None


# --- the wrappers ----------------------------------------------------------------------------------


class _Method:
    """Stands in for one of modal's methods: called as it is, or awaited through .aio, like the original.
    `before()` runs first; `after(state, result, error)` runs when the call returns (and gives back what the
    app gets) or raises."""

    def __init__(self, method: Any, before: Callable[[], Any], after: Callable[[Any, Any, BaseException | None], Any]):
        self._method, self._before, self._after = method, before, after

    def __getattr__(self, name: str) -> Any:
        return getattr(self._method, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        state = _safe(self._before)
        try:
            result = self._method(*args, **kwargs)
        except BaseException as e:
            _safe(self._after, state, None, e)
            raise
        return self._done(state, result)

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        state = _safe(self._before)
        try:
            result = await self._method.aio(*args, **kwargs)
        except BaseException as e:
            _safe(self._after, state, None, e)
            raise
        return self._done(state, result)

    def _done(self, state: Any, result: Any) -> Any:
        try:
            return self._after(state, result, None)
        except Exception:
            logger.exception("inference_tracker: tracking a Modal call failed")
            return result


class _Function:
    """Stands in for a modal.Function: remote() and spawn() are recorded, the rest is the function's own."""

    def __init__(self, function: Any, model: str, settings: dict[str, Any]):
        self._function, self._model, self._settings = function, model, settings

    def __getattr__(self, name: str) -> Any:
        return getattr(self._function, name)

    def __repr__(self) -> str:
        return f"<tracked {self._function!r}>"

    @property
    def remote(self) -> _Method:
        def after(call: _Call | None, result: Any, error: BaseException | None) -> Any:
            if call is not None:
                call.completed(None, None if error is None else error_name(error))
            return result

        return _Method(self._function.remote, lambda: _start(self._model, self._settings), after)

    @property
    def spawn(self) -> _Method:
        def after(call: _Call | None, function_call: Any, error: BaseException | None) -> Any:
            if call is None:
                return function_call
            if error is not None:  # never got a function call id: one failed "call" record
                call.completed(None, error_name(error))
                return None
            call.spawned(getattr(function_call, "object_id", None))
            return _FunctionCall(function_call, call)

        return _Method(self._function.spawn, lambda: _start(self._model, self._settings), after)

    def with_options(self, **options: Any) -> _Function:
        settings = dict(self._settings)
        if isinstance(options.get("gpu"), (str, list)):
            settings["gpu"] = options["gpu"]
        return _Function(self._function.with_options(**options), self._model, settings)


class _FunctionCall:
    """Stands in for the modal.FunctionCall spawn() returns: get() and cancel() finish the call."""

    def __init__(self, function_call: Any, call: _Call):
        self._function_call, self._call = function_call, call

    def __getattr__(self, name: str) -> Any:
        return getattr(self._function_call, name)

    def __repr__(self) -> str:
        return f"<tracked {self._function_call!r}>"

    @property
    def get(self) -> _Method:
        def after(_: Any, result: Any, error: BaseException | None) -> Any:
            if type(error) is TimeoutError:  # modal's "no result yet" for get(timeout=...): still running
                return result
            self._call.finish(None if error is None else error_name(error))
            return result

        return _Method(self._function_call.get, lambda: None, after)

    @property
    def cancel(self) -> _Method:
        def after(_: Any, result: Any, error: BaseException | None) -> Any:
            if error is None:
                self._call.finish("cancelled")
            return result

        return _Method(self._function_call.cancel, lambda: None, after)


# --- over HTTP ---------------------------------------------------------------------------------------

# Long web requests: Modal answers 303 with the function call id until the result is ready. The first
# request's time is kept by call id until the final answer, which may come to another watched session;
# beyond this many, the oldest is forgotten.
MAX_OPEN = 1000
_redirected: OrderedDict[str, tuple[float, float]] = OrderedDict()
_redirected_lock = threading.Lock()


def track_http(client: Any, *, apps: Iterable[str] = ()) -> Any:
    """Watches an httpx.Client, httpx.AsyncClient or requests.Session for calls to Modal web endpoints
    (https://<workspace>--<label>.modal.run/...) and returns the same client. One "call" record per call.

    Modal names a web endpoint's host "<app>-<function>" (the label), which can't be split where both have
    dashes: list the app names in `apps` and the model is "<app>/<function>"; a host that starts with none
    of them (a custom label) keeps the label as its model. A trailing "-dev" (modal serve) is dropped."""
    try:
        if not _http.watch(client, _HttpWatcher(apps)):
            logger.warning("inference_tracker: can't watch a %s for Modal calls", type(client).__name__)
    except Exception:
        logger.exception("inference_tracker: could not watch a client for Modal calls")
    return client


class _HttpWatcher:
    name = "modal"

    def __init__(self, apps: Iterable[str]):
        # Longest first, so "video-app-v2" wins over "video-app".
        self.apps = sorted({a for a in apps if isinstance(a, str) and a}, key=len, reverse=True)

    def accepts(self, host: str) -> bool:
        return host.endswith(HOST_SUFFIX)

    def wants_body(self, ex: _http.Exchange) -> bool:
        return False

    def seen(self, ex: _http.Exchange) -> None:
        call_id = ex.headers.get(CALL_ID_HEADER) or _param(ex.query)
        if 300 <= ex.status < 400:
            call_id = call_id or _param(parse_qs(urlsplit(ex.headers.get("location") or "").query))
            if call_id:  # not done yet: the final answer comes from the URL Modal redirects to
                with _redirected_lock:
                    _redirected.setdefault(call_id, (ex.started_at, ex.clock))
                    while len(_redirected) > MAX_OPEN:
                        _redirected.popitem(last=False)
            return
        with _redirected_lock:
            started = _redirected.pop(call_id, None) if call_id else None
        call = _Call(self.model(ex.host), {}, started or (ex.started_at, ex.clock))
        call.completed(call_id or None, None if ex.ok else f"http_{ex.status}")

    def model(self, host: str) -> str:
        label = host[: -len(HOST_SUFFIX)].partition("--")[2] or host[: -len(HOST_SUFFIX)]
        label = label.removesuffix("-dev")
        for app in self.apps:
            prefix = app.lower() + "-"
            if label.startswith(prefix) and len(label) > len(prefix):
                return f"{app}/{label[len(prefix) :]}"
        return label


def _param(query: dict[str, list[str]]) -> str | None:
    values = query.get(CALL_ID_PARAM)
    return values[0] if values and values[0] else None


# --- helpers ----------------------------------------------------------------------------------


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.exception("inference_tracker: tracking a Modal call failed")
        return None
