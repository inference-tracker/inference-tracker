"""Tracks calls made through LiteLLM, for every provider it routes to: AWS Bedrock, OpenAI, Azure, Vertex AI and the
rest. Also calls made by libraries built on LiteLLM, like the OpenAI Agents SDK's LitellmModel.

    import inference_tracker

    inference_tracker.litellm.track()  # once, at startup; calling it again does nothing

    with inference_tracker.context(user_id=uid, task_id=task.id):
        response = await litellm.acompletion(model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", ...)

track() adds a callback to litellm.callbacks, LiteLLM's own way to see each call, so nothing is wrapped and the call
itself is never changed. Tracking never raises: a problem in it is logged and the call goes on. An app that sets
litellm.callbacks itself (litellm.callbacks = [...]) replaces the list: call track() after that.

Tracked, one "call" record each, sync or async, streamed or not: completion, text_completion, embedding,
image_generation, image_edit and responses. A streamed call is recorded when its stream ends (that is when LiteLLM
has its usage); a stream the app stops reading early isn't recorded. Answers from LiteLLM's cache aren't recorded:
no provider was called. Every other LiteLLM call passes through unrecorded.

Who a call is for: LiteLLM runs its callbacks on its own thread, or after the call has returned. So the context
(user, task, parent) is taken when LiteLLM sends the request (its pre-call hook, which runs where the app made the
call) and kept by LiteLLM's id for the call until the call is recorded.

When a call is recorded: an async call (acompletion..., and so the OpenAI Agents SDK's Runner.run and run_sync) is
recorded before it returns to the app, from the hook LiteLLM runs on the answer
(async_post_call_success_deployment_hook): LiteLLM's own success callbacks run later on its logging worker, which may
be after the app's flushing block ended, or never, if the process is frozen first. A sync call (completion...) is
recorded by LiteLLM's logging thread a moment after it returns; inference_tracker.flush() (and so
context(flush=True)) first waits for those of the calls that have been answered, so they are in that flush. Streams
are recorded when they end, as LiteLLM reports them.

When the app stops waiting for an async call (its task is cancelled, or a timeout around it runs out) before the
provider answered, LiteLLM records nothing. The call is then recorded with status "cancelled" (error_type
"CancelledError") when the task that made it is done: the provider may still have run it, and billed it.

On the record:
- provider: Bedrock is "aws.bedrock". OpenAI, Azure (azure, azure_ai), Vertex AI and Gemini get the names the openai
  and google wrappers use, so the same bills split them; any other provider keeps LiteLLM's name for it.
- model: the one the provider billed. For Bedrock, the model or inference profile id in the request's URL (no
  "bedrock/" or route prefix; an application inference profile's ARN as it is). For others, the model the response
  names, with the requested one in settings when it differs (an Azure deployment, say).
- request_id: for Bedrock, the x-amzn-requestid header of its answer (LiteLLM keeps the answer's headers). For
  others, the response id, or the request id header when there isn't one (embeddings, images, errors).
- units: input_tokens (all of the input, cached or not, as LiteLLM counts it), and the parts of it read from or
  written to the prompt cache (cached_input_tokens, cache_write_input_tokens); output_tokens and reasoning_tokens
  (part of the output); images.
- settings: those that change the price (service tier, Bedrock's latency-optimized mode, reasoning effort, thinking
  budget, image size, quality, count) and where the call went: the AWS region, the Azure resource, the Google Cloud
  project. Never prompts, inputs or outputs.
- status and error_type: Bedrock's error code (x-amzn-errortype), or the exception's class name.

Each call is recorded once. A call LiteLLM makes with its own client isn't seen by the other wrappers, so tracking
both LiteLLM and an openai client (inference_tracker.openai.track) is fine. Only don't pass a tracked client to
LiteLLM (litellm.completion(..., client=tracked)): that call would be recorded by both.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

from . import _client
from ._context import Context
from ._context import current as _current_context
from ._proxy import get, number, put
from ._record import Record, cancelled, code, error_name
from .google import GEMINI, VERTEX
from .openai import AZURE, OPENAI

logger = logging.getLogger("inference_tracker")

BEDROCK = "aws.bedrock"

# LiteLLM's provider names, as the bills know them. A provider not here keeps LiteLLM's name.
PROVIDERS = {
    "bedrock": BEDROCK,
    "openai": OPENAI,
    "text-completion-openai": OPENAI,
    "azure": AZURE,
    "azure_text": AZURE,
    "azure_ai": AZURE,
    "vertex_ai": VERTEX,
    "vertex_ai_beta": VERTEX,
    "gemini": GEMINI,
}
# LiteLLM's call types that are recorded (sync and async).
CALL_TYPES = frozenset(
    {
        t
        for c in ("completion", "text_completion", "embedding", "image_generation", "image_edit", "responses")
        for t in (c, "a" + c)
    }
)
# Request arguments copied onto the record because they change the price.
SETTINGS = ("service_tier", "reasoning_effort", "size", "quality", "n")
# Calls sent and not recorded yet, by LiteLLM's call id. Beyond this many, the oldest is forgotten.
MAX_OPEN = 1000
# How long flush() waits, at most, for a sync call that was answered this long ago to be recorded. One that isn't
# by then (LiteLLM dropped its logging task, say) isn't waited for again.
LATE_FOR = 5.0

# An id LiteLLM makes up when the provider gives none (Bedrock's answers, for one): not the provider's.
_MADE_UP_ID = re.compile(r"^chatcmpl-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_BEDROCK_HOST = re.compile(r"^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com$")
_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "apim-request-id", "x-amzn-requestid")


@dataclass
class _Open:
    """A call LiteLLM has sent and this module hasn't recorded yet."""

    context: Context
    api_base: str | None
    sync: bool  # completion, not acompletion: recorded on LiteLLM's logging thread
    stream: bool
    kwargs: Any = None  # LiteLLM's own dict for the call, kept only for an async call, to record it if cancelled
    answered: float | None = None  # time.monotonic() when the provider's answer came (LiteLLM's post-call hook)


_open: OrderedDict[str, _Open] = OrderedDict()
# Calls recorded when they returned (_returned): LiteLLM's success callback for them comes later, and is skipped.
_recorded: OrderedDict[str, None] = OrderedDict()
_open_lock = threading.Lock()
_changed = threading.Condition(_open_lock)  # a call was recorded: for flush(), waiting for late records
_recording = 0  # sync calls taken out of _open whose record isn't queued yet
_logger: Any = None
_track_lock = threading.Lock()


def track() -> Any:
    """Adds the tracking callback to litellm.callbacks, once. Returns it (None when LiteLLM can't be loaded)."""
    global _logger
    try:
        import litellm

        with _track_lock:
            if _logger is None:
                _logger = _make_logger()
            if _logger not in litellm.callbacks:
                litellm.callbacks.append(_logger)
            if _wait_for_records not in _client._before_flush:
                _client._before_flush.append(_wait_for_records)
        return _logger
    except Exception:
        logger.exception("inference_tracker: could not add the litellm callback")
        return None


def _make_logger() -> Any:
    from litellm.integrations.custom_logger import CustomLogger

    class Tracker(CustomLogger):
        """LiteLLM calls these itself; the argument names are LiteLLM's (it passes them by name)."""

        def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:
            _before(kwargs)

        def log_post_api_call(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:
            _answered(kwargs)

        async def async_post_call_success_deployment_hook(
            self, request_data: Any, response: Any, call_type: Any
        ) -> Any:
            _returned(request_data, response)
            return None  # the response as it is

        def log_success_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:
            _after(kwargs, response_obj, start_time, end_time)

        def log_failure_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:
            _after(kwargs, response_obj, start_time, end_time, failed=True)

        async def async_log_success_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:
            _after(kwargs, response_obj, start_time, end_time)

        async def async_log_failure_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:
            _after(kwargs, response_obj, start_time, end_time, failed=True)

    return Tracker()


def _before(kwargs: Any) -> None:
    """Runs where the app made the call, just before the request goes out: keeps who it is for, and the URL. An async
    call is watched until the task that made it is done, in case the app stops waiting for it (see _task_done)."""
    try:
        if not _client.active():
            return
        call_id = get(kwargs, "litellm_call_id")
        if not call_id:
            return
        api_base = get(kwargs, "additional_args", "api_base")
        call_type = get(kwargs, "call_type")
        call = _Open(
            _current_context(),
            api_base if isinstance(api_base, str) else None,
            sync=isinstance(call_type, str) and not call_type.startswith("a"),
            stream=get(kwargs, "stream") is True,
        )
        task = None
        if call_type in CALL_TYPES and not call.sync and not call.stream:
            try:
                asyncio.get_running_loop()
                task = asyncio.current_task()
            except RuntimeError:
                task = None  # not on an event loop: LiteLLM sends it from a thread
            call.kwargs = kwargs if task is not None else None
        with _open_lock:
            _open[call_id] = call
            _open.move_to_end(call_id)
            while len(_open) > MAX_OPEN:
                _open.popitem(last=False)
        if task is not None:
            task.add_done_callback(lambda done: _task_done(call_id, done))
    except Exception:
        logger.exception("inference_tracker: could not start tracking a litellm call")


def _answered(kwargs: Any) -> None:
    """LiteLLM's post-call hook: the provider's answer has come, so a record is due (see _wait_for_records)."""
    try:
        with _open_lock:
            call = _open.get(get(kwargs, "litellm_call_id"))
            if call is not None and call.answered is None:
                call.answered = time.monotonic()
    except Exception:
        logger.exception("inference_tracker: could not track a litellm answer")


def _returned(request_data: Any, response: Any) -> None:
    """An async call's answer, before it returns to the app: recorded now. A stream is recorded when it ends."""
    try:
        if not _client.active() or response is None or hasattr(response, "__anext__") or hasattr(response, "__next__"):
            return
        logging_obj = get(request_data, "litellm_logging_obj")
        kwargs = getattr(logging_obj, "model_call_details", None)
        if not isinstance(kwargs, dict):
            return
        _after(kwargs, response, getattr(logging_obj, "start_time", None), datetime.datetime.now(), inline=True)
    except Exception:
        logger.exception("inference_tracker: could not record a litellm call")


def _task_done(call_id: str, task: Any) -> None:
    """The task that made an async call is done. When the call is still open without an answer, the app stopped
    waiting for it (cancelled, or a timeout around it): recorded as "cancelled"."""
    try:
        with _open_lock:
            call = _open.get(call_id)
            if call is None or call.answered is not None or call.kwargs is None:
                return
        _after(call.kwargs, None, get(call.kwargs, "start_time"), datetime.datetime.now(), stopped=True)
    except Exception:
        logger.exception("inference_tracker: could not record a cancelled litellm call")


def _wait_for_records(deadline: float) -> None:
    """Before a flush: waits, until the deadline, for the sync calls the provider has answered to be recorded by
    LiteLLM's logging thread (it runs the callbacks a moment after the call returns)."""
    with _changed:
        while True:
            now = time.monotonic()
            late = [c for c in _open.values() if c.sync and not c.stream and c.answered and now - c.answered < LATE_FOR]
            if (not late and not _recording) or now >= deadline:
                return
            _changed.wait(min(deadline - now, 0.05))


def _after(
    kwargs: Any,
    response: Any,
    start_time: Any,
    end_time: Any,
    failed: bool = False,
    stopped: bool = False,
    inline: bool = False,
) -> None:
    """Records a call. `stopped`: the app stopped waiting before any answer (see _task_done). `inline`: recorded
    as it returns (_returned), so LiteLLM's success callback for it, which comes later, is skipped."""
    global _recording
    counted = False
    try:
        call_id = get(kwargs, "litellm_call_id")
        with _open_lock:
            if call_id is not None and not failed and not stopped and call_id in _recorded:
                if not inline:
                    _recorded.pop(call_id, None)
                return  # recorded when it returned
            call = _open.pop(call_id, None)
            if call is not None and call.sync:
                _recording += 1  # flush() waits for it until it is queued (see _wait_for_records)
                counted = True
            if inline and call_id is not None:
                _recorded[call_id] = None
                while len(_recorded) > MAX_OPEN:
                    _recorded.popitem(last=False)
        ctx, api_base = (call.context, call.api_base) if call is not None else (None, None)
        if not _client.active():
            return
        if get(kwargs, "call_type") not in CALL_TYPES or get(kwargs, "cache_hit") is True:
            return
        error = get(kwargs, "exception") if failed else None
        if failed or stopped:
            response = None
        llm_provider = get(kwargs, "custom_llm_provider") or get(kwargs, "litellm_params", "custom_llm_provider")
        provider = PROVIDERS.get(llm_provider, llm_provider) if isinstance(llm_provider, str) else None
        api_base = (
            api_base
            or _text(get(kwargs, "litellm_params", "api_base"))
            or _text(get(response, "_hidden_params", "api_base"))
        )
        url = urlparse(api_base or "")
        host = url.hostname or ""
        headers = _headers(kwargs, response, error)
        requested = _text(get(kwargs, "model"))
        settings = _settings(get(kwargs, "optional_params"), response)

        if provider == BEDROCK:
            model = _bedrock_model(url.path) or _text(get(response, "model")) or requested
            region = _BEDROCK_HOST.match(host)
            region = (region.group(1) if region else None) or _text(get(response, "_hidden_params", "region_name"))
            if region:
                settings["aws_region"] = region
            request_id = headers.get("x-amzn-requestid")
        else:
            model = _text(get(response, "model")) or requested
            if requested and model != requested:
                settings["requested_model"] = requested
            if host.endswith(".azure.com"):
                settings["azure_resource"] = host.split(".")[0]
            project = re.search(r"/projects/([^/]+)/", url.path)
            if project and provider == VERTEX:
                settings["gcp_project"] = project.group(1)
            request_id = _response_id(response) or next(
                (headers[h] for h in _REQUEST_ID_HEADERS if headers.get(h)), None
            )

        fields: dict[str, Any] = {}
        if ctx is not None:  # who the call was for when it was made, not whatever is current here
            fields = {
                "user_id": ctx.user_id,
                "task_id": ctx.task_id,
                "parent_provider": ctx.parent_provider,
                "parent_request_id": ctx.parent_request_id,
            }
        started, ended = _seconds(start_time), _seconds(end_time)
        record = Record.new(
            provider or "litellm",
            model,
            request_id=request_id or None,
            status="cancelled" if stopped or cancelled(error) else "error" if failed else "ok",
            started_at=started,
            duration_s=round(ended - started, 3) if started is not None and ended is not None else None,
            units=_units(get(kwargs, "call_type"), response),
            settings=settings,
            **fields,
        )
        if stopped:
            record.error_type = "CancelledError"
        elif failed:
            found = code((headers.get("x-amzn-errortype") or "").split(":")[0])
            record.error_type = found or (error_name(error) if isinstance(error, BaseException) else "error")
        _client.record(record)
    except Exception:
        logger.exception("inference_tracker: could not record a litellm call")
    finally:
        if counted:
            with _changed:
                _recording -= 1
                _changed.notify_all()


def _bedrock_model(path: str) -> str | None:
    """The model id in a Bedrock runtime URL: /model/<id>/converse (or converse-stream, invoke...)."""
    found = re.search(r"/model/([^/]+)/", path)
    return unquote(found.group(1)) if found else None


def _response_id(response: Any) -> str | None:
    rid = get(response, "id")
    return rid if isinstance(rid, str) and rid and not _MADE_UP_ID.match(rid) else None


def _headers(kwargs: Any, response: Any, error: Any) -> dict[str, str]:
    """The provider's answer headers, lower-case, from wherever LiteLLM keeps them (on the response, on the error)."""
    sources = (
        lambda: get(kwargs, "response_headers"),
        lambda: get(response, "_hidden_params", "additional_headers"),
        lambda: get(response, "_hidden_params", "headers"),
        lambda: getattr(error, "litellm_response_headers", None),
        lambda: get(error, "response", "headers"),
        lambda: get(kwargs, "standard_logging_object", "hidden_params", "additional_headers"),
    )
    found: dict[str, str] = {}
    for source in sources:
        try:
            items = dict(source() or {}).items()
        except Exception:
            continue
        for k, v in items:
            name = str(k).lower()
            if name.startswith("llm_provider-"):
                name = name[len("llm_provider-") :]
            if isinstance(v, str):
                found.setdefault(name, v)
    return found


def _settings(params: Any, response: Any) -> dict[str, Any]:
    params = params if isinstance(params, dict) else {}
    settings = {k: params[k] for k in SETTINGS if params.get(k) is not None}
    tier = get(params, "serviceTier", "type")  # Bedrock's own form of service_tier
    if tier is not None:
        settings.setdefault("service_tier", tier)
    tier = code(get(response, "service_tier"))  # the tier that served the call, when the answer says
    if tier is not None:
        settings["service_tier"] = tier
    latency = get(params, "performanceConfig", "latency")  # Bedrock's latency-optimized mode
    if latency is not None:
        settings["latency"] = latency
    budget = get(params, "thinking", "budget_tokens")
    if budget is not None:
        settings["thinking_budget"] = budget
    return settings


def _units(call_type: Any, response: Any) -> dict[str, float]:
    usage = get(response, "usage")
    units: dict[str, float] = {}
    if isinstance(call_type, str) and call_type.lstrip("a").startswith("image"):
        data = get(response, "data")
        if isinstance(data, list):
            units["images"] = len(data)
    if number(get(usage, "prompt_tokens")) is not None:  # chat completions, embeddings
        put(units, "input_tokens", get(usage, "prompt_tokens"))
        put(units, "output_tokens", get(usage, "completion_tokens"))
        details, out_details = get(usage, "prompt_tokens_details"), get(usage, "completion_tokens_details")
    else:  # responses, images
        put(units, "input_tokens", get(usage, "input_tokens"))
        put(units, "output_tokens", get(usage, "output_tokens"))
        details, out_details = get(usage, "input_tokens_details"), get(usage, "output_tokens_details")
    cached = number(get(details, "cached_tokens"))
    put(units, "cached_input_tokens", cached if cached is not None else get(usage, "cache_read_input_tokens"))
    written = number(get(usage, "cache_creation_input_tokens"))
    written = written if written is not None else number(get(details, "cache_creation_tokens"))
    if written:  # only providers that bill writing the cache (Anthropic models) report it
        units["cache_write_input_tokens"] = written
    put(units, "reasoning_tokens", get(out_details, "reasoning_tokens"))
    return units


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _seconds(t: Any) -> float | None:
    if isinstance(t, datetime.datetime):
        return t.timestamp()
    return number(t)
