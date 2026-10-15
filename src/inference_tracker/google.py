"""Tracks calls made with Google's google-genai package (genai.Client), on Vertex AI or the Gemini API,
sync or async (client.aio).

    from google import genai
    import inference_tracker

    client = inference_tracker.google.track(genai.Client(vertexai=True, project=..., location=...))
    response = client.models.generate_content(model="gemini-3.1-flash-image", contents=..., config=...)

Tracked:
- models.generate_content, generate_images, edit_image: one "call" record each, with the response id
  and the usage the response reports: tokens in and out, by kind (text, image...), thinking and cached
  tokens, images made.
- models.generate_videos (Veo) runs as a long-running operation: a "start" record when it is sent (its
  request id is the operation's name), and a "finish" record when operations.get() returns it done, with
  the seconds of video made (the requested length times the videos returned).
Everything else, streamed calls included, passes through unchanged. The call itself is never changed, and
tracking never raises: a problem in it is logged and the call goes on.

Settings copied onto the record are those that change the price (resolution, length, audio, number of
videos, image size, aspect ratio, thinking budget or level), plus the Google Cloud project the call went to,
which is how Vertex's bill is split. Never prompts, images or outputs.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from . import _client
from ._proxy import Path, Proxy, get, number, put
from ._record import Record, error_name

logger = logging.getLogger("inference_tracker")

VERTEX = "gcp.vertex_ai"
GEMINI = "gcp.gemini"

_CALLS = {"generate_content", "generate_images", "edit_image"}
PATHS = frozenset(
    {("models", m) for m in _CALLS | {"generate_videos"}}
    | {("aio", "models", m) for m in _CALLS | {"generate_videos"}}
    | {("operations", "get"), ("aio", "operations", "get")}
)
# Config fields copied onto the record because they change the price.
SETTINGS = (
    "duration_seconds",
    "resolution",
    "generate_audio",
    "number_of_videos",
    "number_of_images",
    "aspect_ratio",
    "image_size",
)
# Video operations sent and not seen done yet, by operation name. Beyond this many, the oldest is forgotten.
MAX_OPEN = 1000


def track(client: Any) -> Any:
    """Wraps a google.genai.Client. See the module docstring."""
    api = getattr(client, "_api_client", None)
    provider = VERTEX if getattr(api, "vertexai", False) else GEMINI
    project = getattr(api, "project", None) if provider == VERTEX else None
    open_ops: OrderedDict[str, tuple[Record, float]] = OrderedDict()
    lock = threading.Lock()

    def before(path: Path, kwargs: dict[str, Any]) -> tuple[Record, float] | None:
        if path[-1] == "get":
            return None
        try:
            config = kwargs.get("config")
            settings = {k: v for k in SETTINGS if (v := get(config, k)) is not None}
            for k in ("aspect_ratio", "image_size"):
                v = get(config, "image_config", k)
                if v is not None:
                    settings[k] = v
            for k in ("thinking_budget", "thinking_level"):
                v = get(config, "thinking_config", k)
                if v is not None:
                    settings[k] = getattr(v, "value", v)
            if project:
                settings["gcp_project"] = project
            phase = "start" if path[-1] == "generate_videos" else "call"
            record = Record.new(provider, kwargs.get("model"), phase=phase, started_at=time.time(), settings=settings)
            return record, time.monotonic()
        except Exception:
            logger.exception("inference_tracker: could not start tracking a google-genai call")
            return None

    def after(
        path: Path,
        call: tuple[Record, float] | None,
        args: tuple,
        kwargs: dict,
        result: Any,
        error: BaseException | None,
    ) -> None:
        try:
            if path[-1] == "get":
                _polled(args, kwargs, result, error)
            elif path[-1] == "generate_videos":
                _sent(call, result, error)
            else:
                _called(call, result, error)
        except Exception:
            logger.exception("inference_tracker: could not record a google-genai call")

    def _called(call: tuple[Record, float] | None, response: Any, error: BaseException | None) -> None:
        if call is None:
            return
        record, clock = call
        record.duration_s = round(time.monotonic() - clock, 3)
        _outcome(record, error)
        if error is None:
            record.request_id = get(response, "response_id")
            record.units = _units(response)
        _client.record(record)

    def _sent(call: tuple[Record, float] | None, operation: Any, error: BaseException | None) -> None:
        if call is None:
            return
        record, clock = call
        name = get(operation, "name") if error is None else None
        if error is not None or not name:
            # Never got an operation: the whole call ends here.
            record.phase = "call"
            record.duration_s = round(time.monotonic() - clock, 3)
            _outcome(record, error)
            _client.record(record)
            return
        record.request_id = name
        _client.record(record)
        with lock:
            open_ops[name] = call
            while len(open_ops) > MAX_OPEN:
                open_ops.popitem(last=False)
        if get(operation, "done"):
            _finish(name, operation, None)

    def _polled(args: tuple, kwargs: dict, operation: Any, error: BaseException | None) -> None:
        asked = kwargs.get("operation", args[0] if args else None)
        name = get(asked, "name") if not isinstance(asked, str) else asked
        if not name:
            return
        if error is not None:
            return  # a failed poll isn't the end of the operation; the app may poll again
        if get(operation, "done"):
            _finish(name, operation, None)

    def _finish(name: str, operation: Any, error: BaseException | None) -> None:
        with lock:
            call = open_ops.pop(name, None)
        if call is None:
            return
        start, clock = call
        failure = get(operation, "error")
        videos = get(operation, "response", "generated_videos")
        if videos is None:
            videos = get(operation, "result", "generated_videos")
        units: dict[str, float] = {}
        if isinstance(videos, list):
            units["videos"] = len(videos)
            seconds = number(start.settings.get("duration_seconds"))
            if seconds is not None:
                units["video_seconds"] = seconds * len(videos)
        error_type = None
        if failure:
            code = get(failure, "code")
            error_type = str(code) if code is not None else "operation_error"
        elif isinstance(videos, list) and not videos:
            filtered = get(operation, "response", "rai_media_filtered_count")
            error_type = "filtered" if filtered else "no_video"
        _client.record(
            Record.new(
                start.provider,
                start.model,
                phase="finish",
                request_id=name,
                status="error" if error_type else "ok",
                error_type=error_type,
                duration_s=round(time.monotonic() - clock, 3),
                units=units,
                user_id=start.user_id,
                task_id=start.task_id,
                parent_provider=start.parent_provider,
                parent_request_id=start.parent_request_id,
            )
        )

    def wrap(path: Path, method: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
        if is_async:

            async def tracked_async(*args: Any, **kwargs: Any) -> Any:
                call = before(path, kwargs)
                try:
                    result = await method(*args, **kwargs)
                except BaseException as e:
                    after(path, call, args, kwargs, None, e)
                    raise
                after(path, call, args, kwargs, result, None)
                return result

            return tracked_async

        def tracked(*args: Any, **kwargs: Any) -> Any:
            call = before(path, kwargs)
            try:
                result = method(*args, **kwargs)
            except BaseException as e:
                after(path, call, args, kwargs, None, e)
                raise
            after(path, call, args, kwargs, result, None)
            return result

        return tracked

    return Proxy(client, PATHS, wrap)


def _outcome(record: Record, error: BaseException | None) -> None:
    if error is None:
        record.status = "ok"
        return
    record.status = "error"
    status = getattr(error, "status", None)
    record.error_type = status if isinstance(status, str) and status else error_name(error)


def _units(response: Any) -> dict[str, float]:
    usage = get(response, "usage_metadata")
    units: dict[str, float] = {}
    put(units, "input_tokens", get(usage, "prompt_token_count"))
    put(units, "output_tokens", get(usage, "candidates_token_count"))
    put(units, "thinking_tokens", get(usage, "thoughts_token_count"))
    put(units, "cached_input_tokens", get(usage, "cached_content_token_count"))
    for side, field in (("input", "prompt_tokens_details"), ("output", "candidates_tokens_details")):
        for detail in get(usage, field) or []:
            modality = get(detail, "modality")
            modality = getattr(modality, "value", modality)
            count = number(get(detail, "token_count"))
            if isinstance(modality, str) and count is not None:
                key = f"{side}_{modality.lower()}_tokens"
                units[key] = units.get(key, 0) + count
    images = get(response, "generated_images")
    if isinstance(images, list):
        units["images"] = len(images)
    return units
