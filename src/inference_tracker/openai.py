"""Tracks calls made with the openai package, sync or async: OpenAI's own API, and Azure (AzureOpenAI, or
OpenAI with an Azure base_url, as Azure AI Foundry suggests).

    from openai import OpenAI
    import inference_tracker

    client = inference_tracker.openai.track(OpenAI(base_url=..., api_key=...))
    response = client.responses.parse(model="gpt-5.4-mini", input=..., text_format=...)

Tracked, one "call" record each (the whole call is seen here): responses.create / parse,
chat.completions.create / parse (and beta.chat.completions.parse), images.generate / edit, and a raw
client.post() to an images route ("/images/generations", "/images/edits"), which is how Azure AI Foundry's
other image models (FLUX, for example) are called. Everything else passes through unchanged, and so do
streamed calls (stream=True). The call itself is never changed,
and tracking never raises: a problem in it is logged and the call goes on.

On the record: the model the response names (the one billed; the requested name, an Azure deployment for
example, goes in settings when it differs), the response id, the usage the response reports (tokens by
kind, images), and settings that change the price (reasoning effort, service tier, image size, quality,
count). Never prompts, images or outputs. For Azure, settings also hold the resource the call went to
(the first part of its host name), which is how its bill is split.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from . import _client
from ._proxy import Path, Proxy, get, put
from ._record import Record, error_name

logger = logging.getLogger("inference_tracker")

OPENAI = "openai"
AZURE = "azure.ai.openai"

PATHS = frozenset(
    {
        ("responses", "create"),
        ("responses", "parse"),
        ("chat", "completions", "create"),
        ("chat", "completions", "parse"),
        ("beta", "chat", "completions", "parse"),
        ("images", "generate"),
        ("images", "edit"),
        ("post",),  # only a raw call to an images route, see _images_post
    }
)
# Request arguments copied onto the record because they change the price.
SETTINGS = ("service_tier", "size", "quality", "n", "background", "output_format", "input_fidelity")


def track(client: Any) -> Any:
    """Wraps an openai.OpenAI, AsyncOpenAI, AzureOpenAI or AsyncAzureOpenAI. See the module docstring."""
    provider, resource = _where(client)

    def wrap(path: Path, method: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
        if path == ("post",):
            return _images_post(method, is_async, provider, resource)
        if is_async:

            async def tracked_async(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get("stream"):
                    return await method(*args, **kwargs)
                call = _start(provider, resource, kwargs)
                try:
                    response = await method(*args, **kwargs)
                except BaseException as e:
                    _end(call, path, None, e)
                    raise
                _end(call, path, response, None)
                return response

            return tracked_async

        def tracked(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream"):
                return method(*args, **kwargs)
            call = _start(provider, resource, kwargs)
            try:
                response = method(*args, **kwargs)
            except BaseException as e:
                _end(call, path, None, e)
                raise
            _end(call, path, response, None)
            return response

        return tracked

    return Proxy(client, PATHS, wrap)


def _images_post(method: Callable[..., Any], is_async: bool, provider: str, resource: str | None) -> Callable[..., Any]:
    """client.post(path, body=..., cast_to=...): tracked like images.generate when path is an images route,
    passed through otherwise."""

    def images_body(args: tuple, kwargs: dict) -> dict | None:
        route = kwargs.get("path", args[0] if args else None)
        body = kwargs.get("body")
        if isinstance(route, str) and "/images/" in route and isinstance(body, dict):
            return body
        return None

    if is_async:

        async def tracked_async(*args: Any, **kwargs: Any) -> Any:
            body = images_body(args, kwargs)
            if body is None:
                return await method(*args, **kwargs)
            call = _start(provider, resource, body)
            try:
                response = await method(*args, **kwargs)
            except BaseException as e:
                _end(call, ("images",), None, e)
                raise
            _end(call, ("images",), response, None)
            return response

        return tracked_async

    def tracked(*args: Any, **kwargs: Any) -> Any:
        body = images_body(args, kwargs)
        if body is None:
            return method(*args, **kwargs)
        call = _start(provider, resource, body)
        try:
            response = method(*args, **kwargs)
        except BaseException as e:
            _end(call, ("images",), None, e)
            raise
        _end(call, ("images",), response, None)
        return response

    return tracked


def _where(client: Any) -> tuple[str, str | None]:
    """OpenAI or Azure, and for Azure the resource name (first label of the host)."""
    try:
        host = urlparse(str(getattr(client, "base_url", "") or "")).hostname or ""
        if "azure" in type(client).__name__.lower() or host.endswith(".azure.com"):
            return AZURE, host.split(".")[0] or None
    except Exception:
        logger.debug("inference_tracker: could not read the openai client's base_url", exc_info=True)
    return OPENAI, None


def _start(provider: str, resource: str | None, kwargs: dict[str, Any]) -> tuple[Record, float] | None:
    try:
        settings = {k: kwargs[k] for k in SETTINGS if k in kwargs and kwargs[k] is not None}
        effort = get(kwargs.get("reasoning"), "effort") or kwargs.get("reasoning_effort")
        if effort is not None:
            settings["reasoning_effort"] = effort
        if resource:
            settings["azure_resource"] = resource
        model = kwargs.get("model")
        record = Record.new(
            provider, str(model) if model is not None else None, started_at=time.time(), settings=settings
        )
        return record, time.monotonic()
    except Exception:
        logger.exception("inference_tracker: could not start tracking an openai call")
        return None


def _end(call: tuple[Record, float] | None, path: Path, response: Any, error: BaseException | None) -> None:
    if call is None:
        return
    record, clock = call
    try:
        record.duration_s = round(time.monotonic() - clock, 3)
        if error is not None:
            record.status = "error"
            code = getattr(error, "code", None)
            record.error_type = code if isinstance(code, str) and code else error_name(error)
            record.request_id = getattr(error, "request_id", None) or None
        else:
            record.status = "ok"
            record.request_id = get(response, "id") or getattr(response, "_request_id", None)
            answered = get(response, "model")
            if isinstance(answered, str) and answered and answered != record.model:
                if record.model:
                    record.settings["requested_model"] = record.model
                record.model = answered
            record.units = _units(path, response)
        _client.record(record)
    except Exception:
        logger.exception("inference_tracker: could not record an openai call")


def _units(path: Path, response: Any) -> dict[str, float]:
    usage = get(response, "usage")
    units: dict[str, float] = {}
    if path[0] == "images":
        data = get(response, "data")
        if isinstance(data, list):
            units["images"] = len(data)
        put(units, "input_tokens", get(usage, "input_tokens"))
        put(units, "output_tokens", get(usage, "output_tokens"))
        put(units, "input_text_tokens", get(usage, "input_tokens_details", "text_tokens"))
        put(units, "input_image_tokens", get(usage, "input_tokens_details", "image_tokens"))
        return units
    if path[0] == "responses":
        put(units, "input_tokens", get(usage, "input_tokens"))
        put(units, "output_tokens", get(usage, "output_tokens"))
        put(units, "cached_input_tokens", get(usage, "input_tokens_details", "cached_tokens"))
        put(units, "reasoning_tokens", get(usage, "output_tokens_details", "reasoning_tokens"))
        return units
    put(units, "input_tokens", get(usage, "prompt_tokens"))
    put(units, "output_tokens", get(usage, "completion_tokens"))
    put(units, "cached_input_tokens", get(usage, "prompt_tokens_details", "cached_tokens"))
    put(units, "reasoning_tokens", get(usage, "completion_tokens_details", "reasoning_tokens"))
    return units
