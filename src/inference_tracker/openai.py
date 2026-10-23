"""Tracks calls made with the openai package, sync or async: OpenAI's own API, Azure (AzureOpenAI, or
OpenAI with an Azure base_url, as Azure AI Foundry suggests), and Vertex AI's OpenAI-compatible endpoint
(an OpenAI client whose base_url is https://aiplatform.googleapis.com/v1beta1/projects/<p>/locations/<l>/
endpoints/openapi, or the same on a regional <region>-aiplatform.googleapis.com host).

    from openai import OpenAI
    import inference_tracker

    client = inference_tracker.openai.track(OpenAI(base_url=..., api_key=...))
    response = client.responses.parse(model="gpt-5.4-mini", input=..., text_format=...)

Tracked, one "call" record each (the whole call is seen here): responses.create / parse,
chat.completions.create / parse (and beta.chat.completions.parse), embeddings.create, moderations.create,
images.generate / edit, and a raw client.post() to an images route ("/images/generations",
"/images/edits"), which is how Azure AI Foundry's other image models (FLUX, for example) are called. The
with_raw_response and with_streaming_response versions of responses, chat.completions, embeddings and
moderations .create are tracked too, when the app parses the answer (response.parse()), which is how the
OpenAI Agents SDK streams. A copy of the client (with_options, copy) is tracked like the client.
Everything else passes through unchanged. The call itself is never changed, and tracking never raises: a
problem in it is logged and the call goes on.

When the app stops waiting (its task is cancelled, or a timeout around the call runs out), the call is recorded
with status "cancelled" (error_type the exception's class name): the provider may still have answered, and billed.

Streamed calls (stream=True) are recorded when the stream ends, with the usage its last chunk or event
reports. A Responses stream always reports it. A chat completions stream reports it only when the app
asked for stream_options={"include_usage": True}; it isn't added here, because it adds a chunk the app
would then receive, and some OpenAI-compatible servers refuse it. Without it the call is recorded without
usage.

A parse call whose answer doesn't fit the format the app asked for raises after the provider has answered
and billed. So parse calls go through the SDK's own with_raw_response, and the answer is parsed here, the
same way the SDK does it: when parsing fails, the record still gets the usage the answer reports, and the
app gets the same error as before.

On the record: the model the response names (the one billed; the requested name, an Azure deployment for
example, goes in settings when it differs), the response id (the x-request-id header when the answer has
no id, as with embeddings), the usage the response reports (tokens by kind, images), and settings that
change the price (reasoning effort, service tier, image size, quality, count). Never prompts, images or
outputs. For Azure, settings also hold the resource the call went to (the first part of its host name),
which is how its bill is split. For Vertex, they hold the project and location, the model is named as
Vertex names it (without the "google/" prefix), and the tokens are named as the google wrapper names them
(input_text_tokens, output_text_tokens, thinking_tokens...), so Vertex's bill is split the same way.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from . import _client
from ._proxy import Path, Proxy, get, number, put
from ._record import Record, cancelled, code, error_name

logger = logging.getLogger("inference_tracker")

OPENAI = "openai"
AZURE = "azure.ai.openai"
VERTEX = "gcp.vertex_ai"  # the same name google.py uses: one bill, one provider

# Calls that also have with_raw_response / with_streaming_response versions tracked here.
_CREATES = (
    ("responses", "create"),
    ("chat", "completions", "create"),
    ("embeddings", "create"),
    ("moderations", "create"),
)
RAW = ("with_raw_response", "with_streaming_response")
PATHS = frozenset(
    {
        *_CREATES,
        ("responses", "parse"),
        ("chat", "completions", "parse"),
        ("beta", "chat", "completions", "parse"),
        ("images", "generate"),
        ("images", "edit"),
        ("post",),  # only a raw call to an images route, see _images_post
        ("with_options",),  # a copy of the client, tracked too
        ("copy",),
    }
    | {c[:-1] + (raw, c[-1]) for c in _CREATES for raw in RAW}
)
# Request arguments copied onto the record because they change the price.
SETTINGS = ("service_tier", "size", "quality", "n", "background", "output_format", "input_fidelity")
# Content parts that aren't text. A Vertex call without any has only text tokens in; see _as_vertex.
_MEDIA = {"image_url", "input_image", "input_audio", "input_file", "file", "image", "audio", "video_url"}


@dataclass
class _Where:
    provider: str
    place: dict[str, str] = field(default_factory=dict)  # azure_resource, or gcp_project and gcp_location


@dataclass
class _Call:
    record: Record
    clock: float
    text_in: bool = True  # no image, audio or file in the request (only looked at for Vertex)
    text_out: bool = True  # no output modality other than text was asked for (only looked at for Vertex)


def track(client: Any) -> Any:
    """Wraps an openai.OpenAI, AsyncOpenAI, AzureOpenAI or AsyncAzureOpenAI. See the module docstring."""
    where = _where(client)

    def wrap(path: Path, method: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
        if path == ("post",):
            return _images_post(method, is_async, where)
        if path in (("with_options",), ("copy",)):
            return lambda *args, **kwargs: track(method(*args, **kwargs))
        if len(path) > 2 and path[-2] in RAW:
            return _raw_variant(path, method, is_async, where)
        raw = _raw_parse(path, method)
        if is_async:

            async def tracked_async(*args: Any, **kwargs: Any) -> Any:
                call = _start(where, kwargs)
                try:
                    if kwargs.get("stream"):
                        return _watch(call, path, await method(*args, **kwargs))
                    answer = await (raw or method)(*args, **kwargs)
                except BaseException as e:
                    _end(call, path, None, e)
                    raise
                if raw is None:
                    response = answer
                else:
                    try:
                        response = answer.parse()
                        if inspect.isawaitable(response):
                            response = await response
                    except BaseException as e:
                        _end(call, path, _body(answer), e)
                        raise
                _end(call, path, response, None)
                return response

            return tracked_async

        def tracked(*args: Any, **kwargs: Any) -> Any:
            call = _start(where, kwargs)
            try:
                if kwargs.get("stream"):
                    return _watch(call, path, method(*args, **kwargs))
                answer = (raw or method)(*args, **kwargs)
            except BaseException as e:
                _end(call, path, None, e)
                raise
            if raw is None:
                response = answer
            else:
                try:
                    response = answer.parse()
                except BaseException as e:
                    _end(call, path, _body(answer), e)
                    raise
            _end(call, path, response, None)
            return response

        return tracked

    return Proxy(client, PATHS, wrap)


def _raw_parse(path: Path, method: Callable[..., Any]) -> Callable[..., Any] | None:
    """The SDK's with_raw_response version of a parse method: it sends the same request and returns the
    answer unparsed (its parse() then does what the plain method would have). None when there isn't one."""
    if path[-1] != "parse":
        return None
    try:
        raw = getattr(getattr(method, "__self__", None), "with_raw_response", None)
        found = getattr(raw, "parse", None)
        return found if callable(found) else None
    except Exception:
        return None


def _body(answer: Any) -> Any:
    """The answer's JSON body, for its id, model and usage. None when it can't be read."""
    try:
        return answer.http_response.json()
    except Exception:
        logger.debug("inference_tracker: could not read the body of an answer that failed to parse", exc_info=True)
        return None


def _raw_variant(path: Path, method: Callable[..., Any], is_async: bool, where: _Where) -> Callable[..., Any]:
    """x.with_raw_response.create(...) returns the HTTP answer; x.with_streaming_response.create(...) returns a
    context manager that sends the request when entered and gives the HTTP answer. Either way the call is
    seen when the app parses the answer (answer.parse()): its result is recorded, or, for a stream, watched
    as the app reads it. The answer object is the SDK's own, with only its parse() looked through. An answer
    the app never parses (it reads the bytes itself) isn't recorded."""
    plain = path[:-2] + path[-1:]  # ("responses", "create")

    def look_through(call: _Call | None, answer: Any, stream: bool) -> Any:
        if call is None:
            return answer
        try:
            original = answer.parse
        except Exception:
            return answer
        seen: list[bool] = []

        def done(result: Any) -> Any:
            if seen:
                return result  # parse() again: the SDK returns its cached result, already recorded
            seen.append(True)
            if stream:
                return _watch(call, plain, result)
            _end(call, plain, result, None)
            return result

        def failed(e: BaseException) -> None:
            if not seen:
                seen.append(True)
                _end(call, plain, None if stream else _body(answer), e)

        async def finish_async(pending: Any) -> Any:
            try:
                result = await pending
            except BaseException as e:
                failed(e)
                raise
            return done(result)

        def parse(*args: Any, **kwargs: Any) -> Any:
            try:
                result = original(*args, **kwargs)
            except BaseException as e:
                failed(e)
                raise
            return finish_async(result) if inspect.isawaitable(result) else done(result)

        try:
            answer.parse = parse
        except Exception:
            logger.debug("inference_tracker: could not look through an openai raw answer", exc_info=True)
        return answer

    if path[-2] == "with_streaming_response":

        def streaming(*args: Any, **kwargs: Any) -> Any:
            manager = method(*args, **kwargs)  # nothing is sent until the app enters it
            try:
                if hasattr(manager, "_api_request"):  # AsyncResponseContextManager: an awaitable request
                    call = _start(where, kwargs)
                    pending = manager._api_request

                    async def request() -> Any:
                        try:
                            answer = await pending
                        except BaseException as e:
                            _end(call, plain, None, e)
                            raise
                        return look_through(call, answer, bool(kwargs.get("stream")))

                    manager._api_request = request()
                elif hasattr(manager, "_request_func"):  # ResponseContextManager: a request function
                    call = _start(where, kwargs)
                    send = manager._request_func

                    def request_now() -> Any:
                        try:
                            answer = send()
                        except BaseException as e:
                            _end(call, plain, None, e)
                            raise
                        return look_through(call, answer, bool(kwargs.get("stream")))

                    manager._request_func = request_now
            except Exception:
                logger.exception("inference_tracker: could not track an openai with_streaming_response call")
            return manager

        return streaming

    if is_async:

        async def raw_async(*args: Any, **kwargs: Any) -> Any:
            call = _start(where, kwargs)
            try:
                answer = await method(*args, **kwargs)
            except BaseException as e:
                _end(call, plain, None, e)
                raise
            return look_through(call, answer, bool(kwargs.get("stream")))

        return raw_async

    def raw(*args: Any, **kwargs: Any) -> Any:
        call = _start(where, kwargs)
        try:
            answer = method(*args, **kwargs)
        except BaseException as e:
            _end(call, plain, None, e)
            raise
        return look_through(call, answer, bool(kwargs.get("stream")))

    return raw


def _watch(call: _Call | None, path: Path, stream: Any) -> Any:
    """Returns the same stream, read through a generator that looks at each chunk or event on its way to the
    app, and records the call when the usage arrives (the last chunk, or the Responses API's
    response.completed event) or the stream ends, fails or is closed early. A stream that can't be read
    this way (not the SDK's Stream) is recorded now, without usage."""
    if call is None:
        return stream
    watcher = _Stream(call, path, stream)
    try:
        source = stream._iterator
        stream._iterator = watcher.read_async(source) if hasattr(source, "__anext__") else watcher.read(source)
    except Exception:
        logger.debug("inference_tracker: could not watch an openai stream; recorded without usage", exc_info=True)
        watcher.finish(None)
    return stream


class _Stream:
    """What a stream has said so far: the answer's id, model and usage, as an answer would hold them."""

    def __init__(self, call: _Call, path: Path, stream: Any):
        self.call = call
        self.path = path
        self.stream = stream
        self.answer: dict[str, Any] = {}
        self.images = 0
        self.error_type: str | None = None
        self.done = False

    def see(self, item: Any) -> None:
        try:
            kind = get(item, "type")
            response = get(item, "response")  # a Responses event that carries the whole response
            for source in (item, response):
                for name in ("id", "model", "usage"):
                    value = get(source, name)
                    if value is not None and (name != "id" or isinstance(value, str)):
                        self.answer[name] = value
            if not isinstance(kind, str):
                # A chat completions chunk. The usage-only chunk (no choices) is the last one; some
                # OpenAI-compatible servers put the running usage on every chunk, so the latest one is kept.
                if get(item, "usage") is not None and not get(item, "choices"):
                    self.finish(None)
                return
            if kind.startswith("image_") and kind.endswith(".completed"):
                self.images += 1
            elif kind == "response.failed":
                self.error_type = code(get(response, "error", "code")) or "failed"
            elif kind == "error":
                self.error_type = code(get(item, "code")) or "error"
            if kind in ("response.completed", "response.incomplete", "response.failed", "error"):
                self.finish(None)
        except Exception:
            logger.debug("inference_tracker: could not read an openai stream chunk", exc_info=True)

    def finish(self, error: BaseException | None) -> None:
        if self.done:
            return
        self.done = True
        try:
            if not isinstance(self.answer.get("id"), str):
                self.answer["id"] = get(self.stream, "response", "headers").get("x-request-id")
        except Exception:
            pass
        if self.path[0] == "images":
            self.answer["data"] = [None] * self.images
        _end(self.call, self.path, self.answer, error, self.error_type)

    def read(self, source: Any) -> Any:
        try:
            for item in source:
                self.see(item)
                yield item
        except GeneratorExit:
            raise  # closed early: recorded below with what was seen
        except BaseException as e:
            self.finish(e)
            raise
        finally:
            self.finish(None)

    async def read_async(self, source: Any) -> Any:
        try:
            async for item in source:
                self.see(item)
                yield item
        except GeneratorExit:
            raise
        except BaseException as e:
            self.finish(e)
            raise
        finally:
            self.finish(None)


def _images_post(method: Callable[..., Any], is_async: bool, where: _Where) -> Callable[..., Any]:
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
            call = _start(where, body)
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
        call = _start(where, body)
        try:
            response = method(*args, **kwargs)
        except BaseException as e:
            _end(call, ("images",), None, e)
            raise
        _end(call, ("images",), response, None)
        return response

    return tracked


def _where(client: Any) -> _Where:
    """Which bill the client's calls are on, and where on it: OpenAI; Azure, with the resource name (first
    label of the host); or Vertex AI, with the project and location its base_url names."""
    try:
        url = urlparse(str(getattr(client, "base_url", "") or ""))
        host = url.hostname or ""
        if host == "aiplatform.googleapis.com" or host.endswith("-aiplatform.googleapis.com"):
            found = re.search(r"/projects/([^/]+)/locations/([^/]+)", url.path)
            return _Where(VERTEX, {"gcp_project": found[1], "gcp_location": found[2]} if found else {})
        if "azure" in type(client).__name__.lower() or host.endswith(".azure.com"):
            resource = host.split(".")[0]
            return _Where(AZURE, {"azure_resource": resource} if resource else {})
    except Exception:
        logger.debug("inference_tracker: could not read the openai client's base_url", exc_info=True)
    return _Where(OPENAI)


def _vertex_model(model: Any) -> Any:
    """Vertex's OpenAI-compatible endpoint names Google's models "google/<model>"; the bill and the google
    wrapper name them without the prefix. Other publishers' models keep theirs."""
    if isinstance(model, str) and model.startswith("google/"):
        return model[len("google/") :]
    return model


def _start(where: _Where, kwargs: dict[str, Any]) -> _Call | None:
    if not _client.active():
        return None  # nothing is recorded: nothing is looked at
    try:
        settings = {k: kwargs[k] for k in SETTINGS if _given(kwargs.get(k))}
        effort = get(kwargs.get("reasoning"), "effort") or kwargs.get("reasoning_effort")
        if _given(effort):
            settings["reasoning_effort"] = effort
        settings.update(where.place)
        model = kwargs.get("model") if _given(kwargs.get("model")) else None
        if where.provider == VERTEX:
            model = _vertex_model(model)
        record = Record.new(
            where.provider, str(model) if model is not None else None, started_at=time.time(), settings=settings
        )
        call = _Call(record, time.monotonic())
        if where.provider == VERTEX:
            call.text_in = not any(
                _has_media(kwargs.get(k)) for k in ("messages", "input", "instructions") if k in kwargs
            )
            modalities = kwargs.get("modalities")
            call.text_out = not isinstance(modalities, list) or all(m == "text" for m in modalities)
        return call
    except Exception:
        logger.exception("inference_tracker: could not start tracking an openai call")
        return None


def _given(value: Any) -> bool:
    """Not None, and not the SDK's "not given" markers (openai.omit, NOT_GIVEN), which the OpenAI Agents SDK
    passes for every setting it leaves out."""
    return value is not None and type(value).__name__ not in ("Omit", "NotGiven")


def _has_media(value: Any, depth: int = 0) -> bool:
    """Whether a request's messages or input hold a part that isn't text (an image, audio, a file). Only the
    parts' "type" is looked at; nothing is kept."""
    if depth > 6:
        return False
    if isinstance(value, (list, tuple)):
        return any(_has_media(v, depth + 1) for v in value)
    kind = get(value, "type") if not isinstance(value, str) else None
    if isinstance(kind, str) and kind in _MEDIA:
        return True
    content = get(value, "content") if not isinstance(value, str) else None
    return isinstance(content, (list, tuple)) and _has_media(content, depth + 1)


def _end(
    call: _Call | None, path: Path, response: Any, error: BaseException | None, error_type: str | None = None
) -> None:
    """Records the call. error_type without an error: the provider said the call failed (a failed stream)."""
    if call is None:
        return
    record = call.record
    try:
        record.duration_s = round(time.monotonic() - call.clock, 3)
        # An error can still come with an answer (a parse call whose answer didn't fit the format): the
        # provider billed it, so its id, model and usage go on the record too.
        if error is not None:
            record.status = "cancelled" if cancelled(error) else "error"
            record.error_type = code(getattr(error, "code", None)) or error_name(error)
            record.request_id = getattr(error, "request_id", None) or get(response, "id") or None
        else:
            record.status = "error" if error_type else "ok"
            record.error_type = error_type
            record.request_id = get(response, "id") or getattr(response, "_request_id", None)
        if response is not None:
            answered = get(response, "model")
            if record.provider == VERTEX:
                answered = _vertex_model(answered)
            if isinstance(answered, str) and answered and answered != record.model:
                if record.model:
                    record.settings["requested_model"] = record.model
                record.model = answered
            record.units = _units(path, response)
            if record.provider == VERTEX:
                record.units = _as_vertex(record.units, get(response, "usage", "total_tokens"), call)
        _client.record(record)
    except Exception:
        logger.exception("inference_tracker: could not record an openai call")


def _units(path: Path, response: Any) -> dict[str, float]:
    usage = get(response, "usage")
    units: dict[str, float] = {}
    if path[0] == "moderations":
        return units  # free today, and the answer reports no usage: the record is the call itself
    if path[0] == "embeddings":
        put(units, "input_tokens", get(usage, "prompt_tokens"))
        return units
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


def _as_vertex(units: dict[str, float], total: Any, call: _Call) -> dict[str, float]:
    """OpenAI's names for a Vertex call's tokens, turned into the google wrapper's: output_tokens without
    thinking, thinking_tokens apart, and the text part of input and output named as such
    (input_text_tokens, output_text_tokens), which is what Vertex's bill lines are split by.

    OpenAI counts reasoning inside the output tokens. Vertex's endpoint may count it apart, or report it only
    in the total; the total says which. When the request had an image, audio or a file in it, the input
    isn't all text, and only input_tokens is recorded for it."""
    out = dict(units)
    inp = number(out.get("input_tokens"))
    output = number(out.get("output_tokens"))
    reasoning = number(out.pop("reasoning_tokens", None))
    total = number(total)
    thinking = None
    if inp is not None and output is not None:
        if reasoning is not None:
            thinking = reasoning
            counted_apart = total is not None and total == inp + output + reasoning
            if not counted_apart and output >= reasoning:
                output -= reasoning
        elif total is not None and total > inp + output:
            thinking = total - inp - output  # reported only in the total
    if thinking:
        out["thinking_tokens"] = thinking
    if output is not None:
        out["output_tokens"] = output
        if call.text_out:
            out["output_text_tokens"] = output
    if inp is not None and call.text_in:
        out["input_text_tokens"] = inp
    return out
