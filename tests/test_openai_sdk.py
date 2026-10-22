"""The openai wrapper around the real openai package, with the servers (OpenAI, Azure, Vertex AI's
OpenAI-compatible endpoint) stood in by an httpx mock. Skipped when openai isn't installed."""

import asyncio
import json

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

import inference_tracker  # noqa: E402
from inference_tracker import context  # noqa: E402
from inference_tracker.openai import track  # noqa: E402

VERTEX_GLOBAL = "https://aiplatform.googleapis.com/v1beta1/projects/acme-prod/locations/global/endpoints/openapi"
VERTEX_REGIONAL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/acme-prod/locations/us-central1/endpoints/openapi"
)
AZURE = "https://acme-resource.services.ai.azure.com/openai/v1/"


@pytest.fixture
def records(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    return out


class Server:
    """Answers every request with the next of `answers`: a JSON body, or a list of server-sent events."""

    def __init__(self, *answers, headers=None):
        self.answers = list(answers)
        self.headers = headers or {}
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, list):
            lines = []
            for event in answer:
                if isinstance(event, dict) and "type" in event:
                    lines.append(f"event: {event['type']}\n")
                lines.append(f"data: {json.dumps(event) if isinstance(event, dict) else event}\n\n")
            return httpx.Response(
                200, headers={"content-type": "text/event-stream", **self.headers}, content="".join(lines).encode()
            )
        return httpx.Response(200, headers=self.headers, json=answer)

    def body(self, i=-1):
        return json.loads(self.requests[i].content)


def client(server, base_url="https://api.openai.com/v1/", cls=None):
    cls = cls or openai.OpenAI
    http = (httpx.AsyncClient if cls is openai.AsyncOpenAI else httpx.Client)(transport=httpx.MockTransport(server))
    return cls(api_key="sk-test", base_url=base_url, http_client=http, max_retries=0)


def chat(model, usage, id="chatcmpl-1"):
    return {
        "id": id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "a short answer"}}
        ],
        "usage": usage,
    }


def chunks(model, usage=None, id="chatcmpl-2"):
    def chunk(delta, finish=None):
        return {
            "id": id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    out = [chunk({"role": "assistant", "content": "a short"}), chunk({"content": " answer"}, "stop")]
    if usage:
        base = {"id": id, "object": "chat.completion.chunk", "created": 0, "model": model}
        out.append({**base, "choices": [], "usage": usage})
    return out + ["[DONE]"]


# ---------- Vertex AI behind an OpenAI client ----------


def test_vertex_chat_is_vertex_with_googles_names_for_model_and_tokens(records):
    # Thinking reported only in the total: 120 in + 30 out + 250 thinking = 400.
    usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 400}
    server = Server(chat("google/gemini-3.1-flash", usage, id="vertex-resp-1"))
    with context(user_id="u1", task_id="t1"):
        out = track(client(server, VERTEX_GLOBAL)).chat.completions.create(
            model="google/gemini-3.1-flash", messages=[{"role": "user", "content": "a photo of my kid"}]
        )
    assert out.choices[0].message.content == "a short answer"
    assert str(server.requests[0].url) == VERTEX_GLOBAL + "/chat/completions"
    (r,) = records
    assert (r.provider, r.model, r.request_id, r.status) == ("gcp.vertex_ai", "gemini-3.1-flash", "vertex-resp-1", "ok")
    assert r.settings == {"gcp_project": "acme-prod", "gcp_location": "global"}
    assert r.units == {
        "input_tokens": 120,
        "output_tokens": 30,
        "thinking_tokens": 250,
        "input_text_tokens": 120,
        "output_text_tokens": 30,
    }
    assert (r.user_id, r.task_id) == ("u1", "t1")
    assert "kid" not in repr(r.to_wire())


def test_vertex_regional_async_with_an_image_and_reasoning_counted_in_the_output(records):
    # OpenAI's way: completion_tokens (80) includes reasoning_tokens (50); the total says so.
    usage = {
        "prompt_tokens": 1400,
        "completion_tokens": 80,
        "total_tokens": 1480,
        "completion_tokens_details": {"reasoning_tokens": 50},
        "prompt_tokens_details": {"cached_tokens": 1024},
    }
    server = Server(chat("google/gemini-3.1-pro", usage))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this photo?"},
                {"type": "image_url", "image_url": {"url": "gs://bucket/kid.jpg"}},
            ],
        }
    ]
    c = track(client(server, VERTEX_REGIONAL, cls=openai.AsyncOpenAI))
    asyncio.run(c.chat.completions.create(model="google/gemini-3.1-pro", messages=messages, reasoning_effort="low"))
    (r,) = records
    assert (r.provider, r.model) == ("gcp.vertex_ai", "gemini-3.1-pro")
    assert r.settings == {"reasoning_effort": "low", "gcp_project": "acme-prod", "gcp_location": "us-central1"}
    # Part of the input was an image: only its total is known.
    assert r.units == {
        "input_tokens": 1400,
        "output_tokens": 30,
        "thinking_tokens": 50,
        "cached_input_tokens": 1024,
        "output_text_tokens": 30,
    }
    assert "kid" not in repr(r.to_wire())


def test_vertex_reasoning_counted_apart_from_the_output(records):
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 35,
        "completion_tokens_details": {"reasoning_tokens": 20},
    }
    server = Server(chat("google/gemini-3.1-flash", usage))
    track(client(server, VERTEX_GLOBAL)).chat.completions.create(model="google/gemini-3.1-flash", messages=[])
    assert records[0].units["output_text_tokens"] == 5
    assert records[0].units["thinking_tokens"] == 20


def test_vertex_embeddings(records):
    server = Server(
        {
            "object": "list",
            "model": "google/gemini-embedding-001",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 9, "total_tokens": 9},
        }
    )
    track(client(server, VERTEX_GLOBAL)).embeddings.create(model="google/gemini-embedding-001", input=["hello"])
    (r,) = records
    assert (r.provider, r.model) == ("gcp.vertex_ai", "gemini-embedding-001")
    assert r.units == {"input_tokens": 9, "input_text_tokens": 9}


# ---------- moderations and embeddings ----------


@pytest.mark.parametrize(
    "base_url,provider,place",
    [("https://api.openai.com/v1/", "openai", {}), (AZURE, "azure.ai.openai", {"azure_resource": "acme-resource"})],
)
def test_moderation_is_a_record_with_its_id_and_model(records, base_url, provider, place):
    server = Server(
        {
            "id": "modr-1",
            "model": "omni-moderation-2024-09-26",
            "results": [
                {"flagged": False, "categories": {}, "category_scores": {}, "category_applied_input_types": {}}
            ],
        }
    )
    with context(user_id="u1"):
        out = track(client(server, base_url)).moderations.create(
            model="omni-moderation-latest", input="a photo of my kid"
        )
    assert out.results[0].flagged is False
    (r,) = records
    assert (r.provider, r.model, r.request_id, r.status, r.user_id) == (
        provider,
        "omni-moderation-2024-09-26",
        "modr-1",
        "ok",
        "u1",
    )
    assert r.settings == {"requested_model": "omni-moderation-latest", **place}
    assert r.units == {}
    assert "kid" not in repr(r.to_wire())


def test_embeddings_tokens_in_and_the_request_id_from_the_header(records):
    server = Server(
        {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}] * 2,
            "usage": {"prompt_tokens": 14, "total_tokens": 14},
        },
        headers={"x-request-id": "req_emb"},
    )
    out = track(client(server)).embeddings.create(model="text-embedding-3-small", input=["a", "b"])
    assert len(out.data) == 2
    (r,) = records
    assert (r.provider, r.model, r.request_id) == ("openai", "text-embedding-3-small", "req_emb")
    assert r.units == {"input_tokens": 14}


# ---------- streams ----------


def test_chat_stream_with_usage_is_recorded_when_it_ends(records):
    usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    server = Server(chunks("gpt-5.4-nano", usage))
    stream = track(client(server)).chat.completions.create(
        model="gpt-5.4-nano", messages=[], stream=True, stream_options={"include_usage": True}
    )
    assert isinstance(stream, openai.Stream)
    got = [c.choices[0].delta.content for c in stream if c.choices]
    assert got == ["a short", " answer"]
    (r,) = records
    assert (r.request_id, r.model, r.status) == ("chatcmpl-2", "gpt-5.4-nano", "ok")
    assert r.units == {"input_tokens": 50, "output_tokens": 10}


def test_chat_stream_without_usage_is_left_as_the_app_asked_and_recorded_without_usage(records):
    server = Server(chunks("gpt-5.4-nano"))
    stream = track(client(server)).chat.completions.create(model="gpt-5.4-nano", messages=[], stream=True)
    assert [c.choices[0].delta.content for c in stream] == ["a short", " answer"]  # no extra chunk
    assert "stream_options" not in server.body()
    (r,) = records
    assert (r.request_id, r.status, r.units) == ("chatcmpl-2", "ok", {})


def test_vertex_chat_stream_async(records):
    usage = {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 60}
    server = Server(chunks("google/gemini-3.1-flash", usage))
    c = track(client(server, VERTEX_GLOBAL, cls=openai.AsyncOpenAI))

    async def run():
        stream = await c.chat.completions.create(
            model="google/gemini-3.1-flash", messages=[], stream=True, stream_options={"include_usage": True}
        )
        assert isinstance(stream, openai.AsyncStream)
        return [x async for x in stream]

    assert len(asyncio.run(run())) == 3
    (r,) = records
    assert (r.provider, r.model) == ("gcp.vertex_ai", "gemini-3.1-flash")
    assert r.units == {
        "input_tokens": 40,
        "output_tokens": 8,
        "thinking_tokens": 12,
        "output_text_tokens": 8,
        "input_text_tokens": 40,
    }


def response(status="completed", usage=None):
    return {
        "id": "resp_9",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.4-mini-2026-03-17",
        "status": status,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "a short answer", "annotations": []}],
            }
        ],
        "usage": usage,
    }


USAGE = {
    "input_tokens": 700,
    "output_tokens": 20,
    "total_tokens": 720,
    "input_tokens_details": {"cached_tokens": 512},
    "output_tokens_details": {"reasoning_tokens": 8},
}


def events(status="completed"):
    started = response("in_progress")
    started["output"] = []
    return [
        {"type": "response.created", "sequence_number": 0, "response": started},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "a short answer",
            "logprobs": [],
        },
        {"type": f"response.{status}", "sequence_number": 2, "response": response(status, USAGE)},
    ]


def test_responses_stream_records_the_usage_of_the_completed_event(records):
    server = Server(events())
    with context(user_id="u1"):
        stream = track(client(server, AZURE)).responses.create(model="gpt-5.4-mini", input="x", stream=True)
    types = [e.type for e in stream]
    assert types == ["response.created", "response.output_text.delta", "response.completed"]
    (r,) = records
    assert (r.provider, r.request_id, r.model, r.status, r.user_id) == (
        "azure.ai.openai",
        "resp_9",
        "gpt-5.4-mini-2026-03-17",
        "ok",
        "u1",
    )
    assert r.units == {"input_tokens": 700, "output_tokens": 20, "cached_input_tokens": 512, "reasoning_tokens": 8}


def test_responses_stream_that_fails_is_an_error_with_the_usage(records):
    failed = events("failed")
    failed[-1]["response"]["error"] = {"code": "server_error", "message": "the message quotes the prompt"}
    server = Server(failed)
    list(track(client(server)).responses.create(model="gpt-5.4-mini", input="x", stream=True))
    (r,) = records
    assert (r.status, r.error_type, r.units["input_tokens"]) == ("error", "server_error", 700)


def test_stream_closed_early_is_still_recorded(records):
    server = Server(chunks("gpt-5.4-nano"))
    stream = track(client(server)).chat.completions.create(model="gpt-5.4-nano", messages=[], stream=True)
    next(iter(stream))
    stream._iterator.close()  # what happens to the generator when the app drops the stream
    assert len(records) == 1 and records[0].status == "ok"


# ---------- with_raw_response, with_streaming_response, with_options ----------


def test_with_streaming_response_stream_async(records):
    """How the OpenAI Agents SDK streams the Responses API."""
    server = Server(events(), headers={"x-request-id": "req_9"})
    c = track(client(server, cls=openai.AsyncOpenAI))

    async def run():
        manager = c.responses.with_streaming_response.create(model="gpt-5.4-mini", input="x", stream=True)
        async with manager as answer:
            assert answer.request_id == "req_9"
            stream = await answer.parse()
            return [e.type async for e in stream]

    assert asyncio.run(run())[-1] == "response.completed"
    (r,) = records
    assert (r.request_id, r.units["input_tokens"]) == ("resp_9", 700)


def test_with_raw_response_sync(records):
    server = Server(chat("gpt-5.4-nano", {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}))
    answer = track(client(server)).chat.completions.with_raw_response.create(model="gpt-5.4-nano", messages=[])
    assert records == []  # not parsed yet
    assert answer.parse().id == "chatcmpl-1"
    answer.parse()
    (r,) = records
    assert r.units == {"input_tokens": 5, "output_tokens": 1}


def test_a_copy_of_the_client_is_tracked_too(records):
    server = Server(chat("gpt-5.4-nano", {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}))
    c = track(client(server, AZURE)).with_options(max_retries=0, timeout=5)
    assert c.max_retries == 0
    c.chat.completions.create(model="gpt-5.4-nano", messages=[])
    assert records[0].provider == "azure.ai.openai"


def test_an_error_mid_stream_is_recorded_and_raised(records):
    def broken(request):
        def body():
            yield b'data: {"id": "chatcmpl-3", "object": "chat.completion.chunk", "created": 0, "model": "m", '
            yield b'"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": null}]}\n\n'
            raise httpx.ReadError("connection reset")

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    stream = track(client(broken)).chat.completions.create(model="m", messages=[], stream=True)
    with pytest.raises(openai.APIConnectionError):
        list(stream)
    (r,) = records
    assert (r.status, r.error_type, r.request_id) == ("error", "APIConnectionError", "chatcmpl-3")


def test_with_streaming_response_sync_without_a_stream(records):
    server = Server(chat("gpt-5.4-nano", {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}))
    c = track(client(server))
    with c.chat.completions.with_streaming_response.create(model="gpt-5.4-nano", messages=[]) as answer:
        assert answer.parse().choices[0].message.content == "a short answer"
    (r,) = records
    assert (r.request_id, r.units) == ("chatcmpl-1", {"input_tokens": 5, "output_tokens": 1})


def test_usage_on_every_chunk_keeps_the_last(records):
    """Some OpenAI-compatible servers report the running usage on every chunk."""
    stream = chunks("google/gemini-3.1-flash")[:-1]
    for i, chunk in enumerate(stream):
        chunk["usage"] = {"prompt_tokens": 40, "completion_tokens": 4 * (i + 1), "total_tokens": 40 + 4 * (i + 1)}
    server = Server(stream + ["[DONE]"])
    list(track(client(server, VERTEX_GLOBAL)).chat.completions.create(model="m", messages=[], stream=True))
    (r,) = records
    assert r.units["output_text_tokens"] == 8
