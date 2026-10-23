"""The litellm callback with the real litellm package (and the OpenAI Agents SDK's LitellmModel), with Bedrock and
OpenAI stood in by an httpx mock: no network, no AWS. Skipped when litellm isn't installed (it isn't a dependency of
inference_tracker)."""

import asyncio
import binascii
import contextvars
import json
import struct
import threading
import time

import pytest

litellm = pytest.importorskip("litellm")
httpx = pytest.importorskip("httpx")

import inference_tracker  # noqa: E402
import inference_tracker.litellm as tracker  # noqa: E402
from inference_tracker import context  # noqa: E402

MODEL = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
PROFILE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
MESSAGES = [{"role": "user", "content": "a photo of my kid"}]
USAGE = {
    "inputTokens": 100,
    "outputTokens": 20,
    "totalTokens": 1320,
    "cacheReadInputTokens": 1000,
    "cacheWriteInputTokens": 200,
}
UNITS = {"input_tokens": 1300, "output_tokens": 20, "cached_input_tokens": 1000, "cache_write_input_tokens": 200}
CONVERSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": USAGE,
    "metrics": {"latencyMs": 300},
}
OPENAI_CHAT = {
    "id": "chatcmpl-AbC123",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5.4-mini-2026-03-01",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 10,
        "total_tokens": 60,
        "prompt_tokens_details": {"cached_tokens": 32},
    },
}


def frame(event_type: str, payload: dict) -> bytes:
    """One message of AWS's event stream encoding, the way Bedrock streams converse-stream."""

    def header(name: str, value: str) -> bytes:
        n, v = name.encode(), value.encode()
        return struct.pack(">B", len(n)) + n + b"\x07" + struct.pack(">H", len(v)) + v

    headers = header(":event-type", event_type) + header(":content-type", "application/json")
    headers += header(":message-type", "event")
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 12 + len(headers) + len(body) + 4, len(headers))
    message = prelude + struct.pack(">I", binascii.crc32(prelude)) + headers + body
    return message + struct.pack(">I", binascii.crc32(message))


STREAM = b"".join(
    [
        frame("messageStart", {"role": "assistant"}),
        frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hi"}}),
        frame("contentBlockStop", {"contentBlockIndex": 0}),
        frame("messageStop", {"stopReason": "end_turn"}),
        frame("metadata", {"usage": USAGE, "metrics": {"latencyMs": 300}}),
    ]
)


class FakeProviders:
    """Answers like Bedrock's runtime (converse, converse-stream) and OpenAI's API. `refuse` makes Bedrock a 400."""

    def __init__(self):
        self.refuse = False
        self.hang = False  # never answer (an async request only): for the app giving up on the wait
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.hang:

            async def never():
                await asyncio.sleep(60)

            return never()
        if request.url.host == "api.openai.com":
            return httpx.Response(200, json=OPENAI_CHAT, headers={"x-request-id": "req-oa"})
        if self.refuse:
            headers = {
                "x-amzn-requestid": "rid-err",
                "x-amzn-errortype": "ValidationException:http://internal.amazon.com/",
            }
            return httpx.Response(400, json={"message": "the input quotes a photo of my kid"}, headers=headers)
        if request.url.path.endswith("/converse-stream"):
            headers = {"x-amzn-requestid": "rid-stream", "content-type": "application/vnd.amazon.eventstream"}
            return httpx.Response(200, content=STREAM, headers=headers)
        return httpx.Response(200, json=CONVERSE, headers={"x-amzn-requestid": "rid-1"})


@pytest.fixture
def fake(monkeypatch):
    providers = FakeProviders()
    mock = httpx.MockTransport(providers)

    async def handle_async(self, request):
        return await mock.handle_async_request(request)

    # LiteLLM and the openai SDK make their own httpx clients; every one of them ends up in these two methods.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: mock.handle_request(request))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "suppress_debug_info", True)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-a-secret")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    return providers


@pytest.fixture
def records(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    tracker.track()
    return out


def wait_for(records, n=1, timeout=5.0):
    """LiteLLM records a sync call on its own logging thread, after the call returns."""
    deadline = time.monotonic() + timeout
    while len(records) < n and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)  # and nothing more comes
    return records


async def async_wait_for(records, n=1, timeout=5.0):
    deadline = time.monotonic() + timeout
    while len(records) < n and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    return records


def test_track_adds_one_callback():
    first = tracker.track()
    assert tracker.track() is first
    assert litellm.callbacks.count(first) == 1


def test_bedrock_call_is_one_record_with_usage_by_kind_and_who(fake, records):
    with context(user_id="u1", task_id="t1"):
        response = litellm.completion(model=MODEL, messages=MESSAGES, reasoning_effort="low")
    assert response.choices[0].message.content == "hi"
    (r,) = wait_for(records)
    assert (r.provider, r.model, r.phase, r.status, r.request_id) == ("aws.bedrock", PROFILE, "call", "ok", "rid-1")
    assert r.units == UNITS | {"reasoning_tokens": 0}
    assert r.settings == {"aws_region": "us-east-1", "reasoning_effort": "low"}
    assert (r.user_id, r.task_id) == ("u1", "t1")
    assert r.duration_s is not None and r.duration_s >= 0 and r.started_at is not None
    assert "kid" not in repr(r.to_wire())


def test_bedrock_route_prefix_and_region_argument(fake, records):
    litellm.completion(
        model="bedrock/converse/" + PROFILE, messages=MESSAGES, aws_region_name="eu-west-1", service_tier="flex"
    )
    (r,) = wait_for(records)
    assert r.model == PROFILE
    assert r.settings["aws_region"] == "eu-west-1"
    assert r.settings["service_tier"] == "flex"
    assert fake.requests[0].url.host == "bedrock-runtime.eu-west-1.amazonaws.com"


def test_bedrock_async_call(fake, records):
    async def main():
        with context(user_id="u2"):
            await litellm.acompletion(model=MODEL, messages=MESSAGES)
        return await async_wait_for(records)

    (r,) = asyncio.run(main())
    assert (r.provider, r.model, r.request_id, r.status) == ("aws.bedrock", PROFILE, "rid-1", "ok")
    assert r.units["cache_write_input_tokens"] == 200
    assert r.user_id == "u2"


def test_bedrock_stream_is_recorded_at_its_end_for_who_started_it(fake, records):
    with context(user_id="u3", task_id="t3"):
        stream = litellm.completion(model=MODEL, messages=MESSAGES, stream=True)
    # Read after the block: the record still gets the context the call was made in.
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert text == "hi"
    (r,) = wait_for(records)
    assert (r.provider, r.model, r.request_id, r.status) == ("aws.bedrock", PROFILE, "rid-stream", "ok")
    assert r.units == UNITS | {"reasoning_tokens": 0}
    assert (r.user_id, r.task_id) == ("u3", "t3")


def test_bedrock_async_stream(fake, records):
    async def main():
        with context(user_id="u4"):
            stream = await litellm.acompletion(model=MODEL, messages=MESSAGES, stream=True)
        async for _ in stream:
            pass
        return await async_wait_for(records)

    (r,) = asyncio.run(main())
    assert (r.request_id, r.user_id) == ("rid-stream", "u4")
    assert r.units["cached_input_tokens"] == 1000


def test_bedrock_error_is_recorded_by_its_code_and_raised(fake, records):
    fake.refuse = True
    with context(user_id="u5"), pytest.raises(litellm.BadRequestError):
        litellm.completion(model=MODEL, messages=MESSAGES)
    (r,) = wait_for(records)
    assert (r.provider, r.model, r.status) == ("aws.bedrock", PROFILE, "error")
    assert (r.error_type, r.request_id, r.user_id) == ("ValidationException", "rid-err", "u5")
    assert r.units == {}
    assert "kid" not in repr(r.to_wire())


def test_bedrock_async_error(fake, records):
    fake.refuse = True

    async def main():
        with pytest.raises(litellm.BadRequestError):
            await litellm.acompletion(model=MODEL, messages=MESSAGES)
        return await async_wait_for(records)

    (r,) = asyncio.run(main())
    assert (r.status, r.error_type, r.request_id) == ("error", "ValidationException", "rid-err")


def test_openai_through_litellm_names_it_like_the_openai_wrapper(fake, records):
    litellm.completion(model="openai/gpt-5.4-mini", messages=MESSAGES, api_key="sk-test")
    (r,) = wait_for(records)
    assert (r.provider, r.model, r.request_id) == ("openai", "gpt-5.4-mini-2026-03-01", "chatcmpl-AbC123")
    assert r.settings == {"requested_model": "gpt-5.4-mini"}
    assert r.units == {"input_tokens": 50, "output_tokens": 10, "cached_input_tokens": 32}


def test_cache_hits_and_other_calls_are_not_recorded(records):
    tracker._after({"call_type": "completion", "cache_hit": True, "custom_llm_provider": "bedrock"}, None, 0, 1)
    tracker._after({"call_type": "acreate_batch", "custom_llm_provider": "openai"}, None, 0, 1)
    assert records == []


def test_context_is_the_one_at_pre_call_even_on_another_thread(records):
    kwargs = {
        "litellm_call_id": "call-1",
        "call_type": "acompletion",
        "custom_llm_provider": "bedrock",
        "model": PROFILE,
        "additional_args": {"api_base": "https://bedrock-runtime.us-west-2.amazonaws.com/model/x%3A1/converse"},
    }
    with context(user_id="u6", parent_provider="fal", parent_request_id="fal-1"):
        tracker._before(kwargs)
    with context(user_id="someone-else"):
        thread = threading.Thread(target=contextvars.Context().run, args=(tracker._after, kwargs, None, 0.0, 1.5))
        thread.start()
        thread.join()
    (r,) = records
    assert (r.user_id, r.parent_provider, r.parent_request_id) == ("u6", "fal", "fal-1")
    assert (r.model, r.settings, r.duration_s) == ("x:1", {"aws_region": "us-west-2"}, 1.5)
    assert "call-1" not in tracker._open


def test_tracking_never_raises(records, caplog):
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("broken")

    tracker._before(Broken())
    tracker._after({"call_type": "completion", "custom_llm_provider": "bedrock", "model": Broken()}, Broken(), 0, 1)
    tracker._after(Broken(), Broken(), Broken(), Broken())


def test_openai_agents_litellm_model_run(fake, records):
    agents = pytest.importorskip("agents")
    from agents.extensions.models.litellm_model import LitellmModel

    agents.set_tracing_disabled(True)  # the Agents SDK would otherwise send its traces to OpenAI
    try:
        agent = agents.Agent(name="helper", instructions="Answer briefly.", model=LitellmModel(model=MODEL))

        async def main():
            with context(user_id="u7", task_id="t7"):
                result = await agents.Runner.run(agent, "a photo of my kid")
            await async_wait_for(records)
            return result

        result = asyncio.run(main())
    finally:
        agents.set_tracing_disabled(False)
    assert result.final_output == "hi"
    (r,) = records
    assert (r.provider, r.model, r.request_id, r.status) == ("aws.bedrock", PROFILE, "rid-1", "ok")
    assert r.units["input_tokens"] == 1300
    assert (r.user_id, r.task_id) == ("u7", "t7")


@pytest.fixture
def ingest(monkeypatch):
    """A real client (no background thread) that sends to a fake ingest: what a flush sends is what it gets."""
    from inference_tracker._transport import Result

    bodies = []

    def post(url, key, body, timeout):
        bodies.append(json.loads(json.dumps(body)))
        return Result("sent", 200)

    inference_tracker._client._client = None
    inference_tracker._client._settings = inference_tracker._client._NOT_YET
    inference_tracker.init("https://ingest.test", "key", "myapp", background=False, post=post)
    tracker.track()
    yield lambda: [r for b in bodies for r in b["records"]]
    inference_tracker._client._client = None
    inference_tracker._client._settings = inference_tracker._client._NOT_YET


def test_a_sync_call_is_in_the_flush_right_after_it(fake, ingest, monkeypatch):
    # LiteLLM records a sync call on its logging thread after the call returned; made slow here, flush waits for it.
    after = tracker._after

    def late(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            time.sleep(0.3)
        after(*args, **kwargs)

    monkeypatch.setattr(tracker, "_after", late)
    with context(user_id="u9", flush=True):
        litellm.completion(model=MODEL, messages=MESSAGES)
    (r,) = ingest()  # in that flush, not a later one
    assert (r["gen_ai.response.id"], r["user_id"], r["status"]) == ("rid-1", "u9", "ok")


def test_an_async_call_is_recorded_before_it_returns(fake, ingest, monkeypatch):
    # As with the Agents SDK's Runner.run_sync: the loop may end, or the process freeze, right after.
    seen = []

    async def main():
        with context(user_id="u10", flush=True):
            await litellm.acompletion(model=MODEL, messages=MESSAGES)
        seen.extend(ingest())
        await asyncio.sleep(0.3)  # LiteLLM's own success callback for it comes now, and adds nothing

    asyncio.run(main())
    (r,) = seen
    assert (r["gen_ai.response.id"], r["user_id"], r["status"]) == ("rid-1", "u10", "ok")
    assert r["units"]["input_tokens"] == 1300
    inference_tracker.flush()
    assert len(ingest()) == 1


def test_an_async_call_the_app_stops_waiting_for_is_cancelled(fake, records):
    fake.hang = True

    async def main():
        with context(user_id="u11"):
            task = asyncio.create_task(litellm.acompletion(model=MODEL, messages=MESSAGES))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

    asyncio.run(main())
    (r,) = records
    assert (r.provider, r.model, r.status, r.error_type) == ("aws.bedrock", PROFILE, "cancelled", "CancelledError")
    assert (r.user_id, r.request_id) == ("u11", None)


def test_a_timeout_around_an_async_call_is_cancelled_too(fake, records):
    fake.hang = True

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(litellm.acompletion(model=MODEL, messages=MESSAGES), 0.3)

    asyncio.run(main())
    (r,) = records
    assert (r.status, r.error_type) == ("cancelled", "CancelledError")


def test_nothing_is_done_without_a_client(fake, monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: False)
    tracker.track()
    tracker._open.clear()
    litellm.completion(model=MODEL, messages=MESSAGES)
    time.sleep(0.3)
    assert out == [] and not tracker._open
