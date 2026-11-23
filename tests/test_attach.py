"""The tracking attached at init(): clients nobody wrapped are recorded, a wrapped one once, LiteLLM's not by us."""

import asyncio
import json
import sys
import textwrap

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")
requests = pytest.importorskip("requests")

from test_eachlabs import Adapter, FakeEachlabs  # noqa: E402
from test_openai_sdk import Server  # noqa: E402

import inference_tracker  # noqa: E402
from inference_tracker import _attach, context  # noqa: E402
from inference_tracker import eachlabs as tracked_eachlabs  # noqa: E402
from inference_tracker import openai as tracked_openai  # noqa: E402

CHAT = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    _attach.attach()
    return out


def make_openai(server, cls=None):
    cls = cls or openai.OpenAI
    http = (httpx.AsyncClient if cls is openai.AsyncOpenAI else httpx.Client)(transport=httpx.MockTransport(server))
    return cls(api_key="sk-test", http_client=http, max_retries=0)


def ask(client):
    return client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])


def test_an_openai_client_nobody_wrapped_is_recorded(sent):
    client = make_openai(Server(CHAT))
    with context(user_id="u1"):
        assert ask(client).choices[0].message.content == "hi"
    assert [(r.provider, r.model, r.request_id, r.user_id) for r in sent] == [
        ("openai", "gpt-4o-mini", "chatcmpl-1", "u1")
    ]
    assert sent[0].units["input_tokens"] == 5
    assert sent[0].code[0] == "test_attach.ask"


def test_a_client_made_before_init_is_recorded_too(monkeypatch):
    client = make_openai(Server(CHAT))  # before attach(), as a module-level client would be
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    _attach.attach()
    ask(client)
    assert len(out) == 1


def test_an_async_client_and_copies_are_recorded(sent):
    async def main():
        client = make_openai(Server(CHAT, CHAT), openai.AsyncOpenAI)
        await client.chat.completions.create(model="gpt-4o-mini", messages=[])
        await client.with_options(timeout=5).chat.completions.create(model="gpt-4o-mini", messages=[])

    asyncio.run(main())
    assert len(sent) == 2


def test_a_client_also_wrapped_with_track_is_recorded_once(sent):
    client = tracked_openai.track(make_openai(Server(CHAT)))
    ask(client)
    assert len(sent) == 1


def test_a_client_made_by_litellm_is_left_to_litellm(sent):
    made_by_litellm = {"__name__": "litellm.llms.openai", "make": make_openai, "Server": Server, "CHAT": CHAT}
    exec("client = make(Server(CHAT))", made_by_litellm)  # noqa: S102
    ask(made_by_litellm["client"])
    assert sent == []


def test_nothing_is_recorded_without_init(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: False)
    _attach.attach()
    ask(make_openai(Server(CHAT)))
    assert out == []


def test_a_plain_httpx_session_to_eachlabs_is_watched_on_its_first_request(sent):
    each = FakeEachlabs()
    http = httpx.Client(transport=httpx.MockTransport(each.httpx))  # never given to track_http
    created = http.post(
        "https://api.eachlabs.ai/v1/prediction", json={"model": "p-video-upscale", "input": {"duration": 5}}
    ).json()
    while http.get(f"https://api.eachlabs.ai/v1/prediction/{created['predictionID']}").json()["status"] != "success":
        pass
    assert [(r.phase, r.model, r.request_id) for r in sent] == [
        ("start", "p-video-upscale", "pred-1"),
        ("finish", "p-video-upscale", "pred-1"),
    ]
    assert sent[1].units["cost_usd"] == 0.1617


def test_a_plain_async_httpx_session_is_watched(sent):
    each = FakeEachlabs()

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(each.httpx)) as http:
            await http.post("https://api.eachlabs.ai/v1/prediction", json={"model": "m", "input": {}})

    asyncio.run(main())
    assert [r.phase for r in sent] == ["start"]


def test_a_requests_session_is_watched(sent):
    each = FakeEachlabs()
    http = requests.Session()
    http.mount("https://", Adapter(each))
    http.post("https://api.eachlabs.ai/v1/prediction", json={"model": "m", "input": {}})
    assert [r.phase for r in sent] == ["start"]


def test_track_http_settings_reach_sessions_nobody_wrapped(sent):
    tracked_eachlabs.track_http(httpx.Client(), settings=("seed",))
    each = FakeEachlabs()
    http = httpx.Client(transport=httpx.MockTransport(each.httpx))
    http.post("https://api.eachlabs.ai/v1/prediction", json={"model": "m", "input": {"seed": 7, "duration": 5}})
    assert sent[0].settings == {"seed": 7, "duration": 5}


def test_a_session_given_to_track_http_is_recorded_once(sent):
    each = FakeEachlabs()
    http = tracked_eachlabs.track_http(httpx.Client(transport=httpx.MockTransport(each.httpx)))
    http.post("https://api.eachlabs.ai/v1/prediction", json={"model": "m", "input": {}})
    assert len(sent) == 1


def test_other_hosts_pass_unseen(sent):
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    http.post("https://example.com/v1/prediction", json={})
    assert sent == []


def test_a_package_imported_later_is_attached_when_it_is(sent, tmp_path, monkeypatch):
    (tmp_path / "itk_late_sdk.py").write_text(
        textwrap.dedent("""
        class Client:
            pass
    """)
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    monkeypatch.setitem(_attach.TARGETS, "itk_late_sdk", lambda: seen.append(sys.modules["itk_late_sdk"].Client))
    _attach.attach()
    import itk_late_sdk  # noqa: F401

    assert len(seen) == 1


def test_a_package_that_changed_is_skipped_with_a_warning(sent, monkeypatch, caplog):
    def broken():
        raise AttributeError("no such attribute")

    monkeypatch.setitem(_attach.TARGETS, "json", broken)
    monkeypatch.setattr(_attach, "_done", set(_attach._done) - {"json"})
    _attach.attach()
    assert any("couldn't attach to json" in r.getMessage() for r in caplog.records)


# --- OpenAI's API called without the openai package -----------------------------------------------------------

DECISION = {
    "model": "gpt-6-luna",
    "answers": [{"type": "predicate", "name": "safe", "probability": 0.97}],
    "usage": {
        "input_tokens": 40,
        "input_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 0},
        "output_tokens": 1,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 41,
    },
}


def openai_over_http(*answers, status=200):
    seen = []

    def answer(request):
        seen.append(request)
        body = answers[len(seen) - 1]
        return httpx.Response(status, headers={"x-request-id": f"req_{len(seen)}"}, json=body)

    return httpx.Client(transport=httpx.MockTransport(answer)), seen


def decide(http):
    return http.post(
        "https://api.openai.com/v1/decisions",
        json={"model": "gpt-6-luna", "input": "a cat", "questions": [{"type": "predicate", "instructions": "safe?"}]},
        headers={"Authorization": "Bearer sk-test"},
    )


def test_a_plain_http_call_to_openai_is_recorded(sent):
    http, _ = openai_over_http(DECISION)
    with context(user_id="u1"):
        assert decide(http).json()["answers"][0]["probability"] == 0.97
    [r] = sent
    assert (r.provider, r.model, r.request_id, r.user_id, r.status) == ("openai", "gpt-6-luna", "req_1", "u1", "ok")
    assert r.units == {"input_tokens": 40, "output_tokens": 1, "cached_input_tokens": 8, "cache_write_input_tokens": 0,
                       "reasoning_tokens": 0}  # fmt: skip
    assert r.code[0] == "test_attach.decide"


def test_a_plain_http_chat_completion_reads_its_usage_its_own_way(sent):
    http, _ = openai_over_http(CHAT)
    http.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []})
    assert [(r.request_id, r.units["input_tokens"], r.units["output_tokens"]) for r in sent] == [("chatcmpl-1", 5, 2)]


def test_a_refused_plain_http_call_is_recorded_with_openai_s_code(sent):
    http, _ = openai_over_http({"error": {"code": "rate_limit_exceeded", "message": "slow down"}}, status=429)
    decide(http)
    assert [(r.status, r.error_type, r.request_id) for r in sent] == [("error", "rate_limit_exceeded", "req_1")]


def test_a_plain_http_call_over_requests_is_recorded(sent):
    class OpenAIAdapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            r = requests.Response()
            r.status_code, r.request, r.url = 200, request, request.url
            r.headers["x-request-id"] = "req_9"
            r._content = __import__("json").dumps(DECISION).encode()
            return r

        def close(self):
            pass

    http = requests.Session()
    http.mount("https://", OpenAIAdapter())
    decide(http)
    assert [(r.model, r.request_id) for r in sent] == [("gpt-6-luna", "req_9")]


def test_requests_to_openai_that_are_not_calls_pass_unseen(sent):
    http, _ = openai_over_http({}, {}, {})
    http.post("https://api.openai.com/v1/files", files={"file": ("a.jsonl", b"{}")})
    http.post("https://api.openai.com/v1/responses/resp_1/cancel")
    http.get("https://api.openai.com/v1/models")
    assert sent == []


def test_a_streamed_plain_http_call_is_recorded_without_reading_the_stream(sent):
    http, _ = openai_over_http({})
    with http.stream("POST", "https://api.openai.com/v1/responses", json={"model": "gpt-5.4-mini", "stream": True}):
        assert sent[0].units == {}  # recorded when the answer's headers came, nothing read
    assert [(r.model, r.request_id) for r in sent] == [("gpt-5.4-mini", "req_1")]


def test_a_call_litellm_makes_over_http_is_left_to_litellm(sent):
    http, _ = openai_over_http(DECISION)
    made_by_litellm = {"__name__": "litellm.llms.custom_httpx", "decide": decide, "http": http}
    exec("decide(http)", made_by_litellm)  # noqa: S102
    assert sent == []


# --- fal: fal_client's own session is watched over HTTP ---------------------------------------------------------

fal_client = pytest.importorskip("fal_client")
from test_fal_http import ARGS, RESULT, FakeFal  # noqa: E402

from inference_tracker import fal as tracked_fal  # noqa: E402


@pytest.fixture
def fal(monkeypatch):
    fake = FakeFal()
    real_sync, real_async = httpx.Client, httpx.AsyncClient
    transport = httpx.MockTransport(fake.httpx)
    # fal_client's session sends through a transport that has a session of its own inside (to retry on a
    # backup domain): only that inner session gets the fake, so a request goes through both, as it really does.
    def faked(real):
        return lambda **kw: real(**{**kw, "transport": kw.get("transport") or transport})

    monkeypatch.setattr(httpx, "Client", faked(real_sync))
    monkeypatch.setattr(httpx, "AsyncClient", faked(real_async))
    return fake


def test_a_fal_client_nobody_wrapped_is_recorded(fal, sent):
    client = fal_client.SyncClient(key="test-key")
    with context(user_id="u1"):
        handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
        assert handle.get(interval=0) == RESULT
    assert [(r.phase, r.model, r.request_id, r.user_id) for r in sent] == [
        ("start", "fal-ai/flux/dev", "req-1", "u1"),
        ("finish", "fal-ai/flux/dev", "req-1", "u1"),
    ]


def test_a_fal_client_wrapped_with_track_is_recorded_once(fal, sent):
    client = tracked_fal.track(fal_client.SyncClient(key="test-key"))
    assert client.subscribe("fal-ai/flux/dev", arguments=ARGS, interval=0) == RESULT
    assert [(r.phase, r.request_id) for r in sent] == [("start", "req-1"), ("finish", "req-1")]


def test_an_async_fal_client_nobody_wrapped_is_recorded(fal, sent):
    async def main():
        client = fal_client.AsyncClient(key="test-key")
        handle = await client.submit("fal-ai/flux/dev", arguments=ARGS)
        return await handle.get()

    assert asyncio.run(main()) == RESULT
    assert [r.phase for r in sent] == ["start", "finish"]


# --- google-genai ----------------------------------------------------------------------------------------------

genai = pytest.importorskip("google.genai")
from google.genai import types  # noqa: E402
from test_google_sdk import google as fake_google  # noqa: E402


def test_a_genai_client_nobody_wrapped_is_recorded(sent):
    transport = httpx.MockTransport(fake_google)
    client = genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=transport), httpx_async_client=httpx.AsyncClient(transport=transport)
        ),
    )
    client.models.embed_content(model="gemini-embedding-001", contents=["a", "b"])
    assert [(r.provider, r.model) for r in sent] == [("gcp.gemini", "gemini-embedding-001")]


def test_a_replaced_httpx_client_name_still_attaches_to_the_real_class(monkeypatch):
    each = FakeEachlabs()
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(each.httpx)}))
    monkeypatch.setattr(_attach, "_done", set(_attach._done) - {"httpx"})
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    _attach.attach()
    httpx.Client().post("https://api.eachlabs.ai/v1/prediction", json={"model": "m", "input": {}})
    assert [r.phase for r in out] == ["start"]


LONG_KEY = "sk-proj-" + "x" * 40 + "a1b2"


def test_a_call_records_the_hint_of_the_key_that_made_it_never_the_key(sent):
    http = httpx.Client(transport=httpx.MockTransport(Server(CHAT)))
    client = openai.OpenAI(api_key=LONG_KEY, http_client=http, max_retries=0)
    ask(client)
    [r] = sent
    assert r.settings["key_hint"] == "sk-proj-…a1b2"
    assert LONG_KEY not in json.dumps(r.settings)


def test_a_plain_http_call_records_the_hint_of_its_key(sent):
    http, _ = openai_over_http(CHAT)
    http.post("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []},
              headers={"Authorization": f"Bearer {LONG_KEY}"})  # fmt: skip
    assert sent[0].settings["key_hint"] == "sk-proj-…a1b2"


def test_a_short_key_gives_no_hint():
    from inference_tracker.openai import key_hint

    assert key_hint("sk-test") is None
    assert key_hint(None) is None
    assert key_hint("Bearer " + LONG_KEY) == "sk-proj-…a1b2"
