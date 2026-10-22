"""The openai wrapper, against stand-ins shaped like the openai package's clients (no network, no openai)."""

import asyncio
from types import SimpleNamespace as NS

import pytest

import inference_tracker
from inference_tracker import context
from inference_tracker.openai import track


@pytest.fixture
def records(monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    return out


class APIStatusError(Exception):
    def __init__(self):
        super().__init__("the message quotes the prompt")
        self.code = "content_filter"
        self.request_id = "req_err"


RESPONSE = NS(
    id="resp_1",
    model="gpt-5.4-mini-2026-03-01",
    usage=NS(
        input_tokens=1200,
        output_tokens=80,
        input_tokens_details=NS(cached_tokens=1000),
        output_tokens_details=NS(reasoning_tokens=40),
    ),
)
CHAT = NS(
    id="chatcmpl-1",
    model="gpt-5.4-nano",
    usage=NS(prompt_tokens=50, completion_tokens=10, prompt_tokens_details=None, completion_tokens_details=None),
)
IMAGES = NS(
    data=[NS(b64_json="..."), NS(b64_json="...")],
    usage=NS(input_tokens=300, output_tokens=4000, input_tokens_details=NS(text_tokens=50, image_tokens=250)),
)
IMAGES._request_id = "req_img"


class Responses:
    def __init__(self, fail=False):
        self.fail = fail
        self.seen = []

    def parse(self, **kwargs):
        self.seen.append(kwargs)
        if self.fail:
            raise APIStatusError()
        return RESPONSE

    def create(self, **kwargs):
        return iter(["chunk"]) if kwargs.get("stream") else RESPONSE

    def retrieve(self, response_id):
        return RESPONSE


class AsyncCompletions:
    async def create(self, **kwargs):
        return CHAT


class Client:
    def __init__(self, base_url="https://api.openai.com/v1/", fail=False):
        self.base_url = base_url
        self.api_key = "sk-..."
        self.responses = Responses(fail)
        self.chat = NS(completions=NS(create=lambda **kw: CHAT))
        self.images = NS(edit=lambda **kw: IMAGES)


class Foundry:
    """An AsyncOpenAI pointed at Azure AI Foundry, calling a FLUX model with a raw post."""

    def __init__(self):
        self.base_url = "https://acme-resource.services.ai.azure.com/openai/v1/"
        self.posts = []

    async def post(self, path, *, body=None, cast_to=None):
        self.posts.append(path)
        return IMAGES if "/images/" in path else {"ok": True}


class AsyncClient:
    def __init__(self):
        self.base_url = "https://api.openai.com/v1/"
        self.chat = NS(completions=AsyncCompletions())


def test_responses_call_is_one_record_with_usage_and_who(records):
    client = track(Client())
    with context(user_id="u1", task_id="t1"):
        out = client.responses.parse(model="gpt-5.4-mini", input="a photo of my kid", reasoning={"effort": "low"})
    assert out is RESPONSE
    (r,) = records
    assert (r.provider, r.phase, r.status, r.request_id) == ("openai", "call", "ok", "resp_1")
    assert r.model == "gpt-5.4-mini-2026-03-01"
    assert r.settings == {"reasoning_effort": "low", "requested_model": "gpt-5.4-mini"}
    assert r.units == {"input_tokens": 1200, "output_tokens": 80, "cached_input_tokens": 1000, "reasoning_tokens": 40}
    assert (r.user_id, r.task_id) == ("u1", "t1")
    assert "kid" not in repr(r.to_wire())


def test_azure_base_url_is_azure_with_its_resource(records):
    client = track(Client(base_url="https://acme-resource.services.ai.azure.com/openai/v1/"))
    client.chat.completions.create(model="my-deployment", messages=[])
    (r,) = records
    assert r.provider == "azure.ai.openai"
    assert r.settings["azure_resource"] == "acme-resource"
    assert r.units == {"input_tokens": 50, "output_tokens": 10}


def test_images_count_and_tokens(records):
    track(Client()).images.edit(model="gpt-image-2", image=b"...", prompt="...", size="1024x1024", quality="high", n=2)
    (r,) = records
    assert r.request_id == "req_img"
    assert r.units == {
        "images": 2,
        "input_tokens": 300,
        "output_tokens": 4000,
        "input_text_tokens": 50,
        "input_image_tokens": 250,
    }
    assert r.settings == {"size": "1024x1024", "quality": "high", "n": 2}


def test_error_is_recorded_by_code_and_raised(records):
    client = track(Client(fail=True))
    with pytest.raises(APIStatusError):
        client.responses.parse(model="gpt-5.4-mini", input="...")
    (r,) = records
    assert (r.status, r.error_type, r.request_id) == ("error", "content_filter", "req_err")


def test_async_client(records):
    client = track(AsyncClient())
    out = asyncio.run(client.chat.completions.create(model="gpt-5.4-nano", messages=[]))
    assert out is CHAT
    assert records[0].request_id == "chatcmpl-1"


def test_other_methods_pass_through(records):
    client = track(Client())
    assert client.responses.retrieve("resp_1") is RESPONSE
    assert client.api_key == "sk-..."
    assert records == []


def test_a_stream_that_is_not_the_sdks_is_returned_as_is_and_recorded_without_usage(records):
    client = track(Client())
    assert list(client.responses.create(model="m", input="x", stream=True)) == ["chunk"]
    (r,) = records
    assert (r.model, r.status, r.units) == ("m", "ok", {})


def test_tracking_problems_never_reach_the_app(records, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("tracker bug")

    monkeypatch.setattr(inference_tracker.openai, "_units", boom)
    assert track(Client()).responses.parse(model="m", input="x") is RESPONSE


def test_raw_post_to_an_images_route_is_tracked_like_images(records):
    client = track(Foundry())
    body = {"model": "FLUX.2-pro", "prompt": "a photo of my kid", "size": "1024x1024", "input_image": "..."}
    assert asyncio.run(client.post("/images/generations", body=body, cast_to=object)) is IMAGES
    assert asyncio.run(client.post("/files", body={"x": 1}, cast_to=object)) == {"ok": True}
    (r,) = records
    assert (r.provider, r.model, r.status) == ("azure.ai.openai", "FLUX.2-pro", "ok")
    assert r.units["images"] == 2
    assert r.settings == {"size": "1024x1024", "azure_resource": "acme-resource"}
    assert "kid" not in repr(r.to_wire())


class FormatError(Exception):
    """Stands in for pydantic's ValidationError: the answer didn't fit the format the app asked for."""


class RawAnswer:
    """Shaped like the openai package's LegacyAPIResponse: the answer, not parsed yet."""

    def __init__(self, body, fits=True):
        self.body = body
        self.fits = fits
        self.http_response = NS(json=lambda: body)

    def parse(self):
        if not self.fits:
            raise FormatError("the answer quotes the prompt")
        return RESPONSE


class RawResponses(Responses):
    """Responses with the SDK's with_raw_response next to parse, as the real one has."""

    def __init__(self, fits=True):
        super().__init__()
        body = {"id": "resp_2", "model": "gpt-5.4-nano-2026-03-17", "usage": {"input_tokens": 900, "output_tokens": 30}}
        self.with_raw_response = NS(parse=lambda **kw: RawAnswer(body, fits))


def test_parse_goes_through_the_raw_answer_and_returns_the_same_result(records):
    client = Client()
    client.responses = RawResponses()
    assert track(client).responses.parse(model="gpt-5.4-nano", input="x") is RESPONSE
    assert records[0].status == "ok"


def test_parse_that_fails_the_format_check_keeps_the_billed_usage(records):
    client = Client(base_url="https://acme-resource.services.ai.azure.com/openai/v1/")
    client.responses = RawResponses(fits=False)
    with pytest.raises(FormatError):
        track(client).responses.parse(model="gpt-5.4-nano", input="a photo of my kid")
    (r,) = records
    assert (r.status, r.error_type, r.request_id) == ("error", "FormatError", "resp_2")
    assert r.model == "gpt-5.4-nano-2026-03-17"
    assert r.units == {"input_tokens": 900, "output_tokens": 30}
    assert "kid" not in repr(r.to_wire())


def test_with_the_real_openai_package(records):
    """The openai package itself, over a fake network: a parse call returns what it would untracked, and
    one whose answer doesn't fit the format raises the same error, with the usage on the record."""
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    from pydantic import BaseModel, ValidationError

    class Output(BaseModel):
        result: str
        is_safe: bool

    def answer(text):
        return {
            "id": "resp_3",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5.4-mini-2026-03-17",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 700,
                "output_tokens": 20,
                "total_tokens": 720,
                "input_tokens_details": {"cached_tokens": 512},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    texts = ['{"result": "hello", "is_safe": true}', '{"result": "hel']
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=answer(texts.pop(0))))

    def client():
        return openai.OpenAI(
            api_key="sk-test",
            base_url="https://acme.openai.azure.com/openai/v1/",
            http_client=httpx.Client(transport=transport),
            max_retries=0,
        )

    out = track(client()).responses.parse(model="gpt-5.4-mini", input="x", text_format=Output)
    assert out.output_parsed == Output(result="hello", is_safe=True)
    assert out.id == "resp_3"
    with pytest.raises(ValidationError):
        track(client()).responses.parse(model="gpt-5.4-mini", input="x", text_format=Output)
    ok, failed = records
    assert (ok.status, ok.units["input_tokens"]) == ("ok", 700)
    assert (failed.status, failed.error_type, failed.request_id) == ("error", "ValidationError", "resp_3")
    assert failed.units == {"input_tokens": 700, "output_tokens": 20, "cached_input_tokens": 512, "reasoning_tokens": 0}
