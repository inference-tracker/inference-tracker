"""Pruna's API over plain HTTP: track_http() on httpx and requests sessions, and on a stand-in for pruna_client's
PrunaClient, with Pruna's servers stood in by a fake that answers as its API reference shows
(https://docs.api.pruna.ai/openapi.yaml)."""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")
requests = pytest.importorskip("requests")

import inference_tracker  # noqa: E402
from inference_tracker import Record, context  # noqa: E402
from inference_tracker import pruna as tracked_pruna  # noqa: E402
from inference_tracker.pruna import track_http  # noqa: E402

MODEL = "p-video-2-pro-reference"
INPUT = {
    "prompt": "a video of my kid walking through a sunlit market",
    "image": ["https://api.pruna.ai/v1/files/fqadqq42xq", "https://api.pruna.ai/v1/files/abc123xyz789"],
    "video": ["https://api.pruna.ai/v1/files/vid1"],
    "audio": ["https://api.pruna.ai/v1/files/aud1"],
    "first_frame_image": "https://api.pruna.ai/v1/files/first1",
    "duration": 5,
    "resolution": "768p",
    "aspect_ratio": "adaptive",
    "mode": "speed",
    "prompt_upsampler": "turbo",
    "seed": 42,
}
DELIVERY = "https://api.pruna.ai/v1/predictions/delivery/xezq/52UGr4/output.mp4"


class FakePruna:
    """Answers like api.pruna.ai. `statuses` are the status answers, in turn (the last one repeats); `submit` is
    (status, body) to answer a submit with, instead of queueing it."""

    def __init__(self):
        self.n = 0
        self.requests = []
        self.statuses = [
            {"status": "starting", "message": "Generation in progress"},
            {"status": "processing", "message": "Generation in progress"},
            {"status": "succeeded", "generation_url": DELIVERY, "metadata": {"seed": 42}},
        ]
        self.submit = None

    def answer(self, method, url, headers, content):
        self.requests.append((method, url, headers, content))
        path = httpx.URL(url).path
        if url.startswith("https://api.pruna.ai/v1/predictions/delivery/"):
            return 200, b"\x00\x00\x00 ftypmp4", {"content-type": "video/mp4"}
        if method == "POST" and path == "/v1/predictions":
            if self.submit:
                return self.submit[0], self.submit[1], {}
            body = json.loads(content)
            if headers.get("Try-Sync") == "true":
                return 201, {"status": "succeeded", "generation_url": DELIVERY, "metadata": {}}, {}
            self.n += 1
            prediction = f"7xk3m29nw1rpv0dqq8tvbfrzc{self.n}"
            return (
                201,
                {
                    "id": prediction,
                    "model": headers.get("Model"),
                    "input": body["input"],  # Pruna echoes the whole input, prompt and URLs included
                    "get_url": f"https://api.pruna.ai/v1/predictions/status/{prediction}",
                },
                {},
            )
        if method == "GET" and path.startswith("/v1/predictions/status/"):
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return 200, status, {}
        if method == "POST" and path == "/v1/files":
            return 201, {"id": "f1", "urls": {"get": "https://api.pruna.ai/v1/files/f1"}}, {}
        return 404, {"error": {"code": "NOT_FOUND", "message": "Not found"}, "request_id": "req_404"}, {}

    def httpx(self, request):
        status, body, headers = self.answer(request.method, str(request.url), request.headers, request.content)
        if isinstance(body, bytes):
            return httpx.Response(status, content=body, headers=headers)
        return httpx.Response(status, json=body, headers=headers)


class Adapter(requests.adapters.BaseAdapter):
    def __init__(self, fake):
        super().__init__()
        self.fake = fake

    def send(self, request, **kwargs):
        status, body, headers = self.fake.answer(request.method, request.url, request.headers, request.body)
        response = requests.Response()
        response.status_code = status
        response.headers = requests.structures.CaseInsensitiveDict({"content-type": "application/json", **headers})
        response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
        response.request, response.url = request, request.url
        return response

    def close(self):
        pass


@pytest.fixture
def pruna():
    return FakePruna()


@pytest.fixture(autouse=True)
def no_open_predictions():
    tracked_pruna._open.clear()


@pytest.fixture
def sent(monkeypatch):
    out: list[Record] = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    return out


def phases(sent):
    return [(r.phase, r.model, r.request_id, r.status) for r in sent]


HEADERS = {"apikey": "pk-test", "Model": MODEL, "Content-Type": "application/json"}


async def app_flow(http):
    """What an app does: submit, poll the status URL until it ends, download the video."""
    submitted = (await http.post("https://api.pruna.ai/v1/predictions", headers=HEADERS, json={"input": INPUT})).json()
    while True:
        status = (await http.get(submitted["get_url"], headers={"apikey": "pk-test"})).json()
        if status["status"] not in ("starting", "processing"):
            break
    if status["status"] == "succeeded":
        return (await http.get(status["generation_url"], headers={"apikey": "pk-test"})).content
    return status


def test_submit_and_poll_is_a_start_and_a_finish(pruna, sent):
    async def main():
        async with track_http(httpx.AsyncClient(transport=httpx.MockTransport(pruna.httpx))) as http:
            with context(user_id="u1", task_id="t1", parent_provider="fal", parent_request_id="fal-req-1"):
                return await app_flow(http)

    assert asyncio.run(main()) == b"\x00\x00\x00 ftypmp4"  # what the app reads is Pruna's answer, unchanged
    assert phases(sent) == [
        ("start", MODEL, "7xk3m29nw1rpv0dqq8tvbfrzc1", None),
        ("finish", MODEL, "7xk3m29nw1rpv0dqq8tvbfrzc1", "ok"),
    ]
    start, finish = sent
    assert start.settings == {"duration": 5, "resolution": "768p", "aspect_ratio": "adaptive", "mode": "speed"}
    assert finish.units == {"input_images": 3, "input_videos": 1, "input_audios": 1}
    assert (finish.user_id, finish.task_id, finish.parent_provider, finish.parent_request_id) == (
        "u1",
        "t1",
        "fal",
        "fal-req-1",
    )
    assert start.started_at is not None and finish.duration_s is not None
    assert json.loads(pruna.requests[0][3]) == {"input": INPUT}  # what Pruna received is the app's input
    wire = json.dumps([r.to_wire() for r in sent])
    assert "kid" not in wire and "https://" not in wire  # no prompt, no media URL


def test_a_failed_or_canceled_prediction_finishes_with_a_code_not_the_message(pruna, sent):
    pruna.statuses = [
        {"status": "failed", "message": "Prediction failed", "error": "Number of samples, -5, must be non-negative."}
    ]
    http = track_http(httpx.Client(transport=httpx.MockTransport(pruna.httpx)))
    sync_flow(http)
    pruna.statuses = [{"status": "canceled", "message": "Prediction was canceled"}]
    sync_flow(http)
    assert [(r.phase, r.status, r.error_type) for r in sent] == [
        ("start", None, None),
        ("finish", "error", "failed"),
        ("start", None, None),
        ("finish", "error", "cancelled"),
    ]
    assert "samples" not in json.dumps([r.to_wire() for r in sent])


def sync_flow(http):
    submitted = http.post("https://api.pruna.ai/v1/predictions", headers=HEADERS, json={"input": INPUT}).json()
    while http.get(submitted["get_url"]).json()["status"] in ("starting", "processing"):
        pass


def test_try_sync_is_one_call_without_an_id(pruna, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(pruna.httpx)))
    body = {"input": {"prompt": "a photo of my kid", "duration": 5, "resolution": "480p", "mode": "cost"}}
    answer = http.post("https://api.pruna.ai/v1/predictions", headers={**HEADERS, "Try-Sync": "true"}, json=body)
    assert answer.json()["status"] == "succeeded"
    assert phases(sent) == [("call", MODEL, None, "ok")]
    assert sent[0].settings == {"duration": 5, "resolution": "480p", "mode": "cost"}
    assert sent[0].units == {}


def test_a_refused_submit_is_one_failed_call_with_prunas_code(pruna, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(pruna.httpx)))
    pruna.submit = (
        400,
        {
            "error": {"code": "VALIDATION_ERROR", "message": "prompt mentions my kid", "details": "..."},
            "request_id": "req_1234567890",
        },
    )
    http.post("https://api.pruna.ai/v1/predictions", headers=HEADERS, json={"input": INPUT})
    pruna.submit = (
        422,
        {
            "detail": "- input.prompt: Field required\n",
            "status": 422,
            "title": "Input validation failed",
            "invalid_fields": [{"type": "missing", "field": "input.prompt", "description": "Field required"}],
        },
    )
    http.post("https://api.pruna.ai/v1/predictions", headers=HEADERS, json={"input": {}})
    assert [(r.phase, r.request_id, r.status, r.error_type) for r in sent] == [
        ("call", "req_1234567890", "error", "VALIDATION_ERROR"),
        ("call", None, "error", "http_422"),
    ]
    assert "kid" not in json.dumps([r.to_wire() for r in sent])


def test_a_submit_without_an_answer_is_one_failed_call(pruna, sent):
    def dead(request):
        if request.url.path == "/v1/predictions":
            raise httpx.ConnectTimeout("no answer")
        return pruna.httpx(request)

    http = track_http(httpx.Client(transport=httpx.MockTransport(dead)))
    with pytest.raises(httpx.ConnectTimeout):
        http.post("https://api.pruna.ai/v1/predictions", headers=HEADERS, json={"input": INPUT})
    assert [(r.phase, r.model, r.request_id, r.status, r.error_type) for r in sent] == [
        ("call", MODEL, None, "error", "ConnectTimeout")
    ]


def test_a_poll_the_app_gives_up_on_leaves_the_prediction_open(pruna, sent):
    hang = []

    async def handler(request):
        if hang and "/status/" in request.url.path:
            await asyncio.sleep(10)
        return pruna.httpx(request)

    async def main():
        async with track_http(httpx.AsyncClient(transport=httpx.MockTransport(handler))) as http:
            submitted = (
                await http.post("https://api.pruna.ai/v1/predictions", headers=HEADERS, json={"input": INPUT})
            ).json()
            hang.append(True)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(http.get(submitted["get_url"]), 0.1)
            assert tracked_pruna._open  # still open: Pruna may still be making the video, and billing it
            hang.clear()
            while (await http.get(submitted["get_url"])).json()["status"] != "succeeded":
                pass

    asyncio.run(main())
    assert [r.phase for r in sent] == ["start", "finish"]


def test_requests_session_and_polling_from_another_session(pruna, sent):
    first = requests.Session()
    first.mount("https://", Adapter(pruna))
    first = track_http(first)
    assert isinstance(first, requests.Session)
    submitted = first.post("https://api.pruna.ai/v1/predictions", headers=HEADERS, json={"input": INPUT}).json()
    second = track_http(httpx.Client(transport=httpx.MockTransport(pruna.httpx)))
    while second.get(submitted["get_url"]).json()["status"] != "succeeded":
        pass
    second.get(DELIVERY)  # the download isn't a call
    second.post("https://api.pruna.ai/v1/files", files={"content": b"..."})  # nor an upload
    assert phases(sent) == [
        ("start", MODEL, "7xk3m29nw1rpv0dqq8tvbfrzc1", None),
        ("finish", MODEL, "7xk3m29nw1rpv0dqq8tvbfrzc1", "ok"),
    ]


def test_a_pruna_client_is_watched_through_its_sessions(pruna, sent):
    class PrunaClient:
        """Like pruna_client.PrunaClient: an httpx.Client and an httpx.AsyncClient, as `client` and `_async_client`."""

        def __init__(self):
            self.client = httpx.Client(transport=httpx.MockTransport(pruna.httpx))
            self._async_client = httpx.AsyncClient(transport=httpx.MockTransport(pruna.httpx))

        def generate(self, model, input, sync=False):
            headers = {"apikey": "pk", "Model": model, **({"Try-Sync": "true"} if sync else {})}
            return self.client.post("https://api.pruna.ai/v1/predictions", headers=headers, json={"input": input})

        async def agenerate(self, model, input):
            headers = {"apikey": "pk", "Model": model}
            return await self._async_client.post(
                "https://api.pruna.ai/v1/predictions", headers=headers, json={"input": input}
            )

    client = PrunaClient()
    assert track_http(client) is client
    client.generate("p-image", {"prompt": "a photo of my kid", "aspect_ratio": "1:1"}, sync=True)
    asyncio.run(client.agenerate("p-video", {"prompt": "x", "duration": 5, "resolution": "720p", "draft": True}))
    assert phases(sent) == [("call", "p-image", None, "ok"), ("start", "p-video", "7xk3m29nw1rpv0dqq8tvbfrzc1", None)]
    assert sent[1].settings == {"duration": 5, "resolution": "720p", "draft": True}


def test_other_hosts_and_unknown_predictions_pass_through(pruna, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(pruna.httpx)))
    http.get("https://api.pruna.ai/v1/predictions/status/not-submitted-here")
    http.get("https://example.com/v1/predictions")
    assert sent == []


def test_without_a_client_nothing_is_looked_at(pruna, monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: False)

    def never(*args):
        raise AssertionError("looked at an answer")

    monkeypatch.setattr(tracked_pruna._HttpWatcher, "wants_body", never)
    monkeypatch.setattr(tracked_pruna._HttpWatcher, "seen", never)

    async def main():
        async with track_http(httpx.AsyncClient(transport=httpx.MockTransport(pruna.httpx))) as http:
            return await app_flow(http)

    assert asyncio.run(main()) == b"\x00\x00\x00 ftypmp4"
    assert out == []


def test_a_broken_tracker_does_not_break_the_call(pruna, sent, monkeypatch, caplog):
    def broken(*args):
        raise RuntimeError("bug in the tracker")

    monkeypatch.setattr(tracked_pruna._HttpWatcher, "_start", broken)
    http = track_http(httpx.Client(transport=httpx.MockTransport(pruna.httpx)))
    answer = http.post("https://api.pruna.ai/v1/predictions", headers=HEADERS, json={"input": INPUT})
    assert answer.json()["id"] == "7xk3m29nw1rpv0dqq8tvbfrzc1"
    assert sent == []
