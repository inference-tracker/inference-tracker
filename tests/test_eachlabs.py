"""eachlabs' API over plain HTTP: track_http() on httpx and requests sessions, with eachlabs' servers stood in by a
fake that answers as its API reference shows (docs.eachlabs.ai, read 2026-11-18)."""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")
requests = pytest.importorskip("requests")

import inference_tracker  # noqa: E402
from inference_tracker import Record, context  # noqa: E402
from inference_tracker import eachlabs as tracked  # noqa: E402
from inference_tracker.eachlabs import track_http  # noqa: E402

MODEL = "p-video-upscale"
BASE = "https://api.eachlabs.ai/v1"
INPUT = {"video": "https://cdn.example.com/in.mp4", "prompt": "my kid at the beach", "target_resolution": "1080p", "duration": 5, "seed": 7}
OUTPUT = "https://storage.eachlabs.ai/out/abc.mp4"


class FakeEachlabs:
    """Answers like api.eachlabs.ai. `statuses` are the poll answers, in turn (the last one repeats); `create` is
    (status, body) to answer a create with instead."""

    def __init__(self):
        self.n = 0
        self.statuses = [
            {"status": "created"},
            {"status": "processing"},
            {"status": "success", "output": OUTPUT, "metrics": {"predict_time": 41.2, "cost": 0.1617}},
        ]
        self.create = None
        self.bodies = []

    def answer(self, method, url, content):
        path = httpx.URL(url).path
        if method == "POST" and path == "/v1/prediction":
            self.bodies.append(json.loads(content))
            if self.create:
                return self.create
            self.n += 1
            return 200, {"status": "success", "message": "Prediction created successfully", "predictionID": f"pred-{self.n}"}
        if method == "GET" and path.startswith("/v1/prediction/"):
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return 200, {"id": path.rsplit("/", 1)[1], "input": INPUT, **status}
        return 404, {"error": "Not found"}

    def httpx(self, request):
        status, body = self.answer(request.method, str(request.url), request.content)
        return httpx.Response(status, json=body)


class Adapter(requests.adapters.BaseAdapter):
    def __init__(self, fake):
        super().__init__()
        self.fake = fake

    def send(self, request, **kwargs):
        status, body = self.fake.answer(request.method, request.url, request.body)
        response = requests.Response()
        response.status_code = status
        response.headers = requests.structures.CaseInsensitiveDict({"content-type": "application/json"})
        response._content = json.dumps(body).encode()
        response.request, response.url = request, request.url
        return response

    def close(self):
        pass


@pytest.fixture
def each():
    return FakeEachlabs()


@pytest.fixture(autouse=True)
def no_open_predictions():
    tracked._open.clear()


@pytest.fixture
def sent(monkeypatch):
    out: list[Record] = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    return out


def phases(sent):
    return [(r.phase, r.model, r.request_id, r.status) for r in sent]


BODY = {"model": MODEL, "version": "0.0.1", "input": INPUT}


async def app_flow(http):
    """What an app does: create, then poll until it ends."""
    created = (await http.post(f"{BASE}/prediction", json=BODY, headers={"Authorization": "Bearer k"})).json()
    while True:
        status = (await http.get(f"{BASE}/prediction/{created['predictionID']}")).json()
        if status["status"] not in ("created", "starting", "processing"):
            return status


def test_create_and_poll_is_a_start_and_a_finish_with_the_cost(each, sent):
    async def main():
        async with track_http(httpx.AsyncClient(transport=httpx.MockTransport(each.httpx))) as http:
            with context(user_id="u1", task_id="t1", parent_provider="fal", parent_request_id="fal-1"):
                return await app_flow(http)

    assert asyncio.run(main())["output"] == OUTPUT  # the app reads eachlabs' answer, unchanged
    assert phases(sent) == [("start", MODEL, "pred-1", None), ("finish", MODEL, "pred-1", "ok")]
    start, finish = sent
    assert start.settings == {"target_resolution": "1080p", "duration": 5, "version": "0.0.1"}
    assert finish.units == {"input_videos": 1, "cost_usd": 0.1617, "predict_seconds": 41.2}
    assert (finish.user_id, finish.task_id, finish.parent_request_id) == ("u1", "t1", "fal-1")
    assert each.bodies == [BODY]  # what eachlabs received is the app's body
    wire = json.dumps([r.to_wire() for r in sent])
    assert "kid" not in wire and "https://" not in wire  # no prompt, no media URL


def test_a_failed_or_cancelled_prediction_finishes_with_a_code(each, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(each.httpx)))
    for ended in ({"status": "error", "logs": "moderation: blocked my kid", "metrics": {"cost": 0}}, {"status": "cancelled"}):
        each.statuses = [ended]
        created = http.post(f"{BASE}/prediction", json=BODY).json()
        http.get(f"{BASE}/prediction/{created['predictionID']}")
    assert [(r.phase, r.status, r.error_type) for r in sent] == [
        ("start", None, None),
        ("finish", "error", "error"),
        ("start", None, None),
        ("finish", "error", "cancelled"),
    ]
    assert sent[1].units["cost_usd"] == 0
    assert "kid" not in json.dumps([r.to_wire() for r in sent])


def test_a_refused_create_is_one_failed_call(each, sent):
    each.create = (400, {"error": "Invalid input for my kid"})
    http = requests.Session()
    http.mount("https://", Adapter(each))
    track_http(http)
    assert http.post(f"{BASE}/prediction", json=BODY).status_code == 400
    assert [(r.phase, r.request_id, r.status, r.error_type) for r in sent] == [("call", None, "error", "http_400")]


def test_a_poll_from_another_session_finishes_it_and_other_calls_pass_unseen(each, sent):
    a = track_http(httpx.Client(transport=httpx.MockTransport(each.httpx)))
    b = track_http(httpx.Client(transport=httpx.MockTransport(each.httpx)))
    each.statuses = [{"status": "success", "metrics": {"cost": 0.2, "billed_seconds": 6, "output_gb": 0.01}}]
    created = a.post(f"{BASE}/prediction", json=BODY).json()
    b.get(f"{BASE}/prediction/{created['predictionID']}")
    b.get(f"{BASE}/prediction/{created['predictionID']}")  # polled again after it ended: nothing more
    b.get(f"{BASE}/models/{MODEL}")
    assert phases(sent) == [("start", MODEL, "pred-1", None), ("finish", MODEL, "pred-1", "ok")]
    assert sent[1].units == {"input_videos": 1, "cost_usd": 0.2, "billed_seconds": 6.0, "output_gb": 0.01}
