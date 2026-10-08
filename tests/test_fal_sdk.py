"""The fal wrapper around the real fal_client package, with fal's servers stood in by an httpx mock.
Skipped when fal_client isn't installed (it isn't a dependency of inferledger)."""

import asyncio
import json

import pytest

fal_client = pytest.importorskip("fal_client")
httpx = pytest.importorskip("httpx")

import inferledger  # noqa: E402
from inferledger import Record, context  # noqa: E402
from inferledger.fal import track  # noqa: E402

RESULT = {"images": [{"url": "https://fal.media/x.png"}]}
REFUSED = {
    "detail": [{"type": "content_policy_violation", "msg": "refused: a photo of my kid"}],
    "error_type": "content_policy_violation",
}


class FakeFal:
    """Answers like fal's queue and run hosts. `refuse` makes the result a 422."""

    def __init__(self, refuse=False):
        self.refuse = refuse
        self.n = 0
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url, host = request.url.path, request.url.host
        if host == "fal.run":
            return httpx.Response(200, json=RESULT, headers={"x-fal-request-id": "run-1"})
        if host == "queue.fal.run" and request.method == "POST":
            self.n += 1
            base = f"https://queue.fal.run/fal-ai/flux/dev/requests/req-{self.n}"
            return httpx.Response(
                200,
                json={
                    "request_id": f"req-{self.n}",
                    "response_url": base,
                    "status_url": base + "/status",
                    "cancel_url": base + "/cancel",
                },
            )
        if url.endswith("/status"):
            return httpx.Response(200, json={"status": "COMPLETED", "logs": [], "metrics": {"inference_time": 1.5}})
        if "/requests/" in url:
            return httpx.Response(422, json=REFUSED) if self.refuse else httpx.Response(200, json=RESULT)
        raise AssertionError(f"unexpected request {request.method} {url}")


@pytest.fixture
def fal(monkeypatch):
    """fal_client builds its httpx sessions with httpx.Client / httpx.AsyncClient: give them the fake transport."""
    fake = FakeFal()
    real_sync, real_async = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_sync(**{**kw, "transport": httpx.MockTransport(fake)}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async(**{**kw, "transport": httpx.MockTransport(fake)}))
    return fake


@pytest.fixture
def sent(monkeypatch):
    out: list[Record] = []
    monkeypatch.setattr(inferledger._client, "record", out.append)
    return out


def phases(sent):
    return [(r.phase, r.request_id, r.status) for r in sent]


ARGS = {"prompt": "a photo of my kid", "image_size": "landscape_4_3", "num_images": 2}


def test_sync_client_submit_poll_get_and_run(fal, sent):
    client = track(fal_client.SyncClient(key="test-key"))
    with context(user_id="u1", task_id="t1"):
        handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
        events = list(handle.iter_events(with_logs=True, interval=0))
        assert isinstance(events[-1], fal_client.Completed)
        assert handle.get(interval=0) == RESULT
        assert client.run("fal-ai/flux/dev", arguments=ARGS) == RESULT

    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "ok"), ("call", "run-1", "ok")]
    start, finish, run = sent
    assert start.settings == {"image_size": "landscape_4_3", "num_images": 2}
    assert finish.units == {"inference_seconds": 1.5} and finish.user_id == "u1"
    assert run.user_id == "u1" and run.task_id == "t1"
    # What fal received is exactly the app's arguments
    assert json.loads(fal.requests[0].content) == ARGS
    assert "kid" not in json.dumps([r.to_wire() for r in sent])


def test_sync_subscribe_and_result_by_request_id(fal, sent):
    client = track(fal_client.SyncClient(key="test-key"))
    assert client.subscribe("fal-ai/flux/dev", arguments=ARGS, interval=0) == RESULT
    handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
    assert client.result("fal-ai/flux/dev", handle.request_id) == RESULT
    assert phases(sent) == [
        ("start", "req-1", None),
        ("finish", "req-1", "ok"),
        ("start", "req-2", None),
        ("finish", "req-2", "ok"),
    ]


def test_refused_result_is_a_finish_with_fals_error_type(fal, sent):
    fal.refuse = True
    client = track(fal_client.SyncClient(key="test-key"))
    handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
    with pytest.raises(fal_client.FalClientHTTPError):
        handle.get(interval=0)
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "error")]
    assert sent[1].error_type == "content_policy_violation"


def test_async_client(fal, sent):
    async def main():
        client = track(fal_client.AsyncClient(key="test-key"))
        with context(user_id="u1"):
            handle = await client.submit("fal-ai/flux/dev", arguments=ARGS)
            async for _event in handle.iter_events(interval=0):
                pass
            assert await handle.get(interval=0) == RESULT
            assert await client.subscribe("fal-ai/flux/dev", arguments=ARGS, interval=0) == RESULT
            assert await client.run("fal-ai/flux/dev", arguments=ARGS) == RESULT

    asyncio.run(main())
    assert phases(sent) == [
        ("start", "req-1", None),
        ("finish", "req-1", "ok"),
        ("start", "req-2", None),
        ("finish", "req-2", "ok"),
        ("call", "run-1", "ok"),
    ]
    assert sent[1].units == {"inference_seconds": 1.5}
    assert all(r.user_id == "u1" for r in sent)
