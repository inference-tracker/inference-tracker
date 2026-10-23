"""The Modal wrapper, around the real modal package with Modal's servers stood in: the private methods that would
talk to Modal are replaced, so the public objects (modal.Function, its remote/spawn and their .aio, FunctionCall)
are the real ones. Web endpoints over HTTP are answered by a fake transport. Skipped without modal or httpx."""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

modal = pytest.importorskip("modal")
httpx = pytest.importorskip("httpx")
requests = pytest.importorskip("requests")

from modal import _functions, _object  # noqa: E402
from modal._runtime import execution_context  # noqa: E402

import inference_tracker  # noqa: E402
from inference_tracker import Record, context  # noqa: E402
from inference_tracker import modal as tracked_modal  # noqa: E402


class FakeModal:
    """What Modal's servers would do. `error` makes the function raise; `pending` is how many get(timeout=0)
    find no result yet."""

    def __init__(self):
        self.error = None
        self.slow = False  # calls and results take a minute: for the app giving up on the wait
        self.pending = 0
        self.calls = []  # the arguments each call got
        self.cancelled = []


@pytest.fixture
def fake(monkeypatch):
    fake = FakeModal()

    async def hydrate(self, client=None):
        self._object_id = self._object_id or "fu-1"
        return self

    async def call_function(self, args, kwargs):
        fake.calls.append(args)
        if fake.slow:
            await asyncio.sleep(60)
        if fake.error:
            raise fake.error
        return {"video": "https://cdn/x.mp4"}

    async def cancel(request):
        fake.cancelled.append(request.function_call_id)

    async def call_function_nowait(self, args, kwargs, invocation_type, from_spawn_map=False):
        fake.calls.append(args)
        if fake.error:
            raise fake.error
        return SimpleNamespace(function_call_id=f"fc-{len(fake.calls)}", client=client)

    async def poll_function(timeout=None, index=0):
        if fake.slow:
            await asyncio.sleep(60)
        if fake.pending:
            fake.pending -= 1
            raise TimeoutError()  # what modal raises when get(timeout=...) finds no result yet
        return {"video": "https://cdn/x.mp4"}

    client = SimpleNamespace(_stub=SimpleNamespace(FunctionCallCancel=cancel))
    monkeypatch.setattr(_object._Object, "hydrate", hydrate)
    monkeypatch.setattr(_functions._Function, "_call_function", call_function)
    monkeypatch.setattr(_functions._Function, "_call_function_nowait", call_function_nowait)
    invocation = SimpleNamespace(poll_function=poll_function)
    monkeypatch.setattr(_functions._FunctionCall, "_invocation", lambda self: invocation)
    return fake


@pytest.fixture
def sent(monkeypatch):
    out: list[Record] = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: True)
    return out


def phases(sent):
    return [(r.phase, r.model, r.request_id, r.status) for r in sent]


ARGS = {"prompt": "a video of my kid", "seconds": 5}


# --- the modal package -------------------------------------------------------------------------------


def test_remote_is_one_call_with_timing_and_no_request_id(fake, sent):
    generate = tracked_modal.function("video-app", "generate")
    with context(user_id="u1", task_id="t1"):
        assert generate.remote(ARGS) == {"video": "https://cdn/x.mp4"}
    assert phases(sent) == [("call", "video-app/generate", None, "ok")]
    assert sent[0].provider == "modal" and sent[0].user_id == "u1" and sent[0].task_id == "t1"
    assert sent[0].started_at is not None and sent[0].duration_s is not None
    assert fake.calls == [(ARGS,)]  # the function got exactly the app's arguments
    assert "kid" not in json.dumps(sent[0].to_wire())


def test_remote_aio(fake, sent):
    generate = tracked_modal.track(modal.Function.from_name("video-app", "generate"), "video-app/generate")

    async def main():
        with context(user_id="u1"):
            return await generate.remote.aio(ARGS)

    assert asyncio.run(main()) == {"video": "https://cdn/x.mp4"}
    assert phases(sent) == [("call", "video-app/generate", None, "ok")] and sent[0].user_id == "u1"


def test_remote_that_raises_records_the_class_name_not_the_message(fake, sent):
    fake.error = ValueError("the message quotes the prompt: a video of my kid")
    generate = tracked_modal.function("video-app", "generate")
    with pytest.raises(ValueError):
        generate.remote(ARGS)
    assert phases(sent) == [("call", "video-app/generate", None, "error")]
    assert sent[0].error_type == "ValueError" and "kid" not in json.dumps(sent[0].to_wire())


def test_spawn_is_a_start_with_the_function_call_id_and_get_the_finish(fake, sent):
    generate = tracked_modal.function("video-app", "generate")
    with context(user_id="u1"):
        call = generate.spawn(ARGS)
    assert call.object_id == "fc-1"  # the FunctionCall's own attributes pass through
    fake.pending = 1
    with pytest.raises(TimeoutError):
        call.get(timeout=0)  # still running: not a finish
    assert phases(sent) == [("start", "video-app/generate", "fc-1", None)]
    assert call.get() == {"video": "https://cdn/x.mp4"}
    call.get()  # fetched again: finished once
    assert phases(sent) == [
        ("start", "video-app/generate", "fc-1", None),
        ("finish", "video-app/generate", "fc-1", "ok"),
    ]
    assert sent[1].user_id == "u1" and sent[1].duration_s is not None


def test_spawn_aio_get_aio_and_cancel(fake, sent):
    generate = tracked_modal.function("video-app", "generate")

    async def main():
        call = await generate.spawn.aio(ARGS)
        assert await call.get.aio() == {"video": "https://cdn/x.mp4"}
        other = await generate.spawn.aio(ARGS)
        await other.cancel.aio()

    asyncio.run(main())
    assert [(r.phase, r.request_id, r.status, r.error_type) for r in sent] == [
        ("start", "fc-1", None, None),
        ("finish", "fc-1", "ok", None),
        ("start", "fc-2", None, None),
        ("finish", "fc-2", "error", "cancelled"),
    ]
    assert fake.cancelled == ["fc-2"]


def test_a_spawn_that_fails_is_one_failed_call(fake, sent):
    fake.error = RuntimeError("no capacity")
    with pytest.raises(RuntimeError):
        tracked_modal.function("video-app", "generate").spawn(ARGS)
    assert phases(sent) == [("call", "video-app/generate", None, "error")]


def test_with_options_stays_tracked_and_notes_the_gpu(fake, sent):
    generate = tracked_modal.function("video-app", "generate").with_options(gpu="H100")
    generate.remote(ARGS)
    assert phases(sent) == [("call", "video-app/generate", None, "ok")] and sent[0].settings == {"gpu": "H100"}


def test_everything_else_is_the_functions_own(fake, sent):
    fn = modal.Function.from_name("video-app", "generate")
    generate = tracked_modal.track(fn, "video-app/generate")
    assert generate.map is not None and generate.local == fn.local
    assert "video-app" in repr(generate)
    assert sent == []


def test_a_broken_tracker_does_not_break_the_call(fake, sent, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("tracker bug")

    monkeypatch.setattr(tracked_modal._Call, "completed", broken)
    monkeypatch.setattr(tracked_modal._Call, "spawned", broken)
    generate = tracked_modal.function("video-app", "generate")
    with caplog.at_level(logging.ERROR, logger="inference_tracker"):
        assert generate.remote(ARGS) == {"video": "https://cdn/x.mp4"}
        assert generate.spawn(ARGS).get() == {"video": "https://cdn/x.mp4"}  # the plain FunctionCall comes back
    assert "tracker bug" in caplog.text


# --- inside a Modal function ----------------------------------------------------------------------------


def test_calls_inside_a_modal_function_have_its_call_as_parent(fake, sent):
    token = execution_context._current_function_call_id.set("fc-parent")  # what Modal sets while serving a call
    try:
        assert modal.current_function_call_id() == "fc-parent"
        with context(user_id="u1"), tracked_modal.inside() as ctx:
            assert (ctx.parent_provider, ctx.parent_request_id, ctx.user_id) == ("modal", "fc-parent", "u1")
            tracked_modal.function("video-app", "upscale").remote(ARGS)
    finally:
        execution_context._current_function_call_id.reset(token)
    assert (sent[0].parent_provider, sent[0].parent_request_id) == ("modal", "fc-parent")


def test_inside_outside_modal_does_nothing(sent):
    assert modal.current_function_call_id() is None
    with context(user_id="u1"), tracked_modal.inside() as ctx:
        assert ctx.parent_request_id is None and ctx.user_id == "u1"
    with tracked_modal.inside("fc-given") as ctx:
        assert ctx.parent_request_id == "fc-given"


def test_inside_lets_the_functions_errors_through():
    with pytest.raises(ValueError), tracked_modal.inside("fc-1"):
        raise ValueError("boom")


# --- web endpoints over HTTP -----------------------------------------------------------------------------


def web_endpoint(request: httpx.Request) -> httpx.Response:
    """Answers like a Modal web endpoint. /slow is redirected once, as Modal does past 150 seconds."""
    if request.url.host.endswith(".modal.run"):
        if request.url.path == "/slow":
            location = f"https://{request.url.host}/slow-result?__modal_function_call_id=fc-slow"
            return httpx.Response(303, headers={"location": location})
        if request.url.path == "/slow-result":
            return httpx.Response(200, json={"ok": True}, headers={"modal-function-call-id": "fc-slow"})
        if request.url.path == "/fail":
            return httpx.Response(500, text="Internal Server Error", headers={"modal-function-call-id": "fc-bad"})
        return httpx.Response(200, json={"ok": True}, headers={"modal-function-call-id": "fc-web"})
    return httpx.Response(200, json={"elsewhere": True})


def test_a_web_endpoint_call_is_one_record_with_the_call_id_header(sent):
    http = tracked_modal.track_http(httpx.Client(transport=httpx.MockTransport(web_endpoint)), apps=["video-app"])
    assert isinstance(http, httpx.Client)
    with context(user_id="u1"):
        assert http.post("https://acme--video-app-generate.modal.run/", json=ARGS).json() == {"ok": True}
        http.post("https://acme-staging--video-app-upscale-dev.modal.run/fail", json=ARGS)
        http.get("https://api.example.com/")  # not Modal: unseen
    assert phases(sent) == [
        ("call", "video-app/generate", "fc-web", "ok"),
        ("call", "video-app/upscale", "fc-bad", "error"),
    ]
    assert sent[1].error_type == "http_500" and all(r.user_id == "u1" for r in sent)
    assert "kid" not in json.dumps([r.to_wire() for r in sent])


def test_without_the_app_name_the_label_is_the_model(sent):
    http = tracked_modal.track_http(httpx.Client(transport=httpx.MockTransport(web_endpoint)))
    http.post("https://acme--video-app-generate.modal.run/", json=ARGS)
    http.post("https://acme--api.modal.run/", json=ARGS)  # a custom label
    assert [r.model for r in sent] == ["video-app-generate", "api"]


def test_a_redirected_long_request_is_one_record(sent):
    http = tracked_modal.track_http(
        httpx.Client(transport=httpx.MockTransport(web_endpoint), follow_redirects=True), apps=["video-app"]
    )
    assert http.post("https://acme--video-app-generate.modal.run/slow", json=ARGS).json() == {"ok": True}
    assert phases(sent) == [("call", "video-app/generate", "fc-slow", "ok")]


def test_async_httpx_and_requests(sent):
    class Adapter(requests.adapters.BaseAdapter):
        def send(self, request, **kwargs):
            response = requests.Response()
            response.status_code, response._content = 200, b'{"ok": true}'
            response.headers = requests.structures.CaseInsensitiveDict({"modal-function-call-id": "fc-req"})
            response.request, response.url = request, request.url
            return response

        def close(self):
            pass

    async def main():
        client = httpx.AsyncClient(transport=httpx.MockTransport(web_endpoint))
        async with tracked_modal.track_http(client, apps=["video-app"]) as http:
            await http.post("https://acme--video-app-generate.modal.run/", json=ARGS)

    asyncio.run(main())
    session = requests.Session()
    session.mount("https://", Adapter())
    tracked_modal.track_http(session, apps=["video-app"]).post("https://acme--video-app-generate.modal.run/", json=ARGS)
    assert [r.request_id for r in sent] == ["fc-web", "fc-req"]
    assert all(r.duration_s is not None for r in sent)


def test_a_remote_call_the_app_stops_waiting_for_is_cancelled(fake, sent):
    fake.slow = True
    generate = tracked_modal.function("video-app", "generate")

    async def main():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(generate.remote.aio(ARGS), 0.1)

    asyncio.run(main())
    assert [(r.phase, r.status, r.error_type) for r in sent] == [("call", "cancelled", "CancelledError")]


def test_a_get_the_app_stops_waiting_for_is_no_finish(fake, sent):
    generate = tracked_modal.function("video-app", "generate")

    async def main():
        call = await generate.spawn.aio(ARGS)
        fake.slow = True
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(call.get.aio(), 0.1)

    asyncio.run(main())
    assert phases(sent) == [("start", "video-app/generate", "fc-1", None)]  # still running on Modal


def test_a_web_request_without_an_answer_is_one_failed_call(sent):
    def dead(request):
        raise httpx.ReadTimeout("no answer")

    http = tracked_modal.track_http(httpx.Client(transport=httpx.MockTransport(dead)), apps=["video-app"])
    with context(user_id="u1"), pytest.raises(httpx.ReadTimeout):
        http.post("https://acme--video-app-generate.modal.run/", json=ARGS)
    assert phases(sent) == [("call", "video-app/generate", None, "error")]
    assert (sent[0].error_type, sent[0].user_id) == ("ReadTimeout", "u1")


def test_without_a_client_nothing_is_done(fake, monkeypatch):
    out = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    monkeypatch.setattr(inference_tracker._client, "active", lambda: False)
    generate = tracked_modal.function("video-app", "generate")
    generate.remote(ARGS)
    generate.spawn(ARGS).get()
    http = tracked_modal.track_http(httpx.Client(transport=httpx.MockTransport(web_endpoint)), apps=["video-app"])
    http.post("https://acme--video-app-generate.modal.run/", json=ARGS)
    assert out == []
