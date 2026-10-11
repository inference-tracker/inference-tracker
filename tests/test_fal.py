"""The fal wrapper, against stand-ins shaped like fal_client's clients (no network, no fal_client)."""

import asyncio
import logging
from types import SimpleNamespace

import pytest

import inferledger
from inferledger import Record, context
from inferledger.fal import inside, track, webhook

# --- stand-ins for fal_client ----------------------------------------------------------------


class Queued:
    position = 0


class InProgress:
    logs = None


class Completed:
    def __init__(self, metrics=None):
        self.metrics = metrics or {}


class FalError(Exception):
    def __init__(self, error_type=None):
        super().__init__("the message quotes the prompt: a photo of my kid")
        self.error_type = error_type


class Response:
    def __init__(self, headers):
        self.headers = headers


class Http:
    """Like httpx.Client: a dict of hooks, and every answer runs the response hooks."""

    def __init__(self):
        self.event_hooks = {"request": [], "response": []}

    def answer(self, headers):
        for hook in self.event_hooks["response"]:
            hook(Response(headers))


class Handle:
    def __init__(self, request_id, result=None, error=None, statuses=()):
        self.request_id, self.result, self.error = request_id, result, error
        self.response_url = f"https://queue.fal.run/x/requests/{request_id}"
        self.statuses = list(statuses) or [Completed({"inference_time": 4.2})]
        self.cancelled = False

    def status(self, *, with_logs=False):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def iter_events(self, *, with_logs=False, interval=0):
        while True:
            s = self.status()
            yield s
            if isinstance(s, Completed):
                return

    def get(self, *, interval=0):
        if self.error:
            raise self.error
        return self.result

    def cancel(self):
        self.cancelled = True


class FalClient:
    def __init__(self, result=None, error=None, submit_error=None):
        self.result_, self.error, self.submit_error = (
            result if result is not None else {"images": [{"url": "u"}]},
            error,
            submit_error,
        )
        self.calls = []  # (method, application, arguments, kwargs)
        self.handles = {}
        self._client = Http()
        self.n = 0

    def submit(self, application, arguments, *, webhook_url=None, **kwargs):
        self.calls.append(("submit", application, arguments, {"webhook_url": webhook_url, **kwargs}))
        if self.submit_error:
            raise self.submit_error
        self.n += 1
        handle = Handle(f"req-{self.n}", self.result_, self.error)
        self.handles[handle.request_id] = handle
        return handle

    def subscribe(self, application, arguments, *, on_enqueue=None, **kwargs):
        handle = self.submit(application, arguments, **kwargs)
        if on_enqueue:
            on_enqueue(handle.request_id)
        return handle.get()

    def run(self, application, arguments, **kwargs):
        self.calls.append(("run", application, arguments, kwargs))
        self._client.answer({"x-fal-request-id": "run-1"})
        if self.error:
            raise self.error
        return self.result_

    def result(self, application, request_id):
        return self.handles[request_id].get()

    def get_handle(self, application, request_id):
        return self.handles[request_id]

    def cancel(self, application, request_id):
        self.handles[request_id].cancel()

    def upload(self, data, content_type):
        return "https://fal.media/file"


class AsyncHandle(Handle):
    async def status(self, **kw):
        return Handle.status(self, **kw)

    async def iter_events(self, **kw):
        while True:
            s = Handle.status(self)
            yield s
            if isinstance(s, Completed):
                return

    async def get(self, **kw):
        return super().get(**kw)

    async def cancel(self):
        super().cancel()


class AsyncFalClient(FalClient):
    async def submit(self, application, arguments, **kwargs):
        handle = super().submit(application, arguments, **kwargs)
        async_handle = AsyncHandle(handle.request_id, handle.result, handle.error)
        self.handles[handle.request_id] = async_handle
        return async_handle

    async def subscribe(self, application, arguments, *, on_enqueue=None, **kwargs):
        handle = await self.submit(application, arguments, **kwargs)
        if on_enqueue:
            on_enqueue(handle.request_id)
        return await handle.get()

    async def run(self, application, arguments, **kwargs):
        self.calls.append(("run", application, arguments, kwargs))
        for hook in self._client.event_hooks["response"]:  # httpx awaits async hooks
            await hook(Response({"x-fal-request-id": "run-1"}))
        if self.error:
            raise self.error
        return self.result_

    async def result(self, application, request_id):
        return await self.handles[request_id].get()

    async def get_handle(self, application, request_id):
        return self.handles[request_id]

    async def cancel(self, application, request_id):
        await self.handles[request_id].cancel()


@pytest.fixture
def sent(monkeypatch):
    out: list[Record] = []
    monkeypatch.setattr(inferledger._client, "record", out.append)
    return out


def phases(sent):
    return [(r.phase, r.request_id, r.status) for r in sent]


ARGS = {"prompt": "a photo of my kid", "image_url": "https://x/y.png", "resolution": "720p", "duration": 5}

# --- submit + handle (how myapp-inference calls fal) ------------------------------------------


def test_submit_then_get_is_a_start_and_a_finish(sent):
    fal = FalClient()
    client = track(fal)
    with context(user_id="u1", task_id="t1"):
        handle = client.submit("fal-ai/kling-video/v2.6/pro/motion-control", arguments=ARGS)
        for _event in handle.iter_events(with_logs=True):
            pass
        assert handle.get() == {"images": [{"url": "u"}]}

    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "ok")]
    start, finish = sent
    assert start.settings == {"resolution": "720p", "duration": 5}  # named price fields only, no prompt or url
    assert start.user_id == "u1" and start.task_id == "t1" and start.started_at
    assert finish.user_id == "u1" and finish.task_id == "t1"
    assert finish.units == {"inference_seconds": 4.2}  # from the completed status
    assert finish.duration_s >= 0
    assert fal.calls[0][2] is ARGS  # the arguments go to fal untouched
    assert handle.response_url.endswith("req-1")  # everything else on the handle passes through


def test_a_failed_result_is_a_finish_with_the_providers_error_code(sent):
    client = track(FalClient(error=FalError("content_policy_violation")))
    handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
    with pytest.raises(FalError):
        handle.get()
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "error")]
    assert sent[1].error_type == "content_policy_violation"
    assert "kid" not in str(sent[1].to_wire())


def test_a_failed_submit_is_one_failed_call_without_request_id(sent):
    client = track(FalClient(submit_error=ConnectionError("fal.run: connection refused")))
    with pytest.raises(ConnectionError):
        client.submit("fal-ai/flux/dev", arguments=ARGS)
    assert phases(sent) == [("call", None, "error")]
    assert sent[0].error_type == "ConnectionError"


def test_result_by_request_id_finishes_a_call_this_process_submitted(sent):
    # myapp-api's poll path: submit, poll with iter_events, then client.result(application, request_id)
    fal = FalClient()
    client = track(fal)
    handle = client.submit("acme/prod-myapp-moderation", arguments=ARGS)
    for _event in handle.iter_events():
        pass
    client.result("acme/prod-myapp-moderation", handle.request_id)
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "ok")]
    client.result("acme/prod-myapp-moderation", handle.request_id)  # fetching again adds nothing
    assert len(sent) == 2


def test_a_call_is_finished_once(sent):
    fal = FalClient()
    client = track(fal)
    handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
    handle.get()
    handle.get()
    client.result("fal-ai/flux/dev", "req-1")
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "ok")]


def test_cancel_finishes_the_call(sent):
    client = track(FalClient())
    handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
    handle.cancel()
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "error")]
    assert sent[1].error_type == "cancelled"


def test_result_for_an_unknown_request_id_passes_through(sent):
    fal = FalClient()
    fal.handles["req-9"] = Handle("req-9", {"ok": True})
    client = track(fal)
    assert client.result("fal-ai/flux/dev", "req-9") == {"ok": True}
    assert sent == []


# --- submit with a webhook (how myapp-api calls fal) ---------------------------------------------


def test_submit_with_webhook_is_a_start_and_the_webhook_body_is_the_finish(sent):
    client = track(FalClient())
    with context(user_id="u1", task_id="t1"):
        handle = client.submit("acme/prod-myapp-wan2-2-img2video", arguments=ARGS, webhook_url="https://api/hook")
    assert phases(sent) == [("start", "req-1", None)]

    # Later, in the webhook handler, another process:
    body = {
        "request_id": handle.request_id,
        "gateway_request_id": "gw",
        "status": "OK",
        "payload": {"video": {"url": "u"}},
    }
    with context(user_id="u1", task_id="t1"):
        finish = webhook(body)
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "ok")]
    assert finish.model is None and "gen_ai.request.model" not in finish.to_wire()  # the start record has it
    assert finish.user_id == "u1"


def test_webhook_failure_takes_the_error_code_not_the_message(sent):
    body = {
        "request_id": "req-1",
        "status": "ERROR",
        "error": "Request failed: a photo of my kid was refused",
        "payload": {
            "detail": [
                {
                    "loc": ["body", "prompt"],
                    "msg": "prompt refused: a photo of my kid",
                    "type": "content_policy_violation",
                    "error_code": "CONTENT_POLICY",
                }
            ]
        },
    }
    finish = webhook(body, model="fal-ai/flux/dev")
    assert finish.status == "error" and finish.error_type == "CONTENT_POLICY" and finish.model == "fal-ai/flux/dev"
    assert "kid" not in str(finish.to_wire())
    assert webhook({"request_id": "req-2", "status": "ERROR", "payload": {}}).error_type == "error"


def test_webhook_without_a_request_id_records_nothing(sent):
    assert webhook({"status": "OK"}) is None and webhook("text") is None and webhook(None) is None
    assert sent == []


# --- subscribe and run -------------------------------------------------------------------------


def test_subscribe_is_a_start_and_a_finish_and_keeps_the_apps_callback(sent):
    seen = []
    client = track(FalClient())
    result = client.subscribe("fal-ai/flux/dev", arguments=ARGS, on_enqueue=seen.append)
    assert result == {"images": [{"url": "u"}]}
    assert seen == ["req-1"]
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "ok")]


def test_subscribe_that_fails_after_enqueue_is_finished_with_the_error(sent):
    client = track(FalClient(error=FalError()))
    with pytest.raises(FalError):
        client.subscribe("fal-ai/flux/dev", arguments=ARGS)
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "error")]
    assert sent[1].error_type == "FalError"  # no code from fal: the class name


def test_run_is_one_call_with_the_request_id_from_the_response_header(sent):
    client = track(FalClient())
    with context(user_id="u1"):
        assert client.run("fal-ai/flux/dev", arguments=ARGS) == {"images": [{"url": "u"}]}
    assert phases(sent) == [("call", "run-1", "ok")]
    assert sent[0].user_id == "u1" and sent[0].settings == {"resolution": "720p", "duration": 5}


def test_run_that_fails_is_one_failed_call(sent):
    client = track(FalClient(error=FalError("timeout")))
    with pytest.raises(FalError):
        client.run("fal-ai/flux/dev", arguments=ARGS)
    assert phases(sent) == [("call", "run-1", "error")]


# --- inside our own fal apps -------------------------------------------------------------------------


class App:
    """Like a fal.App while it serves a request: current_request is fal's RequestContext, or None."""

    def __init__(self, request_id):
        self.current_request = SimpleNamespace(request_id=request_id, endpoint="/", headers={})


def test_calls_inside_our_app_have_the_apps_request_as_parent(sent):
    app = App("app-req-1")
    client = track(FalClient())
    with inside(app):
        handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
    handle.get()  # the finish is fetched outside the block and still has the parent
    client.run("fal-ai/flux/dev", arguments=ARGS)  # outside: no parent

    start, finish, outside = sent
    assert (start.parent_provider, start.parent_request_id) == ("fal", "app-req-1")
    assert (finish.parent_provider, finish.parent_request_id) == ("fal", "app-req-1")
    assert outside.parent_provider is None and outside.parent_request_id is None


def test_inside_keeps_the_user_and_task_already_set(sent):
    with context(user_id="u1", task_id="t1"), inside(App("app-req-1")) as ctx:
        assert ctx == inferledger.Context("u1", "t1", "fal", "app-req-1")


def test_inside_without_a_current_request_does_nothing(sent, caplog):
    class Broken:
        @property
        def current_request(self):
            raise RuntimeError("not serving")

    no_request = SimpleNamespace(current_request=None)
    for app in (no_request, App(None), App(""), object(), Broken(), None):
        with caplog.at_level(logging.DEBUG, logger="inferledger"), inside(app) as ctx:
            assert ctx == inferledger.Context()
    assert "no current fal request" in caplog.text


def test_inside_lets_the_apps_errors_through():
    with pytest.raises(ValueError), inside(App("app-req-1")):
        raise ValueError("the app's own error")
    assert inferledger.current() == inferledger.Context()


# --- settings --------------------------------------------------------------------------------------


def test_settings_can_be_chosen(sent):
    client = track(FalClient(), settings=["num_inference_steps"])
    client.submit("fal-ai/flux/dev", arguments={**ARGS, "num_inference_steps": 28})
    assert sent[0].settings == {"num_inference_steps": 28}


# --- it never gets in the way ------------------------------------------------------------------


def test_everything_else_passes_through(sent):
    fal = FalClient()
    client = track(fal)
    assert client.upload(b"x", "image/png") == "https://fal.media/file"
    assert client.calls == [] and sent == []


def test_a_broken_tracker_does_not_break_the_call(sent, monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("bug in inferledger")

    monkeypatch.setattr(Record, "new", boom)
    client = track(FalClient())
    handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
    assert handle.get() == {"images": [{"url": "u"}]}
    assert client.run("fal-ai/flux/dev", arguments=ARGS)
    assert sent == []
    assert "bug in inferledger" in caplog.text


def test_arguments_that_are_not_a_dict_are_left_alone(sent):
    fal = FalClient()
    client = track(fal)
    client.submit("acme/app", arguments=["not", "a", "dict"])
    assert fal.calls[0][2] == ["not", "a", "dict"]
    assert sent[0].settings == {}


# --- the async client ---------------------------------------------------------------------------


def test_async_submit_iter_events_and_get(sent):
    async def main():
        fal = AsyncFalClient()
        client = track(fal)
        with context(user_id="u1", task_id="t1"):
            handle = await client.submit("acme/prod-myapp-image-synthesis", arguments=ARGS)
            async for _event in handle.iter_events(interval=0):
                pass
            result = await handle.get()
        assert result == {"images": [{"url": "u"}]}
        assert fal.calls[0][2] is ARGS
        assert handle.response_url.endswith("req-1")

    asyncio.run(main())
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "ok")]
    assert sent[1].units == {"inference_seconds": 4.2}


def test_async_subscribe_run_result_and_cancel(sent):
    async def main():
        fal = AsyncFalClient()
        client = track(fal)
        assert await client.subscribe("fal-ai/flux/dev", arguments=ARGS) == {"images": [{"url": "u"}]}
        assert await client.run("fal-ai/flux/dev", arguments=ARGS) == {"images": [{"url": "u"}]}
        h = await client.submit("fal-ai/flux/dev", arguments=ARGS)
        await client.result("fal-ai/flux/dev", h.request_id)
        h = await client.submit("fal-ai/flux/dev", arguments=ARGS)
        await h.cancel()
        h = await client.submit("fal-ai/flux/dev", arguments=ARGS)
        await client.cancel("fal-ai/flux/dev", h.request_id)

    asyncio.run(main())
    assert phases(sent) == [
        ("start", "req-1", None),
        ("finish", "req-1", "ok"),
        ("call", "run-1", "ok"),
        ("start", "req-2", None),
        ("finish", "req-2", "ok"),
        ("start", "req-3", None),
        ("finish", "req-3", "error"),
        ("start", "req-4", None),
        ("finish", "req-4", "error"),
    ]


def test_async_failed_get_is_a_finish_with_the_error(sent):
    async def main():
        client = track(AsyncFalClient(error=FalError("content_policy_violation")))
        handle = await client.submit("fal-ai/flux/dev", arguments=ARGS)
        with pytest.raises(FalError):
            await handle.get()

    asyncio.run(main())
    assert phases(sent) == [("start", "req-1", None), ("finish", "req-1", "error")]
    assert sent[1].error_type == "content_policy_violation"
