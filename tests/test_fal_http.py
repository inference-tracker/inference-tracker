"""fal over plain HTTP: track_http() on httpx and requests sessions, with fal's servers stood in by a fake that
answers like the queue and run hosts. Also fal_client's own clients (and subclasses) watched the same way."""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

httpx = pytest.importorskip("httpx")
requests = pytest.importorskip("requests")

import inference_tracker  # noqa: E402
from inference_tracker import Record, context  # noqa: E402
from inference_tracker import fal as tracked_fal  # noqa: E402
from inference_tracker.fal import WORKFLOW_ENVELOPE_HEADER, inside, track_http, webhook  # noqa: E402

RESULT = {"images": [{"url": "https://fal.media/x.png"}]}
WORKFLOW_RESULT = {
    "output": RESULT,
    "nodes": {
        "node-1": {"endpoint": "acme-compute/test-myapp-image-synthesis", "request_id": "step-1"},
        "node-2": {"endpoint": "fal-ai/kling-video/v2.5-turbo/standard/image-to-video", "request_id": "step-2"},
    },
}
REFUSED = {
    "detail": [{"type": "content_policy_violation", "msg": "refused: a photo of my kid"}],
    "error_type": "content_policy_violation",
}
ARGS = {"prompt": "a photo of my kid", "image_size": "landscape_4_3", "num_images": 2}


class FakeFal:
    """Answers like fal's queue and run hosts, and like any other host for the rest. `result` is what the
    result URL answers: (status, body, headers), or a list of them, one per fetch."""

    def __init__(self):
        self.n = 0
        self.requests = []
        self.result = [(200, RESULT, {"x-fal-billable-units": "2"})]
        self.run = (200, RESULT, {"x-fal-request-id": "run-1", "x-fal-billable-units": "1.5"})
        self.submit = None  # (status, body, headers) to answer a submit with, instead of queueing it
        self.envelope = False

    def answer(self, method, url, headers, content):
        self.requests.append((method, url, headers, content))
        parts = httpx.URL(url)
        host, path = parts.host, parts.path
        if host == "fal.run":
            if headers.get(WORKFLOW_ENVELOPE_HEADER) == "true" and self.run[0] == 200:
                return 200, WORKFLOW_RESULT, self.run[2]
            return self.run
        if host == "queue.fal.run" and method == "POST":
            if self.submit:
                return self.submit
            self.n += 1
            self.envelope = headers.get(WORKFLOW_ENVELOPE_HEADER) == "true"
            # fal's request URLs name only the owner and app, without the endpoint's path
            base = f"https://queue.fal.run/fal-ai/flux/requests/req-{self.n}"
            body = {
                "request_id": f"req-{self.n}",
                "response_url": base,
                "status_url": base + "/status",
                "cancel_url": base + "/cancel",
                "queue_position": 0,
            }
            return 200, body, {"x-fal-request-id": f"req-{self.n}"}
        if host == "queue.fal.run" and path.endswith("/status"):
            return 200, {"status": "COMPLETED", "logs": [], "metrics": {"inference_time": 1.5}}, {}
        if host == "queue.fal.run" and path.endswith("/cancel"):
            return 202, {"status": "CANCELLATION_REQUESTED"}, {}
        if host == "queue.fal.run" and "/requests/" in path:
            status, body, extra = self.result.pop(0) if len(self.result) > 1 else self.result[0]
            if status == 200 and self.envelope:
                body = WORKFLOW_RESULT
            return status, body, extra
        return 200, {"elsewhere": True}, {}

    def httpx(self, request):
        status, body, headers = self.answer(request.method, str(request.url), request.headers, request.content)
        return httpx.Response(status, json=body, headers=headers)


class Adapter(requests.adapters.BaseAdapter):
    """requests' side of the fake: builds the Response the way HTTPAdapter does."""

    def __init__(self, fake):
        super().__init__()
        self.fake = fake

    def send(self, request, **kwargs):
        status, body, headers = self.fake.answer(request.method, request.url, request.headers, request.body)
        response = requests.Response()
        response.status_code = status
        response.headers = requests.structures.CaseInsensitiveDict({"content-type": "application/json", **headers})
        response._content = json.dumps(body).encode()
        response.request, response.url = request, request.url
        return response

    def close(self):
        pass


@pytest.fixture
def fal():
    return FakeFal()


@pytest.fixture(autouse=True)
def no_open_requests():
    """Requests submitted over HTTP are remembered process-wide: each test starts with none."""
    tracked_fal._http_calls._open.clear()


@pytest.fixture
def sent(monkeypatch):
    out: list[Record] = []
    monkeypatch.setattr(inference_tracker._client, "record", out.append)
    return out


def phases(sent):
    return [(r.phase, r.model, r.request_id, r.status) for r in sent]


def session(fal):
    s = requests.Session()
    s.mount("https://", Adapter(fal))
    return s


def queue_flow(http):
    """What an app does over the queue: submit, poll the status until it's done, fetch the result."""
    submitted = http.post("https://queue.fal.run/fal-ai/flux/dev", json=ARGS).json()
    assert http.get(submitted["status_url"]).json()["status"] == "COMPLETED"
    return http.get(submitted["response_url"]).json()


# --- httpx and requests ----------------------------------------------------------------------------


def test_httpx_submit_status_result_is_a_start_and_a_finish(fal, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    assert isinstance(http, httpx.Client)  # the same client, of the same type
    with context(user_id="u1", task_id="t1"):
        assert queue_flow(http) == RESULT  # what the app reads is fal's answer, unchanged

    assert phases(sent) == [("start", "fal-ai/flux/dev", "req-1", None), ("finish", "fal-ai/flux/dev", "req-1", "ok")]
    start, finish = sent
    assert start.settings == {"image_size": "landscape_4_3", "num_images": 2}
    assert finish.units == {"inference_seconds": 1.5, "billable_units": 2.0}
    assert finish.user_id == "u1" and finish.task_id == "t1"
    assert start.started_at is not None and finish.duration_s is not None
    assert json.loads(fal.requests[0][3]) == ARGS  # what fal received is exactly the app's arguments
    assert "kid" not in json.dumps([r.to_wire() for r in sent])


def test_async_httpx(fal, sent):
    async def main():
        async with track_http(httpx.AsyncClient(transport=httpx.MockTransport(fal.httpx))) as http:
            with context(user_id="u1"):
                submitted = (await http.post("https://queue.fal.run/fal-ai/flux/dev", json=ARGS)).json()
                await http.get(submitted["status_url"])
                assert (await http.get(submitted["response_url"])).json() == RESULT
                assert (await http.post("https://fal.run/fal-ai/flux/dev", json=ARGS)).json() == RESULT

    asyncio.run(main())
    assert phases(sent) == [
        ("start", "fal-ai/flux/dev", "req-1", None),
        ("finish", "fal-ai/flux/dev", "req-1", "ok"),
        ("call", "fal-ai/flux/dev", "run-1", "ok"),
    ]
    assert sent[1].units == {"inference_seconds": 1.5, "billable_units": 2.0}
    assert all(r.user_id == "u1" for r in sent)


def test_requests_session(fal, sent):
    http = track_http(session(fal))
    assert isinstance(http, requests.Session)
    with context(user_id="u1"):
        assert queue_flow(http) == RESULT
        assert http.post("https://fal.run/fal-ai/flux/dev", json=ARGS).json() == RESULT
    assert phases(sent) == [
        ("start", "fal-ai/flux/dev", "req-1", None),
        ("finish", "fal-ai/flux/dev", "req-1", "ok"),
        ("call", "fal-ai/flux/dev", "run-1", "ok"),
    ]
    assert sent[0].settings == {"image_size": "landscape_4_3", "num_images": 2}
    assert sent[1].units == {"inference_seconds": 1.5, "billable_units": 2.0}
    assert sent[2].units == {"billable_units": 1.5}


def test_the_result_can_be_fetched_with_another_session(fal, sent):
    first = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    submitted = first.post("https://queue.fal.run/fal-ai/flux/dev", json=ARGS).json()
    second = track_http(session(fal))
    second.get(submitted["response_url"])
    assert phases(sent)[-1] == ("finish", "fal-ai/flux/dev", "req-1", "ok")


def test_run_is_one_call_and_a_refusal_has_fals_error_type(fal, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    http.post("https://fal.run/fal-ai/flux/dev", json=ARGS)
    fal.run = (422, REFUSED, {"x-fal-request-id": "run-2"})
    assert http.post("https://fal.run/fal-ai/flux/dev", json=ARGS).status_code == 422
    assert phases(sent) == [("call", "fal-ai/flux/dev", "run-1", "ok"), ("call", "fal-ai/flux/dev", "run-2", "error")]
    assert sent[1].error_type == "content_policy_violation"


def test_a_refused_result_finishes_with_the_error_type_header(fal, sent):
    fal.result = [(422, {"detail": "the message quotes the prompt"}, {"x-fal-error-type": "content_policy_violation"})]
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    queue_flow(http)
    assert phases(sent)[-1] == ("finish", "fal-ai/flux/dev", "req-1", "error")
    assert sent[-1].error_type == "content_policy_violation"
    assert "prompt" not in json.dumps(sent[-1].to_wire())


def test_an_answer_that_doesnt_say_it_failed_leaves_the_request_open(fal, sent):
    fal.result = [(400, {"detail": "Request is still in progress"}, {}), (200, RESULT, {})]
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    submitted = http.post("https://queue.fal.run/fal-ai/flux/dev", json=ARGS).json()
    http.get(submitted["response_url"])
    assert phases(sent) == [("start", "fal-ai/flux/dev", "req-1", None)]
    http.get(submitted["response_url"])
    assert phases(sent)[-1] == ("finish", "fal-ai/flux/dev", "req-1", "ok")
    http.get(submitted["response_url"])  # fetched again: finished once
    assert len(sent) == 2


def test_a_server_error_on_the_result_finishes_it(fal, sent):
    fal.result = [(500, {"detail": "boom"}, {"x-fal-request-id": "req-1"})]
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    queue_flow(http)
    assert phases(sent)[-1] == ("finish", "fal-ai/flux/dev", "req-1", "error")
    assert sent[-1].error_type == "http_500"


def test_a_refused_submit_is_one_failed_call_and_a_gateway_error_is_not_recorded(fal, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    fal.submit = (503, {"detail": "nginx"}, {})  # fal's gateway, the app never saw it (fal_client retries)
    http.post("https://queue.fal.run/fal-ai/flux/dev", json=ARGS)
    assert sent == []
    fal.submit = (422, {"detail": [{"type": "value_error", "msg": "bad prompt: my kid"}]}, {})
    http.post("https://queue.fal.run/fal-ai/flux/dev", json=ARGS)
    assert phases(sent) == [("call", "fal-ai/flux/dev", None, "error")]
    assert sent[0].error_type == "value_error"


def test_cancel_finishes_the_request(fal, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    submitted = http.post("https://queue.fal.run/fal-ai/flux/dev", json=ARGS).json()
    http.put(submitted["cancel_url"])
    assert phases(sent)[-1] == ("finish", "fal-ai/flux/dev", "req-1", "error")
    assert sent[-1].error_type == "cancelled"


def test_a_webhook_submit_is_finished_by_the_webhook(fal, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    with context(user_id="u1"):
        submitted = http.post(
            "https://queue.fal.run/fal-ai/flux/dev", params={"fal_webhook": "https://app/hook"}, json=ARGS
        ).json()
    http.get(submitted["response_url"])  # not kept open: the webhook has the finish
    webhook({"request_id": "req-1", "status": "OK", "payload": RESULT})
    assert [(r.phase, r.request_id, r.status) for r in sent] == [("start", "req-1", None), ("finish", "req-1", "ok")]


def test_workflow_steps_from_the_result_and_from_run(fal, sent):
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    envelope = {WORKFLOW_ENVELOPE_HEADER: "true"}
    submitted = http.post("https://queue.fal.run/workflows/acme/video", json=ARGS, headers=envelope).json()
    assert http.get(submitted["response_url"]).json() == WORKFLOW_RESULT
    http.post("https://fal.run/workflows/acme/video", json=ARGS, headers=envelope)
    steps = [(r.phase, r.request_id, r.parent_request_id) for r in sent if r.parent_request_id]
    assert steps == [("call", "step-1", "req-1"), ("call", "step-2", "req-1")] + [
        ("call", "step-1", "run-1"),
        ("call", "step-2", "run-1"),
    ]
    assert fal.requests[0][2][WORKFLOW_ENVELOPE_HEADER] == "true"  # the app's header reached fal


def test_calls_inside_our_fal_app_have_its_request_as_parent(fal, sent):
    app = SimpleNamespace(current_request=SimpleNamespace(request_id="app-req-1"))
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    with inside(app):
        queue_flow(http)
    assert {(r.parent_provider, r.parent_request_id) for r in sent} == {("fal", "app-req-1")}


def test_other_urls_pass_through_unseen(fal, sent, monkeypatch):
    routed = []
    route = tracked_fal._route
    monkeypatch.setattr(tracked_fal, "_route", lambda ex: routed.append(ex.host) or route(ex))
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    with http.stream("GET", "https://api.example.com/v1/things") as response:
        assert json.loads(response.read()) == {"elsewhere": True}
    http.post("https://rest.fal.ai/storage/upload", content=b"x")
    assert routed == []  # other hosts: not even looked at
    http.get("https://queue.fal.run/fal-ai/flux/requests/req-9/status/stream")
    assert sent == []  # a status stream is left alone (its body is never read)


def test_watching_twice_records_once(fal, sent):
    http = track_http(track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx))))
    track_http(http)
    http.post("https://fal.run/fal-ai/flux/dev", json=ARGS)
    assert len(sent) == 1


def test_a_broken_tracker_does_not_break_the_call(fal, sent, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("tracker bug")

    monkeypatch.setattr(tracked_fal, "_route", broken)
    http = track_http(httpx.Client(transport=httpx.MockTransport(fal.httpx)))
    with caplog.at_level(logging.ERROR, logger="inference_tracker"):
        assert queue_flow(http) == RESULT
    assert sent == [] and "tracker bug" in caplog.text


def test_something_that_isnt_a_client_is_returned_as_it_is(caplog):
    thing = object()
    with caplog.at_level(logging.WARNING, logger="inference_tracker"):
        assert track_http(thing) is thing
    assert "can't watch" in caplog.text


# --- fal_client's own clients, and subclasses -----------------------------------------------------------

try:
    import fal_client
except ImportError:  # not a dependency of inference_tracker
    fal_client = None

needs_fal_client = pytest.mark.skipif(fal_client is None, reason="fal_client isn't installed")


@pytest.fixture
def patched(monkeypatch, fal):
    """fal_client builds its httpx sessions with httpx.Client / httpx.AsyncClient: give them the fake transport."""
    real_sync, real_async = httpx.Client, httpx.AsyncClient
    transport = httpx.MockTransport(fal.httpx)
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_sync(**{**kw, "transport": transport}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async(**{**kw, "transport": transport}))
    return fal


class AppClient(fal_client.SyncClient if fal_client else object):
    """A subclass that calls fal over HTTP from its own methods, the way some apps do."""

    def generate(self, arguments):
        submitted = self._client.post("https://queue.fal.run/fal-ai/flux/dev", json=arguments).json()
        return self._client.get(submitted["response_url"]).json()


@needs_fal_client
def test_a_fal_client_subclass_is_watched_with_its_own_methods(patched, sent):
    client = track_http(AppClient(key="test-key"))
    assert isinstance(client, AppClient)
    with context(user_id="u1"):
        assert client.generate(ARGS) == RESULT
        assert client.run("fal-ai/flux/dev", arguments=ARGS) == RESULT
        handle = client.submit("fal-ai/flux/dev", arguments=ARGS)
        assert handle.get(interval=0) == RESULT
    assert phases(sent) == [
        ("start", "fal-ai/flux/dev", "req-1", None),
        ("finish", "fal-ai/flux/dev", "req-1", "ok"),
        ("call", "fal-ai/flux/dev", "run-1", "ok"),
        ("start", "fal-ai/flux/dev", "req-2", None),
        ("finish", "fal-ai/flux/dev", "req-2", "ok"),
    ]
    assert sent[4].units == {"inference_seconds": 1.5, "billable_units": 2.0}  # from the status get() polled
    assert all(r.user_id == "u1" for r in sent)


@needs_fal_client
def test_a_subclass_can_also_be_wrapped_by_track(patched, sent):
    client = tracked_fal.track(AppClient(key="test-key"))
    assert client.subscribe("fal-ai/flux/dev", arguments=ARGS, interval=0) == RESULT
    assert client.generate(ARGS) == RESULT  # its own methods aren't seen by track(): use track_http for those
    assert [(r.phase, r.request_id) for r in sent] == [("start", "req-1"), ("finish", "req-1")]


@needs_fal_client
def test_an_async_fal_client_is_watched_through_its_session(patched, sent, caplog):
    async def main():
        client = fal_client.AsyncClient(key="test-key")
        with caplog.at_level(logging.WARNING, logger="inference_tracker"):
            assert track_http(client) is client  # its session doesn't exist yet: says what to pass instead
        assert "await client._client" in caplog.text
        track_http(await client._client)
        handle = await client.submit("fal-ai/flux/dev", arguments=ARGS)
        assert await handle.get(interval=0) == RESULT

    asyncio.run(main())
    assert [(r.phase, r.request_id, r.status) for r in sent] == [("start", "req-1", None), ("finish", "req-1", "ok")]
