import json
import logging
import threading
import time

import pytest

import inference_tracker
from inference_tracker import Client, Record
from inference_tracker._transport import Result


class FakeServer:
    """Stands in for transport.post: answers from a script, then "sent" for everything after."""

    def __init__(self, *answers: Result):
        self.answers = list(answers)
        self.bodies: list[dict] = []

    def __call__(self, url, key, body, timeout):
        self.bodies.append(json.loads(json.dumps(body)))
        return self.answers.pop(0) if self.answers else Result("sent", 200)

    def records(self):
        return [r for b in self.bodies for r in b["records"]]


def client(server, **options):
    options.setdefault("background", False)
    options.setdefault("first_retry_delay", 0.01)
    return Client("https://ingest.test", "key", "myapp", "prod", post=server, **options)


def call(n=0):
    return Record.new("fal", "fal-ai/flux/dev", request_id=f"req-{n}", status="ok")


def unsent_lines(caplog):
    return [r for r in caplog.records if r.getMessage().startswith("inference_tracker.unsent ")]


def test_sends_in_batches_with_who_sent_them():
    server = FakeServer()
    c = client(server, batch_size=3)
    for i in range(7):
        c.record(call(i))
    assert server.bodies == []  # nothing goes out until flush (no background thread)
    assert c.flush() is True
    assert [len(b["records"]) for b in server.bodies] == [3, 3, 1]
    assert server.bodies[0]["app"] == "myapp" and server.bodies[0]["environment"] == "prod"
    assert server.bodies[0]["sdk"]["name"] == "inference-tracker-python"
    assert server.bodies[0]["lost"] == 0
    assert c.pending() == 0


def test_shutdown_retries_a_busy_server_until_it_answers():
    server = FakeServer(Result("retry", 503), Result("retry"))
    c = client(server)
    c.record(call())
    c.shutdown()
    assert len(server.bodies) == 3
    assert server.bodies[0]["records"] == server.bodies[2]["records"]  # same batch, same ids
    assert c.pending() == 0


def test_shutdown_respects_retry_after():
    server = FakeServer(Result("retry", 429, retry_after=0.2))
    c = client(server)
    c.record(call())
    started = time.monotonic()
    c.shutdown()
    assert time.monotonic() - started >= 0.2
    assert c.pending() == 0


def test_flush_never_waits_on_a_busy_server_and_keeps_the_records():
    server = FakeServer(Result("retry", 429, retry_after=30), Result("retry", 503))
    c = client(server, flush_pause=0)
    c.record(call(1))
    started = time.monotonic()
    assert c.flush() is False  # one try, no sleeping on Retry-After or backoff
    assert c.flush() is False
    assert time.monotonic() - started < 0.5
    assert len(server.bodies) == 2
    assert c.pending() == 1  # still queued, for the next flush
    assert c.flush() is True
    assert [r["gen_ai.response.id"] for r in server.records()][-1] == "req-1"


def test_after_a_failed_flush_flushes_are_skipped_for_a_while():
    server = FakeServer(Result("retry"))
    c = client(server, flush_pause=0.2)
    c.record(call(1))
    assert c.flush() is False
    c.record(call(2))
    assert c.flush() is False  # skipped: no try at all
    assert len(server.bodies) == 1
    assert c.pending() == 2
    time.sleep(0.25)
    assert c.flush() is True  # the pause is over: everything waiting goes, in one batch
    assert [r["gen_ai.response.id"] for r in server.bodies[-1]["records"]] == ["req-1", "req-2"]


def test_flush_is_short_by_default_and_configurable(no_global_client):
    timeouts = []

    def slow(url, key, body, timeout):
        timeouts.append(timeout)
        time.sleep(timeout)
        return Result("retry")  # no answer in time

    c = client(slow)
    assert c.flush_timeout == 1.0
    inference_tracker.init("https://ingest.test", "key", "myapp", background=False, post=slow, flush_timeout=0.2)
    inference_tracker.record(call())
    started = time.monotonic()
    assert inference_tracker.flush() is False
    assert time.monotonic() - started < 0.5
    assert 0 < timeouts[-1] <= 0.2
    assert inference_tracker._client._client.pending() == 1


def test_when_time_runs_out_records_are_kept_and_logged_once(caplog):
    caplog.set_level(logging.WARNING, logger="inference_tracker")
    server = FakeServer(*[Result("retry")] * 1000)
    c = client(server, first_retry_delay=0.05, flush_pause=0)
    c.record(call(1))
    c.record(call(2))
    assert c.flush(timeout=0.2) is False
    assert c.pending() == 2  # kept for the next try
    assert c.flush(timeout=0.2) is False
    lines = unsent_lines(caplog)
    assert len(lines) == 2  # logged once, not on every failed try
    assert json.loads(lines[0].getMessage().split(" ", 1)[1])["gen_ai.response.id"] == "req-1"

    server.answers.clear()  # the server is back
    assert c.flush() is True
    assert c.pending() == 0
    assert server.bodies[-1]["lost"] == 0  # kept records aren't lost


def test_refused_batch_is_logged_dropped_and_reported(caplog):
    caplog.set_level(logging.WARNING, logger="inference_tracker")
    server = FakeServer(Result("rejected", 401))
    c = client(server)
    c.record(call(1))
    c.record(call(2))
    assert c.flush() is True  # done with it: sending it again won't help
    assert c.pending() == 0
    assert len(unsent_lines(caplog)) == 2
    c.record(call(3))
    c.flush()
    assert server.bodies[-1]["lost"] == 2  # the next batch tells the server
    c.record(call(4))
    c.flush()
    assert server.bodies[-1]["lost"] == 0  # and the count resets once it's through


def test_full_queue_drops_the_oldest_and_logs_it(caplog):
    caplog.set_level(logging.WARNING, logger="inference_tracker")
    server = FakeServer()
    c = client(server, max_queue=3)
    for i in range(5):
        c.record(call(i))
    assert c.pending() == 3
    assert [json.loads(r.getMessage().split(" ", 1)[1])["gen_ai.response.id"] for r in unsent_lines(caplog)] == [
        "req-0",
        "req-1",
    ]
    c.flush()
    assert [r["gen_ai.response.id"] for r in server.records()] == ["req-2", "req-3", "req-4"]
    assert server.bodies[0]["lost"] == 2


def test_errors_never_reach_the_app():
    def broken(*args):
        raise RuntimeError("bug in the transport")

    c = client(broken)
    c.record(call())
    assert c.flush() is False  # reported as "not everything sent", not raised


def test_background_thread_sends_without_flush():
    server = FakeServer()
    c = client(server, background=True, flush_interval=0.05)
    c.record(call())
    _wait_for(lambda: len(server.records()) == 1)
    c.shutdown()


def test_full_batch_is_sent_right_away():
    server = FakeServer()
    c = client(server, background=True, flush_interval=60, batch_size=2)
    c.record(call(1))
    c.record(call(2))
    _wait_for(lambda: len(server.records()) == 2)
    c.shutdown()


def test_shutdown_sends_what_is_left():
    server = FakeServer()
    c = client(server, background=True, flush_interval=60)
    c.record(call())
    c.shutdown()
    assert len(server.records()) == 1


@pytest.fixture
def no_global_client():
    inference_tracker._client._client = None
    inference_tracker._client._settings = inference_tracker._client._NOT_YET
    yield
    inference_tracker._client._client = None
    inference_tracker._client._settings = inference_tracker._client._NOT_YET


def test_without_init_everything_is_a_no_op(no_global_client, monkeypatch):
    for name in ("INFERENCE_TRACKER_URL", "INFERENCE_TRACKER_KEY", "INFERENCE_TRACKER_APP"):
        monkeypatch.delenv(name, raising=False)
    assert inference_tracker.init() is None
    inference_tracker.record(call())
    assert inference_tracker.flush() is True


def test_context_with_flush_sends_when_the_block_ends(no_global_client):
    server = FakeServer()
    inference_tracker.init("https://ingest.test", "key", "myapp", background=False, post=server)
    with inference_tracker.context(user_id="u1", task_id="t1", flush=True):
        inference_tracker.record(call())
        assert server.bodies == []
    assert server.records()[0]["user_id"] == "u1"


def test_only_the_outermost_flushing_block_flushes(no_global_client, monkeypatch):
    flushes = []
    monkeypatch.setattr(inference_tracker._client, "flush", lambda: flushes.append(1))
    with inference_tracker.context(user_id="u1", flush=True):
        with inference_tracker.context(task_id="t1", flush=True):
            with inference_tracker.context(task_id="t2", flush=True):
                pass
            assert flushes == []
        assert flushes == []
    assert flushes == [1]
    with inference_tracker.context(task_id="t3", flush=True):
        pass
    assert flushes == [1, 1]  # a new outer block flushes again


def test_init_is_cheap_to_repeat_and_safe_from_many_threads(no_global_client):
    server = FakeServer()
    got = []

    def start():
        got.append(inference_tracker.init("https://ingest.test", "key", "myapp", background=False, post=server))

    threads = [threading.Thread(target=start) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 20 and got[0] is not None
    assert all(c is got[0] for c in got)  # one client, made once
    inference_tracker.record(call(1))
    assert inference_tracker.init("https://ingest.test", "key", "myapp", background=False, post=server) is got[0]
    assert got[0].pending() == 1  # calling init again didn't restart it, nor drop what it holds

    other = inference_tracker.init("https://ingest.test", "key", "otherapp", background=False, post=server)
    assert other is not got[0]  # other values: a new client, the old one sent what it had
    assert server.records()[0]["gen_ai.response.id"] == "req-1"


def test_init_without_values_warns_once(no_global_client, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="inference_tracker")
    for name in ("INFERENCE_TRACKER_URL", "INFERENCE_TRACKER_KEY", "INFERENCE_TRACKER_APP"):
        monkeypatch.delenv(name, raising=False)
    for _ in range(3):
        assert inference_tracker.init() is None
    assert len([r for r in caplog.records if "won't be sent" in r.getMessage()]) == 1
    assert inference_tracker._client.active() is False


def _wait_for(condition, seconds=2.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)
