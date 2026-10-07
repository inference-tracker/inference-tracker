import json
import logging
import time

import pytest

import inferledger
from inferledger import Client, Record
from inferledger._transport import Result


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
    return [r for r in caplog.records if r.getMessage().startswith("inferledger.unsent ")]


def test_sends_in_batches_with_who_sent_them():
    server = FakeServer()
    c = client(server, batch_size=3)
    for i in range(7):
        c.record(call(i))
    assert server.bodies == []  # nothing goes out until flush (no background thread)
    assert c.flush() is True
    assert [len(b["records"]) for b in server.bodies] == [3, 3, 1]
    assert server.bodies[0]["app"] == "myapp" and server.bodies[0]["environment"] == "prod"
    assert server.bodies[0]["sdk"]["name"] == "inferledger-python"
    assert server.bodies[0]["lost"] == 0
    assert c.pending() == 0


def test_busy_server_is_retried_until_it_answers():
    server = FakeServer(Result("retry", 503), Result("retry"))
    c = client(server)
    c.record(call())
    assert c.flush() is True
    assert len(server.bodies) == 3
    assert server.bodies[0]["records"] == server.bodies[2]["records"]  # same batch, same ids


def test_retry_after_is_respected():
    server = FakeServer(Result("retry", 429, retry_after=0.2))
    c = client(server)
    c.record(call())
    started = time.monotonic()
    assert c.flush() is True
    assert time.monotonic() - started >= 0.2


def test_when_time_runs_out_records_are_kept_and_logged_once(caplog):
    caplog.set_level(logging.WARNING, logger="inferledger")
    server = FakeServer(*[Result("retry")] * 1000)
    c = client(server, first_retry_delay=0.05)
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
    caplog.set_level(logging.WARNING, logger="inferledger")
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
    caplog.set_level(logging.WARNING, logger="inferledger")
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
    yield
    inferledger._client._client = None


def test_without_init_everything_is_a_no_op(no_global_client, monkeypatch):
    for name in ("INFERLEDGER_URL", "INFERLEDGER_KEY", "INFERLEDGER_APP"):
        monkeypatch.delenv(name, raising=False)
    assert inferledger.init() is None
    inferledger.record(call())
    assert inferledger.flush() is True


def test_context_with_flush_sends_when_the_block_ends(no_global_client):
    server = FakeServer()
    inferledger.init("https://ingest.test", "key", "myapp", background=False, post=server)
    with inferledger.context(user_id="u1", task_id="t1", flush=True):
        inferledger.record(call())
        assert server.bodies == []
    assert server.records()[0]["user_id"] == "u1"


def _wait_for(condition, seconds=2.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)
