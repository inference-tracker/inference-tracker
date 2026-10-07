import gzip
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from inferledger._transport import post


@pytest.fixture
def server():
    """A real local HTTP server that answers with whatever the test sets."""
    state = {"status": 200, "headers": {}, "received": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            state["received"].append((dict(self.headers), json.loads(gzip.decompress(body))))
            self.send_response(state["status"])
            for k, v in state["headers"].items():
                self.send_header(k, v)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_port}/api/ingest"
    yield state
    httpd.shutdown()


def test_sends_gzipped_json_with_the_key(server):
    result = post(server["url"], "secret", {"records": [{"id": "a"}]}, timeout=5)
    assert result.outcome == "sent"
    headers, body = server["received"][0]
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Content-Encoding"] == "gzip"
    assert body == {"records": [{"id": "a"}]}


@pytest.mark.parametrize(
    ("status", "outcome"),
    [(202, "sent"), (500, "retry"), (503, "retry"), (429, "retry"), (400, "rejected"), (401, "rejected")],
)
def test_answers_decide_what_happens_next(server, status, outcome):
    server["status"] = status
    assert post(server["url"], "k", {}, timeout=5).outcome == outcome


def test_429_retry_after(server):
    server["status"], server["headers"] = 429, {"Retry-After": "2"}
    assert post(server["url"], "k", {}, timeout=5).retry_after == 2.0


def test_no_server_means_retry():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # a port nothing listens on
    result = post(f"http://127.0.0.1:{port}/", "k", {}, timeout=2)
    assert result.outcome == "retry" and result.status is None
