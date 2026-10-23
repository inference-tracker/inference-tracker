"""Tracks calls to Pruna's API (https://api.pruna.ai), made over plain HTTP or with Pruna's Python client
(the pruna_client package).

    import inference_tracker

    http = inference_tracker.pruna.track_http(httpx.AsyncClient())   # or httpx.Client, requests.Session
    pruna = inference_tracker.pruna.track_http(PrunaClient())        # pruna_client: both of its sessions

How Pruna's API works: a prediction is sent with POST /v1/predictions, the model in the Model header and the
input in {"input": {...}}. Pruna answers with the prediction's id and a status URL (get_url,
/v1/predictions/status/<id>), which the app polls until it says succeeded (with a generation_url to download
the result from), failed or canceled. With the Try-Sync header, Pruna may instead wait (up to a minute) and
answer with the finished result, which carries no id. Pruna has no webhooks.

What is recorded:
- a submit answered with an id: a "start" record with the id as request id, and the status answer that says
  the prediction ended: the "finish", ok, or "failed" / "cancelled" (status "error"). The status may be polled
  with another watched session than the one that submitted.
- a submit answered with the finished result (Try-Sync): one "call" record, without a request id: Pruna gives
  none.
- a submit Pruna refused (an error answer): one "call" record with Pruna's code (VALIDATION_ERROR,
  RATE_LIMIT_EXCEEDED...) or "http_<status>", and the request id the error carries (req_...).
- a submit that got no answer (a timeout, a broken connection): one "call" record with the exception's class
  name, no request id ("cancelled" when the app cancelled it). A poll without an answer, or with an error
  answer, changes nothing: the prediction may still be running.
Downloads (/v1/predictions/delivery/...), uploads (/v1/files) and other hosts pass through unseen. The call
itself is never changed, and tracking never raises: a problem in it is logged and the call goes on.

Usage: Pruna's answers report none (no seconds of video, no cost, no credits), and its API has no usage or
billing endpoint. What the price depends on is in the request: settings copied from the input by name (duration,
resolution, mode, draft...), and the number of reference images, videos and audio files sent (input_images,
input_videos, input_audios: counts, never the URLs). Never the prompt, nor any media URL. Pruna bills some models
by the finished video's length, which only its bill (its dashboard) shows.

The model is Pruna's name for it, from the Model header ("p-video-2-pro-reference"). The parent is taken from the
context, as for every record.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

from . import _client, _http
from ._proxy import get
from ._record import Record, cancelled, code

logger = logging.getLogger("inference_tracker")

PROVIDER = "pruna"
HOSTS = frozenset({"api.pruna.ai"})
MODEL_HEADER = "Model"
# Input fields copied onto the record because they change the price (or say how long the video is).
SETTINGS = (
    "duration",
    "resolution",
    "mode",
    "draft",
    "fps",
    "aspect_ratio",
    "num_frames",
    "num_outputs",
    "width",
    "height",
    "size",
    "image_size",
    "go_fast",
    "turbo",
)
# Input fields holding media URLs, counted (never copied): reference images, videos and audio change the price.
MEDIA = {
    "input_images": ("image", "images", "first_frame_image", "last_frame_image"),
    "input_videos": ("video",),
    "input_audios": ("audio",),
}
# Predictions submitted and not seen ended yet, by id, shared by every watched session. Beyond this many, the
# oldest is forgotten.
MAX_OPEN = 1000
_ENDED = {"succeeded": None, "failed": "failed", "canceled": "cancelled"}

_SUBMIT = re.compile(r"^/v\d+/predictions/?$")
_STATUS = re.compile(r"^/v\d+/predictions/status/(?P<id>[^/]+)/?$")

_open: OrderedDict[str, _Call] = OrderedDict()
_open_lock = threading.Lock()


def track_http(client: Any, *, settings: Iterable[str] = SETTINGS) -> Any:
    """Watches an httpx.Client, httpx.AsyncClient or requests.Session for calls to Pruna's API, and returns the
    same client. A pruna_client.PrunaClient can be passed as it is: both of its httpx sessions (sync and async)
    are watched. See the module docstring."""
    watcher = _HttpWatcher(settings)
    try:
        if _http.watch(client, watcher):
            return client
        sessions = [getattr(client, name, None) for name in ("client", "_async_client")]  # pruna_client's
        watched = [s for s in sessions if s is not None and _http.watch(s, watcher)]
        if not watched:
            logger.warning("inference_tracker: can't watch a %s for Pruna calls", type(client).__name__)
    except Exception:
        logger.exception("inference_tracker: could not watch a client for Pruna calls")
    return client


# --- one call ----------------------------------------------------------------------------------


class _Call:
    """The record of one prediction while this process follows it."""

    def __init__(self, model: str | None, settings: dict[str, Any], units: dict[str, float], started: tuple):
        started_at, self._clock = started
        self.record = Record.new(PROVIDER, model, phase="start", started_at=started_at, settings=settings)
        self.units = units
        self.done = False

    def submitted(self, request_id: str) -> None:
        self.record.request_id = request_id
        _client.record(self.record)
        with _open_lock:
            _open[request_id] = self
            while len(_open) > MAX_OPEN:
                _open.popitem(last=False)

    def finish(self, error_type: str | None) -> None:
        if self.done:
            return
        self.done = True
        start = self.record
        with _open_lock:
            _open.pop(start.request_id, None)
        _client.record(
            Record.new(
                PROVIDER,
                start.model,
                phase="finish",
                request_id=start.request_id,
                status="error" if error_type else "ok",
                error_type=error_type,
                duration_s=self._elapsed(),
                units=dict(self.units),
                user_id=start.user_id,
                task_id=start.task_id,
                parent_provider=start.parent_provider,
                parent_request_id=start.parent_request_id,
            )
        )

    def completed(self, request_id: str | None, error_type: str | None, stopped: bool = False) -> None:
        """The whole call seen here (a Try-Sync answer, a refused or unanswered submit): one "call" record."""
        if self.done:
            return
        self.done = True
        r = self.record
        r.phase, r.request_id = "call", request_id
        r.status, r.error_type = ("cancelled" if stopped else "error" if error_type else "ok"), error_type
        r.duration_s, r.units = self._elapsed(), dict(self.units)
        _client.record(r)

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._clock, 3)


# --- over HTTP ---------------------------------------------------------------------------------------


class _HttpWatcher:
    name = "pruna"

    def __init__(self, settings: Iterable[str]):
        self.settings = tuple(settings)

    def accepts(self, host: str) -> bool:
        return host in HOSTS

    def wants_body(self, ex: _http.Exchange) -> bool:
        kind, request_id = _route(ex)
        if kind == "submit":
            return True  # the id is in it, or the finished result, or Pruna's error code
        return kind == "status" and ex.ok and _peek(request_id) is not None

    def seen(self, ex: _http.Exchange) -> None:
        kind, request_id = _route(ex)
        if kind == "submit":
            self._submitted(ex)
        elif kind == "status" and ex.ok and isinstance(ex.body, Mapping):
            call = _peek(request_id)
            status = ex.body.get("status")
            if call is not None and status in _ENDED:
                call.finish(_ENDED[status])

    def failed(self, ex: _http.Exchange) -> None:
        kind, _ = _route(ex)
        if kind == "submit" and (call := self._start(ex)) is not None:
            call.completed(None, _http.error_type(ex), cancelled(ex.error))

    def _start(self, ex: _http.Exchange) -> _Call | None:
        try:
            model = ex.request_headers.get(MODEL_HEADER)
            body = ex.request_json()
            data = body.get("input") if isinstance(body, Mapping) else None
            data = data if isinstance(data, Mapping) else {}
            settings = {k: data[k] for k in self.settings if data.get(k) is not None}
            units = {name: n for name, keys in MEDIA.items() if (n := sum(_count(data.get(k)) for k in keys))}
            model = model if isinstance(model, str) and model else None
            return _Call(model, settings, units, (ex.started_at, ex.clock))
        except Exception:
            logger.exception("inference_tracker: could not start tracking a Pruna call")
            return None

    def _submitted(self, ex: _http.Exchange) -> None:
        call = self._start(ex)
        if call is None:
            return
        body = ex.body if isinstance(ex.body, Mapping) else {}
        if not ex.ok:
            request_id = body.get("request_id")
            error_type = code(get(body, "error", "code")) or f"http_{ex.status}"
            call.completed(request_id if isinstance(request_id, str) and request_id else None, error_type)
            return
        request_id = body.get("id")
        request_id = request_id if isinstance(request_id, str) and request_id else None
        status = body.get("status")
        if status in _ENDED:  # Try-Sync: the finished result (no id in it, as Pruna documents it)
            call.completed(request_id, _ENDED[status])
        elif request_id is not None:
            call.submitted(request_id)


def _route(ex: _http.Exchange) -> tuple[str | None, str | None]:
    """What a request to Pruna was: ("submit" | "status" | None, the prediction id in the URL)."""
    if ex.method == "POST" and _SUBMIT.match(ex.path):
        return "submit", None
    match = _STATUS.match(ex.path)
    if ex.method == "GET" and match is not None:
        return "status", match["id"]
    return None, None


def _peek(request_id: Any) -> _Call | None:
    with _open_lock:
        return _open.get(request_id)


def _count(value: Any) -> int:
    """How many media a field holds: one URL, or a list of them."""
    if isinstance(value, str):
        return 1 if value else 0
    if isinstance(value, (list, tuple)):
        return sum(1 for v in value if v)
    return 0
