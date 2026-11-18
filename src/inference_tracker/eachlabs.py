"""Tracks calls to eachlabs' API (https://api.eachlabs.ai), made over plain HTTP.

    import inference_tracker

    http = inference_tracker.eachlabs.track_http(httpx.AsyncClient())   # or httpx.Client, requests.Session

How eachlabs' API works (docs.eachlabs.ai, read 2026-11-18): a prediction is created with POST /v1/prediction,
{"model": slug, "version": "1.0.0", "input": {...}}, and eachlabs answers {"status": "success", "predictionID": id}
("success" means it was created, not that it ran). The app then polls GET /v1/prediction/<id> until its status is
success, error or cancelled, or eachlabs posts the same answer to the app's webhook_url.

What is recorded:
- a created prediction: a "start" record with the prediction id as request id, and the poll answer that says the
  prediction ended: the "finish", ok, or "error" / "cancelled" (status "error"). The status may be polled with
  another watched session than the one that created it.
- a create eachlabs refused (an error answer): one "call" record with "http_<status>" (eachlabs answers with a
  sentence, which isn't copied).
- a create that got no answer (a timeout, a broken connection): one "call" record with the exception's class name
  ("cancelled" when the app cancelled it). A poll without an answer, or with an error answer, changes nothing: the
  prediction may still be running.
Predictions answered to a webhook aren't seen finishing here. Other paths and hosts pass through unseen. The call
itself is never changed, and tracking never raises: a problem in it is logged and the call goes on.

Usage and cost: the ended prediction's metrics carry its cost, in dollars at list price with any credit rebate taken
off (metrics.cost), and on metered models the billed seconds and output size. They're kept in the finish record's
units as cost_usd, billed_seconds, predict_seconds and output_gb. The settings that change the price are copied from
the input by name (duration, resolution...), with the model's version; reference media are counted, never copied.
Never the prompt, nor any media URL.

The model is eachlabs' slug ("p-video-upscale"). The parent is taken from the context, as for every record.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

from . import _attach, _client, _http
from ._record import Record, cancelled

logger = logging.getLogger("inference_tracker")

PROVIDER = "eachlabs"
HOSTS = frozenset({"api.eachlabs.ai"})
# Input fields copied onto the record because they change the price (or say how long the video is).
SETTINGS = (
    "duration",
    "resolution",
    "target_resolution",
    "upscale_factor",
    "mode",
    "draft",
    "fps",
    "target_fps",
    "aspect_ratio",
    "num_frames",
    "num_outputs",
    "num_images",
    "width",
    "height",
    "size",
    "image_size",
    "quality",
    "go_fast",
    "turbo",
)
# Input fields holding media URLs, counted (never copied): reference images, videos and audio change the price.
MEDIA = {
    "input_images": ("image", "images", "image_url", "image_urls", "first_frame_image", "last_frame_image"),
    "input_videos": ("video", "videos", "video_url"),
    "input_audios": ("audio", "audio_url"),
}
# The ended prediction's metrics, as units.
METRICS = {"cost": "cost_usd", "billed_seconds": "billed_seconds", "predict_time": "predict_seconds", "output_gb": "output_gb"}
# Predictions created and not seen ended yet, by id, shared by every watched session. Beyond this many, the oldest
# is forgotten.
MAX_OPEN = 1000
_ENDED = {"success": None, "succeeded": None, "completed": None, "error": "error", "failed": "error", "cancelled": "cancelled", "canceled": "cancelled"}

_CREATE = re.compile(r"^/v\d+/prediction/?$")
_STATUS = re.compile(r"^/v\d+/prediction/(?P<id>[^/]+)/?$")

_open: OrderedDict[str, _Call] = OrderedDict()
_open_lock = threading.Lock()


def track_http(client: Any, *, settings: Iterable[str] = SETTINGS) -> Any:
    """Watches an httpx.Client, httpx.AsyncClient or requests.Session for calls to eachlabs' API, and returns the
    same client. See the module docstring."""
    try:
        watcher = _HttpWatcher(settings)
        _attach.learn(watcher)
        if not _http.watch(client, watcher):
            logger.warning("inference_tracker: can't watch a %s for eachlabs calls", type(client).__name__)
    except Exception:
        logger.exception("inference_tracker: could not watch a client for eachlabs calls")
    return client


# --- one call ----------------------------------------------------------------------------------


class _Call:
    """The record of one prediction while this process follows it."""

    def __init__(self, model: str | None, settings: dict[str, Any], units: dict[str, float], started: tuple):
        started_at, self._clock = started
        self.record = Record.new(PROVIDER, model, phase="start", started_at=started_at, settings=settings)
        self.units = units
        self.done = False

    def created(self, request_id: str) -> None:
        self.record.request_id = request_id
        _client.record(self.record)
        with _open_lock:
            _open[request_id] = self
            while len(_open) > MAX_OPEN:
                _open.popitem(last=False)

    def finish(self, error_type: str | None, metrics: dict[str, float]) -> None:
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
                units={**self.units, **metrics},
                user_id=start.user_id,
                task_id=start.task_id,
                parent_provider=start.parent_provider,
                parent_request_id=start.parent_request_id,
                call_name=start.call_name,
            )
        )

    def completed(self, error_type: str | None, stopped: bool = False) -> None:
        """A create that was refused or got no answer: one "call" record, without a request id."""
        if self.done:
            return
        self.done = True
        r = self.record
        r.phase = "call"
        r.status, r.error_type = ("cancelled" if stopped else "error" if error_type else "ok"), error_type
        r.duration_s, r.units = self._elapsed(), dict(self.units)
        _client.record(r)

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._clock, 3)


# --- over HTTP ---------------------------------------------------------------------------------------


class _HttpWatcher:
    name = "eachlabs"

    def __init__(self, settings: Iterable[str]):
        self.settings = tuple(settings)

    def accepts(self, host: str) -> bool:
        return host in HOSTS

    def wants_body(self, ex: _http.Exchange) -> bool:
        kind, request_id = _route(ex)
        if kind == "create":
            return True  # the prediction's id is in it
        return kind == "status" and ex.ok and _peek(request_id) is not None

    def seen(self, ex: _http.Exchange) -> None:
        kind, request_id = _route(ex)
        if kind == "create":
            self._created(ex)
        elif kind == "status" and ex.ok and isinstance(ex.body, Mapping):
            call = _peek(request_id)
            status = ex.body.get("status")
            status = status.lower() if isinstance(status, str) else status
            if call is not None and status in _ENDED:
                call.finish(_ENDED[status], _metrics(ex.body.get("metrics")))

    def failed(self, ex: _http.Exchange) -> None:
        kind, _ = _route(ex)
        if kind == "create" and (call := self._start(ex)) is not None:
            call.completed(_http.error_type(ex), cancelled(ex.error))

    def _start(self, ex: _http.Exchange) -> _Call | None:
        try:
            body = ex.request_json()
            body = body if isinstance(body, Mapping) else {}
            data = body.get("input")
            data = data if isinstance(data, Mapping) else {}
            settings = {k: data[k] for k in self.settings if data.get(k) is not None}
            version = body.get("version")
            if isinstance(version, str) and version:
                settings["version"] = version
            units = {name: n for name, keys in MEDIA.items() if (n := sum(_count(data.get(k)) for k in keys))}
            model = body.get("model")
            model = model if isinstance(model, str) and model else None
            return _Call(model, settings, units, (ex.started_at, ex.clock))
        except Exception:
            logger.exception("inference_tracker: could not start tracking an eachlabs call")
            return None

    def _created(self, ex: _http.Exchange) -> None:
        call = self._start(ex)
        if call is None:
            return
        body = ex.body if isinstance(ex.body, Mapping) else {}
        request_id = body.get("predictionID") or body.get("id")
        if not ex.ok or not isinstance(request_id, str) or not request_id:
            call.completed(f"http_{ex.status}" if not ex.ok else "no_prediction_id")
            return
        call.created(request_id)


def _route(ex: _http.Exchange) -> tuple[str | None, str | None]:
    """What a request to eachlabs was: ("create" | "status" | None, the prediction id in the URL)."""
    if ex.method == "POST" and _CREATE.match(ex.path):
        return "create", None
    match = _STATUS.match(ex.path)
    if ex.method == "GET" and match is not None:
        return "status", match["id"]
    return None, None


def _peek(request_id: Any) -> _Call | None:
    with _open_lock:
        return _open.get(request_id)


def _metrics(m: Any) -> dict[str, float]:
    """The ended prediction's cost and metered usage, numbers only."""
    if not isinstance(m, Mapping):
        return {}
    out = {}
    for key, unit in METRICS.items():
        v = m.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[unit] = float(v)
    return out


def _count(value: Any) -> int:
    """How many media a field holds: one URL, or a list of them."""
    if isinstance(value, str):
        return 1 if value else 0
    if isinstance(value, (list, tuple)):
        return sum(1 for v in value if v)
    return 0
