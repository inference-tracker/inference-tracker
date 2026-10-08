"""One record per inference call: who it was for, what was called, and what it used.

Never holds prompts, images, URLs or outputs. Cost is not here either: it comes from
the provider's bill, matched by request_id or split by the usage in `units`.

A call can be recorded in two parts when its result arrives somewhere else (fal's
queue with a webhook, RunPod /run, Veo's long-running operations): a "start" record
when it is sent and a "finish" record when the result comes back. The server joins
them on provider + request_id. A call seen from start to end in one place is one
"call" record.

On the wire, fields that OpenTelemetry's GenAI conventions name use those names.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from ._context import current as _current_context

Phase = Literal["call", "start", "finish"]
Status = Literal["ok", "error"]


@dataclass
class Record:
    provider: str  # "fal", "gcp.vertex_ai", "openai", "azure.ai.openai", "aws.bedrock"
    model: str  # model or endpoint, e.g. "fal-ai/kling-video/v2.6/pro/motion-control"
    phase: Phase = "call"
    request_id: str | None = None  # the provider's id for this call: how the bill is matched
    status: Status | None = None  # None on a "start" record
    error_type: str | None = None  # the exception's class name or the provider's error code, never a message
    started_at: float | None = None  # unix seconds
    duration_s: float | None = None
    # Usage as the provider reports it: input_tokens, output_tokens, cached_input_tokens,
    # images, video_seconds, audio_seconds... Numbers only.
    units: dict[str, float] = field(default_factory=dict)
    # Settings that change the price (resolution, requested duration, audio on/off), copied
    # by name by the wrappers. Never the call's whole input: that is how prompts stay out.
    settings: dict[str, Any] = field(default_factory=dict)
    # Retries the app marks itself (see the README); None when not marked.
    call_id: str | None = None
    attempt: int | None = None
    # Filled from the current context when the record is made.
    user_id: str | None = None
    task_id: str | None = None
    parent_id: str | None = None  # the `id` of the record of the call that made this one
    id: str = field(default_factory=lambda: uuid.uuid4().hex)  # lets the server drop duplicates
    recorded_at: float = field(default_factory=time.time)

    @classmethod
    def new(cls, provider: str, model: str, **fields) -> Record:
        ctx = _current_context()
        fields.setdefault("user_id", ctx.user_id)
        fields.setdefault("task_id", ctx.task_id)
        fields.setdefault("parent_id", ctx.parent_id)
        return cls(provider=provider, model=model, **fields)

    def to_wire(self) -> dict:
        wire = {
            "id": self.id,
            "phase": self.phase,
            "recorded_at": self.recorded_at,
            "gen_ai.provider.name": self.provider,
            "gen_ai.request.model": self.model,
            "gen_ai.response.id": self.request_id,
            "status": self.status,
            "error.type": self.error_type,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "units": _numbers(self.units),
            "settings": _settings(self.settings),
            "call_id": self.call_id,
            "attempt": self.attempt,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "parent_id": self.parent_id,
        }
        return {k: v for k, v in wire.items() if v is not None and v != {}}


def _numbers(units: dict) -> dict[str, float]:
    return {
        str(k): v
        for k, v in units.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }


def _settings(settings: dict) -> dict[str, Any]:
    """Kept as given. Only a value that can't be sent as JSON is left out."""
    out: dict[str, Any] = {}
    for k, v in settings.items():
        try:
            json.dumps(v, allow_nan=False)
        except (TypeError, ValueError):
            continue
        out[str(k)] = v
    return out


def error_name(error: BaseException) -> str:
    """What goes in error_type for an exception: its class name. The message can quote the input."""
    return type(error).__name__
