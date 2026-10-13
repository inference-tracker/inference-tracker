"""One HTTP POST of a batch to the ingest endpoint, and what to do about the answer."""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

Outcome = Literal["sent", "retry", "rejected"]


@dataclass
class Result:
    outcome: Outcome
    status: int | None = None  # HTTP status, None when the request never got an answer
    retry_after: float | None = None  # seconds, from a 429's Retry-After header


def post(url: str, key: str, body: dict, timeout: float) -> Result:
    data = gzip.compress(json.dumps(body, allow_nan=False, separators=(",", ":")).encode())
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Result("sent", response.status)
    except urllib.error.HTTPError as e:
        # 429 and 5xx: the server is busy or broken, the same batch may go through later.
        if e.code == 429 or e.code >= 500:
            return Result("retry", e.code, _seconds(e.headers.get("Retry-After")))
        # Any other 4xx (bad key, rejected body): sending it again won't change the answer.
        return Result("rejected", e.code)
    except (urllib.error.URLError, OSError, TimeoutError):
        return Result("retry")


def _seconds(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None
