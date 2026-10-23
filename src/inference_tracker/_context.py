"""Who a call is for: the user, the task, and the call that made it.

Set once where work starts, read by every tracked call inside it:

    with inference_tracker.context(user_id=uid, task_id=task.id):
        ...  # every tracked call in here is tagged with uid and task.id

A call made inside another call (a model called by one of our fal apps) names that call as its
parent, the way the provider's bill knows it: the provider and its request id.

    with inference_tracker.context(parent_provider="fal", parent_request_id=rid):
        ...

The server finds the user and task of such a call by walking up its parents, so the app inside
doesn't need to be told them.

Built on contextvars, so each asyncio task (and each request in a server that runs several at
once) keeps its own values. asyncio.to_thread copies them into the thread; a plain
ThreadPoolExecutor does not (use contextvars.copy_context().run).
"""

from __future__ import annotations

import contextvars
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

_MAX_LEN = 256


@dataclass(frozen=True)
class Context:
    user_id: str | None = None
    task_id: str | None = None
    # The call that made this one, as its provider knows it: "fal" and fal's request id, say.
    parent_provider: str | None = None
    parent_request_id: str | None = None


_current: contextvars.ContextVar[Context] = contextvars.ContextVar("inference_tracker_context", default=Context())  # noqa: B039 (Context is frozen)
# Inside a context(flush=True) block: an inner block with flush=True leaves the flush to the outer one.
_flushing: contextvars.ContextVar[bool] = contextvars.ContextVar("inference_tracker_flushing", default=False)


def current() -> Context:
    return _current.get()


@contextmanager
def context(
    *,
    user_id: Any = None,
    task_id: Any = None,
    parent_provider: Any = None,
    parent_request_id: Any = None,
    flush: bool = False,
) -> Generator[Context, None, None]:
    """Values left out are kept from the outer context. flush=True sends what's waiting when the
    block ends: for runtimes that freeze the process after the response (see _client.py). Only the
    outermost block with flush=True flushes: blocks inside it leave it to that one, so nesting them
    costs one flush, not one per block."""
    given = {
        "user_id": user_id,
        "task_id": task_id,
        "parent_provider": parent_provider,
        "parent_request_id": parent_request_id,
    }
    changes = {k: c for k, v in given.items() if v is not None and (c := _clean(v)) is not None}
    token = _current.set(replace(_current.get(), **changes))
    flushes = flush and not _flushing.get()
    flushing = _flushing.set(True) if flushes else None
    try:
        yield _current.get()
    finally:
        if flushing is not None:
            _flushing.reset(flushing)
        _current.reset(token)
        if flushes:
            from ._client import flush as _flush

            _flush()


def _clean(value: Any) -> str | None:
    text = str(value).strip()[:_MAX_LEN]
    return text or None
