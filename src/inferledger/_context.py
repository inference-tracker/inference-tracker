"""Who a call is for: the user, the task, and the call that made it.

Set once where work starts, read by every tracked call inside it:

    with inferledger.context(user_id=uid, task_id=task.id):
        ...  # every tracked call in here is tagged with uid and task.id

Built on contextvars, so each asyncio task (and each request in a server that
runs several at once) keeps its own values. asyncio.to_thread copies them into
the thread; a plain ThreadPoolExecutor does not (use contextvars.copy_context().run).
"""

from __future__ import annotations

import contextvars
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

# The field an app adds to a request so the service on the other side can restore the context.
CARRY_FIELD = "inferledger"

_KEYS = ("user_id", "task_id", "parent_id")
_MAX_LEN = 256


@dataclass(frozen=True)
class Context:
    user_id: str | None = None
    task_id: str | None = None
    # The provider request id of the call that made this one (a workflow, our own fal app).
    parent_id: str | None = None


_current: contextvars.ContextVar[Context] = contextvars.ContextVar("inferledger_context", default=Context())  # noqa: B039 (Context is frozen)


def current() -> Context:
    return _current.get()


@contextmanager
def context(
    *, user_id: Any = None, task_id: Any = None, parent_id: Any = None, flush: bool = False
) -> Generator[Context, None, None]:
    """Values left out are kept from the outer context. flush=True sends what's waiting when the
    block ends: for runtimes that freeze the process after the response (see _client.py)."""
    given = {"user_id": user_id, "task_id": task_id, "parent_id": parent_id}
    changes = {k: c for k, v in given.items() if v is not None and (c := _clean(v)) is not None}
    token = _current.set(replace(_current.get(), **changes))
    try:
        yield _current.get()
    finally:
        _current.reset(token)
        if flush:
            from ._client import flush as _flush

            _flush()


def carry() -> dict[str, str]:
    """The current context as plain data, to put in a request: {CARRY_FIELD: carry()}."""
    ctx = _current.get()
    return {k: getattr(ctx, k) for k in _KEYS if getattr(ctx, k) is not None}


@contextmanager
def restore(data: Any) -> Generator[Context, None, None]:
    """The other side of carry(). Takes what carry() made, or the whole request input that holds it.
    Anything missing or malformed is ignored: a bad field never breaks the request."""
    if isinstance(data, Mapping) and isinstance(data.get(CARRY_FIELD), Mapping):
        data = data[CARRY_FIELD]

    def get(key: str) -> str | int | None:
        value = data.get(key) if isinstance(data, Mapping) else None
        return value if isinstance(value, (str, int)) else None

    with context(user_id=get("user_id"), task_id=get("task_id"), parent_id=get("parent_id")) as ctx:
        yield ctx


def _clean(value: Any) -> str | None:
    text = str(value).strip()[:_MAX_LEN]
    return text or None
