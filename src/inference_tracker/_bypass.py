"""Set while a track() wrapper looks up its client's own attribute or runs a call it records itself, so the
tracking attached at init() (_attach) stands aside: one record per call."""

from __future__ import annotations

import contextvars
import functools
import inspect
from collections.abc import Callable
from typing import Any

_on: contextvars.ContextVar[bool] = contextvars.ContextVar("inference_tracker_bypass", default=False)


def on() -> bool:
    return _on.get()


class bypass:
    """with bypass(): ... (the attached tracking doesn't see what runs inside)."""

    def __enter__(self) -> None:
        self._token = _on.set(True)

    def __exit__(self, *exc: Any) -> None:
        _on.reset(self._token)


def methods(cls: type) -> type:
    """Runs each public method the class defines in bypass(): for wrappers that record the calls they make."""
    for name, fn in list(vars(cls).items()):
        if name.startswith("_") or not inspect.isfunction(fn):
            continue
        setattr(cls, name, _wrapped(fn))
    return cls


def _wrapped(fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def run_async(*args: Any, **kwargs: Any) -> Any:
            with bypass():
                return await fn(*args, **kwargs)

        return run_async

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        with bypass():
            return fn(*args, **kwargs)

    return run
