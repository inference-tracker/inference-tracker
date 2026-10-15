"""A stand-in for an SDK client that intercepts only the methods that make inference calls.

    client.chat.completions.create(...)   path ("chat", "completions", "create")

Every other attribute is the client's own, unchanged. The wrappers (openai.py, google.py) say which paths
to intercept and what to do around them.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

Path = tuple[str, ...]
# Given the path, the original method and whether it is async, returns the method to use instead.
Wrap = Callable[[Path, Callable[..., Any], bool], Callable[..., Any]]


class Proxy:
    def __init__(self, target: Any, paths: frozenset[Path], wrap: Wrap, path: Path = ()):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_paths", paths)
        object.__setattr__(self, "_wrap", wrap)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._target, name)
        path = self._path + (name,)
        if path in self._paths and callable(value):
            return self._wrap(path, value, inspect.iscoroutinefunction(value))
        if any(p[: len(path)] == path for p in self._paths):
            return Proxy(value, self._paths, self._wrap, path)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def __repr__(self) -> str:
        return f"<tracked {self._target!r}>"


def get(obj: Any, *names: str) -> Any:
    """obj.a.b or obj["a"]["b"], None when any step is missing. SDK objects and plain dicts alike."""
    for name in names:
        if obj is None:
            return None
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return obj


def number(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def put(units: dict[str, float], name: str, value: Any) -> None:
    n = number(value)
    if n is not None:
        units[name] = n
