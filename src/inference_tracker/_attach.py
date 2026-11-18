"""Attaches to the provider SDKs once, at init(), so every client the app makes is tracked, wherever and whenever
it's made: nothing to wrap, and a call added later is recorded without anyone remembering to.

- openai (OpenAI, AsyncOpenAI and the Azure ones) and google-genai (genai.Client): the client's attributes that
  lead to inference calls (client.chat, client.models...) are looked up through the same wrappers track() uses.
  Clients made by LiteLLM are left alone: LiteLLM's own callback records those calls.
- Plain HTTP (httpx.Client, httpx.AsyncClient, requests.Session): each session is watched on its first request
  for the providers tracked over HTTP: fal (fal_client's own session included), eachlabs, Pruna, Modal web
  endpoints, and OpenAI's API called without the openai package. Other hosts pass through unseen.
- LiteLLM: its callback is added (litellm.track()).

A package imported after init() is attached to when it's imported. Only the class is changed, in memory: the app's
objects and calls stay its own, a call is never changed, and tracking never raises. If a package changed what we
attach to, that package isn't attached (one warning in the log) and the app runs on.

track() still works: a client wrapped with it is recorded by its wrapper only, never twice.
"""

from __future__ import annotations

import importlib.abc
import inspect
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

from . import _bypass, _client, _http

logger = logging.getLogger("inference_tracker")

_VIEW = "_inference_tracker_view"  # an attached client's tracked stand-in, kept on the client
_SKIP = "_inference_tracker_skip"  # a client LiteLLM made: its calls are LiteLLM's to record
_WATCHED = "_inference_tracker_watched"  # an HTTP session already given the watchers

_done: set[str] = set()
_lock = threading.Lock()
_on = False  # set by attach(); the class changes do nothing until then (and after off())


# --- what is attached ------------------------------------------------------------------------------


def _openai() -> None:
    import openai

    from . import openai as tracked

    names = sorted({p[0] for p in tracked.PATHS} - {"post", "with_options", "copy"})
    for cls in (openai.OpenAI, openai.AsyncOpenAI):
        _skip_litellm_made(cls)
        _lookups(cls, names, lambda client: tracked.track(client))


def _google() -> None:
    from google import genai

    from . import google as tracked

    names = sorted({p[0] for p in tracked.PATHS})
    _lookups(genai.Client, names, lambda client: tracked.track(client))


def _httpx() -> None:
    # The classes where httpx defines them: httpx.Client may have been replaced (a test's stand-in, say).
    from httpx import _client as httpx_client

    _sessions(httpx_client.Client, "send", is_async=False)
    _sessions(httpx_client.AsyncClient, "send", is_async=True)


def _requests() -> None:
    from requests import sessions

    _sessions(sessions.Session, "request", is_async=False)  # before the request is prepared: its hooks come from here
    _sessions(sessions.Session, "send", is_async=False)


def _litellm() -> None:
    from . import litellm

    litellm.track()


# Module that triggers it → what to attach.
TARGETS: dict[str, Callable[[], None]] = {
    "openai": _openai,
    "google.genai": _google,
    "httpx": _httpx,
    "requests": _requests,
    "litellm": _litellm,
}


def attach() -> None:
    """Attaches to every target already imported, and to the others when they are. Safe to call again."""
    global _on
    _on = True
    for module in TARGETS:
        if module in sys.modules:
            _run(module)
    _install_import_hook()


def off() -> None:
    """Stops the attached tracking (the classes stay changed, but stand aside). For tests."""
    global _on
    _on = False


def _run(module: str) -> None:
    with _lock:
        if module in _done:
            return
        _done.add(module)
    try:
        TARGETS[module]()
    except Exception:
        logger.warning("inference_tracker: couldn't attach to %s; its calls won't be recorded", module, exc_info=True)


# --- attribute lookups ---------------------------------------------------------------------------


class _Lookup:
    """Replaces one attribute on an SDK client class (client.chat, client.models...). Looked up on a client, it
    gives the attribute as the client's tracked stand-in has it."""

    def __init__(self, name: str, original: Any, make: Callable[[Any], Any]):
        self.name, self.original, self.make = name, original, make

    def _own(self, obj: Any, cls: Any) -> Any:
        get = getattr(self.original, "__get__", None)
        return get(obj, cls) if get is not None else self.original

    def __get__(self, obj: Any, cls: Any = None) -> Any:
        if obj is None or not _on or _bypass.on() or not _client.active() or getattr(obj, _SKIP, False):
            return self._own(obj, cls)
        try:
            view = obj.__dict__.get(_VIEW)
            if view is None:
                view = self.make(obj)
                obj.__dict__[_VIEW] = view
            return getattr(view, self.name)
        except Exception:
            logger.exception("inference_tracker: could not track %s.%s", type(obj).__name__, self.name)
            return self._own(obj, cls)

    def __set__(self, obj: Any, value: Any) -> None:
        setter = getattr(self.original, "__set__", None)
        if setter is not None:
            setter(obj, value)
        else:
            obj.__dict__[self.name] = value

    def __delete__(self, obj: Any) -> None:
        deleter = getattr(self.original, "__delete__", None)
        if deleter is not None:
            deleter(obj)
        else:
            obj.__dict__.pop(self.name, None)


def _lookups(cls: type, names: list[str], make: Callable[[Any], Any]) -> None:
    for name in names:
        original = inspect.getattr_static(cls, name)  # raises if the SDK no longer has it: the target isn't attached
        if isinstance(original, _Lookup):
            continue
        setattr(cls, name, _Lookup(name, original, make))


def _skip_litellm_made(cls: type) -> None:
    """Marks the clients LiteLLM makes, so their calls are recorded once, by LiteLLM's callback."""
    init = cls.__init__
    if getattr(init, "_inference_tracker", False):
        return

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        try:
            f = sys._getframe(1)
            while f is not None:
                if str(f.f_globals.get("__name__", "")).startswith("litellm"):
                    self.__dict__[_SKIP] = True
                    break
                f = f.f_back
        except Exception:
            pass

    __init__._inference_tracker = True  # type: ignore[attr-defined]
    __init__.__wrapped__ = init  # type: ignore[attr-defined]
    cls.__init__ = __init__  # type: ignore[misc]


# --- HTTP sessions --------------------------------------------------------------------------------


def watchers() -> list[_http.Watcher]:
    """The watchers every session gets, one per provider tracked over HTTP."""
    from . import eachlabs, fal, modal, openai, pruna

    with _lock:
        global _watchers
        if _watchers is None:
            _watchers = [
                fal._HttpWatcher(fal.SETTINGS),
                eachlabs._HttpWatcher(eachlabs.SETTINGS),
                pruna._HttpWatcher(pruna.SETTINGS),
                modal._HttpWatcher(()),
                openai._HttpWatcher(),
            ]
        return _watchers


_watchers: list[_http.Watcher] | None = None


def learn(watcher: Any) -> None:
    """A track_http() call's own settings (or Modal app names) are added to the attached watcher of that
    provider, so sessions the app didn't wrap get them too."""
    try:
        for w in watchers():
            if w.name != watcher.name:
                continue
            for attr in ("settings", "apps"):
                mine, theirs = getattr(w, attr, None), getattr(watcher, attr, None)
                if isinstance(mine, (tuple, list)) and isinstance(theirs, (tuple, list)):
                    merged = list(dict.fromkeys([*mine, *theirs]))
                    if attr == "apps":
                        merged.sort(key=len, reverse=True)  # longest first, as modal's watcher keeps them
                    setattr(w, attr, tuple(merged) if isinstance(mine, tuple) else merged)
    except Exception:
        logger.exception("inference_tracker: could not add a client's settings to the attached tracking")


def _watch(session: Any) -> None:
    try:
        session.__dict__[_WATCHED] = True
        for w in watchers():
            _http.watch(session, w)
    except Exception:
        logger.exception("inference_tracker: could not watch an HTTP session")


def _sessions(cls: type, method: str, *, is_async: bool) -> None:
    original = cls.__dict__.get(method) or getattr(cls, method)
    if getattr(original, "_inference_tracker", False):
        return

    def needs(session: Any) -> bool:
        try:
            return _on and _client.active() and not session.__dict__.get(_WATCHED, False)
        except Exception:
            return False

    if is_async:

        async def attached_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            if needs(self):
                _watch(self)
                return await getattr(self, method)(*args, **kwargs)  # the watched session's own method
            return await original(self, *args, **kwargs)

        attached: Any = attached_async
    else:

        def attached_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
            if needs(self):
                _watch(self)
                return getattr(self, method)(*args, **kwargs)
            return original(self, *args, **kwargs)

        attached = attached_sync
    attached._inference_tracker = True
    attached.__wrapped__ = original
    attached.__name__ = getattr(original, "__name__", method)
    setattr(cls, method, attached)


# --- packages imported later ---------------------------------------------------------------------


class _Finder(importlib.abc.MetaPathFinder):
    """Runs the attach for a target module right after it is first imported."""

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name not in TARGETS or name in _done:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            try:
                spec = finder.find_spec(name, path, target)
            except Exception:
                continue
            if spec is not None:
                break
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None or not hasattr(loader, "exec_module"):
            return None
        exec_module = loader.exec_module

        def exec_and_attach(module: Any) -> None:
            exec_module(module)
            _run(name)

        try:
            loader.exec_module = exec_and_attach
        except Exception:
            return None
        return spec


_finder = _Finder()


def _install_import_hook() -> None:
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
