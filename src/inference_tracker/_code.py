"""Which of the app's functions made a call: the record's "code.functions".

    ["app.features.suggest.prompt_suggestions", "app.handlers.on_request"]   # nearest first

The chain is how the product tells calls apart, so names never need to be written in the code: two features
that share one helper differ in the next function up. The product shows a name given there, never these.

Only the app's own code counts: frames from the standard library, from installed packages (site-packages,
the provider SDKs among them) and from this library are skipped. Each frame is "module.function" (the
function's qualified name, so methods keep their class), never a file path, which differs per machine.
An app that is itself installed as a package (in site-packages) shows no chain.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import threading
from types import FrameType

DEPTH = 5  # how many of the app's functions are kept

_HERE = os.path.dirname(os.path.abspath(__file__))
_NOT_APP: tuple[str, ...] = tuple(
    {os.path.abspath(p) for k in ("stdlib", "platstdlib", "purelib", "platlib") if (p := sysconfig.get_paths().get(k))}
    | {_HERE}
)
_known: dict[str, bool] = {}
_known_lock = threading.Lock()


def _is_app(filename: str) -> bool:
    seen = _known.get(filename)
    if seen is not None:
        return seen
    path = os.path.abspath(filename) if not filename.startswith("<") else ""
    app = (
        bool(path)
        and not path.startswith(_NOT_APP)
        and not any(part in ("site-packages", "dist-packages") for part in path.split(os.sep))
    )
    with _known_lock:
        if len(_known) > 10_000:
            _known.clear()
        _known[filename] = app
    return app


def functions(depth: int = DEPTH, frame: FrameType | None = None) -> list[str]:
    """The app's functions on the current stack (or from `frame` up), nearest first. Never raises."""
    out: list[str] = []
    try:
        f = frame if frame is not None else sys._getframe(1)
        while f is not None and len(out) < depth:
            code = f.f_code
            if _is_app(code.co_filename):
                module = f.f_globals.get("__name__") or "?"
                name = getattr(code, "co_qualname", code.co_name)  # Python 3.11+ keeps the class
                out.append(f"{module}.{name}")
            f = f.f_back
    except Exception:
        return out
    return out
