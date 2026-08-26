"""FastAPI application, configuration and shared state.

Also the project's Python version gate. This package is imported before
anything else in every entry point, so a too-old interpreter fails here with an
actionable message rather than as a ``SyntaxError`` from a PEP 604 annotation or
a ``TypeError`` from ``dataclass(slots=True)`` several imports deep.
"""

import sys

#: Minimum supported Python. Two hard requirements set this floor:
#: ``dataclass(slots=True)`` (3.10+), used throughout ``app.models`` to keep the
#: per-frame allocation cost down, and PEP 604 ``X | None`` annotations in the
#: Pydantic schemas, which Pydantic evaluates at runtime.
#:
#: Note this is 3.10, not the 3.9 named in the build spec. Raspberry Pi OS
#: Trixie ships Python 3.13, so the intended target is comfortably satisfied;
#: only an older distro or a hand-built 3.9 would trip this. Kept as a floor
#: rather than raised to match the rig: nothing here needs 3.13, and a floor that
#: tracks whatever the current image ships would reject working interpreters.
MIN_PYTHON = (3, 10)

if sys.version_info < MIN_PYTHON:
    raise RuntimeError(
        f"GhostBall needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer, "
        f"but this is {sys.version_info.major}.{sys.version_info.minor}. "
        "Raspberry Pi OS Trixie (64-bit) ships Python 3.13, which is the "
        "supported target."
    )
