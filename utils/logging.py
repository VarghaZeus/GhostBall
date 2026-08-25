"""Logging setup.

Debugging on a Pi bolted under a pool table is painful: there is often no
keyboard attached and the projector is showing the game, not a terminal. So
logging is deliberately generous, always timestamped with milliseconds, and
mirrored to a rotating file that survives a power cut.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONFIGURED = False

#: Millisecond timestamps matter here -- at 30 FPS, second-resolution logs
#: cannot tell you which frame a dropped-frame warning belongs to.
_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-22s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class _ColorFormatter(logging.Formatter):
    """Colourise levels when writing to a TTY.

    Falls back to plain text when stderr is redirected, so log files do not get
    littered with escape codes.
    """

    _COLORS = {
        logging.DEBUG: "\033[38;5;244m",
        logging.INFO: "\033[38;5;39m",
        logging.WARNING: "\033[38;5;214m",
        logging.ERROR: "\033[38;5;196m",
        logging.CRITICAL: "\033[1;38;5;196m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        color = self._COLORS.get(record.levelno)
        return f"{color}{text}{self._RESET}" if color else text


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Path | None = None,
) -> None:
    """Configure root logging. Idempotent -- safe to call from several entry points.

    Both ``app.main`` and the calibration app call this, and uvicorn may import
    the app twice, so repeat calls are a normal case rather than a bug.

    Args:
        level: Root log level name, e.g. ``"DEBUG"``.
        log_to_file: Also write to a rotating file under ``log_dir``.
        log_dir: Where to put ``ar_pool.log``. Defaults to ``data/logs``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger(__name__).debug("logging already configured; skipping")
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for existing in list(root.handlers):
        root.removeHandler(existing)

    console = logging.StreamHandler(stream=sys.stderr)
    formatter_cls = _ColorFormatter if sys.stderr.isatty() else logging.Formatter
    console.setFormatter(formatter_cls(_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    if log_to_file:
        from app.config import LOG_DIR

        target_dir = log_dir or LOG_DIR
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            # 5 x 5 MB caps the SD card cost of a long session at 25 MB.
            file_handler = logging.handlers.RotatingFileHandler(
                target_dir / "ar_pool.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
            root.addHandler(file_handler)
        except OSError as exc:
            # A read-only or full SD card must not stop the game starting.
            root.warning("file logging disabled (%s): %s", target_dir, exc)

    # These are chatty at DEBUG and drown out our own frame logs.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("picamera2").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    _CONFIGURED = True
    root.debug("logging configured at %s (file=%s)", level, log_to_file)


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Convenience wrapper so callers need not import logging."""
    return logging.getLogger(name)


class _Unset:
    """Distinct from ``None``, which is a legitimate state value."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()


class ChangeLogger:
    """Logs a recurring condition once, and again only when it changes.

    Built for the per-frame path. Anything the vision loop notices, it notices
    thirty times a second, and a plain ``logger.warning`` in that path emits
    1,800 identical lines a minute -- which does not just waste an SD card, it
    buries every other line in the log. That is not hypothetical: the felt
    coverage notice in :mod:`vision.detection` did exactly this.

    The obvious alternatives are both worse. Logging every Nth occurrence still
    scales with runtime and still says nothing new. Logging once and never again
    loses the transition *back*, which is the interesting half -- "detection
    started failing" and "detection recovered" are both events, and a log that
    only ever reports the first leaves you unable to tell a fixed problem from
    an ongoing one.

    So state is compared, not counted::

        _changes.report("cloth", "adaptive", logging.INFO, "thresholds cover %.0f%%", pct)
        ...
        _changes.recovered("cloth", logging.INFO, "thresholds match the cloth again")

    Pick the state value carefully: it is the identity of the condition, not its
    measurement. Passing a raw percentage would re-log on every frame, because
    the percentage jitters. Pass a constant, or the exception message, or
    whatever is stable while the condition persists.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._active: dict[str, object] = {}

    def report(self, key: str, state: object, level: int, message: str, *args: object) -> bool:
        """Log ``message`` only when ``state`` differs from the last one for ``key``.

        Returns whether it logged, so a caller can hang extra one-off work off
        the transition.
        """
        if self._active.get(key, _UNSET) == state:
            return False
        self._active[key] = state
        self._logger.log(level, message, *args)
        return True

    def recovered(self, key: str, level: int, message: str, *args: object) -> bool:
        """Log that ``key``'s condition has cleared -- but only if it was active.

        Silent when nothing was wrong, which is what makes it safe to call
        unconditionally on the success path. Without that guard every healthy
        frame from startup would announce a recovery from a problem that never
        happened.
        """
        if self._active.pop(key, _UNSET) is _UNSET:
            return False
        self._logger.log(level, message, *args)
        return True

    def is_active(self, key: str) -> bool:
        """Whether ``key``'s condition is currently being reported."""
        return key in self._active

    def clear(self, key: str | None = None) -> None:
        """Forget one key, or all of them.

        Module-level instances outlive any single run, so tests -- and anything
        that restarts the pipeline in-process -- need a way to reset. Without
        it, the second run in a process silently logs nothing.
        """
        if key is None:
            self._active.clear()
        else:
            self._active.pop(key, None)
