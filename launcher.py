#!/usr/bin/env python3
"""Start the whole system, with a preflight check first.

``python -m app.main`` is the application. This is the way to *launch* it, and
it differs in three ways that all come from the same place -- the machine this
runs on is a Pi bolted under a pool table, usually with no keyboard attached:

* **It runs from anywhere.** The project is imported as top-level packages
  (``app``, ``vision``, ``physics``...), so ``python -m app.main`` only works
  with the working directory set to this folder. Getting that wrong produces
  ``ModuleNotFoundError: No module named 'app'``, which is a confusing way to
  learn about ``cd``. This file puts its own directory on ``sys.path``, so
  ``python /home/pi/ar_pool_table/launcher.py`` works from anywhere, which is
  what a systemd unit or an SSH one-liner actually does.

* **It checks before it starts.** Every failure below is one that otherwise
  surfaces as a traceback several imports deep, or -- worse -- as a system that
  starts, looks healthy, and projects something wrong. A missing dependency, a
  config file with a typo in it, a port still held by a previous instance, no
  calibration on disk. Each check reports what to *do*, not just what is wrong.

* **It tells you where the panel is.** The control panel is designed to be used
  from a phone while standing at the table, and ``http://localhost:8000`` is
  useless from a phone. This prints the LAN address.

Usage::

    python launcher.py                  # preflight, then camera + projector + web
    python launcher.py --mock           # no hardware needed
    python launcher.py --check          # preflight only, then exit
    python launcher.py --force          # start even if a check failed

Every ``app.main`` flag works here too -- ``--headless``, ``--frames``,
``--profile``, ``--config``, ``--port`` and the rest -- because the parser is
imported from there rather than redeclared.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

#: This directory. Everything is imported relative to it.
PACKAGE_DIR = Path(__file__).resolve().parent

#: Third-party imports the application cannot start without, as
#: ``(import name, pip name)``. The two differ often enough -- ``cv2`` vs
#: ``opencv-python``, ``yaml`` vs ``pyyaml`` -- that telling someone to
#: ``pip install cv2`` would send them to a package that is not the one they
#: need.
REQUIRED_MODULES = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn[standard]"),
    ("pydantic", "pydantic"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("yaml", "pyyaml"),
)

#: Not required, but their absence costs something worth naming.
OPTIONAL_MODULES = (
    ("psutil", "psutil", "the panel cannot show CPU, memory or SoC temperature"),
)

OK, WARN, FAIL = "ok", "warn", "fail"

_GLYPH = {OK: "+", WARN: "!", FAIL: "x"}
_COLOR = {OK: "\033[38;5;40m", WARN: "\033[38;5;214m", FAIL: "\033[38;5;196m"}
_RESET = "\033[0m"


@dataclass(frozen=True)
class Check:
    """One preflight result.

    ``fix`` is the whole point of the type. A check that reports "camera not
    found" and stops has told the person under the table nothing they did not
    already know; one that also says which command to run has.
    """

    name: str
    status: str
    detail: str
    fix: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _supports_color() -> bool:
    """Whether to emit ANSI colour.

    ``NO_COLOR`` is honoured because this output gets piped into journalctl and
    into ``systemctl status``, where escape codes are noise.
    """
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _bootstrap_path() -> None:
    """Put this directory first on ``sys.path``.

    Prepended rather than appended: if the machine happens to have anything else
    called ``app`` or ``vision`` installed -- and ``app`` is a common enough name
    that this is not paranoid -- the local package has to win, or the failure is
    an import that succeeds and gives you somebody else's module.
    """
    here = str(PACKAGE_DIR)
    if sys.path and sys.path[0] == here:
        return
    while here in sys.path:
        sys.path.remove(here)
    sys.path.insert(0, here)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_python() -> Check:
    """Interpreter version, checked before importing anything of ours.

    ``app/__init__.py`` gates this too, but it does so by raising during import
    -- which is correct there and unhelpful here, because it would come out as a
    traceback instead of as a line in this report.
    """
    from app import MIN_PYTHON  # noqa: PLC0415 - after _bootstrap_path()

    current = sys.version_info[:2]
    version = f"{current[0]}.{current[1]}.{sys.version_info.micro}"
    if current >= MIN_PYTHON:
        return Check("python", OK, f"{version} at {sys.executable}")
    return Check(
        "python",
        FAIL,
        f"{version}, but {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required",
        "Raspberry Pi OS Bookworm ships Python 3.11. Check you are inside the venv: "
        "source .venv/bin/activate",
    )


def check_dependencies() -> list[Check]:
    """Import each dependency by name.

    Importing rather than consulting pip metadata, because the failure mode this
    catches is an OpenCV that installed cleanly and cannot load its shared
    libraries -- extremely common on a Pi, invisible to ``pip list``, and
    reported here as the ImportError it is.
    """
    checks: list[Check] = []
    missing: list[str] = []

    for module, package in REQUIRED_MODULES:
        try:
            __import__(module)
        except ImportError as exc:
            missing.append(package)
            checks.append(
                Check(
                    f"import {module}",
                    FAIL,
                    str(exc),
                    f"pip install {package}",
                )
            )

    if not missing:
        names = ", ".join(name for name, _ in REQUIRED_MODULES)
        checks.append(Check("dependencies", OK, names))
    elif len(missing) > 1:
        # One instruction that fixes all of them beats N individual ones.
        checks.append(
            Check(
                "dependencies",
                FAIL,
                f"{len(missing)} missing",
                "pip install -r requirements.txt",
            )
        )

    for module, package, consequence in OPTIONAL_MODULES:
        try:
            __import__(module)
        except ImportError:
            checks.append(
                Check(f"import {module}", WARN, f"absent -- {consequence}", f"pip install {package}")
            )
    return checks


def check_config(config_path: Path | None) -> tuple[Check, object]:
    """Load and validate the YAML config. Returns the check and the settings.

    A malformed config is a hard failure rather than a fallback to defaults, and
    that is deliberate: silently running on default HSV ranges when someone has
    spent an evening tuning them against their own felt is far more confusing
    than refusing to start.
    """
    from app.config import DEFAULT_CONFIG_PATH, load_settings

    path = config_path or DEFAULT_CONFIG_PATH
    try:
        settings = load_settings(config_path)
    except Exception as exc:  # noqa: BLE001 - pydantic raises a family of these
        return (
            Check(
                "config",
                FAIL,
                f"{path.name} is invalid: {exc}",
                f"Fix the error above in {path}, or move it aside to boot on defaults.",
            ),
            None,
        )

    if not path.is_file():
        return (
            Check(
                "config",
                WARN,
                f"no {path.name}; running on built-in defaults",
                "Copy the shipped config.yaml here if you want to tune anything.",
            ),
            settings,
        )
    return (
        Check(
            "config",
            OK,
            f"{path.name}: {settings.table_preset} table, {settings.system.target_fps} FPS target, "
            f"{settings.render.theme} theme",
        ),
        settings,
    )


def check_data_dirs() -> Check:
    """Make sure the data directories exist and can be written to.

    Created here rather than left to fail later, because the two things that
    live in them -- the log file and the calibration -- are written at moments
    when nobody is watching a terminal, and a read-only SD card is a real
    failure that otherwise shows up hours later as a calibration that did not
    save.
    """
    from app.config import CALIBRATION_DIR, DATA_DIR, LOG_DIR

    try:
        for directory in (DATA_DIR, CALIBRATION_DIR, LOG_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        probe = LOG_DIR / ".write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(
            "data dirs",
            FAIL,
            f"cannot write to {DATA_DIR}: {exc}",
            "Check the filesystem is not read-only (a full or corrupted SD card does this) "
            f"and that the user owns {DATA_DIR}.",
        )
    return Check("data dirs", OK, str(DATA_DIR))


def check_calibration() -> Check:
    """Whether a solved projector calibration is on disk.

    A warning rather than a failure, but a loud one. Without it the projector
    falls back to an identity mapping, which does not look broken: it puts a
    perfectly plausible overlay onto the wrong part of the table, and the
    natural reading of that is that the *detection* is wrong. People lose
    evenings to this.
    """
    from projection.mapper import load_calibration

    calibration = load_calibration()
    if calibration is None:
        return Check(
            "calibration",
            WARN,
            "none saved -- overlays will not line up with the felt",
            "python -m calibration_ui.calibration_app",
        )
    if not calibration.is_calibrated:
        return Check(
            "calibration",
            WARN,
            "file exists but was never solved",
            "python -m calibration_ui.calibration_app",
        )
    quality = "good" if calibration.rmse_px <= 5 else "usable" if calibration.rmse_px <= 20 else "poor"
    status = OK if calibration.rmse_px <= 20 else WARN
    return Check(
        "calibration",
        status,
        f"{calibration.rmse_px:.1f} px RMSE ({quality})"
        + (f", solved {calibration.created_at[:10]}" if calibration.created_at else ""),
        "" if status == OK else "Re-run the wizard: python -m calibration_ui.calibration_app",
    )


def check_camera(settings) -> Check:
    """Report what the camera is likely to come up as, without opening it.

    Deliberately does not open the device. Opening it here and closing it again
    leaves a window in which some V4L2 drivers report the camera busy, so the
    check would occasionally cause the failure it is looking for. What it can do
    without touching the hardware is answer the question that is actually
    ambiguous: whether a *real* backend is available at all, because
    ``vision.camera`` falls back to the synthetic one rather than failing, and a
    mock camera that nobody noticed looks exactly like a detection bug.
    """
    if settings.camera.use_mock:
        return Check("camera", WARN, "mock forced in config -- frames will be synthetic")

    backends: list[str] = []
    try:
        __import__("picamera2")
        backends.append("picamera2")
    except ImportError:
        pass

    devices: list[str] = []
    if sys.platform.startswith("linux"):
        devices = sorted(str(p) for p in Path("/dev").glob("video*"))
        if devices:
            backends.append(f"v4l2 ({', '.join(devices)})")
    else:
        # No device nodes to enumerate off Linux; OpenCV indexes cameras itself.
        backends.append("opencv (device enumeration is platform-specific)")

    if backends:
        return Check("camera", OK, ", ".join(backends))
    return Check(
        "camera",
        WARN,
        "no camera backend found -- the loop will fall back to synthetic frames",
        "Check the ribbon cable is seated, then: libcamera-hello --list-cameras\n"
        "        Preview what the camera sees with: python -m tools.camera_preview",
    )


def check_display(settings) -> Check:
    """Whether there is somewhere to project.

    On the Pi this is the check that catches a headless boot: with no ``DISPLAY``
    the full-screen OpenCV window cannot open, the display layer falls back to
    discarding frames, and the system runs perfectly while projecting nothing.
    """
    if settings.projector.use_mock:
        return Check("display", WARN, "mock forced in config -- projector output is discarded")

    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return Check(
            "display",
            WARN,
            "no DISPLAY or WAYLAND_DISPLAY -- projector output will be discarded",
            "Run from the Pi's desktop session, or export DISPLAY=:0 first. "
            "A systemd unit needs Environment=DISPLAY=:0 in it.",
        )
    return Check(
        "display",
        OK,
        f"{settings.projector.width}x{settings.projector.height}"
        + (f" on {os.environ['DISPLAY']}" if os.environ.get("DISPLAY") else ""),
    )


def check_port(host: str, port: int) -> Check:
    """Whether the web port is free.

    Worth its own check because the failure it prevents is misleading: uvicorn
    reports "address already in use" and exits, and the usual cause is a
    previous instance of *this* application still running -- which means the
    projector is already lit and the camera already claimed, so every other
    symptom points somewhere else.
    """
    # 0.0.0.0 means "every interface"; bind-testing it is the correct check for
    # whether the port is free, but it is not a connectable address.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # No SO_REUSEADDR: it would let this bind succeed against a socket in
        # TIME_WAIT, which is exactly the case we want to know about.
        try:
            sock.bind((probe_host, port))
        except OSError as exc:
            return Check(
                "web port",
                FAIL,
                f"{probe_host}:{port} is already in use ({exc.strerror or exc})",
                f"Something is already listening -- most likely another copy of this app. "
                f"Stop it, or pick another port: --port {port + 1}",
            )
    return Check("web port", OK, f"{host}:{port} is free")


def lan_url(host: str, port: int) -> str | None:
    """The address to type into a phone, or ``None`` if it is not reachable.

    The route to a public address is asked for and then dropped without sending
    anything -- a UDP ``connect`` only fixes the socket's peer, so this is a
    routing-table lookup rather than traffic. It beats ``gethostbyname`` on a Pi,
    which frequently answers 127.0.1.1 from ``/etc/hosts``.
    """
    if host not in ("0.0.0.0", "::", ""):
        return None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.2)
            sock.connect(("8.8.8.8", 80))
            address = sock.getsockname()[0]
    except OSError:
        return None
    if not address or address.startswith("127."):
        return None
    return f"http://{address}:{port}/"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight(args: argparse.Namespace) -> tuple[list[Check], object]:
    """Run every check in dependency order. Returns the results and settings.

    Ordered so that each check can assume the previous ones passed -- there is
    no point validating the config before confirming pydantic imports -- and it
    stops at the first failure that would make the rest meaningless rather than
    printing a cascade of errors that all have one cause.
    """
    checks: list[Check] = []

    python_check = check_python()
    checks.append(python_check)
    if python_check.failed:
        return checks, None

    dependency_checks = check_dependencies()
    checks.extend(dependency_checks)
    if any(c.failed for c in dependency_checks):
        return checks, None

    config_check, settings = check_config(args.config)
    checks.append(config_check)
    if settings is None:
        return checks, None

    if args.mock:
        # --mock forces both, and it does so after config load, so reflect it
        # here or the report contradicts what is about to happen.
        settings.camera.use_mock = True
        settings.projector.use_mock = True

    checks.append(check_data_dirs())

    # Below here the checks are about hardware, and which of them apply depends
    # on what is actually going to run.
    if not args.no_loop:
        checks.append(check_camera(settings))
        if not args.mock:
            checks.append(check_calibration())
        checks.append(check_display(settings))

    if not args.headless:
        host = args.host or settings.web.host
        port = args.port or settings.web.port
        checks.append(check_port(host, port))

    return checks, settings


def print_report(checks: list[Check]) -> None:
    """Print the preflight results, aligned, with fixes indented underneath."""
    color = _supports_color()
    width = max((len(c.name) for c in checks), default=0)

    print()
    print("  Preflight")
    print("  " + "-" * (width + 40))
    for check in checks:
        glyph = _GLYPH[check.status]
        if color:
            glyph = f"{_COLOR[check.status]}{glyph}{_RESET}"
        print(f"  {glyph} {check.name.ljust(width)}  {check.detail}")
        if check.fix:
            for line in check.fix.splitlines():
                print(f"      {' ' * width}  -> {line}")
    print()
    # Explicit, because stdout is block-buffered whenever it is not a terminal.
    # Under systemd or a pipe -- which is most of the time on the Pi -- the
    # report would otherwise surface long after the log lines it is supposed to
    # precede, interleaved into the middle of them.
    sys.stdout.flush()


def print_ready(args: argparse.Namespace, settings) -> None:
    """Say what is about to start, and where to reach it."""
    if args.headless:
        print("  Starting the vision loop. No web server (--headless).")
        print("  Ctrl-C to stop.\n")
        sys.stdout.flush()
        return

    host = args.host or settings.web.host
    port = args.port or settings.web.port
    print(f"  Control panel:  http://localhost:{port}/")
    url = lan_url(host, port)
    if url:
        print(f"  From a phone:   {url}")
    if args.no_loop:
        print("  Web panel only. No camera, no projection (--no-loop).")
    print("  Ctrl-C to stop.\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_launcher_parser() -> argparse.ArgumentParser:
    """The application's flags, plus the launcher's own.

    The application's half is imported rather than redeclared, so a flag added
    to ``app.main`` works here on the same day without anyone remembering to
    mirror it.
    """
    from app.main import build_parser

    parser = argparse.ArgumentParser(
        prog="launcher.py",
        description="Start GhostBall, with a preflight check first.",
        parents=[build_parser(add_help=False)],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python launcher.py                 preflight, then start everything\n"
            "  python launcher.py --mock          no hardware needed\n"
            "  python launcher.py --check         preflight only, then exit\n"
            "  python launcher.py --headless      vision loop only, no web server\n"
        ),
    )
    launcher = parser.add_argument_group("launcher options")
    launcher.add_argument(
        "--check",
        action="store_true",
        help="run the preflight checks and exit without starting anything",
    )
    launcher.add_argument(
        "--force",
        action="store_true",
        help="start even if a preflight check failed",
    )
    launcher.add_argument(
        "--skip-checks",
        action="store_true",
        help="start immediately, without running the preflight checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Preflight, then hand over to the application."""
    _bootstrap_path()

    parser = build_launcher_parser()
    args = parser.parse_args(argv)

    if args.check and args.skip_checks:
        parser.error("--check and --skip-checks ask for opposite things")

    from app.main import run

    if args.skip_checks:
        return run(args)

    print("\n  GhostBall")
    checks, settings = preflight(args)
    print_report(checks)

    failures = [c for c in checks if c.failed]
    if failures:
        names = ", ".join(c.name for c in failures)
        if not args.force:
            print(f"  Not starting: {len(failures)} check(s) failed ({names}).")
            print("  Fix the items marked 'x' above, or re-run with --force.\n")
            sys.stdout.flush()
            return 1
        print(f"  {len(failures)} check(s) failed ({names}); starting anyway (--force).\n")

    if args.check:
        print("  Preflight only (--check). Nothing was started.\n")
        return 0

    print_ready(args, settings)
    # run() re-reads the config itself. That is one redundant YAML parse at
    # startup, and it is worth it: the application stays runnable without this
    # launcher, which matters because `python -m app.main` is what the tests and
    # the docs use.
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
