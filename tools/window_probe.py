"""Find out what the projector window actually is, and when it becomes it.

Written because a warning in this codebase reported the projector window as
1920x548 on a 1920x1080 desktop -- a number corresponding to no mode, no half,
and nothing else on the machine. Two bugs in that warning explain it:
``getWindowImageRect`` reports the *image* area rather than the window, and it
was read immediately after the fullscreen request, which the window manager
answers asynchronously. So the number was an image area that did not exist yet,
measured mid-map.

Rather than guess again, this measures. It opens the projector window exactly
the way the app does, then watches it over the first second, and cross-checks
OpenCV's idea of the geometry against the X server's -- which are different
questions, and the difference is the whole answer.

    python -m tools.window_probe
    python -m tools.window_probe --seconds 5 --mode borderless

Run it over SSH with ``DISPLAY=:0`` set, or from a terminal on the desktop.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import load_settings  # noqa: E402
from projection.display import WINDOW_NAME, _gui_backend  # noqa: E402


def run(command: list[str]) -> str:
    """A command's output, or a note that it is not installed."""
    if shutil.which(command[0]) is None:
        return f"({command[0]} not installed)"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            # Explicit: text=True alone decodes with the locale encoding.
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"({command[0]} failed: {exc})"
    return (result.stdout or result.stderr).strip()


def environment() -> None:
    print("  Session")
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "QT_QPA_PLATFORM"):
        print(f"    {name:<18} {os.environ.get(name, '(unset)')}")
    print(f"    OpenCV GUI         {_gui_backend()}")
    print()

    print("  Outputs (xrandr)")
    for line in run(["xrandr", "--listmonitors"]).splitlines():
        print(f"    {line}")
    print()


def x_geometry() -> str:
    """The window as the X server sees it. The authoritative answer."""
    if shutil.which("xdotool"):
        ids = run(["xdotool", "search", "--name", WINDOW_NAME])
        window = ids.splitlines()[-1] if ids and not ids.startswith("(") else None
        if window:
            geometry = run(["xdotool", "getwindowgeometry", window])
            state = run(["xprop", "-id", window, "_NET_WM_STATE"])
            return f"{geometry}\n      {state}"
    if shutil.which("xwininfo"):
        return run(["xwininfo", "-name", WINDOW_NAME])
    return "(install xdotool or x11-utils to read the real window geometry)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the projector window.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seconds", type=float, default=3.0, help="how long to watch")
    parser.add_argument(
        "--mode",
        choices=["property", "borderless", "none"],
        default="property",
        help=(
            "property: WND_PROP_FULLSCREEN (what the app does). "
            "borderless: resize and move to 0,0 without asking for fullscreen. "
            "none: leave the window as created."
        ),
    )
    args = parser.parse_args(argv)

    import cv2

    settings = load_settings(args.config).projector
    print(f"\n  Requested: {settings.width}x{settings.height}\n")
    environment()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    frame = np.zeros((settings.height, settings.width, 3), dtype=np.uint8)
    # A visible border, so what is on the wall can be compared with the numbers.
    cv2.rectangle(frame, (2, 2), (settings.width - 3, settings.height - 3), (0, 255, 0), 4)
    cv2.line(frame, (0, 0), (settings.width, settings.height), (0, 90, 0), 2)
    cv2.line(frame, (settings.width, 0), (0, settings.height), (0, 90, 0), 2)

    if args.mode == "property":
        cv2.resizeWindow(WINDOW_NAME, settings.width, settings.height)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif args.mode == "borderless":
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.resizeWindow(WINDOW_NAME, settings.width, settings.height)

    print("  OpenCV image area over time (this is the DRAWING area, not the window):")
    print(f"    {'t':>7}  {'image area':>14}   note")
    started = time.perf_counter()
    shown = False
    while time.perf_counter() - started < args.seconds:
        elapsed = time.perf_counter() - started
        note = ""
        if elapsed > 0.25 and not shown:
            cv2.imshow(WINDOW_NAME, frame)
            shown = True
            note = "<- first frame shown"
        cv2.waitKey(30)
        try:
            rect = cv2.getWindowImageRect(WINDOW_NAME)
        except cv2.error:
            rect = None
        area = f"{rect[2]}x{rect[3]}" if rect else "(unavailable)"
        print(f"    {elapsed:>6.2f}s  {area:>14}   {note}")
        time.sleep(0.2)

    print("\n  The window as the X server sees it:")
    for line in x_geometry().splitlines():
        print(f"      {line}")

    print(
        "\n  Read it like this:\n"
        "    - If X reports 1920x1080 and _NET_WM_STATE has _FULLSCREEN, the window is\n"
        "      fine and any smaller OpenCV number is toolkit chrome (a Qt toolbar) or\n"
        "      an aspect-fit inside it.\n"
        "    - If X reports something smaller, fullscreen was genuinely refused; try\n"
        "      --mode borderless, and compare.\n"
        "    - If the number changes across the rows above, it was still settling and\n"
        "      any single reading taken at open time is meaningless.\n"
    )
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
