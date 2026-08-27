"""Projector-assisted focus calibration, from a terminal.

Run this before there is any wizard around it, and before there is a table:
pattern on the floor, camera on a box, report to stdout. The point of having it
standalone is that blob detection and the exposure lock can be proved against
real projected light without four screens of UI in the way.

What it does, in order::

    project the five checkerboards
    find them in the camera image        <- fails fast if the projector is off
    lock exposure, gain and white balance
    sweep focus, measuring inside each target only
    check the lock held
    report per-target peaks, tilt, and a verdict
    blank the projector and take a bare-cloth reference
    write data/calibration/focus.json

Usage::

    python -m tools.focus_calibrate                # full run, saves
    python -m tools.focus_calibrate --dry-run      # report only, writes nothing
    python -m tools.focus_calibrate --step 32      # finer sweep, slower
    python -m tools.focus_calibrate --detect-only  # just find the targets and stop

**Focus the projector first**, with its own focus ring, before running this. The
camera cannot resolve detail the projector never put on the surface -- a blurry
projected checkerboard produces a real, confident, wrong answer, because the
sweep will happily find the lens position that best resolves a blurry pattern.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CameraSettings, load_settings  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from vision.focus import (  # noqa: E402
    FocusCalibration,
    FocusError,
    apply_focus,
    find_lens_subdev,
    query_focus_range,
    save_focus_calibration,
)
from vision.focus_calibration import (  # noqa: E402
    DEFAULT_TILT_THRESHOLD,
    analyse,
    bare_reference,
    coarse_step,
    detect_targets,
    focus_positions,
    sweep_focus,
)

logger = logging.getLogger("focus_calibrate")


def say(message: str = "") -> None:
    """Print and flush.

    Flushed because this is watched from an SSH session while standing at a
    table, and block-buffered progress that arrives in one lump at the end is
    indistinguishable from a hang.
    """
    print(message, flush=True)


def project_targets(display, canvas) -> None:
    """Push the target pattern. Must be repeated -- see below."""
    display.send_frame(canvas)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Projector-assisted focus calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Focus the PROJECTOR first, using its own focus ring. The camera cannot\n"
            "resolve detail the projector never put on the surface, and a blurry\n"
            "pattern yields a confident wrong answer rather than an obvious failure."
        ),
    )
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    parser.add_argument(
        "--step", type=int, default=None,
        help="coarse focus step (default: derived from the lens range, ~32 stops)",
    )
    parser.add_argument("--start", type=int, default=None, help="first focus value")
    parser.add_argument("--end", type=int, default=None, help="last focus value")
    parser.add_argument(
        "--settle", type=float, default=0.35, help="seconds for the motor to settle per stop"
    )
    parser.add_argument(
        "--frames", type=int, default=3, help="frames discarded, then measured, per stop"
    )
    parser.add_argument(
        "--tilt-threshold",
        type=int,
        default=DEFAULT_TILT_THRESHOLD,
        help=f"focus-count spread that counts as a tilted mount (default {DEFAULT_TILT_THRESHOLD})",
    )
    parser.add_argument(
        "--expect", type=int, default=5, help="how many targets to expect (default 5)"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="find the targets, report where they are, and stop",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the result without writing focus.json"
    )
    parser.add_argument(
        "--save-frame", type=Path, default=None, help="write the detection frame here, for a look"
    )
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    setup_logging(args.log_level, log_to_file=False)

    # -- lens ---------------------------------------------------------------
    lens = find_lens_subdev(settings.camera.lens_driver)
    if lens is None:
        say(f"\n  No focus motor matching {settings.camera.lens_driver!r}.")
        say("  This needs the real camera; there is nothing to calibrate on a dev box.\n")
        return 2
    try:
        focus_range = query_focus_range(lens.path)
    except FocusError as exc:
        say(f"\n  {exc}\n")
        return 2

    # -- hardware -----------------------------------------------------------
    from projection.display import Display
    from projection.patterns import TestPattern, render_test_pattern
    from vision.camera import Camera, CameraError

    camera_settings = CameraSettings(**{**settings.camera.model_dump(), "use_mock": False})
    try:
        camera = Camera(camera_settings).open()
        display = Display(settings.projector).open()
    except (CameraError, RuntimeError) as exc:
        say(f"\n  Cannot open the hardware: {exc}\n")
        return 2
    if camera.is_mock:
        # A synthetic frame has nothing to do with the lens, and a confident
        # wrong number is worse than refusing.
        say("\n  The camera fell back to the mock backend; there is no lens to sweep.\n")
        camera.close()
        display.close()
        return 2

    try:
        return _run(args, settings, camera, display, lens, focus_range,
                    render_test_pattern, TestPattern)
    finally:
        display.clear()
        display.close()
        camera.close()


def _run(args, settings, camera, display, lens, focus_range, render_test_pattern, TestPattern):
    say("\n  Projector-assisted focus calibration")
    say(f"  Lens:  {lens.name} at {lens.path}, range {focus_range.minimum}-{focus_range.maximum}")
    say("  Focus the PROJECTOR by hand first -- this cannot resolve what it never drew.\n")

    canvas = render_test_pattern(TestPattern.FOCUS_TARGETS, settings=settings)

    # -- find the targets ---------------------------------------------------
    # Before anything expensive. "The projector is off or misaimed" is the
    # commonest failure and it is answerable in two seconds; discovering it as a
    # flat curve after a two-minute sweep would be its own small tragedy.
    say("  Looking for the targets...")
    display.send_frame(canvas)
    for _ in range(15):  # let AE settle on the projected pattern
        camera.capture_frame()

    frame = camera.capture_frame()
    if frame is None:
        say("  No frames from the camera.\n")
        return 2

    regions = detect_targets(frame.image, expected=args.expect)
    if args.save_frame:
        import cv2

        annotated = frame.image.copy()
        for region in regions:
            x0, y0, x1, y1 = region.box
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                annotated, region.name, (x0, max(14, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_frame), annotated)
        say(f"  Wrote {args.save_frame}")

    if len(regions) < args.expect:
        say(f"\n  Found {len(regions)} of {args.expect} targets.")
        say("  Is the projector on, awake and aimed at the surface? The five")
        say("  checkerboards should be plainly visible before this runs.")
        say("  Re-run with --save-frame shot.png to see what the camera sees.\n")
        return 1

    say(f"  Found {len(regions)}:")
    for region in regions:
        cx, cy = region.center_px
        say(f"    {region.name:<14} at ({cx:>6.0f}, {cy:>6.0f})  area {region.area_px} px")
    if args.detect_only:
        say("\n  --detect-only; stopping here.\n")
        return 0

    # -- sweep --------------------------------------------------------------
    step = coarse_step(focus_range) if args.step is None else args.step
    positions = focus_positions(focus_range, step, args.start, args.end)
    say(f"\n  Sweeping {len(positions)} positions, {positions[0]}-{positions[-1]}...")

    def on_step(index, total, position, measured):
        # The projector window stops repainting if it is not fed, so the pattern
        # is re-sent every step rather than once at the start.
        display.send_frame(canvas)
        best = max(measured.values()) if measured else 0.0
        bar = "#" * int(min(30, best / 40))
        say(f"    [{index:>3}/{total}] {position:>5}  {best:>9.1f}  {bar}")

    with camera.exposure_lock() as exposure:
        if not exposure.locked:
            # Not fatal on its own -- some backends have no ISP controls -- but
            # it must be said, because the result is only as trustworthy as the
            # exposure was steady.
            say(f"\n  WARNING: exposure is NOT locked ({exposure.detail}).")
            say("  Auto-exposure moves with focus, so the peak may be wrong.\n")
        else:
            say(f"  Exposure locked: {exposure.detail}\n")

        outcome = sweep_focus(
            camera,
            lens.path,
            regions,
            positions,
            focus_range,
            settle_seconds=args.settle,
            frames=args.frames,
            exposure_status=exposure,
            on_step=on_step,
        )
        outcome = analyse(outcome, focus_range, tilt_threshold=args.tilt_threshold)

    # -- report -------------------------------------------------------------
    say("\n  Per-target peaks:")
    if outcome.peaks:
        for peak in outcome.peaks:
            say(
                f"    {peak.name:<14} best {peak.peak_focus:>5}   "
                f"sharpness {peak.peak_sharpness:>9.1f}   "
                f"peak/median {peak.prominence:>5.2f}x"
            )
        say(f"    spread across targets: {outcome.tilt_spread} focus counts")
    else:
        say("    none")

    if outcome.diagnosis is not None:
        marker = "REFUSED" if outcome.diagnosis.fatal else "WARNING"
        say(f"\n  {marker}: {outcome.diagnosis.message}")
    if not outcome.ok:
        say("\n  No value written. A confident wrong number is worse than no number.\n")
        return 1

    say(f"\n  Best focus: focus_absolute={outcome.best_focus}")

    # -- bare reference -----------------------------------------------------
    # With the targets off, at the chosen focus. This is what the runtime health
    # check compares against; the peak above was measured on high-contrast
    # projected checkerboards and is an order of magnitude larger than anything
    # bare cloth produces, so comparing against it would fire on every boot.
    status = apply_focus(outcome.best_focus, device=lens.path)
    display.clear()
    time.sleep(0.4)
    for _ in range(10):
        camera.capture_frame()
    bare = bare_reference(camera, regions, settings=settings)
    say(f"  Bare-surface reference at that focus: {bare:.1f}")

    if args.dry_run:
        say("\n  --dry-run; nothing written.\n")
        return 0

    calibration = FocusCalibration(
        focus_absolute=outcome.best_focus,
        peak_sharpness=outcome.best_sharpness,
        bare_table_sharpness=bare,
        per_target=tuple(outcome.peaks),
        tilt_spread=outcome.tilt_spread,
        tilt_note=outcome.tilt_note,
        camera_resolution=f"{settings.camera.width}x{settings.camera.height}",
        lens_name=lens.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    path = save_focus_calibration(calibration)
    say(f"  Saved to {path}")
    say(f"  Lens left at {status.actual} ({'confirmed' if status.ok else status.detail})")
    say("  This is applied automatically on every startup.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
