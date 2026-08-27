"""Find the right ``focus_absolute`` for the camera where it is actually mounted.

There is no correct value in the abstract -- it depends on how far the lens ends
up from the cloth, which is only known once the thing is bolted to the ceiling.
So this steps the lens across its range, measures how sharp the picture is at
each stop, and tells you the number to put in ``config.yaml``.

Sharpness is the variance of the Laplacian: the second spatial derivative
responds to edges, an out-of-focus image has none, and the variance across the
frame collapses toward zero. It is the standard autofocus metric and it is
scale-free enough to compare between two stops of the same scene, which is all
that is asked of it here. It is *not* comparable between scenes -- a busy table
scores higher than an empty one at the same focus -- so the absolute numbers
below mean nothing on their own. Only the peak does.

Two things this gets right that a naive sweep does not:

* **It waits for the motor.** A VCM takes a moment to settle and the first frames
  after a step are mid-travel. Stepping and immediately grabbing measures the
  previous position, which produces a curve shifted by one stop.
* **It discards frames after the step**, for the same reason plus the ISP's own
  pipeline depth -- a frame handed back right after a control change was already
  in flight before it.

Usage::

    python -m tools.focus_sweep                 # coarse sweep, then refine
    python -m tools.focus_sweep --step 32       # finer, slower
    python -m tools.focus_sweep --no-refine     # single pass
    python -m tools.focus_sweep --roi 0.4       # measure the middle 40%
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import CameraSettings, load_settings  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from vision.focus import (  # noqa: E402
    FocusCalibration,
    FocusError,
    FocusRange,
    apply_focus,
    approach_focus,
    find_lens_subdev,
    query_focus_range,
    read_focus,
    resolve_focus_value,
    save_focus_calibration,
)
from vision.focus_calibration import coarse_step  # noqa: E402

logger = logging.getLogger("focus_sweep")


@dataclass(slots=True)
class Sample:
    """One focus stop and how sharp the picture was there."""

    position: int
    sharpness: float
    #: What the lens reported after being told to go to ``position``. A drift
    #: here means the motor is not tracking, and every sample after it is
    #: measuring something other than what the table says.
    readback: int


def sharpness(image: np.ndarray, roi_fraction: float) -> float:
    """Variance of the Laplacian over a centred region of the frame.

    Centred and cropped on purpose. The edges of an overhead frame are cushion,
    floor and whatever else is in the room, all at a different distance from the
    lens than the cloth -- include them and the sweep optimises focus for the
    carpet.
    """
    import cv2

    height, width = image.shape[:2]
    half = max(0.05, min(1.0, roi_fraction)) / 2.0
    y0, y1 = int(height * (0.5 - half)), int(height * (0.5 + half))
    x0, x1 = int(width * (0.5 - half)), int(width * (0.5 + half))
    patch = image[y0:y1, x0:x1]

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_at(
    camera,
    device: Path,
    position: int,
    settle_seconds: float,
    frames: int,
    roi_fraction: float,
    focus_range: FocusRange,
) -> Sample:
    """Drive the lens to ``position`` and score the picture there."""
    # Always from below. A voice coil lands in a slightly different place
    # depending on which way it travelled, and the sweep has to arrive the same
    # way startup does or the calibrated value is soft at boot for no visible
    # reason. See vision.focus.approach_focus.
    approach_focus(device, position, focus_range)
    time.sleep(settle_seconds)

    # Drop the first frames: they were already in the ISP pipeline before the
    # control changed, so they show the previous focus.
    for _ in range(frames):
        camera.capture_frame()

    scores = []
    for _ in range(frames):
        frame = camera.capture_frame()
        if frame is not None:
            scores.append(sharpness(frame.image, roi_fraction))

    if not scores:
        raise FocusError(f"no frames captured at focus_absolute={position}")
    # Median, not mean: one frame caught mid-exposure-change is a big outlier
    # and would move a mean enough to pick the wrong stop.
    return Sample(
        position=position, sharpness=float(np.median(scores)), readback=read_focus(device)
    )


def sweep(
    camera,
    device: Path,
    positions: list[int],
    settle_seconds: float,
    frames: int,
    roi_fraction: float,
    focus_range: FocusRange,
) -> list[Sample]:
    """Measure every position in turn, printing as it goes."""
    samples: list[Sample] = []
    peak = 0.0
    for index, position in enumerate(positions, start=1):
        sample = measure_at(
            camera, device, position, settle_seconds, frames, roi_fraction, focus_range
        )
        samples.append(sample)
        peak = max(peak, sample.sharpness)

        # A bar as it runs, because this takes a couple of minutes and the shape
        # of the curve is the thing you want to see while standing on a ladder.
        bar = "#" * int(40 * sample.sharpness / peak) if peak > 0 else ""
        drift = "" if sample.readback == position else f"  READBACK {sample.readback}"
        print(
            f"  [{index:>3}/{len(positions)}] {position:>5}  {sample.sharpness:>10.1f}  {bar}{drift}",
            flush=True,
        )
    return samples


def best_of(samples: list[Sample]) -> Sample:
    return max(samples, key=lambda s: s.sharpness)


def refine_range(samples: list[Sample], span: int) -> tuple[int, int]:
    """A window around the coarse peak, for the second pass.

    Bracketed by the *neighbours* of the peak rather than a fixed span around
    it, so a peak that sits between two coarse stops is still inside the window.
    """
    ordered = sorted(samples, key=lambda s: s.position)
    index = ordered.index(best_of(samples))
    low = ordered[max(0, index - 1)].position
    high = ordered[min(len(ordered) - 1, index + 1)].position
    if low == high:
        low, high = low - span, high + span
    return low, high


def is_peak_credible(samples: list[Sample]) -> tuple[bool, str]:
    """Whether the curve actually has a peak, or is just noise.

    Worth checking before printing a confident number. A lens pointed at a blank
    wall, or one that is not moving at all, produces a flat curve whose maximum
    is whichever sample got the most sensor noise -- and reporting that as "the
    best focus" would send someone to bolt the camera at the wrong height.
    """
    scores = [s.sharpness for s in samples]
    if not scores or max(scores) <= 0:
        return False, "no detail found at any focus position"

    spread = max(scores) / max(1e-9, float(np.median(scores)))
    if spread < 1.5:
        return False, (
            f"the sharpness curve is nearly flat (peak is only {spread:.2f}x the median). "
            "Either the lens is not moving, or there is nothing with edges in the "
            "middle of the frame -- point it at the table with the balls racked."
        )

    drifted = [s for s in samples if s.readback != s.position]
    if drifted:
        return False, (
            f"{len(drifted)} of {len(samples)} positions did not read back the value "
            "they were set to; the lens is not tracking. Check the camera ribbon."
        )
    return True, ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep lens focus and report the sharpest position.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Point the camera at the table, racked, with the room lit the way it will "
            "be during play, then run this once. Put the reported number in\n"
            "config.yaml under camera.focus_absolute."
        ),
    )
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    parser.add_argument("--start", type=int, default=None, help="first focus value")
    parser.add_argument("--end", type=int, default=None, help="last focus value")
    parser.add_argument(
        "--step", type=int, default=None,
        help="coarse step size (default: derived from the lens range, ~32 stops)",
    )
    parser.add_argument(
        "--settle", type=float, default=0.35, help="seconds to let the motor settle (default 0.35)"
    )
    parser.add_argument(
        "--frames", type=int, default=3, help="frames to discard and then measure per stop"
    )
    parser.add_argument(
        "--roi",
        type=float,
        default=0.5,
        help="fraction of the frame to measure, centred (default 0.5)",
    )
    parser.add_argument(
        "--no-refine", action="store_true", help="skip the fine pass around the peak"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="return the lens to its saved position afterwards instead of the best one",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="report the best value without writing data/calibration/focus.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    setup_logging(settings.system.log_level, log_to_file=False)

    lens = find_lens_subdev(settings.camera.lens_driver)
    if lens is None:
        logger.error(
            "no focus motor matching %r found. This tool needs the real camera; "
            "there is nothing to sweep on a dev box.",
            settings.camera.lens_driver,
        )
        return 2

    try:
        focus_range = query_focus_range(lens.path)
    except FocusError as exc:
        logger.error("%s", exc)
        return 2

    start = focus_range.minimum if args.start is None else focus_range.clamp(args.start)
    end = focus_range.maximum if args.end is None else focus_range.clamp(args.end)
    if start > end:
        start, end = end, start

    # Derived from the range the driver reported unless overridden. A fixed
    # stride means a different sweep resolution on every lens.
    step = coarse_step(focus_range) if args.step is None else max(1, args.step)

    print(f"\n  Lens:  {lens.name} at {lens.path}")
    print(f"  Range: {focus_range.minimum}-{focus_range.maximum} step {focus_range.step}")
    print(f"  Sweep: step {step}{'' if args.step is not None else ' (derived from the range)'}")
    print(f"  ROI:   centre {args.roi:.0%} of the frame\n")

    from vision.camera import Camera, CameraError

    # The mock camera would produce a synthetic image whose sharpness has
    # nothing to do with the lens, and a confident wrong answer is worse than
    # refusing -- so this insists on real hardware.
    camera_settings = CameraSettings(**{**settings.camera.model_dump(), "use_mock": False})
    try:
        camera = Camera(camera_settings).open()
    except CameraError as exc:
        logger.error("cannot open the camera: %s", exc)
        return 2
    if camera.is_mock:
        logger.error("the camera fell back to the mock backend; there is no lens to sweep")
        camera.close()
        return 2

    try:
        coarse_positions = list(range(start, end + 1, max(1, step)))
        if coarse_positions[-1] != end:
            coarse_positions.append(end)

        print(f"  Coarse sweep, {len(coarse_positions)} positions:")
        samples = sweep(
            camera, lens.path, coarse_positions, args.settle, args.frames, args.roi, focus_range
        )

        if not args.no_refine and len(samples) > 2:
            low, high = refine_range(samples, step)
            fine_step = max(focus_range.step, 1, step // 8)
            fine_positions = [
                p for p in range(low, high + 1, fine_step) if p not in {s.position for s in samples}
            ]
            if fine_positions:
                print(f"\n  Fine sweep around {low}-{high}, step {fine_step}:")
                samples.extend(
                    sweep(
                        camera, lens.path, fine_positions, args.settle, args.frames,
                        args.roi, focus_range,
                    )
                )

        best = best_of(samples)
        credible, complaint = is_peak_credible(samples)

        print()
        if not credible:
            # Still print the number, but do not dress it up as an answer.
            print(f"  INCONCLUSIVE: {complaint}")
            print(f"  (the highest score was at focus_absolute={best.position})\n")
            return 1

        print(f"  Best focus: focus_absolute={best.position}  (sharpness {best.sharpness:.1f})")

        target = best.position
        if args.restore:
            saved, _ = resolve_focus_value(settings.camera)
            target = best.position if saved is None else saved
        status = apply_focus(target, device=lens.path)

        if args.save and not args.restore:
            # peak and bare-table sharpness are the same measurement here,
            # and that is correct: this tool sweeps whatever is in front of
            # the camera with no projected targets, so what it measured *is*
            # the bare table. The two only diverge in the projector-assisted
            # flow, where the peak is taken on high-contrast targets and
            # sits an order of magnitude above anything the runtime health
            # check could ever see.
            calibration = FocusCalibration(
                focus_absolute=best.position,
                peak_sharpness=best.sharpness,
                bare_table_sharpness=best.sharpness,
                camera_resolution=f"{settings.camera.width}x{settings.camera.height}",
                lens_name=lens.name,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            path = save_focus_calibration(calibration)
            print(f"  Saved to {path}")
            print("  Applied automatically on every startup.\n")
        else:
            print("  Not saved. To use this value, put it in config.yaml:\n")
            print("    camera:")
            print(f"      focus_absolute: {best.position}\n")

        print(f"  Lens left at {status.actual} ({'confirmed' if status.ok else status.detail})\n")
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
