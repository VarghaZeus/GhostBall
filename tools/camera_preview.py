"""Live camera preview and felt-threshold tuning tool.

The first thing to run on new hardware, and the answer to two questions Phase 2
cannot start without:

1. **Is the camera working, in focus, and holding 30 FPS?** Run with no flags.
2. **What are the right felt HSV thresholds for this table and this room?**
   Run with ``--mask``. Table detection is the next stage to implement and it
   segments the felt by colour, so these numbers are its single most important
   input. Guessing them and debugging detection afterwards is much slower than
   tuning them here against a live image.

Works headless. A Pi under a pool table is usually reached over SSH with no
display, so ``--headless`` writes periodic JPEGs instead of opening a window and
``--report`` prints timing with no imagery at all.

Usage::

    python -m tools.camera_preview                  # live window with HUD
    python -m tools.camera_preview --mask           # felt threshold tuning
    python -m tools.camera_preview --headless       # write JPEGs, no window
    python -m tools.camera_preview --report --seconds 10   # timing only
    python -m tools.camera_preview --mock           # no hardware needed

Keys (windowed mode)::

    q / Esc   quit
    s         save a snapshot
    h         toggle the HUD
    m         toggle the felt mask view
    [ / ]     widen / narrow the felt hue range
    - / =     lower / raise the saturation floor
    f / g     nudge manual focus nearer / further (picamera2 only)
    p         print the current thresholds as YAML, ready to paste into config
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config import DATA_DIR, Settings, load_settings
from utils.logging import setup_logging
from utils.performance import PerformanceTracker

logger = logging.getLogger("camera_preview")

SNAPSHOT_DIR = DATA_DIR / "snapshots"

#: How often to write a frame in headless mode, in seconds. Frequent enough to
#: watch progress over SSH, rare enough not to thrash the SD card.
HEADLESS_WRITE_INTERVAL = 2.0


class ThresholdTuner:
    """Mutable felt HSV thresholds, adjustable from the keyboard.

    Starts from whatever is in ``config.yaml`` so a tuning session picks up
    where the last one left off, rather than from hardcoded defaults.
    """

    def __init__(self, settings: Settings) -> None:
        low, high = settings.vision.felt_hue_range
        self.hue_low = int(low)
        self.hue_high = int(high)
        self.sat_min = int(settings.vision.felt_sat_min)
        self.val_min = int(settings.vision.felt_val_min)

    def widen_hue(self, delta: int) -> None:
        """Expand or contract the hue band symmetrically.

        Clamped to OpenCV's 0-179 hue scale -- not 0-359, which is the usual
        source of confusion when copying HSV values from a colour picker.
        """
        self.hue_low = max(0, min(self.hue_low - delta, 179))
        self.hue_high = max(0, min(self.hue_high + delta, 179))
        if self.hue_low > self.hue_high:
            self.hue_low = self.hue_high

    def adjust_saturation(self, delta: int) -> None:
        self.sat_min = max(0, min(self.sat_min + delta, 255))

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Build the felt mask the way table detection will.

        Kept deliberately identical in spirit to what ``detect_table_boundaries``
        should do -- threshold, then close to bridge the holes punched by balls
        and the cue, then open to drop specular speckle. Tuning against a
        different pipeline than the one that will consume the numbers would make
        the numbers wrong.
        """
        import cv2

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([self.hue_low, self.sat_min, self.val_min], dtype=np.uint8),
            np.array([self.hue_high, 255, 255], dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def coverage_pct(self, mask: np.ndarray) -> float:
        """Fraction of the frame the mask claims.

        The number to tune by. An overhead shot framed on the table should land
        somewhere around 55-80%: far below and the mask is missing felt, far
        above and it has leaked onto a green wall, carpet or the floor.
        """
        return 100.0 * float(np.count_nonzero(mask)) / mask.size

    def largest_contour_bounds(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        """Bounding box of the largest masked region, or ``None`` if empty.

        A proxy for what table detection will find. If this box does not hug the
        table, detection will not either, however good the coverage number looks.
        """
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return cv2.boundingRect(max(contours, key=cv2.contourArea))

    def as_yaml(self) -> str:
        """The current thresholds, formatted to paste straight into config.yaml."""
        return (
            "vision:\n"
            f"  felt_hue_range: [{self.hue_low}, {self.hue_high}]\n"
            f"  felt_sat_min: {self.sat_min}\n"
            f"  felt_val_min: {self.val_min}"
        )


def draw_hud(
    frame: np.ndarray,
    tracker: PerformanceTracker,
    backend: str,
    focus: float | None,
    tuner: ThresholdTuner | None,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Overlay diagnostics onto a copy of the frame.

    A copy, not in place: the same frame is also written to snapshots, and
    baking the HUD into a saved image makes it useless for offline threshold
    work.
    """
    import cv2

    canvas = frame.copy()
    snap = tracker.snapshot()

    lines = [
        f"backend={backend}  {frame.shape[1]}x{frame.shape[0]}",
        f"fps={snap.fps:.1f}  frame={snap.frame_ms_avg:.1f}ms  p95={snap.frame_ms_p95:.1f}ms",
        f"frames={snap.total_frames}  slow={snap.dropped_frames}",
    ]
    if focus is not None:
        lines.append(f"lens_position={focus:.2f}")
    if tuner is not None and mask is not None:
        coverage = tuner.coverage_pct(mask)
        lines.append(
            f"felt hue=[{tuner.hue_low},{tuner.hue_high}] "
            f"sat>={tuner.sat_min} val>={tuner.val_min}"
        )
        lines.append(f"mask coverage={coverage:.1f}%  (aim 55-80% on an overhead shot)")

    # Dark plate behind the text: white-on-felt is unreadable, and this is being
    # read on a phone over VNC as often as on a monitor. Size the plate to the
    # longest line rather than a fixed width, or long lines spill onto the felt
    # and become illegible.
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    text_widths = [cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines]
    plate_w = min(canvas.shape[1], max(text_widths) + 24)
    plate_h = 24 * len(lines) + 16
    cv2.rectangle(canvas, (0, 0), (plate_w, plate_h), (0, 0, 0), thickness=-1)
    for i, line in enumerate(lines):
        cv2.putText(
            canvas, line, (12, 28 + i * 24), font, scale, (60, 255, 120), thickness, cv2.LINE_AA
        )

    if tuner is not None and mask is not None:
        bounds = tuner.largest_contour_bounds(mask)
        if bounds is not None:
            x, y, w, h = bounds
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 255), 2)
            # Label above the box normally, but inside it when that would land
            # under the HUD plate -- on a well-framed overhead shot the felt
            # starts near the top of the frame, so the collision is the common
            # case, not the edge case.
            label_y = y - 8 if y - 8 > plate_h else y + 24
            label_x = x + 6 if x + 6 > plate_w else plate_w + 12
            cv2.putText(
                canvas,
                "largest felt region",
                (label_x, label_y),
                font,
                0.5,
                (0, 220, 255),
                thickness,
                cv2.LINE_AA,
            )
    return canvas


def compose_mask_view(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint the masked region green and dim everything else.

    Better than showing the raw binary mask: keeping the underlying image
    visible is what lets you see *which* real-world thing leaked into the mask,
    rather than just that something did.
    """
    import cv2

    dimmed = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    tint = np.zeros_like(frame)
    tint[:, :, 1] = 255
    highlighted = cv2.addWeighted(frame, 0.65, tint, 0.35, 0)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(bool)
    return np.where(mask_3ch, highlighted, dimmed)


def save_snapshot(frame: np.ndarray, label: str = "snapshot") -> Path:
    """Write a frame to ``data/snapshots`` with a wall-clock filename.

    Wall clock here, not the monotonic frame timestamp: these files are for a
    human to correlate with what they were doing at the table.
    """
    import cv2

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    path = SNAPSHOT_DIR / f"{label}-{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    logger.info("saved %s", path)
    return path


def run_preview(args: argparse.Namespace, settings: Settings) -> int:
    """Main capture and display loop. Returns a process exit code."""
    from vision.camera import Camera, CameraError

    tracker = PerformanceTracker(target_fps=settings.system.target_fps)
    tuner = ThresholdTuner(settings)
    show_hud = True
    show_mask = bool(args.mask)
    windowed = not (args.headless or args.report)
    window = "ar_pool camera preview"

    if windowed:
        try:
            import cv2

            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            # Half size by default: a 1080p window on a Pi desktop over VNC is
            # unusable, and the preview does not need pixel accuracy.
            cv2.resizeWindow(window, settings.camera.width // 2, settings.camera.height // 2)
        except Exception as exc:  # noqa: BLE001 - headless build, no X, no GL
            logger.warning("cannot open a window (%s); falling back to headless", exc)
            windowed = False

    deadline = time.perf_counter() + args.seconds if args.seconds else None
    last_headless_write = 0.0
    last_summary = time.perf_counter()
    last_coverage: float | None = None
    using_mock = False
    exit_code = 0

    try:
        with Camera(settings.camera) as camera:
            logger.info(
                "camera open: backend=%s %dx%d target=%d FPS%s",
                camera.backend_name,
                settings.camera.width,
                settings.camera.height,
                settings.camera.fps,
                "  [MOCK - synthetic frames]" if camera.is_mock else "",
            )
            using_mock = camera.is_mock
            if camera.is_mock and not args.mock:
                # Loud, because silently previewing a synthetic table is exactly
                # how an afternoon gets lost tuning thresholds against a fake.
                logger.warning(
                    "no real camera found -- these frames are SYNTHETIC. "
                    "Any thresholds tuned now will not apply to your table."
                )

            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    logger.info("reached --seconds limit of %.1fs", args.seconds)
                    break

                tracker.begin_frame()
                # Every step is staged, so the report attributes all of the
                # frame time. Unattributed time is what makes a "too slow"
                # verdict impossible to act on.
                with tracker.stage("capture"):
                    try:
                        frame = camera.capture_frame()
                    except CameraError as exc:
                        logger.error("camera lost: %s", exc)
                        exit_code = 1
                        break
                if frame is None:
                    continue

                mask = None
                if show_mask or args.mask:
                    with tracker.stage("mask"):
                        mask = tuner.mask(frame.image)

                view = frame.image
                with tracker.stage("compose"):
                    if show_mask and mask is not None:
                        view = compose_mask_view(frame.image, mask)
                    if show_hud and not args.report:
                        view = draw_hud(
                            view,
                            tracker,
                            camera.backend_name,
                            frame.focus_distance,
                            tuner if (show_mask or args.mask) else None,
                            mask,
                        )

                # Close the frame before any snapshot write. JPEG encoding a
                # 1080p frame costs tens of ms, and counting it would make the
                # FPS verdict a measurement of the SD card rather than of the
                # camera.
                tracker.end_frame(capture_timestamp=frame.timestamp)
                if mask is not None:
                    last_coverage = tuner.coverage_pct(mask)

                if windowed:
                    import cv2

                    cv2.imshow(window, view)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("s"):
                        save_snapshot(frame.image)  # clean frame, no HUD
                    elif key == ord("h"):
                        show_hud = not show_hud
                    elif key == ord("m"):
                        show_mask = not show_mask
                    elif key == ord("]"):
                        tuner.widen_hue(1)
                    elif key == ord("["):
                        tuner.widen_hue(-1)
                    elif key == ord("="):
                        tuner.adjust_saturation(5)
                    elif key == ord("-"):
                        tuner.adjust_saturation(-5)
                    elif key == ord("p"):
                        print("\n" + tuner.as_yaml() + "\n", flush=True)
                    elif key in (ord("f"), ord("g")):
                        _nudge_focus(camera, settings, -0.1 if key == ord("f") else 0.1)
                elif args.headless:
                    now = time.perf_counter()
                    if now - last_headless_write >= HEADLESS_WRITE_INTERVAL:
                        with tracker.stage("jpeg_write"):
                            save_snapshot(view, label="preview")
                        last_headless_write = now

                # Time-based, not frame-count-based: this loop is deliberately
                # unthrottled to measure raw throughput, so a frame-count
                # interval fires many times a second on fast hardware.
                now = time.perf_counter()
                if now - last_summary >= settings.system.perf_log_interval_seconds:
                    tracker.log_summary(settings.system.latency_warn_ms)
                    last_summary = now

    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        if windowed:
            import cv2

            cv2.destroyAllWindows()
            cv2.waitKey(1)

    _print_report(
        tracker, settings, tuner, show_mask or args.mask, last_coverage, using_mock
    )
    return exit_code


def _nudge_focus(camera: object, settings: Settings, delta: float) -> None:
    """Adjust manual focus live, for the picamera2 backend only.

    Focus is set once and locked in production, so getting ``lens_position``
    right here and writing it into config is the whole point -- there is no
    autofocus to fall back on at play time.
    """
    settings.camera.lens_position = max(0.0, min(15.0, settings.camera.lens_position + delta))
    backend = getattr(camera, "_backend", None)
    cam = getattr(backend, "_cam", None)
    if cam is None:
        logger.info(
            "focus control needs the picamera2 backend; lens_position now %.2f (not applied)",
            settings.camera.lens_position,
        )
        return
    try:
        cam.set_controls({"LensPosition": settings.camera.lens_position})
        logger.info("lens_position=%.2f", settings.camera.lens_position)
    except (RuntimeError, AttributeError) as exc:
        logger.warning("could not set focus: %s", exc)


def _print_report(
    tracker: PerformanceTracker,
    settings: Settings,
    tuner: ThresholdTuner,
    include_thresholds: bool,
    coverage_pct: float | None = None,
    is_mock: bool = False,
) -> None:
    """Final summary, with an explicit verdict against the 30 FPS target.

    A verdict rather than raw numbers: the whole reason to run this on new
    hardware is to find out whether the camera can keep up, and that should not
    require interpreting a percentile.
    """
    snap = tracker.snapshot()
    target = settings.system.target_fps

    print("\n--- camera report ---", flush=True)
    print(f"frames captured   {snap.total_frames}", flush=True)
    print(f"throughput        {snap.fps:.1f} FPS (target {target})", flush=True)
    print(f"frame time        avg {snap.frame_ms_avg:.1f} ms  p95 {snap.frame_ms_p95:.1f} ms"
          f"  max {snap.frame_ms_max:.1f} ms", flush=True)
    print(f"capture latency   {snap.latency_ms:.1f} ms", flush=True)
    if snap.stage_ms:
        stages = "  ".join(f"{k}={v:.1f}ms" for k, v in sorted(snap.stage_ms.items()))
        print(f"stages            {stages}", flush=True)

    capture_ms = snap.stage_ms.get("capture", 0.0)

    if snap.total_frames == 0:
        verdict = "NO FRAMES CAPTURED -- check the ribbon cable and `libcamera-hello`"
    elif is_mock:
        # Refusing to judge is the honest answer. The synthetic backend redraws
        # a 1080p scene with OpenCV primitives every frame, which costs far more
        # than a real camera handing over a DMA buffer -- so a "too slow"
        # verdict here would be a measurement of the mock, not of any camera.
        verdict = (
            "NOT MEASURED -- these are synthetic frames. The mock backend "
            "redraws the scene in software each frame, so this says nothing "
            "about real camera throughput. Re-run on the Pi with the camera "
            "attached."
        )
    elif snap.fps >= target:
        verdict = f"OK -- comfortably holds {target} FPS"
    elif snap.fps >= target * 0.9:
        verdict = f"MARGINAL -- near {target} FPS with no headroom for detection"
    else:
        verdict = (
            f"TOO SLOW -- {snap.fps:.1f} FPS against a {target} FPS target. "
            "Detection and rendering still have to fit in the same budget."
        )
    print(f"verdict           {verdict}", flush=True)

    # Capture alone against the frame budget is the number that matters for
    # hardware sizing: everything else in this tool is diagnostic overhead that
    # the real pipeline will not pay.
    if capture_ms > 0 and not is_mock:
        budget_ms = 1000.0 / target
        print(
            f"capture alone     {capture_ms:.1f} ms of the {budget_ms:.1f} ms budget "
            f"({100.0 * capture_ms / budget_ms:.0f}%)",
            flush=True,
        )

    if include_thresholds:
        if coverage_pct is not None:
            # The number to tune by, so it belongs in the report and not just in
            # the on-screen HUD -- headless and --report users never see the HUD.
            if 55.0 <= coverage_pct <= 80.0:
                note = "in the expected band for an overhead shot"
            elif coverage_pct < 55.0:
                note = "LOW -- the mask is probably missing felt; widen the hue range"
            else:
                note = "HIGH -- the mask may have leaked off the table; narrow it"
            print(f"\nfelt mask coverage  {coverage_pct:.1f}%  ({note})", flush=True)
        print("\nfelt thresholds (paste into config.yaml):", flush=True)
        print(tuner.as_yaml(), flush=True)
    print(flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera preview and felt-threshold tuning tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    parser.add_argument(
        "--mask",
        action="store_true",
        help="start in felt-mask view, for tuning the HSV thresholds",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="write periodic JPEGs to data/snapshots instead of opening a window",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="timing only, no imagery -- for checking FPS over SSH",
    )
    parser.add_argument(
        "--seconds", type=float, default=None, help="stop after this many seconds"
    )
    parser.add_argument("--mock", action="store_true", help="use the synthetic camera")
    parser.add_argument("--width", type=int, default=None, help="override capture width")
    parser.add_argument("--height", type=int, default=None, help="override capture height")
    parser.add_argument("--fps", type=int, default=None, help="override target FPS")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)

    # File logging off by default: this is an interactive diagnostic, and
    # cluttering the application log with tuning sessions makes the real log
    # harder to read.
    setup_logging(args.log_level, log_to_file=False)

    if args.mock:
        settings.camera.use_mock = True
    if args.width:
        settings.camera.width = args.width
    if args.height:
        settings.camera.height = args.height
    if args.fps:
        settings.camera.fps = args.fps
        settings.system.target_fps = args.fps

    return run_preview(args, settings)


if __name__ == "__main__":
    sys.exit(main())
