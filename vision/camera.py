"""Camera capture for the Arducam 16MP autofocus (IMX519).

Three backends behind one :class:`Camera` interface, selected automatically:

``Picamera2Backend``
    The production path on the Pi. Uses the libcamera-based ``picamera2``
    stack, which is how the IMX519 is driven on Raspberry Pi OS -- the legacy
    ``cv2.VideoCapture`` route does not expose autofocus or the ISP controls we
    need. Installed via apt, not pip.
``OpenCVBackend``
    Any UVC webcam over ``cv2.VideoCapture``. Useful for developing on a laptop
    and as a fallback if picamera2 is unavailable but a camera is present.
``MockBackend``
    Synthesises a green felt surface with balls on it. This is what makes the
    rest of the pipeline developable with no hardware attached at all, and what
    the tests run against.

Frames are **BGR uint8** everywhere, matching OpenCV's convention, so the
detection code does not need to care which backend produced them. picamera2
natively hands back RGB, and the conversion happens inside that backend.
"""

from __future__ import annotations

import contextlib
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from app.config import CameraSettings, get_settings
from vision.crop import CropRect

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Frame:
    """One captured frame plus the metadata the pipeline needs.

    ``timestamp`` is ``time.perf_counter()`` taken as close to the capture as the
    backend allows. Latency accounting depends on it, so it must never be a
    wall-clock reading.
    """

    image: np.ndarray  # HxWx3 uint8, BGR
    timestamp: float
    index: int
    focus_distance: float | None = None  # dioptres, when the backend reports it

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` in pixels."""
        return self.image.shape[0], self.image.shape[1]


@dataclass(slots=True)
class ExposureLockStatus:
    """Whether exposure and white balance are pinned, and to what."""

    locked: bool = False
    #: The ISP settings that were pinned, for drift checking afterwards.
    baseline: dict = field(default_factory=dict)
    detail: str = "exposure lock not attempted"


class CameraError(RuntimeError):
    """Raised when no camera backend can be brought up."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _Backend(ABC):
    """Minimal contract a capture source must satisfy."""

    name: str = "base"

    @abstractmethod
    def start(self, settings: CameraSettings) -> None:
        """Open the device and begin streaming. Raises on failure."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Grab one BGR frame, or ``None`` if the grab failed."""

    @abstractmethod
    def stop(self) -> None:
        """Release the device. Must be safe to call twice."""

    def focus_status(self):
        """Lens focus state, for ``/api/status``.

        On the base class because the panel should be able to say "this backend
        does not do focus" rather than showing nothing -- an empty field reads
        as a bug, and "not applicable" does not.
        """
        from vision.focus import FocusStatus

        return getattr(self, "_focus", None) or FocusStatus(
            detail=f"{self.name} backend does not control focus"
        )

    def focus_distance(self) -> float | None:
        """Current lens position in dioptres, if the backend exposes it."""
        return None


class Picamera2Backend(_Backend):
    """libcamera / picamera2 path -- the production backend on the Pi."""

    name = "picamera2"

    def __init__(self) -> None:
        self._cam = None
        #: Outcome of the last focus attempt. Held on the backend rather than
        #: recomputed on demand: reading it back means an ioctl, and the panel
        #: polls twice a second.
        self._focus = None

    def start(self, settings: CameraSettings) -> None:
        from picamera2 import Picamera2  # imported lazily: apt-only dependency

        self._cam = Picamera2()
        # RGB888 rather than a YUV format: the ISP does the conversion for free,
        # and doing it ourselves per frame would cost several ms on the CPU.
        config = self._cam.create_video_configuration(
            main={"size": (settings.width, settings.height), "format": "RGB888"},
            # frame_rate is enforced by the sensor's frame duration limits, in
            # microseconds. Pinning both ends stops the ISP from lengthening
            # exposure in dim rooms and silently dropping us to 15 FPS.
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 / settings.fps),
                    int(1_000_000 / settings.fps),
                )
            },
        )
        self._cam.configure(config)
        # Before start, so libcamera's AF -- if this sensor has one -- is already
        # in manual mode by the time it would otherwise take the lens.
        self._release_libcamera_af()
        self._cam.start()
        # After start, deliberately. libcamera writes the lens once at start-up
        # on any sensor with an AF algorithm bound, so a V4L2 write issued
        # beforehand is overwritten a moment later and the log says it succeeded.
        # Writing last means ours is the value that survives.
        self._apply_focus(settings)

        if settings.warmup_seconds > 0:
            # AE/AWB need a moment; frames captured before this are dark or
            # colour-shifted enough to break HSV felt segmentation.
            logger.info("camera warm-up %.1fs", settings.warmup_seconds)
            time.sleep(settings.warmup_seconds)

    def _release_libcamera_af(self) -> None:
        """Put libcamera's autofocus in manual mode, where the sensor has one.

        The module docstring in :mod:`vision.focus` explains why focus is driven
        straight at the VCM over V4L2: on the IMX519 the stock tuning file binds
        no AF algorithm, so ``AfMode`` and ``LensPosition`` are silently dropped.
        That reasoning has a hole in it, and swapping to a Camera Module 3 found
        it -- the IMX708's tuning *does* bind ``rpi.af``, and then libcamera owns
        the very VCM this code writes to.

        Two processes driving one voice coil produces a specific and thoroughly
        confusing symptom: every V4L2 write is accepted, and the readback
        disagrees at *every* position, because libcamera has moved the lens in
        between. Meanwhile a manual ``v4l2-ctl`` write with nothing streaming
        sticks perfectly -- so the cable, which is the thing the old diagnosis
        blamed, tests fine.

        Best-effort by design. A sensor with no AF algorithm has no ``AfMode``
        control and there is nothing to do; a failure to set it is worth a log
        line and not worth refusing to start over, since on such a sensor the
        control was never being honoured anyway.
        """
        try:
            controls = self._cam.camera_controls
        except Exception as exc:  # noqa: BLE001 - any failure here is non-fatal
            logger.debug("focus: could not read camera controls (%s)", exc)
            return

        if "AfMode" not in controls:
            # The IMX519 case. Stated at debug rather than passed over in
            # silence, because "no AF to disable" and "failed to disable AF" are
            # different and the next person will want to know which happened.
            logger.debug("focus: no AfMode control; libcamera has no AF bound here")
            return

        try:
            # 0 is AfModeEnum.Manual. The integer rather than the enum import so
            # this does not depend on a picamera2 version exposing it.
            self._cam.set_controls({"AfMode": 0})
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.warning(
                "focus: this sensor has an AF algorithm and AfMode=Manual was refused "
                "(%s). libcamera may drive the lens against us; expect focus readbacks "
                "to disagree at every position.",
                exc,
            )
            return

        logger.info(
            "focus: this sensor binds an AF algorithm; set AfMode=Manual so libcamera "
            "leaves the lens to us"
        )

    def _apply_focus(self, settings: CameraSettings) -> None:
        """Drive the lens to the configured position over V4L2, and verify it.

        Deliberately *not* ``AfMode``/``LensPosition`` through picamera2. The
        stock Pi tuning file for this sensor binds no AF algorithm, so libcamera
        drops both controls with an internal warning and no exception -- the
        Python call returns cleanly having done nothing, which reports success
        while every frame stays soft. See :mod:`vision.focus`.

        Failure is recorded, not raised. A camera that will not focus is a
        degraded system; refusing to start would be a dead one.
        """
        from vision.focus import FocusStatus, apply_focus, resolve_focus_value

        if not settings.focus_enabled:
            self._focus = FocusStatus(detail="focus control disabled in config")
            return

        value, source = resolve_focus_value(settings)
        if value is None:
            # Never calibrated. Deliberately does not guess a number: the lens
            # powers up at 0 and stays there, which is visibly soft and points
            # at the real fix, where a plausible-looking guess would leave a
            # permanently mediocre picture with no symptom to chase.
            self._focus = FocusStatus(
                detail=(
                    "no focus calibration for this rig. Run the focus calibration "
                    "(python -m tools.focus_sweep) -- the lens is at its power-on "
                    "position and the picture will be soft."
                ),
            )
            return

        self._focus = apply_focus(value, name_fragment=settings.lens_driver, source=source)

    # -- exposure and white balance ----------------------------------------

    def capture_controls(self) -> dict[str, object]:
        """Read back the ISP settings currently in force, from frame metadata."""
        if self._cam is None:
            return {}
        try:
            metadata = self._cam.capture_metadata()
        except (RuntimeError, OSError):
            return {}
        return {
            key: metadata[key]
            for key in ("ExposureTime", "AnalogueGain", "ColourGains")
            if key in metadata
        }

    def set_controls(self, controls: dict[str, object]) -> bool:
        """Push ISP controls. Returns whether the call was accepted.

        "Accepted" is not "applied" -- that lesson is already learned in
        :mod:`vision.focus`, and it is why callers here verify with
        :meth:`capture_controls` rather than trusting this.
        """
        if self._cam is None:
            return False
        try:
            self._cam.set_controls(controls)
            return True
        except (RuntimeError, AttributeError, KeyError) as exc:
            logger.warning("could not set camera controls %s: %s", sorted(controls), exc)
            return False

    def read(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        # Despite the "RGB888" label, picamera2 hands back BGR-ordered bytes for
        # this format, which is what OpenCV wants -- so no conversion here. If
        # colours ever come out inverted, this is the line to look at.
        return self._cam.capture_array("main")

    def focus_distance(self) -> float | None:
        """Always ``None`` on this stack, and that is the honest answer.

        ``LensPosition`` is a libcamera control, and libcamera is not driving
        this lens -- see :meth:`_apply_focus`. The metadata key is simply absent.
        The real focus state is :meth:`focus_status`, which reports the raw
        ``focus_absolute`` the motor actually holds.
        """
        return None

    def focus_status(self):
        return self._focus

    def stop(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except RuntimeError as exc:
                logger.warning("error closing picamera2: %s", exc)
            self._cam = None


class OpenCVBackend(_Backend):
    """UVC webcam over ``cv2.VideoCapture``. Development fallback."""

    name = "opencv"

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._cap = None

    def start(self, settings: CameraSettings) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise CameraError(f"cv2.VideoCapture({self.device_index}) would not open")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        self._cap.set(cv2.CAP_PROP_FPS, settings.fps)
        # A 1-frame buffer keeps latency down; without it VideoCapture hands
        # back stale queued frames and the overlay lags the real balls.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (actual_w, actual_h) != (settings.width, settings.height):
            # Not fatal, but the homography and expected ball radius are
            # resolution-dependent, so the user needs to know.
            logger.warning(
                "camera gave %dx%d, not the requested %dx%d",
                actual_w,
                actual_h,
                settings.width,
                settings.height,
            )

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class MockBackend(_Backend):
    """Synthetic table, for development and tests with no camera present.

    Draws a green felt rectangle, six dark pockets and a rack of balls, with the
    cue ball tracking slowly across the table so that motion-dependent code
    (tracking, strike detection) has something to chew on. It is deliberately
    crude -- enough structure for the detector to find, not a renderer.
    """

    name = "mock"

    def __init__(self) -> None:
        self._settings: CameraSettings | None = None
        self._tick = 0
        self._base: np.ndarray | None = None

    def start(self, settings: CameraSettings) -> None:
        self._settings = settings
        self._base = self._build_base(settings.width, settings.height)
        logger.info("mock camera active (%dx%d) -- no hardware in use", settings.width, settings.height)

    @staticmethod
    def _build_base(width: int, height: int) -> np.ndarray:
        """Render the static parts: felt, rails and pockets."""
        import cv2

        frame = np.full((height, width, 3), (30, 25, 20), dtype=np.uint8)  # dark rail
        margin_x, margin_y = int(width * 0.08), int(height * 0.12)
        # Felt green in BGR. Chosen to sit inside the default felt_hue_range.
        cv2.rectangle(
            frame,
            (margin_x, margin_y),
            (width - margin_x, height - margin_y),
            (70, 120, 45),
            thickness=-1,
        )
        pocket_r = int(min(width, height) * 0.028)
        xs = (margin_x, width // 2, width - margin_x)
        ys = (margin_y, height - margin_y)
        for y in ys:
            for x in xs:
                cv2.circle(frame, (x, y), pocket_r, (12, 12, 12), thickness=-1)
        return frame

    def read(self) -> np.ndarray | None:
        import cv2

        if self._base is None or self._settings is None:
            return None
        frame = self._base.copy()
        height, width = frame.shape[:2]
        ball_r = int(min(width, height) * 0.022)

        # Object balls in a loose spread, BGR.
        palette = [
            (40, 200, 220),  # yellow
            (200, 80, 40),  # blue
            (40, 40, 200),  # red
            (150, 40, 140),  # purple
            (40, 130, 230),  # orange
            (60, 160, 60),  # green
            (40, 40, 90),  # maroon
            (20, 20, 20),  # eight
        ]
        for i, color in enumerate(palette):
            cx = int(width * (0.55 + 0.05 * (i % 4)))
            cy = int(height * (0.38 + 0.10 * (i // 4)))
            cv2.circle(frame, (cx, cy), ball_r, color, thickness=-1)
            # A specular highlight, so the detector sees the same glare it will
            # have to cope with on a real table.
            cv2.circle(frame, (cx - ball_r // 3, cy - ball_r // 3), max(1, ball_r // 4), (245, 245, 245), -1)

        # Cue ball sweeps back and forth so velocity-dependent code has signal.
        phase = (self._tick % 120) / 120.0
        sweep = abs(phase * 2.0 - 1.0)  # triangle wave 1 -> 0 -> 1
        cue_x = int(width * (0.20 + 0.12 * sweep))
        cue_y = int(height * 0.50)
        cv2.circle(frame, (cue_x, cue_y), ball_r, (250, 250, 250), thickness=-1)

        # A cue stick pointing at the cue ball from the left.
        cv2.line(
            frame,
            (cue_x - int(width * 0.16), cue_y + int(height * 0.03)),
            (cue_x - ball_r - 4, cue_y),
            (90, 140, 190),
            thickness=max(2, ball_r // 3),
        )

        self._tick += 1
        return frame

    def stop(self) -> None:
        self._base = None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class Camera:
    """Capture source with automatic backend selection and graceful degradation.

    Use as a context manager::

        with Camera() as cam:
            for frame in cam.stream_frames():
                ...
    """

    def __init__(self, settings: CameraSettings | None = None) -> None:
        self.settings = settings or get_settings().camera
        self._backend: _Backend | None = None
        self._frame_index = 0
        self._consecutive_failures = 0
        #: Give up on a backend after this many consecutive failed grabs. At
        #: 30 FPS this is a third of a second of nothing -- a real fault, not a
        #: transient hiccup.
        self._max_consecutive_failures = 10
        #: The crop actually applied to the last frame, and the frame it was
        #: applied to. Both ``None`` until the first frame, because a crop is
        #: clamped against the real frame size and that is not known until one
        #: arrives -- see :meth:`_apply_crop`.
        self._crop: CropRect | None = None
        self._sensor_size: tuple[int, int] | None = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> Camera:
        """Bring up the best available backend.

        Tries picamera2, then OpenCV, then the mock. Falls through on
        ``ImportError`` (dependency absent) and on backend errors (hardware
        absent or busy), because on a dev box both are the normal case. Only
        raises if even the mock cannot start, which would indicate a broken
        NumPy/OpenCV install rather than a camera problem.
        """
        candidates: list[_Backend] = []
        if self.settings.use_mock:
            candidates = [MockBackend()]
        else:
            candidates = [Picamera2Backend(), OpenCVBackend(), MockBackend()]

        for backend in candidates:
            try:
                backend.start(self.settings)
            except ImportError as exc:
                logger.info("backend %s unavailable: %s", backend.name, exc)
                continue
            except (CameraError, RuntimeError, OSError, ValueError) as exc:
                logger.warning("backend %s failed to start: %s", backend.name, exc)
                backend.stop()
                continue

            self._backend = backend
            logger.info(
                "camera open via %s at %dx%d target %d FPS",
                backend.name,
                self.settings.width,
                self.settings.height,
                self.settings.fps,
            )
            return self

        raise CameraError("no camera backend could be started, not even the mock")

    def close(self) -> None:
        """Release the device. Safe to call more than once."""
        if self._backend is not None:
            logger.info("closing camera backend %s", self._backend.name)
            self._backend.stop()
            self._backend = None

    def __enter__(self) -> Camera:
        return self.open() if self._backend is None else self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- capture ------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "none"

    @property
    def focus(self):
        """Lens focus state, as a :class:`vision.focus.FocusStatus`.

        Always answers, even with no backend open, so the panel can distinguish
        "focus failed" from "the camera never started" -- which look identical
        from an empty field.
        """
        from vision.focus import FocusStatus

        if self._backend is None:
            return FocusStatus(detail="camera not open")
        return self._backend.focus_status()

    @property
    def is_mock(self) -> bool:
        """Whether frames are synthetic. The web panel surfaces this, so nobody
        spends an hour tuning HSV ranges against a fake table."""
        return isinstance(self._backend, MockBackend)

    # -- exposure lock ------------------------------------------------------

    @contextlib.contextmanager
    def exposure_lock(self, settle_frames: int = 10):
        """Freeze exposure, gain and white balance for the duration of the block.

        The single most dangerous thing in a focus sweep. Sharpness is measured
        as the variance of the Laplacian, which scales with *contrast* -- so if
        auto-exposure moves while the lens is stepping, the brightness change is
        indistinguishable from a focus change and lands the peak in the wrong
        place. It fails silently and produces a confident wrong number, which is
        the worst possible shape for this particular bug.

        Auto-exposure will move, too, and not by coincidence: the sweep starts
        with a badly defocused frame, which is *dimmer and flatter* than a
        focused one, so AE ramps gain up at the start of the sweep and back down
        as focus improves. That is a monotonic brightness trend pointing the same
        way as the focus trend.

        The lock is: let AE settle, read what it chose, then pin those exact
        values with the auto algorithms disabled. Restored on the way out, in a
        ``finally``, so an exception mid-sweep does not leave the camera pinned
        to a dark room's exposure for the rest of the session.

        Yields:
            A :class:`ExposureLockStatus`. Check ``.locked`` -- a backend with
            no ISP controls yields an unlocked status rather than raising, and
            the caller decides whether to proceed or refuse.
        """
        backend = self._backend
        status = ExposureLockStatus()

        if backend is None or not hasattr(backend, "set_controls"):
            status = ExposureLockStatus(
                detail=f"the {self.backend_name} backend has no exposure controls"
            )
            yield status
            return

        # Let AE and AWB converge on the scene as it is now, with the targets
        # already up -- locking to a reading taken before they appeared would
        # pin the exposure for a black frame.
        for _ in range(settle_frames):
            self.capture_frame()

        baseline = backend.capture_controls()
        if not baseline:
            status = ExposureLockStatus(detail="the camera reported no exposure metadata")
            yield status
            return

        controls: dict[str, object] = {"AeEnable": False, "AwbEnable": False}
        if "ExposureTime" in baseline:
            controls["ExposureTime"] = int(baseline["ExposureTime"])
        if "AnalogueGain" in baseline:
            controls["AnalogueGain"] = float(baseline["AnalogueGain"])
        if "ColourGains" in baseline:
            controls["ColourGains"] = tuple(baseline["ColourGains"])

        accepted = backend.set_controls(controls)
        for _ in range(settle_frames):
            self.capture_frame()

        status = ExposureLockStatus(
            locked=accepted,
            baseline=dict(baseline),
            detail=(
                f"exposure {baseline.get('ExposureTime', '?')}us "
                f"gain {baseline.get('AnalogueGain', '?')}"
                if accepted
                else "the camera did not accept the exposure lock"
            ),
        )
        try:
            yield status
        finally:
            backend.set_controls({"AeEnable": True, "AwbEnable": True})
            logger.info("exposure lock released; auto exposure and white balance back on")

    def exposure_drifted(self, status, tolerance: float = 0.02) -> str | None:
        """Whether exposure has moved since the lock was taken.

        The verification half, and the reason the lock is not simply trusted.
        ``set_controls`` succeeding means the request was accepted, which is a
        different claim from the ISP having honoured it -- exactly the
        distinction that made the libcamera focus path report success while
        doing nothing.

        Returns a description of the drift, or ``None`` if it held.
        """
        backend = self._backend
        if backend is None or not status.locked or not status.baseline:
            return None
        current = backend.capture_controls()
        if not current:
            return None

        for key in ("ExposureTime", "AnalogueGain"):
            was, now = status.baseline.get(key), current.get(key)
            if was in (None, 0) or now is None:
                continue
            if abs(float(now) - float(was)) / abs(float(was)) > tolerance:
                return f"{key} moved from {was} to {now} during the measurement"
        return None

    def capture_frame(self) -> Frame | None:
        """Grab a single frame.

        Returns ``None`` on a failed grab rather than raising -- a dropped frame
        is routine (USB hiccup, ISP reconfiguration) and the main loop should
        skip it and carry on. Sustained failure raises :class:`CameraError`,
        because at that point the device is genuinely gone and looping silently
        would hide it.
        """
        if self._backend is None:
            raise CameraError("capture_frame() before open()")

        image = self._backend.read()
        if image is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_consecutive_failures:
                raise CameraError(
                    f"{self._consecutive_failures} consecutive capture failures "
                    f"from backend {self._backend.name}; device lost"
                )
            logger.debug("frame grab failed (%d in a row)", self._consecutive_failures)
            return None

        self._consecutive_failures = 0
        image = self._apply_rotation(image)
        image = self._apply_crop(image)
        frame = Frame(
            image=image,
            timestamp=time.perf_counter(),
            index=self._frame_index,
            focus_distance=self._backend.focus_distance(),
        )
        self._frame_index += 1
        return frame

    def _apply_crop(self, image: np.ndarray) -> np.ndarray:
        """Take the configured sub-rectangle, after rotation, before anything else.

        The last thing that happens to a frame before the rest of the program
        sees it, which is the point: there is one frame coordinate system at a
        time and "full frame" simply means "whatever came out of here". No
        consumer is handed an offset, so no consumer can forget to apply one.

        After rotation because the panel's pan controls have to mean the same
        thing at any ``rotation_deg``; see :mod:`vision.crop`.

        A copy, not a view. ``image[y0:y1, x0:x1]`` is a non-contiguous view of
        the parent buffer, which keeps the whole full-resolution frame alive for
        as long as anything holds the crop, and which several OpenCV calls
        quietly copy anyway. Copying once here is cheaper than the alternatives
        and makes the memory behaviour obvious.

        Records the crop actually applied on ``self._crop``, so the panel can
        report what is in force rather than what was requested -- those differ
        whenever a saved crop is clamped to a frame it no longer fits.
        """
        height, width = image.shape[:2]
        self._sensor_size = (width, height)

        crop = self.settings.crop
        if not crop.enabled or crop.width <= 0 or crop.height <= 0:
            self._crop = CropRect.full(width, height)
            return image

        rect = CropRect(
            x=crop.x, y=crop.y, width=crop.width, height=crop.height
        ).clamped(width, height)
        self._crop = rect
        if rect.is_full(width, height):
            return image
        return np.ascontiguousarray(image[rect.y : rect.y1, rect.x : rect.x1])

    @property
    def crop(self) -> CropRect:
        """The crop in force, in sensor space. Full-frame when not cropping.

        ``None`` until the first frame: the rectangle is clamped against the real
        frame, and the real frame size is not known until one arrives. Callers
        that need it before then should treat the configured value as a request
        rather than a fact.
        """
        return self._crop

    @property
    def sensor_size(self) -> tuple[int, int] | None:
        """Full post-rotation frame size, ``(width, height)``, or ``None``.

        Measured from a real frame rather than taken from config, because a
        backend can hand back something other than what it was asked for -- the
        OpenCV path already logs when it does.
        """
        return self._sensor_size

    def _apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """Rotate to correct for a sideways-mounted camera."""
        if self.settings.rotation_deg == 0:
            return image
        import cv2

        codes = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        return cv2.rotate(image, codes[self.settings.rotation_deg])

    def stream_frames(self, max_frames: int | None = None) -> Iterator[Frame]:
        """Yield frames continuously.

        Failed grabs are skipped rather than yielded, so consumers never have to
        null-check. Does not rate-limit -- the main loop owns pacing via
        :class:`utils.performance.RateLimiter`, since it needs to account for
        detection and render time too.

        Args:
            max_frames: Stop after this many successful frames. Bounded runs
                keep tests from hanging; ``None`` streams until closed.
        """
        if self._backend is None:
            self.open()

        produced = 0
        while max_frames is None or produced < max_frames:
            frame = self.capture_frame()
            if frame is None:
                continue
            yield frame
            produced += 1


def init_camera(settings: CameraSettings | None = None) -> Camera:
    """Open a camera. Convenience wrapper matching the spec's function name."""
    return Camera(settings).open()
