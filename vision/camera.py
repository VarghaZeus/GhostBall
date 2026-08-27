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

    def focus_controller(self, settings: CameraSettings):
        """How this backend drives the lens, or ``None`` if it cannot.

        On the base class so the focus sweep and the wizard can ask the camera
        rather than going around it. They used to resolve a V4L2 subdev path
        themselves, which is fine right up until the lens is not reachable that
        way -- and on an AF-bound sensor it is not.
        """
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
        #: The chosen focus control. Cached because the sweep asks per step, and
        #: because selecting it reads the camera's control list.
        self._controller = None
        #: One line describing that choice and the evidence for it, for the log
        #: and the panel. The choice used to be invisible until a sweep failed.
        self._focus_path = ""

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
        self._cam.start()
        # Everything lens-related happens after start(), and that ordering is
        # the whole fix. libcamera writes the lens once at stream start on any
        # AF-bound sensor, so both halves -- AfMode=Manual and the position
        # itself -- have to come afterwards or they are overwritten a moment
        # later while every log line reports success. Measured symptom: the lens
        # parked at 477 and every one of 34 sweep positions read back 477.
        self._apply_focus(settings)

        if settings.warmup_seconds > 0:
            # AE/AWB need a moment; frames captured before this are dark or
            # colour-shifted enough to break HSV felt segmentation.
            logger.info("camera warm-up %.1fs", settings.warmup_seconds)
            time.sleep(settings.warmup_seconds)

    def _control_names(self) -> tuple[tuple[str, ...], str]:
        """Every control libcamera advertises, and why the list is empty if it is.

        Separated out and returning its *reason* because the previous version
        collapsed four different situations into one silent fallback: no camera
        object yet, an empty control list, an exception of a type that was
        caught, and an exception of a type that was not. Only the last of those
        is visible without this, and the fallback it produced -- V4L2 on an
        AF-bound sensor -- cannot work.

        Catches ``Exception`` deliberately. ``camera_controls`` walks libcamera's
        control descriptors and the interesting failures are third-party
        exception types this module has no business enumerating; narrowing it is
        how one of them ends up propagating out of ``start()`` and getting
        mistaken for the camera being absent.
        """
        if self._cam is None:
            return (), "no Picamera2 instance yet -- the camera has not been opened"
        try:
            controls = self._cam.camera_controls
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            return (), f"reading camera_controls raised {type(exc).__name__}: {exc}"
        if not controls:
            return (), "camera_controls is empty (called before configure()?)"
        try:
            return tuple(str(name) for name in controls), ""
        except TypeError as exc:
            return (), f"camera_controls is not iterable ({type(controls).__name__}): {exc}"

    def focus_controller(self, settings: CameraSettings):
        """Pick the control this sensor actually responds to, and say so out loud.

        Selected by capability, not configuration: ``AfMode`` in the camera's
        control list means libcamera has an AF algorithm bound, which means it
        owns the VCM and will fight anything written straight at the subdev.

        * **``AfMode`` present** -- IMX708 and friends. Drive ``LensPosition``
          through libcamera, in dioptres. Nothing else works: the V4L2 write is
          accepted and the lens does not move, because AF has already parked it.
        * **``AfMode`` absent** -- IMX519. Drive ``focus_absolute`` at the subdev
          in raw counts, which is the original path and stays exactly as it was.
          libcamera cannot interfere because it has no AF to interfere with.

        **The choice is logged with its evidence, every time.** It was previously
        inferred from a debug line and a fallback, which meant a wrong choice was
        invisible until a focus sweep failed several minutes later with a message
        about a ribbon cable. A decision this consequential has to announce
        itself, and it has to announce *what it saw* -- "AfMode absent" is only
        useful next to the list it was absent from.

        ``settings.focus_path`` forces the answer. Capability detection is still
        the default and the right mechanism; the override exists because when
        detection disagrees with a human looking at the same control list, the
        human needs a way to proceed while the disagreement is worked out.

        Cached: the sweep asks per step.
        """
        from vision.focus import FOCUS_COUNTS, FOCUS_DIOPTRES, LibcameraFocus, V4L2Focus

        if self._controller is not None:
            return self._controller

        names, problem = self._control_names()
        has_af = "AfMode" in names
        forced = (getattr(settings, "focus_path", "auto") or "auto").lower()

        if forced == "libcamera":
            want, why = FOCUS_DIOPTRES, "forced by camera.focus_path=libcamera in config"
        elif forced == "v4l2":
            want, why = FOCUS_COUNTS, "forced by camera.focus_path=v4l2 in config"
        elif has_af:
            want = FOCUS_DIOPTRES
            why = f"AfMode is present among {len(names)} camera controls"
        elif problem:
            # The dangerous case, and the one that used to be a debug line. There
            # is no evidence either way here, so falling back to V4L2 is a guess,
            # and on an AF-bound sensor it is a guess that cannot work.
            want = FOCUS_COUNTS
            why = f"could not determine AF support ({problem}) -- GUESSING V4L2"
        else:
            want = FOCUS_COUNTS
            why = f"AfMode is absent from {len(names)} camera controls"

        if want == FOCUS_DIOPTRES:
            self._controller = LibcameraFocus(self._cam)
        else:
            self._controller = V4L2Focus.find(settings.lens_driver)

        self._focus_path = self._describe_focus_path(self._controller, why, names, problem)
        # WARNING, not INFO, when the answer was guessed rather than observed:
        # that is the line somebody needs to see in a log they are skimming.
        (logger.warning if problem and forced == "auto" else logger.info)(
            "focus: %s", self._focus_path
        )
        if names:
            logger.debug("focus: camera controls seen: %s", ", ".join(sorted(names)))
        return self._controller

    @staticmethod
    def _describe_focus_path(controller, why: str, names, problem: str) -> str:
        """One line naming the control, its unit, and the evidence for choosing it.

        Shown on the panel as well as logged, so the choice is visible from the
        phone rather than only to somebody with journalctl open.
        """
        from vision.focus import FOCUS_DIOPTRES

        if controller is None:
            return f"no focus control available ({why})"
        if controller.kind == FOCUS_DIOPTRES:
            return f"libcamera LensPosition, dioptres -- {why}"
        return f"V4L2 focus_absolute, raw counts -- {why}"

    @property
    def focus_path(self) -> str:
        """How focus is being driven, and why. Empty until the camera is open."""
        return self._focus_path

    def _apply_focus(self, settings: CameraSettings) -> None:
        """Drive the lens to the configured position and verify it arrived.

        Whichever control this sensor responds to -- see :meth:`focus_controller`.
        Called after ``start()``, which is load-bearing: on an AF-bound sensor
        libcamera moves the lens at stream start, so anything set earlier is
        silently overwritten.

        The configured value is looked up *in this controller's unit*, so a
        dioptre calibration is never applied to a counts lens or the reverse.

        Failure is recorded, not raised. A camera that will not focus is a
        degraded system; refusing to start would be a dead one.
        """
        from vision.focus import FocusStatus, apply_focus, resolve_focus_value

        # Selected -- and therefore logged -- before the enabled check, so the
        # startup log states which control this sensor would use even on a rig
        # that has focus switched off. Selection touches no hardware; only
        # prepare() and the write do.
        controller = self.focus_controller(settings)

        if not settings.focus_enabled:
            self._focus = FocusStatus(detail="focus control disabled in config")
            return
        if controller is None:
            self._focus = FocusStatus(
                detail=(
                    f"no focus control available: this sensor has no AF algorithm and "
                    f"no V4L2 subdev matching {settings.lens_driver!r} was found."
                ),
            )
            logger.error("focus: %s", self._focus.detail)
            return

        value, source = resolve_focus_value(settings, kind=controller.kind)
        if value is None:
            # Never calibrated. Deliberately does not guess a number: the lens
            # sits where it powered up, which is visibly soft and points at the
            # real fix, where a plausible-looking guess would leave a
            # permanently mediocre picture with no symptom to chase.
            self._focus = FocusStatus(
                kind=controller.kind,
                detail=(
                    "no focus calibration for this rig. Run the focus calibration "
                    "(python -m tools.focus_sweep) -- the lens is at its power-on "
                    "position and the picture will be soft."
                ),
            )
            return

        self._focus = apply_focus(value, controller=controller, source=source)

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

    def focus_path(self) -> str:
        """How focus is being driven and why, or why it is not. For the panel.

        Answers the question "which path did it pick?" without anyone reading a
        log or the selection code -- which is what made a wrong choice here cost
        a whole diagnosis.
        """
        if self._backend is None:
            return "camera not open"
        described = getattr(self._backend, "focus_path", "")
        if described:
            return described
        # Force the selection so the answer exists. Cheap and side-effect free:
        # nothing is written to the lens until prepare().
        self.focus_controller()
        return getattr(self._backend, "focus_path", "") or (
            f"{self._backend.name} backend does not control focus"
        )

    def focus_controller(self):
        """How this rig's lens is driven, or ``None``.

        The single place the sweep, the wizard and the tools should get this
        from. They previously each resolved a V4L2 subdev themselves, which
        cannot reach an AF-bound lens at all.
        """
        if self._backend is None:
            return None
        return self._backend.focus_controller(self.settings)

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
