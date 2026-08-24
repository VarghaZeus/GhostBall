"""Camera capture for the Arducam 16MP autofocus (IMX519).

Three backends behind one :class:`Camera` interface, selected automatically:

``Picamera2Backend``
    The production path on the Pi. Uses the libcamera-based ``picamera2``
    stack, which is how the IMX519 is driven on Bookworm -- the legacy
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

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from app.config import CameraSettings, get_settings

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

    def focus_distance(self) -> float | None:
        """Current lens position in dioptres, if the backend exposes it."""
        return None


class Picamera2Backend(_Backend):
    """libcamera / picamera2 path -- the production backend on the Pi."""

    name = "picamera2"

    def __init__(self) -> None:
        self._cam = None

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
        self._apply_focus(settings)
        self._cam.start()

        if settings.warmup_seconds > 0:
            # AE/AWB need a moment; frames captured before this are dark or
            # colour-shifted enough to break HSV felt segmentation.
            logger.info("camera warm-up %.1fs", settings.warmup_seconds)
            time.sleep(settings.warmup_seconds)

    def _apply_focus(self, settings: CameraSettings) -> None:
        """Set the autofocus mode, tolerating lenses that do not support it."""
        from libcamera import controls  # type: ignore[import-not-found]

        try:
            if settings.autofocus_mode == "manual":
                # Manual is the right production choice: continuous AF hunts
                # when a hand or cue enters frame, and a focus shift mid-shot
                # changes the apparent ball radius and breaks detection.
                self._cam.set_controls(
                    {
                        "AfMode": controls.AfModeEnum.Manual,
                        "LensPosition": settings.lens_position,
                    }
                )
            elif settings.autofocus_mode == "continuous":
                self._cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            else:
                self._cam.set_controls({"AfMode": controls.AfModeEnum.Auto})
                self._cam.autofocus_cycle()
        except (RuntimeError, AttributeError) as exc:
            logger.warning("could not set autofocus mode (%s); using lens default", exc)

    def read(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        # Despite the "RGB888" label, picamera2 hands back BGR-ordered bytes for
        # this format, which is what OpenCV wants -- so no conversion here. If
        # colours ever come out inverted, this is the line to look at.
        return self._cam.capture_array("main")

    def focus_distance(self) -> float | None:
        if self._cam is None:
            return None
        try:
            return float(self._cam.capture_metadata().get("LensPosition"))
        except (RuntimeError, TypeError, ValueError):
            return None

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
    def is_mock(self) -> bool:
        """Whether frames are synthetic. The web panel surfaces this, so nobody
        spends an hour tuning HSV ranges against a fake table."""
        return isinstance(self._backend, MockBackend)

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
        frame = Frame(
            image=image,
            timestamp=time.perf_counter(),
            index=self._frame_index,
            focus_distance=self._backend.focus_distance(),
        )
        self._frame_index += 1
        return frame

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
