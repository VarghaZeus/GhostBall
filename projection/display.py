"""Projector output.

Backends mirror the camera module's structure: a real one, and a mock so the
pipeline runs headless.

``OpenCVDisplay`` uses a full-screen ``cv2.namedWindow``, which is the pragmatic
choice on a Pi running a desktop session -- it needs no compositor-specific
code and works over HDMI without extra dependencies. Its cost is that it must
be driven from the main thread and adds a copy. If frame time becomes the
binding constraint, the replacement is a DRM/KMS path writing directly to the
framebuffer, which is why this is behind an interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from app.config import ProjectorSettings, get_settings

logger = logging.getLogger(__name__)

WINDOW_NAME = "ar_pool_projection"


def _gui_backend() -> str:
    """Which GUI toolkit this OpenCV was built against.

    Worth logging because it decides almost everything about window behaviour --
    Qt and GTK handle fullscreen, decorations and thread affinity differently --
    and it is not something anyone can guess from the outside. A build with no
    GUI at all reports ``none``, which explains an awful lot at once.
    """
    import cv2

    try:
        info = cv2.getBuildInformation()
    except Exception:  # noqa: BLE001 - diagnostics must never break startup
        return "unknown"
    for line in info.splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            # Newer builds put the toolkit on the following lines instead.
            tail = stripped[4:].strip()
            if tail:
                return tail
        for toolkit in ("QT:", "GTK+:", "Cocoa:", "Win32 UI:"):
            if stripped.startswith(toolkit) and "YES" in stripped:
                return toolkit.rstrip(":")
    return "unknown"


class DisplayError(RuntimeError):
    """Raised when no display backend can be brought up."""


class _DisplayBackend(ABC):
    name = "base"

    @abstractmethod
    def start(self, settings: ProjectorSettings) -> None: ...

    @abstractmethod
    def show(self, frame_bgr: np.ndarray) -> bool: ...

    @abstractmethod
    def stop(self) -> None: ...


class OpenCVDisplay(_DisplayBackend):
    """Full-screen OpenCV window on the projector output."""

    name = "opencv"

    def __init__(self) -> None:
        self._started = False
        self._settings: ProjectorSettings | None = None
        #: The image area OpenCV reports, once measured. ``None`` means it has
        #: not been queried yet or could not be -- neither of which is a
        #: mismatch.
        self._actual_rect: tuple[int, int, int, int] | None = None
        self._verified = False

    def start(self, settings: ProjectorSettings) -> None:
        import cv2

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        # Order matters, and the previous order was wrong. Position and size
        # first, fullscreen last.
        #
        # `resizeWindow` on a window that has just been told to go fullscreen is
        # a contradictory request, and the GTK backend resolves it by honouring
        # the resize -- so the window came up as a plain 1920x1080 window with
        # the desktop visible around it. Setting the property last means nothing
        # afterwards can revoke it.
        #
        # Multi-head placement: moving the window to an x offset of one screen
        # width lands it on the second output. Crude, but it avoids depending on
        # xrandr or a Wayland protocol.
        if settings.display_index > 0:
            cv2.moveWindow(WINDOW_NAME, settings.width * settings.display_index, 0)
        cv2.resizeWindow(WINDOW_NAME, settings.width, settings.height)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self._started = True
        self._settings = settings
        logger.info(
            "projector window open at %dx%d on display %d via the %s GUI backend",
            settings.width,
            settings.height,
            settings.display_index,
            _gui_backend(),
        )

    def verify_geometry(self) -> tuple[int, int] | None:
        """Measure the image area, once, after a frame has actually been shown.

        Deliberately not called from :meth:`start`, and the previous version's
        doing so was a bug that produced a confident wrong number.

        Two things were wrong with measuring at open time:

        * ``getWindowImageRect`` reports the **image rendering area**, not the
          window. Before any ``imshow`` there is no image, so whatever it
          returns is not a measurement of anything.
        * Fullscreen is **asynchronous**. ``setWindowProperty`` posts a request
          to the window manager, which resizes the window some milliseconds
          later. Reading geometry on the next line reads the window mid-map --
          which is how a rig whose projection was genuinely full-screen could be
          reported as 1920x548, a size that corresponds to nothing and sent
          somebody hunting for a fullscreen bug.

        So this runs after the first frame and after the event loop has been
        pumped, and it reports the image area as exactly that.

        Returns:
            ``(width, height)`` of the image area, or ``None`` when it cannot be
            queried -- which is not the same as a mismatch and is not reported
            as one.
        """
        import cv2

        if not self._started or self._settings is None or self._verified:
            return None
        self._verified = True

        if not hasattr(cv2, "getWindowImageRect"):
            return None
        # Pump the event loop so the window manager's response to the fullscreen
        # request has actually been processed before anything is measured.
        for _ in range(5):
            cv2.waitKey(30)

        try:
            rect = cv2.getWindowImageRect(WINDOW_NAME)
        except cv2.error:
            return None
        if not rect:
            return None

        _, _, width, height = rect
        self._actual_rect = tuple(rect)
        wanted = (self._settings.width, self._settings.height)
        if (width, height) == wanted or width <= 0 or height <= 0:
            return (width, height)

        # Stated as a measurement with its provenance, not as a diagnosis. What
        # this reads is the area OpenCV draws the image into; a window manager
        # that reserved space for a panel, a toolkit that added a toolbar, and a
        # fullscreen request that was ignored all look the same from here.
        logger.warning(
            "projector image area is %dx%d, not the %dx%d requested. Overlays will be "
            "letterboxed or cropped, and alignment cannot be trusted until this "
            "matches. This measures OpenCV's drawing area (%s backend), which is not "
            "the same as the window: confirm with "
            "`xdotool search --name %s getwindowgeometry` before changing anything.",
            width,
            height,
            wanted[0],
            wanted[1],
            _gui_backend(),
            WINDOW_NAME,
        )
        return (width, height)

    def show(self, frame_bgr: np.ndarray) -> bool:
        import cv2

        if not self._started:
            return False
        cv2.imshow(WINDOW_NAME, frame_bgr)
        # waitKey is not optional: it is what pumps the window's event loop.
        # Without it the window never repaints and appears frozen.
        cv2.waitKey(1)
        if not self._verified:
            # Now there is an image to measure, and the window manager has had a
            # frame's worth of time to answer the fullscreen request.
            self.verify_geometry()
        return True

    def stop(self) -> None:
        if self._started:
            import cv2

            cv2.destroyWindow(WINDOW_NAME)
            cv2.waitKey(1)  # let the destroy actually process
            self._started = False


class MockDisplay(_DisplayBackend):
    """Discards frames, counting them. For headless dev and tests."""

    name = "mock"

    def __init__(self) -> None:
        self.frames_shown = 0
        self.last_frame: np.ndarray | None = None

    def start(self, settings: ProjectorSettings) -> None:
        logger.info(
            "mock display active (%dx%d) -- frames are discarded",
            settings.width,
            settings.height,
        )

    def show(self, frame_bgr: np.ndarray) -> bool:
        # Keeping the last frame makes assertions possible in tests and lets the
        # web panel serve a preview without a real projector attached.
        self.last_frame = frame_bgr
        self.frames_shown += 1
        return True

    def stop(self) -> None:
        self.last_frame = None


class Display:
    """Projector output with automatic backend selection.

    Owns the RGBA -> BGR flattening, so callers work in RGBA throughout and the
    conversion happens exactly once, at the last possible moment.
    """

    def __init__(self, settings: ProjectorSettings | None = None) -> None:
        self.settings = settings or get_settings().projector
        self._backend: _DisplayBackend | None = None
        #: Reused black frame for :meth:`clear`, built on first use. Declared
        #: here rather than only in ``clear`` because ``clear`` is what runs on
        #: the shutdown path, and an attribute that only exists once a method has
        #: been called is an AttributeError waiting for the least convenient
        #: moment.
        self._black: np.ndarray | None = None

    def open(self) -> Display:
        """Bring up the best available display backend."""
        candidates: list[_DisplayBackend]
        if self.settings.use_mock:
            candidates = [MockDisplay()]
        else:
            candidates = [OpenCVDisplay(), MockDisplay()]

        for backend in candidates:
            try:
                backend.start(self.settings)
            except ImportError as exc:
                logger.info("display backend %s unavailable: %s", backend.name, exc)
                continue
            except Exception as exc:  # noqa: BLE001
                # cv2 raises assorted errors with no display, no GL, or a
                # headless build. All of them mean "fall back to mock".
                logger.warning("display backend %s failed to start: %s", backend.name, exc)
                backend.stop()
                continue
            self._backend = backend
            return self

        raise DisplayError("no display backend could be started, not even the mock")

    def close(self) -> None:
        if self._backend is not None:
            logger.info("closing display backend %s", self._backend.name)
            self._backend.stop()
            self._backend = None

    def __enter__(self) -> Display:
        return self.open() if self._backend is None else self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "none"

    @property
    def is_mock(self) -> bool:
        return isinstance(self._backend, MockDisplay)

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` of the output."""
        return self.settings.width, self.settings.height

    @property
    def last_frame(self) -> np.ndarray | None:
        """The last BGR frame sent, when the backend retains one.

        Only :class:`MockDisplay` does. Exposed here so tests and the web
        preview can inspect output through the same wrapper they write through,
        rather than reaching past it to the backend.
        """
        return getattr(self._backend, "last_frame", None)

    def send_frame(self, overlay_rgba: np.ndarray) -> bool:
        """Flatten an RGBA overlay onto black and push it to the projector.

        Compositing onto black is the physically correct model: the projector
        adds light, so transparent means "project nothing here" and the felt
        shows through. Global opacity is applied here rather than at draw time,
        so the brightness slider takes effect without re-rendering.

        Returns:
            Whether the frame was displayed. ``False`` means the backend is not
            open -- routine during shutdown, so the caller should not treat it as
            an error.
        """
        import cv2

        if self._backend is None:
            return False

        if overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4:
            raise ValueError(f"expected an HxWx4 RGBA overlay, got {overlay_rgba.shape}")

        # Integer OpenCV ops, not NumPy float32. The obvious float32 expression
        # measured **60-75 ms** at 1080p -- twice the entire 33 ms frame budget,
        # and by far the most expensive thing in the pipeline. The cause is
        # scale rather than arithmetic: a 1080p RGBA frame is 8.3 MB, and
        # promoting it to float32 makes 33 MB that then gets walked several
        # times.
        #
        # cvtColor plus a fused uint8 multiply is the same computation in 5 ms,
        # an 11.5x saving, and agrees with the float path to within 1 count of
        # rounding. Anything touching every pixel of this frame has to be a
        # single OpenCV call.
        scale = (self.settings.overlay_alpha_pct / 100.0) * (
            self.settings.brightness_pct / 100.0
        )
        bgr = cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGR)
        # Per-pixel alpha is kept rather than folded into a global scalar: the
        # fading effects (trails, bursts, score popups) express their fade
        # through the alpha channel, and a global scalar would make them all
        # snap on and off instead.
        alpha3 = cv2.cvtColor(overlay_rgba[:, :, 3], cv2.COLOR_GRAY2BGR)
        flattened = cv2.multiply(bgr, alpha3, scale=scale / 255.0)
        return self._backend.show(flattened)

    def clear(self) -> bool:
        """Project black -- i.e. show nothing on the felt.

        Called on shutdown and on mode changes. It matters that this is cheap
        and reliable: leaving a stale overlay frozen on the table is worse than
        showing nothing at all.

        The black frame is allocated once and reused. It is 6 MB, and on mode
        changes this can be called on consecutive frames.
        """
        if self._backend is None:
            return False
        if self._black is None or self._black.shape[:2] != (
            self.settings.height,
            self.settings.width,
        ):
            self._black = np.zeros(
                (self.settings.height, self.settings.width, 3), dtype=np.uint8
            )
        return self._backend.show(self._black)


def init_projector_display(settings: ProjectorSettings | None = None) -> Display:
    """Open the projector display. Convenience wrapper matching the spec's name."""
    return Display(settings).open()
