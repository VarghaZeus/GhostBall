"""Shared application state.

One object holding everything the vision loop and the web handlers both need to
see. It exists because those two run on different threads: the loop drives
capture and rendering, while uvicorn's event loop serves the panel. Rather than
scatter locks, all cross-thread data lives here, the loop is the only writer of
per-frame fields, and handlers only read them.

Per-frame fields are single references (``latest_game_state``,
``table_boundary``) assigned atomically, so a handler either sees the previous
frame's object or the new one -- never a half-updated one. That is enough for a
status panel and avoids a lock in the 33 ms hot path.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.config import Settings, get_settings
from app.models import GameState, ShotPrediction, TableBoundary

logger = logging.getLogger(__name__)

#: Pipeline stages still raising ``NotImplementedError``. Surfaced through
#: ``/api/status`` so a blank projection explains itself instead of looking like
#: a crash, and used by the API to answer 503 rather than 500 on routes that
#: depend on them.
#:
#: Keep this in sync as phases land -- it is a hand-maintained list because the
#: alternative (probing each function by calling it) would need real frames and
#: has side effects.
PENDING_STAGES: set[str] = {
    "hailo",  # vision.inference.* (optional)
}


@dataclass
class AppState:
    """Everything shared between the vision loop and the web layer."""

    settings: Settings = field(default_factory=get_settings)

    # -- long-lived resources, owned by the loop, read by handlers ----------
    camera: object | None = None  # vision.camera.Camera
    display: object | None = None  # projection.display.Display
    mapper: object = None  # projection.mapper.ProjectionMapper
    tracker: object = None  # utils.performance.PerformanceTracker
    mode_manager: object = None  # modes.mode_manager.ModeManager

    # -- per-frame state, written only by the loop --------------------------
    latest_frame: np.ndarray | None = None
    latest_game_state: GameState | None = None
    latest_prediction: ShotPrediction | None = None
    #: The last RGBA overlay sent to the projector, for the web preview.
    #:
    #: A reference, not a copy. The loop reuses its canvas between frames, so a
    #: handler reading this while the loop is drawing can get a half-drawn frame
    #: -- a torn *preview image*, once, on a diagnostic view. Copying 8 MB per
    #: frame to avoid that would cost more of the frame budget than the entire
    #: render stage.
    latest_overlay: np.ndarray | None = None
    #: Cached table boundary. Table detection is expensive and the table does
    #: not move, so it runs on an interval rather than per frame.
    table_boundary: TableBoundary | None = None
    camera_to_table: np.ndarray | None = None
    table_to_camera: np.ndarray | None = None
    #: Ball tracker, giving stable ids and velocities across frames.
    tracker_balls: object = None  # vision.detection.BallTracker
    #: Cross-frame shot prediction cache. Quantises the inputs so an unchanged
    #: aim is a dictionary lookup, and incidentally stops detection jitter from
    #: making the projected line shimmer.
    prediction_cache: object = None  # physics.simulator.PredictionCache
    #: Pockets are derived from the table boundary and cached with it -- they do
    #: not move, so re-deriving them per frame is pure waste.
    pockets: list = field(default_factory=list)
    #: The running setup wizard, or ``None``. Owned here rather than in a
    #: separate process so that the projector and the phone drive one machine:
    #: the loop ticks it, the API reads and acts on it, and both always agree.
    wizard: object = None  # app.wizard.Wizard
    #: Exclusive claim on the projector while the wizard runs. Guarded by a real
    #: mutex, unlike the rest of this object -- two phones tapping Start at once
    #: genuinely race, and the outcome decides who owns the table.
    wizard_lock: object = None  # app.exclusive.ExclusiveLock
    #: Latest bare-cloth sharpness reading, for the calibration staleness check.
    #: ``None`` until one has been taken.
    live_sharpness: float | None = None

    #: Whether the system has a table to play on, and what to say if not.
    #: Owned by the vision loop, read by the renderer and the panel. Separate
    #: from the mode manager's ``SessionState``, which is about a shot.
    readiness: object = None  # app.readiness.ReadinessTracker
    #: How many table-detection passes have come up empty in a row. Feeds the
    #: retry backoff, so a rig pointed at a ceiling stops paying full price for
    #: a search that is not going anywhere.
    table_attempts: int = 0
    last_shot_confidence: float = 0.0

    # -- health and degradation bookkeeping ---------------------------------
    #
    # The target is a system that runs for hours. Over that span the interesting
    # failures are not crashes -- those are obvious -- but slow ones: a camera
    # that drops off USB and comes back, a stage that starts throwing on one
    # frame in fifty, a loop that falls behind and never catches up. None of
    # those show up in an FPS average, so each gets a counter here and a line on
    # the panel.

    #: ``perf_counter`` at construction. Uptime is measured from here rather
    #: than from process start: this object is built once per run and a
    #: monotonic origin cannot be moved by NTP mid-session.
    started_at: float = field(default_factory=time.perf_counter)
    #: Wall clock at construction, ISO-8601. For the panel only -- never for
    #: arithmetic, since it can step.
    started_wall: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    #: Frames the loop has fully processed. Distinct from
    #: ``tracker.total_frames``, which resets when stats are cleared per mode.
    frames_processed: int = 0
    #: How many times the camera was lost and successfully reopened. A nonzero
    #: value with a healthy FPS is the signature of a marginal USB cable, which
    #: is otherwise near-impossible to diagnose from a projected overlay.
    camera_reconnects: int = 0
    #: Stage name -> count of unexpected exceptions. Counted rather than logged
    #: per occurrence: at 30 FPS a stage that throws every frame produces 1800
    #: log lines a minute and buries everything else.
    stage_errors: dict[str, int] = field(default_factory=dict)
    #: Stages switched off after failing on too many consecutive frames. The
    #: loop keeps running without them -- a broken renderer should cost the
    #: overlay, not the game.
    disabled_stages: set[str] = field(default_factory=set)
    #: 0 = full pipeline, higher = shedding optional work to hold frame rate.
    #: See ``VisionLoop._adapt``.
    degradation_level: int = 0
    #: Last unexpected error, as ``stage: message``. One string, for the panel.
    last_error: str | None = None
    #: Where the projector calibration came from: ``file`` or ``identity``.
    #: Worth surfacing because an identity fallback projects a plausible-looking
    #: overlay that is aligned to nothing, and that reads as a vision bug.
    calibration_source: str = "identity"
    #: Set by the watchdog when frames stop arriving; cleared when they resume.
    loop_stalled: bool = False
    stall_count: int = 0

    # -- control flags ------------------------------------------------------
    is_running: bool = False
    auto_detect_cue: bool = True
    #: Set by the web layer and shutdown handlers to stop the loop cleanly.
    stop_event: threading.Event = field(default_factory=threading.Event)

    # -- projection requests, set by the web layer, consumed by the loop -----
    #
    # These exist because of which *thread* owns the projector. The display is a
    # full-screen OpenCV window, and OpenCV requires its windows to be driven
    # from the thread that created them -- the vision loop. A request handler
    # calling ``display.send_frame`` from uvicorn's event loop is undefined
    # behaviour that ranges from a silently ignored repaint to a segfault,
    # depending on the platform's GUI backend.
    #
    # So the web layer never draws. It leaves a note here, and the loop picks it
    # up on its next pass -- at 30 FPS, within 33 ms, which is imperceptible to
    # someone tapping a button on a phone.

    #: Name of a :class:`projection.patterns.TestPattern` to project instead of
    #: the active mode's overlay, or ``None`` for normal play. Persistent: a
    #: pattern stays up until it is explicitly cleared, because the person using
    #: it is holding a projector with both hands and cannot keep re-tapping.
    projection_override: str | None = None
    #: One-shot request to blank the projector. Consumed and reset by the loop.
    #: A plain bool rather than an ``Event``: only the loop reads it and only the
    #: web layer sets it, and a lost race costs one frame of a stale overlay.
    blank_requested: bool = False

    # -- calibration wizard scratch space ----------------------------------
    #: corner name -> (camera_px, projector_px). A dict rather than a list so
    #: re-recording a corner overwrites it instead of adding a near-duplicate.
    calibration_points: dict[str, tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=dict
    )

    pending_stages: set[str] = field(default_factory=lambda: set(PENDING_STAGES))

    def __post_init__(self) -> None:
        # Built here rather than as default_factory to keep the import graph
        # shallow -- app.config must not depend on projection or modes.
        if self.tracker is None:
            from utils.performance import PerformanceTracker

            self.tracker = PerformanceTracker(target_fps=self.settings.system.target_fps)
        if self.mapper is None:
            from projection.mapper import init_projector_calibration

            self.mapper = init_projector_calibration(self.settings)
        # Recorded whether or not we built the mapper, so a mapper injected by a
        # test or by the calibration wizard reports its own provenance rather
        # than inheriting the default.
        self.calibration_source = (
            "file" if getattr(self.mapper.calibration, "is_calibrated", False) else "identity"
        )
        if self.tracker_balls is None:
            from vision.detection import BallTracker

            self.tracker_balls = BallTracker(self.settings)
        if self.prediction_cache is None:
            from physics.simulator import PredictionCache

            self.prediction_cache = PredictionCache()
        if self.mode_manager is None:
            from modes.mode_manager import ModeManager

            self.mode_manager = ModeManager(self.settings)
        if self.wizard_lock is None:
            from app.exclusive import ExclusiveLock

            self.wizard_lock = ExclusiveLock()
        if self.readiness is None:
            from app.readiness import ReadinessTracker

            self.readiness = ReadinessTracker(
                confirm_frames=self.settings.system.readiness_confirm_frames,
                min_confidence=self.settings.vision.table_min_confidence,
            )
        # The state machine reads ball motion from the tracker, and the modes
        # draw through the mapper. Neither is constructed by the manager: the
        # tracker is populated by detection and the mapper can be replaced at
        # runtime by the calibration wizard, so both belong to the loop.
        self.mode_manager.tracker = self.tracker_balls
        self.mode_manager.mapper = self.mapper

    # -- helpers ------------------------------------------------------------

    def detection_summary(self) -> dict[str, object]:
        """Counts for ``/api/status``.

        Tolerates a ``None`` game state, which is the state before the first
        frame and after a camera loss -- the panel needs to render either way.
        """
        gs = self.latest_game_state
        if gs is None:
            return {
                "balls": 0,
                "cue_ball_visible": False,
                "cue_stick_visible": False,
                "pockets": 0,
                "table_detected": self.table_boundary is not None,
                "confidence": 0.0,
            }
        return {
            "balls": len(gs.balls),
            "cue_ball_visible": gs.cue_ball is not None,
            "cue_stick_visible": gs.cue_stick is not None and gs.cue_stick.visible,
            "pockets": len(gs.pockets),
            "table_detected": gs.table_boundary is not None,
            "confidence": round(gs.confidence, 3),
        }

    def readiness_summary(self) -> dict[str, object]:
        """Current readiness, for ``/api/status`` and the panel."""
        return self.readiness.current().as_dict()

    def calibration_status(self) -> dict[str, object]:
        """What is calibrated, what looks stale, and which flow fixes it.

        One place, with the flow attached to each item, so the panel can put a
        button next to the problem. Previously "not calibrated" appeared in
        three places and none of them were actionable.
        """
        from app.calibration_status import assess
        from projection.mapper import load_calibration
        from vision.focus import load_focus_calibration

        return assess(
            load_calibration(),
            load_focus_calibration(),
            boundary=self.table_boundary,
            live_sharpness=self.live_sharpness,
            # The width the pipeline actually sees, not the capture width. With
            # a crop in force those differ, and the drift limit is a fraction of
            # it -- so the capture width would set a threshold generous by
            # exactly the zoom factor.
            frame_width=self.effective_frame_size()[0],
            current_crop=self.current_crop(),
            # The same reconciled status the Diagnostics card shows, so the two
            # cards cannot describe the lens differently.
            focus_status=self.focus_summary(),
        ).as_dict()

    def current_crop(self) -> list[int] | None:
        """The crop in force, ``[x, y, width, height]`` in sensor px, or ``None``.

        Read from the camera when there is one, because the camera clamps the
        configured rectangle against the frame it actually got -- so the request
        and the reality can differ, and this has to be the reality. Falls back to
        the configured value so the answer is still right before the first frame.
        """
        if not self.settings.camera.crop.enabled:
            return None
        rect = self.crop_rect()
        return [rect.x, rect.y, rect.width, rect.height]

    def crop_rect(self):
        """The crop in force as a :class:`vision.crop.CropRect`.

        From the camera when there is one, because the camera clamps the
        configured rectangle against the frame it actually received -- so the
        request and the reality can differ, and every geometric decision has to
        be made against the reality. Falls back to the configured value, then to
        the full frame, so this is always answerable.
        """
        from vision.crop import CropRect

        width, height = self.sensor_frame_size()
        configured = self.settings.camera.crop
        if configured.enabled and configured.width > 0 and configured.height > 0:
            return CropRect(
                x=configured.x, y=configured.y,
                width=configured.width, height=configured.height,
            ).clamped(width, height)
        return CropRect.full(width, height)

    def shift_detections(self, dx: float, dy: float) -> None:
        """Move the cached frame-space detections by ``(dx, dy)`` px.

        Used when the digital crop changes. A pure crop is exactly a translation
        in frame space, so this is a correction rather than an approximation: a
        point at frame ``(x, y)`` under the old crop is at ``(x + dx, y + dy)``
        under the new one.

        The homography is re-solved from the moved boundary rather than
        translated as a matrix. Composing a translation onto it would work, and
        would also be the sort of thing that is right until somebody changes the
        convention at one end; re-solving from the corners is the same operation
        the loop performs and cannot drift from it.

        Called with ``(0, 0)`` when only the crop's *size* changed, which is
        still a real change -- the cached boundary may now sit outside the frame
        -- but not one that moves any coordinate.
        """
        if self.table_boundary is None:
            return

        from app.models import TableBoundary, Vec2

        def moved(point) -> Vec2:
            return Vec2(point.x + dx, point.y + dy)

        old = self.table_boundary
        self.table_boundary = TableBoundary(
            top_left=moved(old.top_left),
            top_right=moved(old.top_right),
            bottom_right=moved(old.bottom_right),
            bottom_left=moved(old.bottom_left),
            center=moved(old.center),
            width_px=old.width_px,
            height_px=old.height_px,
            confidence=old.confidence,
            length_ft=old.length_ft,
            width_ft=old.width_ft,
            pixels_per_ft=old.pixels_per_ft,
            scale_source=old.scale_source,
        )

        for pocket in self.pockets or []:
            pocket.center_px = moved(pocket.center_px)

        from vision.calibration import CalibrationError, compute_perspective_transform

        try:
            self.camera_to_table, self.table_to_camera = compute_perspective_transform(
                self.table_boundary, self.settings
            )
        except CalibrationError as exc:
            # The moved corners no longer solve, which means the crop cut into
            # the cloth badly enough that the cached boundary is not usable.
            # Drop it and let the next detection pass find the table afresh --
            # keeping a boundary with no homography would leave every consumer
            # half-configured.
            logger.warning("crop moved the table out of solvable range (%s); re-detecting", exc)
            self.table_boundary = None
            self.camera_to_table = None
            self.table_to_camera = None
            self.pockets = []

    def sensor_frame_size(self) -> tuple[int, int]:
        """Full post-rotation frame size, ``(width, height)``.

        Measured from the camera when it has seen a frame, because a backend can
        hand back something other than what it was asked for -- the OpenCV path
        already logs when it does. Falls back to the configured size.
        """
        measured = getattr(self.camera, "sensor_size", None)
        if measured:
            return int(measured[0]), int(measured[1])
        return self.settings.camera.rotated_size

    def effective_frame_size(self) -> tuple[int, int]:
        """Frame size the pipeline sees, ``(width, height)``.

        Distinct from ``settings.camera.width/height``, which is what the sensor
        was asked to capture. Under a crop the two differ, and every consumer
        that means "the image I am working on" wants this one.
        """
        # From the crop rather than from ``latest_frame``, deliberately. The
        # cached frame is whatever the loop produced last, which for the first
        # frame after a re-frame is the *old* size -- so keying off it would make
        # the panel report the previous framing at the exact moment somebody
        # changed it. The crop is authoritative and immediate.
        rect = self.crop_rect()
        return rect.width, rect.height

    def focus_summary(self) -> dict[str, object]:
        """Lens focus state. The single source of truth, for every consumer.

        Two facts, and they are genuinely different: whether this rig has ever
        been focus-calibrated (a file on disk), and whether the lens is at that
        position right now (a readback from the motor). The Setup card and the
        Diagnostics card were each computing one of them independently and
        disagreeing in public -- "Camera focus: OK, set to 1280" beside "Lens
        focus: not calibrated", both true, neither complete.

        Reconciled here so both read the same thing. The interesting case is a
        calibration that exists but is not applied -- after a wizard run on a
        camera that has since been reopened, say -- which is now reported as
        exactly that rather than as either "fine" or "never calibrated".
        """
        from vision.focus import FocusStatus, load_focus_calibration

        camera = self.camera
        status = camera.focus if camera is not None else FocusStatus(detail="camera not open")
        summary = status.as_dict()

        calibration = load_focus_calibration()
        summary["saved"] = calibration.focus_absolute if calibration else None
        summary["saved_at"] = calibration.created_at if calibration else ""

        if calibration is not None and not status.calibrated:
            # A calibration exists that this camera never applied.
            summary["calibrated"] = True
            summary["source"] = "file"
            summary["ok"] = False
            summary["detail"] = (
                f"A saved focus of {calibration.focus_absolute} exists but the lens was not "
                "set to it this run. Restart, or re-run the camera focus step."
            )
        return summary

    def frame_age_ms(self) -> float | None:
        """Milliseconds since the last frame was captured, or ``None`` if none was.

        The panel's most useful single number for "is this display live?". FPS
        can read 30 from a rolling window that stopped updating a minute ago,
        because it is an average over the frames it did see; a frame age of
        60000 ms cannot be misread.

        Derived from ``GameState.timestamp``, which is ``perf_counter`` at
        capture -- deliberately not wall clock, so it is immune to NTP stepping
        the clock mid-game.
        """
        gs = self.latest_game_state
        if gs is None:
            return None
        return max(0.0, (time.perf_counter() - gs.timestamp) * 1000.0)

    def uptime_seconds(self) -> float:
        """Seconds since this state object was built."""
        return max(0.0, time.perf_counter() - self.started_at)

    def note_stage_error(self, stage: str, exc: BaseException) -> int:
        """Record an unexpected stage failure and return its running count."""
        count = self.stage_errors.get(stage, 0) + 1
        self.stage_errors[stage] = count
        self.last_error = f"{stage}: {exc}"
        return count

    def health_summary(self) -> dict[str, object]:
        """The long-run health block for ``/api/status`` and ``/health``.

        Everything here is a *cumulative* counter or a latched flag, on purpose.
        The rest of the status payload is instantaneous, which is the right
        shape for FPS and wrong for reliability: a camera that dropped out twice
        an hour ago and is fine now is exactly the thing a two-hour session
        needs to surface, and any instantaneous view of it reads "healthy".
        """
        return {
            "uptime_seconds": round(self.uptime_seconds(), 1),
            "started_at": self.started_wall,
            "frames_processed": self.frames_processed,
            "camera_reconnects": self.camera_reconnects,
            "stage_errors": dict(self.stage_errors),
            "disabled_stages": sorted(self.disabled_stages),
            "degradation_level": self.degradation_level,
            "loop_stalled": self.loop_stalled,
            "stall_count": self.stall_count,
            "calibration_source": self.calibration_source,
            "last_error": self.last_error,
        }

    def request_blank(self) -> None:
        """Ask the loop to blank the projector on its next pass.

        Also drops any test-pattern override, so "clear the projection" means
        what it says rather than blanking for one frame and then having the
        pattern reappear.
        """
        self.projection_override = None
        self.blank_requested = True

    def mark_stage_done(self, stage: str) -> None:
        """Remove a stage from the pending set once it is implemented."""
        self.pending_stages.discard(stage)

    def request_stop(self) -> None:
        """Ask the vision loop to exit at the end of the current frame."""
        logger.info("stop requested")
        self.stop_event.set()
        self.is_running = False
