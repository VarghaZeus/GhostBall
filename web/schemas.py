"""Request and response models for the REST API.

Pydantic is the right tool here -- this is untrusted input arriving over HTTP,
validated once per request, and the validation cost is irrelevant next to the
network round trip. Contrast :mod:`app.models`, which is the hot path.

Response models mirror what the control panel needs rather than the internal
domain objects. Keeping them separate means the panel's contract does not move
every time an internal dataclass gains a field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models import DrillType, GameModeName, PhysicsAccuracy, SessionState

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class ModeRequest(BaseModel):
    """``POST /api/mode``"""

    mode: GameModeName
    #: Player names, for competitive modes. Ignored by freeplay and training.
    players: list[str] = Field(default_factory=list, max_length=8)


class SettingsRequest(BaseModel):
    """``POST /api/settings``

    Every field is optional so the panel can PATCH a single slider without
    resending the whole config -- and, more importantly, so that a panel built
    against an older version cannot blank out settings it does not know about.
    """

    brightness: int | None = Field(None, ge=0, le=100)
    overlay_alpha: int | None = Field(None, ge=0, le=100)
    trajectory_smoothing: int | None = Field(None, ge=0, le=100)
    physics_accuracy: PhysicsAccuracy | None = None
    auto_detect_cue: bool | None = None
    theme: str | None = Field(None, max_length=64)


class BrightnessRequest(BaseModel):
    """``POST /api/projector/brightness``"""

    value: int = Field(..., ge=0, le=100)


class DrillRequest(BaseModel):
    """``POST /api/training/start_drill``"""

    drill_type: DrillType = DrillType.POTTING


class DifficultyRequest(BaseModel):
    """``POST /api/mode/difficulty``"""

    difficulty: Literal["easy", "medium", "hard"] = "medium"


class ChallengeRequest(BaseModel):
    """``POST /api/mode/challenge``

    Zero-based position in the challenge list rather than the challenge's own
    ``id``. The panel shows them in order and the user picks the third one; it
    should not have to know that the third one is called 7 because two were
    deleted from the file.
    """

    index: int = Field(0, ge=0)


class CalibrationPointRequest(BaseModel):
    """``POST /api/calibration/corner/{corner}``

    Both coordinate pairs are required: the camera pixel where the physical
    corner appears, and the projector pixel the user aligned onto it. One
    without the other is not a correspondence and cannot contribute to the
    solve.
    """

    camera_px: tuple[float, float]
    projector_px: tuple[float, float]


class NudgeRequest(BaseModel):
    """``POST /api/calibration/nudge`` -- the fine-tune arrow controls."""

    dx: float = 0.0
    dy: float = 0.0
    dscale: float = 0.0
    drotation: float = 0.0


class PatternRequest(BaseModel):
    """``POST /api/projector/pattern``

    ``pattern`` is a :class:`projection.patterns.TestPattern` value, or ``None``
    to stop projecting one and hand the projector back to the active mode. It is
    typed as a plain string rather than the enum so that the 400 lists the valid
    names -- a Pydantic enum rejection produces a 422 whose body is harder to
    render usefully in the panel.
    """

    pattern: str | None = Field(None, max_length=32)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class PerfResponse(BaseModel):
    """Timing metrics. Percentiles as well as means, since an average that hides
    a per-shot stutter is not a useful health signal."""

    fps: float = 0.0
    frame_ms_avg: float = 0.0
    frame_ms_p95: float = 0.0
    latency_ms: float = 0.0
    dropped_frames: int = 0
    total_frames: int = 0
    stage_ms: dict[str, float] = Field(default_factory=dict)
    #: Age of the most recent frame in ms, or ``None`` before the first one.
    #:
    #: The panel's most trustworthy liveness signal, and the reason it is here
    #: rather than being inferred from ``fps``: FPS is a mean over a rolling
    #: window, so a loop that died a minute ago still reports 30. A frame age of
    #: 60000 cannot be misread.
    frame_age_ms: float | None = None


class DetectionCountsResponse(BaseModel):
    """What vision currently sees. The panel shows this so a user can tell
    "nothing is projected" (a render problem) from "nothing is detected" (a
    vision problem) without reading logs."""

    balls: int = 0
    cue_ball_visible: bool = False
    cue_stick_visible: bool = False
    pockets: int = 0
    table_detected: bool = False
    confidence: float = 0.0


class SystemResponse(BaseModel):
    """Host metrics. All optional -- ``psutil`` may be absent, and the SoC
    temperature sensor is not exposed on every platform."""

    cpu_pct: float | None = None
    mem_pct: float | None = None
    temp_c: float | None = None
    camera_backend: str = "none"
    display_backend: str = "none"
    #: True when frames are synthetic or output is discarded. Surfaced
    #: prominently, so nobody tunes detection thresholds against a mock table.
    using_mock_camera: bool = False
    using_mock_display: bool = False
    #: Configured capture and output geometry, as ``WxH`` plus the target rate.
    #: Shown because the commonest cause of "detection is bad" on new hardware is
    #: the camera not running at the resolution the thresholds were tuned for.
    camera_resolution: str = ""
    camera_target_fps: int = 0
    projector_resolution: str = ""
    #: Name of the test pattern currently overriding the projection, if any.
    projection_override: str | None = None


class HealthResponse(BaseModel):
    """Cumulative reliability counters, for a session measured in hours.

    Everything else in the status payload is instantaneous, which is the right
    shape for FPS and the wrong shape for reliability. A camera that dropped out
    twice an hour ago and is fine now is exactly what a long session needs to
    surface, and every instantaneous view of it reads "healthy" -- so these are
    counters and latched flags rather than current values.
    """

    uptime_seconds: float = 0.0
    #: Wall clock at start, ISO-8601. Display only; it can step.
    started_at: str = ""
    frames_processed: int = 0
    #: Camera losses recovered from. Nonzero alongside a healthy FPS is the
    #: signature of a marginal USB cable, which is otherwise near-impossible to
    #: diagnose from a projected overlay.
    camera_reconnects: int = 0
    #: Stage name -> unexpected exceptions seen. Cumulative for the run.
    stage_errors: dict[str, int] = Field(default_factory=dict)
    #: Stages switched off after failing on too many consecutive frames.
    disabled_stages: list[str] = Field(default_factory=list)
    #: 0 = full pipeline; higher means optional work is being shed to hold the
    #: frame rate.
    degradation_level: int = 0
    #: Whether the watchdog currently considers the loop wedged. Distinct from
    #: ``running: false``: a stalled loop is alive and producing nothing, which
    #: from outside looks exactly like a healthy idle one.
    loop_stalled: bool = False
    stall_count: int = 0
    #: ``file`` or ``identity``. An identity mapping projects a plausible
    #: overlay aligned to nothing, and that reads as a vision bug -- so the
    #: panel says which one is in force.
    calibration_source: str = "identity"
    last_error: str | None = None


class StatusResponse(BaseModel):
    """``GET /api/status`` -- everything the panel polls."""

    running: bool = False
    current_mode: GameModeName = GameModeName.FREEPLAY
    session_state: SessionState = SessionState.IDLE
    is_calibrated: bool = False
    calibration_rmse_px: float = 0.0
    last_shot_confidence: float = 0.0
    performance: PerfResponse = Field(default_factory=PerfResponse)
    detections: DetectionCountsResponse = Field(default_factory=DetectionCountsResponse)
    system: SystemResponse = Field(default_factory=SystemResponse)
    health: HealthResponse = Field(default_factory=HealthResponse)
    #: Names of pipeline stages that are still unimplemented. The panel shows
    #: these during the build-out so a blank projection is self-explaining
    #: rather than looking like a crash.
    pending_stages: list[str] = Field(default_factory=list)


class PlayerResponse(BaseModel):
    name: str
    score: int = 0
    shots_taken: int = 0
    accuracy_pct: float = 0.0
    is_eliminated: bool = False


class SessionResponse(BaseModel):
    """``GET /api/session`` -- the scoreboard."""

    mode: GameModeName = GameModeName.FREEPLAY
    state: SessionState = SessionState.IDLE
    players: list[PlayerResponse] = Field(default_factory=list)
    current_player: str | None = None
    combo_count: int = 0
    elapsed_seconds: float = 0.0


class CalibrationStatusResponse(BaseModel):
    """``GET /api/calibration/status``"""

    is_calibrated: bool = False
    table_detected: bool = False
    rmse_px: float = 0.0
    #: Derived from RMSE against the 20 px target: excellent / good / poor /
    #: uncalibrated. A word is more actionable to a user than a pixel count.
    alignment_quality: str = "uncalibrated"
    corners_recorded: int = 0
    created_at: str = ""


class CalibrationFinalizeResponse(BaseModel):
    """``POST /api/calibration/finalize``"""

    success: bool
    rmse_error_pixels: float = 0.0
    message: str = ""


class TrainingResultResponse(BaseModel):
    """``GET /api/training/result``"""

    has_result: bool = False
    success: bool = False
    accuracy_pct: float = 0.0
    stars: int = 0
    feedback: str = ""
    next_instruction: str = ""
    attempts: int = 0
    success_rate_pct: float = 0.0


class SettingsResponse(BaseModel):
    """Echo of the live settings, so the panel can render current slider values
    rather than guessing at defaults on load."""

    brightness: int
    overlay_alpha: int
    trajectory_smoothing: int
    physics_accuracy: PhysicsAccuracy
    theme: str
    target_fps: int
    table_preset: str
    auto_detect_cue: bool = True
    #: Every selectable theme, so the panel builds its selector from what the
    #: server actually supports. A hardcoded list in the HTML would silently go
    #: stale the moment a theme is added, and the panel is the only place a user
    #: can discover the names.
    available_themes: list[str] = Field(default_factory=list)
    #: Game modes that can actually be loaded. The panel greys out the rest --
    #: a button that reports success and silently gives you freeplay instead is
    #: worse than a button that is visibly unavailable.
    available_modes: list[GameModeName] = Field(default_factory=list)


class PatternResponse(BaseModel):
    """``POST /api/projector/pattern`` and ``GET /api/projector/patterns``."""

    active: str | None = None
    available: list[str] = Field(default_factory=list)
    message: str = ""


class ActionResponse(BaseModel):
    """Generic acknowledgement for endpoints with nothing else to return."""

    success: bool = True
    message: str = ""
