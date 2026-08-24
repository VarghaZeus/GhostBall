"""Core domain objects shared by the vision, physics, projection and game layers.

Design note
-----------
These are plain :mod:`dataclasses`, not Pydantic models, even though the build
spec nominally called for Pydantic here. Every object in this module is
constructed on the hot path -- ``GameState`` and its ``Ball`` list are rebuilt
from scratch 30 times a second -- and Pydantic's per-field validation would
burn CPU we need for detection. Pydantic is still used where it earns its cost:
``app.config`` (validating user-edited YAML once at startup) and
``web.schemas`` (validating untrusted HTTP input).

Coordinate systems
------------------
Three distinct spaces are used throughout the codebase. Keeping them straight
is the single largest source of bugs in a projection-mapped system, so every
function that crosses a boundary names both spaces in its signature.

``camera px``
    Pixel coordinates in the raw camera frame. Origin top-left, +x right,
    +y down. Range is ``0..camera.width`` / ``0..camera.height``.
``table``
    Physical coordinates on the playing surface, in **inches**. Origin at the
    inside of the top-left cushion, +x along the long axis of the table,
    +y along the short axis. A 7 ft table is therefore ``0..76`` x ``0..38``.
    This is the space physics runs in -- the only space where distances are
    physically meaningful and independent of camera placement.
``projector px``
    Pixel coordinates in the frame sent out over HDMI. Origin top-left.
    Range is ``0..projector.width`` / ``0..projector.height``.

Conversions live in :mod:`vision.calibration` (camera <-> table) and
:mod:`projection.mapper` (table <-> projector). Nothing else should be doing
coordinate math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, NamedTuple

__all__ = [
    "Vec2",
    "BallKind",
    "BallColor",
    "PocketId",
    "GameModeName",
    "SessionState",
    "PhysicsAccuracy",
    "DrillType",
    "Severity",
    "TableBoundary",
    "Ball",
    "CueStick",
    "Pocket",
    "GameState",
    "CollisionResult",
    "ImpactEvent",
    "ShotPrediction",
    "ProjectorCalibration",
    "CalibrationState",
    "AlignmentError",
    "Player",
    "GameSession",
    "ModeOutput",
]


class Vec2(NamedTuple):
    """A 2-D point or vector.

    A ``NamedTuple`` rather than a dataclass: it unpacks like a tuple (so it
    drops straight into OpenCV calls that want ``(x, y)``), compares by value,
    and allocates far less than a regular object. Units depend on the field
    holding it -- see the module docstring.
    """

    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:  # type: ignore[override]
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def scaled(self, k: float) -> Vec2:
        return Vec2(self.x * k, self.y * k)

    def length(self) -> float:
        return (self.x * self.x + self.y * self.y) ** 0.5

    def distance_to(self, other: Vec2) -> float:
        return (self - other).length()

    def as_int(self) -> tuple[int, int]:
        """Round to integer pixels, for handing to OpenCV drawing calls."""
        return (int(round(self.x)), int(round(self.y)))


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BallKind(str, Enum):
    """What role a ball plays, independent of its colour."""

    CUE = "cue"
    SOLID = "solid"  # 1-7
    STRIPE = "stripe"  # 9-15
    EIGHT = "eight"  # 8
    UNKNOWN = "unknown"


class BallColor(str, Enum):
    """Detected colour band.

    Used by HSV detection and by the renderer, which tints each predicted path
    to match the ball it belongs to.
    """

    WHITE = "white"
    YELLOW = "yellow"
    BLUE = "blue"
    RED = "red"
    PURPLE = "purple"
    ORANGE = "orange"
    GREEN = "green"
    MAROON = "maroon"
    BLACK = "black"
    UNKNOWN = "unknown"


class PocketId(str, Enum):
    """The six pockets, named by position in table coordinates."""

    TOP_LEFT = "top_left"
    TOP_MIDDLE = "top_middle"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_MIDDLE = "bottom_middle"
    BOTTOM_RIGHT = "bottom_right"


class GameModeName(str, Enum):
    """Registered game modes. ``mode_manager.load_mode`` maps these to classes."""

    FREEPLAY = "freeplay"
    CLASSIC = "classic"
    KING_OF_THE_HILL = "king_of_the_hill"
    KNOCKOUT = "knockout"
    TRICK_SHOTS = "trick_shots"
    TRAINING = "training"
    CALIBRATION = "calibration"


class SessionState(str, Enum):
    """Game state machine. Transitions are owned by :mod:`modes.mode_manager`."""

    IDLE = "idle"  # table at rest, nobody aiming
    AIMING = "aiming"  # cue stick visible, showing prediction
    SHOT_IN_PROGRESS = "shot_in_progress"  # balls moving
    SETTLING = "settling"  # balls stopped, waiting out the settle timer
    GAME_OVER = "game_over"


class PhysicsAccuracy(str, Enum):
    """Speed/accuracy trade-off, surfaced as a control in the web panel.

    ``FAST`` stops after the first ball-to-ball impact; ``ACCURATE`` simulates
    every ball until the table settles. :mod:`physics.simulator` maps these to
    concrete step counts and collision depths.
    """

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class DrillType(str, Enum):
    POTTING = "potting"
    POSITION = "position"
    BANK_SHOT = "bank_shot"
    SAFETY = "safety"


Severity = Literal["info", "warning", "error"]


# ---------------------------------------------------------------------------
# Vision outputs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TableBoundary:
    """The playing surface as located in the camera frame.

    Corners are in **camera px** and describe the table as seen in the frame,
    not as physically labelled on the table. They are always stored in the
    clockwise order below so downstream homography code can rely on it.

    Note the axis naming, which is unfortunate and load-bearing: ``width_px`` is
    the **long** axis (top-left to top-right) and pairs with
    ``settings.table.length_in``, while ``height_px`` is the short axis and
    pairs with ``width_in``. The ``*_ft`` fields below use the *settings*
    convention -- ``length_ft`` is the long axis -- because those are physical
    table dimensions rather than image extents.
    """

    top_left: Vec2
    top_right: Vec2
    bottom_right: Vec2
    bottom_left: Vec2
    center: Vec2
    width_px: float  # long axis in px
    height_px: float  # short axis in px
    confidence: float = 0.0

    # -- measured real-world size, when detection could establish a scale -----
    #
    # All ``None`` when the table was found but not measured, which is the
    # normal case for felt-colour detection: it locates the cloth without ever
    # learning how big it is. Callers must treat ``None`` as "fall back to the
    # configured table size", never as zero.

    #: Long axis in feet, as measured. See :mod:`vision.pockets`.
    length_ft: float | None = None
    #: Short axis in feet.
    width_ft: float | None = None
    #: Camera px per foot over the cloth. The conversion factor that makes the
    #: measurement possible; see ``scale_source`` for where it came from.
    pixels_per_ft: float | None = None
    #: Which reference established the scale: ``ball``, ``config``,
    #: ``reference``, or ``""`` when unmeasured. Worth carrying all the way to
    #: the UI -- a size derived from ``reference`` is only as good as the camera
    #: being at the height that reference was taken at.
    scale_source: str = ""
    #: Which detector produced this boundary: ``pockets`` or ``felt``.
    detection_method: str = ""

    def corners(self) -> list[Vec2]:
        """Corners clockwise from top-left (camera px).

        This is the order ``cv2.getPerspectiveTransform`` expects to be paired
        against the canonical table rectangle.
        """
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    @property
    def is_measured(self) -> bool:
        """Whether real-world dimensions were established, not just assumed."""
        return self.length_ft is not None and self.pixels_per_ft is not None

    def length_in(self) -> float | None:
        """Measured long axis in inches, the unit the rest of the system uses."""
        return None if self.length_ft is None else self.length_ft * 12.0

    def width_in(self) -> float | None:
        """Measured short axis in inches."""
        return None if self.width_ft is None else self.width_ft * 12.0


@dataclass(slots=True)
class Ball:
    """A single detected ball.

    ``center_px`` is always populated by the detector. ``table_pos`` is filled
    in afterwards by :func:`vision.calibration.camera_to_table_coords` and stays
    ``None`` until the table homography is known -- physics must never be handed
    a ball whose ``table_pos`` is ``None``.
    """

    id: str  # stable across frames once tracking lands, e.g. "ball_08"
    center_px: Vec2
    radius_px: float
    color: BallColor = BallColor.UNKNOWN
    kind: BallKind = BallKind.UNKNOWN
    number: int | None = None  # 1-15 when the rack number is legible
    table_pos: Vec2 | None = None
    confidence: float = 0.0
    pocketed: bool = False

    @property
    def is_cue(self) -> bool:
        return self.kind is BallKind.CUE


@dataclass(slots=True)
class CueStick:
    """The cue as seen in the frame.

    ``angle_deg`` is the direction the cue points, measured in **table space**:
    0 deg is +x (towards the far short rail), increasing counter-clockwise.
    ``velocity`` is tip speed in inches/sec and is what the state machine
    thresholds on to decide a shot has been struck.
    """

    tip_px: Vec2
    angle_deg: float
    tip_table_pos: Vec2 | None = None
    shaft_visible: bool = False
    velocity: float = 0.0
    confidence: float = 0.0

    @property
    def visible(self) -> bool:
        """Whether the cue was detected well enough to aim from."""
        return self.confidence > 0.0


@dataclass(slots=True)
class Pocket:
    """One of the six pockets."""

    id: PocketId
    center_px: Vec2
    radius_px: float
    table_pos: Vec2 | None = None


@dataclass(slots=True)
class GameState:
    """Everything vision knows about the table for a single frame.

    This is the hand-off point out of the vision layer: physics, game modes and
    rendering all read from a ``GameState`` and never touch a raw frame.
    """

    timestamp: float  # time.perf_counter() at frame capture, NOT wall clock
    frame_index: int
    table_boundary: TableBoundary | None = None
    balls: list[Ball] = field(default_factory=list)
    cue_ball: Ball | None = None
    cue_stick: CueStick | None = None
    pockets: list[Pocket] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def is_usable(self) -> bool:
        """Whether there is enough here to run physics and render a prediction."""
        return (
            self.table_boundary is not None
            and self.cue_ball is not None
            and self.cue_ball.table_pos is not None
        )

    def object_balls(self) -> list[Ball]:
        """Every ball except the cue ball, still on the table."""
        return [b for b in self.balls if not b.is_cue and not b.pocketed]

    def ball_by_id(self, ball_id: str) -> Ball | None:
        return next((b for b in self.balls if b.id == ball_id), None)


# ---------------------------------------------------------------------------
# Physics outputs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CollisionResult:
    """Result of testing one moving ball against one stationary ball."""

    will_collide: bool
    point: Vec2 | None = None  # contact point, table coords
    time_to_impact: float = 0.0  # seconds
    normal_angle_deg: float = 0.0  # angle of the line of centres at contact


@dataclass(slots=True)
class ImpactEvent:
    """A collision that occurred during a simulated shot.

    ``target_id`` is ``None`` for cushion contacts, in which case
    ``is_cushion`` is ``True``.
    """

    position: Vec2  # table coords
    target_id: str | None
    incoming_angle_deg: float
    outgoing_angle_deg: float
    is_cushion: bool = False
    time_offset: float = 0.0  # seconds after the strike


@dataclass(slots=True)
class ShotPrediction:
    """Predicted outcome of a shot, in table coordinates throughout.

    ``trajectory_path`` is the cue ball's polyline. ``ball_paths`` holds the
    post-impact polyline for each object ball that gets moved, keyed by ball id
    -- the renderer draws these dimmer than the cue path.
    """

    trajectory_path: list[Vec2] = field(default_factory=list)
    ball_paths: dict[str, list[Vec2]] = field(default_factory=dict)
    impact_points: list[ImpactEvent] = field(default_factory=list)
    final_positions: dict[str, Vec2] = field(default_factory=dict)
    pocketed_ball_ids: list[str] = field(default_factory=list)
    time_to_settle: float = 0.0
    confidence: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.trajectory_path


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProjectorCalibration:
    """Table -> projector mapping, persisted to disk between runs.

    Two representations are kept deliberately. ``homography`` is the accurate
    3x3 transform solved from the four corner correspondences and is what
    :mod:`projection.mapper` actually uses. The affine fields (``offset_*``,
    ``scale_*``, ``rotation_deg``) are what the fine-tune screen nudges and what
    produces "move the projector left 3 inches" advice -- a human-legible
    summary, not the source of truth.
    """

    projector_width: int
    projector_height: int
    homography: list[list[float]] | None = None  # 3x3, table in -> projector px out
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation_deg: float = 0.0
    rmse_px: float = 0.0
    created_at: str = ""  # ISO-8601 wall clock, for display only
    is_calibrated: bool = False


@dataclass(slots=True)
class AlignmentError:
    """How far off the projection currently is, phrased for a human.

    ``message`` is shown verbatim in the calibration UI, so it should read as a
    physical instruction ("Move projector left 3 inches"), not a number dump.
    """

    total_rmse: float  # px
    x_offset: float = 0.0  # px
    y_offset: float = 0.0  # px
    rotation: float = 0.0  # degrees
    message: str = ""
    severity: Severity = "info"


@dataclass(slots=True)
class CalibrationState:
    """Progress through the seven-screen calibration wizard."""

    step: int = 1  # 1..7
    table_boundary: TableBoundary | None = None
    projector_offset: Vec2 = Vec2(0.0, 0.0)
    projector_scale: Vec2 = Vec2(1.0, 1.0)
    projector_rotation: float = 0.0
    corner_errors: list[float] = field(default_factory=list)  # per-corner px error
    grid_rotation_error: float = 0.0
    trajectory_test_error: float = 0.0
    is_complete: bool = False


# ---------------------------------------------------------------------------
# Game session
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Player:
    name: str
    score: int = 0
    shots_taken: int = 0
    shots_made: int = 0
    is_eliminated: bool = False

    @property
    def accuracy_pct(self) -> float:
        if self.shots_taken == 0:
            return 0.0
        return 100.0 * self.shots_made / self.shots_taken


@dataclass(slots=True)
class GameSession:
    """Mode-independent game bookkeeping.

    Mode-specific extras (challenge ids, combo counters) live here rather than
    inside each mode so the web panel can render a scoreboard without knowing
    which mode is active.
    """

    mode: GameModeName = GameModeName.FREEPLAY
    state: SessionState = SessionState.IDLE
    players: list[Player] = field(default_factory=list)
    current_player_index: int = 0
    started_at: float = 0.0  # time.perf_counter()
    current_challenge_id: int | None = None
    combo_count: int = 0
    balls_pocketed_this_turn: int = 0

    @property
    def current_player(self) -> Player | None:
        if not self.players:
            return None
        return self.players[self.current_player_index % len(self.players)]

    def advance_player(self) -> None:
        """Pass the turn, resetting per-turn counters."""
        if self.players:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.combo_count = 0
        self.balls_pocketed_this_turn = 0


@dataclass(slots=True)
class ModeOutput:
    """What a game mode returns each frame.

    ``overlay`` is typed as ``object | None`` rather than as an ndarray to keep
    this module import-light (no NumPy at import time). In practice it is an
    ``HxWx4`` uint8 RGBA array at projector resolution, or ``None`` when the
    mode has nothing to draw this frame.
    """

    overlay: object | None = None
    feedback_text: str = ""
    next_action: str = ""
    session: GameSession | None = None
