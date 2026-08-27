"""Settings, constants and table geometry.

Everything tunable on the Pi lives in ``config.yaml`` next to this package, so
that adjusting an HSV range or the overlay alpha at 11pm in a dim game room does
not require editing Python. This module loads that file, validates it with
Pydantic, and exposes a single process-wide :func:`get_settings` accessor.

Values are validated once at startup rather than per frame -- that is why
Pydantic is appropriate here but not in :mod:`app.models`.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.models import GameModeName, PhysicsAccuracy

logger = logging.getLogger(__name__)

# --- Paths -----------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.yaml"
DATA_DIR = PACKAGE_ROOT / "data"
CALIBRATION_DIR = DATA_DIR / "calibration"
CALIBRATION_FILE = CALIBRATION_DIR / "projector_calibration.json"
LOG_DIR = DATA_DIR / "logs"

# --- Physical constants ----------------------------------------------------
# Regulation dimensions, inches. These are physical facts, not preferences, so
# they are constants rather than config: a 7 ft table has a 76x38 in playing
# surface measured cushion-nose to cushion-nose.

BALL_DIAMETER_IN = 2.25
BALL_RADIUS_IN = BALL_DIAMETER_IN / 2.0

#: Coefficient of restitution for a ball rebounding off a cushion. Real cloth
#: cushions land around 0.75-0.85; the spec's 0.9 is optimistic. Tune in YAML.
DEFAULT_CUSHION_RESTITUTION = 0.80

#: Effective deceleration of a struck ball on cloth, inches/sec^2.
#:
#: NOT the textbook rolling-resistance figure. The standard ~0.01 rolling
#: coefficient gives 3.86 in/s^2, and that value is correct for a ball which is
#: already rolling -- but a struck ball spends its fastest phase *sliding*, where
#: the coefficient is ~0.2 and the deceleration is roughly 77 in/s^2. A
#: single-constant model has to be an effective blend of the two.
#:
#: Calibrated against settle time, which is the observable a person can actually
#: judge. Measured with this simulator on a 7 ft table:
#:
#:     3.86 in/s^2  ->  9.0-10.5 s to settle   (far too long)
#:    12.0  in/s^2  ->  4.1-5.6 s              <- chosen
#:    25.0  in/s^2  ->  2.4-3.7 s              (too brisk for a firm shot)
#:
#: Real shots settle in roughly 3-8 s, so 12 sits inside the plausible band and
#: errs toward longer paths -- the safer direction for an aiming aid, since
#: over-predicting travel is less misleading than showing a ball stopping short.
#:
#: This is the single most valuable parameter to calibrate against a real table:
#: roll a ball a measured distance, time it, and solve a = s^2/(2d).
DEFAULT_ROLLING_FRICTION = 12.0

#: Ball-to-ball coefficient of restitution. Phenolic resin balls are close to
#: elastic.
DEFAULT_BALL_RESTITUTION = 0.95


class TableSize(BaseModel):
    """Playing-surface dimensions in inches (cushion nose to cushion nose)."""

    length_in: float = Field(76.0, gt=0, description="Long axis, table +x")
    width_in: float = Field(38.0, gt=0, description="Short axis, table +y")
    pocket_radius_in: float = Field(2.25, gt=0)

    @field_validator("width_in")
    @classmethod
    def _width_below_length(cls, v: float, info: Any) -> float:
        length = info.data.get("length_in")
        if length is not None and v >= length:
            raise ValueError(
                f"width_in ({v}) must be less than length_in ({length}); "
                "table +x is the long axis by convention"
            )
        return v


#: Standard playing-surface lengths in feet, for reporting the nearest match to
#: a measured table and for the optional snap. Every regulation table is 2:1, so
#: the short axis is always half of these.
#:
#: Not a constraint on what can be detected -- measurement is what the pocket
#: pipeline reports, and a table that is genuinely 8.2 ft comes back as 8.2 ft.
#: This list only names the ones that have names.
STANDARD_TABLE_LENGTHS_FT: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0)


def nearest_standard_length_ft(length_ft: float) -> float:
    """The named table size closest to a measurement."""
    return min(STANDARD_TABLE_LENGTHS_FT, key=lambda standard: abs(standard - length_ft))


#: Named presets for the common table sizes, selectable by ``table.preset``.
TABLE_PRESETS: dict[str, TableSize] = {
    "7ft": TableSize(length_in=76.0, width_in=38.0),
    "8ft": TableSize(length_in=88.0, width_in=44.0),
    "8ft_pro": TableSize(length_in=92.0, width_in=46.0),
    "9ft": TableSize(length_in=100.0, width_in=50.0),
}


class CropSettings(BaseModel):
    """Digital crop applied to every captured frame. See :mod:`vision.crop`.

    Coordinates are **sensor space**: the full captured frame *after* rotation,
    which is the frame the preview shows and the one the panel's controls work
    in. They are deliberately not a fraction of the frame -- a pixel rectangle
    means the same thing after a resolution change, and a fraction silently
    would not.

    ``enabled`` rather than treating a full-frame rectangle as off, because
    "never cropped" and "cropped back out to the full frame" want different
    words on the panel and different behaviour on save.
    """

    enabled: bool = False
    x: int = Field(0, ge=0)
    y: int = Field(0, ge=0)
    width: int = Field(0, ge=0)
    height: int = Field(0, ge=0)


class CameraSettings(BaseModel):
    """Arducam 16MP (IMX519) capture settings.

    The sensor can do 4656x3496, but detection runs on every frame and we need
    30 FPS end to end, so we capture at 1080p and let the ISP downscale. Going
    higher buys sub-pixel ball accuracy we cannot use and costs frame time we
    cannot spare.
    """

    width: int = Field(1920, gt=0)
    height: int = Field(1080, gt=0)
    fps: int = Field(30, gt=0, le=120)
    #: Hand-set lens position override, in the focus motor's raw units, or
    #: ``None`` to use the calibrated value from
    #: ``data/calibration/focus.json``.
    #:
    #: The units are the *driver's*, and their range is a property of the motor:
    #: 0-4095 on an ak7375 (IMX519), 0-1023 on a dw9807 (IMX708). So the bound
    #: here cannot be the real one -- it is a sanity check against a typo, and
    #: the authoritative clamp happens in :func:`vision.focus.apply_focus`
    #: against the range the driver actually reports. It read ``le=4095``, which
    #: was the ak7375's maximum masquerading as a universal limit.
    #:
    #: ``None`` by default, and that default is load-bearing. With a number here
    #: there is always *a* value, so "this rig has never been focus-calibrated"
    #: is not representable -- and every decision about whether to prompt for a
    #: calibration run depends on being able to tell those apart. The calibrated
    #: value lives in a file because software writes it and this file is full of
    #: human comments.
    #:
    #: Driven over V4L2, not libcamera. On the IMX519 that is because the stock
    #: tuning file binds no AF algorithm, so ``AfMode``/``LensPosition`` are
    #: dropped with an internal warning and no exception. On a sensor whose
    #: tuning *does* bind one, driving V4L2 directly is still right, but the AF
    #: algorithm has to be put in manual mode or it fights us for the same VCM.
    #: See :mod:`vision.focus`.
    focus_absolute: int | None = Field(None, ge=0, le=65535)
    #: Hand-set lens position for a camera driven through **libcamera**, in
    #: dioptres (reciprocal metres: 0.0 is infinity, 2.0 is half a metre).
    #:
    #: A separate key from ``focus_absolute`` on purpose. One key would have to
    #: mean raw counts on an IMX519 and dioptres on an IMX708, and a number whose
    #: unit depends on which camera is plugged in is precisely the ambiguity
    #: worth spending a config key to avoid. Only the one matching this rig's
    #: control is read -- see :func:`vision.focus.resolve_focus_value`.
    focus_dioptres: float | None = Field(None, ge=0.0, le=32.0)
    #: Which focus control to use: ``auto``, ``libcamera`` or ``v4l2``.
    #:
    #: ``auto`` decides by capability -- whether libcamera advertises ``AfMode``
    #: for this sensor -- and is the right mechanism, because using the wrong
    #: path does not degrade focus, it makes the lens undrivable.
    #:
    #: The overrides are a diagnostic escape hatch, not a tuning knob. They exist
    #: because auto-detection is a claim about the running system that a person
    #: can check independently (``python -m tools.focus_probe``), and when the two
    #: disagree somebody needs to be able to proceed. Whichever is chosen, the
    #: startup log says which and why.
    focus_path: str = Field("auto", pattern="^(auto|libcamera|v4l2)$")
    #: Dioptre band the focus sweep covers, ``[low, high]``. Only used on the
    #: libcamera path; ignored for raw counts, which have no physical meaning.
    #:
    #: Dioptres are 1/metres, so the default 0.3-1.5 is 3.33 m down to 0.67 m --
    #: every plausible height for a camera over a pool table. The control's own
    #: range is 0-32, i.e. infinity to 3 cm, and sweeping all of it put 30 of 33
    #: samples at distances nothing will ever be mounted at while crossing the
    #: only band that matters in three jumps.
    #:
    #: Widen it if a sweep reports its peak at either end -- the message says
    #: which end and gives the distance in metres.
    focus_sweep_dioptres: tuple[float, float] = (0.3, 1.5)
    #: Driver name of the focus motor, matched as a fragment against
    #: ``/sys/class/video4linux/*/name``. Resolved by name because the subdev
    #: index is not stable across reboots.
    lens_driver: str = Field("ak7375")
    #: Skip focus entirely. For a fixed-focus lens, where every startup would
    #: otherwise log an error about a motor that is legitimately absent.
    focus_enabled: bool = True
    #: Seconds to wait after opening the camera before trusting a frame. The
    #: IMX519 needs a moment for AE/AWB to converge or early frames are dark.
    warmup_seconds: float = Field(2.0, ge=0.0)
    #: Rotate frames 0/90/180/270 degrees if the camera is mounted sideways.
    rotation_deg: int = Field(0)
    #: Force the mock camera even on a Pi. Lets the pipeline be exercised
    #: without hardware; also the automatic fallback when picamera2 is missing.
    use_mock: bool = False
    #: Digital crop. Written by the panel to ``data/calibration/crop.json``,
    #: which wins over whatever is here -- see :func:`vision.crop_store.load`.
    #: This field is the hand-editable default for a rig with no saved crop.
    crop: CropSettings = Field(default_factory=CropSettings)

    @property
    def rotated_size(self) -> tuple[int, int]:
        """Capture size after rotation -- i.e. sensor space. ``(width, height)``.

        90 and 270 transpose the frame, and the crop is applied after rotation,
        so a crop validated against the *pre*-rotation size would be wrong on a
        sideways-mounted camera in exactly the way that is hardest to spot: it
        would still be a valid rectangle, just not the one anyone chose.
        """
        if self.rotation_deg in (90, 270):
            return self.height, self.width
        return self.width, self.height

    @property
    def crop_scale(self) -> float:
        """Sensor width divided by cropped width. 1.0 when not cropping.

        The multiplier for anything expressed as a *fraction of frame width*.
        Such quantities are invariant under a resize and are **not** invariant
        under a crop -- cropping to half the width doubles how much of the frame
        any given object spans. ``vision.pocket_radius_frac_range`` is the one
        setting in this file that is stated that way, and it is why this exists.
        """
        if not self.crop.enabled or self.crop.width <= 0:
            return 1.0
        sensor_width = self.rotated_size[0]
        if sensor_width <= 0:
            return 1.0
        return sensor_width / float(self.crop.width)

    @field_validator("rotation_deg")
    @classmethod
    def _valid_rotation(cls, v: int) -> int:
        if v not in (0, 90, 180, 270):
            raise ValueError("rotation_deg must be one of 0, 90, 180, 270")
        return v


class ProjectorSettings(BaseModel):
    """GoodDee projector output settings.

    Note the native-vs-output distinction: the projector accepts a 4K signal but
    rendering a 3840x2160 RGBA overlay every frame is ~33 MB of pixel writes and
    will not hold 30 FPS on a Pi 5. We render at 1080p and let the projector
    upscale; the overlay is line art, so the loss is invisible.
    """

    width: int = Field(1920, gt=0, description="Render + output resolution")
    height: int = Field(1080, gt=0)
    #: Which display to open full-screen on. ``0`` is the primary output.
    display_index: int = Field(0, ge=0)
    #: Global overlay opacity, 0-100. Below ~40 the felt washes it out; above
    #: ~80 it starts to obscure the balls.
    overlay_alpha_pct: int = Field(65, ge=0, le=100)
    brightness_pct: int = Field(80, ge=0, le=100)
    #: Seconds of lamp warm-up before colours are stable enough to calibrate.
    warmup_seconds: float = Field(120.0, ge=0.0)
    #: Skip opening a real display; render frames but discard them. Used on the
    #: dev box and in tests.
    use_mock: bool = False


class VisionSettings(BaseModel):
    """Detection thresholds.

    The HSV ranges are the values most likely to need tuning per table and per
    room, which is exactly why they live in YAML. Hue is on OpenCV's 0-179
    scale, not 0-359.
    """

    #: Green felt hue range, used to segment the playing surface.
    felt_hue_range: tuple[int, int] = (35, 85)
    felt_sat_min: int = Field(40, ge=0, le=255)
    #: Saturation *ceiling* for felt. This is what lets a green ball be told
    #: apart from green cloth -- hue cannot do it, because the 6 ball's green
    #: falls inside any felt hue range wide enough for real cloth. Felt is matte
    #: wool; a ball is glossy resin and reads as more saturated. Tune it with
    #: `tools.camera_preview --mask` and watch that the 6 ball stays visible.
    felt_sat_max: int = Field(200, ge=0, le=255)
    felt_val_min: int = Field(40, ge=0, le=255)

    #: Minimum confidence for a detected table to be believed.
    #:
    #: Presence is not enough. Pointed at anything that is not a pool table,
    #: felt segmentation and pocket detection both return *something*
    #: occasionally, and a system that declares itself ready on that is worse
    #: than one that admits it cannot see a table.
    #:
    #: 0.75, and the number is not arbitrary. Pocket detection scores
    #: ``0.55 * (pockets found / 6) + 0.45 * aspect_score``, so six dark blobs
    #: anywhere in frame score 0.55 **on their own** -- a ceiling has plenty of
    #: light fittings and vents. Any threshold at or below 0.55 can therefore be
    #: cleared by blob count alone, with the aspect ratio contributing nothing,
    #: which is exactly how a ceiling was reported as a table at 41%.
    #:
    #: The rule this encodes: the threshold must sit above what any single term
    #: can produce, so both have to agree. At 0.75 a full set of pockets still
    #: needs the quad's aspect ratio to land within about half the configured
    #: tolerance. A real table measures 0.97.
    table_min_confidence: float = Field(0.75, ge=0.0, le=1.0)

    #: Width in px that detection is downscaled to before any processing.
    #:
    #: This is the single most important performance knob in the system. Felt
    #: segmentation alone measures ~18 ms at 1920 px wide, which is over half a
    #: 33 ms frame budget before a single ball has been found. Cost scales with
    #: pixel count, so 960 px is 4x cheaper and still leaves a 2.25 in ball
    #: about 12 px across -- comfortably enough to locate its centre to better
    #: than a tenth of an inch once scaled back up.
    #:
    #: Raise it only if small-ball detection is genuinely failing; lower it to
    #: 720 to buy frame time on a loaded Pi.
    detection_width: int = Field(960, ge=320, le=3840)

    #: Width that *table* detection downscales to, separate from ball detection.
    #:
    #: A table is an enormous object -- metre-long straight rails -- so locating
    #: it needs far less resolution than finding a 2.25 in ball. Measured corner
    #: error is 3.1 px at 640 wide versus 1.5 px at 960, both comfortably inside
    #: the 20 px alignment budget, and 640 costs 4.2 ms against 5.5 ms.
    #:
    #: The reason it is a separate knob: table detection runs on an interval, so
    #: its cost lands as a periodic spike rather than being amortised. Observed
    #: at ~57 ms in a live run, which is a visible hitch every few seconds even
    #: though the average contribution is under a millisecond.
    table_detection_width: int = Field(640, ge=320, le=3840)

    #: Expected ball radius in camera px, as a search window. Recomputed from
    #: the table homography once calibrated -- these are only the cold-start
    #: bounds, and they are in *full-resolution* px.
    ball_radius_px_range: tuple[int, int] = (12, 40)
    hough_dp: float = Field(1.2, gt=0)
    hough_param1: int = Field(100, gt=0)  # Canny high threshold
    hough_param2: int = Field(18, gt=0)  # accumulator threshold
    min_ball_confidence: float = Field(0.4, ge=0.0, le=1.0)

    #: How round a blob must be to be a ball: ``4*pi*area / perimeter^2``, where
    #: 1.0 is a perfect circle. This is the main shadow rejector -- a ball plus
    #: its attached shadow is markedly less circular than the ball alone.
    min_ball_circularity: float = Field(0.62, ge=0.0, le=1.0)
    #: Fractional tolerance on expected ball area. A blob within +/-55% of the
    #: area implied by the table scale is plausible; outside that it is a
    #: cluster of touching balls, or a chalk cube, or noise.
    ball_area_tolerance: float = Field(0.55, gt=0.0, le=1.0)

    #: Cue ball discrimination. It is the least saturated, brightest ball on the
    #: table -- these bound what "white" means before the comparison is made
    #: relative to the other balls actually present.
    cue_ball_max_saturation: int = Field(90, ge=0, le=255)
    cue_ball_min_value: int = Field(150, ge=0, le=255)

    #: Table acceptance checks. The detected quad must cover at least this
    #: fraction of the frame, or the felt mask has found something that is not
    #: the table.
    table_min_area_pct: float = Field(8.0, gt=0.0, le=100.0)
    #: Fractional tolerance on the table's expected length:width aspect ratio.
    #: Generous, because perspective from an off-centre camera mount genuinely
    #: distorts it.
    table_aspect_tolerance: float = Field(0.35, gt=0.0, le=1.0)

    #: Cue detection. The stick is found as a long thin line; these bound what
    #: counts as "long" in *full-resolution* camera px.
    cue_min_line_length_px: int = Field(150, gt=0)
    cue_max_line_gap_px: int = Field(20, ge=0)
    min_cue_confidence: float = Field(0.5, ge=0.0, le=1.0)
    #: Degrees within which two line segments count as collinear and get merged.
    #: The shaft is nearly always broken into pieces by the bridge hand.
    cue_merge_angle_deg: float = Field(8.0, gt=0.0, le=45.0)

    #: Inches/sec of cue-tip motion that counts as a strike. Below this the
    #: player is still aiming.
    strike_velocity_threshold: float = Field(20.0, gt=0)
    #: Inches/sec below which a ball counts as stopped.
    ball_stopped_threshold: float = Field(0.5, gt=0)

    #: Frames of history used to smooth ball positions. Higher is steadier but
    #: adds latency: at 30 FPS each frame is 33 ms.
    tracking_history_frames: int = Field(3, ge=1, le=10)

    #: Use the Hailo AI HAT+ for detection when the runtime is present.
    #: Always falls back to OpenCV, so leaving this on is safe.
    use_hailo: bool = True
    hailo_model_path: str = "models/pool_balls.hef"

    # -- Table detection method ---------------------------------------------

    #: How to find the table.
    #:
    #: ``pockets``
    #:     Locate the six pocket mouths and reconstruct the rails from them.
    #:     Independent of felt colour, so it works on red, blue, burgundy or
    #:     black cloth where the felt thresholds above find nothing.
    #: ``felt``
    #:     Segment the cloth by colour. Faster, and better on a table whose
    #:     pockets are occluded or unusually shallow.
    #: ``auto``
    #:     Try pockets first and fall back to felt. The default, because pocket
    #:     detection needs six visible mouths and felt detection does not.
    table_detection_method: Literal["auto", "pockets", "felt"] = "auto"

    #: How many times the pocket finder splits the dark end of the histogram.
    #:
    #: Each level is an Otsu threshold taken over the pixels below the previous
    #: one, so the ladder walks felt -> surround -> rail -> pocket without
    #: anybody saying in advance how dark a pocket is. Four is enough for every
    #: frame tested; more costs about 1 ms each and finds nothing new.
    #:
    #: A fixed percentile was tried first and does not work. "The darkest 1.5%
    #: of the frame" is the pockets on a 7 ft table filling the frame and is the
    #: *rails* on a 9 ft table or a high camera, because the pockets no longer
    #: cover 1.5% of anything -- so detection failed on exactly the tables the
    #: retrofit was for. Otsu splits by pixel value rather than by area.
    pocket_threshold_depth: int = Field(4, ge=1, le=8)
    #: Roundness gate for a pocket mouth, 4*pi*area/perimeter^2. Looser than the
    #: ball gate: a corner pocket is a rounded wedge, not a disc.
    pocket_min_circularity: float = Field(0.45, gt=0.0, le=1.0)
    #: How much brighter the ring around a pocket must be than the pocket
    #: itself, as a *fraction* of the ring's own brightness. This is what
    #: separates a pocket -- a dark spot inside a brighter field -- from the dark
    #: surround beyond the rails, which has nothing bright around it.
    #:
    #: A fraction rather than grey levels, because an absolute threshold is a
    #: vignetting detector. Room lighting falls off toward the frame corners,
    #: which is exactly where the corner pockets are: at 30% falloff their
    #: absolute contrast measured 17 grey levels against an 18-level threshold,
    #: and four of six pockets were rejected on a frame where all six were
    #: plainly visible. Relative contrast is unchanged by a brightness scale.
    pocket_min_contrast_frac: float = Field(0.35, ge=0.0, le=1.0)
    #: Loose first-pass pocket radius bounds, as a fraction of frame width.
    #: Deliberately wide -- the point of the first pass is to work without
    #: knowing the table size, and the second pass tightens these from what it
    #: measured.
    pocket_radius_frac_range: tuple[float, float] = (0.004, 0.075)

    # -- Real-world scale ----------------------------------------------------

    #: Where the pixels-per-inch figure comes from. See
    #: :func:`vision.pockets.resolve_scale` -- this is the one number that turns
    #: a picture of a table into a measurement of one, and it cannot be guessed
    #: from the pixels alone.
    #:
    #: ``ball``
    #:     From the imaged size of a ball, whose diameter is a known constant.
    #:     The only source that is independent of camera height. Needs at least
    #:     one ball on the cloth.
    #: ``config``
    #:     Assume the table is the size ``table.length_in`` says it is.
    #: ``reference``
    #:     The fixed camera-height reference below. A last resort.
    #: ``auto``
    #:     ball, then config, then reference. The default.
    scale_source: Literal["auto", "ball", "config", "reference"] = "auto"

    #: Fallback calibration for ``scale_source: reference``: a table of
    #: ``reference_table_length_ft`` measured this many px across the long axis.
    #:
    #: This is only valid at the camera height it was measured at. Raising or
    #: lowering the camera changes the pixel width of an unchanged table, so this
    #: reports a different table size for the same table -- measured at 6.45 ft,
    #: 5.11 ft and 2.96 ft for one 6.33 ft table at three heights. Prefer
    #: ``ball``, which has no such dependency.
    reference_table_width_px: float = Field(2000.0, gt=0)
    reference_table_length_ft: float = Field(7.0, gt=0)

    #: Round a measured table to the nearest standard size when it is within
    #: this fraction of one. Off by default: the point of measuring is to
    #: support the sizes nobody standardised, and a table that really is 8.2 ft
    #: should be reported as 8.2 ft.
    snap_to_standard_size: bool = False
    snap_tolerance_frac: float = Field(0.03, gt=0.0, lt=0.25)

    #: Fall back to an adaptive cloth mask -- "whatever colour most of the table
    #: is" -- when the configured felt thresholds match less of the table
    #: interior than this. Without it, pocket-based detection succeeds on red
    #: cloth and then ball detection finds nothing, which is a worse failure
    #: than not detecting the table at all because it looks like it is working.
    adaptive_cloth_min_coverage: float = Field(0.35, ge=0.0, le=1.0)

    #: Let a measured table size replace ``table.length_in`` at runtime.
    #:
    #: **Off by default, and the reason is the measurement's accuracy.** A
    #: ball-derived size lands within about 2% when the ball images large and
    #: drifts to 8% when the camera is high and the ball is a dozen pixels
    #: across. Adjacent standard tables are 7% apart -- 7.0 ft against 7.5 ft --
    #: so the measurement cannot reliably tell them apart, and a wrong automatic
    #: resize silently rescales every physics prediction with nothing on screen
    #: to say so. The measurement is logged and carried on the boundary either
    #: way; setting ``table_preset`` from it is a decision for a person.
    #:
    #: Worth turning on when the camera is low, several balls are always on the
    #: cloth, and the table is genuinely not a standard size.
    adopt_measured_table_size: bool = False
    #: Fractional disagreement with the configured size before adoption fires,
    #: when it is enabled at all. Wide enough that noise cannot trigger it.
    adopt_table_size_tolerance: float = Field(0.10, gt=0.0, lt=1.0)


class PowerBucket(BaseModel):
    """One named power level, defined by how far it free-rolls the cue ball.

    Defined as a *distance* rather than as a value on the 0-100 power scale on
    purpose. Power is a UI abstraction whose mapping to speed is arbitrary;
    distance is a thing a person can pace out on the cloth and check. It also
    makes the level survive a friction retune -- recalibrating the cloth moves
    the power that produces "two table lengths" without moving what the label
    promises.

    Expressed in table lengths rather than inches so a level authored on a 7 ft
    table means the same thing on a 9 ft one, for the same reason
    ``challenges.json`` holds table inches instead of pixels.
    """

    #: Projected verbatim under the tick, so keep it to one or two short words.
    name: str
    table_lengths: float = Field(gt=0.0)


class PhysicsSettings(BaseModel):
    """Simulation parameters. Defaults come from the constants above."""

    accuracy: PhysicsAccuracy = PhysicsAccuracy.BALANCED
    cushion_restitution: float = Field(DEFAULT_CUSHION_RESTITUTION, gt=0.0, le=1.0)
    ball_restitution: float = Field(DEFAULT_BALL_RESTITUTION, gt=0.0, le=1.0)
    rolling_friction: float = Field(DEFAULT_ROLLING_FRICTION, gt=0.0)
    #: Integration step, seconds. 4 ms keeps a hard-struck ball (~250 in/s)
    #: moving under 1 inch per step, well below a ball radius, so collisions
    #: are not tunnelled through.
    timestep: float = Field(0.004, gt=0.0, le=0.05)
    #: Hard cap on simulated time so a pathological shot cannot stall a frame.
    max_sim_seconds: float = Field(8.0, gt=0.0)
    #: How many ball-to-ball impacts to follow. Ignored in FAST (always 1).
    max_collision_depth: int = Field(3, ge=1, le=10)
    #: Default power when it cannot be estimated from the cue, 0-100.
    #:
    #: **Do not read this as a plausible shot.** On the ``power_to_velocity``
    #: scale, 50 is 160 in/s, which at the default friction free-rolls 13 table
    #: lengths -- no shot can produce it. It survives only as the input to an
    #: aiming *line*, whose direction is what the value is used for. Anything
    #: that cares how far the ball travels must use :attr:`power_buckets`
    #: instead; see :func:`physics.models.power_for_table_lengths`.
    default_power: int = Field(50, ge=0, le=100)

    #: The five power levels offered to the player, as cue-ball **free-roll**
    #: distance in table lengths. Free roll, not "travel on a straight shot":
    #: a straight shot stuns the cue ball dead at contact, so that anchor
    #: measures zero.
    #:
    #: Spaced evenly in *speed* rather than in distance, because speed is what
    #: the player's arm controls and distance goes as its square -- even
    #: distance steps would bunch up at the hard end and read as three levels
    #: rather than five.
    #:
    #: TUNE, and only after ``rolling_friction`` -- these are distances, so they
    #: mean nothing until the deceleration they are converted through has been
    #: measured on the actual cloth. Fast cloth puts "medium" further down the
    #: table than slow cloth does.
    #:
    #: Note what is *not* here: the break. On this scale a break would sit at
    #: 20-plus table lengths, and stretching the five levels to reach it would
    #: compress every positional shot into the first tick. Break power is a
    #: separate thing if it is ever wanted.
    power_buckets: list[PowerBucket] = Field(
        default_factory=lambda: [
            PowerBucket(name="very soft", table_lengths=0.5),
            PowerBucket(name="soft", table_lengths=1.0),
            PowerBucket(name="medium", table_lengths=2.0),
            PowerBucket(name="strong", table_lengths=3.0),
            PowerBucket(name="very hard", table_lengths=4.5),
        ]
    )
    #: Which bucket to assume when nothing has prescribed one. Index into
    #: :attr:`power_buckets`; the middle level by default.
    default_bucket_index: int = Field(2, ge=0)


class RenderSettings(BaseModel):
    """Overlay appearance. Colours are RGB, 0-255."""

    cue_path_color: tuple[int, int, int] = (80, 255, 120)
    object_path_color: tuple[int, int, int] = (150, 220, 255)
    impact_color: tuple[int, int, int] = (255, 220, 80)
    ghost_ball_color: tuple[int, int, int] = (220, 220, 220)
    pocket_highlight_color: tuple[int, int, int] = (255, 120, 200)
    text_color: tuple[int, int, int] = (255, 255, 255)

    line_thickness_px: int = Field(4, gt=0)
    #: 0-100. Exponential smoothing on the rendered trajectory so the overlay
    #: does not jitter with per-frame detection noise. 0 disables smoothing.
    trajectory_smoothing_pct: int = Field(60, ge=0, le=100)
    show_ghost_balls: bool = True
    show_impact_angles: bool = True
    theme: str = Field("classic", description="Named theme in the animation system")
    #: Seconds an effect such as a pocket burst stays on screen.
    effect_duration_seconds: float = Field(1.2, gt=0)


class WebSettings(BaseModel):
    """FastAPI control panel."""

    host: str = "0.0.0.0"
    port: int = Field(8000, gt=0, lt=65536)
    #: Reload on source change. Never enable on the Pi in production -- the
    #: watcher competes with the vision loop for CPU.
    reload: bool = False
    #: Origins allowed to call the API. Left permissive because this runs on a
    #: LAN with no auth; do not expose the port to the internet.
    cors_origins: list[str] = ["*"]


class SystemSettings(BaseModel):
    """Loop timing and logging."""

    target_fps: int = Field(30, gt=0, le=120)
    #: Warn when camera-to-projector latency exceeds this. The stated goal is
    #: under 100 ms.
    latency_warn_ms: float = Field(100.0, gt=0)
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_to_file: bool = True
    #: Seconds between performance summary log lines.
    perf_log_interval_seconds: float = Field(10.0, gt=0)
    #: Seconds between host-health lines (CPU, memory, SoC temperature). Much
    #: rarer than the perf line: these move slowly, and reading them costs a
    #: trip through /proc that the frame budget does not need every 10 seconds.
    health_log_interval_seconds: float = Field(60.0, gt=0)
    startup_mode: GameModeName = GameModeName.FREEPLAY

    # -- long-run resilience ------------------------------------------------

    #: Seconds without a completed frame before the watchdog calls the loop
    #: stalled. Well above the frame budget: a slow frame is not a stall, and
    #: table detection alone can legitimately take a second on a busy Pi.
    watchdog_stall_seconds: float = Field(5.0, gt=0)
    #: Consecutive frames a readiness condition must hold before the projected
    #: state changes. At 15-30 FPS this is one to two seconds -- long enough to
    #: ride out an arm reaching across the table, short enough to give feedback
    #: while somebody is still holding the bracket.
    readiness_confirm_frames: int = Field(30, gt=0)
    #: How long to keep retrying a lost camera before giving up and stopping the
    #: loop. USB dropouts recover in seconds; a camera that has been unplugged
    #: never will, and spinning on it forever hides the problem behind a
    #: process that still looks alive.
    camera_recovery_seconds: float = Field(30.0, gt=0)
    #: Consecutive failures before a non-essential stage is switched off for the
    #: rest of the run. At 30 FPS this is one second of a stage failing on every
    #: single frame -- comfortably past "transient" and short of "wait, is it
    #: ever going to give up".
    stage_failure_limit: int = Field(30, gt=0)
    #: Whether to shed optional work when frame times exceed the budget. Off
    #: makes performance problems visible as dropped frames instead of as a
    #: quietly coarser overlay, which is what you want while tuning.
    adaptive_degradation: bool = True


class Settings(BaseModel):
    """Root config object. One instance per process, via :func:`get_settings`."""

    table_preset: str = Field("7ft", description="Key into TABLE_PRESETS")
    #: ``model_copy``, not the preset itself. Pydantic calls the factory but does
    #: not copy what it returns, so handing back the shared ``TABLE_PRESETS``
    #: entry makes every ``Settings()`` in the process alias one ``TableSize``
    #: -- and anything that writes to ``settings.table`` silently edits the
    #: preset for everybody. Table detection now measures the table and can
    #: adopt the result, so that write really happens.
    table: TableSize = Field(default_factory=lambda: TABLE_PRESETS["7ft"].model_copy())
    camera: CameraSettings = Field(default_factory=CameraSettings)
    projector: ProjectorSettings = Field(default_factory=ProjectorSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    physics: PhysicsSettings = Field(default_factory=PhysicsSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)

    @field_validator("table_preset")
    @classmethod
    def _known_preset(cls, v: str) -> str:
        if v not in TABLE_PRESETS:
            raise ValueError(
                f"unknown table_preset {v!r}; expected one of {sorted(TABLE_PRESETS)}"
            )
        return v

    @property
    def frame_interval(self) -> float:
        """Target seconds per frame."""
        return 1.0 / self.system.target_fps

    @property
    def overlay_alpha(self) -> float:
        """Overlay opacity as a 0.0-1.0 float, for blending calls."""
        return self.projector.overlay_alpha_pct / 100.0


def _apply_saved_crop(settings: Settings) -> None:
    """Overlay the panel-saved crop onto the loaded settings, if there is one.

    Here rather than in the camera so that ``settings.camera.crop`` is correct
    for everyone -- ``crop_scale`` is read by pocket detection, which never sees
    a :class:`vision.camera.Camera`.

    The saved file wins over ``camera.crop`` in the YAML. Both exist on purpose:
    the YAML entry is the hand-editable default for a rig that has never had a
    crop set from the panel, and deleting the saved file returns to it. See
    :mod:`vision.crop_store` for why the panel does not write the YAML.

    Deliberately tolerant. A crop is a convenience, and no configuration of it
    is worth refusing to boot over -- an unreadable file is logged and skipped
    by ``crop_store.load`` itself.
    """
    try:
        from vision import crop_store
    except ImportError:  # pragma: no cover - dependency guard
        return

    rect, present = crop_store.load(settings.camera.rotated_size)
    if not present:
        return

    if rect is None:
        # A saved "full frame" is a choice, not an absence, so it overrides a
        # stale YAML crop rather than letting one quietly come back.
        settings.camera.crop = CropSettings()
        return

    settings.camera.crop = CropSettings(
        enabled=True, x=rect.x, y=rect.y, width=rect.width, height=rect.height
    )
    logger.info(
        "applied saved camera crop: %dx%d at %d,%d (of %dx%d)",
        rect.width, rect.height, rect.x, rect.y, *settings.camera.rotated_size,
    )


def load_settings(path: Path | str | None = None) -> Settings:
    """Load and validate settings from a YAML file.

    A missing config file is not an error -- every field has a working default,
    so the system boots on defaults and logs that it did so. A *malformed* file
    is an error and is raised, because silently running on defaults when the
    user has hand-tuned HSV ranges would be far more confusing than a crash.

    ``table_preset`` is applied before validation so that setting the preset in
    YAML populates ``table`` without the user restating the dimensions; explicit
    ``table`` values in the file still win.
    """
    config_path = Path(path) if path else Path(os.environ.get("AR_POOL_CONFIG", DEFAULT_CONFIG_PATH))

    if not config_path.is_file():
        logger.warning("no config file at %s; using built-in defaults", config_path)
        return Settings()

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pyyaml is required to read config.yaml; run `pip install -r requirements.txt`"
        ) from exc

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping, got {type(raw).__name__}")

    preset = raw.get("table_preset", "7ft")
    if preset in TABLE_PRESETS:
        merged = TABLE_PRESETS[preset].model_dump()
        merged.update(raw.get("table") or {})
        raw["table"] = merged

    settings = Settings.model_validate(raw)
    _apply_saved_crop(settings)
    logger.info(
        "loaded config from %s (table=%s %.0fx%.0f in, target %d FPS)",
        config_path,
        settings.table_preset,
        settings.table.length_in,
        settings.table.width_in,
        settings.system.target_fps,
    )
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that the vision loop and the web workers see the same object and a
    change made over the API is visible everywhere. Call
    :func:`reload_settings` to pick up on-disk edits.
    """
    return load_settings()


def reload_settings() -> Settings:
    """Drop the cached settings and re-read the YAML file."""
    get_settings.cache_clear()
    return get_settings()
