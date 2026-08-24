"""Measuring how far off the projection is, and saying so in inches.

Phase 6.3. Everything the wizard *claims* about alignment is computed here, and
nothing here touches a camera, a window or a projector -- which is the point.
The screens are hard to test because they block on a human; these functions are
pure, so the arithmetic the user is trusting is under test even though the
screens around it are not.

Two halves, and they are separated deliberately:

**Finding the marks** (:func:`locate_projected_marks`, :func:`assign_marks_to_corners`)
    Image processing. Where did the projected crosshairs actually land in the
    camera frame? Answered by differencing a lit frame against a blanked one,
    which is far more robust than looking for bright pixels -- felt under a
    ceiling light is not dark, and a chalk mark is.

**Judging the result** (:func:`compute_alignment_error`, :func:`compute_grid_metrics`)
    Pure geometry on homographies. No pixels.

:func:`projection_error_in` spans both, and deliberately: it is the wizard's
only end-to-end check, and the reason it takes *observed* mark positions rather
than composing the transforms is written out in its docstring. The short
version is that the composed version is a tautology that returns zero for any
input, including a projector aimed at the floor.

Directions and signs
--------------------
Every direction here is stated **as it appears on the console's camera view**,
because that is what the user is looking at while they push the projector
around. "Right" means right on screen. Table space inherits the camera's
handedness (the homography maps the camera's top-left corner to table ``(0,0)``
and its top-right to ``(length, 0)``), so +y is down and a positive rotation is
**clockwise on screen** in both spaces. Getting this backwards would send the
user the wrong way, which is worse than saying nothing.

Why not tell the user how far to move the *projector*
-----------------------------------------------------
We can measure how far the projected marks must move on the felt. How far the
projector body must move to achieve that depends on the throw ratio, the lens
shift and the mounting angle, none of which are known. So the advice names the
displacement of the marks and asks the user to nudge the projector until it
happens -- which is a closed loop with live feedback, and self-corrects whatever
the true ratio turns out to be.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from app.config import Settings, get_settings
from app.models import AlignmentError, Severity, TableBoundary, Vec2

logger = logging.getLogger(__name__)

__all__ = [
    "CORNER_NAMES",
    "CORNER_LABELS",
    "GridMetrics",
    "MarkDetection",
    "locate_projected_marks",
    "assign_marks_to_corners",
    "solve_projector_to_camera",
    "similarity_fit",
    "compute_alignment_error",
    "compute_grid_metrics",
    "ProjectionCheck",
    "projection_error_in",
]

#: Corner order, matching :meth:`app.models.TableBoundary.corners` -- clockwise
#: from top-left as the camera sees it. Every dict keyed by corner in this
#: package uses these names, so a plain ``zip`` against ``boundary.corners()``
#: is always correct.
CORNER_NAMES: tuple[str, ...] = ("top_left", "top_right", "bottom_right", "bottom_left")

#: What the user is shown. Spelled out rather than "TL", because the wizard is
#: read by someone who has never seen the codebase.
CORNER_LABELS: dict[str, str] = {
    "top_left": "TOP LEFT",
    "top_right": "TOP RIGHT",
    "bottom_right": "BOTTOM RIGHT",
    "bottom_left": "BOTTOM LEFT",
}

#: Below this the difference image is indistinguishable from sensor noise and
#: any "mark" found is a fiction. 0-255 on the max channel of the absdiff.
_MIN_MARK_CONTRAST = 28

#: A projected crosshair at a couple of metres covers hundreds of camera px.
#: Anything much smaller is a speck of dust catching the light.
_MIN_MARK_AREA_PX = 24

#: Rotation big enough to be worth mentioning before offset. Below this, asking
#: someone to twist a projector by eye does more harm than good.
_ROTATION_ADVICE_DEG = 1.5
#: Scale error worth a zoom instruction, as a fraction.
_SCALE_ADVICE_FRAC = 0.04
#: Offset worth an instruction, in inches on the felt.
_OFFSET_ADVICE_IN = 0.5


# ---------------------------------------------------------------------------
# Finding the projected marks in the camera frame
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MarkDetection:
    """One bright blob found in the lit-minus-dark difference image."""

    center: Vec2  # camera px
    area_px: float
    contrast: float  # 0-255, peak difference inside the blob

    @property
    def confidence(self) -> float:
        """How much to trust this as a real projected mark, 0-1.

        Contrast dominates: a faint blob is the thing that turns out to be a
        reflection off a rail, and area alone cannot tell those apart.
        """
        by_contrast = min(1.0, self.contrast / 120.0)
        by_area = min(1.0, self.area_px / (_MIN_MARK_AREA_PX * 6.0))
        return round(0.75 * by_contrast + 0.25 * by_area, 3)


def locate_projected_marks(
    dark_bgr: np.ndarray,
    lit_bgr: np.ndarray,
    *,
    max_marks: int = 8,
    min_area_px: int = _MIN_MARK_AREA_PX,
) -> list[MarkDetection]:
    """Find where projected marks landed, by differencing two camera frames.

    Differencing rather than thresholding, because the alternative does not
    work. The wizard runs in a lit room on cloth that is nowhere near black, so
    "find the bright things" finds the rails, the chalk, the reflection off a
    ball and occasionally a window. Subtracting a frame captured with the
    projector blanked removes everything that is not the projector's own light,
    and what remains is exactly the marks.

    Args:
        dark_bgr: Camera frame with the projector showing black.
        lit_bgr: Camera frame with the marks projected. Same size as ``dark_bgr``.
        max_marks: Cap on returned blobs, brightest first. Guards against a
            pathological difference image yielding thousands of contours.
        min_area_px: Reject blobs smaller than this, in full-resolution px.

    Returns:
        Detections sorted by descending contrast, so taking the first N takes
        the most convincing N. Empty when the projector is not reaching the
        table at all -- which is itself the answer to "why is nothing aligning".

    Raises:
        ValueError: If the two frames are different sizes, which means they came
            from different sources and the difference is meaningless.
    """
    import cv2

    if dark_bgr.shape != lit_bgr.shape:
        raise ValueError(f"frame size mismatch: dark {dark_bgr.shape} vs lit {lit_bgr.shape}")

    diff = cv2.absdiff(lit_bgr, dark_bgr)
    # Max across channels, not a luma conversion: a saturated cyan mark on green
    # felt barely moves the green channel and barely moves luma, but slams the
    # blue one. Taking the max keeps every mark colour equally visible.
    gray = diff.max(axis=2) if diff.ndim == 3 else diff
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    peak = int(gray.max())
    if peak < _MIN_MARK_CONTRAST:
        logger.info(
            "no projected light reached the camera (peak difference %d < %d); "
            "check the projector is on and aimed at the table",
            peak,
            _MIN_MARK_CONTRAST,
        )
        return []

    # Relative to the peak rather than absolute, so the same code works for a
    # dim projector in a bright room and a bright one in the dark. Half the peak
    # is comfortably above the noise floor and comfortably below a mark's core.
    level = max(_MIN_MARK_CONTRAST, int(peak * 0.5))
    _, mask = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
    # Close before contouring: a projected crosshair is thin lines, and sensor
    # noise breaks them into fragments that would each become a "mark".
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[MarkDetection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area_px:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            continue
        center = Vec2(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

        blob = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        contrast = float(gray[blob > 0].max())
        found.append(MarkDetection(center=center, area_px=area, contrast=contrast))

    found.sort(key=lambda m: m.contrast, reverse=True)
    logger.info(
        "located %d projected mark(s) (peak difference %d, threshold %d)",
        len(found),
        peak,
        level,
    )
    return found[:max_marks]


def assign_marks_to_corners(
    marks: list[MarkDetection], boundary: TableBoundary
) -> dict[str, Vec2]:
    """Label detected marks by which table corner they belong to.

    Assignment is by quadrant about the table centre, not by nearest corner. A
    badly aimed projection can put its top-left mark closer to the table's
    *centre* than to the top-left corner, and nearest-corner matching then
    assigns two marks to one corner and none to another -- silently producing a
    calibration solved from three points pretending to be four. Quadrants are
    unambiguous as long as the projection is roughly the right way up, which is
    a far weaker assumption.

    Only the most convincing mark in each quadrant is kept, so a stray
    reflection loses to the real crosshair.

    Returns:
        A dict keyed by :data:`CORNER_NAMES`, with entries only for quadrants
        that had a mark. Callers must handle a partial result -- an unlit corner
        is the normal symptom of a projection that does not cover the table.
    """
    center = boundary.center
    best: dict[str, MarkDetection] = {}
    for mark in marks:
        right = mark.center.x >= center.x
        below = mark.center.y >= center.y
        if below:
            name = "bottom_right" if right else "bottom_left"
        else:
            name = "top_right" if right else "top_left"
        if name not in best or mark.contrast > best[name].contrast:
            best[name] = mark

    result = {name: detection.center for name, detection in best.items()}
    missing = [name for name in CORNER_NAMES if name not in result]
    if missing:
        logger.warning(
            "no projected mark found near %s; the projection may not cover the whole table",
            ", ".join(missing),
        )
    return result


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def solve_projector_to_camera(
    correspondences: dict[str, tuple[Vec2, Vec2]],
) -> np.ndarray | None:
    """Solve the projector px -> camera px homography from recorded corners.

    This is the transform the wizard cannot get any other way, and the one every
    later measurement depends on: it is what lets the app predict where a given
    projector pixel will physically appear, and therefore how far that is from
    where it was wanted.

    Args:
        correspondences: ``name -> (camera_px, projector_px)``, as recorded by
            the corner-mapping screen. At least four, and not collinear.

    Returns:
        A 3x3 float64 matrix, or ``None`` if there are too few points or the
        solve degenerates. ``None`` rather than an exception because the caller
        is a screen that must keep running and tell the user to redo a corner.
    """
    import cv2

    if len(correspondences) < 4:
        logger.debug(
            "need 4 correspondences to solve projector->camera, have %d", len(correspondences)
        )
        return None

    camera = np.array([[c.x, c.y] for c, _p in correspondences.values()], dtype=np.float32)
    projector = np.array([[p.x, p.y] for _c, p in correspondences.values()], dtype=np.float32)

    if len(correspondences) == 4:
        matrix = cv2.getPerspectiveTransform(projector, camera)
    else:
        matrix, _mask = cv2.findHomography(projector, camera, method=0)

    if matrix is None or not np.all(np.isfinite(matrix)):
        logger.error(
            "projector->camera solve failed; two corners were probably recorded "
            "at nearly the same place"
        )
        return None
    return matrix.astype(np.float64)


def similarity_fit(source: list[Vec2], target: list[Vec2]) -> tuple[float, float, Vec2]:
    """Least-squares similarity taking ``source`` onto ``target``.

    Returns ``(rotation_deg, scale, translation)`` -- the three corrections a
    user can physically apply to a projector: twist it, zoom it, slide it.
    Decomposing the residual this way is what turns "18 px of error" into an
    instruction, which is the whole job of the corner-mapping screen.

    Solved in closed form rather than through ``cv2.estimateAffinePartial2D``.
    With exactly four points there is nothing for a robust estimator to be
    robust against, and that call's RANSAC default makes the answer depend on a
    random seed -- so a user who nudges nothing would still see the reported
    rotation twitch between frames.

    Rotation is degrees **clockwise on screen** (both spaces have +y down; see
    the module docstring). Scale > 1 means the projection must grow.

    Raises:
        ValueError: On mismatched lengths or fewer than two points, neither of
            which has a meaningful answer.
    """
    if len(source) != len(target):
        raise ValueError(f"{len(source)} source points but {len(target)} target points")
    if len(source) < 2:
        raise ValueError(f"need at least 2 points for a similarity fit, got {len(source)}")

    src = np.array([[p.x, p.y] for p in source], dtype=np.float64)
    dst = np.array([[p.x, p.y] for p in target], dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    s = src - src_mean
    d = dst - dst_mean

    # 2D Umeyama without reflection: the rotation that maximises correlation is
    # the argument of the complex inner product, and the scale is its magnitude
    # over the source's variance.
    a = float((s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum())
    b = float((s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum())
    denominator = float((s**2).sum())
    if denominator < 1e-9:
        # Every source point is the same point. No orientation to recover.
        logger.debug("similarity fit on coincident points; reporting identity")
        return 0.0, 1.0, Vec2(*(dst_mean - src_mean))

    rotation = math.degrees(math.atan2(b, a))
    scale = math.hypot(a, b) / denominator
    theta = math.radians(rotation)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    translation = dst_mean - scale * (rot @ src_mean)
    return rotation, scale, Vec2(float(translation[0]), float(translation[1]))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def compute_alignment_error(
    detected: dict[str, Vec2],
    projected_seen: dict[str, Vec2],
    px_per_inch: float,
) -> AlignmentError:
    """How far the projected marks are from the table corners, and what to do.

    Both inputs are in **camera px**, which is the only space where the two
    things can honestly be compared: one is where the cushion nose physically
    is, the other is where the projector's light physically landed. Comparing
    them anywhere else would mean applying the very transform being measured.

    Args:
        detected: Table corners from detection, keyed by :data:`CORNER_NAMES`.
        projected_seen: Where each projected mark was observed, same keys. Only
            corners present in both are used.
        px_per_inch: Camera scale over the felt, from
            :func:`vision.calibration.pixels_per_inch`. Converts the error into
            the units the advice is phrased in.

    Returns:
        An :class:`~app.models.AlignmentError` whose ``message`` is a single
        physical instruction and whose ``severity`` decides whether the user is
        allowed to continue.
    """
    shared = [name for name in CORNER_NAMES if name in detected and name in projected_seen]
    if not shared:
        return AlignmentError(
            total_rmse=float("inf"),
            message="Cannot see the projected marks on the table.",
            severity="error",
        )

    marks = [projected_seen[name] for name in shared]
    corners = [detected[name] for name in shared]

    residuals = [corner.distance_to(mark) for corner, mark in zip(corners, marks, strict=True)]
    rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    dx = sum(c.x - m.x for c, m in zip(corners, marks, strict=True)) / len(shared)
    dy = sum(c.y - m.y for c, m in zip(corners, marks, strict=True)) / len(shared)

    if len(shared) >= 2:
        rotation, scale, _translation = similarity_fit(marks, corners)
    else:
        rotation, scale = 0.0, 1.0

    scale_px = px_per_inch if px_per_inch > 1e-6 else 1.0
    message, severity = _alignment_advice(
        dx_in=dx / scale_px,
        dy_in=dy / scale_px,
        rotation_deg=rotation,
        scale=scale,
        rmse_in=rmse / scale_px,
    )

    logger.info(
        "alignment over %d corner(s): rmse %.1f px (%.2f in), offset (%.1f, %.1f) px, "
        "rotation %.2f deg, scale %.3f",
        len(shared),
        rmse,
        rmse / scale_px,
        dx,
        dy,
        rotation,
        scale,
    )
    return AlignmentError(
        total_rmse=rmse,
        x_offset=dx,
        y_offset=dy,
        rotation=rotation,
        message=message,
        severity=severity,
    )


def _alignment_advice(
    *, dx_in: float, dy_in: float, rotation_deg: float, scale: float, rmse_in: float
) -> tuple[str, Severity]:
    """Pick the one correction worth making next, and phrase it.

    One instruction, not four. A user holding a projector cannot act on "shift
    left, rotate 2 degrees clockwise, zoom out 5% and raise the front" -- and
    they do not have to, because the screen re-measures after every nudge. So
    the largest error is named and the rest are left to the metrics panel.

    Order is rotation, then zoom, then slide, because that is the order in which
    they interfere: a rotated projection cannot be slid into place, and a
    mis-sized one cannot be twisted into place, but once both are right, sliding
    finishes the job.
    """
    if abs(rotation_deg) >= _ROTATION_ADVICE_DEG:
        direction = "clockwise" if rotation_deg > 0 else "anti-clockwise"
        severity: Severity = "warning" if abs(rotation_deg) < 5.0 else "error"
        return f"Twist the projector {abs(rotation_deg):.0f} degrees {direction}.", severity

    if abs(scale - 1.0) >= _SCALE_ADVICE_FRAC:
        percent = abs(scale - 1.0) * 100.0
        verb = "in" if scale > 1.0 else "out"
        severity = "warning" if percent < 12.0 else "error"
        return f"Zoom the projector {verb} about {percent:.0f}%.", severity

    offset = math.hypot(dx_in, dy_in)
    if offset >= _OFFSET_ADVICE_IN:
        parts = []
        if abs(dx_in) >= 0.25:
            parts.append(f"{abs(dx_in):.1f} in {'right' if dx_in > 0 else 'left'}")
        if abs(dy_in) >= 0.25:
            parts.append(f"{abs(dy_in):.1f} in {'down' if dy_in > 0 else 'up'}")
        # Under a quarter-inch on both axes but over the threshold diagonally:
        # there is no useful direction to name, so give the size and let them
        # nudge while watching the arrows.
        movement = " and ".join(parts) if parts else f"{offset:.1f} in"
        severity = "info" if offset < 1.5 else "warning" if offset < 4.0 else "error"
        return f"Nudge the projector so the marks move {movement}.", severity

    if rmse_in > 0.5:
        # Offset, rotation and scale are all small, yet corners are still off --
        # so what is left is keystone, which no rigid move can fix.
        return (
            "The projection is skewed. Raise or lower the front of the projector.",
            "warning",
        )
    return "Alignment looks excellent.", "info"


# ---------------------------------------------------------------------------
# Grid quality
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GridMetrics:
    """How square and how complete the projection is over the felt.

    Derived analytically from the two homographies rather than by looking for
    projected grid lines in the image. Same answer, no detection to fail, and it
    stays correct when the user has switched the grid off to see the cloth.
    """

    #: Angle between the projector's own x and y axes once they land on the
    #: felt. 90 is perfect; a departure means keystone.
    perpendicularity_deg: float
    #: Projected x axis against the table's long axis, degrees clockwise.
    rotation_deg: float
    #: Percentage of the table's long axis the projection reaches, measured
    #: along the centre line.
    coverage_x_pct: float
    #: The same across the short axis.
    coverage_y_pct: float

    @property
    def is_square(self) -> bool:
        """Whether the projection is square enough to play under."""
        return abs(self.perpendicularity_deg - 90.0) <= 2.0 and abs(self.rotation_deg) <= 2.0

    @property
    def covers_table(self) -> bool:
        """Whether the projection reaches far enough into every corner.

        95% rather than 100%: the last couple of percent is the cushion nose
        itself, where nothing is ever drawn, and demanding it would fail setups
        that are perfectly good to play on.
        """
        return self.coverage_x_pct >= 95.0 and self.coverage_y_pct >= 95.0

    @property
    def is_acceptable(self) -> bool:
        return self.is_square and self.covers_table


def compute_grid_metrics(
    projector_to_camera: np.ndarray,
    camera_to_table: np.ndarray,
    settings: Settings | None = None,
    *,
    samples: int = 101,
) -> GridMetrics:
    """Measure squareness and coverage of the projection over the table.

    Args:
        projector_to_camera: From :func:`solve_projector_to_camera`.
        camera_to_table: From :func:`vision.calibration.compute_perspective_transform`.
        settings: Config supplying the table size. Defaults to the global.
        samples: Points tested along each centre line for coverage. 101 gives
            1% resolution, which is finer than the metric is quoted to.

    Returns:
        The four numbers the fine-tune screen displays.
    """
    settings = settings or get_settings()
    length, width = settings.table.length_in, settings.table.width_in
    projector_to_table = camera_to_table @ projector_to_camera

    pw, ph = float(settings.projector.width), float(settings.projector.height)
    quad = _apply(
        projector_to_table,
        np.array([[0.0, 0.0], [pw, 0.0], [pw, ph], [0.0, ph]], dtype=np.float64),
    )

    # Squareness is read at the frame centre, where the user is looking and
    # where a homography's local behaviour is most representative of the whole.
    step = min(pw, ph) * 0.25
    origin, along_x, along_y = _apply(
        projector_to_table,
        np.array(
            [
                [pw / 2.0, ph / 2.0],
                [pw / 2.0 + step, ph / 2.0],
                [pw / 2.0, ph / 2.0 + step],
            ],
            dtype=np.float64,
        ),
    )
    ux, uy = along_x - origin, along_y - origin
    perpendicularity = _angle_between(ux, uy)
    rotation = math.degrees(math.atan2(ux[1], ux[0]))
    # Wrap into (-90, 90]: a projector mounted end-on is a mounting choice, not
    # an alignment error, and reporting 179 degrees would send someone off to
    # unbolt a perfectly good rig.
    rotation = (rotation + 90.0) % 180.0 - 90.0

    metrics = GridMetrics(
        perpendicularity_deg=perpendicularity,
        rotation_deg=rotation,
        coverage_x_pct=_covered_fraction(quad, length, width, axis=0, samples=samples) * 100.0,
        coverage_y_pct=_covered_fraction(quad, length, width, axis=1, samples=samples) * 100.0,
    )
    logger.info(
        "grid metrics: perpendicularity %.1f deg, rotation %.1f deg, coverage %.0f%% x %.0f%%",
        metrics.perpendicularity_deg,
        metrics.rotation_deg,
        metrics.coverage_x_pct,
        metrics.coverage_y_pct,
    )
    return metrics


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an Nx2 array, returning Nx2.

    Points on the transform's horizon come back as NaN rather than as a huge
    finite number. A caller comparing NaN gets ``False``, which is the right
    answer for "is this point on the table"; a caller comparing 1e17 gets a
    plausible-looking wrong answer.
    """
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return projected[:, :2] / w


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned angle between two 2-D vectors, in degrees."""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return 90.0
    cosine = float(np.dot(u, v)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _covered_fraction(
    quad: np.ndarray, length: float, width: float, *, axis: int, samples: int
) -> float:
    """Fraction of one table centre line that falls inside the projected quad.

    A centre line rather than the whole surface, because the two numbers then
    answer the question the user actually has -- "does it reach both ends?" --
    instead of an area percentage that stays high while a whole rail is dark.
    """
    import cv2

    if not np.all(np.isfinite(quad)):
        logger.warning("projected frame does not map onto the table plane; coverage unknown")
        return 0.0

    contour = quad.astype(np.float32).reshape(-1, 1, 2)
    if axis == 0:
        points = [(float(t), width / 2.0) for t in np.linspace(0.0, length, samples)]
    else:
        points = [(length / 2.0, float(t)) for t in np.linspace(0.0, width, samples)]

    inside = sum(1 for point in points if cv2.pointPolygonTest(contour, point, False) >= 0)
    return inside / float(samples)


# ---------------------------------------------------------------------------
# End-to-end check
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProjectionCheck:
    """Result of the end-to-end check: how far the light landed from the balls.

    Carries the ``pairs`` as well as the mean, because the wizard draws an arrow
    per pair and the drawing must use the *same* matching the number came from.
    Matching twice -- once to measure, once to draw -- is how a screen ends up
    showing arrows between a ball and some other ball's ring while reporting a
    small error, which is worse than showing nothing.
    """

    #: Mean distance over the matched pairs, inches. ``inf`` when nothing matched.
    mean_error_in: float
    #: ``(ball_px, mark_px)`` for each ball a mark was matched to.
    pairs: list[tuple[Vec2, Vec2]]
    #: How many balls were offered for checking, matched or not.
    balls_checked: int

    @property
    def matched(self) -> int:
        return len(self.pairs)

    @property
    def all_matched(self) -> bool:
        """Whether every ball got a ring. The number that actually matters.

        A small mean over one ball out of six means the projection missed five
        of them -- the opposite of a pass, and invisible in the mean alone.
        """
        return self.balls_checked > 0 and self.matched == self.balls_checked


def projection_error_in(
    ball_positions_px: list[Vec2],
    mark_positions_px: list[Vec2],
    px_per_inch: float,
    *,
    max_match_px: float,
) -> ProjectionCheck:
    """How far projected marks landed from the balls they were aimed at.

    The wizard's only honest end-to-end measurement, and it has to be made
    against **light the camera actually saw** rather than against arithmetic.

    The tempting version of this function composes the transforms instead:
    send a ball's table position out through the calibration to a projector
    pixel, then back through the measured projector-to-camera homography, and
    compare with where the camera sees the ball. That is worthless. Both
    matrices are solved from the same four correspondences, so their
    composition *is* the table-to-camera homography exactly -- not just at the
    four corners but everywhere, because four points determine a homography.
    The result is identically zero for any input, including a projector aimed
    at the floor. It looks like a rigorous check and it is a tautology.

    So the caller projects a mark at each ball, photographs the felt, finds the
    marks by differencing, and passes the observed positions here. Any error in
    the calibration, any drift since the corners were placed, and any keystone
    the four corners failed to capture all show up, because the light went
    through the real optics.

    Args:
        ball_positions_px: Detected ball centres, camera px.
        mark_positions_px: Observed projected marks, camera px, in any order.
        px_per_inch: Camera scale over the felt.
        max_match_px: A mark further than this from every ball is not that
            ball's mark. Callers should size it from the ball radius -- too
            generous and a stray reflection gets matched to a ball, flattering
            the result; too tight and a genuinely bad projection matches
            nothing and reports success by reporting nothing.

    Returns:
        A :class:`ProjectionCheck`. Read ``all_matched`` before ``mean_error_in``.
    """
    if not ball_positions_px or not mark_positions_px or px_per_inch <= 1e-6:
        return ProjectionCheck(float("inf"), [], len(ball_positions_px))

    # Greedy nearest-mark matching, each mark used once. Balls are separated by
    # at least a diameter, so the assignment is unambiguous whenever the
    # projection is good enough to be worth measuring -- and when it is not, the
    # unmatched count says so more clearly than an optimal assignment would.
    remaining = list(mark_positions_px)
    pairs: list[tuple[Vec2, Vec2]] = []
    for ball in ball_positions_px:
        if not remaining:
            break
        nearest = min(remaining, key=ball.distance_to)
        if ball.distance_to(nearest) > max_match_px:
            continue
        remaining.remove(nearest)
        pairs.append((ball, nearest))

    if not pairs:
        logger.info(
            "no projected mark landed within %.0f px of any of the %d detected ball(s)",
            max_match_px,
            len(ball_positions_px),
        )
        return ProjectionCheck(float("inf"), [], len(ball_positions_px))

    errors = [ball.distance_to(mark) for ball, mark in pairs]
    mean_error_in = float(np.mean(errors)) / px_per_inch
    logger.info(
        "projection error over %d of %d ball(s): %.2f in (worst %.2f in)",
        len(pairs),
        len(ball_positions_px),
        mean_error_in,
        max(errors) / px_per_inch,
    )
    return ProjectionCheck(mean_error_in, pairs, len(ball_positions_px))
