"""Pocket-based table detection, independent of felt colour.

Session 2 retrofit. The felt-segmentation path in :mod:`vision.calibration`
finds the table by looking for green cloth, which is fast and accurate and stops
working the moment somebody re-covers their table in burgundy. This module finds
the same table by locating its **six pocket mouths** and reconstructing the rails
from them, which works on any cloth colour because a hole is a hole.

The pipeline
------------
Two passes, because the parameters that find a pocket depend on how big the
pockets are, which is what the first pass is for:

1. :func:`detect_pockets_loose` -- size-agnostic. Sweeps a range of darkness
   thresholds and keeps round, locally-dark blobs anywhere from 0.4% to 7.5% of
   the frame width in radius.
2. :func:`dynamic_detect_table_size` -- fits a quad to those blobs and measures
   the table: pocket spacing, pocket radius, and the real-world size.
3. :func:`get_dynamic_hough_params` -- turns those measurements into detection
   parameters scaled to *this* table. Nothing hardcoded.
4. :func:`detect_pockets_refined` -- second pass with those parameters.
5. :func:`classify_pockets_by_geometry` -- four corners, two sides.
6. :func:`detect_table_boundaries_dynamic` -- fits the rails and returns a
   :class:`~app.models.TableBoundary` with the same interface everything
   downstream already consumes.

Scale, and the thing that cannot work
-------------------------------------
The retrofit brief specifies deriving real-world size from a fixed reference::

    scale_factor  = measured_width_px / 2000
    actual_ft     = 7.0 * scale_factor

This cannot recover table size, and the reason is worth stating plainly because
the formula looks reasonable. A camera sees ``f * L / h`` pixels for a table of
size ``L`` at height ``h``. Doubling ``L`` and doubling ``h`` give an identical
image. So pixel width constrains only the *ratio* ``L/h``, and a single number
cannot be split into two unknowns. The 2:1 aspect ratio does not help: every
pool table from 6 ft to 10 ft is 2:1, so the aspect carries no information about
which one this is.

Measured against the synthetic harness -- one unchanged 6.33 ft table, three
camera heights:

| Camera  | Table px | Reference formula | Ball ratio |
|---------|---------:|------------------:|-----------:|
| low     |     1843 |           6.45 ft |    6.33 ft |
| typical |     1459 |           5.11 ft |    6.33 ft |
| high    |      845 |           2.96 ft |    6.33 ft |

The fix is to put a second known length in the image, and there already is one:
**a ball**. A pool ball is 2.25 in regardless of the table it is sitting on, so

    table_px / ball_px = table_in / 2.25

and the height and focal length cancel. That is what :func:`resolve_scale`
prefers, and it is what makes "measure a 7.5 ft table as 7.5 ft" achievable at
all. The configured table size comes second, and the brief's fixed reference is
kept as a last resort with a warning, since it is the only option when the cloth
is bare.

The regulation-ball assumption is the one thing to know about the ball path: a
mini table with undersized balls will measure large, because the code has no way
to tell a small ball on a small table from a regulation ball on a big one. Set
``vision.scale_source: config`` and the table preset for those.

Cost
----
Roughly 12-18 ms on a 640 px frame against felt segmentation's ~4 ms, because
the threshold sweep runs the contour pass several times. Table detection runs on
an interval rather than per frame, so this lands as a periodic spike rather than
in the 33 ms budget -- see ``vision.table_detection_width``. It is the price of
not caring what colour the cloth is.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from app.config import (
    BALL_DIAMETER_IN,
    BALL_RADIUS_IN,
    Settings,
    get_settings,
    nearest_standard_length_ft,
)
from app.models import PocketId, TableBoundary, Vec2
from vision.calibration import downscale_for_detection, fit_line, intersect_lines, order_corners

logger = logging.getLogger(__name__)

__all__ = [
    "PocketBlob",
    "HoughParams",
    "TableMeasurement",
    "detect_pockets_loose",
    "detect_pockets_refined",
    "get_dynamic_hough_params",
    "classify_pockets_by_geometry",
    "dynamic_detect_table_size",
    "detect_table_boundaries_dynamic",
    "resolve_scale",
    "adaptive_cloth_mask",
    "adopt_measured_table_size",
    "report_size_disagreement",
]

#: Where each pocket sits in normalised table coordinates, clockwise from the
#: top-left. Identical to ``vision.detection.POCKET_LAYOUT`` and deliberately
#: so: detection places pockets from a boundary using that table, and this
#: module derives a boundary from detected pockets using this one. If the two
#: disagreed, every pocket would move the moment the detector changed.
POCKET_UV: list[tuple[PocketId, float, float]] = [
    (PocketId.TOP_LEFT, 0.0, 0.0),
    (PocketId.TOP_MIDDLE, 0.5, 0.0),
    (PocketId.TOP_RIGHT, 1.0, 0.0),
    (PocketId.BOTTOM_RIGHT, 1.0, 1.0),
    (PocketId.BOTTOM_MIDDLE, 0.5, 1.0),
    (PocketId.BOTTOM_LEFT, 0.0, 1.0),
]

_CORNER_IDS = (PocketId.TOP_LEFT, PocketId.TOP_RIGHT, PocketId.BOTTOM_RIGHT, PocketId.BOTTOM_LEFT)
_SIDE_IDS = (PocketId.TOP_MIDDLE, PocketId.BOTTOM_MIDDLE)


#: A regulation ball is 2.25 in, and regulation tables run 72-100 in on the long
#: axis, so a ball's radius is always between about 1.1% and 1.6% of the table's
#: imaged length. Searched wider than that in both directions to leave room for
#: measurement error and for the non-standard sizes the brief asks about.
_BALL_RADIUS_FRAC_RANGE = (0.006, 0.025)

#: Minimum balls needed before a ball-derived scale is trusted. One ball is
#: probably right and has nothing to be cross-checked against; two agreeing
#: radii rule out a chalk cube or a reflection being measured as a ball.
_MIN_BALLS_FOR_SCALE = 2

#: Pool tables are 2:1 on the playing surface, every size. Used only to reject
#: things that are not tables -- never to derive one axis from the other, since
#: both are measured.
_TABLE_ASPECT = 2.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PocketBlob:
    """One pocket-mouth candidate found in the frame.

    ``center`` and ``radius_px`` are in **full-resolution camera px**, already
    scaled back up from whatever resolution detection ran at.
    """

    center: Vec2
    radius_px: float
    #: 4*pi*area/perimeter^2. 1.0 is a perfect disc.
    circularity: float
    #: How much brighter the surrounding ring is than the blob, as a fraction
    #: of the ring's brightness -- 0 for a blob no darker than its surroundings,
    #: approaching 1 for a black hole in bright cloth. The discriminator that
    #: separates a pocket from the dark surround: a pocket is a dark spot inside
    #: a bright field, the surround is just dark.
    contrast: float
    pocket_id: PocketId | None = None

    @property
    def confidence(self) -> float:
        """How much this looks like a pocket rather than a shadow, 0-1."""
        by_shape = max(0.0, min(1.0, (self.circularity - 0.3) / 0.6))
        by_contrast = max(0.0, min(1.0, self.contrast / 0.75))
        return round(0.55 * by_shape + 0.45 * by_contrast, 3)


@dataclass(slots=True)
class HoughParams:
    """Detection parameters scaled to one specific table.

    Named for the brief's ``get_dynamic_hough_params``, and the three headline
    values are exactly the ones it specifies. They are applied to a contour pass
    rather than to ``cv2.HoughCircles``:

    ``min_dist``
        Minimum separation between accepted blobs. Suppresses the second
        detection when one pocket is found twice at different sweep levels.
    ``min_radius`` / ``max_radius``
        The size gate, which is the whole point of the second pass -- the loose
        pass has to admit anything from 0.4% to 7.5% of frame width, and this
        narrows it to what was actually measured.

    Contours rather than Hough for the same reason ball detection uses them (see
    the README): a Hough pass per frame costs more than the whole frame budget,
    and a corner pocket is a rounded wedge rather than a circle, which Hough
    scores poorly and a circularity gate accepts.
    """

    min_dist: float
    min_radius: float
    max_radius: float
    #: Carried through so a caller that does want ``cv2.HoughCircles`` has a
    #: complete parameter set without re-deriving these from config.
    dp: float = 1.2
    param1: float = 100.0
    param2: float = 18.0

    def as_dict(self) -> dict[str, float]:
        """The brief's dict shape, ready to splat into an OpenCV call."""
        return {
            "minDist": self.min_dist,
            "minRadius": self.min_radius,
            "maxRadius": self.max_radius,
            "dp": self.dp,
            "param1": self.param1,
            "param2": self.param2,
        }


@dataclass(slots=True)
class TableMeasurement:
    """What the pockets say about the table.

    Pixel fields are always populated. The real-world fields depend on a scale
    reference having been found -- see :func:`resolve_scale`.
    """

    #: Table corners in full-resolution camera px, clockwise from top-left,
    #: long axis first.
    quad: np.ndarray
    #: Long axis in px, averaged over the two long rails.
    length_px: float
    #: Short axis in px.
    width_px: float
    #: Mean distance between adjacent pockets along a rail.
    pocket_spacing_px: float
    #: Median detected pocket radius.
    pocket_radius_px: float
    #: Camera px per inch over the cloth.
    pixels_per_inch: float
    #: ``ball``, ``config`` or ``reference``.
    scale_source: str
    #: Matched pockets, keyed by id. May hold fewer than six.
    pockets: dict[PocketId, PocketBlob] = field(default_factory=dict)

    @property
    def length_ft(self) -> float:
        return self.length_px / self.pixels_per_inch / 12.0

    @property
    def width_ft(self) -> float:
        return self.width_px / self.pixels_per_inch / 12.0

    @property
    def pixels_per_ft(self) -> float:
        return self.pixels_per_inch * 12.0

    @property
    def aspect(self) -> float:
        return self.length_px / max(self.width_px, 1e-6)

    def as_dict(self) -> dict[str, float | str]:
        """The measurement as a plain dict, for logging and for the API.

        Note the axis naming against the brief's. It asks for
        ``table_width_ft`` as the measured dimension with
        ``table_length_ft = width * 2``, but a "7 ft table" is 7 ft on its
        **long** axis -- 76 in by 38 in of playing surface. Following the brief
        literally would report a 7 ft table as 7 ft wide and 14 ft long. The
        keys below therefore use this codebase's convention, which matches
        ``settings.table``: ``length`` is the long axis.
        """
        return {
            "table_length_ft": round(self.length_ft, 3),
            "table_width_ft": round(self.width_ft, 3),
            "table_length_px": round(self.length_px, 1),
            "table_width_px": round(self.width_px, 1),
            "scale_factor": round(self.length_ft / 7.0, 4),
            "pocket_spacing_px": round(self.pocket_spacing_px, 1),
            "pocket_radius_px": round(self.pocket_radius_px, 1),
            "pixels_per_ft": round(self.pixels_per_ft, 2),
            "pixels_per_inch": round(self.pixels_per_inch, 3),
            "scale_source": self.scale_source,
            "nearest_standard_ft": nearest_standard_length_ft(self.length_ft),
        }


# ---------------------------------------------------------------------------
# Pass 1: finding dark round things
# ---------------------------------------------------------------------------


def detect_pockets_loose(
    frame: np.ndarray, settings: Settings | None = None
) -> list[PocketBlob]:
    """First-pass pocket detection, with no idea how big the table is.

    Size-agnostic by construction: the only size constraint is
    ``vision.pocket_radius_frac_range``, a span wide enough to cover a mini
    table filling the frame and a 10 ft table shot from the ceiling.

    Returns:
        Candidates in full-resolution camera px, most convincing first. Expect
        more than six -- shadows and dark rail sections get through, and it is
        :func:`dynamic_detect_table_size`'s job to pick the six that form a
        table.
    """
    settings = settings or get_settings()
    if frame is None or frame.size == 0:
        return []

    small, scale = downscale_for_detection(frame, settings.vision.table_detection_width)
    low_frac, high_frac = settings.vision.pocket_radius_frac_range
    frame_width = float(small.shape[1])
    blobs = _dark_round_blobs(
        small,
        settings,
        min_radius=low_frac * frame_width,
        max_radius=high_frac * frame_width,
        min_dist=0.0,
    )
    logger.debug("loose pocket pass found %d candidate(s)", len(blobs))
    return _rescale_blobs(blobs, scale)


def detect_pockets_refined(
    frame: np.ndarray,
    hough_params: HoughParams,
    settings: Settings | None = None,
) -> list[PocketBlob]:
    """Second-pass pocket detection, using parameters scaled to this table.

    The same finder as the loose pass with the size gate closed down to what the
    first pass measured, which is what turns "any dark round thing" into "a
    pocket on this table". ``min_dist`` additionally suppresses the duplicate
    detections the threshold sweep produces when one pocket is visible at
    several darkness levels.
    """
    settings = settings or get_settings()
    if frame is None or frame.size == 0:
        return []

    small, scale = downscale_for_detection(frame, settings.vision.table_detection_width)
    blobs = _dark_round_blobs(
        small,
        settings,
        min_radius=hough_params.min_radius * scale,
        max_radius=hough_params.max_radius * scale,
        min_dist=hough_params.min_dist * scale,
    )
    logger.debug("refined pocket pass found %d candidate(s)", len(blobs))
    return _rescale_blobs(blobs, scale)


def _rescale_blobs(blobs: list[PocketBlob], scale: float) -> list[PocketBlob]:
    """Return blobs in full-resolution px.

    Every detector in this package owes its caller full-resolution coordinates;
    one that forgets produces results that are silently wrong by a constant
    factor and look like a calibration fault rather than a bug.
    """
    if scale == 1.0:
        return blobs
    return [
        PocketBlob(
            center=Vec2(blob.center.x / scale, blob.center.y / scale),
            radius_px=blob.radius_px / scale,
            circularity=blob.circularity,
            contrast=blob.contrast,
            pocket_id=blob.pocket_id,
        )
        for blob in blobs
    ]


def _otsu_ladder(gray: np.ndarray, depth: int) -> list[float]:
    """Successive Otsu splits of the dark end of the histogram.

    Each level is Otsu over the pixels below the previous level, so the ladder
    descends felt -> surround -> rail -> pocket. Nothing in it assumes how dark
    a pocket is or what fraction of the frame it covers, which is exactly what
    a fixed percentile got wrong.

    Returns the levels from brightest to darkest. Callers try all of them; a
    pocket that is clean at one level is a merged blob at the level above and
    fragments at the level below, and the deduplication picks the best version.
    """
    import cv2

    levels: list[float] = []
    remaining = gray.reshape(-1)
    for _ in range(depth):
        # Below this there is not enough population left for Otsu to say
        # anything, and it starts splitting sensor noise.
        if remaining.size < 64:
            break
        level, _mask = cv2.threshold(
            remaining.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if level <= 1.0:
            break
        levels.append(float(level))
        remaining = remaining[remaining < level]
    return levels


def _dark_round_blobs(
    small: np.ndarray,
    settings: Settings,
    *,
    min_radius: float,
    max_radius: float,
    min_dist: float,
) -> list[PocketBlob]:
    """Collect round, locally-dark blobs across every threshold in the ladder.

    Works in the downscaled frame's own coordinates.

    Several thresholds rather than one, because no single one works. A corner
    pocket opens into the rail, which is itself dark, so at any level loose
    enough to include the rail the two merge into a blob that fails every
    roundness test; the pocket is only separable at a level below the rail. But
    a level that low buries a shallow side pocket on a brightly lit table. The
    ladder covers both ends and :func:`_deduplicate` reconciles the results.
    """
    import cv2

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    found: list[PocketBlob] = []
    # Open with a kernel well under the smallest pocket: clears sensor speckle
    # and the thin dark seam along a cushion without touching a pocket mouth.
    k = max(3, int(min_radius * 0.6) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for level in _otsu_ladder(gray, settings.vision.pocket_threshold_depth):
        _, mask = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            blob = _blob_from_contour(contour, gray, settings, min_radius, max_radius)
            if blob is not None:
                found.append(blob)

    return _deduplicate(found, min_dist)


def _blob_from_contour(
    contour: np.ndarray,
    gray: np.ndarray,
    settings: Settings,
    min_radius: float,
    max_radius: float,
) -> PocketBlob | None:
    """Score one contour as a pocket candidate, or reject it."""
    import cv2

    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return None
    perimeter = float(cv2.arcLength(contour, closed=True))
    if perimeter <= 0.0:
        return None

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    if circularity < settings.vision.pocket_min_circularity:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if not (min_radius <= radius <= max_radius):
        return None

    contrast = _local_contrast(gray, cx, cy, radius)
    if contrast < settings.vision.pocket_min_contrast_frac:
        return None

    return PocketBlob(
        center=Vec2(float(cx), float(cy)),
        radius_px=float(radius),
        circularity=float(circularity),
        contrast=float(contrast),
    )


#: Grey levels the ring must clear the blob by before the ratio is believed.
#: Below this the two medians are separated by sensor noise, and a dark patch of
#: floor with a marginally lighter patch of floor around it would otherwise score
#: a respectable ratio.
_MIN_ABSOLUTE_CONTRAST = 6.0


def _local_contrast(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """How much brighter the ring around a blob is than the blob, 0-1.

    The single most useful pocket test, and the one that does the work a global
    threshold cannot. Both a pocket and the floor beyond the rails are dark;
    only the pocket has something brighter all the way around it. Sampled as a
    ring rather than a box so a pocket near the frame edge is judged on the
    cloth beside it rather than on whatever is outside the table.

    Reported as a fraction of the ring's own brightness rather than in grey
    levels, so that dimming the whole neighbourhood does not change the answer.
    That matters because room light falls off toward the frame corners and the
    corner pockets live there -- see ``vision.pocket_min_contrast_frac``.
    """
    import cv2

    height, width = gray.shape[:2]
    outer = int(round(radius * 2.6))
    x0, x1 = max(0, int(cx) - outer), min(width, int(cx) + outer + 1)
    y0, y1 = max(0, int(cy) - outer), min(height, int(cy) + outer + 1)
    window = gray[y0:y1, x0:x1]
    if window.size == 0:
        return 0.0

    local_x, local_y = cx - x0, cy - y0
    inner_mask = np.zeros(window.shape, dtype=np.uint8)
    cv2.circle(inner_mask, (int(local_x), int(local_y)), max(1, int(radius * 0.7)), 255, -1)
    ring_mask = np.zeros(window.shape, dtype=np.uint8)
    cv2.circle(ring_mask, (int(local_x), int(local_y)), max(2, int(radius * 2.4)), 255, -1)
    cv2.circle(ring_mask, (int(local_x), int(local_y)), max(1, int(radius * 1.3)), 0, -1)

    if not inner_mask.any() or not ring_mask.any():
        return 0.0
    # Medians, not means: a ring that clips a neighbouring pocket or a ball
    # should not have its brightness dragged around by the outlier.
    inner = float(np.median(window[inner_mask > 0]))
    ring = float(np.median(window[ring_mask > 0]))
    if ring - inner < _MIN_ABSOLUTE_CONTRAST:
        return 0.0
    return (ring - inner) / max(ring, 1.0)


def _deduplicate(blobs: list[PocketBlob], min_dist: float) -> list[PocketBlob]:
    """Keep the best blob in each cluster, strongest first.

    The threshold sweep finds the same pocket several times, once per level it
    survives. Two blobs are the same pocket when their centres are closer than
    ``min_dist``, or -- during the loose pass, which has no distance to work
    with -- than the larger of their radii.
    """
    kept: list[PocketBlob] = []
    for blob in sorted(blobs, key=lambda b: b.confidence, reverse=True):
        limit = min_dist if min_dist > 0.0 else blob.radius_px
        if any(blob.center.distance_to(other.center) < limit for other in kept):
            continue
        kept.append(blob)
    return kept


# ---------------------------------------------------------------------------
# Pass 2: measuring the table
# ---------------------------------------------------------------------------


def get_dynamic_hough_params(
    measurement: TableMeasurement, settings: Settings | None = None
) -> HoughParams:
    """Detection parameters derived from what the first pass measured.

    The three ratios are the brief's, and they are sound: pockets never sit
    closer than their own spacing, and a second-pass radius window of 0.8x-1.3x
    the measured radius is wide enough for perspective across the table and
    narrow enough to exclude everything the loose pass let through.
    """
    settings = settings or get_settings()
    return HoughParams(
        min_dist=measurement.pocket_spacing_px * 0.95,
        min_radius=measurement.pocket_radius_px * 0.8,
        max_radius=measurement.pocket_radius_px * 1.3,
        dp=settings.vision.hough_dp,
        param1=float(settings.vision.hough_param1),
        param2=float(settings.vision.hough_param2),
    )


def classify_pockets_by_geometry(
    pockets: list[PocketBlob],
) -> tuple[list[PocketBlob], list[PocketBlob]]:
    """Sort candidates into four corner pockets and the side pockets.

    By position within the cloud's own rotated bounding box, so it holds for a
    camera mounted at any angle to the table. Corners are the candidates nearest
    the box's four corners; whatever is left near the middle of a long rail is a
    side pocket.

    Returns:
        ``(corners, sides)``. ``corners`` is clockwise from top-left when four
        were found, and may be shorter -- an occluded pocket is a normal event
        and the caller reconstructs from what it has.
    """
    import cv2

    if len(pockets) < 4:
        return list(pockets), []

    centers = np.array([[blob.center.x, blob.center.y] for blob in pockets], dtype=np.float32)
    box = order_corners(cv2.boxPoints(cv2.minAreaRect(centers)).astype(np.float64))

    corners: list[PocketBlob] = []
    used: set[int] = set()
    for corner in box:
        best, best_distance = None, float("inf")
        for index, blob in enumerate(pockets):
            if index in used:
                continue
            distance = math.hypot(blob.center.x - corner[0], blob.center.y - corner[1])
            if distance < best_distance:
                best, best_distance = index, distance
        if best is not None:
            used.add(best)
            corners.append(pockets[best])

    sides = [blob for index, blob in enumerate(pockets) if index not in used]
    return corners, sides


def dynamic_detect_table_size(
    frame: np.ndarray,
    pockets: list[PocketBlob],
    settings: Settings | None = None,
    scale: tuple[float, str] | None = None,
) -> TableMeasurement | None:
    """Measure the table from detected pockets.

    Fits a quad to the pocket cloud, matches each of the six expected positions
    to a candidate, and refines the quad from the matches -- twice, because the
    first quad comes from a bounding box that includes any outliers and the
    second comes only from pockets that matched.

    Args:
        frame: The full-resolution BGR frame, needed to look for balls when the
            scale is being taken from one.
        pockets: Candidates from :func:`detect_pockets_loose`.
        settings: Config. Defaults to the global settings.
        scale: A ``(pixels_per_inch, source)`` pair to use instead of resolving
            one. The two-pass pipeline passes the cheap configured scale on its
            first pass and resolves properly on the second -- finding balls is
            the most expensive thing in this module by a wide margin, and the
            first pass only needs a quad, not a size.

    Returns:
        A :class:`TableMeasurement`, or ``None`` when the candidates do not form
        anything table-shaped.
    """
    settings = settings or get_settings()
    if len(pockets) < 4:
        logger.debug("need 4 pocket candidates to measure a table, have %d", len(pockets))
        return None

    quad = _fit_quad(pockets)
    if quad is None:
        return None

    matched: dict[PocketId, PocketBlob] = {}
    for _pass in range(3):
        matched = _match_to_layout(quad, pockets)
        refined = _quad_from_matched(matched, quad)
        if refined is None:
            break
        quad = refined

    corner_count = sum(1 for pocket_id in _CORNER_IDS if pocket_id in matched)
    if corner_count < 3:
        logger.debug("only %d corner pocket(s) matched; not a table", corner_count)
        return None

    length_px, width_px = _quad_extents(quad)
    if length_px < 1.0 or width_px < 1.0:
        return None

    aspect = length_px / width_px
    tolerance = settings.vision.table_aspect_tolerance
    if abs(aspect - _TABLE_ASPECT) / _TABLE_ASPECT > tolerance:
        logger.debug(
            "rejecting pocket quad: aspect %.2f vs the 2:1 every pool table has "
            "(tolerance %.0f%%)",
            aspect,
            100.0 * tolerance,
        )
        return None

    radii = [blob.radius_px for blob in matched.values()]
    pocket_radius_px = float(np.median(radii)) if radii else length_px * 0.03
    pixels_per_inch, scale_source = (
        scale if scale is not None else resolve_scale(frame, quad, length_px, settings, matched)
    )

    measurement = TableMeasurement(
        quad=quad,
        length_px=length_px,
        width_px=width_px,
        pocket_spacing_px=_pocket_spacing(matched, length_px),
        pocket_radius_px=pocket_radius_px,
        pixels_per_inch=pixels_per_inch,
        scale_source=scale_source,
        pockets=matched,
    )

    if scale is None and settings.vision.snap_to_standard_size:
        measurement = _snap(measurement, settings)

    logger.debug(
        "pocket quad: %d pocket(s), %.0f x %.0f px, aspect %.2f",
        len(matched),
        length_px,
        width_px,
        measurement.aspect,
    )
    return measurement


def _fit_quad(pockets: list[PocketBlob]) -> np.ndarray | None:
    """Initial table quad: the rotated bounding box of the pocket cloud.

    Approximate on purpose. It only has to be close enough to match candidates
    to layout positions; the rail fit that follows is what makes it accurate.
    """
    import cv2

    centers = np.array([[blob.center.x, blob.center.y] for blob in pockets], dtype=np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(centers)).astype(np.float64)
    return _orient(order_corners(box))


def _orient(quad: np.ndarray) -> np.ndarray:
    """Rotate a clockwise quad so its first edge is the long one.

    Table coordinates define +x as the long axis and the homography maps corner
    0 to (0,0) and corner 1 to (length, 0), so corner 0 -> 1 must be the long
    side. Without this a portrait-framed table solves to a transform with the
    axes swapped, every table coordinate afterwards is wrong, and the symptom is
    a plausible-looking table with zero balls on it.
    """
    side_a = float(np.linalg.norm(quad[1] - quad[0]))
    side_b = float(np.linalg.norm(quad[3] - quad[0]))
    if side_b > side_a:
        return np.roll(quad, -1, axis=0)
    return quad


def _layout_positions(quad: np.ndarray) -> dict[PocketId, np.ndarray]:
    """Where each pocket should be, given a table quad.

    Bilinear across the quad, matching ``vision.detection.pocket_positions`` so
    the two agree under perspective.
    """
    top_left, top_right, bottom_right, bottom_left = quad
    positions: dict[PocketId, np.ndarray] = {}
    for pocket_id, u, v in POCKET_UV:
        top = top_left + (top_right - top_left) * u
        bottom = bottom_left + (bottom_right - bottom_left) * u
        positions[pocket_id] = top + (bottom - top) * v
    return positions


def _match_to_layout(quad: np.ndarray, pockets: list[PocketBlob]) -> dict[PocketId, PocketBlob]:
    """Assign candidates to the six expected pocket positions.

    Each expected position takes the nearest unused candidate within a tolerance
    of a fifth of the short axis. The tolerance is what rejects outliers: a
    shadow halfway down the cloth is near no expected position and simply goes
    unmatched, rather than displacing a real pocket.
    """
    _length, width = _quad_extents(quad)
    tolerance = max(8.0, width * 0.2)
    expected = _layout_positions(quad)

    matched: dict[PocketId, PocketBlob] = {}
    used: set[int] = set()
    # Strongest candidates first, so a real pocket claims its slot before a
    # weaker blob that happens to sit slightly closer.
    order = sorted(range(len(pockets)), key=lambda i: pockets[i].confidence, reverse=True)
    for pocket_id, position in expected.items():
        best, best_distance = None, tolerance
        for index in order:
            if index in used:
                continue
            blob = pockets[index]
            distance = math.hypot(blob.center.x - position[0], blob.center.y - position[1])
            if distance < best_distance:
                best, best_distance = index, distance
        if best is not None:
            used.add(best)
            blob = pockets[best]
            blob.pocket_id = pocket_id
            matched[pocket_id] = blob
    return matched


def _quad_from_matched(
    matched: dict[PocketId, PocketBlob], fallback: np.ndarray
) -> np.ndarray | None:
    """Re-fit the table quad from matched pockets by intersecting the rails.

    The rails are fitted rather than the corners taken directly, and the two
    side pockets are what makes that worth doing: a long rail fitted through
    three points is far better conditioned than a line through two, and the
    corner it produces absorbs the error in any single detection.

    Returns ``None`` when there is not enough to fit, leaving the caller with
    the quad it already had.
    """
    def point(pocket_id: PocketId) -> np.ndarray | None:
        blob = matched.get(pocket_id)
        return None if blob is None else np.array([blob.center.x, blob.center.y])

    def rail(*ids: PocketId) -> list[np.ndarray]:
        return [p for p in (point(pocket_id) for pocket_id in ids) if p is not None]

    top = rail(PocketId.TOP_LEFT, PocketId.TOP_MIDDLE, PocketId.TOP_RIGHT)
    bottom = rail(PocketId.BOTTOM_LEFT, PocketId.BOTTOM_MIDDLE, PocketId.BOTTOM_RIGHT)
    left = rail(PocketId.TOP_LEFT, PocketId.BOTTOM_LEFT)
    right = rail(PocketId.TOP_RIGHT, PocketId.BOTTOM_RIGHT)

    if min(len(top), len(bottom), len(left), len(right)) < 2:
        return None

    rails = {
        "top": fit_line(np.array(top)),
        "bottom": fit_line(np.array(bottom)),
        "left": fit_line(np.array(left)),
        "right": fit_line(np.array(right)),
    }
    corners = [
        intersect_lines(rails["top"], rails["left"]),
        intersect_lines(rails["top"], rails["right"]),
        intersect_lines(rails["bottom"], rails["right"]),
        intersect_lines(rails["bottom"], rails["left"]),
    ]
    if any(corner is None for corner in corners):
        logger.debug("adjacent rails came out parallel; keeping the previous quad")
        return None

    quad = np.array(corners, dtype=np.float64)
    # A rail intersection that lands far outside the pocket cloud means a rail
    # was fitted through the wrong points. The previous quad is worse but sane.
    centroid = np.mean([[b.center.x, b.center.y] for b in matched.values()], axis=0)
    span = float(np.linalg.norm(fallback[2] - fallback[0]))
    if np.any(np.linalg.norm(quad - centroid, axis=1) > span):
        logger.debug("rail intersections landed implausibly far out; keeping the previous quad")
        return None
    return _orient(order_corners(quad))


def _quad_extents(quad: np.ndarray) -> tuple[float, float]:
    """``(long_px, short_px)``, each averaged over its two opposite sides.

    Averaged because perspective makes the near rail image longer than the far
    one, and either alone misstates the table's scale.
    """
    top_left, top_right, bottom_right, bottom_left = quad
    length = (
        float(np.linalg.norm(top_right - top_left)) + float(np.linalg.norm(bottom_right - bottom_left))
    ) / 2.0
    width = (
        float(np.linalg.norm(bottom_left - top_left)) + float(np.linalg.norm(bottom_right - top_right))
    ) / 2.0
    return length, width


def _pocket_spacing(matched: dict[PocketId, PocketBlob], length_px: float) -> float:
    """Mean gap between adjacent pockets along the long rails.

    Falls back to half the long axis, which is what the spacing would be with
    the side pockets where they belong -- so a table whose side pockets were
    missed still gets a usable number rather than a zero that would collapse
    every parameter derived from it.
    """
    pairs = (
        (PocketId.TOP_LEFT, PocketId.TOP_MIDDLE),
        (PocketId.TOP_MIDDLE, PocketId.TOP_RIGHT),
        (PocketId.BOTTOM_LEFT, PocketId.BOTTOM_MIDDLE),
        (PocketId.BOTTOM_MIDDLE, PocketId.BOTTOM_RIGHT),
    )
    gaps = [
        matched[a].center.distance_to(matched[b].center)
        for a, b in pairs
        if a in matched and b in matched
    ]
    return float(np.mean(gaps)) if gaps else length_px / 2.0


def _snap(measurement: TableMeasurement, settings: Settings) -> TableMeasurement:
    """Round a measurement onto the nearest named table size, if it is close.

    Off by default. Measurement noise genuinely does put a 7 ft table at 7.06
    ft, and rounding is right when the table is one of the standard sizes -- but
    the brief also asks for a nominally 8.2 ft table to be reported as 8.2 ft,
    and only the person who owns the table knows which case they are in.
    """
    standard = nearest_standard_length_ft(measurement.length_ft)
    if abs(standard - measurement.length_ft) / standard > settings.vision.snap_tolerance_frac:
        return measurement
    logger.info(
        "snapping measured %.2f ft to the standard %.1f ft (within %.0f%%)",
        measurement.length_ft,
        standard,
        100.0 * settings.vision.snap_tolerance_frac,
    )
    measurement.pixels_per_inch = measurement.length_px / (standard * 12.0)
    measurement.scale_source = f"{measurement.scale_source}+snap"
    return measurement


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def resolve_scale(
    frame: np.ndarray,
    quad: np.ndarray,
    length_px: float,
    settings: Settings | None = None,
    pockets: dict[PocketId, PocketBlob] | None = None,
) -> tuple[float, str]:
    """Establish camera px per inch. See the module docstring for why this is hard.

    Tries, in order of how much the answer can be trusted:

    1. **A ball.** Its diameter is a known constant, so ``table_px / ball_px``
       gives the table in ball-diameters and the camera height cancels. The only
       source that measures rather than assumes.
    2. **Config.** Take ``settings.table.length_in`` at its word. Correct
       whenever the user set the right preset, which is most of the time.
    3. **The fixed reference.** ``reference_table_width_px`` px per
       ``reference_table_length_ft`` ft. Note this yields a *constant* px/inch
       regardless of the image, which is exactly the assumption it encodes: that
       the camera is at the height the reference was taken at.

    Returns:
        ``(pixels_per_inch, source)`` where source is ``ball``, ``config`` or
        ``reference``.
    """
    settings = settings or get_settings()
    preference = settings.vision.scale_source

    if preference in ("auto", "ball"):
        from_ball = _scale_from_balls(frame, quad, length_px, settings, pockets or {})
        if from_ball is not None:
            return from_ball, "ball"
        if preference == "ball":
            logger.warning(
                "scale_source is 'ball' but no balls were found; falling back to the "
                "configured table size. Put a couple of balls on the cloth to measure it."
            )

    if preference in ("auto", "ball", "config"):
        return length_px / settings.table.length_in, "config"

    reference_px_per_inch = settings.vision.reference_table_width_px / (
        settings.vision.reference_table_length_ft * 12.0
    )
    logger.warning(
        "using the fixed camera-height reference for scale (%.1f px/in). This is only "
        "valid at the height it was calibrated at; move the camera and the same table "
        "measures a different size.",
        reference_px_per_inch,
    )
    return reference_px_per_inch, "reference"


#: Roundness a blob must reach to be measured as a ball for scale purposes.
#: Stricter than ``vision.min_ball_circularity``, which exists to *find* balls
#: and rightly tolerates a partly-occluded one. Here a marginal blob is not
#: worth having: only two clean balls are needed, and a crescent measured as a
#: ball moves the reported table size by feet. Real balls score 0.89+; the
#: clipped pocket mouths that share the search score 0.73-0.75.
_SCALE_BALL_MIN_CIRCULARITY = 0.85


def _scale_from_balls(
    frame: np.ndarray,
    quad: np.ndarray,
    length_px: float,
    settings: Settings,
    pockets: dict[PocketId, PocketBlob],
) -> float | None:
    """Camera px per inch from the imaged size of the balls on the cloth.

    Colour-agnostic, which it has to be: the whole point of this module is
    working on cloth the configured felt thresholds do not match, so finding
    balls by inverting the felt mask would fail on exactly the tables that need
    this most. Instead the cloth is whatever colour most of the table is -- see
    :func:`adaptive_cloth_mask` -- and a ball is a round thing that is not that.

    Accuracy is about 2%, and the bias has a known cause: a ball's antialiased
    edge reads as not-cloth, so the blob is around half a pixel fat all the way
    round. On a 23 px radius that is +2%, and it is why the radius is taken as
    ``sqrt(area/pi)`` rather than from ``minEnclosingCircle`` -- the enclosing
    circle is set by the outermost stray pixel and measured +6% on the same
    balls.

    Returns ``None`` rather than a guess when fewer than
    :data:`_MIN_BALLS_FOR_SCALE` balls agree on a radius.
    """
    import cv2

    if frame is None or frame.size == 0:
        return None

    small, scale = downscale_for_detection(frame, settings.vision.detection_width)
    quad_small = quad * scale
    cloth = adaptive_cloth_mask(small, quad_small, settings)
    if cloth is None:
        return None

    low_frac, high_frac = _BALL_RADIUS_FRAC_RANGE
    min_radius = low_frac * length_px * scale
    max_radius = high_frac * length_px * scale

    interior = _quad_mask(small.shape[:2], quad_small, inset=min_radius * 0.5)
    # Punch out the pocket mouths, for the same reason ball detection does: a
    # pocket is dark, round and ball-sized, so it is not cloth and therefore
    # looks exactly like a ball. Worse than a false ball, a pocket clipped by
    # the interior edge becomes a crescent whose area radius is two thirds of
    # its true one -- measured here as four phantom balls that dragged the
    # median radius down and the reported table size up by 17%.
    for blob in pockets.values():
        cv2.circle(
            interior,
            (int(round(blob.center.x * scale)), int(round(blob.center.y * scale))),
            int(round(blob.radius_px * scale + max_radius)),
            0,
            thickness=-1,
        )

    candidates = cv2.bitwise_and(cv2.bitwise_not(cloth), interior)
    k = max(3, int(min_radius * 0.7) | 1)
    candidates = cv2.morphologyEx(
        candidates, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )

    contours, _ = cv2.findContours(candidates, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    radii: list[float] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        if area <= 0.0 or perimeter <= 0.0:
            continue
        if 4.0 * math.pi * area / (perimeter * perimeter) < _SCALE_BALL_MIN_CIRCULARITY:
            continue
        radius = math.sqrt(area / math.pi)
        if min_radius <= radius <= max_radius:
            radii.append(float(radius))

    if len(radii) < _MIN_BALLS_FOR_SCALE:
        logger.debug("found %d ball(s); need %d to fix the scale", len(radii), _MIN_BALLS_FOR_SCALE)
        return None

    # Cluster around the median before averaging. A cue butt or a chalk cube can
    # pass the circularity gate, and one outlier in a set of three would move
    # the measured table size by feet.
    median = float(np.median(radii))
    consistent = [r for r in radii if abs(r - median) <= 0.2 * median]
    if len(consistent) < _MIN_BALLS_FOR_SCALE:
        logger.debug("ball radii disagree (%s px); not trusting them for scale", radii)
        return None

    radius_full = float(np.median(consistent)) / scale
    pixels_per_inch = radius_full / BALL_RADIUS_IN
    logger.info(
        "scale from %d ball(s): radius %.1f px -> %.2f px/in (table reads %.2f ft)",
        len(consistent),
        radius_full,
        pixels_per_inch,
        length_px / pixels_per_inch / 12.0,
    )
    return pixels_per_inch


def adaptive_cloth_mask(
    frame: np.ndarray, quad: np.ndarray, settings: Settings | None = None
) -> np.ndarray | None:
    """Mask of "whatever colour this table's cloth is", in the frame's own scale.

    Takes the median hue, saturation and value inside the table and keeps
    everything close to it. No assumption that cloth is green -- which is the
    point, since a red table is exactly the case felt thresholds cannot serve.

    The one cloth colour this handles poorly is black, where the 8 ball is the
    same colour as the table. That is a real limitation and there is no colour
    trick that fixes it; geometry does, and ball detection already leans on
    circularity for the same reason.

    Args:
        frame: BGR frame, any resolution.
        quad: Table corners in *this frame's* coordinates.
        settings: Config. Defaults to the global settings.

    Returns:
        A uint8 mask, or ``None`` if the quad encloses no pixels.
    """
    import cv2

    settings = settings or get_settings()
    interior = _quad_mask(frame.shape[:2], quad, inset=0.0)
    if not interior.any():
        return None

    # Everything below is per-pixel work, and only the table matters -- outside
    # it the "cloth colour" was never sampled and the answer means nothing. On a
    # typical frame the table is around half the area, so cropping to it roughly
    # halves the cost of the most expensive step in the module.
    x0, y0, box_w, box_h = cv2.boundingRect(interior)
    roi = frame[y0 : y0 + box_h, x0 : x0 + box_w]
    roi_interior = interior[y0 : y0 + box_h, x0 : x0 + box_w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sample = hsv[roi_interior > 0]
    if sample.size == 0:
        return None
    median = np.median(sample, axis=0)

    hue, saturation, value = hsv[:, :, 0].astype(np.int16), hsv[:, :, 1], hsv[:, :, 2]
    # Hue is circular on 0-179, so the distance has to wrap: red cloth sits at
    # both ends of the scale and a plain subtraction would split it in half.
    hue_distance = np.abs(hue - int(median[0]))
    hue_distance = np.minimum(hue_distance, 180 - hue_distance)

    mask = (
        (hue_distance <= 14)
        & (np.abs(saturation.astype(np.int16) - int(median[1])) <= 80)
        & (np.abs(value.astype(np.int16) - int(median[2])) <= 80)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    full = np.zeros(frame.shape[:2], dtype=np.uint8)
    full[y0 : y0 + box_h, x0 : x0 + box_w] = mask
    return full


def _quad_mask(shape: tuple[int, int], quad: np.ndarray, inset: float) -> np.ndarray:
    """Filled mask of a quad, optionally shrunk toward its centroid."""
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    corners = np.asarray(quad, dtype=np.float64)
    if inset > 0.0:
        centroid = corners.mean(axis=0)
        vectors = corners - centroid
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        corners = corners - vectors / norms * inset
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    return mask


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------


def detect_table_boundaries_dynamic(
    frame: np.ndarray, settings: Settings | None = None
) -> TableBoundary | None:
    """Find and measure the table from its pockets. Any cloth colour, any size.

    The full workflow: loose detection, measure, scale the parameters to what
    was measured, detect again, re-measure, build the boundary.

    The second pass is not ceremony. The loose pass has to accept radii spanning
    nearly twenty to one, which lets through shadows and dark rail sections; the
    refined pass knows the pocket radius to within a few px and rejects all of
    it. When the second pass finds *fewer* pockets than the first -- which
    happens on a table with unusually varied pocket sizes -- the first
    measurement is kept, since a worse-conditioned fit from more points beats a
    better-conditioned one from three.

    Returns:
        A :class:`~app.models.TableBoundary` with ``length_ft``, ``width_ft``
        and ``pixels_per_ft`` populated, or ``None`` if no table was found.
        Callers must treat ``None`` as "keep the previous boundary".
    """
    settings = settings or get_settings()
    if frame is None or frame.size == 0:
        logger.warning("detect_table_boundaries_dynamic called with an empty frame")
        return None

    loose = detect_pockets_loose(frame, settings)
    if len(loose) < 4:
        logger.debug("pocket detection found only %d candidate(s); no table", len(loose))
        return None

    # Both geometry passes run on a placeholder scale. Resolving the real one
    # means hunting for balls at full detection resolution, which is 16 ms of a
    # 25 ms pipeline -- worth doing once, on the final quad, and not at all on a
    # pass whose only output is a set of corners.
    placeholder = (1.0, "pending")
    measurement = dynamic_detect_table_size(frame, loose, settings, placeholder)
    if measurement is None:
        return None

    params = get_dynamic_hough_params(measurement, settings)
    refined = detect_pockets_refined(frame, params, settings)
    if len(refined) >= 4:
        second = dynamic_detect_table_size(frame, refined, settings, placeholder)
        if second is not None and len(second.pockets) >= len(measurement.pockets):
            measurement = second
        else:
            logger.debug("refined pass did not improve on the loose one; keeping it")

    measurement.pixels_per_inch, measurement.scale_source = resolve_scale(
        frame, measurement.quad, measurement.length_px, settings, measurement.pockets
    )
    if settings.vision.snap_to_standard_size:
        measurement = _snap(measurement, settings)
    logger.info(
        "table measured from %d pocket(s): %.2f x %.2f ft (%.0f x %.0f px), "
        "%.1f px/in via %s, nearest standard %.1f ft",
        len(measurement.pockets),
        measurement.length_ft,
        measurement.width_ft,
        measurement.length_px,
        measurement.width_px,
        measurement.pixels_per_inch,
        measurement.scale_source,
        nearest_standard_length_ft(measurement.length_ft),
    )
    return _boundary_from_measurement(measurement, settings)


def _boundary_from_measurement(
    measurement: TableMeasurement, settings: Settings
) -> TableBoundary:
    """Package a measurement as the boundary every other stage consumes."""
    quad = measurement.quad
    top_left, top_right, bottom_right, bottom_left = (
        Vec2(float(x), float(y)) for x, y in quad
    )
    center = Vec2(float(quad[:, 0].mean()), float(quad[:, 1].mean()))

    # Confidence blends how many pockets were found with how close the quad is
    # to the 2:1 every pool table has. Six pockets on a 2:1 quad is as sure as
    # this method gets; four pockets on a lopsided quad is a guess.
    found = len(measurement.pockets) / 6.0
    aspect_error = abs(measurement.aspect - _TABLE_ASPECT) / _TABLE_ASPECT
    aspect_score = max(0.0, 1.0 - aspect_error / max(settings.vision.table_aspect_tolerance, 1e-6))
    confidence = float(np.clip(0.55 * found + 0.45 * aspect_score, 0.0, 1.0))

    boundary = TableBoundary(
        top_left=top_left,
        top_right=top_right,
        bottom_right=bottom_right,
        bottom_left=bottom_left,
        center=center,
        width_px=measurement.length_px,
        height_px=measurement.width_px,
        confidence=confidence,
        length_ft=measurement.length_ft,
        width_ft=measurement.width_ft,
        pixels_per_ft=measurement.pixels_per_ft,
        scale_source=measurement.scale_source,
        detection_method="pockets",
    )
    logger.info(
        "table detected from pockets: %.0fx%.0f px = %.2fx%.2f ft, confidence %.2f",
        boundary.width_px,
        boundary.height_px,
        measurement.length_ft,
        measurement.width_ft,
        confidence,
    )
    return boundary


# ---------------------------------------------------------------------------
# Feeding the measurement back into the rest of the system
# ---------------------------------------------------------------------------


def report_size_disagreement(settings: Settings, boundary: TableBoundary) -> bool:
    """Log it when the measured table is not the configured one. Changes nothing.

    Always safe to call, and the reason adoption can stay off by default: the
    user finds out that their config says 7 ft while the table measures 9 ft,
    with the nearest standard size named, and can fix ``table_preset`` in one
    line. Silence here would leave every physics prediction scaled wrong with
    nothing to suggest why.

    Returns whether a disagreement was worth reporting.
    """
    measured_length_in = boundary.length_in()
    if measured_length_in is None or not boundary.scale_source.startswith("ball"):
        return False

    current = settings.table.length_in
    if abs(measured_length_in - current) / current <= settings.vision.adopt_table_size_tolerance:
        return False

    measured_ft = measured_length_in / 12.0
    logger.warning(
        "table measures %.2f ft on the long axis but config says %.2f ft. "
        "If that is wrong, set table_preset (nearest standard size: %.1f ft). "
        "Measurement is +/-2%% at best and +/-8%% with a high camera, so it is "
        "advice rather than an instruction.",
        measured_ft,
        current / 12.0,
        nearest_standard_length_ft(measured_ft),
    )
    return True


def adopt_measured_table_size(
    settings: Settings, boundary: TableBoundary, *, tolerance_frac: float | None = None
) -> bool:
    """Update the configured table size from a measurement, if it disagrees.

    Physics, the renderer and the projector mapper all build their geometry from
    ``settings.table``, so measuring an 8 ft table while the config says 7 ft
    leaves every prediction scaled wrong by 15%. This closes that gap -- when
    the user asks for it.

    **Opt-in**, via ``vision.adopt_measured_table_size``, and off by default. A
    ball-derived size is accurate to about 2% at best and 8% when the ball
    images small, while adjacent standard tables are only 7% apart. So this
    cannot reliably tell a 7 ft table from a 7.5 ft one, and guessing wrong
    silently rescales the whole system. :func:`report_size_disagreement` is the
    default behaviour instead: say so, change nothing.

    Deliberately **not** called from the detector. Table detection runs on an
    interval, and a detector that rewrites global settings would let the table
    change size mid-shot on a frame where a player leaned over a pocket. The
    caller decides when a measurement is worth adopting, and does it once; see
    ``app.main``.

    Only adopts a ``ball``-sourced measurement. A ``config``-sourced one is
    circular by construction -- it was derived *from* ``settings.table`` -- and a
    ``reference``-sourced one is only as good as the camera height.

    Returns:
        Whether the settings were changed.
    """
    if not settings.vision.adopt_measured_table_size:
        return False
    if not boundary.is_measured or not boundary.scale_source.startswith("ball"):
        return False

    measured_length_in = boundary.length_in()
    if measured_length_in is None:
        return False

    tolerance = (
        settings.vision.adopt_table_size_tolerance
        if tolerance_frac is None
        else tolerance_frac
    )
    current = settings.table.length_in
    if abs(measured_length_in - current) / current <= tolerance:
        return False

    logger.warning(
        "adopting measured table size %.1f in (%.2f ft) over the configured %.1f in "
        "(%.2f ft). Physics, rendering and projection all rescale to match.",
        measured_length_in,
        measured_length_in / 12.0,
        current,
        current / 12.0,
    )
    settings.table.length_in = measured_length_in
    # The short axis is taken from the same 2:1 the aspect check enforced rather
    # than from its own measurement: the short axis spans fewer pixels, so its
    # relative error is roughly double, and a table that is not 2:1 was rejected
    # long before reaching here.
    settings.table.width_in = measured_length_in / _TABLE_ASPECT
    return True


def ball_diameter_note() -> str:
    """One line on the assumption the ball-based scale rests on.

    Exposed as a function so the calibration wizard and the web panel can show
    the same sentence rather than each inventing their own wording.
    """
    return (
        f"Table size is measured against a {BALL_DIAMETER_IN:g} in ball. "
        "On a table with non-regulation balls, set vision.scale_source to "
        "'config' and pick the right table_preset."
    )
