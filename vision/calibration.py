"""Table detection and the camera <-> table coordinate mapping.

Two separable jobs live here:

1. **Finding the table** in a camera frame (``detect_table_boundaries``).
2. **Mapping between camera pixels and table inches** once the four corners are
   known -- pure projective geometry.

Table detection runs on a downscaled frame and is called on an interval, not per
frame: the table does not move, and felt segmentation at full resolution costs
~18 ms of a 33 ms budget. See ``vision.detection`` for the per-frame path.

The homography maps camera px -> table inches. Its inverse goes the other way.
Both are kept because the calibration UI needs to draw table-space guides back
into the camera preview.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from app.config import Settings, get_settings
from app.models import TableBoundary, Vec2

logger = logging.getLogger(__name__)


class CalibrationError(RuntimeError):
    """Raised when a transform cannot be solved from the given correspondences."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def downscale_for_detection(
    frame: np.ndarray, target_width: int
) -> tuple[np.ndarray, float]:
    """Shrink a frame for detection and report the scale factor applied.

    The scale factor is the multiplier from full-resolution px to downscaled px,
    so dividing a detected coordinate by it returns to full resolution. Every
    detector in this package works this way, and every one of them must scale
    its results back before returning -- a detector that forgets produces
    coordinates that are silently wrong by a constant factor, which looks like a
    calibration problem rather than a bug.

    ``INTER_AREA`` is the correct interpolation for downscaling: it averages the
    source pixels in each destination footprint, which is exactly the
    antialiasing needed before colour-thresholding. ``INTER_LINEAR`` samples too
    sparsely and makes the felt mask noisy along the rails.

    Returns:
        ``(downscaled_frame, scale)``. When the frame is already at or below the
        target width, the original array is returned with a scale of ``1.0`` --
        no copy, no work.
    """
    import cv2

    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0

    scale = target_width / float(width)
    new_size = (target_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), scale


def boundary_interior_mask(
    shape: tuple[int, int], boundary: TableBoundary, inset_px: float = 0.0
) -> np.ndarray:
    """Filled mask of the table interior, for restricting a search to the cloth.

    The cheapest and most effective false-positive filter in the whole pipeline:
    pockets, rails, the floor and anything a player is holding are all outside
    this mask, and none of them can be mistaken for a ball once it is applied.

    Args:
        shape: ``(height, width)`` of the mask to build, matching the frame the
            search runs on.
        boundary: Table corners, in the same coordinate space as ``shape``.
        inset_px: Shrink the polygon toward its centre by this many pixels.
            Worth a ball radius or so: a ball frozen against the cushion is
            still in play, but the cushion itself is a strong edge that generates
            spurious detections right at the boundary.
    """
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    corners = np.array([[c.x, c.y] for c in boundary.corners()], dtype=np.float64)
    if inset_px > 0:
        centroid = corners.mean(axis=0)
        vectors = corners - centroid
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Guard a degenerate quad whose corners sit on the centroid.
        norms = np.where(norms < 1e-6, 1.0, norms)
        corners = corners - vectors / norms * inset_px
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    return mask


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------


def felt_mask(frame: np.ndarray, settings: Settings | None = None) -> np.ndarray:
    """Binary mask of the green felt.

    Shared by table detection and by ball detection, which needs the *inverse*
    -- balls are the non-felt blobs inside the table. Computing it once per frame
    and passing it around is worth real time at 30 FPS.

    Morphology order is close-then-open, and the order matters. Closing first
    bridges the holes punched by balls and the cue so the felt reads as one
    region; opening afterwards removes specular speckle without re-opening those
    holes. Doing it the other way round fragments the mask.
    """
    import cv2

    settings = settings or get_settings()
    vision = settings.vision
    hue_low, hue_high = vision.felt_hue_range

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # The saturation *ceiling* is what separates a green ball from green felt.
    # Hue cannot: the 6 ball's green sits squarely inside any felt hue range wide
    # enough to cope with real cloth. But felt is matte woven wool and a ball is
    # glossy phenolic resin, so the ball is markedly more saturated. Without this
    # ceiling the 6 ball is classified as part of the table and disappears.
    mask = cv2.inRange(
        hsv,
        np.array([hue_low, vision.felt_sat_min, vision.felt_val_min], dtype=np.uint8),
        np.array([hue_high, vision.felt_sat_max, 255], dtype=np.uint8),
    )
    # Kernel scaled to the frame: a fixed 9 px kernel behaves completely
    # differently at 640 px wide than at 1920.
    k = max(3, (frame.shape[1] // 160) | 1)  # odd, ~6 px at 960 wide
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points clockwise starting from the top-left.

    Sorting by angle about the centroid, rather than the common
    sum/difference trick. The sum/difference method silently produces the wrong
    order once the table is rotated more than ~30 degrees in frame, and a camera
    bolted to a ceiling joist is rarely square to the table. A mis-ordered quad
    yields a homography that mirrors or rotates the whole projection, which is
    an unpleasant bug to chase.

    Args:
        points: A 4x2 array of ``(x, y)`` in camera px.

    Returns:
        A 4x2 array, clockwise from top-left.
    """
    centroid = points.mean(axis=0)
    # Image y grows downward, so atan2 angles increase clockwise on screen.
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    clockwise = points[np.argsort(angles)]

    # Rotate the cycle so it starts at the corner nearest the frame origin.
    # Among the two "upper" corners this reliably picks the left one.
    start = int(np.argmin(clockwise.sum(axis=1)))
    return np.roll(clockwise, -start, axis=0)


def fit_line(points: np.ndarray) -> tuple[float, float, float]:
    """Total-least-squares line through points, as ``(a, b, c)`` for ax+by+c=0.

    Public because :mod:`vision.pockets` fits table rails through detected
    pocket centres and needs exactly this. Two implementations of a line fit
    that must agree on what a rail is would be one too many.

    Orthogonal regression via the principal axis, not ``y = mx + b`` least
    squares. Ordinary regression has no answer for a vertical line, and two of
    the four table rails are near-vertical in a landscape frame.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Principal direction = eigenvector of the larger eigenvalue.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    # Normal is perpendicular to the direction.
    a, b = -direction[1], direction[0]
    c = -(a * centroid[0] + b * centroid[1])
    return float(a), float(b), float(c)


def intersect_lines(
    line1: tuple[float, float, float], line2: tuple[float, float, float]
) -> np.ndarray | None:
    """Intersection of two lines in ``ax+by+c=0`` form, or ``None`` if parallel."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return np.array([(b1 * c2 - b2 * c1) / det, (a2 * c1 - a1 * c2) / det])


def _order_rails(
    lines: list[tuple[float, float, float]], centroid: np.ndarray
) -> list[tuple[float, float, float]]:
    """Order four rail lines so consecutive entries are adjacent, not opposite.

    The corners are intersections of *adjacent* rails, so the ordering has to be
    rotational. Sorting by the direction of each line's outward normal achieves
    that, and makes consecutive pairs adjacent by construction.
    """
    keyed = []
    for a, b, c in lines:
        norm = max(math.hypot(a, b), 1e-9)
        normal = np.array([a, b]) / norm
        # The signed distance from the centroid tells us which way is "out".
        if (a * centroid[0] + b * centroid[1] + c) / norm > 0:
            normal = -normal
        keyed.append((math.atan2(normal[1], normal[0]), (a, b, c)))
    keyed.sort(key=lambda item: item[0])
    return [line for _angle, line in keyed]


def _outward_normal(
    line: tuple[float, float, float], centroid: np.ndarray
) -> np.ndarray:
    """Unit normal of a line, oriented away from a reference point."""
    a, b, c = line
    norm = max(math.hypot(a, b), 1e-9)
    normal = np.array([a, b], dtype=np.float64) / norm
    if (a * centroid[0] + b * centroid[1] + c) / norm > 0:
        normal = -normal
    return normal


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Recover the table's four corners from a felt contour.

    The problem is not finding a quadrilateral -- it is that the felt is *not*
    one. Three properties of a real table defeat the obvious approaches, and
    each cost measurable accuracy before being handled:

    **The corner pockets cut the corners off.** A corner pocket is a hole centred
    on the corner, so the cloth never reaches the point the homography needs; the
    outline is chamfered by about a pocket radius. Reading polygon vertices
    directly lands 30-50 px inside the true corner and no approximation tolerance
    fixes it. The rails must be fitted as lines and *intersected*, reconstructing
    a corner that is absent from the image. This is what the spec's Hough-lines
    suggestion is really for.

    **Anything that removes cloth biases a fit inward.** Pocket notches, a cue
    crossing the rail, a hand on the cushion -- all subtract, none add. So a rail
    is defined here as the *outer envelope* of contour points along its own
    outward normal, not as a fit through all points near it. Fitting through all
    of them pulled rails inward by ~25 px.

    **Keystone means the rails are not axis-aligned to any rectangle.** Each rail
    is therefore re-fitted against its own current direction, iteratively:
    ``minAreaRect`` supplies the initial four directions (exact for rotation,
    wrong for keystone) and three passes converge on the true rails.

    Selecting by direction rather than by clustering points into groups is the
    key simplification. An earlier version assigned points to their nearest rail
    and refitted; on a frame where a cue crossed the cushion, two of the four
    rails captured each other's points and the reconstructed quad came out
    1231x2464 px -- taller than the frame. Defining each rail as "the outermost
    points in this direction" cannot mix two rails up, because opposite rails
    have opposite normals.

    Returns:
        A 4x2 array of corners in the contour's own coordinate space, or ``None``
        if the shape could not be resolved.
    """
    import cv2

    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) < 4:
        return None

    extent = points.max(axis=0) - points.min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    if diagonal < 1e-6:
        return None
    centroid = points.mean(axis=0)

    box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32))).astype(np.float64)
    # Four seed lines from the rectangle's four edges. Exact under rotation,
    # approximate under keystone -- which the iteration below corrects.
    lines = [fit_line(np.array([box[i], box[(i + 1) % 4]])) for i in range(4)]

    # Anneal the selection band from fat to thin. Both ends are necessary and
    # the reason is worth stating, because it is not obvious:
    #
    # It must *start* fat. Under keystone the seed direction is wrong by a few
    # degrees, which spreads a rail's own points across a wide range of
    # projections; a thin band would capture only one end of the rail and the
    # fit would pivot.
    #
    # It must *end* thin. A band is a wedge of depth `band`, and the chamfered
    # corners inside it curve inward, so a least-squares fit through the whole
    # band settles at roughly its mid-depth -- inset by band/2. A fixed 4% band
    # inset every rail and cost 33 px of corner error even when the seed
    # rectangle was pixel-exact.
    for band_factor in (0.05, 0.02, 0.008, 0.004):
        refit: list[tuple[float, float, float]] = []
        for line in lines:
            normal = _outward_normal(line, centroid)
            projections = points @ normal
            span = float(projections.max() - projections.min())
            if span < 1e-6:
                refit.append(line)
                continue
            band = max(1.5, band_factor * span)
            keep = points[projections >= projections.max() - band]
            refit.append(fit_line(keep) if len(keep) >= 4 else line)
        lines = refit

    # Order the rails rotationally so consecutive pairs are adjacent, since the
    # corners are intersections of adjacent rails.
    lines = _order_rails(lines, centroid)

    corners: list[np.ndarray] = []
    for i in range(4):
        point = intersect_lines(lines[i], lines[(i + 1) % 4])
        if point is None:
            logger.debug("adjacent rails %d/%d are parallel; using minAreaRect", i, i + 1)
            return box
        corners.append(point)

    quad = np.array(corners)
    # A reconstructed corner legitimately sits outside the cloth -- that is the
    # whole point -- but only by about a pocket radius. Anything further means
    # the rails were mis-fitted, and the honest rectangle beats a wild quad.
    if np.any(np.linalg.norm(quad - centroid, axis=1) > 0.75 * diagonal):
        logger.debug("rail intersections landed implausibly far out; using minAreaRect")
        return box
    return quad


def detect_table_boundaries(
    frame: np.ndarray, settings: Settings | None = None
) -> TableBoundary | None:
    """Locate the playing surface in a camera frame.

    The entry point every other stage calls, and a dispatcher over two
    detectors selected by ``vision.table_detection_method``:

    ``pockets``
        :func:`vision.pockets.detect_table_boundaries_dynamic`. Finds the six
        pocket mouths and reconstructs the rails, which works on any cloth
        colour and additionally *measures* the table in feet.
    ``felt``
        :func:`detect_table_boundaries_by_felt`, below. Segments green cloth.
        Cheaper and unbothered by occluded pockets, but blind to a table that
        is not the configured colour.
    ``auto`` (default)
        Pockets first, felt as a fallback. Pocket detection needs six visible
        mouths and felt detection does not, so falling back the other way round
        would give up the colour independence for no gain.

    The return type is unchanged either way, which is what keeps every later
    stage from caring which one ran. A pocket-derived boundary carries extra
    populated fields -- ``length_ft``, ``pixels_per_ft`` -- and a felt-derived
    one leaves them ``None``.
    """
    settings = settings or get_settings()
    method = settings.vision.table_detection_method

    if method in ("auto", "pockets"):
        from vision.pockets import detect_table_boundaries_dynamic

        boundary = detect_table_boundaries_dynamic(frame, settings)
        if boundary is not None:
            return boundary
        if method == "pockets":
            return None
        logger.debug("pocket detection found no table; falling back to felt segmentation")

    return detect_table_boundaries_by_felt(frame, settings)


def detect_table_boundaries_by_felt(
    frame: np.ndarray, settings: Settings | None = None
) -> TableBoundary | None:
    """Locate the playing surface by segmenting the cloth colour.

    Colour, not edges. Felt is the largest saturated-green region in an overhead
    shot, and under a projector that is painting bright lines across the cloth,
    a colour cue survives where edge detection fills with spurious lines.

    The original detector, kept because it is the better choice on some tables
    and the only choice on a table whose pockets the camera cannot see -- a
    tight overhead crop, or a bar table with recessed returns. What it cannot
    do is work on cloth outside ``vision.felt_hue_range``, which is why it is no
    longer the default; see :func:`detect_table_boundaries`.

    Runs on a frame downscaled to ``vision.detection_width`` and scales the
    corners back up, which is a ~4x saving at no useful cost in accuracy -- the
    table edge is a metre-long straight line, so locating it to the nearest
    couple of full-resolution pixels is ample.

    Two acceptance checks reject a mask that has leaked off the table, which is
    the realistic failure mode (a green carpet, a plant, a green wall):

    * **Area** -- the quad must cover at least ``vision.table_min_area_pct`` of
      the frame.
    * **Aspect ratio** -- it must be within ``vision.table_aspect_tolerance`` of
      the configured table's length:width ratio.

    Returns:
        The boundary in camera px, or ``None`` if no plausible table was found.
        Callers must treat ``None`` as "keep the previous boundary" rather than
        blanking the overlay -- someone leaning over the table should not make
        the projection flicker.
    """
    import cv2

    settings = settings or get_settings()
    if frame is None or frame.size == 0:
        logger.warning("detect_table_boundaries called with an empty frame")
        return None

    small, scale = downscale_for_detection(frame, settings.vision.table_detection_width)
    mask = felt_mask(small, settings)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("no felt region found; check felt_hue_range against the actual cloth")
        return None

    largest = max(contours, key=cv2.contourArea)
    frame_area = float(small.shape[0] * small.shape[1])
    area_pct = 100.0 * cv2.contourArea(largest) / frame_area
    if area_pct < settings.vision.table_min_area_pct:
        logger.debug(
            "largest felt region is only %.1f%% of the frame (need %.1f%%)",
            area_pct,
            settings.vision.table_min_area_pct,
        )
        return None

    quad = _quad_from_contour(largest)
    if quad is None:
        return None

    ordered = order_corners(quad) / scale  # back to full-resolution px

    # Force the long axis onto the table's +x. Table coordinates define +x as
    # the long axis, and compute_perspective_transform maps corner 0 -> (0,0)
    # and corner 1 -> (length, 0), so corner 0 -> 1 *must* be the long side.
    #
    # Without this the table can be accepted rotated 90 degrees: the aspect
    # check compares the ratio either way up (deliberately, since the camera
    # may be mounted sideways), so a portrait quad passes it and then solves to
    # a homography with the axes swapped. Every table coordinate afterwards is
    # wrong, balls fall outside the expected size range, and nothing is
    # detected -- observed as a table reported at 1231x2464 px with zero balls.
    side_a = float(np.linalg.norm(ordered[1] - ordered[0]))
    side_b = float(np.linalg.norm(ordered[3] - ordered[0]))
    if side_b > side_a:
        logger.debug("table found in portrait orientation; rotating corners to landscape")
        ordered = np.roll(ordered, -1, axis=0)

    top_left, top_right, bottom_right, bottom_left = (Vec2(float(x), float(y)) for x, y in ordered)

    # Average the opposite sides: with perspective the near and far rails differ
    # in length, and either one alone misstates the table's scale.
    width_px = (top_left.distance_to(top_right) + bottom_left.distance_to(bottom_right)) / 2.0
    height_px = (top_left.distance_to(bottom_left) + top_right.distance_to(bottom_right)) / 2.0
    if height_px <= 1.0 or width_px <= 1.0:
        logger.debug("degenerate table quad: %.1f x %.1f px", width_px, height_px)
        return None

    # Aspect check. The quad may be found rotated 90 degrees relative to the
    # table's long axis, so compare against the ratio either way up.
    detected_aspect = width_px / height_px
    expected_aspect = settings.table.length_in / settings.table.width_in
    relative_error = min(
        abs(detected_aspect - expected_aspect) / expected_aspect,
        abs(1.0 / detected_aspect - expected_aspect) / expected_aspect,
    )
    if relative_error > settings.vision.table_aspect_tolerance:
        logger.debug(
            "rejecting table: aspect %.2f vs expected %.2f (%.0f%% off, tolerance %.0f%%)",
            detected_aspect,
            expected_aspect,
            100.0 * relative_error,
            100.0 * settings.vision.table_aspect_tolerance,
        )
        return None

    # Confidence blends how rectangular the quad is (its area against its own
    # bounding box) with how well the aspect ratio matches. Both are needed: a
    # skewed quad can have the right aspect, and a correct rectangle can be the
    # wrong object.
    quad_area = _polygon_area(ordered)
    rect_area = max(width_px * height_px, 1e-6)
    rectangularity = min(1.0, quad_area / rect_area)
    aspect_score = max(0.0, 1.0 - relative_error / settings.vision.table_aspect_tolerance)
    confidence = float(np.clip(0.6 * rectangularity + 0.4 * aspect_score, 0.0, 1.0))

    center = Vec2(
        float(ordered[:, 0].mean()),
        float(ordered[:, 1].mean()),
    )
    boundary = TableBoundary(
        top_left=top_left,
        top_right=top_right,
        bottom_right=bottom_right,
        bottom_left=bottom_left,
        center=center,
        width_px=width_px,
        height_px=height_px,
        confidence=confidence,
        detection_method="felt",
    )
    logger.info(
        "table detected by felt: %.0fx%.0f px, aspect %.2f, confidence %.2f",
        width_px,
        height_px,
        detected_aspect,
        confidence,
    )
    return boundary


def create_calibration_markers(
    frame: np.ndarray, boundary: TableBoundary | None = None
) -> np.ndarray:
    """Annotate a camera frame with corner markers, centre and an alignment grid.

    Returns a copy -- the input frame is also handed to the detector, and
    scribbling on it in place would corrupt detection.

    The grid is spaced in *table* coordinates and mapped back into camera space,
    so it appears to lie on the cloth and converge with perspective. A grid of
    evenly spaced screen lines would look wrong on an angled table and invite the
    user to "correct" a distortion that is not there.
    """
    import cv2

    canvas = frame.copy()
    if boundary is None:
        cv2.putText(
            canvas,
            "no table detected",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    corners = boundary.corners()
    outline = np.array([c.as_int() for c in corners], dtype=np.int32)
    cv2.polylines(canvas, [outline], isClosed=True, color=(0, 255, 255), thickness=2)

    # Perspective-correct grid, drawn only when a homography can be solved.
    try:
        _, table_to_camera = compute_perspective_transform(boundary)
    except CalibrationError:
        table_to_camera = None

    if table_to_camera is not None:
        settings = get_settings()
        length, width = settings.table.length_in, settings.table.width_in
        divisions = 8
        # Semi-transparent so the grid guides the eye without hiding the cloth.
        overlay = canvas.copy()
        for i in range(1, divisions):
            fx = length * i / divisions
            a = table_to_camera_coords(Vec2(fx, 0.0), table_to_camera)
            b = table_to_camera_coords(Vec2(fx, width), table_to_camera)
            cv2.line(overlay, a.as_int(), b.as_int(), (160, 160, 160), 1, cv2.LINE_AA)
        for i in range(1, divisions // 2):
            fy = width * i / (divisions // 2)
            a = table_to_camera_coords(Vec2(0.0, fy), table_to_camera)
            b = table_to_camera_coords(Vec2(length, fy), table_to_camera)
            cv2.line(overlay, a.as_int(), b.as_int(), (160, 160, 160), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)

    labels = ("TL", "TR", "BR", "BL")
    for corner, label in zip(corners, labels, strict=True):
        point = corner.as_int()
        cv2.circle(canvas, point, 16, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, point, 2, (0, 220, 255), -1)
        cv2.putText(
            canvas, label, (point[0] + 20, point[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA,
        )

    center = boundary.center.as_int()
    cv2.drawMarker(canvas, center, (255, 120, 200), cv2.MARKER_CROSS, 28, 2)
    cv2.putText(
        canvas,
        f"table {boundary.width_px:.0f}x{boundary.height_px:.0f}px  conf {boundary.confidence:.2f}",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


# ---------------------------------------------------------------------------
# Coordinate transforms (implemented -- pure geometry, no hardware needed)
# ---------------------------------------------------------------------------


def compute_perspective_transform(
    boundary: TableBoundary, settings: Settings | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the camera px <-> table inches homography from four table corners.

    The destination rectangle is the physical playing surface in inches, taken
    from config, with the origin at the top-left cushion. Solving straight to
    inches (rather than to some normalised square) means every downstream
    distance is physically meaningful without a second scale factor to get
    wrong.

    Args:
        boundary: Detected table corners, camera px.
        settings: Config supplying the table dimensions. Defaults to the global.

    Returns:
        ``(camera_to_table, table_to_camera)`` as 3x3 float64 matrices.

    Raises:
        CalibrationError: If the corners are degenerate -- collinear, coincident
            or otherwise not forming a convex quad -- which would make the
            solve singular. This happens in practice when felt segmentation
            leaks and two "corners" land on top of each other, so it must fail
            loudly rather than yield a garbage transform.
    """
    import cv2

    settings = settings or get_settings()
    src = np.array([[c.x, c.y] for c in boundary.corners()], dtype=np.float32)

    length, width = settings.table.length_in, settings.table.width_in
    # Clockwise from top-left, matching TableBoundary.corners().
    dst = np.array(
        [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]],
        dtype=np.float32,
    )

    # Reject degenerate input before handing it to OpenCV: getPerspectiveTransform
    # will happily return a matrix full of inf/nan for collinear points.
    if _polygon_area(src) < 1.0:
        raise CalibrationError(
            f"table corners enclose almost no area ({_polygon_area(src):.2f} px^2); "
            "detection probably failed"
        )

    camera_to_table = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    if not np.all(np.isfinite(camera_to_table)):
        raise CalibrationError("homography solve produced non-finite values")

    table_to_camera = np.linalg.inv(camera_to_table)
    logger.info(
        "solved table homography: %.0fx%.0f px -> %.0fx%.0f in",
        boundary.width_px,
        boundary.height_px,
        length,
        width,
    )
    return camera_to_table, table_to_camera


def _polygon_area(points: np.ndarray) -> float:
    """Shoelace area of a polygon given as an Nx2 array. Used as a degeneracy check."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def transform_point(point: Vec2, matrix: np.ndarray) -> Vec2:
    """Apply a 3x3 homography to a single point.

    Kept separate from the batch version because the per-frame cost of building
    a NumPy array for one point is real, and this is called for the cue tip on
    every frame.

    Raises:
        CalibrationError: If the point maps to the plane at infinity (``w == 0``),
            which means it lies on the horizon of the transform -- geometrically
            behind the table plane. Returning a huge finite number instead would
            silently poison the physics.
    """
    x, y = point
    denom = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if abs(denom) < 1e-12:
        raise CalibrationError(f"point {point} maps to infinity under this transform")
    return Vec2(
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom,
    )


def transform_points(points: list[Vec2], matrix: np.ndarray) -> list[Vec2]:
    """Batch form of :func:`transform_point`.

    Worth using for a whole ball list or trajectory polyline: one vectorised
    call instead of N Python-level ones.
    """
    if not points:
        return []
    array = np.array([[p.x, p.y] for p in points], dtype=np.float64)
    homogeneous = np.hstack([array, np.ones((len(points), 1))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2:3]
    # Guard the same infinity case as the scalar path, but without failing the
    # whole batch: a single bad point in a trajectory should not discard it.
    safe = np.where(np.abs(w) < 1e-12, np.nan, w)
    result = projected[:, :2] / safe
    return [Vec2(float(px), float(py)) for px, py in result]


def camera_to_table_coords(pixel: Vec2, camera_to_table: np.ndarray) -> Vec2:
    """Camera px -> table inches."""
    return transform_point(pixel, camera_to_table)


def table_to_camera_coords(table_point: Vec2, table_to_camera: np.ndarray) -> Vec2:
    """Table inches -> camera px."""
    return transform_point(table_point, table_to_camera)


def pixels_per_inch(boundary: TableBoundary, settings: Settings | None = None) -> float:
    """Mean camera scale over the table, px per inch.

    Only an average -- perspective makes the true scale vary across the frame --
    but it is the right way to seed the expected ball radius for Hough circle
    detection, which needs a search window rather than an exact value.
    """
    settings = settings or get_settings()
    scale_x = boundary.width_px / settings.table.length_in
    scale_y = boundary.height_px / settings.table.width_in
    return (scale_x + scale_y) / 2.0


def expected_ball_radius_px(
    boundary: TableBoundary, settings: Settings | None = None
) -> tuple[int, int]:
    """Ball radius search window in camera px, derived from the table scale.

    Much tighter than the cold-start bounds in config, so detection should use
    this as soon as the table has been found. The +/-25% band absorbs
    perspective variation between the near and far rails.
    """
    settings = settings or get_settings()
    from app.config import BALL_RADIUS_IN

    nominal = pixels_per_inch(boundary, settings) * BALL_RADIUS_IN
    return max(3, int(nominal * 0.75)), int(nominal * 1.25) + 1
