"""Ball, cue-ball, cue-stick and pocket detection.

The per-frame hot path. Everything here has to fit, together with capture and
rendering, inside a 33 ms budget on a Pi 5.

Two design decisions shape the whole module.

**Detection runs on a downscaled frame.** Felt segmentation alone measures
~18 ms at 1920 px wide. At ``vision.detection_width`` (960 by default) the same
work is ~4x cheaper, and a 2.25 in ball is still ~12 px across -- enough to
locate its centre to a fraction of an inch after scaling back up. Every function
here scales its output back to full-resolution camera px before returning.

**Balls are found as non-felt blobs, not by per-colour Hough circles.** The spec
suggests running a circular Hough transform per colour band, which would mean
eight passes over the frame; a single Hough pass alone costs more than the whole
budget at this resolution. Instead the felt mask is inverted inside the table
outline, which yields every ball in one pass regardless of colour, and colour is
then classified per blob by sampling its interior. This is both faster and more
robust -- it finds a ball whose colour is being altered by projected light, which
a colour-keyed search cannot.

A consequence worth stating: this system *projects light onto the thing it is
looking at*. Geometry (roundness, size, continuity between frames) is trusted
over absolute colour wherever there is a choice, and colour is sampled from the
ball centre, which is the least contaminated part of the disc.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from app.config import BALL_DIAMETER_IN, Settings, get_settings
from app.models import (
    Ball,
    BallColor,
    BallKind,
    CueStick,
    GameState,
    Pocket,
    PocketId,
    TableBoundary,
    Vec2,
)
from utils.logging import ChangeLogger
from vision.calibration import (
    CalibrationError,
    boundary_interior_mask,
    camera_to_table_coords,
    downscale_for_detection,
    felt_mask,
    pixels_per_inch,
)
from vision.colors import (
    classify_color,
    classify_stripe_or_solid,
    guess_number,
    sample_ball_hsv,
)

logger = logging.getLogger(__name__)
#: Conditions in the per-frame path that are worth reporting once rather
#: than thirty times a second. See :class:`utils.logging.ChangeLogger`.
_changes = ChangeLogger(logger)

#: Pocket positions in normalised table coordinates, ``(u, v)`` with ``u`` along
#: the long axis. Derived from the table quad rather than found in the image:
#: pockets do not move relative to the cushions, and geometry is far more
#: reliable than dark-region detection, which cheerfully finds shadows.
POCKET_LAYOUT: list[tuple[PocketId, float, float]] = [
    (PocketId.TOP_LEFT, 0.0, 0.0),
    (PocketId.TOP_MIDDLE, 0.5, 0.0),
    (PocketId.TOP_RIGHT, 1.0, 0.0),
    (PocketId.BOTTOM_RIGHT, 1.0, 1.0),
    (PocketId.BOTTOM_MIDDLE, 0.5, 1.0),
    (PocketId.BOTTOM_LEFT, 0.0, 1.0),
]


# ---------------------------------------------------------------------------
# Ball detection
# ---------------------------------------------------------------------------


def pocket_positions(
    boundary: TableBoundary, settings: Settings | None = None
) -> list[tuple[PocketId, Vec2]]:
    """The six pocket centres, in the boundary's own coordinate space.

    Bilinear interpolation across the table quad, so the positions follow
    perspective instead of assuming a rectangle. Shared by pocket detection and
    by ball detection, which has to exclude these regions -- if the two used
    different geometry, balls would be rejected in the wrong places.
    """
    settings = settings or get_settings()
    corners = np.array([[c.x, c.y] for c in boundary.corners()], dtype=np.float64)
    top_left, top_right, bottom_right, bottom_left = corners

    positions: list[tuple[PocketId, Vec2]] = []
    for pocket_id, u, v in POCKET_LAYOUT:
        top = top_left + (top_right - top_left) * u
        bottom = bottom_left + (bottom_right - bottom_left) * u
        point = top + (bottom - top) * v
        positions.append((pocket_id, Vec2(float(point[0]), float(point[1]))))
    return positions


def _cloth_mask(
    small: np.ndarray,
    boundary: TableBoundary | None,
    scale: float,
    settings: Settings,
) -> np.ndarray:
    """The cloth mask to subtract balls from, whatever colour the cloth is.

    Uses the configured felt thresholds, and falls back to an adaptive mask when
    they match almost none of the table interior.

    The fallback exists because pocket-based table detection made a new failure
    mode possible, and it is a nasty one. A red table is now found perfectly
    well -- pockets are holes regardless of cloth colour -- and then ball
    detection, which finds balls by inverting a *green* mask, sees the entire
    table as one enormous non-felt blob and reports zero balls. The system looks
    like it is working right up until nothing is ever detected, which is a far
    more confusing failure than not finding the table at all.

    Triggered on coverage rather than on the configured colour, because the same
    thing happens to a green table under coloured room lighting, and because
    coverage is measurable while "is this the right green" is not.
    """
    import cv2

    cloth = felt_mask(small, settings)
    if boundary is None:
        return cloth

    scaled = _scale_boundary(boundary, scale)
    interior = boundary_interior_mask(small.shape[:2], scaled)
    interior_area = int(np.count_nonzero(interior))
    if interior_area == 0:
        return cloth

    covered = int(np.count_nonzero(cv2.bitwise_and(cloth, interior))) / interior_area
    if covered >= settings.vision.adaptive_cloth_min_coverage:
        _changes.recovered(
            "cloth_mask",
            logging.INFO,
            "configured felt thresholds match the cloth again; adaptive mask no longer needed",
        )
        return cloth

    from vision.pockets import adaptive_cloth_mask

    quad = np.array([[c.x, c.y] for c in scaled.corners()], dtype=np.float64)
    adaptive = adaptive_cloth_mask(small, quad, settings)
    if adaptive is None:
        return cloth

    # Keyed on the constant "adaptive", not on the coverage figure: coverage
    # jitters by a percent or two frame to frame, so keying on it would re-log
    # every frame and defeat the whole point.
    _changes.report(
        "cloth_mask",
        "adaptive",
        logging.INFO,
        "configured felt thresholds cover only %.0f%% of the table; using an "
        "adaptive cloth mask instead. Re-tune felt_hue_range with "
        "`python -m tools.camera_preview --mask` to avoid the extra work.",
        100.0 * covered,
    )
    # Outside the table the adaptive mask means nothing -- it was sampled from
    # the cloth -- so keep the configured mask there. Ball detection clips to the
    # interior anyway; this only matters when it is called without a boundary.
    outside = cv2.bitwise_and(cloth, cv2.bitwise_not(interior))
    return cv2.bitwise_or(cv2.bitwise_and(adaptive, interior), outside)


def _ball_radius_bounds_small(
    boundary: TableBoundary | None, scale: float, settings: Settings
) -> tuple[float, float]:
    """Plausible ball radius range, in downscaled px.

    Derived from the table homography when it is available, which is far tighter
    than any configured guess: the table's pixel width and its known physical
    length give the scale directly. Falls back to the configured cold-start
    bounds before the table has been found.
    """
    if boundary is not None:
        ppi = pixels_per_inch(boundary, settings) * scale
        nominal = ppi * (BALL_DIAMETER_IN / 2.0)
        return max(2.0, nominal * 0.6), nominal * 1.5

    low, high = settings.vision.ball_radius_px_range
    return max(2.0, low * scale), high * scale


def detect_balls(
    frame: np.ndarray,
    boundary: TableBoundary | None = None,
    settings: Settings | None = None,
    *,
    _prepared: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> list[Ball]:
    """Find every ball on the table.

    Args:
        frame: BGR camera frame, full resolution.
        boundary: Table boundary in full-resolution px. Passing it improves both
            accuracy and speed substantially -- it restricts the search to the
            cloth and tightens the expected ball size. ``None`` searches the
            whole frame with looser size bounds.
        settings: Config. Defaults to the global settings.
        _prepared: Internal. Pre-computed ``(small, hsv, scale)`` so
            :func:`extract_game_state` can share one downscale and one HSV
            conversion across every detector instead of repeating them.

    Returns:
        Balls with ``center_px`` and ``radius_px`` in full-resolution camera px.
        ``table_pos`` is left ``None``; filling it in is
        :func:`extract_game_state`'s job, since only it holds the homography.
        Sorted by descending confidence, so a caller taking the best N gets the
        best N.
    """
    import cv2

    settings = settings or get_settings()
    if frame is None or frame.size == 0:
        return []

    if _prepared is not None:
        small, hsv, scale = _prepared
    else:
        small, scale = downscale_for_detection(frame, settings.vision.detection_width)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    cloth = _cloth_mask(small, boundary, scale, settings)

    # Balls are what is *not* felt, inside the table. Inset by a ball radius so
    # the cushion edge itself does not read as an object.
    min_r, max_r = _ball_radius_bounds_small(boundary, scale, settings)
    if boundary is not None:
        scaled = _scale_boundary(boundary, scale)
        interior = boundary_interior_mask(small.shape[:2], scaled, inset_px=min_r * 0.5)

        # Punch out the pocket mouths. A pocket is dark, round and almost
        # exactly ball-sized, so it is a textbook false 8 ball -- measured as
        # three phantom balls at the corners. Insetting the whole boundary far
        # enough to cover them would also reject balls frozen against the
        # cushion, which are legitimately in play, so only the six mouths are
        # removed. A ball inside a pocket mouth has been potted anyway.
        pocket_radius = settings.table.pocket_radius_in * pixels_per_inch(
            boundary, settings
        ) * scale
        exclude = int(round(pocket_radius + min_r))
        for _pocket_id, center in pocket_positions(scaled, settings):
            cv2.circle(interior, center.as_int(), exclude, 0, thickness=-1)
    else:
        interior = np.full(small.shape[:2], 255, dtype=np.uint8)

    candidates = cv2.bitwise_and(cv2.bitwise_not(cloth), interior)
    # Open with a kernel a bit under the smallest plausible ball: removes chalk
    # dust and mask speckle while leaving every real ball intact.
    k = max(3, int(min_r * 0.8) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)

    # RETR_LIST, not RETR_EXTERNAL. Without a table boundary to clip against,
    # the candidate mask's outermost region is the dark surround, which encloses
    # the felt, which encloses the balls -- so the balls sit two levels down the
    # hierarchy and RETR_EXTERNAL discards every one of them. Nesting carries no
    # meaning here, so ask for all contours.
    contours, _ = cv2.findContours(candidates, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    expected_area = math.pi * ((min_r + max_r) / 2.0) ** 2
    tolerance = settings.vision.ball_area_tolerance
    balls: list[Ball] = []

    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0:
            continue

        # Circularity is the primary shadow rejector. A ball with its shadow
        # attached is a lopsided blob; the ball alone is nearly a perfect disc.
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < settings.vision.min_ball_circularity:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not (min_r <= radius <= max_r):
            continue

        area_error = abs(area - expected_area) / expected_area
        if area_error > 1.0 + tolerance:
            continue

        median_hsv, value_std = sample_ball_hsv(hsv, (int(round(cx)), int(round(cy))), radius)
        color, color_confidence = classify_color(median_hsv)

        # Only reject a dark blob if it is *darker than any real ball*. The 8
        # ball images around value 25-45 under normal lighting, so a threshold
        # set to reject shadows at 35 throws the 8 ball away with them. Pockets
        # and shadows are excluded by the interior inset and the circularity
        # test instead, which do not have to make this trade-off.
        if color is BallColor.BLACK and float(median_hsv[2]) < 12:
            logger.debug("rejecting near-black blob at (%.0f, %.0f) as pocket", cx, cy)
            continue

        kind = classify_stripe_or_solid(color, value_std)
        # Geometry is trusted over colour, so shape carries more weight than the
        # colour match -- see the module docstring.
        shape_score = min(1.0, circularity) * max(0.0, 1.0 - area_error / (1.0 + tolerance))
        confidence = float(np.clip(0.65 * shape_score + 0.35 * color_confidence, 0.0, 1.0))
        if confidence < settings.vision.min_ball_confidence:
            continue

        balls.append(
            Ball(
                # Provisional id; BallTracker replaces it with a stable one.
                id=f"det_{index:02d}",
                center_px=Vec2(cx / scale, cy / scale),
                radius_px=radius / scale,
                color=color,
                kind=kind,
                number=guess_number(color, kind),
                confidence=confidence,
            )
        )

    balls.sort(key=lambda b: b.confidence, reverse=True)
    logger.debug("detected %d balls from %d contours", len(balls), len(contours))
    return balls


def _scale_boundary(boundary: TableBoundary, scale: float) -> TableBoundary:
    """Re-express a full-resolution boundary in downscaled px."""
    return TableBoundary(
        top_left=boundary.top_left.scaled(scale),
        top_right=boundary.top_right.scaled(scale),
        bottom_right=boundary.bottom_right.scaled(scale),
        bottom_left=boundary.bottom_left.scaled(scale),
        center=boundary.center.scaled(scale),
        width_px=boundary.width_px * scale,
        height_px=boundary.height_px * scale,
        confidence=boundary.confidence,
    )


def detect_cue_ball(
    frame: np.ndarray,
    balls: list[Ball] | None = None,
    settings: Settings | None = None,
) -> Ball | None:
    """Identify the cue ball among already-detected balls.

    Picks from ``balls`` rather than re-scanning: the cue ball is simply the
    least saturated, brightest ball present, and a second detection pass would
    double the frame cost for information already in hand.

    The choice is made **relative to the other balls on the table**, not against
    a fixed white threshold. Under a projector an absolute threshold fails in
    both directions -- a dim room makes the real cue ball too dark to pass, and a
    white overlay landing on the 4 ball makes it pass. Ranking is stable under
    both.

    Args:
        frame: BGR frame. Used only if ``balls`` is not supplied.
        balls: Already-detected balls. Strongly preferred.
        settings: Config. Defaults to the global settings.

    Returns:
        The cue ball with ``kind`` set to ``CUE``, or ``None`` if no candidate is
        white enough. ``confidence`` reflects the margin over the runner-up, so
        an ambiguous frame produces a low score rather than a confident mistake --
        callers should prefer the last known cue ball over a weak match.
    """
    settings = settings or get_settings()
    if balls is None:
        balls = detect_balls(frame, settings=settings)
    if not balls:
        return None

    whites = [b for b in balls if b.color is BallColor.WHITE]
    if not whites:
        logger.debug("no white ball among %d detections; cue ball not visible", len(balls))
        return None

    # Highest confidence white wins. classify_color already folds "how white is
    # it" into that confidence for achromatic balls.
    whites.sort(key=lambda b: b.confidence, reverse=True)
    cue = whites[0]

    if len(whites) > 1:
        # Two white candidates means either a projected highlight on another
        # ball or the eight-ball's white number showing. Scale confidence by the
        # margin so the caller knows this frame was ambiguous.
        margin = cue.confidence - whites[1].confidence
        cue.confidence *= float(np.clip(0.4 + margin * 3.0, 0.4, 1.0))
        logger.debug("%d white candidates; margin %.2f", len(whites), margin)

    cue.kind = BallKind.CUE
    cue.number = None
    return cue


# ---------------------------------------------------------------------------
# Cue stick detection
# ---------------------------------------------------------------------------


def _merge_collinear(
    segments: np.ndarray, angle_tolerance_deg: float
) -> list[tuple[np.ndarray, float]]:
    """Group near-collinear segments and return one span per group.

    The bridge hand almost always splits the shaft into two or more pieces, and
    the longest single fragment is often shorter than a rail highlight -- so
    merging before picking the longest is what makes the cue win. Grouping is by
    orientation *and* perpendicular offset: two parallel lines on opposite sides
    of the table are not the same cue.

    Returns:
        ``[(endpoints, length)]``, endpoints as a 2x2 array, longest first.
    """
    groups: list[list[np.ndarray]] = []
    angles: list[float] = []

    for x1, y1, x2, y2 in segments:
        # Modulo 180: a segment has no direction, only an orientation.
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        placed = False
        for i, group_angle in enumerate(angles):
            delta = abs(angle - group_angle)
            delta = min(delta, 180.0 - delta)
            if delta > angle_tolerance_deg:
                continue
            # Perpendicular distance from this segment's midpoint to the group's
            # existing line, to keep parallel-but-separate lines apart.
            ref = groups[i][0]
            rx1, ry1, rx2, ry2 = ref
            dx, dy = rx2 - rx1, ry2 - ry1
            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                continue
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            offset = abs(dy * (mx - rx1) - dx * (my - ry1)) / norm
            if offset > max(12.0, norm * 0.08):
                continue
            groups[i].append(np.array([x1, y1, x2, y2], dtype=np.float64))
            placed = True
            break
        if not placed:
            groups.append([np.array([x1, y1, x2, y2], dtype=np.float64)])
            angles.append(angle)

    merged: list[tuple[np.ndarray, float]] = []
    for group in groups:
        points = np.array([[s[0], s[1]] for s in group] + [[s[2], s[3]] for s in group])

        # Fit the group's axis and project every endpoint onto it, rather than
        # taking the farthest-apart pair. Canny finds *both* sides of the shaft,
        # so the two most distant endpoints are diagonally opposite corners of a
        # long thin rectangle -- and that diagonal is several degrees off the
        # shaft's true axis. At a table's length a few degrees is a miss.
        centroid = points.mean(axis=0)
        centered = points - centroid
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        projections = centered @ axis
        start = centroid + axis * projections.min()
        end = centroid + axis * projections.max()
        merged.append((np.array([start, end]), float(projections.max() - projections.min())))

    merged.sort(key=lambda item: item[1], reverse=True)
    return merged


def _overlay_colors_bgr(settings: Settings) -> list[np.ndarray]:
    """The render palette as BGR triples, for self-projection rejection.

    Taken from the *resolved theme* rather than from the config colours
    directly. The two agree for the default ``classic`` theme, and diverge for
    every other one -- so reading config here would mean that switching to
    ``neon`` silently reintroduced the overlay-detected-as-cue failure, with the
    projected cyan trajectory being tracked as a cue stick. Anything that
    changes what colour goes onto the felt has to change this list with it.
    """
    from projection.themes import palette_rgb, resolve_theme

    # Themes store RGB, like the config they came from; OpenCV works in BGR.
    return [
        np.array([c[2], c[1], c[0]], dtype=np.float32)
        for c in palette_rgb(resolve_theme(settings))
    ]


def _looks_like_overlay(
    image: np.ndarray,
    endpoints: np.ndarray,
    overlay_bgr: list[np.ndarray],
    match_distance: float = 70.0,
) -> bool:
    """Whether a line's own colour matches the projected overlay palette.

    Samples along the line rather than at its midpoint: a cue crossing a
    projected trajectory would fail a single-point test, and sampling the median
    of many points makes the crossing irrelevant.
    """
    p1, p2 = endpoints
    samples = 12
    ts = np.linspace(0.15, 0.85, samples)  # skip the ends, which touch the felt
    xs = np.clip((p1[0] + (p2[0] - p1[0]) * ts).astype(int), 0, image.shape[1] - 1)
    ys = np.clip((p1[1] + (p2[1] - p1[1]) * ts).astype(int), 0, image.shape[0] - 1)
    pixels = image[ys, xs].astype(np.float32)
    if pixels.size == 0:
        return False
    median = np.median(pixels, axis=0)
    return any(
        float(np.linalg.norm(median - color)) < match_distance for color in overlay_bgr
    )


def detect_cue_stick(
    frame: np.ndarray,
    boundary: TableBoundary | None = None,
    settings: Settings | None = None,
    cue_ball: Ball | None = None,
    camera_to_table: np.ndarray | None = None,
    *,
    _prepared: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> CueStick | None:
    """Find the cue and the direction it is aiming.

    Canny plus a probabilistic Hough transform, restricted to the table and a
    margin around it, then collinear merging before the longest line is taken as
    the shaft.

    Deciding **which end is the tip** matters more than anything else here.
    Getting it backwards points the prediction 180 degrees out, which is the most
    visible possible failure. When the cue ball is known the tip is the nearer
    endpoint, which is reliable; without it the endpoint closer to the table
    centre is used, which is a guess, and the confidence returned says so.

    ``angle_deg`` is returned in **table space** when ``camera_to_table`` is
    supplied, so the physics layer never has to think about the camera. Without
    it the angle is in camera space and confidence is reduced, because an
    uncalibrated angle is not directly usable for aiming.

    Returns:
        The cue, or ``None`` when no plausible stick is visible -- the normal
        state between shots, not an error.
    """
    import cv2

    settings = settings or get_settings()
    if frame is None or frame.size == 0:
        return None

    if _prepared is not None:
        small, _hsv, scale = _prepared
    else:
        small, scale = downscale_for_detection(frame, settings.vision.detection_width)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Blur before Canny: the cloth's weave generates a dense field of tiny edges
    # that swamp the Hough accumulator otherwise.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)

    if boundary is not None:
        # Restrict to the cloth, inset past the cushion. Expanding *outward* to
        # catch the cue's butt is the intuitive move and it is wrong: the rails
        # are the longest straight lines anywhere in the frame, so including
        # them means the "cue" found is always a rail. Truncating the shaft at
        # the cushion costs some length -- which the confidence score reflects --
        # but the on-cloth portion is what carries the aim direction anyway.
        scaled = _scale_boundary(boundary, scale)
        inset = 0.02 * min(scaled.width_px, scaled.height_px)
        edges = cv2.bitwise_and(edges, boundary_interior_mask(small.shape[:2], scaled, inset))

    min_length = max(20.0, settings.vision.cue_min_line_length_px * scale)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=50,
        minLineLength=int(min_length),
        maxLineGap=int(max(4, settings.vision.cue_max_line_gap_px * scale)),
    )
    if lines is None or len(lines) == 0:
        return None

    merged = _merge_collinear(lines.reshape(-1, 4), settings.vision.cue_merge_angle_deg)
    if not merged:
        return None

    # Reject our own projection before picking a winner. A rendered trajectory
    # is a long bright line lying on the cloth -- geometrically indistinguishable
    # from a cue, and measured at 118 degrees of aim error when it wins. What
    # separates them is colour: the overlay palette is *known*, because we chose
    # it, whereas a cue is wood or a muted composite.
    #
    # This is a mitigation, not a cure. The robust fix is to blank the overlay
    # for a single frame periodically and detect against that clean frame; see
    # the module docstring. Until then, colour rejection handles the common case
    # cheaply.
    overlay_bgr = _overlay_colors_bgr(settings)
    candidate = None
    for endpoints, length in merged:
        if length < min_length:
            break  # merged is sorted by length, so nothing after this qualifies
        if _looks_like_overlay(small, endpoints, overlay_bgr):
            logger.debug("rejecting a line matching the overlay palette as our own projection")
            continue
        candidate = (endpoints, length)
        break

    if candidate is None:
        return None
    endpoints, length = candidate

    # Which end is the tip.
    if cue_ball is not None:
        target = np.array([cue_ball.center_px.x * scale, cue_ball.center_px.y * scale])
        tip_is_confident = True
    elif boundary is not None:
        scaled = _scale_boundary(boundary, scale)
        target = np.array([scaled.center.x, scaled.center.y])
        tip_is_confident = False
    else:
        target = np.array([small.shape[1] / 2.0, small.shape[0] / 2.0])
        tip_is_confident = False

    d0 = float(np.linalg.norm(endpoints[0] - target))
    d1 = float(np.linalg.norm(endpoints[1] - target))
    tip_small, butt_small = (endpoints[0], endpoints[1]) if d0 < d1 else (endpoints[1], endpoints[0])

    tip_px = Vec2(float(tip_small[0]) / scale, float(tip_small[1]) / scale)
    butt_px = Vec2(float(butt_small[0]) / scale, float(butt_small[1]) / scale)

    # Aim direction runs butt -> tip: that is where the ball will go.
    tip_table_pos: Vec2 | None = None
    angle_in_table_space = False
    if camera_to_table is not None:
        try:
            tip_table_pos = camera_to_table_coords(tip_px, camera_to_table)
            butt_table = camera_to_table_coords(butt_px, camera_to_table)
            angle_deg = math.degrees(
                math.atan2(tip_table_pos.y - butt_table.y, tip_table_pos.x - butt_table.x)
            )
            angle_in_table_space = True
        except CalibrationError:
            # A cue extending past the transform's horizon. Fall back to the
            # camera-space angle rather than dropping the detection.
            logger.debug("cue endpoint outside the table plane; using camera-space angle")
            angle_deg = math.degrees(math.atan2(tip_px.y - butt_px.y, tip_px.x - butt_px.x))
    else:
        angle_deg = math.degrees(math.atan2(tip_px.y - butt_px.y, tip_px.x - butt_px.x))

    # Confidence: length relative to the table, penalised for a guessed tip and
    # for an uncalibrated angle. Both caveats make the reading less usable, and
    # the mode layer should be able to see that.
    # Full credit at ~20% of the table's width of visible shaft. A real cue is
    # 58 in long, but most of it is off the table and behind the player; the
    # detectable on-cloth portion is far shorter, and scoring against the full
    # cue length would mark every genuine detection as low confidence.
    reference = boundary.width_px if boundary is not None else frame.shape[1]
    length_score = min(1.0, (length / scale) / (reference * 0.20))
    confidence = length_score
    if not tip_is_confident:
        confidence *= 0.6
    if not angle_in_table_space:
        confidence *= 0.7
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if confidence < settings.vision.min_cue_confidence:
        logger.debug("cue candidate rejected: confidence %.2f", confidence)
        return None

    return CueStick(
        tip_px=tip_px,
        angle_deg=angle_deg,
        tip_table_pos=tip_table_pos,
        shaft_visible=True,
        velocity=0.0,  # filled in by BallTracker across frames
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Pocket detection
# ---------------------------------------------------------------------------


def detect_pocket_openings(
    frame: np.ndarray,
    boundary: TableBoundary | None = None,
    settings: Settings | None = None,
) -> list[Pocket]:
    """Locate the six pockets.

    Geometry-first, refined by image evidence. Pockets sit at known positions
    relative to the table corners and side midpoints, so their expected locations
    come from ``boundary`` and dark-region detection only nudges them. Pure
    dark-region detection is the tempting approach and a bad one -- it finds
    shadows, the cue's butt and a player's dark sleeve, and cheerfully returns
    five or seven pockets.

    Always returns six when a boundary is known, so downstream code never has to
    handle a partial set. Results should be cached by the caller: pockets do not
    move, and re-deriving them every frame is wasted work.

    Returns:
        Six pockets in full-resolution camera px, or an empty list if the table
        has not been found yet.
    """
    import cv2

    settings = settings or get_settings()
    if boundary is None:
        logger.debug("cannot place pockets without a table boundary")
        return []

    ppi = pixels_per_inch(boundary, settings)
    radius_px = settings.table.pocket_radius_in * ppi

    gray = None
    if frame is not None and frame.size > 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pockets: list[Pocket] = []
    for pocket_id, center in pocket_positions(boundary, settings):
        expected = np.array([center.x, center.y], dtype=np.float64)
        refined = expected
        if gray is not None:
            found = _refine_pocket(gray, expected, radius_px)
            if found is not None:
                refined = found

        pockets.append(
            Pocket(
                id=pocket_id,
                center_px=Vec2(float(refined[0]), float(refined[1])),
                radius_px=float(radius_px),
            )
        )
    return pockets


def _refine_pocket(
    gray: np.ndarray, expected: np.ndarray, radius_px: float
) -> np.ndarray | None:
    """Nudge a pocket estimate onto the darkest nearby spot.

    Searches only a small window around the geometric estimate, which is what
    keeps this from wandering onto an unrelated shadow. Returns ``None`` when
    the window holds nothing convincingly dark, leaving the geometric estimate
    in place -- the right outcome, since geometry is the more trustworthy source.
    """
    import cv2

    search = int(max(6.0, radius_px * 1.5))
    cx, cy = int(round(expected[0])), int(round(expected[1]))
    y0, y1 = max(0, cy - search), min(gray.shape[0], cy + search + 1)
    x0, x1 = max(0, cx - search), min(gray.shape[1], cx + search + 1)
    window = gray[y0:y1, x0:x1]
    if window.size == 0:
        return None

    # A pocket is a hole: near-black and much darker than its surroundings. If
    # nothing in the window qualifies, the pocket is occluded or the estimate is
    # off, and geometry should win.
    threshold = max(45, int(np.percentile(window, 5)) + 12)
    dark = (window < threshold).astype(np.uint8) * 255
    if not dark.any():
        return None

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (radius_px * radius_px * 0.3):
        return None

    moments = cv2.moments(largest)
    if moments["m00"] <= 0:
        return None
    return np.array(
        [x0 + moments["m10"] / moments["m00"], y0 + moments["m01"] / moments["m00"]]
    )


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


class BallTracker:
    """Assigns stable ids across frames and estimates per-ball velocity.

    Detection alone returns an unordered bag of balls per frame, which is not
    enough for the game layer: deciding a shot has been taken needs to know that
    *this* ball moved, and identifying pocketed balls needs ids that persist. So
    this greedily matches each detection to the nearest previous ball within a
    gate, carries the id forward, and differentiates position over time.

    Greedy nearest-neighbour rather than a Hungarian assignment or a Kalman
    filter: with at most 16 well-separated objects moving smoothly, the greedy
    match is right essentially always, and it costs microseconds instead of
    milliseconds. Revisit only if id swapping is observed during a hard break.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._tracks: dict[str, Ball] = {}
        self._velocities: dict[str, Vec2] = {}
        self._last_timestamp: float | None = None
        self._next_id = 0
        self._missing_frames: dict[str, int] = {}
        #: Frames a track survives without a detection before being dropped.
        #: Generous, because a hand passing over the table occludes balls for
        #: several frames and dropping them would fire a spurious "pocketed".
        self._max_missing = 15

    def update(self, balls: list[Ball], timestamp: float) -> list[Ball]:
        """Match detections to existing tracks, returning balls with stable ids.

        Args:
            balls: This frame's detections, with ``table_pos`` already populated
                -- matching happens in table inches, not pixels, so the gate is a
                physical distance and does not change with camera placement.
            timestamp: ``time.perf_counter()`` at capture.

        Returns:
            The same balls, with ``id`` rewritten to a stable track id.
        """
        dt = 0.0 if self._last_timestamp is None else timestamp - self._last_timestamp
        self._last_timestamp = timestamp

        # Gate: how far a ball could plausibly move between frames. A hard-struck
        # ball does ~250 in/s, so at 30 FPS that is ~8 in. Too tight and a fast
        # ball spawns a new id every frame; too loose and neighbouring balls swap.
        gate_in = max(4.0, 300.0 * max(dt, 1.0 / 120.0))

        unmatched = list(self._tracks.items())
        assigned: list[Ball] = []
        used: set[str] = set()

        for ball in balls:
            if ball.table_pos is None:
                # Cannot match without table coordinates; give it a fresh id and
                # let the next frame pick it up properly.
                ball.id = self._make_id()
                assigned.append(ball)
                continue

            best_id, best_distance = None, float("inf")
            for track_id, previous in unmatched:
                if track_id in used or previous.table_pos is None:
                    continue
                distance = ball.table_pos.distance_to(previous.table_pos)
                if distance < best_distance:
                    best_id, best_distance = track_id, distance

            if best_id is not None and best_distance <= gate_in:
                previous = self._tracks[best_id]
                ball.id = best_id
                used.add(best_id)
                if dt > 1e-6 and previous.table_pos is not None:
                    delta = ball.table_pos - previous.table_pos
                    self._velocities[best_id] = delta.scaled(1.0 / dt)
                self._missing_frames[best_id] = 0
            else:
                ball.id = self._make_id()
                self._velocities[ball.id] = Vec2(0.0, 0.0)

            assigned.append(ball)

        # Age out tracks that went unseen.
        seen = {b.id for b in assigned}
        for track_id in list(self._tracks):
            if track_id in seen:
                continue
            self._missing_frames[track_id] = self._missing_frames.get(track_id, 0) + 1
            if self._missing_frames[track_id] > self._max_missing:
                self._tracks.pop(track_id, None)
                self._velocities.pop(track_id, None)
                self._missing_frames.pop(track_id, None)

        self._tracks.update({b.id: b for b in assigned})
        return assigned

    def _make_id(self) -> str:
        self._next_id += 1
        return f"ball_{self._next_id:03d}"

    def velocity(self, ball_id: str) -> Vec2:
        """Last estimated velocity in inches/sec. Zero for an unknown id."""
        return self._velocities.get(ball_id, Vec2(0.0, 0.0))

    def max_speed(self) -> float:
        """Fastest tracked ball, inches/sec.

        What the shot state machine thresholds on to decide the table is still
        in motion.
        """
        if not self._velocities:
            return 0.0
        return max(v.length() for v in self._velocities.values())

    def any_moving(self, threshold: float | None = None) -> bool:
        """Whether any ball exceeds the stopped threshold."""
        limit = (
            self.settings.vision.ball_stopped_threshold if threshold is None else threshold
        )
        return self.max_speed() > limit

    def reset(self) -> None:
        """Forget all tracks. Called on table re-detection and on game reset."""
        self._tracks.clear()
        self._velocities.clear()
        self._missing_frames.clear()
        self._last_timestamp = None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def extract_game_state(
    frame: np.ndarray,
    frame_index: int,
    timestamp: float,
    boundary: TableBoundary | None = None,
    camera_to_table: np.ndarray | None = None,
    settings: Settings | None = None,
    tracker: BallTracker | None = None,
    pockets: list[Pocket] | None = None,
) -> GameState:
    """Run the full detection pass and assemble a :class:`~app.models.GameState`.

    The single entry point the main loop calls, and the only place holding the
    homography -- so it is also where every detection gets its ``table_pos``.

    Shares one downscale and one BGR->HSV conversion across all the detectors.
    That is not micro-optimisation: at 960 px the conversion is ~1.5 ms, and
    doing it three times would spend 10% of the frame budget on repeated work.

    Deliberately defensive. A frame where nothing is found is a normal event --
    someone leaning over the table -- and the response is a ``GameState`` with
    empty lists and a low ``confidence``, never an exception. The main loop
    decides what to do about it.

    Args:
        frame: BGR camera frame, full resolution.
        frame_index: Monotonic frame counter, carried through for log correlation.
        timestamp: ``time.perf_counter()`` at capture, not now -- latency
            accounting depends on it.
        boundary: Cached table boundary. Detecting the table every frame is
            wasteful; the main loop detects on an interval and passes it in.
        camera_to_table: Cached homography matching ``boundary``.
        settings: Config. Defaults to the global settings.
        tracker: Ball tracker for stable ids and velocities. Without one, ids are
            per-frame only and no velocity is available.
        pockets: Cached pockets. Recomputed from ``boundary`` when omitted.

    Returns:
        The frame's game state, with ``confidence`` aggregating the individual
        detections so callers can gate on quality with one check.
    """
    import cv2

    settings = settings or get_settings()
    state = GameState(timestamp=timestamp, frame_index=frame_index, table_boundary=boundary)

    if frame is None or frame.size == 0:
        _changes.report(
            "empty_frame",
            True,
            logging.WARNING,
            "extract_game_state called with an empty frame (silenced until it recovers)",
        )
        return state
    _changes.recovered("empty_frame", logging.INFO, "frames are arriving again")

    small, scale = downscale_for_detection(frame, settings.vision.detection_width)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    prepared = (small, hsv, scale)

    # -- balls -------------------------------------------------------------
    try:
        balls = detect_balls(frame, boundary, settings, _prepared=prepared)
        _changes.recovered("ball_detect", logging.INFO, "ball detection recovered")
    except cv2.error as exc:
        # OpenCV raises on malformed input rather than returning empty. One bad
        # frame must not take the loop down.
        #
        # Keyed on the message, so a *different* failure still gets reported --
        # the thing being suppressed is repetition, not information.
        _changes.report(
            "ball_detect",
            str(exc),
            logging.WARNING,
            "ball detection failing from frame %d: %s (silenced until it changes)",
            frame_index,
            exc,
        )
        balls = []

    if camera_to_table is not None:
        for ball in balls:
            try:
                ball.table_pos = camera_to_table_coords(ball.center_px, camera_to_table)
            except CalibrationError:
                # Outside the table plane: keep the pixel detection but leave
                # table_pos None so physics skips it rather than using garbage.
                logger.debug("ball at %s has no table-plane position", ball.center_px)

    cue_ball = detect_cue_ball(frame, balls, settings)
    if tracker is not None:
        balls = tracker.update(balls, timestamp)
        # update() rewrites ids in place, so re-resolve the cue ball reference.
        if cue_ball is not None:
            cue_ball = next((b for b in balls if b.is_cue), cue_ball)

    state.balls = balls
    state.cue_ball = cue_ball

    # -- cue stick ---------------------------------------------------------
    try:
        state.cue_stick = detect_cue_stick(
            frame,
            boundary,
            settings,
            cue_ball=cue_ball,
            camera_to_table=camera_to_table,
            _prepared=prepared,
        )
        _changes.recovered("cue_detect", logging.INFO, "cue detection recovered")
    except cv2.error as exc:
        _changes.report(
            "cue_detect",
            str(exc),
            logging.WARNING,
            "cue detection failing from frame %d: %s (silenced until it changes)",
            frame_index,
            exc,
        )

    # -- pockets -----------------------------------------------------------
    state.pockets = (
        pockets if pockets is not None else detect_pocket_openings(frame, boundary, settings)
    )
    if camera_to_table is not None:
        for pocket in state.pockets:
            try:
                pocket.table_pos = camera_to_table_coords(pocket.center_px, camera_to_table)
            except CalibrationError:
                pass

    state.confidence = _aggregate_confidence(state)
    logger.debug(
        "frame %d: %d balls, cue_ball=%s, cue=%s, confidence %.2f",
        frame_index,
        len(state.balls),
        state.cue_ball is not None,
        state.cue_stick is not None,
        state.confidence,
    )
    return state


def _aggregate_confidence(state: GameState) -> float:
    """Blend per-element confidences into one score for the frame.

    Weighted by what downstream code actually depends on. The table dominates,
    because without it nothing can be expressed in table coordinates and every
    other detection is unusable. The cue stick is excluded entirely -- no cue is
    the normal state between shots, and counting its absence as low quality would
    make a settled table look like a detection failure.
    """
    if state.table_boundary is None:
        return 0.0

    score = 0.45 * state.table_boundary.confidence
    if state.cue_ball is not None:
        score += 0.25 * state.cue_ball.confidence
    object_balls = state.object_balls()
    if object_balls:
        score += 0.2 * (sum(b.confidence for b in object_balls) / len(object_balls))
    if len(state.pockets) == 6:
        score += 0.1
    return float(np.clip(score, 0.0, 1.0))
