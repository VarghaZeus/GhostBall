"""Synthetic pool table images with exact ground truth.

Testing computer vision needs images whose correct answer is known. Real photos
would have to be hand-labelled, and hand-labelled ball centres are accurate to a
few pixels at best -- which is the same order as the error being measured. So
these frames are *constructed*: ball positions are chosen in table inches, then
projected into camera pixels through a homography this module builds, so the
expected answer is known to floating-point precision.

That inverts the usual relationship. Rather than detecting and then checking
against a label, the test states where a ball is, renders it, and asserts the
detector recovers that position. Errors are then reportable in inches, which is
the unit the physics engine and the player actually care about.

Deliberately not photorealistic. It reproduces the specific things that break
detection -- perspective, specular highlights, cast shadows, sensor noise,
projected overlay light, uneven illumination -- and nothing else. Passing here is
necessary, not sufficient; the felt-threshold tuning in
``tools.camera_preview --mask`` is what validates against a real table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import BALL_DIAMETER_IN, Settings
from app.models import BallColor, Vec2
from vision.colors import BALL_COLOR_REFERENCE

#: Felt colour in HSV. Mid-way inside the default ``felt_hue_range`` of
#: (35, 85) so tests are not sensitive to that boundary, and at a saturation
#: typical of matte wool cloth -- well under the glossy balls, which is what
#: makes the 6 ball separable from the cloth it sits on.
FELT_HSV = (62, 145, 115)

#: Rail and surround colour, BGR. Dark and desaturated so it cannot be confused
#: with felt by a hue threshold.
RAIL_BGR = (38, 32, 28)

#: Cue colour, BGR -- a light tan maple shaft.
CUE_BGR = (120, 165, 205)


#: Cloth colours found on real tables, OpenCV HSV. Only ``green`` falls inside
#: the default ``felt_hue_range``; the rest exist to prove that pocket-based
#: detection does not care.
#:
#: ``black`` is deliberately included and is the hard one -- on black cloth the
#: 8 ball is the same colour as the table, so the adaptive cloth mask cannot
#: separate them and ball-derived scale degrades. No colour trick fixes that.
FELT_COLOURS: dict[str, tuple[int, int, int]] = {
    "green": (62, 145, 115),
    "blue": (110, 160, 120),
    "red": (2, 170, 110),
    "burgundy": (172, 150, 80),
    "black": (0, 12, 32),
}


def hsv_to_bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert a single OpenCV HSV triple to BGR, for drawing calls."""
    patch = np.array([[list(hsv)]], dtype=np.uint8)
    bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


@dataclass
class BallSpec:
    """A ball to render, positioned in table inches."""

    x_in: float
    y_in: float
    color: BallColor
    #: Draw a white stripe band, so stripe/solid classification has a subject.
    striped: bool = False


@dataclass
class GroundTruth:
    """The exact answer for a rendered frame."""

    #: Table corners in camera px, clockwise from top-left.
    corners: list[Vec2]
    #: Ball centres in camera px, in the order they were specified.
    ball_centers_px: list[Vec2]
    #: The same balls in table inches, as originally specified.
    ball_positions_in: list[Vec2]
    ball_colors: list[BallColor]
    #: Expected ball radius in camera px at the table centre. Approximate under
    #: strong perspective, since a near ball images larger than a far one.
    ball_radius_px: float
    #: Cue tip in camera px, when a cue was drawn.
    cue_tip_px: Vec2 | None = None
    #: Cue aim direction in table space, degrees, matching the convention in
    #: ``app.models.CueStick``.
    cue_angle_deg: float | None = None
    table_to_camera: np.ndarray = field(default_factory=lambda: np.eye(3))

    def cue_ball_index(self) -> int | None:
        """Index of the white ball, or ``None`` if none was rendered."""
        return next(
            (i for i, c in enumerate(self.ball_colors) if c is BallColor.WHITE), None
        )


def _table_to_camera_matrix(
    settings: Settings,
    frame_width: int,
    frame_height: int,
    margin: float,
    rotation_deg: float,
    perspective: float,
) -> np.ndarray:
    """Build the table-inches -> camera-px homography for a rendered scene.

    ``perspective`` shrinks the top edge relative to the bottom, which is what an
    overhead camera mounted off the table's centre actually produces. It is the
    single most important thing to simulate: an axis-aligned rectangle is a much
    easier target than any real installation, and code tested only on rectangles
    tends to have corner-ordering bugs.
    """
    length, width = settings.table.length_in, settings.table.width_in

    half_w = frame_width * (1.0 - 2.0 * margin) / 2.0
    half_h = frame_height * (1.0 - 2.0 * margin) / 2.0
    cx, cy = frame_width / 2.0, frame_height / 2.0

    # Destination quad, centred, with the top edge narrowed by `perspective`.
    shrink = 1.0 - perspective
    quad = np.array(
        [
            [cx - half_w * shrink, cy - half_h],
            [cx + half_w * shrink, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ],
        dtype=np.float32,
    )

    if rotation_deg:
        theta = math.radians(rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        centered = quad - np.array([cx, cy], dtype=np.float32)
        rotated = np.stack(
            [
                centered[:, 0] * cos_t - centered[:, 1] * sin_t,
                centered[:, 0] * sin_t + centered[:, 1] * cos_t,
            ],
            axis=1,
        )
        quad = rotated + np.array([cx, cy], dtype=np.float32)

    source = np.array(
        [[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(source, quad).astype(np.float64)


def _project(matrix: np.ndarray, x: float, y: float) -> Vec2:
    """Apply a homography to one point."""
    denom = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return Vec2(
        (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom,
        (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom,
    )


def render_table(
    settings: Settings,
    balls: list[BallSpec] | None = None,
    *,
    frame_width: int = 1920,
    frame_height: int = 1080,
    margin: float = 0.08,
    rotation_deg: float = 0.0,
    perspective: float = 0.0,
    cue_from: tuple[float, float] | None = None,
    cue_to: tuple[float, float] | None = None,
    highlights: bool = True,
    shadows: bool = False,
    noise_sigma: float = 0.0,
    vignette: float = 0.0,
    overlay_streak: bool = False,
    draw_pockets: bool = True,
    felt_hsv: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, GroundTruth]:
    """Render a synthetic table and return ``(bgr_frame, ground_truth)``.

    Args:
        settings: Supplies the table dimensions, so rendering and detection agree
            on what the table is.
        balls: Balls to draw, in table inches. ``None`` draws none.
        frame_width: Output width in px.
        frame_height: Output height in px.
        margin: Fraction of the frame left as surround on each side.
        rotation_deg: Rotate the table in frame, simulating a camera that is not
            square to it.
        perspective: 0 for a perfect rectangle, up to ~0.3 for a strongly
            keystoned view. Realistic installations are 0.05-0.15.
        cue_from: Cue butt position in table inches.
        cue_to: Cue tip position in table inches. Aim runs from ``cue_from``
            toward ``cue_to``.
        highlights: Draw a specular highlight on each ball. Real balls are
            glossy, and the highlight is exactly what pulls a mean colour sample
            toward white -- which is why colour sampling uses a median.
        shadows: Draw an offset dark ellipse under each ball. The main source of
            false positives, and what the circularity filter must reject.
        noise_sigma: Gaussian sensor noise standard deviation.
        vignette: Corner darkening, 0-1. Simulates uneven room lighting, which
            is what makes a single global value threshold fail.
        overlay_streak: Draw a bright projected line across the felt, simulating
            the system's own overlay. Tests that detection survives the light it
            is projecting.
        draw_pockets: Draw the six dark pocket mouths.
        felt_hsv: Cloth colour, OpenCV HSV. Defaults to :data:`FELT_HSV`, a
            green inside the configured hue range. Pass something outside that
            range -- red, blue, burgundy -- to exercise the detectors that are
            supposed to be colour-independent. Note that felt-based table
            detection is *expected* to fail on those, which is the entire
            reason ``vision.pockets`` exists.

    Returns:
        ``(frame, ground_truth)``. The frame is ``HxWx3`` uint8 BGR.
    """
    balls = balls or []
    length, width = settings.table.length_in, settings.table.width_in

    matrix = _table_to_camera_matrix(
        settings, frame_width, frame_height, margin, rotation_deg, perspective
    )

    frame = np.full((frame_height, frame_width, 3), RAIL_BGR, dtype=np.uint8)

    # Felt.
    corners = [
        _project(matrix, 0.0, 0.0),
        _project(matrix, length, 0.0),
        _project(matrix, length, width),
        _project(matrix, 0.0, width),
    ]
    felt_bgr = hsv_to_bgr(felt_hsv or FELT_HSV)
    cv2.fillConvexPoly(
        frame, np.array([c.as_int() for c in corners], dtype=np.int32), felt_bgr
    )

    # Scale at the table centre, used for ball radii.
    center_px = _project(matrix, length / 2.0, width / 2.0)
    one_inch = _project(matrix, length / 2.0 + 1.0, width / 2.0)
    px_per_inch = center_px.distance_to(one_inch)
    ball_radius_px = px_per_inch * (BALL_DIAMETER_IN / 2.0)

    if draw_pockets:
        pocket_r = int(settings.table.pocket_radius_in * px_per_inch)
        for u, v in ((0, 0), (0.5, 0), (1, 0), (1, 1), (0.5, 1), (0, 1)):
            p = _project(matrix, length * u, width * v)
            cv2.circle(frame, p.as_int(), pocket_r, (10, 10, 10), -1, cv2.LINE_AA)

    # A projected overlay streak, drawn before the balls so balls sit on top of
    # it -- which is what actually happens on a real table.
    if overlay_streak:
        a = _project(matrix, length * 0.1, width * 0.25)
        b = _project(matrix, length * 0.9, width * 0.75)
        # The configured cue-path colour, not an arbitrary green: the point is
        # to test that detection rejects the palette this system actually
        # projects. RGB in config, BGR for OpenCV.
        rgb = settings.render.cue_path_color
        cv2.line(
            frame, a.as_int(), b.as_int(), (rgb[2], rgb[1], rgb[0]),
            max(3, int(px_per_inch * 1.2)), cv2.LINE_AA,
        )

    # Shadows first, so balls occlude their own shadow.
    if shadows:
        offset = max(2, int(ball_radius_px * 0.35))
        for spec in balls:
            c = _project(matrix, spec.x_in, spec.y_in)
            cv2.ellipse(
                frame,
                (int(c.x) + offset, int(c.y) + offset),
                (int(ball_radius_px * 1.05), int(ball_radius_px * 0.8)),
                0,
                0,
                360,
                (28, 46, 22),
                -1,
                cv2.LINE_AA,
            )

    ball_centers: list[Vec2] = []
    for spec in balls:
        c = _project(matrix, spec.x_in, spec.y_in)
        ball_centers.append(c)
        radius = int(round(ball_radius_px))

        if spec.color is BallColor.WHITE:
            bgr = (248, 248, 246)
        elif spec.color is BallColor.BLACK:
            bgr = (26, 26, 26)
        else:
            bgr = hsv_to_bgr(BALL_COLOR_REFERENCE[spec.color])

        if spec.striped:
            # White ball with a coloured band across the middle: the real
            # appearance, and what gives the value channel its bimodal spread.
            cv2.circle(frame, c.as_int(), radius, (245, 245, 243), -1, cv2.LINE_AA)
            band = max(2, int(radius * 0.9))
            cv2.ellipse(
                frame, c.as_int(), (radius, band // 2), 0, 0, 360, bgr, -1, cv2.LINE_AA
            )
        else:
            cv2.circle(frame, c.as_int(), radius, bgr, -1, cv2.LINE_AA)

        if highlights:
            hx = int(c.x - radius * 0.33)
            hy = int(c.y - radius * 0.33)
            cv2.circle(frame, (hx, hy), max(1, radius // 4), (252, 252, 252), -1, cv2.LINE_AA)

    # Cue.
    cue_tip_px: Vec2 | None = None
    cue_angle: float | None = None
    if cue_from is not None and cue_to is not None:
        butt = _project(matrix, *cue_from)
        tip = _project(matrix, *cue_to)
        cv2.line(
            frame,
            butt.as_int(),
            tip.as_int(),
            CUE_BGR,
            max(2, int(px_per_inch * 0.55)),
            cv2.LINE_AA,
        )
        cue_tip_px = tip
        cue_angle = math.degrees(
            math.atan2(cue_to[1] - cue_from[1], cue_to[0] - cue_from[0])
        )

    if vignette > 0:
        yy, xx = np.mgrid[0:frame_height, 0:frame_width]
        ny = (yy - frame_height / 2.0) / (frame_height / 2.0)
        nx = (xx - frame_width / 2.0) / (frame_width / 2.0)
        falloff = 1.0 - vignette * np.clip(np.sqrt(nx**2 + ny**2) / math.sqrt(2), 0, 1)
        frame = (frame.astype(np.float32) * falloff[:, :, None]).clip(0, 255).astype(np.uint8)

    if noise_sigma > 0:
        # Seeded for reproducibility: a test that fails one run in twenty is
        # worse than no test.
        rng = np.random.default_rng(1234)
        noisy = frame.astype(np.float32) + rng.normal(0.0, noise_sigma, frame.shape)
        frame = noisy.clip(0, 255).astype(np.uint8)

    truth = GroundTruth(
        corners=corners,
        ball_centers_px=ball_centers,
        ball_positions_in=[Vec2(b.x_in, b.y_in) for b in balls],
        ball_colors=[b.color for b in balls],
        ball_radius_px=ball_radius_px,
        cue_tip_px=cue_tip_px,
        cue_angle_deg=cue_angle,
        table_to_camera=matrix,
    )
    return frame, truth


def standard_rack(settings: Settings) -> list[BallSpec]:
    """A plausible mid-game spread: cue ball plus seven object balls.

    Spaced well apart on purpose. Touching balls merge into one blob under the
    non-felt segmentation and are a known limitation, so a test fixture that
    happens to place two balls in contact would be testing that limitation rather
    than the detector.
    """
    length, width = settings.table.length_in, settings.table.width_in
    return [
        BallSpec(length * 0.25, width * 0.50, BallColor.WHITE),
        BallSpec(length * 0.60, width * 0.30, BallColor.YELLOW),
        BallSpec(length * 0.60, width * 0.70, BallColor.BLUE),
        BallSpec(length * 0.70, width * 0.50, BallColor.RED),
        BallSpec(length * 0.80, width * 0.30, BallColor.PURPLE),
        BallSpec(length * 0.80, width * 0.70, BallColor.ORANGE),
        BallSpec(length * 0.90, width * 0.50, BallColor.GREEN),
        BallSpec(length * 0.50, width * 0.20, BallColor.BLACK),
    ]
