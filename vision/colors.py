"""Ball colour classification.

Reference data rather than tuning knobs, which is why it lives here and not in
``config.yaml``: the colours of a standard set of pool balls are a fixed
property of the balls, unlike the felt hue, which varies per table and per room.

All hues are on **OpenCV's 0-179 scale**, not 0-359. Halving a hue value read
from a colour picker is the single most common mistake in this area.

Classification is deliberately nearest-match rather than threshold-based. A
threshold scheme has to answer "what if nothing matched?", and on a table lit by
a projector the honest answer is "often" -- so instead every ball gets its
closest colour plus a confidence derived from how close that match actually was.
The caller can then decide what to do with a weak match, rather than being
handed a silent ``UNKNOWN``.
"""

from __future__ import annotations

import logging

import numpy as np

from app.models import BallColor, BallKind

logger = logging.getLogger(__name__)

#: Reference HSV centroids for a standard American set, plus the rack number
#: each colour maps to. Solids are 1-7, stripes 9-15 sharing the same colours,
#: so colour alone cannot separate a 2 from a 10 -- that is what
#: :func:`classify_stripe_or_solid` is for.
#:
#: Hue is only meaningful for chromatic colours. White and black are classified
#: by saturation and value instead, so their hue entries are ignored.
BALL_COLOR_REFERENCE: dict[BallColor, tuple[int, int, int]] = {
    BallColor.YELLOW: (28, 200, 210),
    BallColor.BLUE: (110, 200, 170),
    BallColor.RED: (2, 210, 190),
    BallColor.PURPLE: (140, 150, 120),
    BallColor.ORANGE: (12, 220, 210),
    # Deliberately high saturation. A green ball on green cloth is the single
    # hardest discrimination in the whole pipeline -- hue alone cannot separate
    # them -- and saturation is the channel that does it. See felt_sat_max.
    BallColor.GREEN: (68, 240, 125),
    BallColor.MAROON: (175, 190, 120),
}

#: Solid ball numbers by colour. The 8 is black and the cue ball is white, both
#: handled separately.
SOLID_NUMBER_BY_COLOR: dict[BallColor, int] = {
    BallColor.YELLOW: 1,
    BallColor.BLUE: 2,
    BallColor.RED: 3,
    BallColor.PURPLE: 4,
    BallColor.ORANGE: 5,
    BallColor.GREEN: 6,
    BallColor.MAROON: 7,
}

#: Hue is circular: red sits at both ~0 and ~179, so a naive absolute
#: difference reports maximum distance between two nearly identical reds.
HUE_PERIOD = 180


def hue_distance(a: float, b: float) -> float:
    """Shortest distance between two hues on the circular 0-179 scale."""
    diff = abs(a - b) % HUE_PERIOD
    return min(diff, HUE_PERIOD - diff)


def sample_ball_hsv(
    hsv: np.ndarray, center: tuple[int, int], radius: float
) -> tuple[np.ndarray, float]:
    """Median HSV of a ball's interior, plus the value spread across it.

    Samples only the inner 60% of the disc. The rim is where felt bleeds in
    through antialiasing and where the sphere curves away from the light, so
    including it drags every colour toward the felt's green and toward black.

    The **median**, not the mean: a specular highlight is a small number of
    near-white pixels, and a mean is pulled hard by them while a median ignores
    them. That highlight is present on every ball under a projector.

    Returns:
        ``(median_hsv, value_std)``. The standard deviation of the value channel
        is what separates stripes from solids -- see
        :func:`classify_stripe_or_solid`.
    """
    cx, cy = center
    inner = max(1, int(radius * 0.6))
    y0, y1 = max(0, cy - inner), min(hsv.shape[0], cy + inner + 1)
    x0, x1 = max(0, cx - inner), min(hsv.shape[1], cx + inner + 1)
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0, 0, 0], dtype=np.float32), 0.0

    # Circular mask over the patch: a square patch on a small ball catches
    # enough corner felt to shift the median.
    h, w = patch.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    disc = (yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2 <= inner**2
    pixels = patch[disc] if disc.any() else patch.reshape(-1, 3)

    median = np.median(pixels, axis=0).astype(np.float32)
    value_std = float(np.std(pixels[:, 2]))
    return median, value_std


def classify_color(hsv_median: np.ndarray) -> tuple[BallColor, float]:
    """Nearest reference colour for a sampled HSV median.

    White and black are tested first and by different criteria, because for an
    achromatic ball the hue channel is pure noise -- a white ball's measured hue
    wanders across the whole scale between frames, so matching it by hue would
    be matching noise.

    Returns:
        ``(color, confidence)`` with confidence in 0-1, derived from how close
        the match was. A low confidence is meaningful information and should be
        propagated, not thresholded away here.
    """
    hue, sat, val = float(hsv_median[0]), float(hsv_median[1]), float(hsv_median[2])

    # Achromatic first.
    if sat < 60 and val > 150:
        # Confidence rises as saturation falls and brightness climbs: an ideal
        # cue ball is sat 0, val 255.
        confidence = min(1.0, (1.0 - sat / 60.0) * 0.5 + (val / 255.0) * 0.5)
        return BallColor.WHITE, confidence
    if val < 70:
        confidence = min(1.0, 1.0 - val / 70.0)
        return BallColor.BLACK, max(0.35, confidence)

    best_color = BallColor.UNKNOWN
    best_distance = float("inf")
    for color, (ref_h, ref_s, ref_v) in BALL_COLOR_REFERENCE.items():
        # Hue dominates, with saturation and value as weak tie-breakers. Hue is
        # the channel that actually identifies a ball; sat and val move a lot
        # with lighting and with whatever the projector is painting nearby.
        distance = (
            hue_distance(hue, ref_h) * 1.0
            + abs(sat - ref_s) * 0.12
            + abs(val - ref_v) * 0.06
        )
        if distance < best_distance:
            best_distance = distance
            best_color = color

    # 45 is roughly the distance at which a match stops being meaningful: a
    # quarter of the hue circle away, with the weak channels contributing.
    confidence = max(0.0, 1.0 - best_distance / 45.0)
    return best_color, confidence


def classify_stripe_or_solid(
    color: BallColor, value_std: float, stripe_std_threshold: float = 42.0
) -> BallKind:
    """Tell a stripe from a solid by how much its brightness varies.

    A striped ball is a coloured band on white, so the value channel across its
    face is bimodal and its standard deviation is high. A solid is uniform apart
    from shading and one highlight, so its deviation is low.

    This is genuinely unreliable at the resolution detection runs at, and it is
    treated as a hint rather than a fact: an overhead camera sees whatever part
    of the ball happens to be facing up, and a stripe rotated away from the
    camera looks exactly like a solid. Recovering the rack number properly needs
    the trained detector in :mod:`vision.inference`.
    """
    if color is BallColor.WHITE:
        return BallKind.CUE
    if color is BallColor.BLACK:
        return BallKind.EIGHT
    return BallKind.STRIPE if value_std > stripe_std_threshold else BallKind.SOLID


def guess_number(color: BallColor, kind: BallKind) -> int | None:
    """Best-effort rack number from colour and stripe/solid.

    Returns ``None`` rather than guessing when the pairing is not determined --
    an unknown number is harmless, whereas a wrong one shows the player the
    wrong ball highlighted and destroys trust in the overlay.
    """
    if kind is BallKind.EIGHT:
        return 8
    if kind is BallKind.CUE:
        return None
    base = SOLID_NUMBER_BY_COLOR.get(color)
    if base is None:
        return None
    if kind is BallKind.SOLID:
        return base
    if kind is BallKind.STRIPE:
        return base + 8  # 1->9, 2->10, ... 7->15
    return None
