"""What the projector shows when there is nothing to play on.

Before this existed the system started in freeplay and projected a scoreboard
over an empty room, which is a confusing thing for a machine to do: it looks
like it is working, and it looks like the detection is broken, and there is no
way to tell those apart from the felt.

Two constraints shape everything here.

**It is drawn without a calibration.** By definition these screens appear when
the table has not been found, so there is no homography and the mapper is an
identity stretch of the table onto the projector frame. Nothing here may depend
on table geometry being meaningful -- so it is laid out in fractions of the
output, not in inches.

**It is read from across a room, in the dark, by somebody holding a bracket.**
Large type, one instruction, high contrast, and no more than two lines of it.
Anything longer belongs on the phone, which is why the panel carries the same
state with room for detail.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from app.config import Settings, get_settings
from app.models import SystemState, Vec2
from projection import draw
from projection.themes import Theme, resolve_theme

logger = logging.getLogger(__name__)

__all__ = ["render_readiness_overlay", "should_draw_readiness"]

#: Colour per state. Not the theme's accent: these are status, not decoration,
#: and a red "no camera" has to read as a fault whatever palette is loaded.
_TINT = {
    SystemState.STARTING: (170, 200, 255),
    SystemState.NO_CAMERA: (255, 90, 80),
    SystemState.NO_TABLE: (255, 190, 90),
    SystemState.READY: (140, 230, 170),
}


def should_draw_readiness(state: SystemState) -> bool:
    """Whether this state wants the projector.

    ``READY`` does not: the mode owns the felt from that point, and a "Ready"
    banner sitting under a game would be exactly the interference the whole
    overlay design is meant to avoid.
    """
    return state is not SystemState.READY


def render_readiness_overlay(
    readiness,
    settings: Settings | None = None,
    canvas: np.ndarray | None = None,
    *,
    theme: Theme | None = None,
    now: float | None = None,
) -> np.ndarray:
    """Draw the current readiness state, full-frame.

    Args:
        readiness: An :class:`~app.readiness.Readiness`.
        canvas: Reusable RGBA buffer, cleared in place.
    """
    settings = settings or get_settings()
    theme = theme or resolve_theme(settings)
    now = time.perf_counter() if now is None else now

    canvas = draw.ensure_canvas(canvas, settings)
    height, width = canvas.shape[:2]
    colour = _TINT.get(readiness.state, theme.accent)

    # Sized as a fraction of the output rather than in inches, because the whole
    # point is that we do not know where the table is.
    title_scale = max(1.2, width / 900.0)
    body_scale = max(0.6, width / 2100.0)

    draw.draw_text(
        canvas,
        readiness.headline.upper(),
        Vec2(width / 2.0, height * 0.42),
        colour,
        scale=title_scale,
        thickness=max(2, int(width / 640)),
        anchor="c",
    )

    if readiness.detail:
        lines = _wrap_to_width(readiness.detail, body_scale, width * 0.8)
        for index, line in enumerate(lines):
            draw.draw_text(
                canvas,
                line,
                Vec2(width / 2.0, height * 0.52 + index * height * 0.055),
                colour,
                scale=body_scale,
                thickness=max(1, int(width / 1400)),
                anchor="c",
                alpha=225,
            )

    if readiness.state is SystemState.STARTING:
        _draw_pulse(canvas, colour, now)
    elif readiness.state is SystemState.NO_TABLE:
        # A rectangle in the middle of the projection, showing the shape the
        # cloth should fill. Far more use than the words: it turns "point it at
        # the table" into something you can line up against while you are still
        # on the ladder.
        _draw_target_frame(canvas, colour, settings, now)

    return canvas


def _wrap_to_width(text: str, scale: float, max_px: float) -> list[str]:
    """Greedy word wrap, measured in pixels rather than characters.

    Character counts are the wrong unit here: the font is proportional and the
    scale follows the output resolution, so a limit that fits at 1080p runs off
    the edge at 720p. Measuring is barely more code and cannot be wrong.

    Two lines maximum. A third will not be read from across a room, and if the
    message needs one it belongs on the phone instead.
    """
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.text_size(candidate, scale, 1)[0] > max_px and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:2]


def _draw_pulse(canvas: np.ndarray, colour, now: float) -> None:
    """A slow breathing ring. Says "alive and working" without claiming progress.

    Deliberately not a progress bar: startup has no measurable progress, and a
    bar that jumps to 100% instantly or crawls for an unknown time is a small
    lie either way.
    """
    height, width = canvas.shape[:2]
    phase = 0.5 + 0.5 * math.sin(now * 2.0)
    radius = min(width, height) * (0.05 + 0.012 * phase)
    draw.draw_ring(
        canvas,
        Vec2(width / 2.0, height * 0.62),
        radius,
        colour,
        thickness=3,
        alpha=int(90 + 120 * phase),
    )


def _draw_target_frame(canvas: np.ndarray, colour, settings: Settings, now: float) -> None:
    """An outline of the table's aspect ratio, for aiming the camera.

    Drawn at the configured table's proportions so that lining the cloth up
    inside it also gets the framing roughly right -- which is the next thing
    that has to be true anyway, since detection wants the whole table in view
    with a little margin.
    """
    height, width = canvas.shape[:2]
    aspect = settings.table.length_in / max(1e-6, settings.table.width_in)

    box_w = width * 0.5
    box_h = box_w / aspect
    if box_h > height * 0.3:
        box_h = height * 0.3
        box_w = box_h * aspect

    cx, cy = width / 2.0, height * 0.78
    x0, y0 = cx - box_w / 2.0, cy - box_h / 2.0
    x1, y1 = cx + box_w / 2.0, cy + box_h / 2.0

    # Corner brackets rather than a closed rectangle. A full outline reads as a
    # thing to look at; brackets read as a thing to put something inside.
    arm = min(box_w, box_h) * 0.28
    alpha = int(140 + 60 * (0.5 + 0.5 * math.sin(now * 2.0)))
    for (corner_x, corner_y), (dx, dy) in (
        ((x0, y0), (1, 1)),
        ((x1, y0), (-1, 1)),
        ((x1, y1), (-1, -1)),
        ((x0, y1), (1, -1)),
    ):
        draw.draw_polyline(
            canvas,
            [
                Vec2(corner_x + dx * arm, corner_y),
                Vec2(corner_x, corner_y),
                Vec2(corner_x, corner_y + dy * arm),
            ],
            colour,
            thickness=3,
            alpha=alpha,
        )
