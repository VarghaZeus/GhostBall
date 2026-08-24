"""Projector test patterns.

The answer to "is the projection aligned, in focus and the right size?" without
needing a game, a camera or a cue. Every pattern is drawn from *table*
coordinates through the calibration, so what the user is checking is the
transform and the physical placement together -- which is the only thing worth
checking, since either one being wrong looks identical on the felt.

Reading the patterns
--------------------
``grid``
    Lines every 6 inches with the rail diamonds marked. Judge **scale and
    rotation**: the grid should be square to the rails, and the diamond marks
    should sit on the real diamonds.
``corners``
    An L bracket in each corner of the playing surface plus a tick at each rail
    midpoint. Judge **alignment**: each bracket should hug the cushion nose. A
    consistent offset means the projector needs sliding; brackets off in
    different directions mean it needs squaring, or the homography needs
    re-solving.
``full_table``
    The whole surface: outline, pockets, spots and the head string. Judge
    **keystone and coverage**, and confirm nothing is clipped.
``crosshair``
    Centre cross with rings at known radii, and the output resolution. Judge
    **focus** -- the thin rings are the first thing to go soft -- and confirm the
    resolution the projector is actually receiving.

All four render at any calibration quality, including none: with an identity
calibration the pattern simply stretches the table across the whole frame, which
is itself the useful first observation.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

import numpy as np

from app.config import BALL_RADIUS_IN, Settings, get_settings
from app.models import Vec2
from physics.models import TableGeometry
from projection import draw
from projection.mapper import ProjectionMapper, identity_calibration
from projection.themes import Theme, resolve_theme

logger = logging.getLogger(__name__)

__all__ = ["TestPattern", "render_test_pattern", "default_mapper"]

#: Grid spacing in inches. Six is a divisor of both 76 and 38 in the sense that
#: matters -- it lands a line on the centre spot and near the diamonds -- and it
#: gives about 12 x 6 cells, dense enough to see a rotation of a degree or two
#: and sparse enough not to look like a mesh.
GRID_SPACING_IN = 6.0


class TestPattern(str, Enum):
    """The available patterns. String-valued so a CLI flag maps straight in."""

    GRID = "grid"
    CORNERS = "corners"
    FULL_TABLE = "full_table"
    CROSSHAIR = "crosshair"


def default_mapper(settings: Settings | None = None) -> ProjectionMapper:
    """A mapper for callers that have none, stretching the table to the frame.

    Uses :func:`~projection.mapper.identity_calibration` rather than loading the
    saved one: a test pattern is often the thing being used to *decide* whether
    the saved calibration is any good, and quietly applying it would hide the
    answer.
    """
    return ProjectionMapper(identity_calibration(settings or get_settings()))


def render_test_pattern(
    pattern: TestPattern | str,
    mapper: ProjectionMapper | None = None,
    settings: Settings | None = None,
    canvas: np.ndarray | None = None,
    *,
    theme: Theme | None = None,
) -> np.ndarray:
    """Render one test pattern to an RGBA canvas at projector resolution.

    Args:
        pattern: A :class:`TestPattern` or its string value.
        mapper: Transform to draw through. Defaults to :func:`default_mapper`.
        settings: Config. Defaults to the global settings.
        canvas: Reusable buffer. Zeroed in place when the shape matches, which
            is what lets the calibration overlay stack two patterns by passing
            the same canvas twice -- the second call must *not* clear the first,
            so it is the caller's job to pass a canvas it already drew into and
            accept that it will not be re-zeroed.
        theme: Palette override. Defaults to the configured theme.

    Returns:
        The canvas, for chaining.
    """
    settings = settings or get_settings()
    theme = theme or resolve_theme(settings)
    mapper = mapper or default_mapper(settings)
    pattern = TestPattern(pattern)

    # Only allocate-or-clear when handed nothing. See the ``canvas`` note above:
    # stacking patterns depends on a passed-in canvas being left alone.
    if canvas is None:
        canvas = draw.new_canvas(settings)

    geometry = TableGeometry.from_settings(settings)
    if pattern is TestPattern.GRID:
        _draw_grid(canvas, mapper, geometry, theme, settings)
    elif pattern is TestPattern.CORNERS:
        _draw_corners(canvas, mapper, geometry, theme, settings)
    elif pattern is TestPattern.FULL_TABLE:
        _draw_full_table(canvas, mapper, geometry, theme, settings)
    else:
        _draw_crosshair(canvas, mapper, geometry, theme, settings)
    return canvas


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def _label_scale(ppi: float, inches_tall: float) -> float:
    """Font scale for text roughly ``inches_tall`` on the felt.

    Same sizing rule as the renderer's, and for the same reason: these labels
    are read off cloth from a few feet away while the reader has both hands on a
    projector, so they have to be sized physically rather than in pixels.
    Hershey Simplex at scale 1.0 is about 22 px of cap height.
    """
    return max(0.5, ppi * inches_tall / 22.0)


def _table_line(
    canvas: np.ndarray,
    mapper: ProjectionMapper,
    start: Vec2,
    end: Vec2,
    color: tuple[int, int, int],
    thickness: int = 2,
    alpha: int = 200,
    subdivisions: int = 1,
) -> None:
    """Draw a straight table-space line through the transform.

    ``subdivisions`` matters under a homography: a straight line in table space
    is still straight in projector space (a homography preserves lines), so two
    endpoints suffice -- but subdividing is kept as an option because it is the
    honest way to *verify* that, and a curved rendering of a subdivided line is
    an immediate sign the transform is not a homography at all.
    """
    if subdivisions <= 1:
        points = [start, end]
    else:
        points = [
            Vec2(
                start.x + (end.x - start.x) * i / subdivisions,
                start.y + (end.y - start.y) * i / subdivisions,
            )
            for i in range(subdivisions + 1)
        ]
    draw.draw_polyline(canvas, mapper.table_to_projector_batch(points), color, thickness, alpha)


def _draw_grid(
    canvas: np.ndarray,
    mapper: ProjectionMapper,
    geometry: TableGeometry,
    theme: Theme,
    settings: Settings,
) -> None:
    """Grid at :data:`GRID_SPACING_IN`, with the rail diamonds marked."""
    length, width = geometry.length_in, geometry.width_in
    ppi = mapper.pixels_per_inch()

    # Interior lines dimmer than the border, so the table edge stays the
    # dominant reference and the grid reads as secondary.
    x = 0.0
    while x <= length + 1e-6:
        emphasis = abs(x - length / 2.0) < 1e-6
        _table_line(
            canvas,
            mapper,
            Vec2(x, 0.0),
            Vec2(x, width),
            theme.object_path,
            thickness=3 if emphasis else 1,
            alpha=220 if emphasis else 120,
        )
        x += GRID_SPACING_IN

    y = 0.0
    while y <= width + 1e-6:
        emphasis = abs(y - width / 2.0) < 1e-6
        _table_line(
            canvas,
            mapper,
            Vec2(0.0, y),
            Vec2(length, y),
            theme.object_path,
            thickness=3 if emphasis else 1,
            alpha=220 if emphasis else 120,
        )
        y += GRID_SPACING_IN

    # Border last so it sits on top of the grid lines meeting it.
    _draw_outline(canvas, mapper, geometry, theme.cue_path, thickness=3, alpha=250)

    # Rail diamonds: eight along each long rail, four along each short one, which
    # is the standard layout. These are the marks a player already knows the
    # position of, so they are the most sensitive alignment check available
    # without a measuring tape.
    for i in range(1, 8):
        x = length * i / 8.0
        for y_rail in (0.0, width):
            draw.draw_circle(
                canvas, mapper.table_to_projector(Vec2(x, y_rail)), max(3.0, ppi * 0.18),
                theme.impact, alpha=235, filled=True,
            )
    for i in range(1, 4):
        y = width * i / 4.0
        for x_rail in (0.0, length):
            draw.draw_circle(
                canvas, mapper.table_to_projector(Vec2(x_rail, y)), max(3.0, ppi * 0.18),
                theme.impact, alpha=235, filled=True,
            )

    draw.draw_text(
        canvas,
        f'{GRID_SPACING_IN:.0f}" grid  |  {length:.0f} x {width:.0f}"',
        mapper.table_to_projector(Vec2(length / 2.0, width / 2.0)),
        theme.text,
        scale=_label_scale(ppi, 1.3),
        thickness=2,
        alpha=200,
        anchor="c",
    )


def _draw_corners(
    canvas: np.ndarray,
    mapper: ProjectionMapper,
    geometry: TableGeometry,
    theme: Theme,
    settings: Settings,
) -> None:
    """An L bracket in each corner, plus a tick at each rail midpoint.

    Brackets rather than crosses: a cross centred on a corner has half of itself
    off the table, and "is the centre of that cross on the cushion nose" is a
    harder judgement than "do these two lines lie along the two cushions".
    """
    length, width = geometry.length_in, geometry.width_in
    arm = min(length, width) * 0.18

    corners = [
        (Vec2(0.0, 0.0), 1.0, 1.0, "TL"),
        (Vec2(length, 0.0), -1.0, 1.0, "TR"),
        (Vec2(length, width), -1.0, -1.0, "BR"),
        (Vec2(0.0, width), 1.0, -1.0, "BL"),
    ]
    for corner, sx, sy, label in corners:
        _table_line(
            canvas, mapper, corner, Vec2(corner.x + arm * sx, corner.y), theme.cue_path, 4, 250
        )
        _table_line(
            canvas, mapper, corner, Vec2(corner.x, corner.y + arm * sy), theme.cue_path, 4, 250
        )
        draw.draw_text(
            canvas,
            label,
            mapper.table_to_projector(Vec2(corner.x + arm * 0.35 * sx, corner.y + arm * 0.35 * sy)),
            theme.cue_path,
            scale=_label_scale(mapper.pixels_per_inch(), 1.6),
            thickness=3,
            anchor="c",
        )

    midpoints = [
        Vec2(length / 2.0, 0.0),
        Vec2(length / 2.0, width),
        Vec2(0.0, width / 2.0),
        Vec2(length, width / 2.0),
    ]
    tick = min(length, width) * 0.05
    for point in midpoints:
        inward_x = tick if point.x == 0.0 else -tick if point.x == length else 0.0
        inward_y = tick if point.y == 0.0 else -tick if point.y == width else 0.0
        _table_line(
            canvas,
            mapper,
            point,
            Vec2(point.x + inward_x, point.y + inward_y),
            theme.impact,
            thickness=3,
            alpha=240,
        )


def _draw_full_table(
    canvas: np.ndarray,
    mapper: ProjectionMapper,
    geometry: TableGeometry,
    theme: Theme,
    settings: Settings,
) -> None:
    """Outline, cushion inset, pockets, spots and the head string."""
    length, width = geometry.length_in, geometry.width_in
    ppi = mapper.pixels_per_inch()

    _draw_outline(canvas, mapper, geometry, theme.cue_path, thickness=4, alpha=250)

    # The ball-radius inset the simulator actually bounces off. Showing it makes
    # the difference between "cushion" and "where a ball centre can go" visible,
    # which is otherwise a confusing 1.1 inch discrepancy when comparing a
    # predicted rebound against the felt.
    inset = BALL_RADIUS_IN
    _draw_rect(
        canvas,
        mapper,
        Vec2(inset, inset),
        Vec2(length - inset, width - inset),
        theme.object_path,
        thickness=1,
        alpha=110,
    )

    for pocket in geometry.pocket_centers():
        draw.draw_ring(
            canvas,
            mapper.table_to_projector(pocket),
            ppi * settings.table.pocket_radius_in,
            theme.pocket_highlight,
            thickness=3,
            alpha=240,
            glow=theme.glow,
        )

    # Foot spot at 1/4 length (where the rack apex goes), centre spot, and the
    # head string across the table at 3/4 length.
    for spot_x in (length * 0.25, length * 0.5):
        draw.draw_circle(
            canvas,
            mapper.table_to_projector(Vec2(spot_x, width / 2.0)),
            max(3.0, ppi * 0.25),
            theme.impact,
            alpha=240,
            filled=True,
        )
    _table_line(
        canvas,
        mapper,
        Vec2(length * 0.75, 0.0),
        Vec2(length * 0.75, width),
        theme.impact,
        thickness=2,
        alpha=170,
    )

    draw.draw_text(
        canvas,
        f'{settings.table_preset}  {length:.0f} x {width:.0f}"',
        mapper.table_to_projector(Vec2(length / 2.0, width * 0.62)),
        theme.text,
        scale=_label_scale(ppi, 1.6),
        thickness=3,
        alpha=210,
        anchor="tc",
    )


def _draw_crosshair(
    canvas: np.ndarray,
    mapper: ProjectionMapper,
    geometry: TableGeometry,
    theme: Theme,
    settings: Settings,
) -> None:
    """Centre cross, rings at known inch radii, and the output resolution."""
    length, width = geometry.length_in, geometry.width_in
    center = Vec2(length / 2.0, width / 2.0)
    center_px = mapper.table_to_projector(center)
    ppi = mapper.pixels_per_inch(center)

    _table_line(canvas, mapper, Vec2(0.0, width / 2.0), Vec2(length, width / 2.0), theme.cue_path, 2, 200)
    _table_line(canvas, mapper, Vec2(length / 2.0, 0.0), Vec2(length / 2.0, width), theme.cue_path, 2, 200)

    # Rings at physical radii, labelled. Reading off which ring is sharp and
    # which is soft is a far more sensitive focus test than looking at text.
    #
    # Labels are staggered around the circle rather than all placed to the right.
    # Stacked on one radius they collide: the inner two rings here are 30 and
    # 80 px apart while a label is ~50 px wide, so the small-radius labels --
    # the ones that matter most for focus -- are exactly the ones that get
    # buried.
    radii = (BALL_RADIUS_IN, 3.0, 6.0, 12.0)
    for index, radius_in in enumerate(radii):
        radius_px = ppi * radius_in
        draw.draw_circle(canvas, center_px, radius_px, theme.object_path, thickness=1, alpha=200)
        theta = math.radians(-45.0 - index * 30.0)
        draw.draw_text(
            canvas,
            f'{radius_in:g}"',
            Vec2(
                center_px.x + math.cos(theta) * radius_px,
                center_px.y + math.sin(theta) * radius_px,
            ),
            theme.object_path,
            scale=_label_scale(ppi, 0.8),
            thickness=2,
            alpha=190,
            anchor="c",
        )

    draw.draw_cross(canvas, center_px, ppi * 1.5, theme.impact, thickness=3, alpha=255)
    draw.draw_text(
        canvas,
        f"{settings.projector.width} x {settings.projector.height}  |  {ppi:.1f} px/in",
        Vec2(center_px.x, center_px.y + ppi * 14.0),
        theme.text,
        scale=_label_scale(ppi, 1.4),
        thickness=3,
        anchor="tc",
    )


# ---------------------------------------------------------------------------
# Shared shapes
# ---------------------------------------------------------------------------


def _draw_outline(
    canvas: np.ndarray,
    mapper: ProjectionMapper,
    geometry: TableGeometry,
    color: tuple[int, int, int],
    thickness: int = 3,
    alpha: int = 250,
) -> None:
    """The playing surface border, as a closed quad through the transform."""
    _draw_rect(
        canvas, mapper, Vec2(0.0, 0.0), Vec2(geometry.length_in, geometry.width_in), color, thickness, alpha
    )


def _draw_rect(
    canvas: np.ndarray,
    mapper: ProjectionMapper,
    top_left: Vec2,
    bottom_right: Vec2,
    color: tuple[int, int, int],
    thickness: int = 2,
    alpha: int = 200,
) -> None:
    """A closed table-space rectangle, mapped corner by corner."""
    corners = [
        top_left,
        Vec2(bottom_right.x, top_left.y),
        bottom_right,
        Vec2(top_left.x, bottom_right.y),
        top_left,
    ]
    draw.draw_polyline(canvas, mapper.table_to_projector_batch(corners), color, thickness, alpha)
