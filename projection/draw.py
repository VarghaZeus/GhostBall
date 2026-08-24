"""RGBA canvas primitives.

The one module in the projection package that touches pixels. It exists so that
:mod:`projection.renderer` (what to draw) and :mod:`projection.effects` (when to
draw it) can both depend on the same drawing vocabulary without depending on
each other -- which they would otherwise have to, since effects are drawn into
the renderer's canvas and the renderer composes effects into its overlays.

Canvas contract
---------------
``HxWx4`` uint8, channel order **RGBA**, at projector resolution. Not BGRA:
:meth:`projection.display.Display.send_frame` converts once with
``cv2.COLOR_RGBA2BGR`` at the last moment, so every layer above it works in the
same order the config file uses. OpenCV's drawing calls do not care about
channel *meaning* -- they write whatever 4-tuple they are handed -- so the swap
happens exactly once, here, in :func:`_scalar`.

Alpha is the fade channel
-------------------------
The display multiplies RGB by per-pixel alpha before output, so a mark at
``alpha=128`` projects at half brightness. Everything that fades -- trails,
bursts, score popups -- fades by lowering alpha, never by dimming RGB toward
black. Dimming RGB works for a single mark on an empty canvas and breaks as soon
as two marks overlap.

Overdraw, not blending
----------------------
Marks are written over whatever is already in the canvas rather than blended
into it. Read-modify-write blending at 1080p costs more than the entire frame
budget (see the note in ``display.send_frame`` about the 60-75 ms float path),
and the projector's own additive optics do the job for free where the felt is
concerned. The consequence is that **draw order is z order**, so every render
function documents its order back to front.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.models import Vec2
from projection.themes import RGB

__all__ = [
    "new_canvas",
    "reset_canvas",
    "ensure_canvas",
    "polyline_length",
    "resample_polyline",
    "dash_segments",
    "draw_polyline",
    "draw_dashed_polyline",
    "draw_fading_polyline",
    "draw_circle",
    "draw_ring",
    "draw_glow_dot",
    "draw_cross",
    "draw_arrow",
    "draw_arc",
    "draw_text",
    "text_size",
    "draw_starburst",
    "draw_radial_lines",
    "FONT",
]

#: Hershey Simplex throughout. It is the only OpenCV font that stays legible
#: when rendered small and then thrown through a projector's own upscaler, and
#: unlike the duplex/triplex faces its strokes are single-width, so a 3 px
#: thickness at a 1.5 scale reads cleanly rather than filling in.
FONT = cv2.FONT_HERSHEY_SIMPLEX

_MAX_COORD = 1 << 14  # OpenCV silently misbehaves on absurd integer coordinates


# ---------------------------------------------------------------------------
# Canvas management
# ---------------------------------------------------------------------------


def new_canvas(settings: Settings | None = None) -> np.ndarray:
    """Allocate a transparent ``HxWx4`` canvas at projector resolution."""
    settings = settings or get_settings()
    return np.zeros((settings.projector.height, settings.projector.width, 4), dtype=np.uint8)


def reset_canvas(canvas: np.ndarray) -> np.ndarray:
    """Zero a canvas in place and return it.

    Reusing one buffer is worth real frame time, but not for the obvious reason.
    ``np.zeros`` of an 8.3 MB canvas measures 0.03 ms, because it gets
    already-zeroed pages from the OS and does no work -- so the *allocation* is
    free. The cost lands later, as a page fault on every 4 KB the drawing
    touches. Measured at 1080p on an x86 dev box, filling the frame once:

        allocate a new canvas, then draw   3.26 ms
        zero a reused canvas, then draw    0.54 ms

    2.7 ms, or 8% of a 33 ms budget, for nothing. ``fill`` over
    ``canvas[...] = 0`` is a wash (0.259 vs 0.253 ms); ``fill`` is used because
    it says what it means.
    """
    canvas.fill(0)
    return canvas


def ensure_canvas(
    canvas: np.ndarray | None,
    settings: Settings | None = None,
    *,
    clear: bool = True,
) -> np.ndarray:
    """Return a canvas of the right shape, reusing ``canvas`` if possible.

    A wrong-shaped buffer is replaced rather than rejected. It means the
    projector resolution changed mid-run -- which the API allows -- and raising
    would turn a settings change into a crash in the vision loop.

    ``clear`` defaults to ``True``, so a render function handed a buffer wipes
    it first. That default is load-bearing: without it a stale overlay
    accumulates frame on frame and fills the felt with light within a second.

    Pass ``clear=False`` to *stack* one overlay on another -- a trajectory under
    a scoreboard, say. The caller then owns clearing the buffer exactly once per
    frame, which is what :class:`modes.rendering.ModeRenderer` does. Getting
    this wrong in the other direction is visible immediately (the lower layers
    vanish) rather than slowly, which is why the unsafe direction is the one
    that has to be asked for.
    """
    settings = settings or get_settings()
    want = (settings.projector.height, settings.projector.width, 4)
    if canvas is None:
        return new_canvas(settings)
    if canvas.shape != want or canvas.dtype != np.uint8:
        return new_canvas(settings)
    return reset_canvas(canvas) if clear else canvas


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _scalar(color: RGB, alpha: int) -> tuple[int, int, int, int]:
    """An RGB triple plus alpha as the 4-tuple OpenCV writes into the canvas.

    The single point where channel order is decided. Values are clamped because
    a caller computing a fade can easily arrive at 256 or -1 by rounding, and
    OpenCV wraps rather than clamps -- an alpha of 256 becoming 0 would make an
    effect vanish at exactly its brightest frame.
    """
    return (
        min(255, max(0, int(color[0]))),
        min(255, max(0, int(color[1]))),
        min(255, max(0, int(color[2]))),
        min(255, max(0, int(alpha))),
    )


def _points_array(points: np.ndarray | list[Vec2] | list[tuple[float, float]]) -> np.ndarray:
    """Coerce any accepted point container to an ``Nx2`` int32 array.

    Non-finite points are dropped rather than clipped. They arrive from
    :meth:`~projection.mapper.ProjectionMapper.table_to_projector_batch`, which
    marks points on the transform's horizon as NaN; clipping one to the frame
    edge would draw a line to a place the ball is not going.
    """
    if isinstance(points, np.ndarray):
        array = points.astype(np.float64, copy=False).reshape(-1, 2)
    else:
        array = np.array([(p[0], p[1]) for p in points], dtype=np.float64).reshape(-1, 2)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    array = array[np.isfinite(array).all(axis=1)]
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.clip(array, -_MAX_COORD, _MAX_COORD).round().astype(np.int32)


def _int_point(point: Vec2 | tuple[float, float]) -> tuple[int, int] | None:
    """Round a point to int px, or ``None`` if it is not finite."""
    x, y = float(point[0]), float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (
        int(round(min(_MAX_COORD, max(-_MAX_COORD, x)))),
        int(round(min(_MAX_COORD, max(-_MAX_COORD, y)))),
    )


# ---------------------------------------------------------------------------
# Polyline geometry
# ---------------------------------------------------------------------------


def polyline_length(points: np.ndarray | list[Vec2]) -> float:
    """Total arc length of a polyline, in whatever units it is expressed in."""
    array = _points_array(points).astype(np.float64)
    if len(array) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def resample_polyline(points: np.ndarray | list[Vec2], spacing: float) -> np.ndarray:
    """Resample a polyline to roughly even spacing.

    Physics emits an event-driven path: a handful of points at contacts, with
    long straight runs between them. That is ideal for drawing a solid line and
    useless for anything that needs to walk the path at a fixed rate -- a
    particle, a gradient, a per-point alpha ramp. This turns one into the other.

    ``spacing`` is a target, not a guarantee: each segment gets a whole number of
    subdivisions, so the actual spacing varies by up to a factor of two on short
    segments. Even enough for a fade ramp, and it keeps the point count bounded
    without a second pass.
    """
    array = _points_array(points).astype(np.float64)
    if len(array) < 2 or spacing <= 0:
        return array
    out: list[np.ndarray] = [array[0]]
    for start, end in zip(array[:-1], array[1:], strict=True):
        seg = end - start
        length = float(np.linalg.norm(seg))
        steps = max(1, int(length / spacing))
        for i in range(1, steps + 1):
            out.append(start + seg * (i / steps))
    return np.array(out, dtype=np.float64)


def dash_segments(
    points: np.ndarray | list[Vec2],
    dash_px: float,
    gap_px: float,
    phase_px: float = 0.0,
) -> list[np.ndarray]:
    """Split a polyline into the drawn parts of a dash pattern.

    Walks the polyline in arc length so dashes stay a constant *length* around
    corners. The naive alternative -- alternating whole segments -- makes the
    dash length depend on where physics happened to put its vertices, so a
    rebound would visibly change the rhythm of the line at the cushion.

    Args:
        dash_px: Length of a drawn dash.
        gap_px: Length of the gap after it.
        phase_px: Distance to shift the pattern along the path. Advancing this
            with time is what makes the line crawl; a *negative* rate crawls
            forward, i.e. away from the cue ball, because increasing the phase
            moves the pattern backwards along the path.

    Returns:
        One ``Nx2`` float array per dash, ready for :func:`draw_polyline`.
    """
    array = _points_array(points).astype(np.float64)
    if len(array) < 2:
        return []
    dash_px = max(1.0, float(dash_px))
    gap_px = max(0.0, float(gap_px))
    period = dash_px + gap_px
    if period <= 0:
        return [array]

    # Where in the pattern the path starts. Modulo before the walk so a phase
    # that has been accumulating for an hour does not lose precision.
    offset = float(phase_px) % period

    dashes: list[np.ndarray] = []
    current: list[np.ndarray] = []
    travelled = 0.0
    for start, end in zip(array[:-1], array[1:], strict=True):
        seg = end - start
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 0:
            continue
        direction = seg / seg_len
        pos = 0.0
        while pos < seg_len:
            phase = (travelled + pos + offset) % period
            drawing = phase < dash_px
            # Distance until the pattern flips state, clipped to this segment.
            remaining = (dash_px - phase) if drawing else (period - phase)
            step = min(remaining, seg_len - pos)
            if drawing:
                a = start + direction * pos
                b = start + direction * (pos + step)
                if current and np.allclose(current[-1], a):
                    current.append(b)
                else:
                    if len(current) >= 2:
                        dashes.append(np.array(current))
                    current = [a, b]
            pos += max(step, 1e-6)
        travelled += seg_len

    if len(current) >= 2:
        dashes.append(np.array(current))
    return dashes


# ---------------------------------------------------------------------------
# Line drawing
# ---------------------------------------------------------------------------


def draw_polyline(
    canvas: np.ndarray,
    points: np.ndarray | list[Vec2],
    color: RGB,
    thickness: int = 4,
    alpha: int = 255,
    glow: bool = False,
) -> None:
    """Draw a connected polyline.

    ``glow`` adds two wider, dimmer passes underneath. That is a halo rather
    than a true bloom -- a real bloom needs a Gaussian blur of the whole canvas,
    which is 15-20 ms at 1080p and unaffordable per frame -- but on felt the
    difference is not visible, because the projector's own optics blur the edges
    anyway.
    """
    array = _points_array(points)
    if len(array) < 2:
        return
    poly = [array]
    if glow:
        cv2.polylines(canvas, poly, False, _scalar(color, alpha // 5), thickness * 4, cv2.LINE_AA)
        cv2.polylines(canvas, poly, False, _scalar(color, alpha // 2), thickness * 2, cv2.LINE_AA)
    cv2.polylines(canvas, poly, False, _scalar(color, alpha), max(1, thickness), cv2.LINE_AA)


def draw_dashed_polyline(
    canvas: np.ndarray,
    points: np.ndarray | list[Vec2],
    color: RGB,
    thickness: int = 4,
    alpha: int = 255,
    dash_px: float = 22.0,
    gap_px: float = 14.0,
    phase_px: float = 0.0,
    glow: bool = False,
) -> None:
    """Draw a polyline as an animated dash pattern.

    Dashed rather than solid for the cue path on purpose: a solid bright line
    laid over the felt hides the cloth under it, and the spec's "no
    interference" principle wants the player to still see the table through the
    guidance. The gaps also give the crawl something to be visible against.
    """
    for dash in dash_segments(points, dash_px, gap_px, phase_px):
        draw_polyline(canvas, dash, color, thickness, alpha, glow=glow)


def draw_fading_polyline(
    canvas: np.ndarray,
    points: np.ndarray | list[Vec2],
    color: RGB,
    thickness_head: int = 6,
    thickness_tail: int = 1,
    alpha_head: int = 230,
    alpha_tail: int = 0,
    glow: bool = False,
) -> None:
    """Draw a polyline whose brightness and width ramp from tail to head.

    The head is the *last* point, which is where a ball trail's newest sample
    is. Drawn as one short line per segment, tail first, so brighter head
    segments overwrite dimmer tail ones where they overlap -- a ball curving back
    across its own path otherwise gets a bright patch of old trail on top of new.
    """
    array = _points_array(points)
    if len(array) < 2:
        return
    count = len(array) - 1
    for i in range(count):
        t = (i + 1) / count
        alpha = int(round(alpha_tail + (alpha_head - alpha_tail) * t))
        if alpha <= 2:
            continue
        width = max(1, int(round(thickness_tail + (thickness_head - thickness_tail) * t)))
        segment = array[i : i + 2]
        if glow:
            cv2.polylines(
                canvas, [segment], False, _scalar(color, alpha // 3), width * 3, cv2.LINE_AA
            )
        cv2.polylines(canvas, [segment], False, _scalar(color, alpha), width, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def draw_circle(
    canvas: np.ndarray,
    center: Vec2 | tuple[float, float],
    radius: float,
    color: RGB,
    thickness: int = 2,
    alpha: int = 255,
    filled: bool = False,
) -> None:
    """Draw a circle, outlined by default."""
    point = _int_point(center)
    if point is None or radius < 0.5:
        return
    cv2.circle(
        canvas,
        point,
        int(round(radius)),
        _scalar(color, alpha),
        -1 if filled else max(1, thickness),
        cv2.LINE_AA,
    )


def draw_ring(
    canvas: np.ndarray,
    center: Vec2 | tuple[float, float],
    radius: float,
    color: RGB,
    thickness: int = 3,
    alpha: int = 255,
    glow: bool = False,
) -> None:
    """Draw a ring, optionally haloed. The building block of every burst."""
    if glow:
        draw_circle(canvas, center, radius, color, thickness * 3, alpha // 4)
    draw_circle(canvas, center, radius, color, thickness, alpha)


def draw_glow_dot(
    canvas: np.ndarray,
    center: Vec2 | tuple[float, float],
    radius: float,
    color: RGB,
    alpha: int = 255,
) -> None:
    """A filled dot with a soft halo, for spark particles and impact points.

    Three concentric fills rather than a blur: the same reasoning as
    :func:`draw_polyline`'s glow, and at these radii (2-12 px) three steps are
    already finer than the projector can resolve.
    """
    draw_circle(canvas, center, radius * 2.4, color, alpha=alpha // 6, filled=True)
    draw_circle(canvas, center, radius * 1.5, color, alpha=alpha // 3, filled=True)
    draw_circle(canvas, center, radius, color, alpha=alpha, filled=True)


def draw_cross(
    canvas: np.ndarray,
    center: Vec2 | tuple[float, float],
    size: float,
    color: RGB,
    thickness: int = 2,
    alpha: int = 255,
    rotate_deg: float = 0.0,
) -> None:
    """Draw a cross or, at 45 degrees, an X. Used for calibration targets."""
    point = _int_point(center)
    if point is None:
        return
    theta = math.radians(rotate_deg)
    for angle in (theta, theta + math.pi / 2):
        dx, dy = math.cos(angle) * size, math.sin(angle) * size
        cv2.line(
            canvas,
            (int(round(point[0] - dx)), int(round(point[1] - dy))),
            (int(round(point[0] + dx)), int(round(point[1] + dy))),
            _scalar(color, alpha),
            max(1, thickness),
            cv2.LINE_AA,
        )


def draw_arrow(
    canvas: np.ndarray,
    origin: Vec2 | tuple[float, float],
    angle_deg: float,
    length: float,
    color: RGB,
    thickness: int = 3,
    alpha: int = 255,
) -> None:
    """Draw an arrow from ``origin`` along a projector-space angle.

    ``angle_deg`` is measured in **projector** space -- screen-style, clockwise
    from +x, because +y is down. Table angles are counter-clockwise, so anything
    converting a physics angle must map it through the transform rather than
    passing the number straight in; see ``renderer._projector_angle``.
    """
    point = _int_point(origin)
    if point is None or length < 1.0:
        return
    theta = math.radians(angle_deg)
    tip = (
        int(round(point[0] + math.cos(theta) * length)),
        int(round(point[1] + math.sin(theta) * length)),
    )
    cv2.arrowedLine(
        canvas,
        point,
        tip,
        _scalar(color, alpha),
        max(1, thickness),
        cv2.LINE_AA,
        tipLength=0.35,
    )


def draw_arc(
    canvas: np.ndarray,
    center: Vec2 | tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    color: RGB,
    thickness: int = 2,
    alpha: int = 255,
) -> None:
    """Draw a circular arc. Angles are in projector space, clockwise from +x."""
    point = _int_point(center)
    if point is None or radius < 1.0:
        return
    cv2.ellipse(
        canvas,
        point,
        (int(round(radius)), int(round(radius))),
        0.0,
        start_deg,
        end_deg,
        _scalar(color, alpha),
        max(1, thickness),
        cv2.LINE_AA,
    )


def draw_radial_lines(
    canvas: np.ndarray,
    center: Vec2 | tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    color: RGB,
    count: int = 6,
    rotation_deg: float = 0.0,
    thickness: int = 2,
    alpha: int = 255,
) -> None:
    """Draw evenly spaced spokes between two radii.

    The vortex in the pocketing animation: spinning it by advancing
    ``rotation_deg`` and collapsing ``inner_radius`` toward zero is what sells
    the ball being swallowed.
    """
    point = _int_point(center)
    if point is None or count <= 0:
        return
    inner = max(0.0, inner_radius)
    for i in range(count):
        theta = math.radians(rotation_deg + i * 360.0 / count)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cv2.line(
            canvas,
            (int(round(point[0] + cos_t * inner)), int(round(point[1] + sin_t * inner))),
            (
                int(round(point[0] + cos_t * outer_radius)),
                int(round(point[1] + sin_t * outer_radius)),
            ),
            _scalar(color, alpha),
            max(1, thickness),
            cv2.LINE_AA,
        )


def draw_starburst(
    canvas: np.ndarray,
    center: Vec2 | tuple[float, float],
    radius: float,
    color: RGB,
    count: int = 8,
    rotation_deg: float = 0.0,
    alpha: int = 255,
    dot_radius: float = 3.0,
) -> None:
    """Scatter glowing dots on a circle. The spark half of a burst effect."""
    point = _int_point(center)
    if point is None or count <= 0:
        return
    for i in range(count):
        theta = math.radians(rotation_deg + i * 360.0 / count)
        draw_glow_dot(
            canvas,
            (point[0] + math.cos(theta) * radius, point[1] + math.sin(theta) * radius),
            dot_radius,
            color,
            alpha,
        )


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    """Rendered ``(width, height)`` of a string in px, for layout."""
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, max(1, thickness))
    return w, h + baseline


def draw_text(
    canvas: np.ndarray,
    text: str,
    position: Vec2 | tuple[float, float],
    color: RGB,
    scale: float = 1.0,
    thickness: int = 2,
    alpha: int = 255,
    anchor: str = "tl",
    outline: bool = True,
) -> tuple[int, int]:
    """Draw a string and return its ``(width, height)``.

    ``anchor`` names which corner of the text box ``position`` refers to:
    ``tl``, ``tr``, ``bl``, ``br``, ``c`` (centre), ``tc`` (top centre) or
    ``bc``. Anchoring rather than always top-left matters because most of this
    text is right- or centre-aligned against a rail, and computing the offset at
    every call site is where alignment bugs come from.

    ``outline`` draws a darker, thicker copy underneath. Counter-intuitive on a
    projector, where nothing can be darker than the felt -- but the outline is
    not for contrast against the felt, it is for contrast against the *overlay's
    own lines*, and text sitting on top of a bright trajectory is otherwise
    unreadable.
    """
    point = _int_point(position)
    if point is None or not text:
        return (0, 0)

    thickness = max(1, thickness)
    width, height = text_size(text, scale, thickness)
    x, y = point

    if anchor in ("tr", "br"):
        x -= width
    elif anchor in ("c", "tc", "bc"):
        x -= width // 2
    if anchor in ("tl", "tr", "tc", "c"):
        # cv2 positions text by its baseline; shift down so the caller's y is
        # the top of the box, which is how every other draw call here behaves.
        y += height if anchor != "c" else height // 2

    if outline:
        cv2.putText(
            canvas, text, (x, y), FONT, scale, _scalar((0, 0, 0), alpha), thickness + 3, cv2.LINE_AA
        )
    cv2.putText(canvas, text, (x, y), FONT, scale, _scalar(color, alpha), thickness, cv2.LINE_AA)
    return width, height
