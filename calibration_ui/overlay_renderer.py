"""Camera-preview overlays for the calibration wizard.

Phase 6.2. Distinct from :mod:`projection.renderer`, and the distinction is the
whole point of the wizard: that module draws what goes *out to the projector*,
this one annotates what the *camera sees*, and calibration is the act of making
the two agree. Sharing one renderer between them would make it impossible to
show the user the discrepancy they are correcting.

Everything here draws onto a BGR camera frame at camera resolution, and returns
a copy -- the input frame is also handed to the detector, and drawing on it in
place would corrupt detection.

Colours are **BGR**, not RGB
----------------------------
The opposite convention to :mod:`projection.draw`, and deliberately so: that
module builds an RGBA overlay for the projector and swaps once on the way out,
whereas these functions annotate a frame that came straight off the camera in
BGR and goes straight to a window in BGR. Converting here would mean two
pointless conversions per frame and one more place to get the order wrong. The
severity palette in :data:`SEVERITY_BGR` is the one to reach for.

Text is sized against the frame, not in pixels
----------------------------------------------
The console scales the finished frame down to fit a laptop or a Pi touchscreen,
so a size chosen in pixels comes out at whatever the scale factor happens to
make it. Every string here is sized as a fraction of frame *height* via
:func:`_scale_for`, which is what keeps the wizard's headline text at roughly
48pt on the console regardless of camera resolution.

Chaining, and why ``copy`` exists
---------------------------------
Copying by default is the safe contract and it is not free: a 1080p BGR frame is
6.2 MB, and copying one measures **2-2.6 ms** on an x86 dev box. The busiest
screen stacks five of these, so the copies alone were over half of a **22 ms**
annotation pass -- on the screen whose entire job is live feedback while the user
physically pushes a projector around.

So every function takes a keyword-only ``copy``. Leave it ``True`` for the call
that starts a chain and pass ``False`` for the rest::

    view = render_step_instructions(frame, state, text)     # copies once
    view = draw_table_outline(view, boundary, copy=False)   # draws in place
    view = draw_alignment_feedback(view, error, copy=False)

That takes the same five-overlay pass to **9.9 ms**, and a whole console frame
including the downscale to **13.5 ms** -- comfortably inside the wizard's ~25 ms
input poll, so the error readout tracks the user's hands.

The rule is simply never to pass ``copy=False`` with the *camera's own frame* as
the input: that frame also goes to the detector, and a wizard banner drawn
across it would be detected as part of the table. ``tests/test_calibration_ui.py``
pins that invariant across a whole wizard run rather than trusting the twenty-odd
call sites to respect it.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.models import AlignmentError, CalibrationState, Severity, TableBoundary, Vec2

logger = logging.getLogger(__name__)

__all__ = [
    "SEVERITY_BGR",
    "STEP_TITLES",
    "TOTAL_STEPS",
    "draw_table_outline",
    "draw_alignment_grid",
    "draw_corner_target",
    "draw_alignment_feedback",
    "draw_projected_vs_detected",
    "render_step_instructions",
    "draw_checklist",
    "draw_confidence_bar",
    "draw_metric_rows",
    "draw_countdown",
    "draw_banner_message",
    "draw_notice",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
TOTAL_STEPS = 7

#: One title per screen, shown in the banner so the user always knows where
#: they are in a seven-step process they did not choose to be in.
STEP_TITLES: dict[int, str] = {
    1: "Welcome",
    2: "Find the table",
    3: "Projector warm-up",
    4: "Corner mapping",
    5: "Fine-tune",
    6: "Test projection",
    7: "Done",
}

#: Verdict colours, BGR. Green / amber / red, per the spec's colour coding.
#: Chosen bright rather than saturated so they stay legible over green felt in
#: a dim room, which is where every one of them will actually be read.
SEVERITY_BGR: dict[str, tuple[int, int, int]] = {
    "info": (120, 235, 120),
    "warning": (60, 205, 255),
    "error": (80, 80, 255),
}

_WHITE = (255, 255, 255)
_DIM = (170, 170, 170)
_PANEL = (28, 24, 20)

#: Cap height as a fraction of frame height, per text role. The headline lands
#: at ~59 px on a 1080p frame, which survives the console's downscale as
#: roughly 48pt -- the size the spec asks for.
_HEADLINE = 0.055
_BODY = 0.030
_SMALL = 0.022

#: Corner handle radius in px at 1080p, scaled with the frame. Around 20 px is
#: the smallest reliable touch target, and the wizard is expected to be driven
#: from a tablet propped on the rail rather than from a keyboard.
_HANDLE_FRACTION = 0.019


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _canvas(frame: np.ndarray, copy: bool) -> np.ndarray:
    """The surface to draw on: a copy of the frame, or the frame itself.

    See the module note on chaining for when ``copy=False`` is safe.
    """
    return frame.copy() if copy else frame


def _scale_for(frame: np.ndarray, cap_fraction: float) -> float:
    """Font scale giving a cap height of ``cap_fraction`` of the frame height.

    Hershey Simplex at scale 1.0 is about 22 px of cap height, which is the same
    constant :mod:`projection.patterns` sizes its felt labels by.
    """
    return max(0.4, frame.shape[0] * cap_fraction / 22.0)


def _thickness_for(scale: float) -> int:
    """Stroke weight that keeps letterforms solid as the scale grows."""
    return max(1, int(round(scale * 1.2)))


def _fit_scale(text: str, scale: float, max_width: int | None) -> float:
    """Shrink ``scale`` until ``text`` fits ``max_width``.

    Necessary rather than cosmetic. Every headline in the wizard is a sentence
    written for a human, and sentences vary in length -- "Done." and "The
    projection is skewed. Raise or lower the front of the projector." both go
    through the same call. Sizing purely by frame height clips the long ones,
    and a clipped instruction is worse than a small one: the user cannot act on
    the half they cannot see.

    Floored at 40% of the requested size, so a pathologically long string comes
    out small rather than invisible. Callers with room for more than one line
    should use :func:`_fit_lines`, which stays legible where this bottoms out.
    """
    if max_width is None or max_width <= 0 or not text:
        return scale
    width, _ = cv2.getTextSize(text, FONT, scale, _thickness_for(scale))[0]
    if width <= max_width:
        return scale
    return max(scale * 0.4, scale * max_width / float(width))


def _wrap(text: str, scale: float, max_width: int) -> list[str]:
    """Greedily break ``text`` into lines no wider than ``max_width``.

    Word wrapping, not character wrapping: a single word longer than the line is
    left to overflow rather than hyphenated, because the only strings that long
    in this wizard are file paths, and a path broken across lines cannot be
    typed back in.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and cv2.getTextSize(candidate, FONT, scale, _thickness_for(scale))[0][0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _fit_lines(
    text: str, scale: float, max_width: int, max_lines: int = 3
) -> tuple[list[str], float]:
    """Wrap ``text``, shrinking until it fits in ``max_lines``.

    Wrapping rather than shrinking alone, because shrinking alone runs out. The
    wizard's own rule is one sentence per screen, but "The projection is skewed.
    Raise or lower the front of the projector." is already wide enough at
    headline size to overflow a 1080p frame, and the fallback advice strings are
    longer still. Below about 40% of headline size the text stops being readable
    across a dim room, which is the situation the whole wizard is designed for
    -- so past that point the answer is another line, not smaller type.

    Returns ``(lines, scale)``; the caller lays them out, since only it knows how
    much vertical room it has.
    """
    for _attempt in range(6):
        lines = _wrap(text, scale, max_width)
        if len(lines) <= max_lines:
            return lines, scale
        scale *= 0.85
    return _wrap(text, scale, max_width)[:max_lines], scale


def _draw_lines(
    canvas: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    *,
    anchor: str = "tl",
) -> int:
    """Draw wrapped lines downward from ``origin``. Returns the height used."""
    line_height = int(scale * 30)
    for index, line in enumerate(lines):
        _text(canvas, line, (origin[0], origin[1] + index * line_height), color, scale, anchor=anchor)
    return line_height * len(lines)


def _text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    *,
    anchor: str = "tl",
    thickness: int | None = None,
    max_width: int | None = None,
) -> tuple[int, int]:
    """Draw one string with a dark halo, returning its ``(width, height)``.

    The halo is not for contrast against the felt -- it is for contrast against
    the wizard's own bright overlay lines, which the text frequently sits on
    top of and would otherwise disappear into.

    ``max_width`` shrinks the text to fit rather than letting it run off the
    frame; see :func:`_fit_scale`.
    """
    if not text:
        return (0, 0)
    scale = _fit_scale(text, scale, max_width)
    thickness = thickness if thickness is not None else _thickness_for(scale)
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    if anchor in ("tr", "br"):
        x -= width
    elif anchor in ("tc", "bc", "c"):
        x -= width // 2
    if anchor in ("tl", "tr", "tc"):
        y += height
    elif anchor == "c":
        y += height // 2

    cv2.putText(canvas, text, (x, y), FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return width, height + baseline


def _panel(
    canvas: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    *,
    border: tuple[int, int, int] | None = None,
    opacity: float = 0.72,
) -> None:
    """Darken a rectangle in place so text over it is readable.

    Blended rather than filled: the user is judging a live camera image, and a
    solid box over a third of it hides the very thing they are being asked to
    look at. Clamped to the frame, because a panel positioned from a metric
    that came back as ``inf`` would otherwise be an OpenCV assertion.
    """
    height, width = canvas.shape[:2]
    x0 = max(0, min(int(top_left[0]), width - 1))
    y0 = max(0, min(int(top_left[1]), height - 1))
    x1 = max(x0 + 1, min(int(bottom_right[0]), width))
    y1 = max(y0 + 1, min(int(bottom_right[1]), height))

    region = canvas[y0:y1, x0:x1]
    tint = np.full(region.shape, _PANEL, dtype=np.uint8)
    cv2.addWeighted(tint, opacity, region, 1.0 - opacity, 0.0, dst=region)
    if border is not None:
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), border, 2, cv2.LINE_AA)


def _severity_color(severity: str) -> tuple[int, int, int]:
    return SEVERITY_BGR.get(severity, _WHITE)


# ---------------------------------------------------------------------------
# Table and grid
# ---------------------------------------------------------------------------


def draw_table_outline(
    frame: np.ndarray,
    boundary: TableBoundary,
    color: tuple[int, int, int] = (0, 255, 255),
    show_handles: bool = True,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Outline the detected table with draggable corner handles.

    Handles need to be large enough to hit on a touchscreen -- around 20 px
    radius -- since the wizard is expected to be driven from a tablet at the
    table rather than from a keyboard. They are drawn as a ring with a small
    solid centre rather than a filled disc: the point being dragged is a
    *corner*, and a filled disc hides the pixels the user is trying to line the
    corner up with.
    """
    canvas = _canvas(frame, copy)
    corners = boundary.corners()
    outline = np.array([corner.as_int() for corner in corners], dtype=np.int32)
    cv2.polylines(canvas, [outline], True, color, 3, cv2.LINE_AA)

    if not show_handles:
        return canvas

    radius = max(12, int(frame.shape[0] * _HANDLE_FRACTION))
    scale = _scale_for(frame, _SMALL)
    from calibration_ui.metrics import CORNER_LABELS, CORNER_NAMES

    for corner, name in zip(corners, CORNER_NAMES, strict=True):
        point = corner.as_int()
        cv2.circle(canvas, point, radius, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, point, 3, color, -1, cv2.LINE_AA)
        # Label pushed toward the table centre, so a corner near the frame edge
        # does not have its name clipped off.
        toward_x = -1 if corner.x > boundary.center.x else 1
        toward_y = -1 if corner.y > boundary.center.y else 1
        _text(
            canvas,
            CORNER_LABELS[name],
            (point[0] + toward_x * (radius + 8), point[1] + toward_y * (radius + 8)),
            color,
            scale,
            anchor="tl" if toward_x > 0 else "tr",
        )
    return canvas


def draw_alignment_grid(
    frame: np.ndarray,
    boundary: TableBoundary,
    divisions: int = 8,
    color: tuple[int, int, int] = (120, 120, 120),
    settings: Settings | None = None,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Draw a perspective-correct grid over the felt.

    Perspective-correct matters: a grid drawn as evenly spaced screen lines will
    look wrong against a table shot from an angle, and the user will try to
    correct a distortion that is not there. Space the lines in table
    coordinates, then map each one back into camera space.

    Returns the frame unchanged (still a copy) when the corners are too
    degenerate to solve a homography from. A missing grid is a small loss; a
    traceback out of a drawing call would end the wizard.
    """
    canvas = _canvas(frame, copy)
    from vision.calibration import CalibrationError, compute_perspective_transform, table_to_camera_coords

    try:
        _camera_to_table, table_to_camera = compute_perspective_transform(boundary, settings)
    except CalibrationError as exc:
        logger.warning("cannot draw the alignment grid: %s", exc)
        return canvas

    settings = settings or get_settings()
    length, width = settings.table.length_in, settings.table.width_in
    # Blended in one pass at the end rather than per line: the grid has to be
    # faint enough to see the cloth through, and drawing each line at low alpha
    # would make the intersections twice as bright as the lines.
    overlay = canvas.copy()
    for index in range(1, divisions):
        x = length * index / divisions
        start = table_to_camera_coords(Vec2(x, 0.0), table_to_camera)
        end = table_to_camera_coords(Vec2(x, width), table_to_camera)
        cv2.line(overlay, start.as_int(), end.as_int(), color, 1, cv2.LINE_AA)

    # Half as many divisions across the short axis, so the cells come out
    # roughly square on a 2:1 table instead of tall and thin.
    for index in range(1, max(2, divisions // 2)):
        y = width * index / max(2, divisions // 2)
        start = table_to_camera_coords(Vec2(0.0, y), table_to_camera)
        end = table_to_camera_coords(Vec2(length, y), table_to_camera)
        cv2.line(overlay, start.as_int(), end.as_int(), color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0.0, dst=canvas)
    return canvas


# ---------------------------------------------------------------------------
# Corner targets
# ---------------------------------------------------------------------------


def draw_corner_target(
    frame: np.ndarray,
    position: Vec2,
    label: str,
    is_active: bool = False,
    is_recorded: bool = False,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Mark one corner target on the preview.

    Three visual states, and they need to be distinguishable at a glance from a
    few feet away: pending, active (the one being placed now), and recorded.
    Colour alone is not sufficient -- vary the shape too.

    ==========  ==========================================  ==========
    State       Shape                                       Colour
    ==========  ==========================================  ==========
    pending     small hollow ring                           grey
    active      long crosshair inside a double ring         amber
    recorded    diagonal X inside a ring, centre filled     green
    ==========  ==========================================  ==========

    The active target is a *crosshair* rather than a dot on purpose. A dot's
    centre is ambiguous by several pixels once it is a few inches across on
    cloth, and that ambiguity propagates straight into the alignment RMSE --
    which is the number the whole system's accuracy is judged by.
    """
    canvas = _canvas(frame, copy)
    point = position.as_int()
    unit = max(10, int(frame.shape[0] * _HANDLE_FRACTION))

    if is_recorded:
        color = SEVERITY_BGR["info"]
        cv2.circle(canvas, point, unit, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, point, 4, color, -1, cv2.LINE_AA)
        arm = int(unit * 0.62)
        cv2.line(
            canvas, (point[0] - arm, point[1] - arm), (point[0] + arm, point[1] + arm),
            color, 2, cv2.LINE_AA,
        )
        cv2.line(
            canvas, (point[0] - arm, point[1] + arm), (point[0] + arm, point[1] - arm),
            color, 2, cv2.LINE_AA,
        )
    elif is_active:
        color = SEVERITY_BGR["warning"]
        arm = unit * 2
        cv2.line(canvas, (point[0] - arm, point[1]), (point[0] + arm, point[1]), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (point[0], point[1] - arm), (point[0], point[1] + arm), color, 2, cv2.LINE_AA)
        cv2.circle(canvas, point, unit, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, point, int(unit * 1.7), color, 1, cv2.LINE_AA)
    else:
        color = _DIM
        cv2.circle(canvas, point, int(unit * 0.7), color, 1, cv2.LINE_AA)

    scale = _scale_for(frame, _SMALL if not is_active else _BODY)
    _text(canvas, label, (point[0], point[1] + unit * 2 + 6), color, scale, anchor="tc")
    return canvas


def draw_projected_vs_detected(
    frame: np.ndarray,
    projected_points: list[Vec2],
    detected_points: list[Vec2],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Show where the projection landed versus where it should have.

    The most informative view in the wizard: pair each projected mark with its
    intended position and connect them with a line, so the residual error is
    visible as a set of little arrows. A consistent direction across all four
    means an offset error; a rotational pattern means the projector is not
    square. That distinction is what tells the user whether to slide the
    projector or turn it.

    Arrows point **from the projected mark to the table corner** -- the
    direction the light has to travel, so "follow the arrows" is literally the
    instruction. Reversing them would read as "here is where it came from",
    which is true and useless.

    Extra points in either list are ignored rather than treated as an error:
    losing sight of one mark mid-adjustment is routine, and the other three
    still say something useful.
    """
    canvas = _canvas(frame, copy)
    pairs = list(zip(projected_points, detected_points, strict=False))
    if len(projected_points) != len(detected_points):
        logger.debug(
            "residual view got %d projected and %d detected points; pairing the first %d",
            len(projected_points),
            len(detected_points),
            len(pairs),
        )

    radius = max(8, int(frame.shape[0] * 0.012))
    scale = _scale_for(frame, _SMALL)

    for projected, detected in pairs:
        start, end = projected.as_int(), detected.as_int()
        error_px = projected.distance_to(detected)

        cv2.circle(canvas, end, radius, SEVERITY_BGR["info"], 2, cv2.LINE_AA)
        cv2.circle(canvas, start, radius, SEVERITY_BGR["error"], 2, cv2.LINE_AA)
        # Only draw the arrow when there is a visible gap. An arrowhead on a
        # two-pixel shaft is a blob that reads as a third marker, exactly when
        # the user has succeeded and wants confirmation that nothing is left.
        if error_px > radius:
            cv2.arrowedLine(
                canvas, start, end, _WHITE, 2, cv2.LINE_AA, tipLength=min(0.4, radius / error_px)
            )
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            _text(
                canvas,
                f"{error_px:.0f} px",
                (midpoint[0], midpoint[1] - radius),
                _WHITE,
                scale,
                anchor="bc",
            )

    if pairs:
        _legend(canvas, scale)
    return canvas


def _legend(canvas: np.ndarray, scale: float) -> None:
    """Explain the two circle colours, once, in the top-right corner.

    Without it the view is two sets of rings and an arrow, and which one is the
    table and which one is the light is exactly the thing the user cannot infer
    -- they are looking at the screen precisely because they cannot tell.
    """
    height, width = canvas.shape[:2]
    entries = [(SEVERITY_BGR["info"], "table corner"), (SEVERITY_BGR["error"], "projected mark")]
    line_h = int(scale * 34)
    right = width - 16
    top = int(height * 0.16)

    _panel(canvas, (right - int(width * 0.20), top - 8), (right + 8, top + line_h * len(entries) + 8))
    for index, (color, text) in enumerate(entries):
        y = top + line_h * index + line_h // 2
        cv2.circle(canvas, (right - int(width * 0.19), y), max(5, line_h // 4), color, 2, cv2.LINE_AA)
        _text(canvas, text, (right - int(width * 0.185) + line_h // 3, y), color, scale, anchor="bl")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def draw_alignment_feedback(
    frame: np.ndarray,
    error: AlignmentError,
    settings: Settings | None = None,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Overlay the alignment verdict and its instruction.

    ``error.message`` is written for a human and is rendered verbatim. The panel
    is tinted by ``error.severity`` so the user can tell good from bad without
    reading, then read for the detail.

    The numbers underneath are secondary on purpose. They are there for whoever
    is tuning the rig and for the log; the person holding the projector needs
    the sentence.
    """
    settings = settings or get_settings()
    rmse = "unmeasured" if not math.isfinite(error.total_rmse) else f"{error.total_rmse:.1f} px"
    detail = (
        f"RMSE {rmse}   |   offset {error.x_offset:+.0f}, {error.y_offset:+.0f} px"
        f"   |   rotation {error.rotation:+.1f} deg"
    )
    return _verdict_panel(frame, error.message, error.severity, detail, copy)


def draw_notice(
    frame: np.ndarray,
    text: str,
    severity: Severity = "info",
    *,
    copy: bool = True,
) -> np.ndarray:
    """The same bottom panel, for a message that is not about alignment.

    Split from :func:`draw_alignment_feedback` because that one prints the RMSE
    and offsets underneath, and a screen with no alignment to report was showing
    "RMSE unmeasured | offset +0, +0 px" under a warning about the camera. Zeros
    presented as measurements are worse than no numbers.
    """
    return _verdict_panel(frame, text, severity, None, copy)


def _verdict_panel(
    frame: np.ndarray,
    message: str,
    severity: Severity,
    detail: str | None,
    copy: bool = True,
) -> np.ndarray:
    """The bottom message strip, tinted by severity.

    Deliberately shallow. An earlier version started at 79% of frame height and
    covered both bottom table corners -- on the corner-mapping screen, which is
    the one screen whose entire purpose is looking at the corners. It is now
    about half that, and the headline shrinks to fit rather than the panel
    growing to hold it.
    """
    canvas = _canvas(frame, copy)
    height, width = canvas.shape[:2]
    color = _severity_color(severity)

    lines, scale = _fit_lines(message, _scale_for(canvas, _HEADLINE * 0.82), width - 64, max_lines=2)
    message_height = int(scale * 30) * len(lines)
    detail_height = int(height * 0.045) if detail else int(height * 0.015)
    top = min(
        int(height * (0.855 if detail else 0.885)),
        height - message_height - detail_height - 12,
    )

    _panel(canvas, (0, top), (width, height), border=color)
    _draw_lines(canvas, lines, (width // 2, top + int(height * 0.012)), color, scale, anchor="tc")
    if detail:
        _text(
            canvas,
            detail,
            (width // 2, height - int(height * 0.012)),
            _DIM,
            _scale_for(canvas, _SMALL),
            anchor="bc",
            max_width=width - 64,
        )
    return canvas


def render_step_instructions(
    frame: np.ndarray,
    state: CalibrationState,
    instruction: str,
    settings: Settings | None = None,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Draw the step counter and current instruction.

    Keep it to one sentence. The under-ten-minutes target in the spec is not met
    by a user reading paragraphs on each of seven screens.

    The row of pips is worth its space: "step 4 of 7" tells someone how far
    through they are, but the pips tell them at a glance without reading, and
    the wizard is used by someone whose attention is mostly on the table.
    """
    settings = settings or get_settings()
    canvas = _canvas(frame, copy)
    height, width = canvas.shape[:2]

    instruction_top = int(height * 0.075)
    # Two lines at headline size do not fit the default banner, so a wrapped
    # instruction is drawn a size down. One line -- which is what the wizard's
    # own text always is -- keeps the full size.
    lines, scale = _fit_lines(instruction, _scale_for(canvas, _HEADLINE), width - 48, max_lines=1)
    if len(lines) > 1 or scale < _scale_for(canvas, _HEADLINE):
        lines, scale = _fit_lines(
            instruction, _scale_for(canvas, _HEADLINE * 0.62), width - 48, max_lines=3
        )

    # The panel is sized to the text rather than the text squeezed into a fixed
    # panel: a clipped instruction is the one thing on screen the user cannot do
    # without.
    bottom = max(int(height * 0.155), instruction_top + int(scale * 30) * len(lines) + 12)
    _panel(canvas, (0, 0), (width, bottom))

    title = STEP_TITLES.get(state.step, "")
    _text(
        canvas,
        f"STEP {state.step} OF {TOTAL_STEPS}   {title.upper()}",
        (24, int(height * 0.015)),
        _WHITE,
        _scale_for(canvas, _BODY),
    )
    _draw_lines(canvas, lines, (24, instruction_top), SEVERITY_BGR["warning"], scale)

    pip_r = max(5, int(height * 0.009))
    gap = pip_r * 3
    origin_x = width - 24 - gap * (TOTAL_STEPS - 1) - pip_r
    pip_y = int(height * 0.035)
    for step in range(1, TOTAL_STEPS + 1):
        center = (origin_x + gap * (step - 1), pip_y)
        if step < state.step:
            cv2.circle(canvas, center, pip_r, SEVERITY_BGR["info"], -1, cv2.LINE_AA)
        elif step == state.step:
            cv2.circle(canvas, center, pip_r, SEVERITY_BGR["warning"], -1, cv2.LINE_AA)
            cv2.circle(canvas, center, pip_r * 2, SEVERITY_BGR["warning"], 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, center, pip_r, _DIM, 1, cv2.LINE_AA)
    return canvas


def draw_banner_message(
    frame: np.ndarray, text: str, severity: Severity = "info",
    *,
    copy: bool = True,
) -> np.ndarray:
    """A single centred line over the middle of the frame.

    For the moments the wizard is doing something the user must wait through --
    blanking the projector, capturing a reference frame -- where saying nothing
    reads as a freeze.
    """
    canvas = _canvas(frame, copy)
    height, width = canvas.shape[:2]
    color = _severity_color(severity)
    scale = _fit_scale(text, _scale_for(canvas, _HEADLINE), int(width * 0.86))
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, scale, _thickness_for(scale))
    center_y = height // 2
    _panel(
        canvas,
        (width // 2 - text_w // 2 - 32, center_y - text_h),
        (width // 2 + text_w // 2 + 32, center_y + text_h),
        border=color,
    )
    _text(canvas, text, (width // 2, center_y), color, scale, anchor="c")
    return canvas


def draw_checklist(
    frame: np.ndarray, title: str, items: list[tuple[str, bool | None]],
    *,
    copy: bool = True,
) -> np.ndarray:
    """A pre-flight checklist with live pass/fail marks.

    ``None`` for an item's state means "we cannot check this, you must" -- which
    covers most of a physical setup, and is honest in a way that a tick the
    software did not earn would not be. Those get a hollow bullet, so a user
    scanning for red sees only the checks that genuinely failed.
    """
    canvas = _canvas(frame, copy)
    height, width = canvas.shape[:2]

    title_scale = _scale_for(canvas, _BODY)
    item_scale = _scale_for(canvas, _BODY)
    line_h = int(item_scale * 40)
    top = int(height * 0.20)
    panel_w = int(width * 0.62)

    _panel(canvas, (24, top), (24 + panel_w, top + line_h * (len(items) + 2)))
    _text(canvas, title, (48, top + line_h // 2), _WHITE, title_scale)

    for index, (text, ok) in enumerate(items):
        y = top + line_h * (index + 2) - line_h // 3
        if ok is True:
            mark, color = "OK", SEVERITY_BGR["info"]
        elif ok is False:
            mark, color = "X", SEVERITY_BGR["error"]
        else:
            mark, color = "-", _DIM
        _text(canvas, mark, (48, y), color, item_scale, anchor="bl")
        _text(
            canvas,
            text,
            (48 + int(item_scale * 60), y),
            color,
            item_scale,
            anchor="bl",
            max_width=panel_w - int(item_scale * 84),
        )
    return canvas


def draw_confidence_bar(
    frame: np.ndarray, label: str, value: float, thresholds: tuple[float, float] = (0.70, 0.85),
    *,
    copy: bool = True,
) -> np.ndarray:
    """A labelled 0-1 bar, coloured by two thresholds.

    A bar rather than a percentage because the user's question is not "what is
    the confidence" but "is it enough yet", and they are asking it repeatedly
    while shifting a camera with both hands. The threshold marks are drawn on
    the bar so the answer is positional, not numeric.
    """
    canvas = _canvas(frame, copy)
    height, width = canvas.shape[:2]
    value = max(0.0, min(1.0, float(value)))
    low, high = thresholds

    severity = "info" if value >= high else "warning" if value >= low else "error"
    color = SEVERITY_BGR[severity]

    bar_x0, bar_x1 = 48, int(width * 0.55)
    bar_y = int(height * 0.24)
    bar_h = max(18, int(height * 0.035))

    scale = _scale_for(canvas, _BODY)
    _panel(canvas, (24, bar_y - int(scale * 44)), (bar_x1 + 220, bar_y + bar_h + 20))
    _text(canvas, label, (bar_x0, bar_y - int(scale * 38)), _WHITE, scale)

    cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), _DIM, 1, cv2.LINE_AA)
    filled = bar_x0 + int((bar_x1 - bar_x0) * value)
    if filled > bar_x0:
        cv2.rectangle(canvas, (bar_x0, bar_y), (filled, bar_y + bar_h), color, -1)
    for threshold in (low, high):
        tick = bar_x0 + int((bar_x1 - bar_x0) * threshold)
        cv2.line(canvas, (tick, bar_y - 4), (tick, bar_y + bar_h + 4), _WHITE, 1, cv2.LINE_AA)

    # Left-aligned clear of the bar, not centred just past its end: at high
    # values the fill reaches the label and the digits sit on top of it.
    _text(canvas, f"{value * 100:.0f}%", (bar_x1 + 20, bar_y + bar_h), color, scale, anchor="bl")
    return canvas


def draw_metric_rows(
    frame: np.ndarray, title: str, rows: list[tuple[str, str, Severity]],
    *,
    copy: bool = True,
) -> np.ndarray:
    """A table of named measurements, each coloured by its own verdict.

    Used by the fine-tune, test and completion screens, which all have the same
    shape of thing to say: several independent checks, each of which passes or
    does not, and a user who needs to see which one is the problem rather than
    a single aggregate that says "poor".
    """
    canvas = _canvas(frame, copy)
    height, width = canvas.shape[:2]

    scale = _scale_for(canvas, _BODY)
    line_h = int(scale * 42)
    top = int(height * 0.20)
    panel_w = int(width * 0.56)
    value_x = 48 + int(panel_w * 0.68)

    _panel(canvas, (24, top), (24 + panel_w, top + line_h * (len(rows) + 2)))
    _text(canvas, title, (48, top + line_h // 2), _WHITE, scale)

    for index, (label, value, severity) in enumerate(rows):
        y = top + line_h * (index + 2) - line_h // 3
        color = _severity_color(severity)
        _text(canvas, label, (48, y), _DIM, scale, anchor="bl", max_width=value_x - 72)
        _text(canvas, value, (value_x, y), color, scale, anchor="bl")
    return canvas


def draw_countdown(
    frame: np.ndarray, remaining_s: float, total_s: float, message: str,
    *,
    copy: bool = True,
) -> np.ndarray:
    """A large seconds-remaining readout with a progress bar underneath.

    The projector warm-up screen is the one place the wizard asks the user to do
    nothing for a while, and a screen that looks identical for two minutes is
    indistinguishable from a hung one.
    """
    canvas = _canvas(frame, copy)
    height, width = canvas.shape[:2]
    remaining_s = max(0.0, remaining_s)
    fraction = 1.0 if total_s <= 0 else max(0.0, min(1.0, 1.0 - remaining_s / total_s))

    top = int(height * 0.34)
    bottom = int(height * 0.62)
    _panel(canvas, (int(width * 0.16), top), (int(width * 0.84), bottom))

    minutes, seconds = divmod(int(math.ceil(remaining_s)), 60)
    _text(
        canvas,
        f"{minutes}:{seconds:02d}",
        (width // 2, top + int(height * 0.04)),
        _WHITE,
        _scale_for(canvas, _HEADLINE * 1.8),
        anchor="tc",
    )
    _text(
        canvas,
        message,
        (width // 2, top + int(height * 0.155)),
        _DIM,
        _scale_for(canvas, _BODY),
        anchor="tc",
        max_width=int(width * 0.64),
    )

    bar_x0, bar_x1 = int(width * 0.22), int(width * 0.78)
    bar_y = bottom - int(height * 0.045)
    bar_h = max(12, int(height * 0.02))
    cv2.rectangle(canvas, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), _DIM, 1, cv2.LINE_AA)
    filled = bar_x0 + int((bar_x1 - bar_x0) * fraction)
    if filled > bar_x0:
        cv2.rectangle(canvas, (bar_x0, bar_y), (filled, bar_y + bar_h), SEVERITY_BGR["info"], -1)
    return canvas
