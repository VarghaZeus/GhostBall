"""Overlay rendering.

Phase 4.2. Composes what physics predicted and what the game knows into an RGBA
canvas at projector resolution. The pixel primitives live in
:mod:`projection.draw`, the palette in :mod:`projection.themes`, and anything
time-varying in :mod:`projection.effects`; this module is the layout and the
draw order.

Two things about rendering *for a projector onto green felt* that differ from
normal screen rendering, and that the implementations respect:

**Black is transparent.** The projector cannot subtract light. An RGB of
``(0,0,0)`` leaves the felt as it is; there is no way to darken a region. So
overlays are built from bright marks on black, never from dark marks on a light
fill -- a "dimmed" panel background is not physically possible. The one use of
black is deliberate erasure: writing zeros over the overlay's own lines, which
is how text stays readable where it crosses a trajectory.

**Saturated primaries fight the felt.** Green overlay marks on green felt have
poor contrast whatever the brightness. Cyan, white, magenta and yellow read far
better. The default palette in config reflects this; the green cue path is
deliberately shifted towards mint.

The canvas is ``HxWx4`` uint8 RGBA at projector resolution. Alpha is carried
through so effects can fade independently of the global overlay opacity, and is
flattened by the display layer at the last moment.

Where UI goes
-------------
Nothing informational is drawn inside the cushions. Balls, hands and cues occlude
that region constantly, and the spec's "zero interference with natural play" is a
hard constraint rather than a preference. Text and scores are anchored at table
coordinates *outside* the playing surface -- negative ``y`` is beyond the top
rail -- so they land on the rail or the floor. :func:`rail_anchor` is the one
place that geometry is decided.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from app.config import BALL_RADIUS_IN, Settings, get_settings
from app.models import (
    AlignmentError,
    CalibrationState,
    GameSession,
    GameState,
    ImpactEvent,
    ShotPrediction,
    Vec2,
)
from projection import draw
from projection.effects import EffectContext, EffectSystem, draw_combo_badge, draw_timer_countdown
from projection.mapper import ProjectionMapper
from projection.themes import RGB, Theme, ball_display_color, resolve_theme

logger = logging.getLogger(__name__)

__all__ = [
    "blank_overlay",
    "blend_overlay",
    "render_trajectory_overlay",
    "render_training_overlay",
    "render_calibration_overlay",
    "render_game_ui",
    "smooth_path",
    "TrajectorySmoother",
    "rail_anchor",
    "calibration_target_points",
]

#: Corner targets are inset from the frame edge by this fraction of each axis.
#: Projectors overscan, so a target at pixel (0, 0) may be physically off the
#: table or lost to the soft edge of the lens; the inset also leaves room to
#: print the corner's name under it.
CALIBRATION_TARGET_INSET = 0.08

#: Inches beyond the cushion at which rail-anchored UI sits. Roughly the width
#: of a rail on a 7 ft table, so the text lands on wood rather than on cloth.
RAIL_MARGIN_IN = 5.0

#: Px kept clear of the frame edge when a rail anchor has to be clamped inward.
#: Projectors overscan and lens edges are soft; text right at the boundary is
#: the first thing to be lost.
EDGE_PADDING_PX = 28


# ---------------------------------------------------------------------------
# Canvas helpers
# ---------------------------------------------------------------------------


def blank_overlay(settings: Settings | None = None) -> np.ndarray:
    """A fully transparent canvas at projector resolution.

    For one-off callers. Per-frame rendering should keep one buffer and zero it
    in place instead -- see :func:`projection.draw.reset_canvas` for the measured
    2.7 ms/frame that saves, which is not where you would expect it to come
    from. Every ``render_*`` function here takes a ``canvas`` argument for
    exactly that reason.
    """
    return draw.new_canvas(settings)


def blend_overlay(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """Composite an RGBA overlay over a BGR base image.

    Used for the *camera preview* in the calibration UI -- showing the user what
    will be projected, superimposed on what the camera sees. Not used for actual
    projector output, which needs no base image.

    The overlay's own alpha channel is multiplied by ``alpha``, so a fading
    effect fades correctly rather than snapping to the global opacity.

    Two things worth knowing about this implementation:

    **It converts RGBA to BGR.** The overlay is RGBA and a camera frame is BGR,
    so the channels have to be swapped before they are mixed. Skipping the swap
    is invisible on greys and wrong on everything else -- the mint cue path would
    preview as a different green -- which matters here specifically, because the
    whole purpose of this preview is judging whether what is projected matches
    what was intended.

    **It is integer OpenCV ops, not float32 NumPy.** Same reasoning as
    ``display.send_frame``: the obvious float expression measures ~130 ms at
    1080p, which would cap the calibration preview at 7 FPS. The wizard's job is
    live feedback while the user physically nudges the projector, and feedback a
    seventh of a second behind the nudge is not usable. This path is ~10x
    faster and agrees with the float one to within a count of rounding.
    """
    if base.shape[:2] != overlay.shape[:2]:
        raise ValueError(
            f"size mismatch: base {base.shape[:2]} vs overlay {overlay.shape[:2]}"
        )

    import cv2

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR)
    # Per-pixel alpha, pre-scaled by the global opacity, broadcast to 3 channels
    # so the multiplies are plain elementwise uint8 ops.
    weight = cv2.cvtColor(
        cv2.convertScaleAbs(overlay[:, :, 3], alpha=min(1.0, max(0.0, alpha))),
        cv2.COLOR_GRAY2BGR,
    )
    foreground = cv2.multiply(overlay_bgr, weight, scale=1.0 / 255.0)
    background = cv2.multiply(base, cv2.bitwise_not(weight), scale=1.0 / 255.0)
    return cv2.add(foreground, background)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def rail_anchor(
    mapper: ProjectionMapper,
    where: str,
    settings: Settings | None = None,
    margin_in: float = RAIL_MARGIN_IN,
) -> Vec2:
    """A UI anchor point in projector px, positioned outside the cushions.

    The anchor is specified in *table* coordinates just off the playing surface
    and then mapped, which is what makes it land on the physical rail under any
    calibration -- including a keystoned one, where a fixed pixel offset from the
    frame edge would drift off the table at one end.

    ``where`` is one of ``top_left``, ``top_center``, ``top_right``,
    ``bottom_left``, ``bottom_center``, ``bottom_right``.

    A mapped anchor can land off-frame: an identity calibration stretches the
    table to fill the output exactly, leaving nothing outside it. Rather than
    drop the UI, it is clamped just inside the frame -- so on an uncalibrated
    setup the score sits on the cloth near the rail instead of vanishing, which
    is worse-looking but still usable.
    """
    settings = settings or get_settings()
    length, width = settings.table.length_in, settings.table.width_in
    xs = {"left": 0.0, "center": length / 2.0, "right": length}
    if where.startswith("top_"):
        y = -margin_in
    elif where.startswith("bottom_"):
        y = width + margin_in
    else:
        raise ValueError(f"unknown anchor {where!r}")
    key = where.split("_", 1)[1]
    if key not in xs:
        raise ValueError(f"unknown anchor {where!r}")

    # Nudge the left and right anchors inboard so a right-aligned string is not
    # hanging off the corner of the table.
    x = xs[key]
    if key == "left":
        x += margin_in
    elif key == "right":
        x -= margin_in

    point = mapper.table_to_projector(Vec2(x, y))
    max_x = settings.projector.width - EDGE_PADDING_PX
    max_y = settings.projector.height - EDGE_PADDING_PX
    return Vec2(
        min(max(point.x, EDGE_PADDING_PX), max_x),
        min(max(point.y, EDGE_PADDING_PX), max_y),
    )


def _projector_angle(mapper: ProjectionMapper, at: Vec2, table_deg: float) -> float:
    """A table-space heading expressed as a projector-space angle at a point.

    Measured through the transform rather than negated, because under a
    homography the angle between two directions is not preserved -- a 30 degree
    cut at the far end of a keystoned table is not 30 degrees on screen. Cut
    angles are the one thing in the overlay a player will actually judge by eye
    against the real balls, so this one is worth two extra matrix applications.
    """
    theta = math.radians(table_deg)
    tip = Vec2(at.x + math.cos(theta), at.y + math.sin(theta))
    origin_px = mapper.table_to_projector(at)
    tip_px = mapper.table_to_projector(tip)
    return math.degrees(math.atan2(tip_px.y - origin_px.y, tip_px.x - origin_px.x))


def _alpha(base: int, theme_factor: float = 1.0) -> int:
    """Clamp an alpha, scaled by a theme factor."""
    return min(255, max(0, int(round(base * theme_factor))))


def _text_scale(ctx: EffectContext, inches_tall: float) -> float:
    """Font scale that renders roughly ``inches_tall`` on the felt.

    Sized in inches for the same reason everything else is: text that is
    readable from six feet away on a 7 ft table has to be around an inch and a
    half tall physically, and that is a different number of pixels on every
    projector and at every throw distance. Hershey Simplex at scale 1.0 is about
    22 px cap height, which is where the divisor comes from.
    """
    return max(0.4, ctx.inches(inches_tall) / 22.0)


# ---------------------------------------------------------------------------
# Trajectory smoothing
# ---------------------------------------------------------------------------


def smooth_path(
    previous: list[tuple[float, float]] | None,
    current: list[tuple[float, float]],
    smoothing_pct: int,
) -> list[tuple[float, float]]:
    """Exponentially smooth a trajectory against the previous frame's.

    Per-frame detection noise on the cue angle translates into a visibly
    twitching aiming line, and this is the fix. It costs latency in exact
    proportion to the smoothing, so the control is exposed to the user.

    Handles the differing-length case by smoothing the common prefix and taking
    the new tail verbatim -- a shot prediction changes length whenever the
    collision set changes, and refusing to smooth in that case would make the
    line jump exactly when it matters most.

    Args:
        previous: Last frame's smoothed path, or ``None`` on the first frame.
        current: This frame's raw path.
        smoothing_pct: 0-100. 0 returns ``current`` untouched; 100 would freeze
            the line, so it is capped just below -- an unresponsive aiming line
            is a bug however the user got there.

    Returns:
        The smoothed path. Points are :class:`~app.models.Vec2`, which is a
        ``tuple[float, float]``, so the result can be fed straight back in as
        ``previous`` next frame.
    """
    if not current:
        return []
    weight = min(0.95, max(0.0, smoothing_pct / 100.0))
    if weight <= 0.0 or not previous:
        return [Vec2(float(p[0]), float(p[1])) for p in current]

    out: list[Vec2] = []
    shared = min(len(previous), len(current))
    for i in range(shared):
        px, py = previous[i][0], previous[i][1]
        cx, cy = current[i][0], current[i][1]
        out.append(Vec2(px * weight + cx * (1.0 - weight), py * weight + cy * (1.0 - weight)))
    # The tail beyond the shared prefix has nothing to smooth against. Taking it
    # verbatim means a newly appeared rebound segment snaps in at full length,
    # which is correct: it is new information, not noise.
    out.extend(Vec2(float(p[0]), float(p[1])) for p in current[shared:])
    return out


class TrajectorySmoother:
    """Per-path smoothing state across frames.

    :func:`smooth_path` is pure and needs last frame's path handed to it; this
    keeps them, keyed by path (``"cue"``, or a ball id). Separate from the
    render functions so that smoothing can be shared between the projector
    overlay and the calibration preview -- both draw the same prediction and
    they must not disagree about where it is.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._paths: dict[str, list[Vec2]] = {}

    def apply(self, key: str, path: list[Vec2]) -> list[Vec2]:
        """Smooth one path and remember the result."""
        smoothed = smooth_path(
            self._paths.get(key), path, self.settings.render.trajectory_smoothing_pct
        )
        self._paths[key] = smoothed
        return smoothed

    def forget(self, key: str | None = None) -> None:
        """Drop remembered paths.

        Called when the cue leaves the frame or a shot starts. Without it, the
        next aim would be smoothed against a path from a different shot and the
        line would sweep across the table over half a second.
        """
        if key is None:
            self._paths.clear()
        else:
            self._paths.pop(key, None)


# ---------------------------------------------------------------------------
# Trajectory overlay
# ---------------------------------------------------------------------------


def render_trajectory_overlay(
    prediction: ShotPrediction,
    game_state: GameState,
    mapper: ProjectionMapper,
    settings: Settings | None = None,
    canvas: np.ndarray | None = None,
    *,
    theme: Theme | None = None,
    smoother: TrajectorySmoother | None = None,
    clear: bool = True,
    now: float | None = None,
) -> np.ndarray:
    """Draw the shot prediction.

    ``clear=False`` stacks this onto whatever is already in ``canvas``; see
    :func:`projection.draw.ensure_canvas`.

    Draw order, back to front, so that later marks are not buried:

    1. Pocket highlights for pockets a predicted ball is heading into.
    2. Object-ball post-impact paths, in the dimmer secondary colour.
    3. The cue-ball path, brightest, as the primary aiming line.
    4. Impact markers with cut-angle indicators at each collision.
    5. Ghost-ball outlines at predicted resting positions.

    Every path arrives in table inches and goes through
    ``mapper.table_to_projector_batch`` -- one batched call per polyline, not one
    per point.

    Args:
        prediction: What physics predicted. An empty prediction is valid and
            yields a blank canvas rather than raising.
        game_state: Current observed state, for ball colours and positions.
        mapper: Table -> projector transform.
        settings: Config. Defaults to the global settings.
        canvas: Reusable buffer, zeroed in place. Strongly preferred over
            allocating -- see :func:`blank_overlay`.
        theme: Palette override. Defaults to the configured theme.
        smoother: Cross-frame smoothing state. Without one the line is drawn raw
            and will jitter with detection noise; the caller owns it because it
            has to persist between frames.
        now: Clock reading for the dash animation. Defaults to
            ``time.perf_counter()``.
    """
    settings = settings or get_settings()
    theme = theme or resolve_theme(settings)
    now = time.perf_counter() if now is None else now
    canvas = draw.ensure_canvas(canvas, settings, clear=clear)

    if prediction is None or prediction.is_empty:
        return canvas

    ctx = EffectContext.build(mapper, settings, theme)
    render = settings.render
    thickness = render.line_thickness_px
    balls_by_id = {ball.id: ball for ball in game_state.balls}

    # 1. Pocket highlights -------------------------------------------------
    _draw_pocket_highlights(canvas, prediction, game_state, ctx, now)

    # 2. Object-ball paths -------------------------------------------------
    # Held well below the cue path's 248. Object paths are the longest lines on
    # the table -- a firm shot moves four balls several feet each -- so at equal
    # alpha they carry more total ink than the aiming line and invert the visual
    # hierarchy the whole overlay depends on.
    secondary_alpha = _alpha(190, theme.secondary_alpha)
    for ball_id, path in prediction.ball_paths.items():
        if len(path) < 2:
            continue
        if smoother is not None:
            path = smoother.apply(ball_id, list(path))
        ball = balls_by_id.get(ball_id)
        color = ball_display_color(ball, theme) if ball is not None else theme.object_path
        draw.draw_polyline(
            canvas,
            mapper.table_to_projector_batch(list(path)),
            color,
            thickness=max(1, thickness - 1),
            alpha=secondary_alpha,
            glow=theme.glow,
        )

    # 3. Cue path ----------------------------------------------------------
    cue_path = list(prediction.trajectory_path)
    if smoother is not None:
        cue_path = smoother.apply("cue", cue_path)
    cue_px = mapper.table_to_projector_batch(cue_path)
    # Negative rate so the dashes crawl *away* from the cue ball: increasing the
    # phase shifts the pattern backwards along the path, and a line whose dashes
    # travel toward the player reads as the ball coming back at them.
    phase = -now * theme.dash_speed_px_s
    draw.draw_dashed_polyline(
        canvas,
        cue_px,
        theme.cue_path,
        thickness=thickness,
        alpha=248,
        dash_px=theme.dash_px,
        gap_px=theme.gap_px,
        phase_px=phase,
        glow=theme.glow,
    )

    # 4. Impact markers ----------------------------------------------------
    for impact in prediction.impact_points:
        _draw_impact(canvas, impact, ctx, show_angle=render.show_impact_angles)

    # 5. Ghost balls -------------------------------------------------------
    if render.show_ghost_balls:
        for ball_id, position in prediction.final_positions.items():
            if ball_id in prediction.pocketed_ball_ids:
                continue  # a potted ball has no resting position to show
            ball = balls_by_id.get(ball_id)
            color = ball_display_color(ball, theme) if ball is not None else theme.ghost_ball
            _draw_ghost_ball(canvas, position, ctx, color)

    return canvas


def _draw_pocket_highlights(
    canvas: np.ndarray,
    prediction: ShotPrediction,
    game_state: GameState,
    ctx: EffectContext,
    now: float,
) -> None:
    """Ring the pockets a predicted ball is going into, pulsing gently.

    The pocket is found from where the ball's path *ends* rather than from its
    resting position, because a potted ball has no resting position. Falls back
    to the table's nominal pocket geometry when the detector has not located the
    real openings, so the highlight works before pocket detection has run.
    """
    if not prediction.pocketed_ball_ids:
        return
    pockets = _pocket_positions(game_state, ctx.settings)
    if not pockets:
        return

    # 1.5 Hz pulse. Slow: this is a persistent indicator, not an event, and a
    # fast pulse next to a crawling dash pattern is visual noise.
    pulse = 0.5 + 0.5 * math.sin(now * math.pi * 3.0)
    radius = ctx.inches(ctx.settings.table.pocket_radius_in * (1.5 + 0.35 * pulse))
    alpha = _alpha(150 + int(70 * pulse))

    for ball_id in prediction.pocketed_ball_ids:
        end = _path_end(prediction, ball_id)
        if end is None:
            continue
        pocket = min(pockets, key=end.distance_to)
        draw.draw_ring(
            canvas,
            ctx.to_px(pocket),
            radius,
            ctx.theme.pocket_highlight,
            thickness=4,
            alpha=alpha,
            glow=True,
        )


def _pocket_positions(game_state: GameState, settings: Settings) -> list[Vec2]:
    """Pocket centres in table inches, detected if available, nominal if not."""
    detected = [p.table_pos for p in game_state.pockets if p.table_pos is not None]
    if detected:
        return detected
    from physics.models import TableGeometry

    return TableGeometry.from_settings(settings).pocket_centers()


def _path_end(prediction: ShotPrediction, ball_id: str) -> Vec2 | None:
    """Where a ball's predicted path finishes, cue ball included."""
    path = prediction.ball_paths.get(ball_id)
    if path:
        return path[-1]
    if prediction.trajectory_path and ball_id in ("cue", "cue_ball"):
        return prediction.trajectory_path[-1]
    return prediction.final_positions.get(ball_id)


def _draw_impact(
    canvas: np.ndarray,
    impact: ImpactEvent,
    ctx: EffectContext,
    show_angle: bool = True,
) -> None:
    """Mark one collision with a circle, an outgoing arrow and its cut angle.

    Cushion contacts get the circle and arrow but no text. A rebound angle is
    not a decision the player makes -- they can see the rail -- whereas the cut
    angle on a ball contact is exactly the number being judged, so labelling
    both would bury the useful one in clutter.
    """
    center = ctx.to_px(impact.position)
    theme = ctx.theme
    radius = ctx.inches(BALL_RADIUS_IN * 0.8)

    draw.draw_ring(canvas, center, radius, theme.impact, thickness=2, alpha=230, glow=theme.glow)
    draw.draw_arrow(
        canvas,
        center,
        _projector_angle(ctx.mapper, impact.position, impact.outgoing_angle_deg),
        ctx.inches(BALL_RADIUS_IN * 2.6),
        theme.impact,
        thickness=2,
        alpha=220,
    )

    if not show_angle or impact.is_cushion or theme.minimal_ui:
        return

    # Cut angle: how far the ball is deflected, 0 for a straight-on hit. Reported
    # as the deflection rather than as the raw outgoing bearing, because that is
    # the quantity a player thinks in ("a thin 70 degree cut").
    deflection = abs(
        (impact.outgoing_angle_deg - impact.incoming_angle_deg + 180.0) % 360.0 - 180.0
    )
    label_at = Vec2(center.x, center.y - radius - ctx.inches(0.6))
    draw.draw_text(
        canvas,
        f"{deflection:.0f}",
        label_at,
        theme.impact,
        scale=_text_scale(ctx, 0.9),
        thickness=2,
        alpha=235,
        anchor="bc",
    )


def _draw_ghost_ball(canvas: np.ndarray, position: Vec2, ctx: EffectContext, color: RGB) -> None:
    """Outline where a ball is predicted to come to rest.

    An outline at true ball radius, plus a centre dot. True radius matters: the
    player compares the ghost against real balls, and a ghost drawn a few pixels
    large reads as a prediction that the ball will end up somewhere it will not
    fit.
    """
    center = ctx.to_px(position)
    radius = ctx.inches(BALL_RADIUS_IN)
    draw.draw_circle(canvas, center, radius, color, thickness=2, alpha=170)
    draw.draw_circle(canvas, center, max(1.5, radius * 0.12), color, alpha=200, filled=True)


# ---------------------------------------------------------------------------
# Training overlay
# ---------------------------------------------------------------------------


def render_training_overlay(
    game_state: GameState,
    target_prediction: ShotPrediction | None,
    user_prediction: ShotPrediction | None,
    mapper: ProjectionMapper,
    feedback_text: str = "",
    settings: Settings | None = None,
    canvas: np.ndarray | None = None,
    *,
    theme: Theme | None = None,
    clear: bool = True,
    now: float | None = None,
) -> np.ndarray:
    """Draw training-mode guidance: the target shot, the user's aim, and feedback.

    ``clear=False`` stacks this onto whatever is already in ``canvas``; see
    :func:`projection.draw.ensure_canvas`.

    The two predictions must be visually distinguishable at a glance from across
    a table -- differing by colour alone is not enough under projector light.
    The target is dashed and marked with a ghost ball at its contact point; the
    user's live aim is solid. Style, not just hue, so the distinction survives a
    washed-out projection and a colour-blind player.

    Text is a real constraint here: it is being projected onto cloth at an angle
    and read from several feet away. It is kept short, large, and anchored to the
    rail rather than centred where the balls are.
    """
    settings = settings or get_settings()
    theme = theme or resolve_theme(settings)
    now = time.perf_counter() if now is None else now
    canvas = draw.ensure_canvas(canvas, settings, clear=clear)
    ctx = EffectContext.build(mapper, settings, theme)
    thickness = settings.render.line_thickness_px

    # The target shot, dashed and static. A static pattern next to the animated
    # live line is another axis of difference -- movement means "this is you".
    if target_prediction is not None and not target_prediction.is_empty:
        target_px = mapper.table_to_projector_batch(list(target_prediction.trajectory_path))
        draw.draw_dashed_polyline(
            canvas,
            target_px,
            theme.impact,
            thickness=max(2, thickness - 1),
            alpha=_alpha(210, theme.secondary_alpha),
            dash_px=14.0,
            gap_px=18.0,
            phase_px=0.0,
        )
        for impact in target_prediction.impact_points[:1]:
            # Only the first contact: the drill is about hitting the right ball
            # at the right place, and drawing the whole chain of a bank shot
            # would obscure the one contact being taught.
            _draw_ghost_ball(canvas, impact.position, ctx, theme.impact)

    # The player's current aim, solid and crawling.
    if user_prediction is not None and not user_prediction.is_empty:
        draw.draw_polyline(
            canvas,
            mapper.table_to_projector_batch(list(user_prediction.trajectory_path)),
            theme.cue_path,
            thickness=thickness,
            alpha=248,
            glow=theme.glow,
        )

    if feedback_text:
        anchor = rail_anchor(mapper, "top_center", settings)
        draw.draw_text(
            canvas,
            feedback_text,
            anchor,
            theme.text,
            scale=_text_scale(ctx, 1.6),
            thickness=3,
            anchor="tc",
        )

    return canvas


# ---------------------------------------------------------------------------
# Calibration overlay
# ---------------------------------------------------------------------------


def render_calibration_overlay(
    state: CalibrationState,
    mapper: ProjectionMapper | None,
    alignment: AlignmentError | None = None,
    settings: Settings | None = None,
    *,
    theme: Theme | None = None,
    canvas: np.ndarray | None = None,
    now: float | None = None,
    targets: list[tuple[Vec2, str]] | None = None,
) -> np.ndarray:
    """Draw the calibration wizard's projected content for the current step.

    Note this renders what goes *to the projector* -- corner targets, the grid,
    the test trajectory. The camera-preview side of the wizard is
    :mod:`calibration_ui.overlay_renderer`. Keeping the two apart matters
    because the whole point of the wizard is comparing them.

    During corner mapping the marks must be projected without any calibration
    applied, since establishing the transform is the goal -- so ``mapper`` is
    ``None`` for step 4 and the drawing is in raw projector coordinates.

    Steps, and what each projects:

    ==== ===========================================================
    1-3  Nothing but a centred instruction: these are camera-side
         steps (mount, focus, find the table) and projecting marks
         over the felt during them only gets in the detector's way.
    4    Four corner targets in raw projector space, one highlighted.
    5    A table-space grid, to judge scale and rotation.
    6    Corner targets plus the grid, with the live alignment error.
    7    A full-table pattern for the final confirmation.
    ==== ===========================================================

    ``targets`` overrides where the step-4 marks are drawn, defaulting to
    :func:`calibration_target_points`. The wizard passes its own list because
    its arrow keys walk individual marks across the felt, and a mark drawn
    anywhere other than the pixel the wizard records as its correspondence is a
    calibration wrong by exactly that gap -- with nothing on screen to show it.
    """
    settings = settings or get_settings()
    theme = theme or resolve_theme(settings)
    now = time.perf_counter() if now is None else now
    canvas = draw.ensure_canvas(canvas, settings)

    # Imported here rather than at module scope: patterns imports this module's
    # sibling draw helpers and is only needed on the wizard path, which runs at
    # human speed and not in the 33 ms loop.
    from projection.patterns import TestPattern, render_test_pattern

    width, height = settings.projector.width, settings.projector.height

    if state.step <= 3:
        draw.draw_text(
            canvas,
            _CALIBRATION_INSTRUCTIONS.get(state.step, ""),
            Vec2(width / 2.0, height / 2.0),
            theme.text,
            scale=2.0,
            thickness=3,
            anchor="c",
        )
        return canvas

    if state.step == 4:
        _draw_raw_corner_targets(canvas, state, theme, settings, now, targets)
    elif state.step == 5 and mapper is not None:
        render_test_pattern(TestPattern.GRID, mapper, settings, canvas=canvas, theme=theme)
    elif state.step == 6 and mapper is not None:
        render_test_pattern(TestPattern.GRID, mapper, settings, canvas=canvas, theme=theme)
        render_test_pattern(TestPattern.CORNERS, mapper, settings, canvas=canvas, theme=theme)
    elif mapper is not None:
        render_test_pattern(TestPattern.FULL_TABLE, mapper, settings, canvas=canvas, theme=theme)

    if alignment is not None:
        _draw_alignment_readout(canvas, alignment, theme, settings)

    return canvas


#: One sentence per camera-side step. The five-minute setup target in the spec
#: is not met by a user reading paragraphs off the felt.
_CALIBRATION_INSTRUCTIONS: dict[int, str] = {
    1: "Mount the camera above the table centre",
    2: "Adjust focus until the balls are sharp",
    3: "Checking the table is fully in frame...",
}


def calibration_target_points(settings: Settings | None = None) -> list[tuple[Vec2, str]]:
    """Where the wizard's four corner targets are projected, in projector px.

    Public because the calibration wizard needs the same numbers this module
    draws at: it records each target's projector pixel as one half of a
    correspondence, and if the two ever disagreed the solved transform would be
    wrong by exactly the discrepancy -- with nothing on screen to suggest it.
    Two copies of an inset constant is not a risk worth taking for four points.

    Returned clockwise from top-left, matching
    :data:`calibration_ui.metrics.CORNER_NAMES` and
    :meth:`app.models.TableBoundary.corners`, so the three can be zipped.
    """
    settings = settings or get_settings()
    width, height = settings.projector.width, settings.projector.height
    inset_x = width * CALIBRATION_TARGET_INSET
    inset_y = height * CALIBRATION_TARGET_INSET
    return [
        (Vec2(inset_x, inset_y), "TOP LEFT"),
        (Vec2(width - inset_x, inset_y), "TOP RIGHT"),
        (Vec2(width - inset_x, height - inset_y), "BOTTOM RIGHT"),
        (Vec2(inset_x, height - inset_y), "BOTTOM LEFT"),
    ]


def _draw_raw_corner_targets(
    canvas: np.ndarray,
    state: CalibrationState,
    theme: Theme,
    settings: Settings,
    now: float,
    targets: list[tuple[Vec2, str]] | None = None,
) -> None:
    """Project four corner targets in raw projector coordinates.

    Positions come from :func:`calibration_target_points`; see there for why
    they are inset from the frame edge.

    The active target -- the one the user is placing now -- pulses and gets a
    second ring. The recorded ones become static crosses. Shape carries the
    state as well as colour, per the wizard's own requirement.
    """
    corners = targets if targets is not None else calibration_target_points(settings)
    recorded = len(state.corner_errors)
    pulse = 0.5 + 0.5 * math.sin(now * math.pi * 2.5)

    for index, (point, label) in enumerate(corners):
        is_recorded = index < recorded
        is_active = index == recorded
        color = theme.cue_path if is_recorded else theme.impact
        size = 34.0 + (10.0 * pulse if is_active else 0.0)

        draw.draw_cross(canvas, point, size, color, thickness=3, alpha=250)
        if is_recorded:
            draw.draw_cross(canvas, point, size * 0.7, color, thickness=2, alpha=200, rotate_deg=45)
        else:
            draw.draw_ring(canvas, point, size * 0.8, color, thickness=2, alpha=220)
        if is_active:
            draw.draw_ring(
                canvas, point, size * (1.3 + 0.4 * pulse), color, thickness=2, alpha=int(200 * pulse)
            )
        draw.draw_text(
            canvas,
            label,
            Vec2(point.x, point.y + size + 12.0),
            color,
            scale=0.9,
            thickness=2,
            anchor="tc",
        )


def _draw_alignment_readout(
    canvas: np.ndarray, alignment: AlignmentError, theme: Theme, settings: Settings
) -> None:
    """Project the alignment verdict near the bottom of the frame.

    ``alignment.message`` is written for a human and is rendered verbatim.
    Severity picks the colour, so the user gets the verdict before reading the
    words -- and the RMSE is appended small, because the number is for the
    person tuning, not the person playing.
    """
    color = {
        "info": theme.cue_path,
        "warning": theme.impact,
        "error": theme.alert,
    }.get(alignment.severity, theme.text)
    center_x = settings.projector.width / 2.0
    base_y = settings.projector.height * 0.86

    if alignment.message:
        draw.draw_text(
            canvas, alignment.message, Vec2(center_x, base_y), color, scale=1.6, thickness=3, anchor="tc"
        )
    draw.draw_text(
        canvas,
        f"RMSE {alignment.total_rmse:.1f} px",
        Vec2(center_x, base_y + 56.0),
        color,
        scale=0.9,
        thickness=2,
        alpha=200,
        anchor="tc",
    )


# ---------------------------------------------------------------------------
# Game UI
# ---------------------------------------------------------------------------


def render_game_ui(
    session: GameSession,
    mapper: ProjectionMapper,
    settings: Settings | None = None,
    canvas: np.ndarray | None = None,
    *,
    theme: Theme | None = None,
    effects: EffectSystem | None = None,
    seconds_remaining: float | None = None,
    feedback_text: str = "",
    leaderboard: bool = False,
    clear: bool = True,
    now: float | None = None,
) -> np.ndarray:
    """Draw the score, turn indicator and mode UI.

    ``clear=False`` stacks this onto whatever is already in ``canvas``, which is
    what every game mode does -- the scoreboard goes *over* the aiming line. See
    :func:`projection.draw.ensure_canvas`.

    Placement is the design problem, not the drawing. Anything inside the
    cushions gets occluded by balls and hands and interferes with play, so
    scoreboards belong projected onto the rails or the floor beyond the table.
    That the spec's stated principle is "zero interference with natural play"
    makes this a hard constraint, not a preference -- so every element here is
    positioned via :func:`rail_anchor`.

    Layout, all rail-anchored:

    - **Top left**: players and scores, current player brighter and marked.
    - **Top right**: countdown, when the mode has a clock.
    - **Top centre**: combo badge.
    - **Bottom centre**: feedback text and the mode name.

    Args:
        effects: Effect system whose score popups and bursts should be drawn on
            top of the UI. Passed in rather than created here because effects
            outlive a frame -- see :mod:`projection.effects`.
        seconds_remaining: Countdown to show, or ``None`` for modes without one.
        leaderboard: Rank the scoreboard by score and number it, instead of
            listing players in turn order. King of the Hill wants a ranking --
            the whole mode is about who is winning -- while a two-player game
            wants turn order, where re-sorting the list mid-game would make the
            player positions jump around.
    """
    settings = settings or get_settings()
    theme = theme or resolve_theme(settings)
    now = time.perf_counter() if now is None else now
    canvas = draw.ensure_canvas(canvas, settings, clear=clear)
    ctx = EffectContext.build(mapper, settings, theme)

    _draw_scoreboard(canvas, session, ctx, mapper, settings, ranked=leaderboard)

    if seconds_remaining is not None:
        draw_timer_countdown(
            canvas, seconds_remaining, ctx, rail_anchor(mapper, "top_right", settings), now
        )

    draw_combo_badge(canvas, session.combo_count, ctx, rail_anchor(mapper, "top_center", settings), now)

    if feedback_text:
        draw.draw_text(
            canvas,
            feedback_text,
            rail_anchor(mapper, "bottom_center", settings),
            theme.text,
            scale=_text_scale(ctx, 1.5),
            thickness=3,
            anchor="bc",
        )

    if not theme.minimal_ui:
        anchor = rail_anchor(mapper, "bottom_right", settings)
        draw.draw_text(
            canvas,
            session.mode.value.replace("_", " ").upper(),
            anchor,
            theme.text,
            scale=_text_scale(ctx, 0.8),
            thickness=2,
            alpha=_alpha(200, theme.secondary_alpha),
            anchor="br",
        )

    if effects is not None:
        effects.render(canvas, ctx, now)

    return canvas


def _draw_scoreboard(
    canvas: np.ndarray,
    session: GameSession,
    ctx: EffectContext,
    mapper: ProjectionMapper,
    settings: Settings,
    *,
    ranked: bool = False,
) -> None:
    """List players and scores down from the top-left rail anchor.

    The current player is brighter and carries a leading marker; eliminated
    players are dimmed rather than removed, because a knockout scoreboard that
    silently shrinks loses the information that someone was knocked out.

    ``ranked`` sorts by score and numbers the rows. Off by default: in turn
    order the player knows where to look, whereas a list that re-sorts itself
    the moment somebody scores makes everyone hunt for their own name.
    """
    theme = ctx.theme
    anchor = rail_anchor(mapper, "top_left", settings)
    scale = _text_scale(ctx, 1.3)
    line_height = ctx.inches(1.9)

    order = list(enumerate(session.players))
    if ranked:
        order.sort(key=lambda pair: pair[1].score, reverse=True)

    for row, (index, player) in enumerate(order):
        is_current = index == session.current_player_index % max(1, len(session.players))
        if player.is_eliminated:
            color, alpha = theme.text, _alpha(110)
        elif is_current:
            color, alpha = theme.accent, 255
        else:
            color, alpha = theme.text, _alpha(190, theme.secondary_alpha)
        prefix = "> " if is_current and not player.is_eliminated else "  "
        if ranked:
            prefix = f"{row + 1}. " if not is_current else f"{row + 1}.>"
        draw.draw_text(
            canvas,
            f"{prefix}{player.name}  {player.score}",
            Vec2(anchor.x, anchor.y + row * line_height),
            color,
            scale=scale,
            thickness=3 if is_current else 2,
            alpha=alpha,
            anchor="tl",
        )


def render_ball_trails(
    game_state: GameState,
    effects: EffectSystem,
    mapper: ProjectionMapper,
    settings: Settings | None = None,
    canvas: np.ndarray | None = None,
    *,
    theme: Theme | None = None,
    now: float | None = None,
) -> np.ndarray:
    """Observe this frame's ball motion and draw trails plus live effects.

    The overlay for the ``shot_in_progress`` state, where there is no prediction
    to draw -- the balls are already doing what physics guessed at -- and the
    job is to make the motion legible. One call so the mode layer does not have
    to remember to both feed and draw the effect system, which is the sort of
    two-call contract that gets half-implemented.
    """
    settings = settings or get_settings()
    theme = theme or resolve_theme(settings)
    now = time.perf_counter() if now is None else now
    canvas = draw.ensure_canvas(canvas, settings)

    effects.observe(game_state, now)
    effects.render(canvas, EffectContext.build(mapper, settings, theme), now)
    return canvas
