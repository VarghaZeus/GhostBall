"""Overlay composition shared by every game mode.

Phase 7. Each mode decides *what* to say; this decides *when* and *where* it
lands on the felt, so that switching modes changes the rules rather than the
visual grammar. Four modes each assembling their own overlay would drift apart
within a week, and the drift would show up as the aiming line moving between
modes.

What a frame is made of, back to front:

1. **The state layer.** One of: the predicted trajectory (aiming), nothing at
   all (balls moving), or nothing (idle). See :meth:`ModeRenderer.compose`.
2. **The mode layer.** Whatever the mode wants on top -- a highlighted target
   ball, a challenge's ball placements, a drill's ideal line. Passed in as a
   ``decorate`` callback rather than returned by the mode, so the mode never
   has to know about canvas lifetimes.
3. **The UI layer.** Scores, timer, combo badge, feedback text, and the effect
   system's trails and celebrations. All rail-anchored by
   :func:`projection.renderer.render_game_ui`, which is what keeps it off the
   cloth.

Nothing is drawn while the balls are moving
-------------------------------------------
The ``shot_in_progress`` branch deliberately draws no prediction. A stale aiming
line under a rolling ball reads as a bug, and the spec's "zero interference with
natural play" rules out painting over live play. What does appear is the trail
behind each ball, which is the effect system's job and is drawn by the UI layer
anyway -- so the branch is empty rather than absent.

One observe per frame
---------------------
:meth:`projection.effects.EffectSystem.observe` extends every ball's trail by
one sample, so calling it twice in a frame doubles the trail's sample rate and
makes a still ball look like it is vibrating. It is called exactly once here,
and ``render_game_ui`` is given the effect system so that ``render`` is also
called exactly once. That is why this module does not use
:func:`projection.renderer.render_ball_trails`, which does both in one call and
would double up against the UI layer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from app.config import Settings, get_settings
from app.models import GameSession, GameState, SessionState, ShotPrediction, Vec2
from projection import draw
from projection.effects import EffectContext, EffectSystem
from projection.mapper import ProjectionMapper
from projection.renderer import TrajectorySmoother, render_game_ui, render_trajectory_overlay
from projection.themes import RGB, Theme, resolve_theme

logger = logging.getLogger(__name__)

__all__ = ["ModeRenderer", "Decorator", "highlight_ball", "draw_placement_marker"]

#: A mode's extra drawing, called with the canvas and the frame's effect
#: context. Returning anything is ignored -- draw in place.
Decorator = Callable[[np.ndarray, EffectContext], None]


class ModeRenderer:
    """Per-mode overlay scaffolding: one canvas, one smoother, one effect system.

    Held by each :class:`~modes.mode_manager.GameMode` instance, so loading a
    new mode drops the previous mode's in-flight celebrations with it -- a score
    popup from the last game floating over a fresh rack is worse than no popup.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.effects = EffectSystem(self.settings)
        #: Smooths the aiming line across frames. Stateful, so it lives here
        #: rather than being rebuilt per frame -- a fresh smoother every frame
        #: would smooth nothing.
        self.smoother = TrajectorySmoother(self.settings)
        self._canvas: np.ndarray | None = None

    def reset(self) -> None:
        """Drop effects and smoothing history. Called on mode entry and reset."""
        self.effects.clear()
        self.smoother.forget()

    def compose(
        self,
        game_state: GameState,
        prediction: ShotPrediction | None,
        session: GameSession,
        mapper: ProjectionMapper | None,
        *,
        feedback: str = "",
        seconds_remaining: float | None = None,
        leaderboard: bool = False,
        show_prediction: bool = True,
        decorate: Decorator | None = None,
        now: float | None = None,
    ) -> np.ndarray | None:
        """Build one frame's overlay, or ``None`` if there is nothing to draw on.

        Args:
            game_state: This frame's observations.
            prediction: The current shot prediction, or ``None``.
            session: Scores, turn and combo state, drawn by the UI layer.
            mapper: Table -> projector transform. ``None`` means the loop has
                not wired one up yet, and the answer is ``None`` rather than an
                exception -- a mode being asked to draw before calibration is
                normal at startup and in unit tests.
            feedback: One short line along the bottom rail.
            seconds_remaining: Countdown for modes that have a clock.
            leaderboard: Sort the scoreboard by score and number it, rather than
                listing players in turn order.
            show_prediction: Draw the aiming line. Modes turn this off when the
                player is not supposed to be aiming yet -- placing balls for a
                trick shot, for instance.
            decorate: Mode-specific drawing, between the trajectory and the UI.
            now: Injectable clock, so effect animation can be tested without
                sleeping.

        Returns:
            An ``HxWx4`` RGBA overlay at projector resolution, reused between
            frames.
        """
        if mapper is None:
            return None

        now = time.perf_counter() if now is None else now
        theme = resolve_theme(self.settings)
        self._canvas = draw.ensure_canvas(self._canvas, self.settings)
        canvas = self._canvas

        # Exactly once per frame, whatever the state: a ball can be pocketed on
        # a shot the state machine has already settled, and the celebration
        # should still fire.
        self.effects.observe(game_state, now)

        if (
            show_prediction
            and session.state is SessionState.AIMING
            and prediction is not None
            and not prediction.is_empty
        ):
            render_trajectory_overlay(
                prediction,
                game_state,
                mapper,
                self.settings,
                canvas=canvas,
                theme=theme,
                smoother=self.smoother,
                clear=False,
                now=now,
            )
        elif session.state is not SessionState.AIMING:
            # Nothing to draw over live play; see the module docstring. The
            # smoother is reset so the next aim starts from the new position
            # instead of easing across the table from the last one.
            self.smoother.forget()

        if decorate is not None:
            decorate(canvas, EffectContext.build(mapper, self.settings, theme))

        render_game_ui(
            session,
            mapper,
            self.settings,
            canvas=canvas,
            theme=theme,
            effects=self.effects,
            seconds_remaining=seconds_remaining,
            feedback_text=feedback,
            leaderboard=leaderboard,
            clear=False,
            now=now,
        )
        return canvas


# ---------------------------------------------------------------------------
# Decorations modes share
# ---------------------------------------------------------------------------


def highlight_ball(
    canvas: np.ndarray,
    ctx: EffectContext,
    position: Vec2,
    color: RGB | None = None,
    *,
    label: str = "",
    now: float | None = None,
) -> None:
    """Ring a ball to mark it as the one to hit, pulsing so it reads as live.

    Pulsing rather than static, and the reason is the felt: a static ring around
    a ball competes with the ball's own outline under projector light, and
    players look straight past it. A ring that breathes is unmistakably an
    overlay.
    """
    import math

    now = time.perf_counter() if now is None else now
    theme = ctx.theme
    color = color or theme.accent
    center = ctx.to_px(position)
    pulse = 0.5 + 0.5 * math.sin(now * math.pi * 2.2)
    base = ctx.inches(1.6)

    draw.draw_ring(canvas, center, base, color, thickness=3, alpha=250, glow=theme.glow)
    draw.draw_ring(
        canvas,
        center,
        base * (1.35 + 0.25 * pulse),
        color,
        thickness=2,
        alpha=int(70 + 130 * pulse),
    )
    if label:
        draw.draw_text(
            canvas,
            label,
            Vec2(center.x, center.y - base * 2.0),
            color,
            scale=max(0.6, ctx.inches(1.1) / 22.0),
            thickness=2,
            anchor="bc",
        )


def draw_placement_marker(
    canvas: np.ndarray,
    ctx: EffectContext,
    position: Vec2,
    *,
    satisfied: bool,
    label: str = "",
) -> None:
    """Mark where a ball needs to be placed, and whether it is there yet.

    Trick shots need the balls in specific spots and the system cannot move
    them, so it projects the layout onto the cloth and waits. The satisfied
    state matters as much as the position: without it the user has no way to
    know whether "close enough" has been reached, and they will either fuss for
    a minute or shoot from the wrong place.
    """
    theme = ctx.theme
    center = ctx.to_px(position)
    radius = ctx.inches(1.125)  # a ball
    color = theme.cue_path if satisfied else theme.impact

    draw.draw_ring(canvas, center, radius, color, thickness=3, alpha=250)
    if satisfied:
        draw.draw_circle(canvas, center, radius * 0.35, color, alpha=220, filled=True)
    else:
        # A cross reads as "put it here" where an empty ring reads as "something
        # is already here", which is exactly the wrong message on an empty spot.
        draw.draw_cross(canvas, center, radius * 0.7, color, thickness=2, alpha=230)
    if label:
        draw.draw_text(
            canvas,
            label,
            Vec2(center.x, center.y + radius * 1.8),
            color,
            scale=max(0.5, ctx.inches(0.9) / 22.0),
            thickness=2,
            anchor="tc",
        )


def theme_for(settings: Settings) -> Theme:
    """The active palette. A one-liner, so modes need not import themes."""
    return resolve_theme(settings)
