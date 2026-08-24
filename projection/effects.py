"""Animated effects: trails, collision bursts, pocket celebrations, score popups.

Everything here is a function of *time*, which is what separates it from
:mod:`projection.renderer`. The renderer draws the current prediction, which is
a pure function of this frame's state; an effect outlives the frame that spawned
it and has to be advanced. So effects need somewhere to live between frames, and
that is :class:`EffectSystem`.

Positions are table inches, sizes are inches
--------------------------------------------
Every effect stores its position in table coordinates and its size in inches,
and converts to pixels at draw time. It would be less work to store pixels --
the spawn site usually has them -- and it would be wrong: a burst spawned before
the projector is nudged would then animate in the old alignment, and effects
sized in pixels would be physically bigger at the far end of a keystoned
projection than the near end. Inches are the only space in which "a burst the
size of two balls" is a stable statement.

Time is ``perf_counter``
------------------------
Consistent with ``GameState.timestamp`` and :mod:`utils.performance`, so an
effect can be scheduled against the timestamp of the frame that caused it.
Every entry point takes ``now`` explicitly and defaults it, which is also what
makes these testable without sleeping.

What this module does and does not decide
-----------------------------------------
:meth:`EffectSystem.observe` will spawn trails, collision bursts and pocket
vortexes on its own, from ball motion alone -- so the effects work before the
game modes land, and so they stay correct if a mode forgets to report something.
It never invents a *score*: a points popup only appears if a caller passes a
number, because how many points a pot is worth is a game rule and rules do not
belong in the renderer.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.config import BALL_RADIUS_IN, Settings, get_settings
from app.models import Ball, GameState, Vec2
from physics.models import TableGeometry
from projection import draw
from projection.mapper import ProjectionMapper
from projection.themes import RGB, Theme, ball_display_color, resolve_theme

logger = logging.getLogger(__name__)

__all__ = [
    "EffectContext",
    "Effect",
    "CollisionBurst",
    "PocketVortex",
    "ScorePopup",
    "BallTrail",
    "EffectSystem",
    "draw_combo_badge",
    "draw_timer_countdown",
    "ease_out_cubic",
    "ease_in_cubic",
    "ease_in_out_cubic",
    "ease_out_back",
    "ghost_ball_radius_px",
]

#: Seconds of ball path kept in a trail. The spec's figure, and it holds up: at
#: 30 FPS it is 15 samples, and a firmly struck ball covers roughly 40 inches in
#: that time -- about half the table, which reads as motion without the trail
#: becoming a second trajectory line competing with the prediction.
TRAIL_SECONDS = 0.5

#: Drop a ball's trail once it has been missing this long. Long enough to ride
#: out a few frames of failed detection (which is routine when a hand crosses
#: the table), short enough that a pocketed ball's trail does not linger.
TRAIL_EXPIRY_SECONDS = 0.4

#: Collision burst duration. The spec's 200 ms, rounded up slightly so the
#: sparks get a few frames of fade rather than snapping off.
BURST_SECONDS = 0.28

#: Minimum inches/sec for a direction change to count as a collision. Below
#: this, a "collision" is detection jitter on a nearly stationary ball.
COLLISION_MIN_SPEED = 8.0

#: Degrees of heading change that counts as a collision. A cushion rebound is
#: 60-180 degrees and a cut is 20-90; tracking noise on a rolling ball is a few
#: degrees per frame. 25 sits well clear of the noise floor.
COLLISION_ANGLE_DEG = 25.0

#: Per-ball cooldown between auto-detected bursts. Without it, one cushion
#: contact spawns a burst on several consecutive frames as the heading estimate
#: settles.
COLLISION_COOLDOWN_SECONDS = 0.2

#: Inches from a pocket mouth within which a vanished ball is assumed potted
#: rather than lost by the detector. Generous relative to the 2.25 in pocket
#: radius because a ball is *gone* by the time it is unobservable, and its last
#: good sample is a frame or two before that.
POCKET_CAPTURE_IN = 4.0

#: Hard cap on concurrent effects. A cap rather than a warning because the
#: failure this guards against is a detector flapping and spawning a burst every
#: frame -- which would grow the list without bound and quietly eat the frame
#: budget. Oldest are dropped first.
MAX_EFFECTS = 48


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------
#
# Linear interpolation is what makes an animation look computer-generated. Every
# effect here eases, and the choice of curve carries the meaning: things that are
# *arriving* decelerate (ease-out), things being *swallowed* accelerate
# (ease-in), and things celebrating overshoot and settle (back).


def ease_out_cubic(t: float) -> float:
    """Fast then slow. The default for anything expanding or appearing."""
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_cubic(t: float) -> float:
    """Slow then fast. For the ball accelerating into the pocket."""
    t = min(1.0, max(0.0, t))
    return t * t * t


def ease_in_out_cubic(t: float) -> float:
    """Symmetric. For anything that starts and ends at rest, such as a pulse."""
    t = min(1.0, max(0.0, t))
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease_out_back(t: float, overshoot: float = 1.7) -> float:
    """Overshoots past 1 and settles back. The "pop" in a score popup."""
    t = min(1.0, max(0.0, t))
    c3 = overshoot + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + overshoot * (t - 1.0) ** 2


# ---------------------------------------------------------------------------
# Draw context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EffectContext:
    """Everything an effect needs to turn inches into pixels.

    Built once per frame and shared by every effect drawn in it, so the
    projector scale is one measurement per frame rather than one per effect --
    :meth:`~projection.mapper.ProjectionMapper.pixels_per_inch` costs three
    matrix applications, which is nothing once and adds up across forty sparks.
    """

    mapper: ProjectionMapper
    theme: Theme
    settings: Settings
    px_per_inch: float

    @classmethod
    def build(
        cls,
        mapper: ProjectionMapper,
        settings: Settings | None = None,
        theme: Theme | None = None,
    ) -> EffectContext:
        settings = settings or get_settings()
        return cls(
            mapper=mapper,
            theme=theme or resolve_theme(settings),
            settings=settings,
            px_per_inch=mapper.pixels_per_inch(),
        )

    def to_px(self, point: Vec2) -> Vec2:
        """Table inches -> projector px."""
        return self.mapper.table_to_projector(point)

    def inches(self, value: float) -> float:
        """Inches -> projector px, at the frame's measured scale."""
        return value * self.px_per_inch


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


class Effect(ABC):
    """One time-limited animation.

    Subclasses implement :meth:`draw` and are otherwise pure data. They are
    driven from outside -- an effect never reads the clock itself -- so a whole
    frame renders at one consistent instant, and so tests can step time.
    """

    __slots__ = ("start", "duration")

    def __init__(self, start: float, duration: float) -> None:
        self.start = start
        self.duration = max(1e-3, duration)

    def progress(self, now: float) -> float:
        """Position through the effect's life, clamped to 0-1."""
        return min(1.0, max(0.0, (now - self.start) / self.duration))

    def is_alive(self, now: float) -> bool:
        return now < self.start + self.duration

    @abstractmethod
    def draw(self, canvas: np.ndarray, ctx: EffectContext, now: float) -> None:
        """Render this effect's current frame into the canvas."""


class CollisionBurst(Effect):
    """Spark burst at a ball-to-ball or ball-to-cushion contact.

    Three parts, which together read as an impact rather than as a decoration:
    expanding rings (the energy), radiating sparks (the debris), and an arrow
    along the outgoing direction (the information). The arrow is the part worth
    keeping if the budget ever forces a choice -- it tells the player where the
    ball went, and the rest is delight.
    """

    __slots__ = ("position", "outgoing_deg", "color", "scale")

    def __init__(
        self,
        position: Vec2,
        start: float,
        outgoing_deg: float | None = None,
        color: RGB | None = None,
        duration: float = BURST_SECONDS,
        scale: float = 1.0,
    ) -> None:
        super().__init__(start, duration)
        self.position = position
        self.outgoing_deg = outgoing_deg
        self.color = color
        self.scale = scale

    def draw(self, canvas: np.ndarray, ctx: EffectContext, now: float) -> None:
        t = self.progress(now)
        eased = ease_out_cubic(t)
        color = self.color or ctx.theme.impact
        center = ctx.to_px(self.position)
        fade = int(round(255 * (1.0 - t) ** 1.5))
        if fade <= 4:
            return

        # Two rings, the second lagging a third of the duration behind, so the
        # burst reads as a pulse rather than a single expanding circle.
        for lag in (0.0, 0.33):
            ring_t = (t - lag) / (1.0 - lag) if t > lag else None
            if ring_t is None:
                continue
            radius = ctx.inches((0.6 + 3.4 * ease_out_cubic(ring_t)) * self.scale)
            alpha = int(round(fade * (1.0 - ring_t)))
            draw.draw_ring(canvas, center, radius, color, thickness=3, alpha=alpha, glow=ctx.theme.glow)

        # Sparks fly outward slightly ahead of the rings and shrink as they go.
        spark_radius = ctx.inches((0.8 + 4.2 * eased) * self.scale)
        draw.draw_starburst(
            canvas,
            center,
            spark_radius,
            color,
            count=8,
            rotation_deg=18.0 * eased,
            alpha=fade,
            dot_radius=max(1.5, ctx.inches(0.16) * (1.0 - t)),
        )

        if self.outgoing_deg is not None:
            draw.draw_arrow(
                canvas,
                center,
                self.outgoing_deg,
                ctx.inches(3.0 + 2.0 * eased),
                color,
                thickness=3,
                alpha=fade,
            )


class PocketVortex(Effect):
    """Pocket celebration: halo, collapsing vortex, then a sparkle burst.

    Follows the spec's sequence, with the phases overlapping rather than running
    strictly one after another -- five sequential 100 ms stages inside half a
    second gives each one three frames at 30 FPS, which is not enough to read as
    an animation. Overlapping them means every frame has something moving.

    The vortex spokes collapse toward the centre on an *ease-in* curve, which is
    the detail that makes it look like the pocket is pulling: a linear or
    ease-out collapse looks like the spokes are simply shrinking.
    """

    __slots__ = ("position", "color", "spokes")

    def __init__(
        self,
        position: Vec2,
        start: float,
        color: RGB | None = None,
        duration: float = 0.9,
        spokes: int = 8,
    ) -> None:
        super().__init__(start, duration)
        self.position = position
        self.color = color
        self.spokes = spokes

    def draw(self, canvas: np.ndarray, ctx: EffectContext, now: float) -> None:
        t = self.progress(now)
        color = self.color or ctx.theme.accent
        center = ctx.to_px(self.position)

        # Phase 1 (0 - 0.6): halo swells and fades.
        if t < 0.6:
            u = t / 0.6
            draw.draw_ring(
                canvas,
                center,
                ctx.inches(1.4 + 2.6 * ease_out_cubic(u)),
                color,
                thickness=4,
                alpha=int(round(220 * (1.0 - u))),
                glow=True,
            )

        # Phase 2 (0 - 0.65): spokes rotate and collapse inward.
        if t < 0.65:
            u = t / 0.65
            draw.draw_radial_lines(
                canvas,
                center,
                inner_radius=ctx.inches(2.6 * (1.0 - ease_in_cubic(u))),
                outer_radius=ctx.inches(3.2),
                color=color,
                count=self.spokes,
                rotation_deg=200.0 * u,
                thickness=2,
                alpha=int(round(200 * (1.0 - u * 0.7))),
            )

        # Phase 3 (0.35 - 1.0): sparkles fly out of the pocket.
        if t > 0.35:
            u = (t - 0.35) / 0.65
            draw.draw_starburst(
                canvas,
                center,
                ctx.inches(1.0 + 5.0 * ease_out_cubic(u)),
                color,
                count=12,
                rotation_deg=-40.0 * u,
                alpha=int(round(235 * (1.0 - u) ** 1.4)),
                dot_radius=max(1.5, ctx.inches(0.22) * (1.0 - u)),
            )


class ScorePopup(Effect):
    """Floating text that pops in, drifts and fades.

    Drifts in *inches*, not pixels, so it travels the same physical distance
    wherever on the table it was spawned. Colour comes from the caller rather
    than being derived from the value: what counts as a big score is a game rule.

    It drifts **inboard**, toward the middle of the table, rather than "up".
    There is no up: the projection surface is horizontal, and a score popup is
    spawned at a pocket, which is by definition at the edge. Drifting along a
    fixed screen direction sends every popup at a top-rail pocket straight off
    the canvas within a frame or two, where OpenCV silently discards it -- so the
    celebration for half the pockets on the table simply never appears.
    """

    __slots__ = ("position", "text", "color", "rise_in", "scale")

    def __init__(
        self,
        position: Vec2,
        text: str,
        start: float,
        color: RGB | None = None,
        duration: float = 1.0,
        rise_in: float = 7.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(start, duration)
        self.position = position
        self.text = text
        self.color = color
        self.rise_in = rise_in
        self.scale = scale

    def draw(self, canvas: np.ndarray, ctx: EffectContext, now: float) -> None:
        t = self.progress(now)
        color = self.color or ctx.theme.accent

        # Drift toward the table's mid-line on the short axis: away from
        # whichever long rail the pocket sits on, and therefore always onto felt
        # the projector actually covers.
        half_width = ctx.settings.table.width_in / 2.0
        direction = 1.0 if self.position.y < half_width else -1.0
        drifted = Vec2(
            self.position.x,
            self.position.y + direction * self.rise_in * ease_out_cubic(t),
        )
        center = ctx.to_px(drifted)

        # Pop in over the first 20%, then hold. Fade only over the last 45%, so
        # the number is fully legible for most of its life -- fading throughout
        # would make it unreadable at exactly the moment the player looks up.
        pop = ease_out_back(min(1.0, t / 0.2))
        text_scale = self.scale * (0.6 + 0.6 * pop) * max(0.4, ctx.px_per_inch / 26.0)
        alpha = 255 if t < 0.55 else int(round(255 * (1.0 - (t - 0.55) / 0.45)))
        draw.draw_text(
            canvas,
            self.text,
            center,
            color,
            scale=text_scale,
            thickness=3,
            alpha=alpha,
            anchor="c",
        )


# ---------------------------------------------------------------------------
# Ball trails
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BallTrail:
    """Recent table positions of one ball, newest last.

    A deque of samples rather than a fixed-length ring: the trail is trimmed by
    *age*, not by count, so it stays half a second long whether the loop is
    running at 30 FPS or has dropped to 12 under load. A count-based trail would
    get longer in wall-clock terms exactly when the system is struggling.
    """

    ball_id: str
    color: RGB
    samples: deque[tuple[Vec2, float]] = field(default_factory=lambda: deque(maxlen=64))

    def add(self, position: Vec2, now: float) -> None:
        self.samples.append((position, now))

    def trim(self, now: float, window: float = TRAIL_SECONDS) -> None:
        while self.samples and now - self.samples[0][1] > window:
            self.samples.popleft()

    @property
    def last_seen(self) -> float:
        return self.samples[-1][1] if self.samples else float("-inf")

    def speed(self) -> float:
        """Inches/sec from the two most recent samples, or 0.

        Two samples rather than a fit over the whole trail: the trail is used to
        decide *right now* whether the ball is moving and how thick to draw it,
        and a fit over half a second lags badly at the start and end of a shot,
        which is when it matters.
        """
        if len(self.samples) < 2:
            return 0.0
        (p0, t0), (p1, t1) = self.samples[-2], self.samples[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0
        return p0.distance_to(p1) / dt

    def heading_deg(self) -> float | None:
        """Table-space heading from the two most recent samples, or ``None``."""
        if len(self.samples) < 2:
            return None
        (p0, _), (p1, _) = self.samples[-2], self.samples[-1]
        delta = p1 - p0
        if delta.length() < 1e-6:
            return None
        return math.degrees(math.atan2(delta.y, delta.x))

    def draw(self, canvas: np.ndarray, ctx: EffectContext, now: float) -> None:
        """Draw the trail as a polyline fading and narrowing toward the tail."""
        self.trim(now)
        if len(self.samples) < 2:
            return
        points = [ctx.to_px(pos) for pos, _ in self.samples]
        # Thickness tracks speed, as the spec asks. Capped at 5 balls/sec worth
        # of speed so a hard break does not produce a stripe wide enough to hide
        # the balls underneath it.
        speed_factor = min(1.0, self.speed() / 120.0)
        head = max(2, int(round(ctx.inches(0.35 + 0.55 * speed_factor))))
        draw.draw_fading_polyline(
            canvas,
            points,
            self.color,
            thickness_head=head,
            thickness_tail=1,
            alpha_head=int(round(235 * ctx.theme.secondary_alpha + 20)),
            alpha_tail=0,
            glow=ctx.theme.glow,
        )


# ---------------------------------------------------------------------------
# The system
# ---------------------------------------------------------------------------


class EffectSystem:
    """Holds live effects and ball trails between frames.

    One instance per session, owned by whatever drives rendering (the mode
    manager once Phase 7 lands; the projection test tool today). Not thread-safe
    -- it is touched only from the vision loop, which is the single thread that
    renders.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._effects: list[Effect] = []
        self._trails: dict[str, BallTrail] = {}
        #: Last known table position per ball id, kept after the ball vanishes
        #: so a pot can be located.
        self._last_positions: dict[str, Vec2] = {}
        self._last_headings: dict[str, float] = {}
        self._last_burst: dict[str, float] = {}
        #: Ball ids already celebrated, so one pot is one vortex.
        self._pocketed_seen: set[str] = set()
        self._pockets: list[Vec2] = TableGeometry.from_settings(self.settings).pocket_centers()

    # -- spawning -----------------------------------------------------------

    def add(self, effect: Effect) -> Effect:
        """Add an effect, dropping the oldest if the cap is reached."""
        if len(self._effects) >= MAX_EFFECTS:
            # Oldest first: a burst that has been on screen longest has least
            # left to show, so dropping it is the least visible choice.
            self._effects.pop(0)
            logger.debug("effect cap %d reached; dropped the oldest", MAX_EFFECTS)
        self._effects.append(effect)
        return effect

    def spawn_collision(
        self,
        position: Vec2,
        outgoing_deg: float | None = None,
        color: RGB | None = None,
        now: float | None = None,
        scale: float = 1.0,
    ) -> Effect:
        """Burst at a contact point. ``outgoing_deg`` is a **table** angle."""
        now = time.perf_counter() if now is None else now
        projector_deg = None if outgoing_deg is None else _table_angle_to_projector(outgoing_deg)
        return self.add(
            CollisionBurst(position, now, projector_deg, color, scale=scale)
        )

    def spawn_pocket(
        self,
        position: Vec2,
        color: RGB | None = None,
        points: int | None = None,
        now: float | None = None,
    ) -> Effect:
        """Celebrate a pot.

        ``points`` is optional and only produces the floating number. The
        renderer does not know what a ball is worth -- see the module docstring.
        """
        now = time.perf_counter() if now is None else now
        effect = self.add(
            PocketVortex(
                position, now, color, duration=self.settings.render.effect_duration_seconds
            )
        )
        if points is not None:
            self.spawn_score(position, f"+{points}", color=color, now=now)
        return effect

    def spawn_score(
        self,
        position: Vec2,
        text: str,
        color: RGB | None = None,
        now: float | None = None,
        scale: float = 1.0,
    ) -> Effect:
        """Floating text at a table position."""
        now = time.perf_counter() if now is None else now
        return self.add(ScorePopup(position, text, now, color, scale=scale))

    # -- per-frame observation ----------------------------------------------

    def observe(
        self,
        game_state: GameState,
        now: float | None = None,
        *,
        detect_collisions: bool = True,
        detect_pockets: bool = True,
    ) -> None:
        """Feed a frame in: extend trails, and spawn effects for what changed.

        Called once per frame while balls are moving. Safe to call every frame
        regardless of state -- a table at rest produces no trails worth drawing,
        because every trail is trimmed by age and a still ball's samples all sit
        on one point.

        Args:
            game_state: This frame's observed state. Balls without a
                ``table_pos`` are skipped: physics coordinates are the only ones
                effects can use, and a camera-space position would animate in
                the wrong place.
            detect_collisions: Spawn a burst when a ball's heading changes
                sharply. This is a *fallback* for running without the game
                modes; when the mode layer reports real impacts, turn it off and
                use :meth:`spawn_collision`, which knows the exact contact point
                rather than inferring one from a frame boundary.
            detect_pockets: Spawn a vortex when a ball is flagged pocketed or
                vanishes at a pocket mouth.
        """
        now = time.perf_counter() if now is None else now
        theme = resolve_theme(self.settings)
        seen: set[str] = set()

        for ball in game_state.balls:
            if ball.table_pos is None:
                continue
            if ball.pocketed:
                if detect_pockets:
                    self._celebrate(ball, ball.table_pos, theme, now)
                continue

            seen.add(ball.id)
            trail = self._trails.get(ball.id)
            if trail is None:
                trail = self._trails[ball.id] = BallTrail(
                    ball.id, ball_display_color(ball, theme)
                )
            else:
                # Re-read the colour: the detector's classification can change
                # between frames, and a trail stuck on the first guess looks like
                # a bug when the ball is finally identified.
                trail.color = ball_display_color(ball, theme)
            trail.add(ball.table_pos, now)
            trail.trim(now)
            self._last_positions[ball.id] = ball.table_pos

            if detect_collisions:
                self._check_collision(trail, now)

        if detect_pockets:
            self._check_vanished(seen, game_state, theme, now)

        self._expire_trails(seen, now)

    def _check_collision(self, trail: BallTrail, now: float) -> None:
        """Spawn a burst when a moving ball's heading changes sharply."""
        heading = trail.heading_deg()
        previous = self._last_headings.get(trail.ball_id)
        if heading is not None:
            self._last_headings[trail.ball_id] = heading
        if heading is None or previous is None:
            return
        if trail.speed() < COLLISION_MIN_SPEED:
            return
        # Signed shortest angular difference, so 350 -> 10 degrees is 20 and not
        # 340. Getting this wrong would fire a burst on every frame of a ball
        # rolling through the wrap-around.
        delta = abs((heading - previous + 180.0) % 360.0 - 180.0)
        if delta < COLLISION_ANGLE_DEG:
            return
        if now - self._last_burst.get(trail.ball_id, float("-inf")) < COLLISION_COOLDOWN_SECONDS:
            return
        self._last_burst[trail.ball_id] = now
        position = trail.samples[-1][0]
        self.spawn_collision(position, outgoing_deg=heading, color=trail.color, now=now)

    def _check_vanished(
        self, seen: set[str], game_state: GameState, theme: Theme, now: float
    ) -> None:
        """Celebrate balls that disappeared at a pocket mouth.

        The detector removes a potted ball from the frame rather than flagging
        it, so absence plus proximity to a pocket is the observable. Absence
        *away* from a pocket is an occlusion -- a hand, a leaning player -- and
        must not fire, which is what the distance test buys.
        """
        for ball_id, position in list(self._last_positions.items()):
            if ball_id in seen or ball_id in self._pocketed_seen:
                continue
            trail = self._trails.get(ball_id)
            if trail is not None and now - trail.last_seen < TRAIL_EXPIRY_SECONDS:
                continue  # too soon to call it: might just be a dropped frame
            pocket = self._nearest_pocket(position)
            if pocket is not None and pocket.distance_to(position) <= POCKET_CAPTURE_IN:
                color = trail.color if trail is not None else theme.accent
                self._pocketed_seen.add(ball_id)
                self.spawn_pocket(pocket, color=color, now=now)
                logger.debug("ball %s vanished at pocket %s; celebrating", ball_id, pocket)
            self._last_positions.pop(ball_id, None)

    def _celebrate(self, ball: Ball, position: Vec2, theme: Theme, now: float) -> None:
        """Spawn a vortex for an explicitly flagged pot, once per ball."""
        if ball.id in self._pocketed_seen:
            return
        self._pocketed_seen.add(ball.id)
        pocket = self._nearest_pocket(position) or position
        self.spawn_pocket(pocket, color=ball_display_color(ball, theme), now=now)

    def _expire_trails(self, seen: set[str], now: float) -> None:
        for ball_id, trail in list(self._trails.items()):
            if ball_id in seen:
                continue
            if now - trail.last_seen > TRAIL_EXPIRY_SECONDS:
                del self._trails[ball_id]
                self._last_headings.pop(ball_id, None)

    def _nearest_pocket(self, position: Vec2) -> Vec2 | None:
        if not self._pockets:
            return None
        return min(self._pockets, key=position.distance_to)

    # -- per-frame update and draw ------------------------------------------

    def update(self, now: float | None = None) -> int:
        """Drop finished effects. Returns how many are still live."""
        now = time.perf_counter() if now is None else now
        self._effects = [e for e in self._effects if e.is_alive(now)]
        return len(self._effects)

    def render(
        self,
        canvas: np.ndarray,
        ctx: EffectContext,
        now: float | None = None,
        *,
        trails: bool = True,
    ) -> np.ndarray:
        """Draw trails then effects into ``canvas``.

        Trails first so that a burst at the end of a trail sits on top of it,
        which is the correct reading -- the impact happened after the travel.
        """
        now = time.perf_counter() if now is None else now
        self.update(now)
        if trails:
            for trail in self._trails.values():
                trail.draw(canvas, ctx, now)
        for effect in self._effects:
            effect.draw(canvas, ctx, now)
        return canvas

    # -- lifecycle ----------------------------------------------------------

    def clear(self) -> None:
        """Drop everything. Called on mode change and on reset.

        Also clears the potted-ball memory, so re-racking does not leave every
        ball permanently un-celebratable.
        """
        self._effects.clear()
        self._trails.clear()
        self._last_positions.clear()
        self._last_headings.clear()
        self._last_burst.clear()
        self._pocketed_seen.clear()

    def refresh_geometry(self, settings: Settings | None = None) -> None:
        """Re-derive pocket positions after a table-size change."""
        if settings is not None:
            self.settings = settings
        self._pockets = TableGeometry.from_settings(self.settings).pocket_centers()

    @property
    def active_count(self) -> int:
        """Live effects, not counting trails. Reported by the test tool."""
        return len(self._effects)

    @property
    def trail_count(self) -> int:
        return len(self._trails)


def _table_angle_to_projector(table_deg: float) -> float:
    """Table-space angle -> projector-space angle.

    Table angles are counter-clockwise from +x; projector angles are clockwise
    from +x, because image +y points down. The conversion is therefore a
    negation -- assuming the calibration does not rotate or mirror the table,
    which is true of any sane projector placement and is only used to point
    decorative arrows. Anything that has to be geometrically exact maps its
    points through the transform instead of converting an angle.
    """
    return -table_deg


# ---------------------------------------------------------------------------
# Animated UI elements
# ---------------------------------------------------------------------------
#
# Stateless: they are a function of a value and the clock, with no history to
# keep, so they are plain functions rather than Effects. The renderer calls them
# while composing game UI.


def draw_combo_badge(
    canvas: np.ndarray,
    combo_count: int,
    ctx: EffectContext,
    position: Vec2,
    now: float | None = None,
) -> None:
    """Draw an ``x3``-style multiplier badge that pulses.

    Nothing is drawn below ``x2``: a badge reading "x1" is not information, and
    a permanent badge stops being noticed.

    ``position`` is in **projector px**, not inches -- this is UI furniture
    anchored to the frame, not something sitting on the felt at a table
    position.
    """
    if combo_count < 2:
        return
    now = time.perf_counter() if now is None else now
    theme = ctx.theme

    # A 2 Hz pulse, easing in and out so it breathes rather than blinks.
    pulse = ease_in_out_cubic(abs(math.sin(now * math.pi * 2.0)))
    scale = 1.5 + 0.35 * pulse
    color = theme.accent if combo_count == 2 else theme.impact
    label = f"x{combo_count}"

    width, height = draw.text_size(label, scale, 3)
    radius = max(width, height) * 0.75
    # The badge is a *ring* around a rail anchor, and the anchor is clamped to
    # the frame edge on an uncalibrated setup -- so the ring gets its top sliced
    # off unless it makes room for itself. Nudging the centre inward is the fix;
    # the alternative, growing the anchor's padding, would push the text of
    # every other UI element inward to accommodate one circle.
    center = Vec2(position.x, max(position.y, radius + 4.0))
    draw.draw_circle(
        canvas,
        center,
        radius,
        color,
        thickness=3,
        alpha=int(round(120 + 100 * pulse)),
    )
    draw.draw_text(canvas, label, center, color, scale=scale, thickness=3, anchor="c")


def draw_timer_countdown(
    canvas: np.ndarray,
    seconds_remaining: float,
    ctx: EffectContext,
    position: Vec2,
    now: float | None = None,
) -> None:
    """Draw a countdown that changes colour and blinks as it runs out.

    Colour bands are the spec's: green above 20 s, amber from 10 to 20, alert
    below 10, plus a blink in the last three seconds. Colour is doing the work
    here rather than the number -- a player mid-shot registers "it went red", not
    "it says 7".

    ``position`` is in **projector px**.
    """
    now = time.perf_counter() if now is None else now
    remaining = max(0.0, seconds_remaining)
    theme = ctx.theme

    if remaining > 20.0:
        color = theme.cue_path
    elif remaining > 10.0:
        color = theme.impact
    else:
        color = theme.alert

    scale = 1.8
    alpha = 255
    if remaining <= 3.0:
        # Square wave at 3 Hz: a smooth fade at this point reads as the display
        # dimming, not as urgency.
        blink = math.sin(now * math.pi * 6.0) > 0
        alpha = 255 if blink else 60
        scale += 0.25 * ease_in_out_cubic(1.0 - (remaining / 3.0) % 1.0)

    minutes, seconds = divmod(int(remaining), 60)
    label = f"{minutes}:{seconds:02d}" if minutes else f"{seconds}"
    draw.draw_text(
        canvas, label, position, color, scale=scale, thickness=4, alpha=alpha, anchor="tr"
    )


def ghost_ball_radius_px(ctx: EffectContext) -> float:
    """A ball's radius in projector px at this frame's scale."""
    return ctx.inches(BALL_RADIUS_IN)
