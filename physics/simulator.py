"""Shot trajectory simulation.

Event-driven, not fixed-step, and that is a measured decision rather than a
stylistic one. A NumPy-vectorised step loop costs ~31 us per step in Python, so
the configured "balanced" profile (4 ms steps over 8 s) would be **62 ms per
shot** and "accurate" would be 374 ms -- against a 33 ms frame budget that
detection already spends 9 ms of. Fixed-step integration cannot work here at any
useful resolution.

Instead the simulator solves analytically for *when the next thing happens* and
jumps straight there. A shot has between two and twenty events, so the cost
scales with the number of collisions rather than with simulated time.

Two consequences beyond speed, both of which matter:

**No tunnelling.** A hard-struck ball moves ~4 inches per 16 ms step, well over
a ball diameter, so a stepped simulation can pass one ball straight through
another. Solving for the contact distance analytically cannot miss a collision.

**Straight segments, so paths are exact polylines.** With rolling friction the
ball decelerates but does not curve, so between two events its path is a
straight line. The returned polyline needs only the event points and is exact --
no sampling, no decimation, and far fewer points for the renderer to draw.

The physics model
-----------------
Rolling resistance is treated as a constant deceleration ``a`` along the
direction of travel, which is the standard model for a rolling (not sliding)
ball and is mass-independent. So for a ball with initial speed ``s``:

    speed(t)     = s - a*t
    distance(t)  = s*t - a*t^2/2
    time to stop = s/a
    total roll   = s^2/(2a)

Side spin is modelled only in its effect on cushion rebound. Top and bottom spin
are deliberately not modelled: doing it properly needs a three-dimensional model
of the sliding-to-rolling transition, and for an *aiming aid* the payoff is small
next to getting cut angles right. See :mod:`physics.models`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import Settings, get_settings
from app.models import (
    Ball,
    CollisionResult,
    CueStick,
    GameState,
    ImpactEvent,
    ShotPrediction,
    Vec2,
)
from physics.models import (
    ACCURACY_PROFILES,
    BallPhysics,
    SimBall,
    SimConfig,
    TableGeometry,
    power_to_velocity,
)

logger = logging.getLogger(__name__)

#: Numerical slack, in inches. After a collision the balls are nudged apart by
#: this much so the very next solve does not re-detect the same contact at t=0
#: and stall the event loop.
EPSILON_IN = 1e-4

#: Minimum spacing between consecutive path points, inches. Larger than
#: EPSILON_IN on purpose: the post-collision separation nudge is exactly
#: EPSILON_IN, so a threshold equal to it fails to dedupe by a hair. A
#: hundredth of an inch is ~0.2 px on the projected table -- invisible.
PATH_DEDUPE_IN = 0.01

#: Hard cap on events per shot. A physical shot cannot produce hundreds of
#: collisions, so hitting this means a numerical problem -- two balls trapped in
#: a contact loop -- and the loop must break rather than hang inside a frame.
MAX_EVENTS = 400


class EventKind:
    """What kind of event ends a ball's free flight.

    Ordering matters when two events land at the same instant, which happens
    constantly at a pocket: the pocket mouth overlaps the cushion line, so
    without a priority a ball that should drop would bounce off the rail
    instead. Lower sorts first.
    """

    POCKET = 0
    BALL = 1
    CUSHION = 2
    STOP = 3


@dataclass(slots=True)
class _Event:
    """The next thing that will happen, and to whom."""

    time: float
    kind: int
    ball_index: int
    other_index: int = -1  # ball-ball target, or cushion axis
    detail: int = 0  # cushion: 0=x_min 1=x_max 2=y_min 3=y_max; pocket: index

    def sort_key(self) -> tuple[float, int]:
        return (self.time, self.kind)


# ---------------------------------------------------------------------------
# Precomputed geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CushionTable:
    """Precomputed cushion lines and pocket centres for one table geometry.

    Built once per geometry and cached. The spec asks for precomputed lookup
    tables; this is the version that actually pays, because these values are
    consulted on every event of every shot and never change.
    """

    bounds: tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    pocket_centers: np.ndarray  # 6x2
    pocket_radius: float


@lru_cache(maxsize=8)
def _cushion_table(
    length: float, width: float, pocket_radius: float, ball_radius: float
) -> _CushionTable:
    """Cached cushion/pocket table, keyed on the scalars that define it.

    Keyed on primitives rather than on the geometry object so ``lru_cache`` can
    hash it, and so two equal geometries share one entry.
    """
    geometry = TableGeometry(
        length_in=length,
        width_in=width,
        pocket_radius_in=pocket_radius,
        ball_radius_in=ball_radius,
    )
    centers = np.array([[p.x, p.y] for p in geometry.pocket_centers()], dtype=np.float64)
    return _CushionTable(
        bounds=(geometry.x_min, geometry.x_max, geometry.y_min, geometry.y_max),
        pocket_centers=centers,
        pocket_radius=pocket_radius,
    )


def cushion_table_for(geometry: TableGeometry) -> _CushionTable:
    """Public accessor for the cached cushion table."""
    return _cushion_table(
        geometry.length_in,
        geometry.width_in,
        geometry.pocket_radius_in,
        geometry.ball_radius_in,
    )


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def total_roll_distance(speed: float, deceleration: float) -> float:
    """How far a ball rolls before stopping, inches."""
    if deceleration <= 0:
        return math.inf
    return speed * speed / (2.0 * deceleration)


def time_for_distance(speed: float, deceleration: float, distance: float) -> float:
    """Time to cover ``distance``, or ``inf`` if the ball stops short.

    Inverts ``d = s*t - a*t^2/2`` and takes the smaller root, which is the first
    time the ball reaches that distance rather than the unphysical second root
    after it would have reversed.
    """
    if distance <= 0.0:
        return 0.0
    if deceleration <= 0:
        return distance / speed if speed > 0 else math.inf
    discriminant = speed * speed - 2.0 * deceleration * distance
    if discriminant < 0.0:
        return math.inf  # stops before getting there
    return (speed - math.sqrt(discriminant)) / deceleration


def distance_at_time(speed: float, deceleration: float, t: float) -> float:
    """Distance travelled by time ``t``, clamped at the stopping distance."""
    stop_time = speed / deceleration if deceleration > 0 else math.inf
    t = min(t, stop_time)
    return speed * t - 0.5 * deceleration * t * t


# ---------------------------------------------------------------------------
# Exact ray intersections (the core of the event solver)
# ---------------------------------------------------------------------------


def ray_circle_entry(
    origin: np.ndarray,
    direction: np.ndarray,
    centers: np.ndarray,
    radius: float,
    max_distance: float,
) -> tuple[int, float]:
    """First circle a ray enters, as ``(index, distance)`` or ``(-1, inf)``.

    Vectorised over ``centers``, which is what keeps a 15-ball table cheap: one
    NumPy pass rather than a Python loop per candidate.

    This is a *geometric* solve along the path, not a temporal one, and that is
    the trick that makes the whole simulator tractable. A decelerating ball
    still travels in a straight line, so "where does the path first come within
    ``radius`` of a centre" has a closed form independent of the speed profile.
    Time is recovered afterwards from the distance.
    """
    if len(centers) == 0:
        return -1, math.inf

    to_center = centers - origin  # Nx2
    along = to_center @ direction  # projection onto the ray
    # Perpendicular distance from each centre to the ray line.
    perpendicular_sq = np.einsum("ij,ij->i", to_center, to_center) - along * along
    radius_sq = radius * radius

    # A circle is reachable only if the line passes within radius and the centre
    # is ahead of the origin.
    reachable = (perpendicular_sq <= radius_sq) & (along > 0.0)
    if not reachable.any():
        return -1, math.inf

    half_chord = np.sqrt(np.maximum(radius_sq - perpendicular_sq, 0.0))
    entry = along - half_chord
    entry = np.where(reachable & (entry >= 0.0) & (entry <= max_distance), entry, np.inf)

    best = int(np.argmin(entry))
    return (best, float(entry[best])) if math.isfinite(entry[best]) else (-1, math.inf)


def ray_cushion_exit(
    origin: np.ndarray,
    direction: np.ndarray,
    bounds: tuple[float, float, float, float],
    max_distance: float,
) -> tuple[int, float]:
    """Distance until the ray leaves the cushion box, as ``(axis, distance)``.

    ``axis`` is 0..3 for x_min, x_max, y_min, y_max. The bounds are already
    inset by a ball radius (see :class:`physics.models.TableGeometry`), so the
    ball *centre* is compared directly and no radius is added here.
    """
    x_min, x_max, y_min, y_max = bounds
    best_distance = math.inf
    best_axis = -1

    for axis, (limit, component) in enumerate(
        ((x_min, 0), (x_max, 0), (y_min, 1), (y_max, 1))
    ):
        d = direction[component]
        if abs(d) < 1e-12:
            continue  # parallel to this cushion
        distance = (limit - origin[component]) / d
        if 0.0 < distance <= max_distance and distance < best_distance:
            best_distance = distance
            best_axis = axis

    return best_axis, best_distance


# ---------------------------------------------------------------------------
# Collision primitives (public, per the spec)
# ---------------------------------------------------------------------------


def predict_collision(
    ball1_pos: Vec2,
    ball1_vel: Vec2,
    ball2_pos: Vec2,
    physics: BallPhysics | None = None,
) -> CollisionResult:
    """Test whether a moving ball will reach a stationary one.

    Solved analytically rather than by stepping. Exact, cheap, and -- crucially
    -- unable to tunnel through the target the way a fixed step can on a
    hard-struck shot.

    Note this uses the ball's *current* velocity and reports a time as though
    that velocity were constant. Deceleration is applied by the caller, which
    knows the friction model; keeping it out of here makes the function a pure
    geometric query and directly testable against hand-computed cases.
    """
    physics = physics or BallPhysics()
    contact = physics.contact_distance

    origin = np.array([ball1_pos.x, ball1_pos.y], dtype=np.float64)
    target = np.array([ball2_pos.x, ball2_pos.y], dtype=np.float64)
    velocity = np.array([ball1_vel.x, ball1_vel.y], dtype=np.float64)

    speed = float(np.linalg.norm(velocity))
    if speed < 1e-12:
        return CollisionResult(will_collide=False)

    separation = float(np.linalg.norm(target - origin))
    if separation < contact - EPSILON_IN:
        # Already overlapping. Real, and worth logging: it means detection
        # merged two balls or a previous resolution left them interpenetrating.
        logger.debug("balls already overlapping (%.4f in apart, contact %.4f)", separation, contact)

    direction = velocity / speed
    index, distance = ray_circle_entry(origin, direction, target[None, :], contact, math.inf)
    if index < 0:
        return CollisionResult(will_collide=False)

    contact_point = origin + direction * distance
    # The line of centres at the moment of contact -- this is the direction the
    # object ball departs along, and the number players actually judge.
    normal = target - contact_point
    normal_angle = math.degrees(math.atan2(normal[1], normal[0]))

    return CollisionResult(
        will_collide=True,
        point=Vec2(float(contact_point[0]), float(contact_point[1])),
        time_to_impact=distance / speed,
        normal_angle_deg=normal_angle,
    )


def resolve_ball_collision(
    ball1: SimBall, ball2: SimBall, physics: BallPhysics | None = None
) -> None:
    """Apply an equal-mass elastic collision between two balls, in place.

    Exchange the velocity components along the line of centres and leave the
    tangential components untouched, scaled by ``ball_restitution``.

    This is the single most important piece of physics in the system. It is what
    produces the **cut angle**: the object ball departs along the line of
    centres, and the cue ball deflects along the tangent, so for a half-ball hit
    the two separate by about 90 degrees. Players know exactly where the object
    ball should go, so an error here is immediately visible in a way that a
    slightly wrong friction coefficient never is.
    """
    physics = physics or BallPhysics()

    p1 = np.array([ball1.position.x, ball1.position.y])
    p2 = np.array([ball2.position.x, ball2.position.y])
    v1 = np.array([ball1.velocity.x, ball1.velocity.y])
    v2 = np.array([ball2.velocity.x, ball2.velocity.y])

    delta = p2 - p1
    separation = float(np.linalg.norm(delta))
    if separation < 1e-9:
        # Exactly coincident centres give no line of centres to work with.
        # Physically impossible; numerically reachable. Pick an arbitrary axis
        # rather than divide by zero.
        logger.debug("coincident ball centres in collision; using an arbitrary normal")
        delta = np.array([1.0, 0.0])
        separation = 1.0

    normal = delta / separation
    # Relative velocity along the normal. Positive means separating, in which
    # case there is nothing to resolve -- resolving anyway would suck them
    # together and cause the pair to vibrate.
    approach = float((v2 - v1) @ normal)
    if approach > 0.0:
        return

    v1n = float(v1 @ normal)
    v2n = float(v2 @ normal)
    v1t = v1 - v1n * normal
    v2t = v2 - v2n * normal

    restitution = physics.ball_restitution
    new_v1 = v1t + normal * (v2n * restitution)
    new_v2 = v2t + normal * (v1n * restitution)

    ball1.velocity = Vec2(float(new_v1[0]), float(new_v1[1]))
    ball2.velocity = Vec2(float(new_v2[0]), float(new_v2[1]))

    # Separate them so the next solve does not re-detect this same contact.
    overlap = physics.contact_distance - separation
    if overlap > 0.0:
        shift = normal * (overlap / 2.0 + EPSILON_IN)
        ball1.position = Vec2(ball1.position.x - shift[0], ball1.position.y - shift[1])
        ball2.position = Vec2(ball2.position.x + shift[0], ball2.position.y + shift[1])


def apply_cushion_bounce(
    ball: SimBall,
    geometry: TableGeometry,
    physics: BallPhysics | None = None,
    axis: int | None = None,
) -> bool:
    """Reflect a ball off any cushion it has reached, in place.

    Reflects the normal component scaled by ``cushion_restitution``, and shrinks
    the tangential component by ``cushion_tangential_loss``. That tangential
    loss is not a detail: real cloth-covered rails grip, so a ball comes off
    *shallower* than mirror reflection predicts, and ignoring it is the main
    reason naive bank-shot predictions are visibly wrong.

    The ball is also repositioned onto the cushion line. Reflecting the velocity
    while leaving the ball outside the bounds means the next solve re-detects
    the same cushion and the ball buzzes along the rail instead of leaving it.

    Args:
        ball: The ball to bounce, modified in place.
        geometry: Table geometry, supplying the inset cushion bounds.
        physics: Restitution parameters.
        axis: Which cushion, 0..3 for x_min/x_max/y_min/y_max. When ``None`` the
            cushion is inferred from whichever bound the ball has passed, which
            is what the standalone (non-event-loop) callers want.

    Returns:
        Whether a bounce happened, so the caller can record an impact event.
    """
    physics = physics or BallPhysics()
    x_min, x_max, y_min, y_max = (
        geometry.x_min,
        geometry.x_max,
        geometry.y_min,
        geometry.y_max,
    )

    if axis is None:
        if ball.position.x <= x_min:
            axis = 0
        elif ball.position.x >= x_max:
            axis = 1
        elif ball.position.y <= y_min:
            axis = 2
        elif ball.position.y >= y_max:
            axis = 3
        else:
            return False

    vx, vy = ball.velocity
    px, py = ball.position
    restitution = physics.cushion_restitution
    tangential = 1.0 - physics.cushion_tangential_loss

    if axis in (0, 1):  # vertical cushion: x is normal, y is tangential
        px = x_min + EPSILON_IN if axis == 0 else x_max - EPSILON_IN
        vx = -vx * restitution
        vy *= tangential
        # Side spin rubs against the cushion and throws the ball along it.
        vy += ball.spin * physics.radius_in * 0.10
    else:  # horizontal cushion: y is normal
        py = y_min + EPSILON_IN if axis == 2 else y_max - EPSILON_IN
        vy = -vy * restitution
        vx *= tangential
        vx += ball.spin * physics.radius_in * 0.10

    # Spin decays sharply in a cushion contact.
    ball.spin *= 0.5
    ball.position = Vec2(px, py)
    ball.velocity = Vec2(vx, vy)
    return True


def check_pocketed(ball: SimBall, geometry: TableGeometry) -> bool:
    """Whether a ball's centre is inside a pocket mouth.

    Modelling the pocket as a plain circle at the mouth centre is deliberately
    generous -- a real pocket rejects a ball arriving at a shallow angle to the
    jaws. That over-predicts made shots, which is the right direction to be
    wrong for an aiming aid: showing a pot that narrowly misses is far less
    annoying than refusing to show one that goes in.
    """
    table = cushion_table_for(geometry)
    position = np.array([ball.position.x, ball.position.y])
    distances = np.linalg.norm(table.pocket_centers - position, axis=1)
    return bool((distances <= table.pocket_radius).any())


# ---------------------------------------------------------------------------
# The event-driven simulator
# ---------------------------------------------------------------------------


def _append_path(ball: SimBall) -> None:
    """Record the ball's current position, skipping duplicates.

    Worth a helper rather than a bare append. Several code paths legitimately
    want to mark a ball's position at an event -- the mover, the target, the
    overlap corrector -- and a stationary ball touched by two of them lands the
    same point in its path twice. Duplicates give the renderer zero-length
    segments to draw, and make any angle derived from consecutive path points
    return atan2(0, 0). That last one is subtle enough that it produced a
    plausible-looking wrong answer during development.
    """
    if ball.path and ball.path[-1].distance_to(ball.position) < PATH_DEDUPE_IN:
        return
    ball.path.append(ball.position)


def _next_event(
    balls: list[SimBall],
    table: _CushionTable,
    physics: BallPhysics,
    deceleration: float,
) -> _Event | None:
    """Find the earliest event across every moving ball.

    Ball-ball timing treats the *other* ball as stationary at its current
    position. That is exact whenever only one ball is moving, which covers the
    entire pre-impact phase -- the part whose accuracy players judge. Once
    several balls are in flight it under-estimates closing speed, so the loop
    also runs an overlap correction after advancing; see
    :func:`_resolve_overlaps`.
    """
    positions = np.array([[b.position.x, b.position.y] for b in balls], dtype=np.float64)
    best: _Event | None = None

    for index, ball in enumerate(balls):
        if ball.pocketed or not ball.is_moving:
            continue

        speed = ball.speed
        direction = np.array([ball.velocity.x, ball.velocity.y]) / speed
        origin = positions[index]
        roll = total_roll_distance(speed, deceleration)

        candidates: list[_Event] = [
            _Event(time=speed / deceleration, kind=EventKind.STOP, ball_index=index)
        ]

        # Pockets first in priority, because a pocket mouth overlaps the cushion
        # line near a corner and the ball should drop rather than bounce.
        pocket_index, pocket_distance = ray_circle_entry(
            origin, direction, table.pocket_centers, table.pocket_radius, roll
        )
        if pocket_index >= 0:
            candidates.append(
                _Event(
                    time=time_for_distance(speed, deceleration, pocket_distance),
                    kind=EventKind.POCKET,
                    ball_index=index,
                    detail=pocket_index,
                )
            )

        # Other balls. Exclude self and anything already pocketed by pushing
        # them far away rather than building a filtered array, which would need
        # an index remap on every event.
        others = positions.copy()
        others[index] = 1e9
        for j, other in enumerate(balls):
            if other.pocketed:
                others[j] = 1e9
        target, distance = ray_circle_entry(
            origin, direction, others, physics.contact_distance, roll
        )
        if target >= 0:
            candidates.append(
                _Event(
                    time=time_for_distance(speed, deceleration, distance),
                    kind=EventKind.BALL,
                    ball_index=index,
                    other_index=target,
                )
            )

        axis, cushion_distance = ray_cushion_exit(origin, direction, table.bounds, roll)
        if axis >= 0:
            candidates.append(
                _Event(
                    time=time_for_distance(speed, deceleration, cushion_distance),
                    kind=EventKind.CUSHION,
                    ball_index=index,
                    detail=axis,
                )
            )

        for candidate in candidates:
            if not math.isfinite(candidate.time):
                continue
            if best is None or candidate.sort_key() < best.sort_key():
                best = candidate

    return best


def _advance(balls: list[SimBall], dt: float, deceleration: float) -> None:
    """Move every ball forward by ``dt`` and record the new position.

    Motion is a straight line between events, so appending only the endpoint
    keeps the path polyline exact -- no interpolation and nothing to decimate.
    """
    if dt <= 0.0:
        return
    for ball in balls:
        if ball.pocketed or not ball.is_moving:
            continue
        speed = ball.speed
        direction = ball.velocity.scaled(1.0 / speed)
        travelled = distance_at_time(speed, deceleration, dt)
        ball.position = ball.position + direction.scaled(travelled)
        new_speed = max(0.0, speed - deceleration * dt)
        ball.velocity = direction.scaled(new_speed)
        _append_path(ball)


def _resolve_overlaps(
    balls: list[SimBall], physics: BallPhysics, impacts: list[ImpactEvent], now: float
) -> int:
    """Resolve any interpenetrating pairs left after advancing.

    This is the correction for the one approximation in the event solver: when
    two balls are both moving, the ball-ball event time is computed as if the
    target were stationary, which under-estimates the closing speed and can let
    them overlap slightly. Detecting and resolving that here keeps the
    simulation stable and physically sensible, and it is much cheaper than
    solving the exact quartic for every moving pair on every event.
    """
    resolved = 0
    contact = physics.contact_distance
    active = [b for b in balls if not b.pocketed]

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            if not (a.is_moving or b.is_moving):
                continue
            if a.position.distance_to(b.position) >= contact - EPSILON_IN:
                continue

            incoming = a.heading_deg() if a.is_moving else b.heading_deg()
            resolve_ball_collision(a, b, physics)
            impacts.append(
                ImpactEvent(
                    position=Vec2(
                        (a.position.x + b.position.x) / 2.0,
                        (a.position.y + b.position.y) / 2.0,
                    ),
                    target_id=b.id,
                    incoming_angle_deg=incoming,
                    outgoing_angle_deg=a.heading_deg(),
                    is_cushion=False,
                    time_offset=now,
                )
            )
            resolved += 1
    return resolved


def simulate_shot(
    cue_ball_pos: Vec2,
    direction_deg: float,
    power: float,
    other_balls: list[Ball] | None = None,
    settings: Settings | None = None,
    spin: float = 0.0,
    config: SimConfig | None = None,
) -> ShotPrediction:
    """Predict the outcome of a shot.

    Args:
        cue_ball_pos: Cue ball centre, table inches.
        direction_deg: Aim direction in table space -- 0 is +x, increasing
            counter-clockwise.
        power: 0-100, mapped to an initial speed by
            :func:`physics.models.power_to_velocity`.
        other_balls: Object balls to simulate against. Any ball whose
            ``table_pos`` is ``None`` is dropped with a warning: treating it as
            being at the origin would put a phantom ball in the corner pocket.
        settings: Config. Defaults to the global settings.
        spin: Side spin in rad/s, positive counter-clockwise from above.
        config: Simulation limits. Defaults to the active accuracy profile.

    Returns:
        A :class:`~app.models.ShotPrediction` in table coordinates throughout.
        ``confidence`` reflects how much the result should be trusted -- it falls
        with collision depth, and falls sharply if a limit was hit and the
        prediction is therefore truncated.
    """
    settings = settings or get_settings()
    config = config or get_sim_config(settings)
    physics = _physics_from_settings(settings)
    geometry = TableGeometry.from_settings(settings)
    table = cushion_table_for(geometry)
    deceleration = physics.rolling_friction

    cue = SimBall(
        id="cue",
        position=cue_ball_pos,
        velocity=Vec2(
            math.cos(math.radians(direction_deg)) * power_to_velocity(power),
            math.sin(math.radians(direction_deg)) * power_to_velocity(power),
        ),
        spin=spin,
        is_cue=True,
    )
    _append_path(cue)

    balls: list[SimBall] = [cue]
    skipped = 0
    for ball in other_balls or []:
        if ball.pocketed:
            continue
        if ball.table_pos is None:
            skipped += 1
            continue
        sim = SimBall(id=ball.id, position=ball.table_pos, is_cue=False)
        _append_path(sim)
        balls.append(sim)

    if skipped:
        logger.warning(
            "%d ball(s) had no table position and were excluded from the simulation", skipped
        )

    impacts: list[ImpactEvent] = []
    pocketed: list[str] = []
    elapsed = 0.0
    events = 0
    truncated = False

    while events < MAX_EVENTS:
        event = _next_event(balls, table, physics, deceleration)
        if event is None:
            break  # everything has stopped

        if elapsed + event.time > config.max_sim_seconds:
            # Out of simulated time. Advance to the limit so the paths end
            # somewhere sensible, then stop and mark the result truncated.
            _advance(balls, config.max_sim_seconds - elapsed, deceleration)
            elapsed = config.max_sim_seconds
            truncated = True
            break

        _advance(balls, event.time, deceleration)
        elapsed += event.time
        events += 1

        ball = balls[event.ball_index]

        if event.kind == EventKind.STOP:
            ball.velocity = Vec2(0.0, 0.0)

        elif event.kind == EventKind.POCKET:
            ball.pocketed = True
            ball.velocity = Vec2(0.0, 0.0)
            pocketed.append(ball.id)
            centre = table.pocket_centers[event.detail]
            ball.position = Vec2(float(centre[0]), float(centre[1]))
            _append_path(ball)

        elif event.kind == EventKind.CUSHION:
            incoming = ball.heading_deg()
            if apply_cushion_bounce(ball, geometry, physics, axis=event.detail):
                impacts.append(
                    ImpactEvent(
                        position=ball.position,
                        target_id=None,
                        incoming_angle_deg=incoming,
                        outgoing_angle_deg=ball.heading_deg(),
                        is_cushion=True,
                        time_offset=elapsed,
                    )
                )

        elif event.kind == EventKind.BALL:
            target = balls[event.other_index]
            incoming = ball.heading_deg()
            resolve_ball_collision(ball, target, physics)
            impacts.append(
                ImpactEvent(
                    position=ball.position,
                    target_id=target.id,
                    incoming_angle_deg=incoming,
                    outgoing_angle_deg=ball.heading_deg(),
                    is_cushion=False,
                    time_offset=elapsed,
                )
            )
            _append_path(ball)
            _append_path(target)

            # FAST stops after the first ball-ball contact: the aiming line and
            # the object ball's departure are already determined, and the rest
            # is the expensive part.
            if not config.simulate_secondary:
                break
            if sum(1 for i in impacts if not i.is_cushion) >= config.max_collision_depth:
                truncated = True
                break

        _resolve_overlaps(balls, physics, impacts, elapsed)

    if events >= MAX_EVENTS:
        logger.warning("shot simulation hit the %d-event cap; result truncated", MAX_EVENTS)
        truncated = True

    return _build_prediction(balls, impacts, pocketed, elapsed, truncated, skipped, config)


def _build_prediction(
    balls: list[SimBall],
    impacts: list[ImpactEvent],
    pocketed: list[str],
    elapsed: float,
    truncated: bool,
    skipped: int,
    config: SimConfig,
) -> ShotPrediction:
    """Assemble the result and score how much to trust it."""
    cue = next((b for b in balls if b.is_cue), None)

    ball_paths = {
        b.id: list(b.path) for b in balls if not b.is_cue and len(b.path) > 1
    }
    final_positions = {b.id: b.position for b in balls}

    # Confidence. Each ball-ball contact compounds real-world effects this model
    # omits -- throw, spin transfer, cloth variation -- so certainty decays with
    # collision depth rather than being flat.
    ball_contacts = sum(1 for i in impacts if not i.is_cushion)
    confidence = 0.95**ball_contacts
    if truncated:
        # A truncated path is not a prediction of where things end up, and must
        # not be presented as one.
        confidence *= 0.5
    if skipped:
        confidence *= 0.7

    return ShotPrediction(
        trajectory_path=list(cue.path) if cue else [],
        ball_paths=ball_paths,
        impact_points=impacts,
        final_positions=final_positions,
        pocketed_ball_ids=pocketed,
        time_to_settle=elapsed,
        confidence=float(min(1.0, max(0.0, confidence))),
    )


def _physics_from_settings(settings: Settings) -> BallPhysics:
    """Build the physics parameters from config.

    Every value is user-tunable, so a table with unusually lively rails or new
    cloth can be matched without touching code.
    """
    return BallPhysics(
        cushion_restitution=settings.physics.cushion_restitution,
        ball_restitution=settings.physics.ball_restitution,
        rolling_friction=settings.physics.rolling_friction,
    )


# ---------------------------------------------------------------------------
# Vision -> physics bridge
# ---------------------------------------------------------------------------


def estimate_shot_from_cue(
    game_state: GameState,
    cue_stick: CueStick,
    power: float | None = None,
    settings: Settings | None = None,
    cache: PredictionCache | None = None,
) -> ShotPrediction | None:
    """Turn observed cue geometry into a shot prediction.

    The bridge between vision and physics, and where two real problems are
    handled rather than hidden.

    **Power is not observable from a single frame.** Until cue-tip velocity
    tracking lands, ``settings.physics.default_power`` is used. That is fine for
    an aiming *line*, whose direction is what matters, and wrong for predicting
    resting positions -- so the returned confidence is reduced to say so.

    **The aim line is not the cue line.** The cue points at a spot on the cue
    ball, and the ball leaves along the line from the cue ball's *centre*
    through that contact point. For a centre-ball hit the two coincide; off
    centre they diverge, and using the raw stick angle puts the prediction
    visibly off. When the cue tip's table position is known the aim is taken
    from the geometry; otherwise the stick angle is used directly, with a
    further confidence penalty.

    Returns:
        ``None`` when the state cannot support a prediction -- no cue ball, or no
        homography so ``table_pos`` is missing. Callers should keep showing the
        previous overlay rather than blanking it.
    """
    settings = settings or get_settings()

    if game_state.cue_ball is None or game_state.cue_ball.table_pos is None:
        return None

    cue_pos = game_state.cue_ball.table_pos
    aim_deg = cue_stick.angle_deg
    aim_is_geometric = False

    tip = cue_stick.tip_table_pos
    if tip is not None:
        offset = cue_pos - tip
        if offset.length() > 1e-6:
            # Aim along tip -> cue-ball-centre, which accounts for where on the
            # ball the cue is actually pointing.
            aim_deg = math.degrees(math.atan2(offset.y, offset.x))
            aim_is_geometric = True

    power_measured = power is not None
    if power is None:
        power = float(settings.physics.default_power)

    object_balls = game_state.object_balls()
    if cache is not None:
        # Cached predictions are shared objects, so copy before scaling the
        # confidence below -- mutating a cache entry would corrupt every later
        # hit on it.
        from dataclasses import replace as _replace

        prediction = _replace(
            simulate_shot_cached(
                cache, cue_pos, aim_deg, power, object_balls, settings
            )
        )
    else:
        prediction = simulate_shot(
            cue_ball_pos=cue_pos,
            direction_deg=aim_deg,
            power=power,
            other_balls=object_balls,
            settings=settings,
        )

    # Fold in how much the *inputs* deserve trust, on top of how much the
    # simulation itself does.
    scale = 1.0
    if not power_measured:
        scale *= 0.75
    if not aim_is_geometric:
        scale *= 0.85
    scale *= max(0.3, cue_stick.confidence)
    prediction.confidence = float(min(1.0, prediction.confidence * scale))
    return prediction


# ---------------------------------------------------------------------------
# Configuration and caching
# ---------------------------------------------------------------------------


def get_sim_config(settings: Settings | None = None) -> SimConfig:
    """Resolve the accuracy setting to a concrete :class:`SimConfig`."""
    settings = settings or get_settings()
    profile = ACCURACY_PROFILES[settings.physics.accuracy]

    # Explicit YAML overrides beat the profile: someone tuning against real
    # hardware should not have their values silently replaced by a preset.
    defaults = type(settings.physics)()
    if (
        settings.physics.max_sim_seconds != defaults.max_sim_seconds
        or settings.physics.max_collision_depth != defaults.max_collision_depth
    ):
        profile = SimConfig(
            timestep=profile.timestep,
            max_sim_seconds=settings.physics.max_sim_seconds,
            max_collision_depth=settings.physics.max_collision_depth,
            simulate_secondary=profile.simulate_secondary,
            path_decimation=profile.path_decimation,
        )
    return profile


class PredictionCache:
    """Caches predictions across frames, keyed on quantised inputs.

    The reason this exists: the aiming line is recomputed every frame while the
    player lines up a shot, but between two frames 33 ms apart the cue has
    usually not moved enough to change the answer. Quantising the inputs and
    reusing the previous result turns most frames into a dictionary lookup.

    It also *stabilises the overlay*. Detection noise jitters the measured cue
    angle by a fraction of a degree per frame, and recomputing on every wobble
    makes the projected line shimmer. Quantising the angle removes that at the
    source, which is cheaper and more effective than smoothing it afterwards.

    Quantisation steps are deliberately coarse relative to the accuracy the
    physics can actually deliver: a quarter inch of cue-ball position and half a
    degree of aim are both well below what a player can perceive on the cloth.
    """

    def __init__(
        self,
        max_entries: int = 64,
        position_step_in: float = 0.25,
        angle_step_deg: float = 0.5,
        power_step: float = 5.0,
    ) -> None:
        self.max_entries = max_entries
        self.position_step_in = position_step_in
        self.angle_step_deg = angle_step_deg
        self.power_step = power_step
        self._entries: dict[tuple, ShotPrediction] = {}
        self.hits = 0
        self.misses = 0

    def key(
        self,
        cue_pos: Vec2,
        direction_deg: float,
        power: float,
        other_balls: list[Ball] | None,
    ) -> tuple:
        """Quantised cache key.

        The object-ball layout is part of the key, coarsely quantised. Leaving it
        out would be a correctness bug: the same aim through a different rack
        gives a completely different result.
        """
        layout = tuple(
            sorted(
                (
                    round(b.table_pos.x / self.position_step_in),
                    round(b.table_pos.y / self.position_step_in),
                )
                for b in (other_balls or [])
                if b.table_pos is not None and not b.pocketed
            )
        )
        return (
            round(cue_pos.x / self.position_step_in),
            round(cue_pos.y / self.position_step_in),
            round(direction_deg / self.angle_step_deg),
            round(power / self.power_step),
            layout,
        )

    def get(self, key: tuple) -> ShotPrediction | None:
        prediction = self._entries.get(key)
        if prediction is None:
            self.misses += 1
        else:
            self.hits += 1
        return prediction

    def put(self, key: tuple, prediction: ShotPrediction) -> None:
        if len(self._entries) >= self.max_entries:
            # Plain FIFO eviction. A true LRU would need access bookkeeping for
            # no measurable benefit at this size -- the working set while
            # someone lines up one shot is a handful of entries.
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = prediction

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0


def simulate_shot_cached(
    cache: PredictionCache,
    cue_ball_pos: Vec2,
    direction_deg: float,
    power: float,
    other_balls: list[Ball] | None = None,
    settings: Settings | None = None,
) -> ShotPrediction:
    """:func:`simulate_shot` with the cross-frame cache applied."""
    key = cache.key(cue_ball_pos, direction_deg, power, other_balls)
    cached = cache.get(key)
    if cached is not None:
        return cached
    prediction = simulate_shot(
        cue_ball_pos, direction_deg, power, other_balls, settings
    )
    cache.put(key, prediction)
    return prediction


def ghost_ball_position(
    target_pos: Vec2, pocket_pos: Vec2, physics: BallPhysics | None = None
) -> Vec2:
    """Where the cue ball must be at contact to send a ball toward a pocket.

    The classic aiming construction, and useful well beyond drawing a marker:
    training mode needs it to generate a drill's ideal shot, and the renderer
    needs it for the ghost-ball outline. The object ball departs along the line
    of centres, so the cue ball's centre at contact sits one ball diameter back
    from the target along the line from the pocket through the target.
    """
    physics = physics or BallPhysics()
    offset = target_pos - pocket_pos
    length = offset.length()
    if length < 1e-9:
        # Target sitting in the pocket; no meaningful aim line.
        return target_pos
    return target_pos + offset.scaled(physics.contact_distance / length)


def aim_angle_for_pocket(
    cue_pos: Vec2,
    target_pos: Vec2,
    pocket_pos: Vec2,
    physics: BallPhysics | None = None,
) -> float:
    """Aim direction, in table degrees, to pot ``target_pos`` into ``pocket_pos``."""
    ghost = ghost_ball_position(target_pos, pocket_pos, physics)
    offset = ghost - cue_pos
    return math.degrees(math.atan2(offset.y, offset.x))
