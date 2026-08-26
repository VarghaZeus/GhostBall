"""Physics parameters and simulation state.

Kept separate from :mod:`app.models` because these are *simulation* objects with
velocity and spin, not *observation* objects. A :class:`~app.models.Ball` is
what vision saw; a :class:`SimBall` is what the simulator is moving. Conflating
them is how you end up accidentally writing predicted positions back over
measured ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.config import (
    BALL_RADIUS_IN,
    DEFAULT_BALL_RESTITUTION,
    DEFAULT_CUSHION_RESTITUTION,
    DEFAULT_ROLLING_FRICTION,
)
from app.models import PhysicsAccuracy, Vec2

__all__ = [
    "BallPhysics",
    "SimBall",
    "TableGeometry",
    "SimConfig",
    "ACCURACY_PROFILES",
    "power_to_velocity",
    "speed_for_distance",
    "power_for_distance",
    "power_for_table_lengths",
    "MAX_TIP_OFFSET",
]


@dataclass(frozen=True, slots=True)
class BallPhysics:
    """Physical constants of a pool ball. Frozen -- these describe the world."""

    radius_in: float = BALL_RADIUS_IN
    mass_oz: float = 5.75  # regulation 5.5-6.0 oz; only ratios matter here
    cushion_restitution: float = DEFAULT_CUSHION_RESTITUTION
    ball_restitution: float = DEFAULT_BALL_RESTITUTION
    rolling_friction: float = DEFAULT_ROLLING_FRICTION  # in/s^2 deceleration

    #: Fraction of tangential velocity lost in a cushion hit. Real rails grip,
    #: so a ball hitting at an angle comes off shallower than pure reflection
    #: predicts -- ignoring this is the main reason naive bank-shot predictions
    #: are visibly wrong.
    cushion_tangential_loss: float = 0.15

    @property
    def contact_distance(self) -> float:
        """Centre separation at which two balls touch."""
        return self.radius_in * 2.0


@dataclass(slots=True)
class SimBall:
    """A ball inside the simulation.

    ``id`` matches the observed :class:`~app.models.Ball` it came from, so
    results can be mapped back to what vision is tracking.
    """

    id: str
    position: Vec2  # table inches
    velocity: Vec2 = Vec2(0.0, 0.0)  # inches/sec
    #: Spin about the vertical axis (side/english), rad/s. Positive is
    #: counter-clockwise seen from above. Top/bottom spin is deliberately not
    #: modelled -- it needs a 3-D model to be worth anything, and the visible
    #: benefit for shot prediction is small next to getting cut angles right.
    spin: float = 0.0
    #: Follow/draw, as a dimensionless fraction of the impact speed that is
    #: returned along the line of centres after a ball-ball contact. Positive is
    #: follow (top spin, cue ball carries on through), negative is draw (bottom
    #: spin, cue ball comes back). ``0.0`` is a struck-centre stun and is a
    #: strict no-op -- see :func:`physics.simulator.resolve_ball_collision`.
    #:
    #: Not an angular velocity, and not the same kind of quantity as
    #: :attr:`spin` above. Modelling this properly needs the sliding-to-rolling
    #: transition in three dimensions; this is a single coefficient tuned to
    #: reproduce the one thing a player watches for, which is whether the cue
    #: ball comes back off a full hit. Anything read off it beyond "does the cue
    #: ball go forwards or backwards, and roughly how far" is over-reading it.
    vertical_spin: float = 0.0
    is_cue: bool = False
    pocketed: bool = False
    #: Positions visited during simulation, for the renderer's polyline.
    path: list[Vec2] = field(default_factory=list)

    @property
    def speed(self) -> float:
        return self.velocity.length()

    @property
    def is_moving(self) -> bool:
        """Whether this ball still needs stepping. Threshold is deliberately
        loose -- below ~0.5 in/s a ball is visually stopped and continuing to
        integrate it just burns frame time."""
        return not self.pocketed and self.speed > 0.5

    def heading_deg(self) -> float:
        """Direction of travel in degrees, table space. ``0.0`` when stopped."""
        if self.speed == 0.0:
            return 0.0
        return math.degrees(math.atan2(self.velocity.y, self.velocity.x))


@dataclass(frozen=True, slots=True)
class TableGeometry:
    """Playing-surface geometry in table inches, precomputed for the simulator.

    The cushion bounds are inset by a ball radius so that collision tests can
    compare ball *centres* against them directly, rather than adding the radius
    at every one of the four comparisons in the inner loop.
    """

    length_in: float
    width_in: float
    pocket_radius_in: float
    ball_radius_in: float = BALL_RADIUS_IN

    @property
    def x_min(self) -> float:
        return self.ball_radius_in

    @property
    def x_max(self) -> float:
        return self.length_in - self.ball_radius_in

    @property
    def y_min(self) -> float:
        return self.ball_radius_in

    @property
    def y_max(self) -> float:
        return self.width_in - self.ball_radius_in

    def pocket_centers(self) -> list[Vec2]:
        """The six pocket mouths, clockwise from top-left."""
        return [
            Vec2(0.0, 0.0),
            Vec2(self.length_in / 2.0, 0.0),
            Vec2(self.length_in, 0.0),
            Vec2(self.length_in, self.width_in),
            Vec2(self.length_in / 2.0, self.width_in),
            Vec2(0.0, self.width_in),
        ]

    def contains(self, point: Vec2) -> bool:
        """Whether a ball centre is within the cushions."""
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max

    @classmethod
    def from_settings(cls, settings: object) -> TableGeometry:
        """Build from an :class:`app.config.Settings`."""
        table = settings.table  # type: ignore[attr-defined]
        return cls(
            length_in=table.length_in,
            width_in=table.width_in,
            pocket_radius_in=table.pocket_radius_in,
        )


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Per-run simulator tuning, derived from the accuracy setting."""

    timestep: float
    max_sim_seconds: float
    max_collision_depth: int
    #: Whether to simulate object balls after the first impact. Off in FAST,
    #: which halves the work in a crowded rack.
    simulate_secondary: bool = True
    #: Keep only every Nth integration point in the output polyline. At a 4 ms
    #: step a 3-second shot is 750 points -- far more than the renderer needs,
    #: and building that list is a real allocation cost.
    path_decimation: int = 8

    @property
    def max_steps(self) -> int:
        return int(self.max_sim_seconds / self.timestep)


#: Concrete parameters behind each user-visible accuracy setting. FAST trades a
#: 4x coarser step for roughly 4x less work, which shows up as slightly wrong
#: rebound points but keeps the frame budget on a loaded Pi.
ACCURACY_PROFILES: dict[PhysicsAccuracy, SimConfig] = {
    PhysicsAccuracy.FAST: SimConfig(
        timestep=0.016,
        max_sim_seconds=4.0,
        max_collision_depth=1,
        simulate_secondary=False,
        path_decimation=2,
    ),
    PhysicsAccuracy.BALANCED: SimConfig(
        timestep=0.004,
        max_sim_seconds=8.0,
        max_collision_depth=3,
        simulate_secondary=True,
        path_decimation=8,
    ),
    PhysicsAccuracy.ACCURATE: SimConfig(
        timestep=0.001,
        max_sim_seconds=12.0,
        max_collision_depth=8,
        simulate_secondary=True,
        path_decimation=32,
    ),
}


def power_to_velocity(power_pct: float) -> float:
    """Map a 0-100 power value to an initial cue-ball speed in inches/sec.

    Anchored at the two ends that matter: a soft touch shot is around 20 in/s,
    and a full-power break is around 300 in/s (~25 ft/s, the high end of
    measured amateur breaks). Linear between them is not physically principled,
    but power is a UI abstraction rather than a measurement, so a curve would
    only make the slider feel less predictable.

    **This scale is much wider than the playable range.** Against the default
    12 in/s^2 deceleration, power 100 free-rolls 3750 inches -- about 49 table
    lengths -- and every positional shot lives below power 30. That is not a
    bug to be rescaled: the mapping is a fixed input contract, and moving it
    would silently change every prediction that has ever been checked against
    a real table. Callers that care how *far* the ball goes should not pick a
    power at all; they should name a distance and use
    :func:`power_for_distance`.
    """
    return 20.0 + (power_pct / 100.0) * 280.0


def speed_for_distance(distance_in: float, deceleration: float) -> float:
    """Initial speed that free-rolls a ball exactly ``distance_in`` inches.

    The inverse of ``distance = speed^2 / (2*a)``, which is the whole of the
    roll model (see the module docstring in :mod:`physics.simulator`). "Free
    roll" means no contact: a ball struck straight into another one stops dead
    at the contact point, so travel on a straight shot is the wrong anchor for
    a power level -- it measures zero regardless of how hard the shot was hit.
    """
    if distance_in <= 0.0:
        return 0.0
    return math.sqrt(2.0 * deceleration * distance_in)


def power_for_distance(distance_in: float, deceleration: float) -> float:
    """The 0-100 power value that free-rolls a ball ``distance_in`` inches.

    Bridges a physically meaningful distance back onto the power scale that
    :func:`simulate_shot` takes. Deliberately **not clamped to 0-100**: a
    caller asking for a distance the scale cannot express should see a value
    outside the range rather than a silently clipped shot that travels the
    wrong distance. Practically this never fires -- the scale reaches 49 table
    lengths -- but a friction retune could make it possible, and a wrong
    distance is exactly the failure this whole mechanism exists to avoid.
    """
    speed = speed_for_distance(distance_in, deceleration)
    return (speed - 20.0) / 2.8


def power_for_table_lengths(
    table_lengths: float, table_length_in: float, deceleration: float
) -> float:
    """:func:`power_for_distance` for a distance given in table lengths."""
    return power_for_distance(table_lengths * table_length_in, deceleration)


#: How far off centre a tip contact can usefully be prescribed, as a fraction of
#: the ball's radius.
#:
#: The limit is the miscue, not the geometry. A tip placed much beyond half a
#: radius slides off the ball instead of gripping it -- the classic
#: "half-tip-of-english maximum" -- so offsets past this describe a shot that
#: does not happen. Prescribing one in a drill would be teaching a miscue, so
#: :func:`tip_offset_to_spin` clamps rather than trusting the caller.
MAX_TIP_OFFSET = 0.5


def tip_offset_to_spin(
    tip_offset: Vec2 | None, speed: float, physics: BallPhysics | None = None
) -> tuple[float, float]:
    """Convert a prescribed tip contact point into the two spin quantities.

    ``tip_offset`` is in ball radii from centre: ``x`` positive is right-hand
    english, ``y`` positive is top (follow). Returns ``(side_spin_rad_s,
    vertical_spin)`` ready for :class:`SimBall`.

    The two axes are different kinds of quantity, which is why this returns a
    pair rather than a vector. Side spin is a real angular velocity that the
    cushion model integrates against; vertical spin is a dimensionless
    coefficient the ball-collision model uses. Combining them into one Vec2
    would invite treating them as interchangeable.

    Side spin scales with speed because a harder stroke at the same tip offset
    imparts proportionally more rotation -- a fixed rad/s would make soft shots
    spin absurdly. Vertical spin does not, because it is already expressed as a
    fraction *of* the impact speed.
    """
    if tip_offset is None:
        return 0.0, 0.0
    physics = physics or BallPhysics()
    x = max(-MAX_TIP_OFFSET, min(MAX_TIP_OFFSET, tip_offset.x))
    y = max(-MAX_TIP_OFFSET, min(MAX_TIP_OFFSET, tip_offset.y))

    # Rolling without slipping gives omega = v/r; an offset of one radius is the
    # scale at which the tip would impart about that much spin, so the offset
    # fraction times v/r is the natural first-order estimate. Negated because a
    # right-hand hit (+x) spins the ball clockwise from above, and `spin` is
    # positive counter-clockwise.
    # `+ 0.0` normalises the signed zero that `-x * ...` produces at x == 0.
    # Harmless arithmetically, but a centre-ball hit returning -0.0 makes the
    # "zero offset changes nothing" guarantee something a reader has to reason
    # about rather than read.
    side = (-x * speed / physics.radius_in + 0.0) if speed > 0.0 else 0.0
    return side, y
