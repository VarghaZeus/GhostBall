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

__all__ = ["BallPhysics", "SimBall", "TableGeometry", "SimConfig", "ACCURACY_PROFILES"]


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
    """
    return 20.0 + (power_pct / 100.0) * 280.0
