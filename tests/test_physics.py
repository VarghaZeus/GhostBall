"""Physics tests against known analytic outcomes.

Physics is testable in a way computer vision is not: the right answer can be
derived on paper. So these tests assert against closed-form results and against
the standard billiards facts a player would recognise, not against whatever the
code currently returns.

The three that matter most, in order:

**The 90-degree rule.** On any cut, the object ball leaves along the line of
centres and the cue ball leaves along the tangent, so the two separate by 90
degrees. Players judge this by eye on every shot, which makes it the most
visible possible error.

**Newton's cradle.** A full-ball head-on hit stops the cue ball dead and sends
the object ball off with the cue ball's speed.

**Free-roll distance.** ``s^2 / (2a)``, exactly.

A note on measurement, learned the hard way while writing these: a ball's
*departure direction* is the first segment of its path, not the vector from its
start to its final resting place. Those differ as soon as it touches a cushion,
and conflating them produced a convincing-looking wrong answer twice during
development.
"""

from __future__ import annotations

import math
import time

import pytest

from app.config import BALL_DIAMETER_IN, Settings
from app.models import (
    Ball,
    BallKind,
    CueStick,
    GameState,
    PhysicsAccuracy,
    TableBoundary,
    Vec2,
)
from physics.models import (
    ACCURACY_PROFILES,
    BallPhysics,
    SimBall,
    TableGeometry,
    power_to_velocity,
)
from physics.simulator import (
    MAX_EVENTS,
    PredictionCache,
    aim_angle_for_pocket,
    apply_cushion_bounce,
    check_pocketed,
    cushion_table_for,
    distance_at_time,
    estimate_shot_from_cue,
    get_sim_config,
    ghost_ball_position,
    predict_collision,
    ray_circle_entry,
    ray_cushion_exit,
    resolve_ball_collision,
    simulate_shot,
    simulate_shot_cached,
    time_for_distance,
    total_roll_distance,
)


@pytest.fixture()
def settings() -> Settings:
    s = Settings()
    s.camera.use_mock = True
    s.projector.use_mock = True
    return s


def _object_ball(ball_id: str, x: float, y: float) -> Ball:
    """An object ball at a table position, as vision would report it."""
    return Ball(
        id=ball_id,
        center_px=Vec2(0.0, 0.0),
        radius_px=10.0,
        kind=BallKind.SOLID,
        table_pos=Vec2(x, y),
    )


def _departure_angle(path: list[Vec2]) -> float | None:
    """Direction of a path's first segment, in degrees.

    Deliberately the *first* segment. Using start-to-end gives the wrong answer
    the moment the ball touches a cushion.
    """
    if len(path) < 2:
        return None
    delta = path[1] - path[0]
    if delta.length() < 1e-9:
        return None
    return math.degrees(math.atan2(delta.y, delta.x))


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def test_roll_distance_closed_form() -> None:
    """``s^2/(2a)``, the foundation everything else is timed against."""
    assert total_roll_distance(100.0, 12.5) == pytest.approx(400.0)
    assert total_roll_distance(0.0, 12.0) == 0.0
    assert total_roll_distance(50.0, 0.0) == math.inf


def test_time_for_distance_inverts_distance_at_time() -> None:
    speed, decel = 120.0, 12.0
    for target in (1.0, 25.0, 100.0, 500.0):
        t = time_for_distance(speed, decel, target)
        if math.isfinite(t):
            assert distance_at_time(speed, decel, t) == pytest.approx(target, abs=1e-6)


def test_time_for_unreachable_distance_is_infinite() -> None:
    """A ball that stops short must report ``inf``, not a complex root.

    This is what stops the event solver from scheduling a collision the ball
    never actually reaches.
    """
    assert total_roll_distance(50.0, 12.0) < 200.0
    assert time_for_distance(50.0, 12.0, 200.0) == math.inf


def test_distance_at_time_clamps_after_stopping() -> None:
    """Past the stopping time the ball must not creep, and definitely must not
    reverse -- which the raw quadratic would do."""
    speed, decel = 60.0, 12.0
    stop_time = speed / decel
    settled = distance_at_time(speed, decel, stop_time)
    assert distance_at_time(speed, decel, stop_time * 3) == pytest.approx(settled)


def test_power_maps_monotonically_to_speed() -> None:
    speeds = [power_to_velocity(p) for p in (0, 25, 50, 75, 100)]
    assert speeds == sorted(speeds)


# ---------------------------------------------------------------------------
# Ray intersection primitives
# ---------------------------------------------------------------------------


def test_ray_circle_entry_head_on() -> None:
    import numpy as np

    origin = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])
    centers = np.array([[10.0, 0.0]])
    index, distance = ray_circle_entry(origin, direction, centers, 2.25, 100.0)
    assert index == 0
    # Entry is one contact distance short of the centre.
    assert distance == pytest.approx(10.0 - 2.25)


def test_ray_circle_entry_misses_and_ignores_behind() -> None:
    import numpy as np

    origin = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])
    assert ray_circle_entry(origin, direction, np.array([[10.0, 5.0]]), 2.25, 100.0)[0] == -1
    # A circle behind the origin must not be reported -- the ball is not
    # reversing.
    assert ray_circle_entry(origin, direction, np.array([[-10.0, 0.0]]), 2.25, 100.0)[0] == -1


def test_ray_circle_entry_picks_the_nearest() -> None:
    import numpy as np

    centers = np.array([[30.0, 0.0], [10.0, 0.0], [50.0, 0.0]])
    index, distance = ray_circle_entry(
        np.array([0.0, 0.0]), np.array([1.0, 0.0]), centers, 2.25, 100.0
    )
    assert index == 1
    assert distance == pytest.approx(10.0 - 2.25)


def test_ray_cushion_exit_finds_the_right_rail(settings: Settings) -> None:
    import numpy as np

    geometry = TableGeometry.from_settings(settings)
    table = cushion_table_for(geometry)
    # Heading in +x from the middle must exit at x_max, which is axis 1.
    axis, distance = ray_cushion_exit(
        np.array([38.0, 19.0]), np.array([1.0, 0.0]), table.bounds, 1000.0
    )
    assert axis == 1
    assert distance == pytest.approx(geometry.x_max - 38.0)


# ---------------------------------------------------------------------------
# Collision primitives
# ---------------------------------------------------------------------------


def test_predict_collision_analytic() -> None:
    """Solved in closed form, so it can be checked exactly."""
    result = predict_collision(Vec2(0.0, 0.0), Vec2(100.0, 0.0), Vec2(10.0, 0.0))
    assert result.will_collide
    assert result.point.x == pytest.approx(10.0 - BALL_DIAMETER_IN)
    assert result.point.y == pytest.approx(0.0)
    assert result.time_to_impact == pytest.approx((10.0 - BALL_DIAMETER_IN) / 100.0)
    # Line of centres is straight ahead for a full-ball hit.
    assert result.normal_angle_deg == pytest.approx(0.0)


def test_predict_collision_grazing_edge() -> None:
    """A ball offset by exactly one contact distance grazes; beyond that it misses.

    The boundary case is worth pinning because an off-by-a-radius here would
    make every thin cut either impossible or guaranteed.
    """
    just_touching = predict_collision(
        Vec2(0.0, 0.0), Vec2(100.0, 0.0), Vec2(10.0, BALL_DIAMETER_IN - 1e-6)
    )
    assert just_touching.will_collide

    clear_miss = predict_collision(
        Vec2(0.0, 0.0), Vec2(100.0, 0.0), Vec2(10.0, BALL_DIAMETER_IN + 0.01)
    )
    assert not clear_miss.will_collide


def test_predict_collision_stationary_ball_never_collides() -> None:
    result = predict_collision(Vec2(0.0, 0.0), Vec2(0.0, 0.0), Vec2(5.0, 0.0))
    assert not result.will_collide


def test_head_on_collision_transfers_all_velocity() -> None:
    """Newton's cradle. Equal masses, full-ball hit: the cue ball stops dead."""
    cue = SimBall(id="cue", position=Vec2(0.0, 0.0), velocity=Vec2(100.0, 0.0), is_cue=True)
    target = SimBall(id="t", position=Vec2(BALL_DIAMETER_IN, 0.0))
    physics = BallPhysics(ball_restitution=1.0)

    resolve_ball_collision(cue, target, physics)

    assert cue.speed == pytest.approx(0.0, abs=1e-9)
    assert target.velocity.x == pytest.approx(100.0)
    assert target.velocity.y == pytest.approx(0.0)


def test_collision_conserves_momentum_along_the_normal() -> None:
    """With restitution 1.0 the normal-direction momentum must be conserved."""
    cue = SimBall(id="cue", position=Vec2(0.0, 0.0), velocity=Vec2(80.0, 30.0), is_cue=True)
    target = SimBall(id="t", position=Vec2(BALL_DIAMETER_IN * 0.8, BALL_DIAMETER_IN * 0.6))
    before = cue.velocity.x + target.velocity.x

    resolve_ball_collision(cue, target, BallPhysics(ball_restitution=1.0))

    assert cue.velocity.x + target.velocity.x == pytest.approx(before, rel=1e-9)


def test_separating_balls_are_not_resolved() -> None:
    """Resolving a separating pair would suck them together and make them buzz."""
    a = SimBall(id="a", position=Vec2(0.0, 0.0), velocity=Vec2(-50.0, 0.0))
    b = SimBall(id="b", position=Vec2(BALL_DIAMETER_IN, 0.0), velocity=Vec2(50.0, 0.0))
    resolve_ball_collision(a, b)
    assert a.velocity.x == pytest.approx(-50.0)
    assert b.velocity.x == pytest.approx(50.0)


def test_overlapping_balls_are_pushed_apart() -> None:
    """Otherwise the next solve re-detects the same contact and the loop stalls."""
    a = SimBall(id="a", position=Vec2(0.0, 0.0), velocity=Vec2(50.0, 0.0))
    b = SimBall(id="b", position=Vec2(BALL_DIAMETER_IN * 0.5, 0.0))
    resolve_ball_collision(a, b)
    assert a.position.distance_to(b.position) >= BALL_DIAMETER_IN


def test_cushion_bounce_reflects_the_normal_component(settings: Settings) -> None:
    """Angle in equals angle out, less restitution on the normal and grip on the
    tangential."""
    geometry = TableGeometry.from_settings(settings)
    physics = BallPhysics()
    ball = SimBall(id="b", position=Vec2(38.0, geometry.y_max), velocity=Vec2(60.0, 60.0))

    assert apply_cushion_bounce(ball, geometry, physics, axis=3)

    # Normal (y) reverses and is scaled by restitution.
    assert ball.velocity.y == pytest.approx(-60.0 * physics.cushion_restitution)
    # Tangential (x) is reduced by cushion grip, not preserved.
    assert ball.velocity.x < 60.0
    assert ball.velocity.x == pytest.approx(60.0 * (1.0 - physics.cushion_tangential_loss))


def test_cushion_bounce_repositions_inside_the_bounds(settings: Settings) -> None:
    """Reflecting the velocity but leaving the ball out of bounds makes it buzz
    along the rail, re-detecting the same cushion every solve."""
    geometry = TableGeometry.from_settings(settings)
    ball = SimBall(id="b", position=Vec2(geometry.x_max + 5.0, 19.0), velocity=Vec2(60.0, 0.0))
    apply_cushion_bounce(ball, geometry, axis=1)
    assert ball.position.x <= geometry.x_max
    assert ball.velocity.x < 0.0


def test_cushion_bounce_loses_energy(settings: Settings) -> None:
    geometry = TableGeometry.from_settings(settings)
    ball = SimBall(id="b", position=Vec2(38.0, geometry.y_min), velocity=Vec2(0.0, -80.0))
    before = ball.speed
    apply_cushion_bounce(ball, geometry, axis=2)
    assert ball.speed < before


def test_ball_in_a_pocket_is_pocketed(settings: Settings) -> None:
    geometry = TableGeometry.from_settings(settings)
    assert check_pocketed(SimBall(id="b", position=Vec2(0.0, 0.0)), geometry)
    assert not check_pocketed(SimBall(id="b", position=Vec2(38.0, 19.0)), geometry)


# ---------------------------------------------------------------------------
# The 90-degree rule -- the headline physics result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut_deg", [15, 30, 45, 60, 75])
def test_ninety_degree_rule(cut_deg: int, settings: Settings) -> None:
    """The object ball leaves along the line of centres; the cue ball leaves
    along the tangent. They must separate by 90 degrees at every cut angle.

    This is the number players judge by eye on every single shot, so it gets
    tested across the whole range rather than at one convenient angle.
    """
    offset = BALL_DIAMETER_IN * math.sin(math.radians(cut_deg))
    target = _object_ball("t", 40.0, 19.0 + offset)

    prediction = simulate_shot(Vec2(20.0, 19.0), 0.0, 40, [target], settings)

    contacts = [i for i in prediction.impact_points if not i.is_cushion]
    assert contacts, f"no ball contact at a {cut_deg} degree cut"

    object_departure = _departure_angle(prediction.ball_paths["t"])
    assert object_departure is not None
    # The object ball departs at exactly the cut angle, by construction.
    assert object_departure == pytest.approx(float(cut_deg), abs=0.5)

    cue_departure = contacts[0].outgoing_angle_deg
    separation = abs(((cue_departure - object_departure + 180.0) % 360.0) - 180.0)
    assert separation == pytest.approx(90.0, abs=1.0), (
        f"cut {cut_deg} deg: cue left at {cue_departure:.1f}, object at "
        f"{object_departure:.1f}, separation {separation:.1f} (expected 90)"
    )


def test_full_ball_hit_sends_object_ball_straight_on(settings: Settings) -> None:
    """A centre-to-centre hit must not deflect the object ball sideways."""
    target = _object_ball("t", 40.0, 19.0)
    prediction = simulate_shot(Vec2(20.0, 19.0), 0.0, 30, [target], settings)

    departure = _departure_angle(prediction.ball_paths["t"])
    assert departure == pytest.approx(0.0, abs=0.01)
    # And the cue ball must stop essentially dead at the contact point.
    contacts = [i for i in prediction.impact_points if not i.is_cushion]
    assert contacts[0].position.x == pytest.approx(40.0 - BALL_DIAMETER_IN, abs=0.01)


def test_thin_cut_barely_deflects_the_cue_ball(settings: Settings) -> None:
    """A very thin cut should send the object ball off at a steep angle while
    the cue ball carries on nearly straight."""
    offset = BALL_DIAMETER_IN * math.sin(math.radians(80))
    target = _object_ball("t", 40.0, 19.0 + offset)
    prediction = simulate_shot(Vec2(20.0, 19.0), 0.0, 40, [target], settings)

    contacts = [i for i in prediction.impact_points if not i.is_cushion]
    assert contacts
    assert abs(contacts[0].outgoing_angle_deg) < 15.0


# ---------------------------------------------------------------------------
# Free roll and cushions in the full simulation
# ---------------------------------------------------------------------------


def test_free_roll_distance_matches_closed_form(settings: Settings) -> None:
    """End-to-end check of the kinematics, on a shot that reaches no cushion."""
    power = 5
    expected = total_roll_distance(
        power_to_velocity(power), settings.physics.rolling_friction
    )
    prediction = simulate_shot(Vec2(2.0, 19.0), 0.0, power, [], settings)

    assert not [i for i in prediction.impact_points if i.is_cushion], "expected no cushion hit"
    travelled = prediction.trajectory_path[0].distance_to(prediction.trajectory_path[-1])
    assert travelled == pytest.approx(expected, abs=0.05)


def test_ball_always_comes_to_rest_inside_the_table(settings: Settings) -> None:
    """Whatever happens, no ball may end up off the table. A ball that escapes
    would put the physics and the projection permanently out of sync."""
    geometry = TableGeometry.from_settings(settings)
    for angle in range(0, 360, 17):
        prediction = simulate_shot(Vec2(38.0, 19.0), float(angle), 85, [], settings)
        final = prediction.final_positions["cue"]
        if prediction.pocketed_ball_ids:
            continue  # pocket centres sit outside the cushion bounds by design
        assert -0.5 <= final.x <= geometry.length_in + 0.5, f"escaped in x at {angle} deg"
        assert -0.5 <= final.y <= geometry.width_in + 0.5, f"escaped in y at {angle} deg"


def test_shot_along_a_rail_is_handled_cleanly(settings: Settings) -> None:
    """Rolling exactly along the cushion line is the numerically awkward case:
    the cushion solve is near-parallel, so a sign error there sends the ball off
    the table.

    Physically the ball drops into the side pocket, because a path hugging the
    cushion passes within a pocket radius of the mouth -- which is exactly what
    happens on a real table. Either outcome is acceptable; escaping is not.
    """
    geometry = TableGeometry.from_settings(settings)
    prediction = simulate_shot(Vec2(5.0, geometry.y_min), 0.0, 60, [], settings)

    if prediction.pocketed_ball_ids:
        assert "cue" in prediction.pocketed_ball_ids
        return
    final = prediction.final_positions["cue"]
    assert geometry.y_min - 0.5 <= final.y <= geometry.y_max + 0.5


def test_straight_at_a_corner_pocket_drops(settings: Settings) -> None:
    prediction = simulate_shot(
        Vec2(38.0, 19.0), math.degrees(math.atan2(-19.0, -38.0)), 60, [], settings
    )
    assert "cue" in prediction.pocketed_ball_ids


def test_pocketed_ball_stops_being_simulated(settings: Settings) -> None:
    """Once a ball drops it must not keep bouncing around inside the table."""
    target = _object_ball("t", 4.0, 4.0)
    prediction = simulate_shot(
        Vec2(38.0, 19.0), math.degrees(math.atan2(-15.0, -34.0)), 70, [target], settings
    )
    for ball_id in prediction.pocketed_ball_ids:
        # A pocketed ball is parked at the pocket centre.
        table = cushion_table_for(TableGeometry.from_settings(settings))
        final = prediction.final_positions[ball_id]
        distances = [
            math.hypot(final.x - c[0], final.y - c[1]) for c in table.pocket_centers
        ]
        assert min(distances) < 1.0


def test_cushion_rebound_is_symmetric(settings: Settings) -> None:
    """Equal-and-opposite entry angles must rebound to mirror-image exits."""
    up = simulate_shot(Vec2(38.0, 19.0), 30.0, 30, [], settings)
    down = simulate_shot(Vec2(38.0, 19.0), -30.0, 30, [], settings)

    up_hits = [i for i in up.impact_points if i.is_cushion]
    down_hits = [i for i in down.impact_points if i.is_cushion]
    assert up_hits and down_hits
    assert up_hits[0].outgoing_angle_deg == pytest.approx(
        -down_hits[0].outgoing_angle_deg, abs=0.5
    )


# ---------------------------------------------------------------------------
# Prediction structure and confidence
# ---------------------------------------------------------------------------


def test_prediction_paths_have_no_duplicate_points(settings: Settings) -> None:
    """Duplicate points give the renderer zero-length segments to draw and make
    any angle derived from consecutive points return atan2(0, 0)."""
    target = _object_ball("t", 40.0, 19.6)
    prediction = simulate_shot(Vec2(20.0, 19.0), 0.0, 60, [target], settings)

    for path in [prediction.trajectory_path, *prediction.ball_paths.values()]:
        for a, b in zip(path[:-1], path[1:], strict=True):
            assert a.distance_to(b) > 1e-3, f"duplicate path point at {a}"


def test_empty_prediction_for_zero_power(settings: Settings) -> None:
    """Power 0 still maps to a slow roll, so the path exists but is short."""
    prediction = simulate_shot(Vec2(38.0, 19.0), 0.0, 0, [], settings)
    assert not prediction.is_empty
    travelled = prediction.trajectory_path[0].distance_to(prediction.trajectory_path[-1])
    assert travelled < 30.0


def test_balls_without_table_positions_are_excluded(settings: Settings) -> None:
    """A ball with no table position must be dropped, not treated as being at
    the origin -- which would put a phantom ball in the corner pocket."""
    ghost = Ball(id="ghost", center_px=Vec2(0.0, 0.0), radius_px=10.0, kind=BallKind.SOLID)
    assert ghost.table_pos is None

    prediction = simulate_shot(Vec2(20.0, 19.0), 0.0, 40, [ghost], settings)
    assert "ghost" not in prediction.final_positions
    # And confidence is docked, because the simulation is knowingly incomplete.
    assert prediction.confidence < 1.0


def test_pocketed_balls_are_not_simulated(settings: Settings) -> None:
    sunk = _object_ball("sunk", 40.0, 19.0)
    sunk.pocketed = True
    prediction = simulate_shot(Vec2(20.0, 19.0), 0.0, 40, [sunk], settings)
    assert "sunk" not in prediction.final_positions
    assert not [i for i in prediction.impact_points if i.target_id == "sunk"]


def test_confidence_falls_with_collision_depth(settings: Settings) -> None:
    """Each contact compounds effects this model omits -- throw, spin transfer --
    so certainty must decay rather than stay flat."""
    clean = simulate_shot(Vec2(20.0, 19.0), 90.0, 10, [], settings)
    crowded = simulate_shot(
        Vec2(20.0, 19.0),
        0.0,
        70,
        [_object_ball(f"b{i}", 30.0 + i * 3.0, 19.0) for i in range(5)],
        settings,
    )
    assert clean.confidence > crowded.confidence


def test_confidence_is_bounded(settings: Settings) -> None:
    for power in (0, 25, 50, 75, 100):
        prediction = simulate_shot(Vec2(20.0, 19.0), 37.0, power, [], settings)
        assert 0.0 <= prediction.confidence <= 1.0


def test_time_to_settle_is_plausible(settings: Settings) -> None:
    """Settle time is the observable that calibrates the friction model. Real
    shots take a few seconds; anything near zero or near a minute means the
    deceleration is wrong."""
    for power in (20, 50, 90):
        prediction = simulate_shot(Vec2(20.0, 19.0), 33.0, power, [], settings)
        assert 0.5 < prediction.time_to_settle < 10.0, (
            f"power {power} settled in {prediction.time_to_settle:.1f}s"
        )


def test_simulation_terminates_on_a_crowded_table(settings: Settings) -> None:
    """A full rack must not run away to the event cap."""
    rack = [
        _object_ball(f"b{i}", 40.0 + (i % 5) * 2.6, 14.0 + (i // 5) * 2.6)
        for i in range(15)
    ]
    prediction = simulate_shot(Vec2(20.0, 19.0), 0.0, 100, rack, settings)
    assert len(prediction.impact_points) < MAX_EVENTS
    assert prediction.time_to_settle <= settings.physics.max_sim_seconds


# ---------------------------------------------------------------------------
# Accuracy profiles
# ---------------------------------------------------------------------------


def test_fast_profile_stops_after_the_first_contact(settings: Settings) -> None:
    """FAST exists to skip the expensive part once the aiming line is known."""
    settings.physics.accuracy = PhysicsAccuracy.FAST
    balls = [_object_ball("a", 40.0, 19.0), _object_ball("b", 55.0, 19.0)]
    prediction = simulate_shot(
        Vec2(20.0, 19.0), 0.0, 60, balls, settings, config=get_sim_config(settings)
    )
    assert len([i for i in prediction.impact_points if not i.is_cushion]) == 1


def test_profiles_trade_depth_for_speed() -> None:
    fast = ACCURACY_PROFILES[PhysicsAccuracy.FAST]
    accurate = ACCURACY_PROFILES[PhysicsAccuracy.ACCURATE]
    assert fast.max_collision_depth < accurate.max_collision_depth
    assert not fast.simulate_secondary
    assert accurate.simulate_secondary


def test_explicit_config_overrides_the_profile(settings: Settings) -> None:
    """Someone tuning against real hardware must not have their values silently
    replaced by a preset."""
    settings.physics.accuracy = PhysicsAccuracy.BALANCED
    settings.physics.max_collision_depth = 9
    assert get_sim_config(settings).max_collision_depth == 9


# ---------------------------------------------------------------------------
# Aiming helpers
# ---------------------------------------------------------------------------


def test_ghost_ball_sits_one_diameter_behind_the_target() -> None:
    """The classic aiming construction, used by training mode and the renderer."""
    target = Vec2(40.0, 19.0)
    pocket = Vec2(76.0, 19.0)
    ghost = ghost_ball_position(target, pocket)
    assert ghost.y == pytest.approx(19.0)
    assert ghost.x == pytest.approx(40.0 - BALL_DIAMETER_IN)
    assert ghost.distance_to(target) == pytest.approx(BALL_DIAMETER_IN)


def test_aim_angle_for_pocket_actually_pots_the_ball(settings: Settings) -> None:
    """The strongest possible test of the aiming maths: use the computed angle
    to take the shot and confirm the ball drops."""
    table = cushion_table_for(TableGeometry.from_settings(settings))
    pocket = Vec2(float(table.pocket_centers[2][0]), float(table.pocket_centers[2][1]))
    target_pos = Vec2(55.0, 12.0)
    cue_pos = Vec2(25.0, 24.0)

    angle = aim_angle_for_pocket(cue_pos, target_pos, pocket)
    prediction = simulate_shot(
        cue_pos, angle, 45, [_object_ball("t", target_pos.x, target_pos.y)], settings
    )
    assert "t" in prediction.pocketed_ball_ids, (
        f"aim of {angle:.2f} deg failed to pot the ball; "
        f"pocketed={prediction.pocketed_ball_ids}"
    )


def test_ghost_ball_handles_target_in_the_pocket() -> None:
    """Degenerate input must not divide by zero."""
    position = Vec2(10.0, 10.0)
    assert ghost_ball_position(position, position) == position


# ---------------------------------------------------------------------------
# The vision -> physics bridge
# ---------------------------------------------------------------------------


def _game_state_with_cue(settings: Settings, aim_deg: float) -> GameState:
    """A minimal game state good enough to aim from."""
    cue_ball = Ball(
        id="cue",
        center_px=Vec2(500.0, 400.0),
        radius_px=20.0,
        kind=BallKind.CUE,
        table_pos=Vec2(20.0, 19.0),
    )
    boundary = TableBoundary(
        Vec2(0, 0), Vec2(100, 0), Vec2(100, 50), Vec2(0, 50), Vec2(50, 25), 100, 50, 0.9
    )
    # Cue tip placed behind the ball along the aim line, so the geometric aim
    # path is exercised rather than the raw stick angle.
    tip = Vec2(
        20.0 - math.cos(math.radians(aim_deg)) * 6.0,
        19.0 - math.sin(math.radians(aim_deg)) * 6.0,
    )
    cue_stick = CueStick(
        tip_px=Vec2(400.0, 400.0),
        angle_deg=aim_deg,
        tip_table_pos=tip,
        shaft_visible=True,
        confidence=0.9,
    )
    return GameState(
        timestamp=1.0,
        frame_index=0,
        table_boundary=boundary,
        balls=[cue_ball, _object_ball("t", 40.0, 19.0)],
        cue_ball=cue_ball,
        cue_stick=cue_stick,
    )


def test_estimate_shot_from_cue_uses_geometric_aim(settings: Settings) -> None:
    state = _game_state_with_cue(settings, 0.0)
    prediction = estimate_shot_from_cue(state, state.cue_stick, settings=settings)
    assert prediction is not None
    assert not prediction.is_empty
    # Aiming straight down the table at a ball 20 in away must produce a contact.
    assert [i for i in prediction.impact_points if not i.is_cushion]


def test_estimate_shot_returns_none_without_a_cue_ball(settings: Settings) -> None:
    """No cue ball means no shot to predict. The caller keeps the previous
    overlay rather than blanking it."""
    state = _game_state_with_cue(settings, 0.0)
    state.cue_ball = None
    assert estimate_shot_from_cue(state, state.cue_stick, settings=settings) is None


def test_estimate_shot_returns_none_without_calibration(settings: Settings) -> None:
    state = _game_state_with_cue(settings, 0.0)
    state.cue_ball.table_pos = None
    assert estimate_shot_from_cue(state, state.cue_stick, settings=settings) is None


def test_assumed_power_lowers_confidence(settings: Settings) -> None:
    """Power is not observable from one frame, and the confidence must say so
    rather than presenting a guess as a measurement."""
    state = _game_state_with_cue(settings, 0.0)
    assumed = estimate_shot_from_cue(state, state.cue_stick, settings=settings)
    measured = estimate_shot_from_cue(state, state.cue_stick, power=50.0, settings=settings)
    assert assumed.confidence < measured.confidence


def test_low_cue_confidence_propagates(settings: Settings) -> None:
    """A weak cue detection must not yield a confident prediction."""
    strong = _game_state_with_cue(settings, 0.0)
    weak = _game_state_with_cue(settings, 0.0)
    weak.cue_stick.confidence = 0.3

    strong_prediction = estimate_shot_from_cue(strong, strong.cue_stick, settings=settings)
    weak_prediction = estimate_shot_from_cue(weak, weak.cue_stick, settings=settings)
    assert weak_prediction.confidence < strong_prediction.confidence


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_returns_a_hit_for_an_unchanged_aim(settings: Settings) -> None:
    cache = PredictionCache()
    balls = [_object_ball("t", 40.0, 19.0)]
    first = simulate_shot_cached(cache, Vec2(20.0, 19.0), 10.0, 50, balls, settings)
    second = simulate_shot_cached(cache, Vec2(20.0, 19.0), 10.0, 50, balls, settings)
    assert second is first
    assert cache.hits == 1


def test_cache_absorbs_sub_threshold_jitter(settings: Settings) -> None:
    """The point of quantising: detection noise wobbles the measured angle by a
    fraction of a degree, and recomputing on every wobble makes the projected
    line shimmer."""
    cache = PredictionCache(angle_step_deg=0.5)
    balls = [_object_ball("t", 40.0, 19.0)]
    simulate_shot_cached(cache, Vec2(20.0, 19.0), 10.0, 50, balls, settings)
    simulate_shot_cached(cache, Vec2(20.0, 19.0), 10.05, 50, balls, settings)
    assert cache.hits == 1


def test_cache_misses_on_a_real_aim_change(settings: Settings) -> None:
    cache = PredictionCache(angle_step_deg=0.5)
    balls = [_object_ball("t", 40.0, 19.0)]
    simulate_shot_cached(cache, Vec2(20.0, 19.0), 10.0, 50, balls, settings)
    simulate_shot_cached(cache, Vec2(20.0, 19.0), 25.0, 50, balls, settings)
    assert cache.misses == 2


def test_cache_key_includes_the_ball_layout(settings: Settings) -> None:
    """Leaving the layout out would be a correctness bug, not just a stale
    result: the same aim through a different rack gives a different answer."""
    cache = PredictionCache()
    a = simulate_shot_cached(
        cache, Vec2(20.0, 19.0), 0.0, 50, [_object_ball("t", 40.0, 19.0)], settings
    )
    b = simulate_shot_cached(
        cache, Vec2(20.0, 19.0), 0.0, 50, [_object_ball("t", 60.0, 30.0)], settings
    )
    assert a is not b
    assert cache.misses == 2


def test_cache_evicts_and_stays_bounded(settings: Settings) -> None:
    cache = PredictionCache(max_entries=4)
    for i in range(20):
        simulate_shot_cached(cache, Vec2(20.0, 19.0), float(i) * 10.0, 50, [], settings)
    assert len(cache._entries) <= 4


def test_cache_clear_resets_statistics(settings: Settings) -> None:
    cache = PredictionCache()
    simulate_shot_cached(cache, Vec2(20.0, 19.0), 0.0, 50, [], settings)
    cache.clear()
    assert cache.hit_rate == 0.0


def test_cached_confidence_scaling_does_not_corrupt_the_entry(
    settings: Settings,
) -> None:
    """``estimate_shot_from_cue`` scales confidence by input quality. Cached
    predictions are shared objects, so scaling one in place would compound the
    penalty on every later hit -- confidence would decay toward zero the longer
    someone aimed at the same spot.
    """
    cache = PredictionCache()
    state = _game_state_with_cue(settings, 0.0)

    confidences = [
        estimate_shot_from_cue(
            state, state.cue_stick, settings=settings, cache=cache
        ).confidence
        for _ in range(5)
    ]
    assert cache.hits >= 3, "expected the cache to be serving hits"
    assert len(set(round(c, 9) for c in confidences)) == 1, (
        f"confidence drifted across cache hits: {confidences}"
    )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_simulation_fits_the_frame_budget(settings: Settings) -> None:
    """Physics shares a 33 ms frame with capture, detection and rendering.

    ``min`` of several runs: scheduler noise only ever adds time, and this box
    has been observed adding tens of milliseconds to identical work.
    """
    rack = [
        _object_ball(f"b{i}", 40.0 + (i % 5) * 3.0, 12.0 + (i // 5) * 3.0)
        for i in range(15)
    ]
    for _ in range(5):
        simulate_shot(Vec2(20.0, 19.0), 12.0, 50, rack, settings)

    best = float("inf")
    for _ in range(10):
        start = time.perf_counter()
        simulate_shot(Vec2(20.0, 19.0), 12.0, 50, rack, settings)
        best = min(best, (time.perf_counter() - start) * 1000.0)

    assert best < 15.0, f"full-rack simulation took {best:.1f} ms"


def test_aiming_line_is_cheap(settings: Settings) -> None:
    """The pre-shot aiming line is what gets recomputed every frame, so it is
    the case that actually has to be fast."""
    target = _object_ball("t", 40.0, 19.0)
    for _ in range(5):
        simulate_shot(Vec2(20.0, 19.0), 0.0, 50, [target], settings)

    best = float("inf")
    for _ in range(20):
        start = time.perf_counter()
        simulate_shot(Vec2(20.0, 19.0), 0.0, 50, [target], settings)
        best = min(best, (time.perf_counter() - start) * 1000.0)

    assert best < 5.0, f"single-ball aiming line took {best:.1f} ms"
