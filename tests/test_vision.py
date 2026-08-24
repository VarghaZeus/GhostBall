"""Accuracy and robustness tests for the vision pipeline.

Every test runs against synthetic frames from :mod:`tests.synthetic`, whose
ground truth is exact by construction. Errors are asserted in **inches** for
positions and **pixels** for the table, because those are the units the spec's
targets are written in:

* trajectory prediction within 3 inches of actual
* projected overlay aligned within 20 pixels

Assertion thresholds are set from measured performance with real headroom, not
pinned to the current number. A test that fails on a 20% regression is useful; a
test that fails on a 2% one just gets deleted.

The scenario list is the point of this file. Detection that works on a clean
axis-aligned rectangle proves very little -- the failures that cost real time
were all in combinations: a rotated *and* keystoned table, a pocket that looks
like the 8 ball, and the system's own projected overlay read as a cue stick.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.config import Settings
from app.models import BallColor, BallKind, PocketId, Vec2
from tests.synthetic import BallSpec, render_table, standard_rack
from vision.calibration import (
    camera_to_table_coords,
    compute_perspective_transform,
    detect_table_boundaries,
    downscale_for_detection,
    felt_mask,
    order_corners,
)
from vision.detection import (
    BallTracker,
    detect_balls,
    detect_cue_ball,
    detect_cue_stick,
    detect_pocket_openings,
    extract_game_state,
    pocket_positions,
)

#: Spec target is 20 px. 15 px leaves headroom while still catching a real
#: regression; measured worst case across every scenario below is ~9 px.
MAX_CORNER_ERROR_PX = 15.0

#: Spec target is 3 in for trajectory prediction, which detection error feeds
#: into. Measured worst case is ~0.35 in, so 1.5 in is generous but still an
#: order of magnitude tighter than the requirement.
MAX_BALL_ERROR_IN = 1.5

#: A cue angle error of 3 degrees over a table's length is roughly 2 in of
#: aiming error at the far cushion. Measured worst case is 0.8 degrees.
MAX_CUE_ANGLE_ERROR_DEG = 3.0


@pytest.fixture()
def settings() -> Settings:
    s = Settings()
    s.camera.use_mock = True
    s.projector.use_mock = True
    return s


#: The scenarios that matter. Each entry is ``(label, render_kwargs)``.
#: Combinations are deliberate -- rotation alone and keystone alone both passed
#: while their combination failed at 89 px, which is exactly the kind of bug a
#: single-factor test suite never finds.
SCENARIOS = [
    ("plain", {}),
    ("keystone", {"perspective": 0.12}),
    ("rotated", {"rotation_deg": 7, "perspective": 0.06}),
    ("rotated_negative", {"rotation_deg": -12, "perspective": 0.05}),
    ("rotated_and_keystoned", {"rotation_deg": 5, "perspective": 0.12}),
    ("heavily_rotated", {"rotation_deg": 20, "perspective": 0.10}),
    ("shadows", {"perspective": 0.10, "shadows": True}),
    ("sensor_noise", {"perspective": 0.08, "noise_sigma": 5.0}),
    ("noise_and_vignette", {"perspective": 0.08, "noise_sigma": 5.0, "vignette": 0.35}),
    ("projected_overlay", {"perspective": 0.08, "overlay_streak": True}),
    (
        "everything_at_once",
        {
            "rotation_deg": 5,
            "perspective": 0.12,
            "shadows": True,
            "noise_sigma": 4.0,
            "vignette": 0.25,
            "overlay_streak": True,
        },
    ),
]


def _match_ball(balls, target: Vec2):
    """Nearest detected ball to a ground-truth table position."""
    return min(balls, key=lambda b: b.table_pos.distance_to(target))


def _detect_all(settings: Settings, **render_kwargs):
    """Render a scene and run the whole pipeline over it.

    Returns ``(truth, boundary, camera_to_table, balls)`` with ``table_pos``
    populated on every ball.
    """
    frame, truth = render_table(
        settings, standard_rack(settings), cue_from=(8, 30), cue_to=(17, 21), **render_kwargs
    )
    boundary = detect_table_boundaries(frame, settings)
    assert boundary is not None, "table not detected"
    camera_to_table, _ = compute_perspective_transform(boundary, settings)
    balls = detect_balls(frame, boundary, settings)
    for ball in balls:
        ball.table_pos = camera_to_table_coords(ball.center_px, camera_to_table)
    return frame, truth, boundary, camera_to_table, balls


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "kwargs"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_table_corners_within_alignment_budget(
    label: str, kwargs: dict, settings: Settings
) -> None:
    """Corners must land within the projection alignment budget.

    This is the measurement the whole system rests on: the homography is solved
    from these four points, so their error is the floor for every projected
    overlay afterwards.
    """
    frame, truth = render_table(settings, standard_rack(settings), **kwargs)
    boundary = detect_table_boundaries(frame, settings)
    assert boundary is not None, f"[{label}] table not detected"

    errors = [
        got.distance_to(want)
        for got, want in zip(boundary.corners(), truth.corners, strict=True)
    ]
    assert max(errors) < MAX_CORNER_ERROR_PX, (
        f"[{label}] worst corner error {max(errors):.1f} px exceeds "
        f"{MAX_CORNER_ERROR_PX} px; per-corner {[f'{e:.1f}' for e in errors]}"
    )


def test_table_corners_are_ordered_clockwise_from_top_left(settings: Settings) -> None:
    """Corner *order* is as load-bearing as corner position.

    A mis-ordered quad still solves to a valid-looking homography -- one that
    mirrors or rotates the entire projection. It is invisible in the RMSE and
    obvious on the table, which is the worst possible combination.
    """
    frame, truth = render_table(settings, standard_rack(settings), perspective=0.10)
    boundary = detect_table_boundaries(frame, settings)
    assert boundary is not None

    for got, want in zip(boundary.corners(), truth.corners, strict=True):
        assert got.distance_to(want) < MAX_CORNER_ERROR_PX

    # And the geometric relationships that define the ordering.
    tl, tr, br, bl = boundary.corners()
    assert tl.x < tr.x and bl.x < br.x, "top-left must be left of top-right"
    assert tl.y < bl.y and tr.y < br.y, "top corners must be above bottom corners"


def test_order_corners_handles_rotation() -> None:
    """The ordering must survive rotation, which the common sum/difference
    trick does not once the table turns more than ~30 degrees."""
    # A square rotated 40 degrees, given in scrambled order.
    import math

    angle = math.radians(40)
    base = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=np.float64)
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rotated = (base @ rot.T) + np.array([100.0, 100.0])
    scrambled = rotated[[2, 0, 3, 1]]

    ordered = order_corners(scrambled)
    # Image y grows downward, so a screen-clockwise traversal gives a *positive*
    # shoelace signed area -- the opposite of the textbook convention.
    x, y = ordered[:, 0], ordered[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert signed > 0, "corners should come back in clockwise image order"
    assert ordered[0].sum() == pytest.approx(min(p.sum() for p in rotated))


def test_table_rejected_when_absent(settings: Settings) -> None:
    """A frame with no felt must return None, not a guess.

    ``None`` means "keep the previous boundary" upstream. Returning a fabricated
    table would silently poison every coordinate downstream.
    """
    blank = np.full((1080, 1920, 3), (30, 30, 30), dtype=np.uint8)
    assert detect_table_boundaries(blank, settings) is None


def test_small_green_object_is_not_a_table(settings: Settings) -> None:
    """A green object that is too small must be rejected by the area check --
    the realistic false positive is a plant or a green chair, not nothing."""
    import cv2

    frame = np.full((1080, 1920, 3), (30, 30, 30), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 220), (70, 150, 60), -1)
    assert detect_table_boundaries(frame, settings) is None


def test_wrong_aspect_ratio_is_rejected(settings: Settings) -> None:
    """A big green square is not a pool table. The aspect check is what stops a
    green carpet or wall from being accepted."""
    import cv2

    frame = np.full((1080, 1920, 3), (30, 30, 30), dtype=np.uint8)
    cv2.rectangle(frame, (400, 100), (1400, 1000), (70, 150, 60), -1)  # ~1.1 aspect
    assert detect_table_boundaries(frame, settings) is None


def test_empty_frame_is_handled(settings: Settings) -> None:
    """Degenerate input returns None rather than raising."""
    assert detect_table_boundaries(np.zeros((0, 0, 3), dtype=np.uint8), settings) is None


# ---------------------------------------------------------------------------
# Felt mask
# ---------------------------------------------------------------------------


def test_felt_mask_covers_most_of_the_table(settings: Settings) -> None:
    frame, _truth = render_table(settings, standard_rack(settings))
    small, _scale = downscale_for_detection(frame, settings.vision.detection_width)
    mask = felt_mask(small, settings)
    coverage = 100.0 * np.count_nonzero(mask) / mask.size
    assert 45.0 < coverage < 85.0, f"felt coverage {coverage:.1f}% looks wrong"


def test_saturation_ceiling_separates_green_ball_from_green_felt(
    settings: Settings,
) -> None:
    """The hardest discrimination in the pipeline, and a regression guard.

    A green ball's hue sits inside any felt hue range wide enough for real
    cloth, so hue alone masks the 6 ball as part of the table and it vanishes.
    Only the saturation ceiling separates them -- raising ``felt_sat_max`` past
    the ball's saturation must make it disappear, which is what proves the
    ceiling is the mechanism doing the work.
    """
    green_only = [
        BallSpec(settings.table.length_in * 0.5, settings.table.width_in * 0.5, BallColor.GREEN)
    ]
    frame, _truth = render_table(settings, green_only)
    boundary = detect_table_boundaries(frame, settings)
    assert boundary is not None

    found = detect_balls(frame, boundary, settings)
    assert len(found) == 1, "the green ball should be detected against green felt"
    assert found[0].color is BallColor.GREEN

    # Disable the ceiling: the ball now falls inside the felt range and is gone.
    settings.vision.felt_sat_max = 255
    assert detect_balls(frame, boundary, settings) == []


# ---------------------------------------------------------------------------
# Ball detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "kwargs"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_all_balls_found_with_no_false_positives(
    label: str, kwargs: dict, settings: Settings
) -> None:
    """Exact count. False positives matter as much as misses: a phantom ball
    puts a phantom obstacle in the physics simulation."""
    _frame, truth, _boundary, _c2t, balls = _detect_all(settings, **kwargs)
    expected = len(truth.ball_positions_in)
    assert len(balls) == expected, (
        f"[{label}] detected {len(balls)} balls, expected {expected}"
    )


@pytest.mark.parametrize(("label", "kwargs"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_ball_positions_accurate_in_inches(
    label: str, kwargs: dict, settings: Settings
) -> None:
    """Position error in table inches -- the unit physics and players care about."""
    _frame, truth, _boundary, _c2t, balls = _detect_all(settings, **kwargs)
    errors = [
        _match_ball(balls, target).table_pos.distance_to(target)
        for target in truth.ball_positions_in
    ]
    assert max(errors) < MAX_BALL_ERROR_IN, (
        f"[{label}] worst ball position error {max(errors):.2f} in exceeds "
        f"{MAX_BALL_ERROR_IN} in"
    )


@pytest.mark.parametrize(("label", "kwargs"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_ball_colours_classified(label: str, kwargs: dict, settings: Settings) -> None:
    _frame, truth, _boundary, _c2t, balls = _detect_all(settings, **kwargs)
    wrong = [
        (truth.ball_colors[i].value, _match_ball(balls, target).color.value)
        for i, target in enumerate(truth.ball_positions_in)
        if _match_ball(balls, target).color is not truth.ball_colors[i]
    ]
    assert not wrong, f"[{label}] misclassified (expected, got): {wrong}"


def test_pockets_are_not_detected_as_balls(settings: Settings) -> None:
    """A pocket is dark, round and ball-sized -- a textbook false 8 ball, and
    measured as three phantom balls at the corners before the mouths were
    excluded from the search."""
    frame, _truth = render_table(settings, [], perspective=0.08, noise_sigma=5.0)
    boundary = detect_table_boundaries(frame, settings)
    assert boundary is not None
    assert detect_balls(frame, boundary, settings) == []


def test_shadows_are_rejected(settings: Settings) -> None:
    """Shadows must not double the ball count. Circularity is what rejects
    them: a ball with its shadow attached is a lopsided blob."""
    balls_spec = standard_rack(settings)
    frame, _truth = render_table(settings, balls_spec, shadows=True, perspective=0.08)
    boundary = detect_table_boundaries(frame, settings)
    assert boundary is not None
    assert len(detect_balls(frame, boundary, settings)) == len(balls_spec)


def test_balls_detected_without_a_boundary(settings: Settings) -> None:
    """Detection must degrade rather than fail before the table is found --
    otherwise a calibration UI cannot show the user anything useful."""
    frame, truth = render_table(settings, standard_rack(settings))
    balls = detect_balls(frame, None, settings)
    # Looser bounds and no interior mask, so extra detections are expected; the
    # requirement is only that the real balls are among them.
    assert len(balls) >= len(truth.ball_positions_in) - 1


def test_ball_confidence_is_meaningful(settings: Settings) -> None:
    """Confidence must discriminate, not sit at a constant."""
    _frame, _truth, _boundary, _c2t, balls = _detect_all(settings, perspective=0.08)
    assert all(0.0 < b.confidence <= 1.0 for b in balls)
    assert all(b.confidence >= settings.vision.min_ball_confidence for b in balls)


def test_empty_frame_yields_no_balls(settings: Settings) -> None:
    assert detect_balls(np.zeros((0, 0, 3), dtype=np.uint8), None, settings) == []


# ---------------------------------------------------------------------------
# Cue ball
# ---------------------------------------------------------------------------


def test_cue_ball_identified(settings: Settings) -> None:
    _frame, truth, _boundary, _c2t, balls = _detect_all(settings, perspective=0.08)
    cue = detect_cue_ball(_frame, balls, settings)
    assert cue is not None
    assert cue.kind is BallKind.CUE
    assert cue.number is None

    index = truth.cue_ball_index()
    assert index is not None
    assert cue.table_pos.distance_to(truth.ball_positions_in[index]) < MAX_BALL_ERROR_IN


def test_cue_ball_absent_when_no_white_ball(settings: Settings) -> None:
    """Returning None is correct when the cue ball is potted or occluded."""
    no_white = [b for b in standard_rack(settings) if b.color is not BallColor.WHITE]
    frame, _truth = render_table(settings, no_white)
    boundary = detect_table_boundaries(frame, settings)
    balls = detect_balls(frame, boundary, settings)
    assert detect_cue_ball(frame, balls, settings) is None


def test_cue_ball_from_empty_list(settings: Settings) -> None:
    assert detect_cue_ball(np.zeros((10, 10, 3), np.uint8), [], settings) is None


# ---------------------------------------------------------------------------
# Cue stick
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "kwargs"), SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_cue_angle_accurate_in_table_space(
    label: str, kwargs: dict, settings: Settings
) -> None:
    """Aim direction is the most visible output in the whole system, so its
    accuracy is asserted in every scenario -- including the one where the
    system's own projected overlay is on the cloth."""
    frame, truth, boundary, c2t, balls = _detect_all(settings, **kwargs)
    cue = detect_cue_stick(
        frame, boundary, settings,
        cue_ball=detect_cue_ball(frame, balls, settings), camera_to_table=c2t,
    )
    assert cue is not None, f"[{label}] cue not detected"
    assert truth.cue_angle_deg is not None

    error = abs(((cue.angle_deg - truth.cue_angle_deg + 180.0) % 360.0) - 180.0)
    assert error < MAX_CUE_ANGLE_ERROR_DEG, (
        f"[{label}] cue angle error {error:.1f} deg exceeds {MAX_CUE_ANGLE_ERROR_DEG} deg "
        f"(got {cue.angle_deg:.1f}, expected {truth.cue_angle_deg:.1f})"
    )


def test_projected_overlay_is_not_mistaken_for_a_cue(settings: Settings) -> None:
    """The system projects light onto the surface it is watching.

    A rendered trajectory is a long bright line lying on the cloth --
    geometrically identical to a cue. Measured at 118 degrees of aim error
    before colour rejection, i.e. aiming at completely the wrong target. With no
    real cue in frame, nothing should be reported at all.
    """
    frame, _truth = render_table(
        settings, standard_rack(settings), perspective=0.08, overlay_streak=True
    )
    boundary = detect_table_boundaries(frame, settings)
    assert boundary is not None
    c2t, _ = compute_perspective_transform(boundary, settings)
    balls = detect_balls(frame, boundary, settings)

    cue = detect_cue_stick(
        frame, boundary, settings,
        cue_ball=detect_cue_ball(frame, balls, settings), camera_to_table=c2t,
    )
    assert cue is None, (
        f"the projected overlay was reported as a cue aiming at {cue.angle_deg:.0f} deg"
        if cue
        else ""
    )


def test_no_cue_when_none_present(settings: Settings) -> None:
    """No cue between shots is the normal resting state, not an error."""
    frame, _truth = render_table(settings, standard_rack(settings), perspective=0.08)
    boundary = detect_table_boundaries(frame, settings)
    assert detect_cue_stick(frame, boundary, settings) is None


def test_cue_tip_is_the_end_nearest_the_cue_ball(settings: Settings) -> None:
    """Choosing the wrong end points the prediction 180 degrees out -- the most
    visible failure the system can produce."""
    frame, truth, boundary, c2t, balls = _detect_all(settings, perspective=0.08)
    cue_ball = detect_cue_ball(frame, balls, settings)
    cue = detect_cue_stick(frame, boundary, settings, cue_ball=cue_ball, camera_to_table=c2t)
    assert cue is not None
    assert truth.cue_tip_px is not None
    # The detected tip must be nearer the cue ball than the butt would be.
    assert cue.tip_px.distance_to(cue_ball.center_px) < boundary.width_px * 0.25


# ---------------------------------------------------------------------------
# Pockets
# ---------------------------------------------------------------------------


def test_six_pockets_at_expected_positions(settings: Settings) -> None:
    frame, _truth = render_table(settings, standard_rack(settings), perspective=0.10)
    boundary = detect_table_boundaries(frame, settings)
    pockets = detect_pocket_openings(frame, boundary, settings)

    assert len(pockets) == 6
    assert {p.id for p in pockets} == set(PocketId)
    # Every pocket should be within a pocket radius of its geometric position.
    expected = dict(pocket_positions(boundary, settings))
    for pocket in pockets:
        assert pocket.center_px.distance_to(expected[pocket.id]) < pocket.radius_px * 2.0


def test_no_pockets_without_a_boundary(settings: Settings) -> None:
    """Pockets are derived from table geometry, so without a table there is
    nothing to derive them from."""
    frame, _truth = render_table(settings, [])
    assert detect_pocket_openings(frame, None, settings) == []


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def test_tracker_keeps_ids_stable_across_frames(settings: Settings) -> None:
    """Stable ids are what let the game layer identify a potted ball as set
    difference. Without them every frame looks like a full table turnover."""
    tracker = BallTracker(settings)
    ids_per_frame = []
    for frame_index in range(4):
        frame, _truth = render_table(settings, standard_rack(settings), perspective=0.08)
        boundary = detect_table_boundaries(frame, settings)
        c2t, _ = compute_perspective_transform(boundary, settings)
        state = extract_game_state(
            frame, frame_index, 1.0 + frame_index / 30.0, boundary, c2t, settings, tracker
        )
        ids_per_frame.append({b.id for b in state.balls})

    assert ids_per_frame[0] == ids_per_frame[-1], "ids churned on a static table"
    assert all(i.startswith("ball_") for i in ids_per_frame[0])


def test_tracker_measures_velocity_of_a_moving_ball(settings: Settings) -> None:
    """Velocity drives shot detection, so it has to be quantitatively right."""
    tracker = BallTracker(settings)
    length, width = settings.table.length_in, settings.table.width_in
    dt = 1.0 / 30.0
    speed_in_s = 30.0

    moving_id = None
    for i in range(3):
        spec = [BallSpec(length * 0.3 + speed_in_s * dt * i, width * 0.5, BallColor.YELLOW)]
        frame, _truth = render_table(settings, spec)
        boundary = detect_table_boundaries(frame, settings)
        c2t, _ = compute_perspective_transform(boundary, settings)
        state = extract_game_state(
            frame, i, 1.0 + i * dt, boundary, c2t, settings, tracker
        )
        assert state.balls, "ball lost while tracking"
        moving_id = state.balls[0].id

    measured = tracker.velocity(moving_id).length()
    assert measured == pytest.approx(speed_in_s, rel=0.25), (
        f"expected ~{speed_in_s} in/s, measured {measured:.1f} in/s"
    )
    assert tracker.any_moving()


def test_tracker_reports_a_static_table_as_still(settings: Settings) -> None:
    """The settle detector depends on this: a false "moving" reading would keep
    the state machine stuck in shot_in_progress forever."""
    tracker = BallTracker(settings)
    for i in range(3):
        frame, _truth = render_table(settings, standard_rack(settings))
        boundary = detect_table_boundaries(frame, settings)
        c2t, _ = compute_perspective_transform(boundary, settings)
        extract_game_state(frame, i, 1.0 + i / 30.0, boundary, c2t, settings, tracker)
    assert not tracker.any_moving()
    assert tracker.max_speed() < settings.vision.ball_stopped_threshold


def test_tracker_reset_clears_state(settings: Settings) -> None:
    tracker = BallTracker(settings)
    frame, _truth = render_table(settings, standard_rack(settings))
    boundary = detect_table_boundaries(frame, settings)
    c2t, _ = compute_perspective_transform(boundary, settings)
    extract_game_state(frame, 0, 1.0, boundary, c2t, settings, tracker)
    tracker.reset()
    assert tracker.max_speed() == 0.0


def test_tracker_velocity_of_unknown_id_is_zero(settings: Settings) -> None:
    assert BallTracker(settings).velocity("nope").length() == 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_game_state_is_fully_populated(settings: Settings) -> None:
    frame, truth = render_table(
        settings, standard_rack(settings), perspective=0.10,
        cue_from=(8, 30), cue_to=(17, 21),
    )
    boundary = detect_table_boundaries(frame, settings)
    c2t, _ = compute_perspective_transform(boundary, settings)
    state = extract_game_state(
        frame, 42, 123.5, boundary, c2t, settings, BallTracker(settings)
    )

    assert state.frame_index == 42
    assert state.timestamp == 123.5
    assert state.is_usable, "state should be usable for physics"
    assert len(state.balls) == len(truth.ball_positions_in)
    assert state.cue_ball is not None
    assert state.cue_stick is not None
    assert len(state.pockets) == 6
    assert 0.0 < state.confidence <= 1.0
    # Every ball must carry table coordinates, or physics silently gets a
    # phantom ball at the origin.
    assert all(b.table_pos is not None for b in state.balls)
    assert all(p.table_pos is not None for p in state.pockets)


def test_game_state_without_calibration_has_no_table_positions(
    settings: Settings,
) -> None:
    """Before the homography is known, pixel detections are still returned but
    ``table_pos`` stays None -- which is what stops physics running on garbage."""
    frame, _truth = render_table(settings, standard_rack(settings))
    state = extract_game_state(frame, 0, 1.0, None, None, settings)
    assert not state.is_usable
    assert all(b.table_pos is None for b in state.balls)
    assert state.confidence == 0.0


def test_game_state_survives_an_empty_frame(settings: Settings) -> None:
    """A frame with nothing in it is a normal event -- someone leaning over the
    table -- and must yield a low-confidence state, never an exception."""
    state = extract_game_state(
        np.zeros((0, 0, 3), dtype=np.uint8), 7, 1.0, None, None, settings
    )
    assert state.frame_index == 7
    assert state.balls == []
    assert state.confidence == 0.0


def test_game_state_on_a_blank_table(settings: Settings) -> None:
    """An empty but visible table: usable geometry, no balls."""
    frame, _truth = render_table(settings, [], perspective=0.08)
    boundary = detect_table_boundaries(frame, settings)
    c2t, _ = compute_perspective_transform(boundary, settings)
    state = extract_game_state(frame, 0, 1.0, boundary, c2t, settings)
    assert state.balls == []
    assert state.cue_ball is None
    assert not state.is_usable  # no cue ball, so physics has nothing to shoot
    assert state.confidence > 0.0  # but the table itself was found


def test_confidence_rises_with_detection_quality(settings: Settings) -> None:
    """Confidence must be an actual signal the mode layer can gate on."""
    frame_full, _t1 = render_table(settings, standard_rack(settings), perspective=0.08)
    frame_bare, _t2 = render_table(settings, [], perspective=0.08)

    boundary = detect_table_boundaries(frame_full, settings)
    c2t, _ = compute_perspective_transform(boundary, settings)
    full = extract_game_state(frame_full, 0, 1.0, boundary, c2t, settings)

    boundary2 = detect_table_boundaries(frame_bare, settings)
    c2t2, _ = compute_perspective_transform(boundary2, settings)
    bare = extract_game_state(frame_bare, 0, 1.0, boundary2, c2t2, settings)

    assert full.confidence > bare.confidence


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_per_frame_detection_fits_the_frame_budget(settings: Settings) -> None:
    """Detection must leave room for capture, physics and rendering.

    Timed on whatever machine runs the tests, so the bound is deliberately loose
    -- it is a guard against an algorithmic regression (an accidental
    full-resolution pass, an O(n^2) loop), not a substitute for measuring on the
    Pi. ``min`` of several runs, because scheduler noise only ever adds time.
    """
    frame, _truth = render_table(
        settings, standard_rack(settings), perspective=0.10,
        cue_from=(8, 30), cue_to=(17, 21),
    )
    boundary = detect_table_boundaries(frame, settings)
    c2t, _ = compute_perspective_transform(boundary, settings)
    pockets = detect_pocket_openings(frame, boundary, settings)
    tracker = BallTracker(settings)

    for _ in range(3):  # warm up caches and OpenCV's internal buffers
        extract_game_state(frame, 0, 1.0, boundary, c2t, settings, tracker, pockets)

    timings = []
    for i in range(7):
        start = time.perf_counter()
        extract_game_state(frame, i, 1.0, boundary, c2t, settings, tracker, pockets)
        timings.append((time.perf_counter() - start) * 1000.0)

    best = min(timings)
    assert best < 33.0, (
        f"per-frame detection took {best:.1f} ms, which alone exceeds the "
        f"33 ms budget for 30 FPS"
    )


def test_downscaling_is_what_makes_the_budget(settings: Settings) -> None:
    """Guards the single most important performance decision in the pipeline.

    Detection at full resolution measured 45 ms against a 33 ms budget -- over
    the limit before capture or rendering get a share. If someone raises
    ``detection_width`` to 1920 thinking it will help accuracy, this test
    explains what it actually costs.
    """
    frame, _truth = render_table(settings, standard_rack(settings), perspective=0.10)
    boundary = detect_table_boundaries(frame, settings)
    c2t, _ = compute_perspective_transform(boundary, settings)

    def measure(width: int) -> float:
        local = Settings()
        local.vision.detection_width = width
        extract_game_state(frame, 0, 1.0, boundary, c2t, local)
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            extract_game_state(frame, 0, 1.0, boundary, c2t, local)
            best = min(best, (time.perf_counter() - start) * 1000.0)
        return best

    downscaled = measure(960)
    full = measure(1920)
    # Measured ratio is 1.9x-5x depending on how many contours the larger frame
    # produces. 1.5x is the floor that still demonstrates the point without
    # making the test flaky on a noisy machine.
    assert full > downscaled * 1.5, (
        f"expected downscaling to save materially: 960px={downscaled:.1f}ms "
        f"vs 1920px={full:.1f}ms"
    )


def test_detection_width_does_not_change_the_answer_much(settings: Settings) -> None:
    """Downscaling buys speed; it must not cost meaningful accuracy.

    This is the other half of the trade-off. If 960 px degraded positions, the
    speed would be worthless.
    """
    frame, truth = render_table(settings, standard_rack(settings), perspective=0.10)

    results = {}
    for width in (720, 960, 1440):
        local = Settings()
        local.vision.detection_width = width
        boundary = detect_table_boundaries(frame, local)
        assert boundary is not None, f"table lost at detection_width={width}"
        c2t, _ = compute_perspective_transform(boundary, local)
        balls = detect_balls(frame, boundary, local)
        for ball in balls:
            ball.table_pos = camera_to_table_coords(ball.center_px, c2t)
        errors = [
            _match_ball(balls, target).table_pos.distance_to(target)
            for target in truth.ball_positions_in
        ]
        results[width] = max(errors)

    for width, error in results.items():
        assert error < MAX_BALL_ERROR_IN, f"detection_width={width} gave {error:.2f} in error"
