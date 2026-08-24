"""Tests for the implemented parts of the scaffold.

Everything here runs with no camera, no projector and no Pi. The stubbed stages
are deliberately not tested -- there is nothing to assert about a
``NotImplementedError`` beyond that it is raised, and pinning that would just
create a test to delete when the stage lands.

What is worth testing now is the coordinate math, since it is pure, it is what a
20 px alignment target depends on, and getting a transform inverted is very hard
to spot by looking at a projected overlay.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.config import TABLE_PRESETS, Settings, load_settings
from app.models import (
    Ball,
    BallKind,
    GameState,
    Player,
    TableBoundary,
    Vec2,
)


@pytest.fixture()
def settings() -> Settings:
    """Default settings with mock hardware, so nothing touches a device."""
    s = Settings()
    s.camera.use_mock = True
    s.projector.use_mock = True
    return s


@pytest.fixture()
def boundary() -> TableBoundary:
    """A perfectly rectangular table in camera space.

    Axis-aligned on purpose: with a known rectangle the expected table
    coordinates can be computed by hand, so a failure points at the transform
    rather than at the fixture.
    """
    return TableBoundary(
        top_left=Vec2(200.0, 100.0),
        top_right=Vec2(1720.0, 100.0),
        bottom_right=Vec2(1720.0, 860.0),
        bottom_left=Vec2(200.0, 860.0),
        center=Vec2(960.0, 480.0),
        width_px=1520.0,
        height_px=760.0,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_defaults_are_valid() -> None:
    """Every field must have a working default, since a missing config.yaml is
    a supported way to boot."""
    s = Settings()
    assert s.table.length_in > s.table.width_in
    assert s.system.target_fps == 30
    assert math.isclose(s.frame_interval, 1 / 30)


def test_missing_config_file_falls_back_to_defaults(tmp_path) -> None:
    """A missing file is a warning, not an error."""
    loaded = load_settings(tmp_path / "nope.yaml")
    assert loaded.table_preset == "7ft"


def test_shipped_config_yaml_is_valid() -> None:
    """Guards against the shipped config drifting out of sync with the schema --
    which would break a fresh checkout on first run."""
    from app.config import DEFAULT_CONFIG_PATH

    assert DEFAULT_CONFIG_PATH.is_file(), "config.yaml should ship with the package"
    loaded = load_settings(DEFAULT_CONFIG_PATH)
    assert loaded.system.target_fps > 0


def test_table_presets_are_landscape() -> None:
    """The +x-is-the-long-axis convention is load-bearing for every transform."""
    for name, preset in TABLE_PRESETS.items():
        assert preset.length_in > preset.width_in, f"{name} preset is not landscape"


def test_width_exceeding_length_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(table={"length_in": 38.0, "width_in": 76.0})


def test_unknown_table_preset_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(table_preset="12ft")


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


def test_vec2_arithmetic() -> None:
    assert Vec2(3, 4).length() == 5.0
    assert Vec2(1, 2) + Vec2(3, 4) == Vec2(4, 6)
    assert Vec2(5, 5) - Vec2(1, 2) == Vec2(4, 3)
    assert Vec2(2, 3).scaled(2) == Vec2(4, 6)
    assert Vec2(1.6, 2.4).as_int() == (2, 2)


def test_game_state_is_unusable_without_table_coords() -> None:
    """The guard that stops physics being handed a ball with no table position --
    which would silently put a phantom ball at the origin."""
    cue = Ball(id="cue", center_px=Vec2(100, 100), radius_px=20, kind=BallKind.CUE)
    state = GameState(timestamp=0.0, frame_index=0, cue_ball=cue)
    assert not state.is_usable, "no table boundary yet"

    state.table_boundary = TableBoundary(
        Vec2(0, 0), Vec2(10, 0), Vec2(10, 10), Vec2(0, 10), Vec2(5, 5), 10, 10
    )
    assert not state.is_usable, "cue ball still has no table_pos"

    cue.table_pos = Vec2(20.0, 19.0)
    assert state.is_usable


def test_object_balls_excludes_cue_and_pocketed() -> None:
    cue = Ball(id="cue", center_px=Vec2(0, 0), radius_px=10, kind=BallKind.CUE)
    solid = Ball(id="b1", center_px=Vec2(1, 1), radius_px=10, kind=BallKind.SOLID)
    sunk = Ball(id="b2", center_px=Vec2(2, 2), radius_px=10, kind=BallKind.SOLID, pocketed=True)
    state = GameState(timestamp=0.0, frame_index=0, balls=[cue, solid, sunk], cue_ball=cue)
    assert [b.id for b in state.object_balls()] == ["b1"]


def test_player_accuracy_handles_zero_shots() -> None:
    """Division-by-zero guard: the panel reads this before the first shot."""
    assert Player(name="a").accuracy_pct == 0.0
    assert Player(name="a", shots_taken=4, shots_made=1).accuracy_pct == 25.0


def test_session_advance_player_wraps_and_resets_combo() -> None:
    from app.models import GameSession

    session = GameSession(players=[Player(name="a"), Player(name="b")], combo_count=3)
    session.advance_player()
    assert session.current_player.name == "b"
    assert session.combo_count == 0
    session.advance_player()
    assert session.current_player.name == "a"


# ---------------------------------------------------------------------------
# Camera <-> table transforms
# ---------------------------------------------------------------------------


def test_camera_to_table_maps_corners_to_table_dimensions(
    boundary: TableBoundary, settings: Settings
) -> None:
    """The corners must land exactly on the table rectangle, in the documented
    clockwise-from-top-left order. An ordering bug here mirrors the whole
    projection and is surprisingly easy to miss on a symmetric table."""
    from vision.calibration import camera_to_table_coords, compute_perspective_transform

    c2t, _ = compute_perspective_transform(boundary, settings)
    length, width = settings.table.length_in, settings.table.width_in

    expected = [(0.0, 0.0), (length, 0.0), (length, width), (0.0, width)]
    for corner, (ex, ey) in zip(boundary.corners(), expected, strict=True):
        got = camera_to_table_coords(corner, c2t)
        assert got.x == pytest.approx(ex, abs=1e-6)
        assert got.y == pytest.approx(ey, abs=1e-6)


def test_camera_table_round_trip(boundary: TableBoundary, settings: Settings) -> None:
    """Forward then inverse must return the original point."""
    from vision.calibration import (
        camera_to_table_coords,
        compute_perspective_transform,
        table_to_camera_coords,
    )

    c2t, t2c = compute_perspective_transform(boundary, settings)
    for point in (Vec2(500.0, 300.0), Vec2(960.0, 480.0), Vec2(1500.0, 800.0)):
        table = camera_to_table_coords(point, c2t)
        back = table_to_camera_coords(table, t2c)
        assert back.x == pytest.approx(point.x, abs=1e-6)
        assert back.y == pytest.approx(point.y, abs=1e-6)


def test_table_center_maps_to_camera_center(
    boundary: TableBoundary, settings: Settings
) -> None:
    from vision.calibration import compute_perspective_transform, table_to_camera_coords

    _, t2c = compute_perspective_transform(boundary, settings)
    center = Vec2(settings.table.length_in / 2, settings.table.width_in / 2)
    got = table_to_camera_coords(center, t2c)
    assert got.x == pytest.approx(boundary.center.x, abs=1e-6)
    assert got.y == pytest.approx(boundary.center.y, abs=1e-6)


def test_degenerate_boundary_is_rejected(settings: Settings) -> None:
    """Collinear corners must raise rather than produce a garbage transform.

    This is the real failure mode when felt segmentation leaks: two "corners"
    collapse onto each other, and without this check the solve returns a matrix
    of infinities that poisons every downstream coordinate.
    """
    from vision.calibration import CalibrationError, compute_perspective_transform

    collapsed = TableBoundary(
        top_left=Vec2(100.0, 100.0),
        top_right=Vec2(100.0, 100.0),
        bottom_right=Vec2(100.0, 100.0),
        bottom_left=Vec2(100.0, 100.0),
        center=Vec2(100.0, 100.0),
        width_px=0.0,
        height_px=0.0,
    )
    with pytest.raises(CalibrationError):
        compute_perspective_transform(collapsed, settings)


def test_batch_transform_matches_scalar(boundary: TableBoundary, settings: Settings) -> None:
    """The vectorised path is an optimisation, so it must agree with the scalar one."""
    from vision.calibration import compute_perspective_transform, transform_point, transform_points

    c2t, _ = compute_perspective_transform(boundary, settings)
    points = [Vec2(300.0, 200.0), Vec2(900.0, 500.0), Vec2(1600.0, 700.0)]
    batched = transform_points(points, c2t)
    for point, got in zip(points, batched, strict=True):
        expected = transform_point(point, c2t)
        assert got.x == pytest.approx(expected.x, abs=1e-9)
        assert got.y == pytest.approx(expected.y, abs=1e-9)


def test_expected_ball_radius_brackets_the_true_radius(
    boundary: TableBoundary, settings: Settings
) -> None:
    from app.config import BALL_RADIUS_IN
    from vision.calibration import expected_ball_radius_px, pixels_per_inch

    low, high = expected_ball_radius_px(boundary, settings)
    nominal = pixels_per_inch(boundary, settings) * BALL_RADIUS_IN
    assert low < nominal < high


# ---------------------------------------------------------------------------
# Table <-> projector transforms
# ---------------------------------------------------------------------------


def test_identity_calibration_fills_the_projector_frame(settings: Settings) -> None:
    """The uncalibrated fallback must still map the table across the whole output,
    so something visible is projected before the wizard has been run."""
    from projection.mapper import ProjectionMapper, identity_calibration

    mapper = ProjectionMapper(identity_calibration(settings))
    assert not mapper.calibration.is_calibrated

    origin = mapper.table_to_projector(Vec2(0.0, 0.0))
    assert (origin.x, origin.y) == pytest.approx((0.0, 0.0))

    far = mapper.table_to_projector(
        Vec2(settings.table.length_in, settings.table.width_in)
    )
    assert far.x == pytest.approx(settings.projector.width)
    assert far.y == pytest.approx(settings.projector.height)


def test_solved_homography_reproduces_its_corners(settings: Settings) -> None:
    from projection.mapper import ProjectionMapper, solve_projector_homography

    length, width = settings.table.length_in, settings.table.width_in
    table_points = [Vec2(0, 0), Vec2(length, 0), Vec2(length, width), Vec2(0, width)]
    # A deliberately keystoned quad -- the top edge narrower than the bottom --
    # which is exactly what an off-axis projector produces and what the affine
    # path cannot represent.
    projector_points = [Vec2(120, 80), Vec2(1800, 140), Vec2(1750, 1000), Vec2(170, 950)]

    calibration = solve_projector_homography(
        table_points, projector_points, settings.projector.width, settings.projector.height
    )
    assert calibration.is_calibrated
    # Four points give an exact fit, hence the zero.
    assert calibration.rmse_px == pytest.approx(0.0, abs=1e-3)

    mapper = ProjectionMapper(calibration)
    for table_point, expected in zip(table_points, projector_points, strict=True):
        got = mapper.table_to_projector(table_point)
        assert got.x == pytest.approx(expected.x, abs=1e-3)
        assert got.y == pytest.approx(expected.y, abs=1e-3)


def test_projector_table_round_trip(settings: Settings) -> None:
    from projection.mapper import ProjectionMapper, solve_projector_homography

    length, width = settings.table.length_in, settings.table.width_in
    calibration = solve_projector_homography(
        [Vec2(0, 0), Vec2(length, 0), Vec2(length, width), Vec2(0, width)],
        [Vec2(120, 80), Vec2(1800, 140), Vec2(1750, 1000), Vec2(170, 950)],
        settings.projector.width,
        settings.projector.height,
    )
    mapper = ProjectionMapper(calibration)
    for point in (Vec2(10.0, 10.0), Vec2(38.0, 19.0), Vec2(70.0, 30.0)):
        back = mapper.projector_to_table(mapper.table_to_projector(point))
        assert back.x == pytest.approx(point.x, abs=1e-6)
        assert back.y == pytest.approx(point.y, abs=1e-6)


def test_solve_rejects_insufficient_points(settings: Settings) -> None:
    from projection.mapper import solve_projector_homography

    with pytest.raises(ValueError, match="at least 4"):
        solve_projector_homography([Vec2(0, 0)], [Vec2(0, 0)], 1920, 1080)


def test_solve_rejects_mismatched_point_counts(settings: Settings) -> None:
    from projection.mapper import solve_projector_homography

    with pytest.raises(ValueError):
        solve_projector_homography(
            [Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)], [Vec2(0, 0)], 1920, 1080
        )


def test_nudge_discards_homography(settings: Settings) -> None:
    """Nudging switches to the affine model, since mixing a nudge into a solved
    keystone transform is incoherent."""
    from projection.mapper import ProjectionMapper, solve_projector_homography

    length, width = settings.table.length_in, settings.table.width_in
    calibration = solve_projector_homography(
        [Vec2(0, 0), Vec2(length, 0), Vec2(length, width), Vec2(0, width)],
        [Vec2(120, 80), Vec2(1800, 140), Vec2(1750, 1000), Vec2(170, 950)],
        settings.projector.width,
        settings.projector.height,
    )
    mapper = ProjectionMapper(calibration)
    assert mapper.calibration.homography is not None
    mapper.nudge(dx=5.0)
    assert mapper.calibration.homography is None
    assert mapper.calibration.offset_x == 5.0


def test_save_and_load_calibration_round_trip(tmp_path, settings: Settings) -> None:
    from projection.mapper import identity_calibration, load_calibration, save_calibration

    path = tmp_path / "cal.json"
    original = identity_calibration(settings)
    original.rmse_px = 4.25
    original.is_calibrated = True
    save_calibration(original, path)

    loaded = load_calibration(path)
    assert loaded is not None
    assert loaded.rmse_px == 4.25
    assert loaded.is_calibrated
    assert loaded.scale_x == pytest.approx(original.scale_x)


def test_load_calibration_tolerates_a_corrupt_file(tmp_path) -> None:
    """A corrupt file must not stop the system booting -- it falls back to
    uncalibrated and tells the user to recalibrate."""
    from projection.mapper import load_calibration

    path = tmp_path / "cal.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert load_calibration(path) is None


def test_load_calibration_missing_file_returns_none(tmp_path) -> None:
    from projection.mapper import load_calibration

    assert load_calibration(tmp_path / "absent.json") is None


# ---------------------------------------------------------------------------
# Physics parameters
# ---------------------------------------------------------------------------


def test_table_geometry_insets_bounds_by_a_ball_radius(settings: Settings) -> None:
    """The bounds are inset so collision tests can compare ball centres directly."""
    from app.config import BALL_RADIUS_IN
    from physics.models import TableGeometry

    geometry = TableGeometry.from_settings(settings)
    assert geometry.x_min == pytest.approx(BALL_RADIUS_IN)
    assert geometry.x_max == pytest.approx(settings.table.length_in - BALL_RADIUS_IN)
    assert len(geometry.pocket_centers()) == 6
    assert geometry.contains(Vec2(38.0, 19.0))
    assert not geometry.contains(Vec2(0.0, 0.0))  # a corner is outside the inset bounds


def test_power_to_velocity_is_monotonic_and_bounded() -> None:
    from physics.models import power_to_velocity

    assert power_to_velocity(0) == pytest.approx(20.0)
    assert power_to_velocity(100) == pytest.approx(300.0)
    assert power_to_velocity(50) < power_to_velocity(75)


def test_sim_ball_movement_threshold() -> None:
    from physics.models import SimBall

    assert not SimBall(id="a", position=Vec2(0, 0)).is_moving
    assert SimBall(id="a", position=Vec2(0, 0), velocity=Vec2(10, 0)).is_moving
    # Below the threshold a ball is visually stopped, so continuing to integrate
    # it just burns frame time.
    assert not SimBall(id="a", position=Vec2(0, 0), velocity=Vec2(0.1, 0)).is_moving


def test_accuracy_profiles_trade_step_for_depth() -> None:
    from app.models import PhysicsAccuracy
    from physics.models import ACCURACY_PROFILES

    fast = ACCURACY_PROFILES[PhysicsAccuracy.FAST]
    accurate = ACCURACY_PROFILES[PhysicsAccuracy.ACCURATE]
    assert fast.timestep > accurate.timestep
    assert fast.max_collision_depth < accurate.max_collision_depth
    assert not fast.simulate_secondary


# ---------------------------------------------------------------------------
# Training score curve
# ---------------------------------------------------------------------------


def test_score_curve_rewards_near_misses() -> None:
    """Exponential decay, not linear: a 1.5 in miss on a 3 in tolerance is a
    decent shot and should not score 50%."""
    from app.models import DrillType
    from modes.training import Drill, score_attempt

    drill = Drill(drill_type=DrillType.POTTING, instruction="pot the 3")
    perfect = score_attempt(drill, 0.0, 0.0, pocketed=True)
    near = score_attempt(drill, 1.5, 2.0, pocketed=True)
    far = score_attempt(drill, 12.0, 30.0, pocketed=False)

    assert perfect.accuracy_pct == pytest.approx(100.0)
    assert perfect.stars == 3
    assert near.accuracy_pct > 60.0
    assert far.accuracy_pct < 20.0
    assert not far.success


def test_potting_success_follows_the_ball_not_the_tolerance() -> None:
    """Potting has an objective outcome, so the ball going in overrides the curve."""
    from app.models import DrillType
    from modes.training import Drill, score_attempt

    drill = Drill(drill_type=DrillType.POTTING, instruction="pot the 3")
    assert score_attempt(drill, 0.2, 0.5, pocketed=False).success is False
    assert score_attempt(drill, 2.5, 4.0, pocketed=True).success is True


def test_position_drill_success_uses_tolerance() -> None:
    """Nothing is potted in a position drill, so tolerance is the only measure."""
    from app.models import DrillType
    from modes.training import Drill, score_attempt

    drill = Drill(drill_type=DrillType.POSITION, instruction="finish near centre")
    assert score_attempt(drill, 4.0, 8.0, pocketed=False).success is True
    assert score_attempt(drill, 20.0, 40.0, pocketed=False).success is False


def test_drill_stats_handle_zero_attempts() -> None:
    from modes.training import DrillStats

    stats = DrillStats()
    assert stats.success_rate_pct == 0.0
    assert stats.mean_accuracy_pct == 0.0


# ---------------------------------------------------------------------------
# Performance instrumentation
# ---------------------------------------------------------------------------


def test_tracker_reports_fps_and_percentiles() -> None:
    from utils.performance import PerformanceTracker

    tracker = PerformanceTracker(window=10, target_fps=30)
    for _ in range(5):
        tracker.begin_frame()
        # Real work per frame: an empty body can measure as 0.0 ms on a coarse
        # clock, and mean-of-zero legitimately reports 0 FPS.
        sum(range(50_000))
        tracker.end_frame()

    snap = tracker.snapshot()
    assert snap.total_frames == 5
    assert snap.fps > 0
    assert snap.frame_ms_p95 >= snap.frame_ms_avg


def test_tracker_counts_stage_time() -> None:
    from utils.performance import PerformanceTracker

    tracker = PerformanceTracker(window=10)
    tracker.begin_frame()
    with tracker.stage("detect"):
        sum(range(10_000))
    tracker.end_frame()
    assert "detect" in tracker.snapshot().stage_ms


def test_tracker_ignores_unpaired_end_frame() -> None:
    """Defensive: the loop's error paths can reach end_frame without begin_frame."""
    from utils.performance import PerformanceTracker

    tracker = PerformanceTracker()
    assert tracker.end_frame() == 0.0
    assert tracker.total_frames == 0


def test_tracker_reset_clears_counters() -> None:
    from utils.performance import PerformanceTracker

    tracker = PerformanceTracker()
    tracker.begin_frame()
    tracker.end_frame()
    tracker.reset()
    assert tracker.snapshot().total_frames == 0


def test_system_metrics_keys_always_present() -> None:
    """psutil is optional, so callers must be able to rely on the keys existing."""
    from utils.performance import get_system_metrics

    metrics = get_system_metrics()
    assert set(metrics) == {"cpu_pct", "mem_pct", "temp_c"}


# ---------------------------------------------------------------------------
# Camera and display, mock backends
# ---------------------------------------------------------------------------


def test_mock_camera_yields_usable_bgr_frames(settings: Settings) -> None:
    from vision.camera import Camera

    with Camera(settings.camera) as camera:
        assert camera.is_mock
        frames = list(camera.stream_frames(max_frames=3))

    assert len(frames) == 3
    assert [f.index for f in frames] == [0, 1, 2]
    for frame in frames:
        assert frame.image.shape == (settings.camera.height, settings.camera.width, 3)
        assert frame.image.dtype == np.uint8
    # Timestamps must be monotonic, since latency accounting subtracts them.
    assert frames[0].timestamp <= frames[-1].timestamp


def test_mock_camera_frames_change_over_time(settings: Settings) -> None:
    """The cue ball sweeps, so motion-dependent code has something to detect.
    Identical frames would make tracking untestable without hardware."""
    from vision.camera import Camera

    with Camera(settings.camera) as camera:
        frames = list(camera.stream_frames(max_frames=40))
    assert not np.array_equal(frames[0].image, frames[-1].image)


def test_camera_capture_before_open_raises(settings: Settings) -> None:
    from vision.camera import Camera, CameraError

    with pytest.raises(CameraError):
        Camera(settings.camera).capture_frame()


def test_camera_close_is_idempotent(settings: Settings) -> None:
    from vision.camera import Camera

    camera = Camera(settings.camera).open()
    camera.close()
    camera.close()  # must not raise
    assert not camera.is_open


def test_mock_display_accepts_rgba_and_flattens(settings: Settings) -> None:
    from projection.display import Display

    with Display(settings.projector) as display:
        assert display.is_mock
        overlay = np.zeros(
            (settings.projector.height, settings.projector.width, 4), dtype=np.uint8
        )
        overlay[100:200, 100:200] = (255, 255, 255, 255)
        assert display.send_frame(overlay)
        assert display.last_frame is not None
        # RGBA in, 3-channel BGR out.
        assert display.last_frame.shape[2] == 3


def test_display_rejects_non_rgba(settings: Settings) -> None:
    from projection.display import Display

    with Display(settings.projector) as display:
        bgr = np.zeros((settings.projector.height, settings.projector.width, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="RGBA"):
            display.send_frame(bgr)


def test_display_clear_projects_black(settings: Settings) -> None:
    from projection.display import Display

    with Display(settings.projector) as display:
        assert display.clear()
        assert display.last_frame is not None
        assert not display.last_frame.any()


def test_transparent_overlay_projects_nothing(settings: Settings) -> None:
    """Alpha zero must produce black, because on a projector black is the only
    way to say "leave the felt alone"."""
    from projection.display import Display

    with Display(settings.projector) as display:
        overlay = np.full(
            (settings.projector.height, settings.projector.width, 4), 255, dtype=np.uint8
        )
        overlay[:, :, 3] = 0
        display.send_frame(overlay)
        assert not display.last_frame.any()


def test_blend_overlay_respects_alpha() -> None:
    from projection.renderer import blend_overlay

    base = np.full((10, 10, 3), 100, dtype=np.uint8)
    overlay = np.zeros((10, 10, 4), dtype=np.uint8)
    overlay[:, :, :3] = 200
    overlay[:, :, 3] = 255

    untouched = blend_overlay(base, np.zeros((10, 10, 4), dtype=np.uint8), alpha=1.0)
    assert np.array_equal(untouched, base)

    full = blend_overlay(base, overlay, alpha=1.0)
    assert full[0, 0, 0] == 200


def test_blend_overlay_rejects_size_mismatch() -> None:
    from projection.renderer import blend_overlay

    with pytest.raises(ValueError, match="size mismatch"):
        blend_overlay(np.zeros((10, 10, 3), np.uint8), np.zeros((20, 20, 4), np.uint8))


# ---------------------------------------------------------------------------
# Application wiring
# ---------------------------------------------------------------------------


def test_app_state_builds_its_dependencies(settings: Settings) -> None:
    from app.state import AppState

    state = AppState(settings=settings)
    assert state.tracker is not None
    assert state.mapper is not None
    assert state.mode_manager is not None
    assert state.pending_stages, "unimplemented stages should be advertised"


def test_detection_summary_survives_no_frames(settings: Settings) -> None:
    """The panel renders before the first frame arrives, so this must not fail."""
    from app.state import AppState

    summary = AppState(settings=settings).detection_summary()
    assert summary["balls"] == 0
    assert summary["table_detected"] is False


def test_mode_manager_defaults_to_freeplay(settings: Settings) -> None:
    from app.models import GameModeName
    from modes.mode_manager import ModeManager

    manager = ModeManager(settings)
    assert manager.session.mode is GameModeName.FREEPLAY


def test_unimplemented_mode_falls_back_to_freeplay(settings: Settings) -> None:
    """A mode named in the spec but not yet built must not take the game down."""
    from app.models import GameModeName
    from modes.mode_manager import ModeManager

    manager = ModeManager(settings)
    manager.load_mode(GameModeName.KNOCKOUT)
    assert manager.session.mode is GameModeName.FREEPLAY


def test_start_game_and_reset(settings: Settings) -> None:
    from app.models import SessionState
    from modes.mode_manager import ModeManager

    manager = ModeManager(settings)
    manager.start_game(["Ada", "Grace"])
    assert [p.name for p in manager.session.players] == ["Ada", "Grace"]

    manager.reset()
    assert manager.session.players == []
    assert manager.session.state is SessionState.IDLE


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(settings: Settings, tmp_path, monkeypatch):
    """A test client with the vision loop disabled, so no hardware is touched.

    ``CALIBRATION_FILE`` is redirected into a temp directory so the app starts
    uncalibrated regardless of the machine it runs on. Without it, anyone who
    has run the calibration wizard on this checkout leaves a saved calibration
    in ``data/`` and ``test_calibration_status_uncalibrated`` starts failing --
    a test that passes or fails depending on what the developer did yesterday.
    """
    from fastapi.testclient import TestClient

    import projection.mapper as mapper
    from app.main import create_app
    from app.state import AppState

    monkeypatch.setattr(mapper, "CALIBRATION_FILE", tmp_path / "projector_calibration.json")
    app = create_app(AppState(settings=settings), start_loop=False)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["loop_running"] is False


def test_status_endpoint_works_with_no_frames(client) -> None:
    """The panel's primary endpoint must return a full payload before the first
    frame -- otherwise a cold start looks like a crash."""
    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["running"] is False
    assert body["detections"]["balls"] == 0
    assert body["performance"]["fps"] == 0.0
    # Vision, physics and rendering have landed, so they are no longer
    # advertised as pending; the game modes have not. This asserts the list
    # still reflects reality rather than that any particular stage is missing --
    # so it is expected to be edited as each phase lands.
    assert "detection" not in body["pending_stages"]
    assert "physics" not in body["pending_stages"]
    assert "rendering" not in body["pending_stages"]
    assert "modes" not in body["pending_stages"]
    # Hailo is optional and genuinely unbuilt, so the list is not empty and the
    # panel's "pending" section still has something to prove it works.
    assert "hailo" in body["pending_stages"]


def test_settings_round_trip(client) -> None:
    original = client.get("/api/settings").json()
    updated = client.post(
        "/api/settings", json={"brightness": 42, "physics_accuracy": "fast"}
    ).json()

    assert updated["brightness"] == 42
    assert updated["physics_accuracy"] == "fast"
    # A partial update must leave unmentioned fields alone.
    assert updated["overlay_alpha"] == original["overlay_alpha"]


def test_settings_rejects_out_of_range(client) -> None:
    assert client.post("/api/settings", json={"brightness": 500}).status_code == 422


def test_mode_switch(client) -> None:
    assert client.post("/api/mode", json={"mode": "training"}).status_code == 200
    assert client.get("/api/status").json()["current_mode"] == "training"


def test_mode_rejects_unknown_value(client) -> None:
    assert client.post("/api/mode", json={"mode": "shuffleboard"}).status_code == 422


def test_calibration_status_uncalibrated(client) -> None:
    body = client.get("/api/calibration/status").json()
    assert body["is_calibrated"] is False
    assert body["alignment_quality"] == "uncalibrated"


def test_calibration_corner_recording_is_idempotent(client) -> None:
    """Re-recording a corner replaces it, so a user who nudges the target three
    times contributes one point rather than three skewed near-duplicates."""
    payload = {"camera_px": [200, 100], "projector_px": [120, 80]}
    client.post("/api/calibration/corner/top_left", json=payload)
    client.post("/api/calibration/corner/top_left", json=payload)
    assert client.get("/api/calibration/status").json()["corners_recorded"] == 1


def test_calibration_corner_rejects_unknown_name(client) -> None:
    response = client.post(
        "/api/calibration/corner/middle_left",
        json={"camera_px": [0, 0], "projector_px": [0, 0]},
    )
    assert response.status_code == 400


def test_finalize_reports_failure_without_enough_corners(client) -> None:
    """Returns success=False with readable text rather than raising -- this
    message is shown verbatim to a non-technical user."""
    body = client.post("/api/calibration/finalize").json()
    assert body["success"] is False
    assert "4 corners" in body["message"]


def test_training_result_before_any_attempt(client) -> None:
    """has_result=False rather than 404: the panel polls this continuously."""
    body = client.get("/api/training/result").json()
    assert body["has_result"] is False


def test_start_drill_blocked_before_the_first_frame(client) -> None:
    """503, not 500 -- "not ready yet" must be distinguishable from "broken".

    Two things can block a drill today: the game modes are unbuilt, and with the
    vision loop disabled there is no frame to pick a drill against. Which one
    fires depends on which phase has landed, so the assertion is on the promise
    -- 503 plus a message naming a real blocker -- rather than on one wording
    that would have to be edited as each phase lands.
    """
    response = client.post("/api/training/start_drill", json={"drill_type": "potting"})
    assert response.status_code == 503
    detail = response.json()["detail"].lower()
    assert "frame" in detail or "modes" in detail, detail


def test_reset_endpoint(client) -> None:
    assert client.post("/api/reset").json()["success"] is True


def test_control_panel_is_served(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "AR Pool Table" in response.text


# ---------------------------------------------------------------------------
# Camera preview / threshold tuning tool
# ---------------------------------------------------------------------------


def test_tuner_seeds_from_config(settings: Settings) -> None:
    """A tuning session must resume from config, not from hardcoded defaults."""
    from tools.camera_preview import ThresholdTuner

    settings.vision.felt_hue_range = (40, 70)
    settings.vision.felt_sat_min = 55
    tuner = ThresholdTuner(settings)
    assert (tuner.hue_low, tuner.hue_high) == (40, 70)
    assert tuner.sat_min == 55


def test_tuner_clamps_to_opencv_hue_scale(settings: Settings) -> None:
    """Hue is 0-179 in OpenCV, not 0-359 -- the usual source of confusion when
    copying values out of a colour picker."""
    from tools.camera_preview import ThresholdTuner

    tuner = ThresholdTuner(settings)
    tuner.widen_hue(500)
    assert tuner.hue_low == 0
    assert tuner.hue_high == 179

    tuner.adjust_saturation(-1000)
    assert tuner.sat_min == 0
    tuner.adjust_saturation(1000)
    assert tuner.sat_min == 255


def test_tuner_narrowing_cannot_invert_the_range(settings: Settings) -> None:
    """An inverted range makes cv2.inRange silently return an empty mask, which
    reads as "no felt found" rather than as a bad threshold."""
    from tools.camera_preview import ThresholdTuner

    tuner = ThresholdTuner(settings)
    for _ in range(200):
        tuner.widen_hue(-1)
    assert tuner.hue_low <= tuner.hue_high


def test_tuner_finds_felt_in_a_synthetic_frame(settings: Settings) -> None:
    """End-to-end check of the mask against a known green rectangle.

    The mock camera's felt is drawn to sit inside the default hue range, so this
    also guards that pairing -- if either the mock's felt colour or the shipped
    default drifts, the tuning tool silently stops finding anything.
    """
    from tools.camera_preview import ThresholdTuner
    from vision.camera import Camera

    with Camera(settings.camera) as camera:
        frame = camera.capture_frame()
    assert frame is not None

    tuner = ThresholdTuner(settings)
    mask = tuner.mask(frame.image)
    assert mask.shape == frame.image.shape[:2]

    coverage = tuner.coverage_pct(mask)
    assert 40.0 < coverage < 90.0, f"expected to find the felt, got {coverage:.1f}% coverage"

    bounds = tuner.largest_contour_bounds(mask)
    assert bounds is not None
    _x, _y, w, h = bounds
    # The mock's felt is a landscape rectangle, so the found region must be too.
    assert w > h


def test_tuner_reports_no_region_on_a_blank_frame(settings: Settings) -> None:
    """No felt must yield None, not an exception -- a covered lens is a normal
    thing to happen while someone is setting up."""
    from tools.camera_preview import ThresholdTuner

    tuner = ThresholdTuner(settings)
    black = np.zeros((240, 320, 3), dtype=np.uint8)
    mask = tuner.mask(black)
    assert tuner.coverage_pct(mask) == 0.0
    assert tuner.largest_contour_bounds(mask) is None


def test_tuner_yaml_is_valid_and_round_trips(settings: Settings) -> None:
    """The printed block is meant to be pasted into config.yaml, so it has to
    parse and validate there."""
    import yaml

    from tools.camera_preview import ThresholdTuner

    tuner = ThresholdTuner(settings)
    tuner.widen_hue(3)
    parsed = yaml.safe_load(tuner.as_yaml())
    assert parsed["vision"]["felt_hue_range"] == [tuner.hue_low, tuner.hue_high]

    # Must survive Settings validation, not merely parse as YAML.
    revalidated = Settings(vision=parsed["vision"])
    assert tuple(revalidated.vision.felt_hue_range) == (tuner.hue_low, tuner.hue_high)


def test_hud_does_not_mutate_the_source_frame(settings: Settings) -> None:
    """Snapshots are saved from the clean frame, so the HUD must draw on a copy."""
    from tools.camera_preview import ThresholdTuner, draw_hud
    from utils.performance import PerformanceTracker

    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    original = frame.copy()
    tuner = ThresholdTuner(settings)
    annotated = draw_hud(
        frame, PerformanceTracker(), "mock", 0.5, tuner, tuner.mask(frame)
    )
    assert np.array_equal(frame, original), "draw_hud mutated its input"
    assert not np.array_equal(annotated, original)


def test_compose_mask_view_preserves_shape() -> None:
    from tools.camera_preview import compose_mask_view

    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[50:150, 60:200] = 255
    view = compose_mask_view(frame, mask)
    assert view.shape == frame.shape
    # Masked region stays bright, unmasked is dimmed -- that contrast is the
    # entire diagnostic value of the view.
    assert view[100, 100].mean() > view[10, 10].mean()


def test_preview_report_mode_runs_headless(settings: Settings, tmp_path) -> None:
    """Smoke test of the whole tool via its argv entry point."""
    from tools.camera_preview import main

    assert main(["--mock", "--report", "--seconds", "0.5", "--log-level", "ERROR"]) == 0
