"""Tests for Phase 4 rendering: themes, primitives, overlays, effects, patterns.

Rendering is mostly judged by eye, so these tests deliberately do not assert
what anything looks like -- a test pinning the exact pixel at (400, 300) would
break on every design tweak and catch nothing. What they assert is the part that
is *not* a matter of taste:

* **Geometry and arithmetic.** Dash patterns walking arc length, smoothing
  converging, table angles becoming projector angles, anchors landing outside
  the cushions.
* **Invariants that are physically meaningful.** Nothing is drawn off-canvas.
  A ghost ball is a ball's radius. The overlay stays RGBA uint8.
* **State machines over time.** Effects expire, trails trim by age, one pot
  produces one celebration. All driven by an injected clock rather than by
  sleeping, which is why every entry point takes ``now``.
* **The failure modes that would blank the projection.** An empty prediction, a
  degenerate calibration, a ball with no table position, a theme that does not
  exist. Each has to yield a usable overlay rather than an exception, because
  the vision loop catches exceptions per stage and the visible result of one is
  a black table.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from app.config import BALL_RADIUS_IN, Settings
from app.models import (
    AlignmentError,
    Ball,
    BallColor,
    BallKind,
    CalibrationState,
    GameModeName,
    GameSession,
    GameState,
    ImpactEvent,
    Player,
    ShotPrediction,
    Vec2,
)
from physics.models import MAX_TIP_OFFSET
from physics.simulator import aim_angle_for_pocket, simulate_shot_fan
from projection import draw
from projection.effects import (
    MAX_EFFECTS,
    POCKET_CAPTURE_IN,
    TRAIL_SECONDS,
    CollisionBurst,
    EffectContext,
    EffectSystem,
    ScorePopup,
    ease_in_out_cubic,
    ease_out_back,
    ease_out_cubic,
)
from projection.mapper import ProjectionMapper, identity_calibration

# Aliased: pytest tries to collect any module-level name starting with "Test"
# as a test class, and warns when it cannot.
from projection.patterns import TestPattern as Pattern
from projection.patterns import render_test_pattern
from projection.renderer import (
    TrajectorySmoother,
    blend_overlay,
    draw_tip_contact_target,
    rail_anchor,
    render_calibration_overlay,
    render_game_ui,
    render_training_overlay,
    render_trajectory_overlay,
    smooth_path,
)
from projection.themes import (
    DEFAULT_THEME,
    THEMES,
    ball_display_color,
    dim,
    get_theme,
    mix,
    palette_rgb,
    resolve_theme,
    theme_names,
)


@pytest.fixture()
def settings() -> Settings:
    """Defaults with mock hardware, and a small projector frame.

    640x360 rather than 1080p: every assertion here is about geometry, which is
    resolution-independent, and a 9x smaller canvas makes the whole module run
    in a fraction of the time. The one test that cares about real resolution
    says so.
    """
    s = Settings()
    s.camera.use_mock = True
    s.projector.use_mock = True
    s.projector.width = 640
    s.projector.height = 360
    return s


@pytest.fixture()
def mapper(settings: Settings) -> ProjectionMapper:
    """Identity calibration: the table stretched to fill the projector frame.

    Deliberately not a homography. It makes the expected pixel position of any
    table point computable by hand, so a failure points at the code under test
    rather than at the fixture.
    """
    return ProjectionMapper(identity_calibration(settings))


@pytest.fixture()
def game_state(settings: Settings) -> GameState:
    """Cue ball plus two object balls, all with table positions."""
    balls = [
        Ball(
            id="cue",
            center_px=Vec2(0.0, 0.0),
            radius_px=12.0,
            color=BallColor.WHITE,
            kind=BallKind.CUE,
            table_pos=Vec2(19.0, 19.0),
        ),
        Ball(
            id="ball_03",
            center_px=Vec2(0.0, 0.0),
            radius_px=12.0,
            color=BallColor.RED,
            kind=BallKind.SOLID,
            number=3,
            table_pos=Vec2(50.0, 14.0),
        ),
        Ball(
            id="ball_09",
            center_px=Vec2(0.0, 0.0),
            radius_px=12.0,
            color=BallColor.YELLOW,
            kind=BallKind.STRIPE,
            number=9,
            table_pos=Vec2(60.0, 26.0),
        ),
    ]
    return GameState(
        timestamp=0.0, frame_index=1, balls=balls, cue_ball=balls[0], confidence=0.9
    )


@pytest.fixture()
def prediction(settings: Settings, game_state: GameState) -> ShotPrediction:
    """A real simulated shot, not a hand-written one.

    Hand-built predictions are the wrong fixture here: they cannot reproduce the
    shapes the renderer actually has to cope with -- an event-driven path with
    long straight runs, a variable number of impacts, ball paths of differing
    lengths -- and every one of those has been a source of drawing bugs.
    """
    from physics.simulator import simulate_shot

    return simulate_shot(
        Vec2(19.0, 19.0),
        -10.0,
        70.0,
        other_balls=game_state.object_balls(),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


def test_every_theme_avoids_colours_the_projector_cannot_make(settings: Settings) -> None:
    """No palette colour may be near-black.

    Not a style rule but a physical one: the projector adds light, so an RGB
    near zero projects nothing at all. A theme with a dark colour in it has an
    invisible element, and the failure looks like a rendering bug rather than a
    palette mistake.
    """
    for name, theme in THEMES.items():
        for field in (
            "cue_path",
            "object_path",
            "impact",
            "ghost_ball",
            "pocket_highlight",
            "text",
            "trail",
            "accent",
            "alert",
        ):
            color = getattr(theme, field)
            assert max(color) >= 120, f"{name}.{field}={color} is too dark to project"


def test_classic_theme_follows_config_and_presets_do_not(settings: Settings) -> None:
    """The documented YAML colour fields must actually do something.

    ``classic`` is the configurable theme, so editing ``render.cue_path_color``
    has to change what is drawn. The presets deliberately ignore it -- picking
    ``neon`` should give neon, not whatever the YAML says -- and that asymmetry
    is surprising enough to be worth pinning.
    """
    settings.render.cue_path_color = (11, 22, 33)

    settings.render.theme = "classic"
    assert resolve_theme(settings).cue_path == (11, 22, 33)

    settings.render.theme = "neon"
    assert resolve_theme(settings).cue_path == THEMES["neon"].cue_path


def test_unknown_theme_falls_back_rather_than_raising() -> None:
    """A typo in config must not stop the projection.

    ``get_theme`` is called every frame from the render path. Raising would turn
    a one-character config error into a black table.
    """
    assert get_theme("does_not_exist").name == DEFAULT_THEME
    assert DEFAULT_THEME in theme_names()


def test_palette_feeds_self_projection_rejection(settings: Settings) -> None:
    """Detection's overlay-rejection palette must track the active theme.

    The cue detector rejects bright lines matching the overlay's own colours. If
    that list came from config while the renderer drew from a theme, switching
    themes would silently make the projected trajectory detectable as a cue --
    which is the exact failure the rejection exists to prevent.
    """
    from vision.detection import _overlay_colors_bgr

    settings.render.theme = "neon"
    bgr = _overlay_colors_bgr(settings)
    neon_cue = THEMES["neon"].cue_path
    # BGR, so the channels come back reversed.
    assert any(np.allclose(c, [neon_cue[2], neon_cue[1], neon_cue[0]]) for c in bgr)
    assert palette_rgb(THEMES["neon"])[0] == neon_cue


def test_ball_paths_get_distinguishable_colours(settings: Settings, game_state: GameState) -> None:
    """Two differently coloured balls must not render as the same line colour.

    This regressed once: an even blend between the palette and the ball tint
    landed both a red and a yellow ball on nearly the same grey, because
    averaging two pale colours of different hue desaturates. A four-ball
    prediction then read as four identical lines.
    """
    theme = resolve_theme(settings)
    red, yellow = game_state.balls[1], game_state.balls[2]
    red_rgb = ball_display_color(red, theme)
    yellow_rgb = ball_display_color(yellow, theme)
    distance = math.dist(red_rgb, yellow_rgb)
    assert distance > 60, f"{red_rgb} and {yellow_rgb} are only {distance:.0f} apart"
    # The cue ball is exempt: it is the aiming line and always uses cue_path.
    assert ball_display_color(game_state.balls[0], theme) == theme.cue_path


def test_colour_arithmetic_clamps() -> None:
    assert mix((0, 0, 0), (100, 200, 255), 0.5) == (50, 100, 128)
    assert mix((10, 10, 10), (20, 20, 20), -5.0) == (10, 10, 10)  # t clamped to 0
    assert dim((200, 200, 200), 4.0) == (255, 255, 255)
    assert dim((200, 200, 200), -1.0) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def test_dash_segments_walk_arc_length_not_vertices() -> None:
    """Dash length must be constant around a corner.

    The naive implementation alternates whole polyline segments, which makes the
    dash length depend on where physics happened to put its vertices -- so a
    cushion rebound would visibly change the rhythm of the line at the rail.
    """
    # An L: 100 px across, then 100 px down, with the corner mid-dash.
    path = [Vec2(0.0, 0.0), Vec2(100.0, 0.0), Vec2(100.0, 100.0)]
    dashes = draw.dash_segments(path, dash_px=20.0, gap_px=10.0, phase_px=0.0)

    assert len(dashes) == pytest.approx(200 / 30, abs=1)
    lengths = [draw.polyline_length(d) for d in dashes]
    # Every dash is 20 px except possibly the last, which the path runs out on.
    for length in lengths[:-1]:
        assert length == pytest.approx(20.0, abs=1.5)
    # And at least one dash must span the corner, i.e. have three points.
    assert any(len(d) == 3 for d in dashes), "no dash bridged the corner"


def test_dash_phase_shifts_the_pattern() -> None:
    """Advancing the phase moves the dashes, which is the crawl animation."""
    path = [Vec2(0.0, 0.0), Vec2(300.0, 0.0)]
    at_zero = draw.dash_segments(path, 20.0, 10.0, phase_px=0.0)
    at_half = draw.dash_segments(path, 20.0, 10.0, phase_px=15.0)
    assert not np.allclose(at_zero[1][0], at_half[1][0])
    # A whole period later the pattern must be back where it started, or the
    # animation would drift instead of cycling.
    at_period = draw.dash_segments(path, 20.0, 10.0, phase_px=30.0)
    assert np.allclose(at_zero[1][0], at_period[1][0])


def test_non_finite_points_are_dropped_not_clipped() -> None:
    """A NaN from the mapper must not become a line to the frame edge.

    ``table_to_projector_batch`` returns NaN for points on the transform's
    horizon. Clipping one to the edge would draw a trajectory into a corner the
    ball is not going anywhere near, which reads as a confident wrong prediction
    -- worse than a short line.
    """
    points = np.array([[10.0, 10.0], [np.nan, 20.0], [30.0, 30.0], [40.0, np.inf]])
    cleaned = draw._points_array(points)
    assert len(cleaned) == 2
    assert cleaned.tolist() == [[10, 10], [30, 30]]


def test_ensure_canvas_reuses_and_replaces(settings: Settings) -> None:
    """The buffer is reused when it fits and replaced when the shape changes.

    Replaced rather than rejected: a changed shape means the projector
    resolution was changed over the API, and raising would turn a settings
    tweak into a crash inside the vision loop.
    """
    canvas = draw.new_canvas(settings)
    canvas[:] = 200
    same = draw.ensure_canvas(canvas, settings)
    assert same is canvas
    assert not same.any(), "reused buffer was not zeroed"

    settings.projector.width = 320
    replaced = draw.ensure_canvas(canvas, settings)
    assert replaced is not canvas
    assert replaced.shape == (settings.projector.height, 320, 4)


def test_text_anchors_position_the_box(settings: Settings) -> None:
    """Right- and centre-anchored text must actually shift left."""
    canvas = draw.new_canvas(settings)
    width, _ = draw.draw_text(canvas, "SCORE", Vec2(300.0, 100.0), (255, 255, 255), anchor="tl")
    left_cols = np.flatnonzero(canvas[:, :, 3].any(axis=0))

    canvas.fill(0)
    draw.draw_text(canvas, "SCORE", Vec2(300.0, 100.0), (255, 255, 255), anchor="tr")
    right_cols = np.flatnonzero(canvas[:, :, 3].any(axis=0))

    assert left_cols.min() >= 295
    assert right_cols.max() <= 305
    assert width > 0


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def test_pixels_per_inch_matches_the_identity_stretch(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """Local scale is the mean of the two axis scales under an affine transform."""
    expected = (
        settings.projector.width / settings.table.length_in
        + settings.projector.height / settings.table.width_in
    ) / 2.0
    assert mapper.pixels_per_inch() == pytest.approx(expected, rel=1e-6)


def test_pixels_per_inch_varies_across_a_keystoned_table(settings: Settings) -> None:
    """Under a homography the scale genuinely differs end to end.

    This is why the renderer measures a local derivative instead of reading
    ``scale_x`` off the calibration: with a single global number, a ghost ball at
    the far end of a keystoned projection would be drawn the same size as one at
    the near end, when the real difference is 10-20%.
    """
    from projection.mapper import solve_projector_homography

    length, width = settings.table.length_in, settings.table.width_in
    # Squeeze the far short rail toward the middle: a plain trapezoid keystone.
    calibration = solve_projector_homography(
        [Vec2(0, 0), Vec2(length, 0), Vec2(length, width), Vec2(0, width)],
        [Vec2(0, 0), Vec2(600, 90), Vec2(600, 270), Vec2(0, 360)],
        settings.projector.width,
        settings.projector.height,
    )
    keystoned = ProjectionMapper(calibration)
    near = keystoned.pixels_per_inch(Vec2(2.0, width / 2.0))
    far = keystoned.pixels_per_inch(Vec2(length - 2.0, width / 2.0))
    assert near > far * 1.2, f"expected a visible scale gradient, got {near:.2f} vs {far:.2f}"


def test_rail_anchors_sit_outside_the_cushions(settings: Settings) -> None:
    """UI anchors must map beyond the playing surface, not onto it.

    The spec's "zero interference with natural play" makes this a hard
    constraint: anything inside the cushions is occluded by balls and hands. The
    check is done with a calibration that leaves room outside the table, because
    the identity one has none -- which is the case the clamping exists for.
    """
    # Table drawn into the middle 60% of the frame, leaving rails visible.
    calibration = identity_calibration(settings)
    calibration.scale_x *= 0.6
    calibration.scale_y *= 0.6
    calibration.offset_x = settings.projector.width * 0.2
    calibration.offset_y = settings.projector.height * 0.2
    roomy = ProjectionMapper(calibration)

    table_top = roomy.table_to_projector(Vec2(settings.table.length_in / 2.0, 0.0))
    table_bottom = roomy.table_to_projector(Vec2(settings.table.length_in / 2.0, settings.table.width_in))

    assert rail_anchor(roomy, "top_center", settings).y < table_top.y
    assert rail_anchor(roomy, "bottom_center", settings).y > table_bottom.y


def test_rail_anchors_clamp_into_frame_when_there_is_no_room(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """With an identity calibration the rails are off-frame, so anchors clamp.

    Clamping rather than dropping the UI: on an uncalibrated setup the score
    ends up on the cloth near the rail, which looks worse but is still readable.
    A vanished scoreboard reads as a broken system.
    """
    for where in ("top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right"):
        point = rail_anchor(mapper, where, settings)
        assert 0 <= point.x < settings.projector.width
        assert 0 <= point.y < settings.projector.height


def test_rail_anchor_rejects_an_unknown_name(settings: Settings, mapper: ProjectionMapper) -> None:
    with pytest.raises(ValueError, match="unknown anchor"):
        rail_anchor(mapper, "middle_middle", settings)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def test_smoothing_off_returns_the_input() -> None:
    current = [Vec2(0.0, 0.0), Vec2(10.0, 10.0)]
    assert smooth_path([Vec2(5.0, 5.0), Vec2(5.0, 5.0)], current, 0) == current


def test_smoothing_converges_and_never_freezes() -> None:
    """Repeated smoothing must approach the target, however high the setting.

    100% smoothing would freeze the aiming line, which is a bug however the user
    got there -- so the weight is capped just below. The test asserts the cap by
    checking that even at 100 the line arrives.
    """
    target = [Vec2(100.0, 0.0)]
    path = [Vec2(0.0, 0.0)]
    for _ in range(200):
        path = smooth_path(path, target, 100)
    assert path[0].x == pytest.approx(100.0, abs=1.0)


def test_smoothing_takes_a_new_tail_verbatim() -> None:
    """A newly appeared rebound segment must snap in at full length.

    The prediction changes length whenever the collision set changes. The tail
    beyond the shared prefix has nothing to smooth against, and averaging it
    toward something arbitrary would make the line lurch exactly when the player
    has just found a new rebound.
    """
    previous = [Vec2(0.0, 0.0)]
    current = [Vec2(10.0, 0.0), Vec2(20.0, 0.0), Vec2(30.0, 0.0)]
    smoothed = smooth_path(previous, current, 80)
    assert len(smoothed) == 3
    assert smoothed[0].x == pytest.approx(2.0)  # smoothed against the old point
    assert smoothed[1] == Vec2(20.0, 0.0)  # verbatim
    assert smoothed[2] == Vec2(30.0, 0.0)


def test_smoothing_handles_a_shrinking_path() -> None:
    """A prediction that gets shorter must not raise or keep stale points."""
    previous = [Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(2.0, 0.0)]
    smoothed = smooth_path(previous, [Vec2(10.0, 0.0)], 50)
    assert len(smoothed) == 1


def test_smoother_forgets_between_shots(settings: Settings) -> None:
    """Dropping the remembered path is what stops the line sweeping across.

    Without it, the first frame of a new aim is smoothed against the previous
    shot's path, and the overlay takes half a second to slide over from
    wherever the last player was aiming.
    """
    smoother = TrajectorySmoother(settings)
    smoother.apply("cue", [Vec2(0.0, 0.0)])
    smoother.forget("cue")
    assert smoother.apply("cue", [Vec2(50.0, 0.0)]) == [Vec2(50.0, 0.0)]


# ---------------------------------------------------------------------------
# Trajectory overlay
# ---------------------------------------------------------------------------


def test_empty_prediction_yields_a_blank_canvas(
    settings: Settings, mapper: ProjectionMapper, game_state: GameState
) -> None:
    """An un-aimed table is the normal resting case, not an error."""
    canvas = render_trajectory_overlay(ShotPrediction(), game_state, mapper, settings)
    assert canvas.shape == (settings.projector.height, settings.projector.width, 4)
    assert not canvas.any()


def test_trajectory_overlay_draws_and_reuses_the_buffer(
    settings: Settings,
    mapper: ProjectionMapper,
    game_state: GameState,
    prediction: ShotPrediction,
) -> None:
    """The overlay must draw something, and must not allocate when handed a buffer."""
    assert not prediction.is_empty
    buffer = draw.new_canvas(settings)
    canvas = render_trajectory_overlay(
        prediction, game_state, mapper, settings, buffer, now=0.0
    )
    assert canvas is buffer
    assert canvas.dtype == np.uint8
    painted = int((canvas[:, :, 3] > 0).sum())
    assert painted > 500, f"only {painted} px drawn for a {len(prediction.trajectory_path)}-point path"

    # A second call must clear the first: a stale overlay accumulating frame on
    # frame would fill the felt with light within a second.
    stale = canvas.copy()
    render_trajectory_overlay(ShotPrediction(), game_state, mapper, settings, buffer)
    assert not np.array_equal(stale, buffer)
    assert not buffer.any()


def test_trajectory_overlay_survives_a_prediction_for_unknown_balls(
    settings: Settings, mapper: ProjectionMapper, game_state: GameState
) -> None:
    """A path for a ball that is no longer detected must not raise.

    Routine, not exotic: physics runs on frame N's ball list and the renderer
    draws with frame N+1's, so a ball the detector has just lost track of
    arrives with a path and no matching ``Ball``.
    """
    prediction = ShotPrediction(
        trajectory_path=[Vec2(10.0, 10.0), Vec2(40.0, 20.0)],
        ball_paths={"ghost_ball_99": [Vec2(40.0, 20.0), Vec2(60.0, 30.0)]},
        final_positions={"ghost_ball_99": Vec2(60.0, 30.0)},
    )
    canvas = render_trajectory_overlay(prediction, game_state, mapper, settings, now=0.0)
    assert canvas.any()


def test_trajectory_overlay_animates_over_time(
    settings: Settings,
    mapper: ProjectionMapper,
    game_state: GameState,
    prediction: ShotPrediction,
) -> None:
    """The dash pattern must move with the clock.

    The crawl is what makes a dashed line read as directional rather than as a
    row of ticks, and it is the one part of the overlay that is invisible in a
    still frame -- so it needs a test rather than an eyeball.
    """
    early = render_trajectory_overlay(prediction, game_state, mapper, settings, now=0.0)
    later = render_trajectory_overlay(prediction, game_state, mapper, settings, now=0.12)
    assert not np.array_equal(early, later)


def test_pocketed_balls_get_a_highlight_not_a_ghost(
    settings: Settings, mapper: ProjectionMapper, game_state: GameState
) -> None:
    """A potted ball has no resting position, so no ghost is drawn for it.

    Drawing one would put a ghost ball inside the pocket, claiming the ball
    comes to rest somewhere it physically cannot.
    """
    pocket = Vec2(settings.table.length_in, 0.0)
    prediction = ShotPrediction(
        trajectory_path=[Vec2(19.0, 19.0), Vec2(50.0, 10.0)],
        ball_paths={"ball_03": [Vec2(50.0, 14.0), pocket]},
        final_positions={"ball_03": pocket},
        pocketed_ball_ids=["ball_03"],
    )
    settings.render.show_ghost_balls = True
    with_pot = render_trajectory_overlay(prediction, game_state, mapper, settings, now=0.0)

    prediction.pocketed_ball_ids = []
    without_pot = render_trajectory_overlay(prediction, game_state, mapper, settings, now=0.0)
    assert not np.array_equal(with_pot, without_pot)


def test_impact_angle_labels_respect_the_setting(
    settings: Settings,
    mapper: ProjectionMapper,
    game_state: GameState,
) -> None:
    """``show_impact_angles`` must actually suppress the numbers."""
    prediction = ShotPrediction(
        trajectory_path=[Vec2(19.0, 19.0), Vec2(50.0, 14.0)],
        impact_points=[
            ImpactEvent(
                position=Vec2(48.0, 14.0),
                target_id="ball_03",
                incoming_angle_deg=0.0,
                outgoing_angle_deg=35.0,
            )
        ],
    )
    settings.render.show_impact_angles = True
    labelled = render_trajectory_overlay(prediction, game_state, mapper, settings, now=0.0)
    settings.render.show_impact_angles = False
    plain = render_trajectory_overlay(prediction, game_state, mapper, settings, now=0.0)
    assert int((labelled[:, :, 3] > 0).sum()) > int((plain[:, :, 3] > 0).sum())


def test_ghost_balls_are_drawn_at_true_ball_radius(
    settings: Settings, mapper: ProjectionMapper, game_state: GameState
) -> None:
    """A ghost must be a ball's size, or it misrepresents whether the ball fits.

    Measured rather than trusted: the player compares the ghost against the real
    balls next to it, so a ghost a few pixels large reads as a prediction that
    the ball will end up somewhere it will not actually fit.
    """
    settings.render.show_ghost_balls = True
    # Kept clear of the cue path's row on purpose: the aiming line runs along
    # y=19 here, and scanning a row it also occupies would measure the line.
    ghost_at = Vec2(settings.table.length_in / 2.0, 8.0)
    prediction = ShotPrediction(
        trajectory_path=[Vec2(19.0, 19.0), Vec2(60.0, 19.0)],
        final_positions={"ball_03": ghost_at},
    )
    canvas = render_trajectory_overlay(prediction, game_state, mapper, settings, now=0.0)

    center_px = mapper.table_to_projector(ghost_at)
    row = canvas[int(round(center_px.y)), :, 3]
    painted = np.flatnonzero(row > 0)
    expected_diameter = 2.0 * BALL_RADIUS_IN * mapper.pixels_per_inch()
    measured = painted.max() - painted.min()
    assert measured == pytest.approx(expected_diameter, abs=6)


# ---------------------------------------------------------------------------
# Game UI, training and calibration overlays
# ---------------------------------------------------------------------------


def test_game_ui_draws_scoreboard_within_the_frame(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """Every UI element must land inside the output frame.

    An element drawn off-canvas is silently discarded by OpenCV, so a layout bug
    here shows up as "the score is missing" with no error anywhere.
    """
    session = GameSession(
        mode=GameModeName.KING_OF_THE_HILL,
        players=[Player("ALICE", 42), Player("BOB", 31, is_eliminated=True)],
        current_player_index=0,
        combo_count=3,
    )
    canvas = render_game_ui(
        session, mapper, settings, seconds_remaining=8.0, feedback_text="NICE SHOT", now=0.0
    )
    assert canvas.any()
    # Something drawn in each half, i.e. the layout is spread rather than piled
    # into one corner.
    assert canvas[: settings.projector.height // 2, :, 3].any()
    assert canvas[settings.projector.height // 2 :, :, 3].any()


def test_game_ui_handles_an_empty_session(settings: Settings, mapper: ProjectionMapper) -> None:
    """A session with no players must render, not divide by zero.

    This is the state at startup and after a reset, which is precisely when the
    projection needs to be working.
    """
    canvas = render_game_ui(GameSession(), mapper, settings, now=0.0)
    assert canvas.shape[2] == 4


def test_timer_changes_colour_as_it_runs_out(settings: Settings, mapper: ProjectionMapper) -> None:
    """Colour is doing the work, not the number.

    A player mid-shot registers "it went red", not "it says 7", so the colour
    bands are the feature and worth pinning.
    """
    session = GameSession(players=[Player("A")])
    plenty = render_game_ui(session, mapper, settings, seconds_remaining=45.0, now=0.0)
    scarce = render_game_ui(session, mapper, settings, seconds_remaining=4.0, now=0.0)

    def dominant(canvas: np.ndarray) -> tuple[int, int, int]:
        mask = canvas[:, :, 3] > 100
        return tuple(int(v) for v in canvas[:, :, :3][mask].mean(axis=0))

    assert dominant(plenty) != dominant(scarce)


def test_training_overlay_distinguishes_target_from_user_aim(
    settings: Settings, mapper: ProjectionMapper, game_state: GameState
) -> None:
    """The two lines must differ in more than hue.

    Colour alone is not enough under projector light on cloth, so the target is
    dashed and the live aim is solid. The check is that removing either one
    changes the picture -- both are actually being drawn.
    """
    target = ShotPrediction(trajectory_path=[Vec2(19.0, 19.0), Vec2(60.0, 19.0)])
    user = ShotPrediction(trajectory_path=[Vec2(19.0, 19.0), Vec2(60.0, 30.0)])

    both = render_training_overlay(
        game_state, target, user, mapper, "CLOSE", settings, now=0.0
    )
    target_only = render_training_overlay(game_state, target, None, mapper, "", settings, now=0.0)
    user_only = render_training_overlay(game_state, None, user, mapper, "", settings, now=0.0)

    assert both.any() and target_only.any() and user_only.any()
    assert int((both[:, :, 3] > 0).sum()) > int((target_only[:, :, 3] > 0).sum())
    assert not np.array_equal(target_only, user_only)


def test_training_overlay_with_nothing_to_draw(
    settings: Settings, mapper: ProjectionMapper, game_state: GameState
) -> None:
    """No drill loaded and no cue detected is a valid state."""
    canvas = render_training_overlay(game_state, None, None, mapper, "", settings, now=0.0)
    assert not canvas.any()


def test_calibration_overlay_draws_corner_targets_without_a_mapper(settings: Settings) -> None:
    """Step 4 must project in raw projector space.

    Establishing the transform is the goal of that step, so applying one would
    be circular -- the user would be aligning targets that had already been
    moved by the calibration they are trying to produce.
    """
    canvas = render_calibration_overlay(CalibrationState(step=4), None, None, settings, now=0.0)
    assert canvas.any()


def test_calibration_overlay_marks_progress_through_the_corners(settings: Settings) -> None:
    """A recorded corner must look different from a pending one."""
    none_done = render_calibration_overlay(
        CalibrationState(step=4, corner_errors=[]), None, None, settings, now=0.0
    )
    two_done = render_calibration_overlay(
        CalibrationState(step=4, corner_errors=[1.0, 2.0]), None, None, settings, now=0.0
    )
    assert not np.array_equal(none_done, two_done)


def test_calibration_overlay_shows_the_alignment_message(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """``AlignmentError.message`` is written for a human and must be rendered."""
    state = CalibrationState(step=6)
    quiet = render_calibration_overlay(state, mapper, None, settings, now=0.0)
    loud = render_calibration_overlay(
        state,
        mapper,
        AlignmentError(total_rmse=18.0, message="Move projector left 3 inches", severity="error"),
        settings,
        now=0.0,
    )
    assert int((loud[:, :, 3] > 0).sum()) > int((quiet[:, :, 3] > 0).sum())


def test_early_calibration_steps_stay_off_the_felt(settings: Settings) -> None:
    """Steps 1-3 are camera-side, so they project only an instruction.

    Marks on the felt during them would land in the detector's way while it is
    trying to find the table.
    """
    canvas = render_calibration_overlay(CalibrationState(step=2), None, None, settings, now=0.0)
    painted = int((canvas[:, :, 3] > 0).sum())
    assert 0 < painted < canvas[:, :, 3].size // 20


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


def test_easing_curves_stay_in_bounds() -> None:
    """Every curve must map 0->0 and 1->1 and stay clamped outside that.

    ``ease_out_back`` overshoots 1 in the middle by design -- that is the pop --
    so only its endpoints are pinned.
    """
    for curve in (ease_out_cubic, ease_in_out_cubic, ease_out_back):
        assert curve(0.0) == pytest.approx(0.0, abs=1e-9)
        assert curve(1.0) == pytest.approx(1.0, abs=1e-9)
        assert curve(-3.0) == pytest.approx(0.0, abs=1e-9)
        assert curve(9.0) == pytest.approx(1.0, abs=1e-9)
    for t in (0.1, 0.5, 0.9):
        assert 0.0 <= ease_out_cubic(t) <= 1.0


def test_effects_expire_on_their_own_clock(settings: Settings, mapper: ProjectionMapper) -> None:
    """A finished effect must be dropped, or the list grows without bound."""
    system = EffectSystem(settings)
    system.spawn_collision(Vec2(30.0, 19.0), outgoing_deg=45.0, now=0.0)
    assert system.update(now=0.1) == 1
    assert system.update(now=5.0) == 0


def test_effect_cap_drops_the_oldest(settings: Settings) -> None:
    """A flapping detector must not be able to grow the effect list forever.

    The oldest is dropped because it has least left to show, so its loss is the
    least visible.
    """
    system = EffectSystem(settings)
    for i in range(MAX_EFFECTS + 10):
        system.spawn_collision(Vec2(10.0 + i % 50, 19.0), now=0.0)
    assert system.active_count == MAX_EFFECTS


def test_trails_are_trimmed_by_age_not_by_count(settings: Settings) -> None:
    """A trail must stay half a second long however fast the loop is running.

    Trimmed by count instead, a trail would get *longer* in wall-clock terms
    exactly when the system is struggling and the frame rate has dropped.
    """
    system = EffectSystem(settings)
    ball = Ball(
        id="cue",
        center_px=Vec2(0.0, 0.0),
        radius_px=12.0,
        kind=BallKind.CUE,
        table_pos=Vec2(10.0, 19.0),
    )
    # 60 samples over 2 seconds: far more than TRAIL_SECONDS worth.
    for i in range(60):
        ball.table_pos = Vec2(10.0 + i, 19.0)
        system.observe(
            GameState(timestamp=i / 30.0, frame_index=i, balls=[ball], cue_ball=ball),
            now=i / 30.0,
            detect_collisions=False,
        )
    trail = system._trails["cue"]
    span = trail.samples[-1][1] - trail.samples[0][1]
    assert span <= TRAIL_SECONDS + 1e-6
    assert len(trail.samples) < 60


def test_collision_is_detected_from_a_sharp_direction_change(settings: Settings) -> None:
    """A rebound must spawn a burst without the mode layer reporting it.

    This is the fallback that makes the effects work before Phase 7 lands. The
    ball is walked in a straight line and then turned hard, which is what a
    cushion contact looks like from the outside.
    """
    system = EffectSystem(settings)
    ball = Ball(
        id="cue", center_px=Vec2(0.0, 0.0), radius_px=12.0, kind=BallKind.CUE, table_pos=Vec2(10.0, 19.0)
    )
    positions = [Vec2(10.0 + i * 4.0, 19.0) for i in range(4)]
    positions += [Vec2(22.0 - i * 4.0, 19.0 + i * 3.0) for i in range(1, 4)]
    for i, position in enumerate(positions):
        ball.table_pos = position
        system.observe(
            GameState(timestamp=i / 30.0, frame_index=i, balls=[ball], cue_ball=ball),
            now=i / 30.0,
        )
    assert system.active_count >= 1
    assert any(isinstance(e, CollisionBurst) for e in system._effects)


def test_slow_jitter_does_not_spawn_collisions(settings: Settings) -> None:
    """Detection noise on a nearly stationary ball must not fire bursts.

    Without the speed floor, a settled table produces a burst every frame -- the
    position estimate wobbles by a fraction of an inch and the *heading* of that
    wobble is essentially random.
    """
    system = EffectSystem(settings)
    ball = Ball(
        id="cue", center_px=Vec2(0.0, 0.0), radius_px=12.0, kind=BallKind.CUE, table_pos=Vec2(10.0, 19.0)
    )
    for i in range(30):
        # A few hundredths of an inch in an arbitrary direction each frame.
        ball.table_pos = Vec2(10.0 + 0.03 * (i % 3), 19.0 + 0.03 * ((i + 1) % 3))
        system.observe(
            GameState(timestamp=i / 30.0, frame_index=i, balls=[ball], cue_ball=ball),
            now=i / 30.0,
        )
    assert system.active_count == 0


def test_a_ball_vanishing_at_a_pocket_is_celebrated_once(settings: Settings) -> None:
    """The detector removes a potted ball rather than flagging it.

    So absence *plus* proximity to a pocket is the observable. And it must fire
    once: a per-frame celebration for as long as the ball stays gone would be a
    permanent vortex.
    """
    system = EffectSystem(settings)
    near_pocket = Vec2(settings.table.length_in - 1.0, 1.0)
    ball = Ball(
        id="ball_03",
        center_px=Vec2(0.0, 0.0),
        radius_px=12.0,
        kind=BallKind.SOLID,
        table_pos=near_pocket,
    )
    system.observe(GameState(timestamp=0.0, frame_index=0, balls=[ball]), now=0.0)

    empty = GameState(timestamp=1.0, frame_index=1, balls=[])
    for i in range(5):
        system.observe(empty, now=1.0 + i * 0.1)
    assert system.active_count >= 1
    vortex_count = sum(1 for e in system._effects if type(e).__name__ == "PocketVortex")
    assert vortex_count == 1


def test_a_ball_vanishing_mid_table_is_not_celebrated(settings: Settings) -> None:
    """An occlusion is not a pot.

    A hand or a leaning player hides balls constantly, and a burst every time
    someone reaches across the table would be unusable. The distance test is
    what separates the two.
    """
    system = EffectSystem(settings)
    middle = Vec2(settings.table.length_in / 2.0, settings.table.width_in / 2.0)
    assert min(p.distance_to(middle) for p in system._pockets) > POCKET_CAPTURE_IN
    ball = Ball(
        id="ball_03", center_px=Vec2(0.0, 0.0), radius_px=12.0, kind=BallKind.SOLID, table_pos=middle
    )
    system.observe(GameState(timestamp=0.0, frame_index=0, balls=[ball]), now=0.0)
    for i in range(5):
        system.observe(GameState(timestamp=1.0, frame_index=1, balls=[]), now=1.0 + i * 0.1)
    assert system.active_count == 0


def test_a_pot_does_not_invent_a_score(settings: Settings) -> None:
    """Points are a game rule, so the renderer must not make one up.

    ``spawn_pocket`` only produces the floating number when a caller passes
    one. Getting this wrong would put "+10" on the felt in a mode where a ball
    is worth 25, or worth nothing.
    """
    system = EffectSystem(settings)
    system.spawn_pocket(Vec2(38.0, 0.0), now=0.0)
    assert not any(isinstance(e, ScorePopup) for e in system._effects)

    system.spawn_pocket(Vec2(38.0, 0.0), points=25, now=0.0)
    popups = [e for e in system._effects if isinstance(e, ScorePopup)]
    assert len(popups) == 1
    assert popups[0].text == "+25"


@pytest.mark.parametrize("pocket_y", [0.0, 38.0])
def test_score_popups_stay_on_canvas_at_either_long_rail(
    settings: Settings, mapper: ProjectionMapper, pocket_y: float
) -> None:
    """A popup must remain visible for its whole life, at every pocket.

    Regression: the popup originally drifted along a fixed screen direction,
    which is fine at one long rail and sends it straight off the canvas at the
    other -- and OpenCV discards off-canvas drawing silently, so the celebration
    for half the pockets simply never appeared. It now drifts inboard.
    """
    system = EffectSystem(settings)
    system.spawn_score(Vec2(settings.table.length_in / 2.0, pocket_y), "+10", now=0.0)
    ctx = EffectContext.build(mapper, settings)

    # Sample across the popup's life, including the end where it has drifted
    # furthest, which is exactly where the old behaviour failed.
    for t in (0.05, 0.25, 0.5, 0.75, 0.95):
        canvas = draw.new_canvas(settings)
        system.render(canvas, ctx, now=t, trails=False)
        assert canvas.any(), f"popup at y={pocket_y} vanished at t={t}"


def test_clear_resets_the_potted_memory(settings: Settings) -> None:
    """Re-racking must not leave every ball permanently un-celebratable.

    One pot is one vortex, which needs a memory of what has already been
    celebrated -- and that memory has to be dropped on a reset, or the second
    game of the evening is silent.
    """
    system = EffectSystem(settings)
    ball = Ball(
        id="ball_03",
        center_px=Vec2(0.0, 0.0),
        radius_px=12.0,
        kind=BallKind.SOLID,
        table_pos=Vec2(0.0, 0.0),
        pocketed=True,
    )
    state = GameState(timestamp=0.0, frame_index=0, balls=[ball])
    system.observe(state, now=0.0)
    assert system.active_count == 1

    system.clear()
    assert system.active_count == 0
    system.observe(state, now=1.0)
    assert system.active_count == 1


def test_effects_draw_inside_the_canvas(settings: Settings, mapper: ProjectionMapper) -> None:
    """Effects spawned at a pocket must not be clipped away entirely.

    A corner pocket is at the very edge of the table, so an effect there is the
    case most likely to be drawn off-canvas and silently discarded.
    """
    system = EffectSystem(settings)
    system.spawn_pocket(Vec2(0.0, 0.0), points=10, now=0.0)
    ctx = EffectContext.build(mapper, settings)
    canvas = draw.new_canvas(settings)
    system.render(canvas, ctx, now=0.15)
    assert canvas.any()


def test_balls_without_table_positions_are_skipped(settings: Settings) -> None:
    """Before the homography is known, ``table_pos`` is ``None``.

    Effects work only in table coordinates, and falling back to camera pixels
    would animate a burst in a completely wrong place -- a camera x of 900 read
    as inches is well off the end of the table.
    """
    system = EffectSystem(settings)
    ball = Ball(id="cue", center_px=Vec2(900.0, 500.0), radius_px=12.0, kind=BallKind.CUE)
    system.observe(GameState(timestamp=0.0, frame_index=0, balls=[ball]), now=0.0)
    assert system.trail_count == 0


# ---------------------------------------------------------------------------
# Test patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", list(Pattern))
def test_every_pattern_draws_something(
    pattern: Pattern, settings: Settings, mapper: ProjectionMapper
) -> None:
    canvas = render_test_pattern(pattern, mapper, settings)
    assert canvas.shape == (settings.projector.height, settings.projector.width, 4)
    assert int((canvas[:, :, 3] > 0).sum()) > 200


def test_patterns_work_with_no_mapper(settings: Settings) -> None:
    """Called before any calibration exists, which is the common case.

    A test pattern is often the thing being used to decide whether a saved
    calibration is any good, so needing one to render would be circular.
    """
    assert render_test_pattern(Pattern.GRID, None, settings).any()


def test_pattern_accepts_its_string_value(settings: Settings, mapper: ProjectionMapper) -> None:
    """So a CLI flag or an API field maps straight in."""
    assert render_test_pattern("corners", mapper, settings).any()
    with pytest.raises(ValueError):
        render_test_pattern("does_not_exist", mapper, settings)


def test_patterns_stack_on_a_shared_canvas(settings: Settings, mapper: ProjectionMapper) -> None:
    """Passing a canvas must not clear it, so overlays can compose.

    The calibration wizard's step 6 draws the grid and the corner brackets
    together. If the second call zeroed the buffer, the first would silently
    vanish.
    """
    canvas = render_test_pattern(Pattern.GRID, mapper, settings)
    grid_px = int((canvas[:, :, 3] > 0).sum())
    render_test_pattern(Pattern.CORNERS, mapper, settings, canvas=canvas)
    assert int((canvas[:, :, 3] > 0).sum()) > grid_px


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


def test_blend_overlay_converts_rgba_to_bgr() -> None:
    """The overlay is RGBA and a camera frame is BGR, so channels must swap.

    Skipping the swap is invisible on greys and wrong on everything else, and it
    matters here specifically: the whole purpose of this preview is judging
    whether what is projected matches what was intended.
    """
    overlay = np.zeros((4, 4, 4), dtype=np.uint8)
    overlay[:, :, 0] = 255  # pure red, in RGB
    overlay[:, :, 3] = 255
    blended = blend_overlay(np.zeros((4, 4, 3), dtype=np.uint8), overlay, alpha=1.0)
    assert tuple(blended[0, 0]) == (0, 0, 255), "expected BGR red"


def test_blend_overlay_scales_by_per_pixel_alpha() -> None:
    """A fading effect must fade, not snap to the global opacity."""
    base = np.full((4, 4, 3), 100, dtype=np.uint8)
    overlay = np.zeros((4, 4, 4), dtype=np.uint8)
    overlay[:, :, :3] = 200
    overlay[:, :, 3] = 128
    assert blend_overlay(base, overlay, alpha=1.0)[0, 0, 0] == pytest.approx(150, abs=2)


# ---------------------------------------------------------------------------
# Cue-ball consequence: the post-contact path and the power ticks
# ---------------------------------------------------------------------------


@pytest.fixture()
def fan_prediction(settings: Settings, game_state: GameState) -> ShotPrediction:
    """A real fan prediction, aimed at the object ball in ``game_state``."""
    cue = game_state.cue_ball
    assert cue is not None and cue.table_pos is not None
    target = game_state.balls[1]
    assert target.table_pos is not None
    # A cut into a pocket, not a shot at the ball's centre. Aimed dead straight
    # the cue ball stuns to a halt and there is no post-contact path at all --
    # correct physics, and it leaves nothing for these tests to look at.
    aim = aim_angle_for_pocket(
        cue.table_pos, target.table_pos, Vec2(settings.table.length_in, 0.0)
    )
    return simulate_shot_fan(
        cue.table_pos, aim, [b for b in game_state.balls if b is not cue], settings
    )


def test_no_cue_ball_ghost_is_drawn_without_a_prescribed_power(
    settings: Settings,
    mapper: ProjectionMapper,
    game_state: GameState,
    fan_prediction: ShotPrediction,
) -> None:
    """Nothing may mark the cue ball's resting place when power is unknown.

    The regression this locks down was live on the felt: the overlay drew a
    ghost at ``final_positions["cue"]`` in every mode, and freeplay reached it
    through ``default_power`` -- a value that free-rolls the cue ball thirteen
    table lengths. A confidently wrong resting place costs trust in every other
    mark on the table, so the absence of one is worth a test.

    Checked by pixel count against the same prediction with the cue's resting
    entry removed: a ghost outline is a ring at true ball radius, so drawing one
    would show up as a clear difference. Comparing rendered output rather than
    inspecting internals keeps the test honest about what reaches the projector.
    """
    assert "cue" in fan_prediction.final_positions, "fixture must exercise the path"

    with_cue = render_trajectory_overlay(
        fan_prediction, game_state, mapper, settings, now=0.0
    )
    painted_with_cue = int((with_cue[:, :, 3] > 0).sum())

    stripped = replace(
        fan_prediction,
        final_positions={
            k: v for k, v in fan_prediction.final_positions.items() if k != "cue"
        },
    )
    without_cue = render_trajectory_overlay(
        stripped, game_state, mapper, settings, now=0.0
    )
    painted_without_cue = int((without_cue[:, :, 3] > 0).sum())

    assert painted_with_cue == painted_without_cue


def test_power_ticks_reach_the_canvas(
    settings: Settings,
    mapper: ProjectionMapper,
    game_state: GameState,
    fan_prediction: ShotPrediction,
) -> None:
    """A prediction carrying ticks must paint more than one without them.

    The ticks are the replacement for the cue ghost, so "we removed the ghost"
    and "we drew nothing instead" have to be distinguishable.
    """
    assert fan_prediction.power_ticks

    with_ticks = render_trajectory_overlay(
        fan_prediction, game_state, mapper, settings, now=0.0
    )
    painted_with = int((with_ticks[:, :, 3] > 0).sum())

    bare = replace(fan_prediction, power_ticks=[], envelope_path=[])
    without_ticks = render_trajectory_overlay(
        bare, game_state, mapper, settings, now=0.0
    )
    painted_without = int((without_ticks[:, :, 3] > 0).sum())

    assert painted_with > painted_without


def test_a_prescribed_level_draws_more_than_an_unprescribed_fan(
    settings: Settings, mapper: ProjectionMapper, game_state: GameState
) -> None:
    """Prescribing a power adds the ghost outline and the highlight.

    The visible difference between "pick one of these" and "hit it this hard".
    With nothing prescribed there is no ghost, because there is nothing to be
    confident about; with a level prescribed the ghost is the whole point.
    """
    cue = game_state.cue_ball
    assert cue is not None and cue.table_pos is not None
    target = game_state.balls[1]
    assert target.table_pos is not None
    aim = aim_angle_for_pocket(
        cue.table_pos, target.table_pos, Vec2(settings.table.length_in, 0.0)
    )
    others = [b for b in game_state.balls if b is not cue]

    loose = simulate_shot_fan(cue.table_pos, aim, others, settings)
    strict = simulate_shot_fan(
        cue.table_pos, aim, others, settings, prescribed_bucket=2
    )

    loose_px = int(
        (render_trajectory_overlay(loose, game_state, mapper, settings, now=0.0)[:, :, 3] > 0).sum()
    )
    strict_px = int(
        (render_trajectory_overlay(strict, game_state, mapper, settings, now=0.0)[:, :, 3] > 0).sum()
    )
    assert strict_px > loose_px


def test_post_contact_path_is_lighter_than_the_aiming_line(
    settings: Settings,
    mapper: ProjectionMapper,
    game_state: GameState,
    fan_prediction: ShotPrediction,
) -> None:
    """Aim and consequence must not read as equally authoritative.

    The aiming line is what the player controls and lines up with; the
    post-contact path follows from the shot. Drawing them at the same weight
    would be the overlay claiming to know as much about where the cue ball ends
    up as about where it is pointed -- and it knows considerably less, because
    distance depends on a power nothing measured.

    Compared as mean alpha along each half rather than as a peak, since the
    consequence path fades along its length and its brightest pixel sits right at
    the contact point where the two nearly meet.
    """
    assert fan_prediction.contact_index > 0
    canvas = render_trajectory_overlay(
        fan_prediction, game_state, mapper, settings, now=0.0
    )

    aim = fan_prediction.trajectory_path[: fan_prediction.contact_index + 1]
    consequence = fan_prediction.post_contact_path
    assert len(consequence) >= 2

    assert _mean_alpha_along(canvas, aim, mapper) > _mean_alpha_along(
        canvas, consequence, mapper
    )


def _mean_alpha_along(
    canvas: np.ndarray, path: list[Vec2], mapper: ProjectionMapper
) -> float:
    """Mean alpha sampled at a path's midpoints, ignoring unpainted pixels.

    Sampled at segment midpoints rather than at the path's own points: the
    vertices are where cushion contacts and impact markers are drawn, and those
    marks belong to neither half of the line.
    """
    samples: list[int] = []
    for a, b in zip(path, path[1:]):  # noqa: B905 -- pairwise, so unequal by design
        mid = mapper.table_to_projector(Vec2((a.x + b.x) / 2.0, (a.y + b.y) / 2.0))
        x, y = int(round(mid.x)), int(round(mid.y))
        window = canvas[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3, 3]
        lit = window[window > 0]
        if lit.size:
            samples.append(int(lit.max()))
    assert samples, "path was not drawn at all"
    return sum(samples) / len(samples)


# ---------------------------------------------------------------------------
# Tip contact target
# ---------------------------------------------------------------------------


def test_tip_diagram_marks_a_different_spot_for_draw_and_follow(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """Top and bottom must land on opposite sides of the diagram's centre.

    The one thing the diagram absolutely has to get right. A drill saying "hit
    it low" beside a mark drawn high teaches the opposite of what it means, and
    the player has no way to tell which of the two is wrong.

    Compared as the vertical centre of mass of the painted pixels: the diagram is
    otherwise symmetric, so the contact mark and its arrow are what move it.
    """
    center = Vec2(38.0, 19.0)
    follow = draw_tip_contact_target(
        draw.new_canvas(settings), center, Vec2(0.0, 0.45), mapper, settings
    )
    draw_shot = draw_tip_contact_target(
        draw.new_canvas(settings), center, Vec2(0.0, -0.45), mapper, settings
    )

    # +y is up on the ball and down in projector pixels, so follow must sit at
    # the smaller row index.
    assert _alpha_centroid_y(follow) < _alpha_centroid_y(draw_shot)


def test_tip_diagram_marks_a_different_spot_for_left_and_right(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """Right-hand and left-hand english must not draw the same picture."""
    center = Vec2(38.0, 19.0)
    right = draw_tip_contact_target(
        draw.new_canvas(settings), center, Vec2(0.45, 0.0), mapper, settings
    )
    left = draw_tip_contact_target(
        draw.new_canvas(settings), center, Vec2(-0.45, 0.0), mapper, settings
    )
    assert _alpha_centroid_x(right) > _alpha_centroid_x(left)


def test_tip_diagram_clamps_past_the_miscue_limit(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """An unplayable offset must draw as the furthest playable one.

    Past about half a radius the tip slides off the ball. Drawing a mark out at
    the edge would be showing the player a miscue as though it were a shot.
    """
    center = Vec2(38.0, 19.0)
    at_limit = draw_tip_contact_target(
        draw.new_canvas(settings), center, Vec2(0.0, MAX_TIP_OFFSET), mapper, settings
    )
    beyond = draw_tip_contact_target(
        draw.new_canvas(settings), center, Vec2(0.0, 4.0), mapper, settings
    )
    assert np.array_equal(at_limit, beyond)


def test_tip_diagram_draws_centre_ball_without_a_spin_arrow(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """A centre-ball prescription still draws, but with no arrow.

    ``Vec2(0, 0)`` is a real instruction -- a drill can be teaching a stun shot
    -- so the diagram appears. But there is no resulting spin, so an arrow would
    be pointing at nothing; it is dropped rather than drawn zero-length.

    Detected by looking outside the face rather than by counting pixels. The
    arrow runs past the outer ring while everything else is contained by it, so
    ink beyond that radius means an arrow -- whereas total pixel counts come out
    near-identical either way, because a centred contact dot lands on top of the
    crosshair it would otherwise have added to.
    """
    center = Vec2(38.0, 19.0)
    centre_ball = draw_tip_contact_target(
        draw.new_canvas(settings), center, Vec2(0.0, 0.0), mapper, settings
    )
    with_spin = draw_tip_contact_target(
        draw.new_canvas(settings), center, Vec2(0.0, 0.45), mapper, settings
    )

    assert centre_ball.any(), "a centre-ball prescription must still draw a diagram"
    assert _ink_outside_face(centre_ball, center, mapper, settings) == 0
    assert _ink_outside_face(with_spin, center, mapper, settings) > 0


def _ink_outside_face(
    canvas: np.ndarray, center: Vec2, mapper: ProjectionMapper, settings: Settings
) -> int:
    """Painted pixels beyond the tip diagram's outer ring, excluding its label."""
    from projection.renderer import TIP_DIAGRAM_RADIUS_IN

    center_px = mapper.table_to_projector(center)
    radius_px = mapper.pixels_per_inch() * TIP_DIAGRAM_RADIUS_IN
    rows = np.arange(canvas.shape[0])[:, None] - center_px.y
    cols = np.arange(canvas.shape[1])[None, :] - center_px.x
    outside = (rows * rows + cols * cols) > (radius_px * 1.08) ** 2
    # The label sits below the face; it is text, not a spin mark.
    outside &= rows < 0
    return int(((canvas[:, :, 3] > 0) & outside).sum())


def test_tip_diagram_stays_on_the_table(settings: Settings, mapper: ProjectionMapper) -> None:
    """Placed near a cushion, the diagram must not be drawn off the cloth.

    The projector cannot draw usefully past the rails, so a diagram anchored to a
    cue ball frozen on a cushion has to be pulled inboard rather than half
    clipped.
    """
    for corner in (Vec2(1.0, 1.0), Vec2(75.0, 37.0), Vec2(1.0, 37.0)):
        canvas = draw_tip_contact_target(
            draw.new_canvas(settings), corner, Vec2(0.0, -0.4), mapper, settings
        )
        # Nothing painted in the outermost row or column of the frame.
        assert not canvas[0, :, 3].any()
        assert not canvas[-1, :, 3].any()
        assert not canvas[:, 0, 3].any()
        assert not canvas[:, -1, 3].any()


def test_training_overlay_draws_the_prescribed_tip_offset(
    settings: Settings, mapper: ProjectionMapper, game_state: GameState
) -> None:
    """Passing a tip offset to the training overlay must add the diagram."""
    prediction = ShotPrediction(
        trajectory_path=[Vec2(19.0, 19.0), Vec2(46.0, 15.0)],
        contact_index=1,
    )
    without = render_training_overlay(
        game_state, prediction, None, mapper, settings=settings, now=0.0
    )
    painted_without = int((without[:, :, 3] > 0).sum())

    with_diagram = render_training_overlay(
        game_state,
        prediction,
        None,
        mapper,
        settings=settings,
        now=0.0,
        tip_offset=Vec2(0.0, -0.4),
    )
    painted_with = int((with_diagram[:, :, 3] > 0).sum())

    assert painted_with > painted_without


def _alpha_centroid_y(canvas: np.ndarray) -> float:
    """Row index of the painted pixels' centre of mass."""
    alpha = canvas[:, :, 3].astype(np.float64)
    total = alpha.sum()
    assert total > 0.0, "nothing was drawn"
    rows = np.arange(alpha.shape[0], dtype=np.float64)[:, None]
    return float((alpha * rows).sum() / total)


def _alpha_centroid_x(canvas: np.ndarray) -> float:
    """Column index of the painted pixels' centre of mass."""
    alpha = canvas[:, :, 3].astype(np.float64)
    total = alpha.sum()
    assert total > 0.0, "nothing was drawn"
    cols = np.arange(alpha.shape[1], dtype=np.float64)[None, :]
    return float((alpha * cols).sum() / total)
