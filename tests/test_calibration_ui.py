"""Tests for Phase 6: the calibration wizard.

The seven screens are loops that block until a human does something, so the
testing strategy is to make the human scriptable and everything they are shown
computable. Two consequences shape this module:

* :class:`calibration_ui.console.Console` takes a list of actions instead of a
  window, so a whole wizard run is a list of strings and an assertion about what
  came out the other end.
* Every number the wizard reports comes from :mod:`calibration_ui.metrics`,
  which touches no hardware. That is where most of these tests are, because
  that is where a bug would silently produce a *plausible* calibration rather
  than a crash.

What is deliberately not asserted: what anything looks like. Pinning the pixel
at (400, 300) would break on every design tweak and catch nothing. What is
asserted is the part that is not a matter of taste -- sign conventions, that
text stays inside the frame, that a copy is a copy, and that the gates which
stop a bad calibration being saved actually stop one.

Three tests here are regressions for bugs found while building this, and each
says so: the tautological end-to-end check, the RMSE-only completion gate, and
the cancel confirmation that could not be cancelled.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.models import AlignmentError, CalibrationState, TableBoundary, Vec2
from calibration_ui import overlay_renderer as ui
from calibration_ui.calibration_app import CalibrationApp, _rebuild_boundary, _score_label
from calibration_ui.console import MAX_BUTTONS_PER_ROW, Button, Console
from calibration_ui.metrics import (
    CORNER_NAMES,
    MarkDetection,
    assign_marks_to_corners,
    compute_alignment_error,
    compute_grid_metrics,
    locate_projected_marks,
    projection_error_in,
    similarity_fit,
    solve_projector_to_camera,
)

FELT_BGR = (60, 110, 60)


@pytest.fixture()
def settings() -> Settings:
    """Defaults with mock hardware, at real camera and projector resolution.

    Not shrunk the way the rendering tests shrink theirs: several assertions
    here are about text fitting inside a frame and about touch-target sizes,
    both of which are only meaningful at the resolution the wizard actually
    runs at.
    """
    s = Settings()
    s.camera.use_mock = True
    s.projector.use_mock = True
    s.projector.warmup_seconds = 0.0
    return s


@pytest.fixture()
def boundary() -> TableBoundary:
    """A slightly keystoned table, so nothing passes by being axis-aligned."""
    return TableBoundary(
        top_left=Vec2(180.0, 120.0),
        top_right=Vec2(1740.0, 140.0),
        bottom_right=Vec2(1760.0, 950.0),
        bottom_left=Vec2(160.0, 930.0),
        center=Vec2(960.0, 535.0),
        width_px=1570.0,
        height_px=810.0,
        confidence=0.93,
    )


def _felt_frame(width: int = 1920, height: int = 1080, noise: float = 0.0) -> np.ndarray:
    """A plain felt-coloured frame, optionally with sensor noise."""
    frame = np.full((height, width, 3), FELT_BGR, dtype=np.uint8)
    if noise > 0.0:
        rng = np.random.default_rng(1234)
        jitter = rng.normal(0.0, noise, frame.shape)
        frame = np.clip(frame.astype(np.float32) + jitter, 0, 255).astype(np.uint8)
    return frame


def _with_marks(frame: np.ndarray, points: list[Vec2], radius: int = 26) -> np.ndarray:
    """Paint bright crosshair marks onto a copy of ``frame``."""
    lit = frame.copy()
    for point in points:
        x, y = point.as_int()
        cv2.circle(lit, (x, y), radius, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.line(lit, (x - radius, y), (x + radius, y), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.line(lit, (x, y - radius), (x, y + radius), (255, 255, 255), 4, cv2.LINE_AA)
    return lit


# ---------------------------------------------------------------------------
# Finding projected marks
# ---------------------------------------------------------------------------


def test_marks_are_found_at_the_positions_they_were_projected_to() -> None:
    dark = _felt_frame(noise=3.0)
    truth = [Vec2(300.0, 250.0), Vec2(1600.0, 260.0), Vec2(1610.0, 830.0), Vec2(310.0, 820.0)]
    lit = _with_marks(dark, truth)

    found = locate_projected_marks(dark, lit)

    assert len(found) == 4
    for expected in truth:
        nearest = min(found, key=lambda m: m.center.distance_to(expected))
        # Two px: the centroid of a symmetric crosshair is its centre, and
        # anything worse than this would eat a fifth of the 20 px RMSE budget
        # before the user has touched the projector.
        assert nearest.center.distance_to(expected) < 2.0


def test_an_unlit_projector_yields_no_marks_rather_than_noise() -> None:
    """The difference of two frames of the same felt is not four corners.

    The failure this guards against is not a crash -- it is
    :func:`locate_projected_marks` confidently returning blobs of sensor noise,
    which would be recorded as correspondences and produce a calibration solved
    from nothing.
    """
    dark = _felt_frame(noise=4.0)
    assert locate_projected_marks(dark, _felt_frame(noise=4.0)) == []


def test_differencing_ignores_things_that_were_bright_all_along() -> None:
    """A chalk cube on the rail is brighter than the marks and is not one."""
    dark = _felt_frame()
    cv2.rectangle(dark, (100, 60), (170, 110), (250, 250, 250), -1)  # chalk, always lit
    lit = _with_marks(dark, [Vec2(900.0, 500.0)])

    found = locate_projected_marks(dark, lit)

    assert len(found) == 1
    assert found[0].center.distance_to(Vec2(900.0, 500.0)) < 2.0


def test_mismatched_frame_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="size mismatch"):
        locate_projected_marks(_felt_frame(), _felt_frame(width=1280, height=720))


# ---------------------------------------------------------------------------
# Assigning marks to corners
# ---------------------------------------------------------------------------


def _mark(x: float, y: float, contrast: float = 200.0) -> MarkDetection:
    return MarkDetection(center=Vec2(x, y), area_px=400.0, contrast=contrast)


def test_marks_are_assigned_by_quadrant(boundary: TableBoundary) -> None:
    marks = [_mark(300, 250), _mark(1600, 260), _mark(1610, 830), _mark(310, 820)]
    assigned = assign_marks_to_corners(marks, boundary)
    assert set(assigned) == set(CORNER_NAMES)
    assert assigned["top_left"] == Vec2(300, 250)
    assert assigned["bottom_right"] == Vec2(1610, 830)


def test_a_badly_aimed_projection_still_assigns_one_mark_per_corner(
    boundary: TableBoundary,
) -> None:
    """Regression for nearest-corner matching, which silently loses a corner.

    All four marks are pulled hard toward the table centre, as they are when the
    projection is much too small. Under nearest-corner matching the top-left and
    bottom-left marks can both be nearest to the same corner, leaving one corner
    with two marks and another with none -- and the wizard then solves a
    "four point" homography from three distinct correspondences.
    """
    marks = [_mark(860, 460), _mark(1060, 465), _mark(1055, 610), _mark(865, 605)]

    assigned = assign_marks_to_corners(marks, boundary)

    assert set(assigned) == set(CORNER_NAMES)
    assert len({(v.x, v.y) for v in assigned.values()}) == 4


def test_the_brightest_mark_in_a_quadrant_wins(boundary: TableBoundary) -> None:
    marks = [_mark(300, 250, contrast=90.0), _mark(340, 300, contrast=240.0)]
    assigned = assign_marks_to_corners(marks, boundary)
    assert assigned["top_left"] == Vec2(340, 300)


def test_an_unlit_corner_is_reported_as_missing_not_guessed(
    boundary: TableBoundary,
) -> None:
    assigned = assign_marks_to_corners([_mark(300, 250)], boundary)
    assert set(assigned) == {"top_left"}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_similarity_fit_recovers_a_known_transform() -> None:
    source = [Vec2(0, 0), Vec2(100, 0), Vec2(100, 50), Vec2(0, 50)]
    angle, scale, shift = math.radians(12.0), 1.35, Vec2(40.0, -25.0)
    target = [
        Vec2(
            (p.x * math.cos(angle) - p.y * math.sin(angle)) * scale + shift.x,
            (p.x * math.sin(angle) + p.y * math.cos(angle)) * scale + shift.y,
        )
        for p in source
    ]

    rotation, recovered_scale, translation = similarity_fit(source, target)

    assert rotation == pytest.approx(12.0, abs=1e-6)
    assert recovered_scale == pytest.approx(1.35, abs=1e-9)
    assert translation.x == pytest.approx(40.0, abs=1e-6)
    assert translation.y == pytest.approx(-25.0, abs=1e-6)


def test_similarity_fit_is_deterministic() -> None:
    """No RANSAC, so an unmoved projector reports an unmoving rotation.

    ``cv2.estimateAffinePartial2D`` defaults to a randomised estimator, and with
    it the fine-tune readout twitches between frames while the user is holding
    perfectly still -- which reads as the rig being unstable.
    """
    source = [Vec2(10, 20), Vec2(300, 25), Vec2(310, 200), Vec2(15, 195)]
    target = [Vec2(p.x + 7.0, p.y - 3.0) for p in source]
    assert similarity_fit(source, target) == similarity_fit(source, target)


def test_similarity_fit_rejects_input_it_cannot_answer_for() -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        similarity_fit([Vec2(0, 0)], [Vec2(1, 1)])
    with pytest.raises(ValueError, match="source points but"):
        similarity_fit([Vec2(0, 0), Vec2(1, 1)], [Vec2(0, 0)])


def test_solving_projector_to_camera_needs_four_distinct_corners() -> None:
    corners = {
        "top_left": (Vec2(100, 100), Vec2(200, 200)),
        "top_right": (Vec2(900, 110), Vec2(1700, 200)),
        "bottom_right": (Vec2(910, 600), Vec2(1700, 900)),
    }
    assert solve_projector_to_camera(corners) is None

    corners["bottom_left"] = (Vec2(105, 590), Vec2(200, 900))
    assert solve_projector_to_camera(corners) is not None


# ---------------------------------------------------------------------------
# Alignment error and its advice
# ---------------------------------------------------------------------------


def _corner_map(points: list[Vec2]) -> dict[str, Vec2]:
    return dict(zip(CORNER_NAMES, points, strict=True))


def test_a_pure_offset_is_reported_in_the_direction_the_marks_must_move() -> None:
    """The sign convention, which is the one thing here that must not be wrong.

    A user sent the wrong way makes the error worse, watches the number grow,
    and concludes the tool is broken. The error vector runs from the projected
    mark to the table corner, so a mark sitting *left* of its corner has to move
    *right*.
    """
    corners = [Vec2(200, 200), Vec2(1700, 200), Vec2(1700, 900), Vec2(200, 900)]
    # Every mark 40 px left of and 20 px above its corner.
    marks = [Vec2(p.x - 40.0, p.y - 20.0) for p in corners]

    error = compute_alignment_error(_corner_map(corners), _corner_map(marks), px_per_inch=20.0)

    assert error.x_offset == pytest.approx(40.0)
    assert error.y_offset == pytest.approx(20.0)
    assert error.total_rmse == pytest.approx(math.hypot(40.0, 20.0))
    assert "right" in error.message and "down" in error.message
    assert "left" not in error.message and " up" not in error.message


def test_rotation_is_advised_before_offset() -> None:
    """A rotated projection cannot be slid into place, so say so first."""
    corners = [Vec2(200, 200), Vec2(1700, 200), Vec2(1700, 900), Vec2(200, 900)]
    center = Vec2(950.0, 550.0)
    angle = math.radians(-4.0)  # marks are rotated; the fit must undo it
    marks = [
        Vec2(
            center.x + (p.x - center.x) * math.cos(angle) - (p.y - center.y) * math.sin(angle) + 30,
            center.y + (p.x - center.x) * math.sin(angle) + (p.y - center.y) * math.cos(angle),
        )
        for p in corners
    ]

    error = compute_alignment_error(_corner_map(corners), _corner_map(marks), px_per_inch=20.0)

    assert "wist" in error.message  # "Twist the projector ..."
    assert error.rotation == pytest.approx(4.0, abs=0.2)


def test_a_projection_that_is_the_wrong_size_is_told_to_zoom() -> None:
    corners = [Vec2(200, 200), Vec2(1700, 200), Vec2(1700, 900), Vec2(200, 900)]
    center = Vec2(950.0, 550.0)
    marks = [Vec2(center.x + (p.x - center.x) * 0.85, center.y + (p.y - center.y) * 0.85) for p in corners]

    error = compute_alignment_error(_corner_map(corners), _corner_map(marks), px_per_inch=20.0)

    assert "Zoom" in error.message and " in " in error.message


def test_a_perfect_alignment_says_so_and_passes() -> None:
    corners = [Vec2(200, 200), Vec2(1700, 200), Vec2(1700, 900), Vec2(200, 900)]
    error = compute_alignment_error(_corner_map(corners), _corner_map(corners), px_per_inch=20.0)
    assert error.severity == "info"
    assert error.total_rmse == pytest.approx(0.0)


def test_invisible_marks_are_an_error_not_a_perfect_score() -> None:
    """Zero shared corners must not average to zero error."""
    corners = _corner_map([Vec2(200, 200), Vec2(1700, 200), Vec2(1700, 900), Vec2(200, 900)])
    error = compute_alignment_error(corners, {}, px_per_inch=20.0)
    assert error.severity == "error"
    assert not math.isfinite(error.total_rmse)


def test_keystone_is_named_when_no_rigid_move_would_fix_it() -> None:
    """Offset, rotation and scale all near zero, yet the corners are off."""
    corners = [Vec2(200, 200), Vec2(1700, 200), Vec2(1700, 900), Vec2(200, 900)]
    # Push the two right-hand marks apart vertically and pull the left pair
    # together: a trapezoid, which no translate/rotate/scale can correct.
    marks = [Vec2(200, 230), Vec2(1700, 170), Vec2(1700, 930), Vec2(200, 870)]

    error = compute_alignment_error(_corner_map(corners), _corner_map(marks), px_per_inch=20.0)

    assert "skewed" in error.message


# ---------------------------------------------------------------------------
# Grid metrics
# ---------------------------------------------------------------------------


def _homography(source: list[Vec2], target: list[Vec2]) -> np.ndarray:
    src = np.array([[p.x, p.y] for p in source], dtype=np.float32)
    dst = np.array([[p.x, p.y] for p in target], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def _camera_to_table(settings: Settings) -> np.ndarray:
    """Camera px -> table inches for a tidy 10 px/in overhead view."""
    length, width = settings.table.length_in, settings.table.width_in
    camera = [Vec2(0, 0), Vec2(length * 10, 0), Vec2(length * 10, width * 10), Vec2(0, width * 10)]
    table = [Vec2(0, 0), Vec2(length, 0), Vec2(length, width), Vec2(0, width)]
    return _homography(camera, table)


def _projector_quad(settings: Settings) -> list[Vec2]:
    pw, ph = float(settings.projector.width), float(settings.projector.height)
    return [Vec2(0, 0), Vec2(pw, 0), Vec2(pw, ph), Vec2(0, ph)]


def test_a_square_projection_covering_the_table_scores_perfectly(settings: Settings) -> None:
    length, width = settings.table.length_in, settings.table.width_in
    onto_table = [Vec2(0, 0), Vec2(length * 10, 0), Vec2(length * 10, width * 10), Vec2(0, width * 10)]
    projector_to_camera = _homography(_projector_quad(settings), onto_table)

    grid = compute_grid_metrics(projector_to_camera, _camera_to_table(settings), settings)

    assert grid.perpendicularity_deg == pytest.approx(90.0, abs=0.01)
    assert grid.rotation_deg == pytest.approx(0.0, abs=0.01)
    assert grid.coverage_x_pct == pytest.approx(100.0)
    assert grid.coverage_y_pct == pytest.approx(100.0)
    assert grid.is_acceptable


def test_a_rotated_projector_is_measured_as_rotated(settings: Settings) -> None:
    length, width = settings.table.length_in, settings.table.width_in
    center = Vec2(length * 5, width * 5)
    angle = math.radians(8.0)
    square = [Vec2(0, 0), Vec2(length * 10, 0), Vec2(length * 10, width * 10), Vec2(0, width * 10)]
    rotated = [
        Vec2(
            center.x + (p.x - center.x) * math.cos(angle) - (p.y - center.y) * math.sin(angle),
            center.y + (p.x - center.x) * math.sin(angle) + (p.y - center.y) * math.cos(angle),
        )
        for p in square
    ]
    projector_to_camera = _homography(_projector_quad(settings), rotated)

    grid = compute_grid_metrics(projector_to_camera, _camera_to_table(settings), settings)

    # A pure rotation is still square; only the alignment to the rails changed.
    assert grid.perpendicularity_deg == pytest.approx(90.0, abs=0.1)
    assert abs(grid.rotation_deg) == pytest.approx(8.0, abs=0.2)
    assert not grid.is_square


def test_a_projection_that_reaches_half_the_table_reports_half_coverage(
    settings: Settings,
) -> None:
    length, width = settings.table.length_in, settings.table.width_in
    half = [
        Vec2(0, 0),
        Vec2(length * 5, 0),
        Vec2(length * 5, width * 10),
        Vec2(0, width * 10),
    ]
    projector_to_camera = _homography(_projector_quad(settings), half)

    grid = compute_grid_metrics(projector_to_camera, _camera_to_table(settings), settings)

    assert grid.coverage_x_pct == pytest.approx(50.0, abs=1.5)
    assert grid.coverage_y_pct == pytest.approx(100.0)
    assert not grid.covers_table
    assert not grid.is_acceptable


# ---------------------------------------------------------------------------
# The end-to-end check
# ---------------------------------------------------------------------------


def test_projection_error_measures_the_gap_between_rings_and_balls() -> None:
    balls = [Vec2(400, 300), Vec2(900, 500), Vec2(1300, 700)]
    marks = [Vec2(b.x + 20.0, b.y) for b in balls]  # every ring 20 px off, 1 in at 20 px/in

    check = projection_error_in(balls, marks, px_per_inch=20.0, max_match_px=100.0)

    assert check.matched == 3
    assert check.all_matched
    assert check.mean_error_in == pytest.approx(1.0)


def test_projection_error_is_not_a_tautology() -> None:
    """Regression: the first implementation could not report a nonzero error.

    It composed the calibration with the measured projector-to-camera
    homography, both of which are solved from the same four correspondences --
    and four points determine a homography, so the composition is the
    table-to-camera transform exactly, everywhere. It returned 0.0 for every
    input including a projector aimed at the floor.

    The check that matters is therefore this one: a displaced mark must produce
    a nonzero number.
    """
    balls = [Vec2(500, 400), Vec2(1000, 600)]
    perfect = projection_error_in(balls, list(balls), px_per_inch=20.0, max_match_px=100.0)
    displaced = projection_error_in(
        balls, [Vec2(b.x + 35.0, b.y + 15.0) for b in balls], px_per_inch=20.0, max_match_px=100.0
    )

    assert perfect.mean_error_in == pytest.approx(0.0)
    assert displaced.mean_error_in > 1.0


def test_a_ring_that_missed_its_ball_entirely_is_unmatched_not_averaged_in() -> None:
    """The count, not the mean, is what catches a projection landing elsewhere."""
    balls = [Vec2(400, 300), Vec2(900, 500), Vec2(1300, 700)]
    marks = [Vec2(405, 302), Vec2(1800, 1000)]  # one good, one nowhere near

    check = projection_error_in(balls, marks, px_per_inch=20.0, max_match_px=60.0)

    assert check.matched == 1
    assert check.balls_checked == 3
    assert not check.all_matched
    # The mean over what matched is excellent, which is exactly why a caller
    # must read all_matched first.
    assert check.mean_error_in < 0.5


def test_no_marks_at_all_is_infinite_error_not_zero() -> None:
    check = projection_error_in([Vec2(400, 300)], [], px_per_inch=20.0, max_match_px=60.0)
    assert not math.isfinite(check.mean_error_in)
    assert check.matched == 0
    assert not check.all_matched


# ---------------------------------------------------------------------------
# Console: layout and input
# ---------------------------------------------------------------------------


def _buttons(count: int) -> list[Button]:
    return [Button(f"B{i}", f"action_{i}", key=chr(ord("a") + i)) for i in range(count)]


@pytest.mark.parametrize("count", [1, 3, 5, 6, 8, 10])
def test_button_rects_do_not_overlap_and_stay_on_canvas(
    settings: Settings, count: int
) -> None:
    console = Console(settings, script=[]).open()
    canvas = console.show(_felt_frame(), _buttons(count))
    rects = console._button_rects(count)

    assert len(rects) == count
    for x0, y0, x1, y1 in rects:
        assert 0 <= x0 < x1 <= canvas.shape[1]
        assert 0 <= y0 < y1 <= canvas.shape[0]
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            separated = a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
            assert separated, f"buttons {a} and {b} overlap"


def test_buttons_wrap_rather_than_shrinking_past_a_thumb(settings: Settings) -> None:
    console = Console(settings, script=[]).open()
    console.show(_felt_frame(), _buttons(MAX_BUTTONS_PER_ROW + 3))
    rects = console._button_rects(MAX_BUTTONS_PER_ROW + 3)
    rows = {y0 for _x0, y0, _x1, _y1 in rects}
    assert len(rows) == 2
    # Balanced, not filled: 8 buttons come out 4 and 4.
    assert len({y0 for _x, y0, _x1, _y1 in rects[:4]}) == 1


def test_clicking_a_button_returns_its_action(settings: Settings) -> None:
    console = Console(settings, script=[]).open()
    buttons = _buttons(4)
    console.show(_felt_frame(), buttons)

    for index, (x0, y0, x1, y1) in enumerate(console._button_rects(4)):
        console._on_mouse(cv2.EVENT_LBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
        assert console._pending_action == buttons[index].action
        console._pending_action = None


def test_a_disabled_button_ignores_clicks(settings: Settings) -> None:
    console = Console(settings, script=[]).open()
    console.show(_felt_frame(), [Button("Nope", "boom", "n", enabled=False)])
    x0, y0, x1, y1 = console._button_rects(1)[0]
    console._on_mouse(cv2.EVENT_LBUTTONDOWN, (x0 + x1) // 2, (y0 + y1) // 2, 0, None)
    assert console._pending_action is None


def test_a_click_in_the_view_comes_back_in_camera_pixels(settings: Settings) -> None:
    """The console downscales the camera frame; a click must scale back up.

    Getting this wrong places a corner tens of pixels out, which is a
    calibration that is quietly wrong rather than visibly broken -- so the
    console is fixed-size specifically to make this a single division.
    """
    console = Console(settings, script=[]).open()
    frame = _felt_frame(width=1920, height=1080)
    canvas = console.show(frame, _buttons(3))
    scale = canvas.shape[1] / 1920

    console._on_mouse(cv2.EVENT_LBUTTONDOWN, int(640 * scale), int(360 * scale), 0, None)
    point = console.take_click()

    assert point is not None
    assert point.x == pytest.approx(640, abs=2)
    assert point.y == pytest.approx(360, abs=2)
    assert console.take_click() is None  # consumed, so one click is one corner


def test_enter_activates_the_primary_button(settings: Settings) -> None:
    console = Console(settings).open()
    console.headless = False
    console._buttons = [Button("No", "no", "o"), Button("Yes", "yes", "y", primary=True)]
    assert console._key_action(ord("\r")) == "yes"


def test_an_exhausted_script_cancels_rather_than_hanging(settings: Settings) -> None:
    """Regression guard for the harness itself.

    A screen polls in a loop, so a console that returns ``None`` forever turns
    every headless test into a hang rather than a failure.
    """
    console = Console(settings, script=["next"]).open()
    assert console.poll() == "next"
    assert console.poll() == "cancel"
    assert console.poll() == "cancel"


def test_scripted_clicks_carry_a_position(settings: Settings) -> None:
    console = Console(settings, script=["click:123,456"]).open()
    assert console.poll() == "click"
    assert console.take_click() == Vec2(123.0, 456.0)


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------

_ALIGNMENT = AlignmentError(
    total_rmse=18.2,
    x_offset=-12.0,
    y_offset=8.0,
    rotation=1.2,
    message="Nudge the projector so the marks move 1.2 in right.",
    severity="warning",
)


def _overlay_calls(boundary: TableBoundary, settings: Settings) -> list[tuple[str, object]]:
    state = CalibrationState(step=4, table_boundary=boundary)
    return [
        ("draw_table_outline", lambda f: ui.draw_table_outline(f, boundary)),
        ("draw_alignment_grid", lambda f: ui.draw_alignment_grid(f, boundary, settings=settings)),
        ("draw_corner_target", lambda f: ui.draw_corner_target(f, Vec2(400, 400), "TOP LEFT", True)),
        ("draw_alignment_feedback", lambda f: ui.draw_alignment_feedback(f, _ALIGNMENT, settings)),
        (
            "draw_projected_vs_detected",
            lambda f: ui.draw_projected_vs_detected(f, [Vec2(210, 210)], [Vec2(200, 200)]),
        ),
        (
            "render_step_instructions",
            lambda f: ui.render_step_instructions(f, state, "Do the thing.", settings),
        ),
        ("draw_checklist", lambda f: ui.draw_checklist(f, "TITLE", [("item", True), ("other", None)])),
        ("draw_confidence_bar", lambda f: ui.draw_confidence_bar(f, "Confidence", 0.92)),
        ("draw_metric_rows", lambda f: ui.draw_metric_rows(f, "T", [("a", "1", "info")])),
        ("draw_countdown", lambda f: ui.draw_countdown(f, 47.0, 120.0, "Warming up")),
        ("draw_banner_message", lambda f: ui.draw_banner_message(f, "Working...")),
        ("draw_notice", lambda f: ui.draw_notice(f, "Something to know", "warning")),
    ]


def _margin_ink(view: np.ndarray, *, skip: int = 6, band: int = 20) -> int:
    """Brightest pixel in a band just inside the right edge of the frame.

    Rendered onto a black frame so the only bright things are the wizard's own
    marks. The outermost ``skip`` columns are excluded because the verdict panel
    draws a severity-coloured border the full width of the frame -- that border
    is meant to reach the edge, and including it would make every one of these
    assertions fail for the wrong reason. Text reaching into the band inside it
    is an overflow.
    """
    return int(view[:, -(skip + band) : -skip].max())


def test_every_overlay_returns_a_copy_and_leaves_the_frame_alone(
    boundary: TableBoundary, settings: Settings
) -> None:
    """The input frame also goes to the detector; drawing on it corrupts detection."""
    for name, call in _overlay_calls(boundary, settings):
        frame = _felt_frame()
        original = frame.copy()
        out = call(frame)
        assert out is not frame, f"{name} drew in place"
        assert out.shape == frame.shape and out.dtype == frame.dtype, name
        assert np.array_equal(frame, original), f"{name} modified its input"


@pytest.mark.parametrize(
    "instruction",
    [
        "Done.",
        "Tap the bottom right mark, or arrow it onto the bottom right cushion nose.",
        "The projection is skewed. Raise or lower the front of the projector, then re-check "
        "every corner before continuing to the next step of the wizard.",
    ],
)
def test_instructions_of_any_length_stay_inside_the_frame(
    boundary: TableBoundary, settings: Settings, instruction: str
) -> None:
    """Text is sized from the frame height, so long sentences must shrink.

    A clipped instruction is worse than a small one: the user cannot act on the
    half they cannot see, and the wizard's whole job is telling them what to do.
    """
    # Black frame: the only bright pixels are then the wizard's own marks.
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    state = CalibrationState(step=4, table_boundary=boundary)

    view = ui.render_step_instructions(frame, state, instruction, settings)
    banner = view[: int(frame.shape[0] * 0.30)]

    # The progress pips live in this band too, so compare against the same
    # banner with no instruction rather than against zero.
    empty = ui.render_step_instructions(frame, state, "", settings)[: banner.shape[0]]
    assert _margin_ink(banner) <= _margin_ink(empty)


#: Every string the wizard can put in a headline, longest first. Collected from
#: the advice branches in ``metrics`` and the screen text in ``calibration_app``
#: so the fitting is tested against what the user will really see.
WIZARD_HEADLINES = [
    "The projection is skewed. Raise or lower the front of the projector.",
    "The projection does not line up with the table. Check that the projector is "
    "square to the table and try again.",
    "Running on simulated hardware -- this calibration will not be usable.",
    "Tap the bottom right mark, or arrow it onto the bottom right cushion nose.",
    "Adjust the projector so the white rectangle covers the whole table.",
    "Nudge the projector so the marks move 12.5 in right and 3.2 in down.",
    "Alignment looks excellent.",
    "Done.",
]


@pytest.mark.parametrize("message", WIZARD_HEADLINES)
def test_every_headline_the_wizard_can_show_fits_the_width(message: str) -> None:
    """Wrapping, not just shrinking, is what keeps a long sentence readable.

    Shrinking alone bottoms out: the fallback advice strings are wide enough at
    headline size to overflow a 1080p frame several times over, and below about
    40% of that size the text stops being readable across the dim room the
    wizard is designed for. Asserted on the fitting itself rather than on
    pixels, because the verdict panel draws a severity border the full width of
    the frame and there is no margin left to inspect.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    budget = frame.shape[1] - 64

    lines, scale = ui._fit_lines(message, ui._scale_for(frame, 0.055 * 0.82), budget, max_lines=2)

    assert lines and all(line.strip() for line in lines)
    assert " ".join(lines) == " ".join(message.split()), "wrapping dropped or reordered words"
    for line in lines:
        width, _ = cv2.getTextSize(line, ui.FONT, scale, max(1, int(round(scale * 1.2))))[0]
        assert width <= budget, f"line does not fit: {line!r}"


def test_a_long_alignment_message_is_rendered_not_dropped(settings: Settings) -> None:
    """The complement to the fitting test: the text reaches the canvas."""
    long_message = AlignmentError(
        total_rmse=44.0,
        message="The projection is skewed. Raise or lower the front of the projector.",
        severity="warning",
    )
    silent = AlignmentError(total_rmse=44.0, message="", severity="warning")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with_text = ui.draw_alignment_feedback(frame, long_message, settings)
    without = ui.draw_alignment_feedback(frame, silent, settings)

    assert not np.array_equal(with_text, without)


def test_the_grid_survives_corners_it_cannot_solve(settings: Settings) -> None:
    """A degenerate boundary must lose the grid, not end the wizard."""
    collapsed = TableBoundary(
        top_left=Vec2(100, 100),
        top_right=Vec2(100, 100),
        bottom_right=Vec2(100, 100),
        bottom_left=Vec2(100, 100),
        center=Vec2(100, 100),
        width_px=0.0,
        height_px=0.0,
        confidence=0.1,
    )
    frame = _felt_frame()
    out = ui.draw_alignment_grid(frame, collapsed, settings=settings)
    assert out is not frame
    assert np.array_equal(out, frame)


def test_corner_targets_are_distinguishable_by_shape_not_only_colour(
    settings: Settings,
) -> None:
    """Pending, active and recorded have to differ at a glance from a few feet.

    Asserted as "the three states put different amounts of ink on the frame",
    which is a proxy for different shapes and does not pin any of them.
    """
    base = _felt_frame()
    inked = []
    for kwargs in ({}, {"is_active": True}, {"is_recorded": True}):
        view = ui.draw_corner_target(base, Vec2(500, 400), "TOP LEFT", **kwargs)
        inked.append(int(np.count_nonzero(cv2.absdiff(view, base))))
    assert len(set(inked)) == 3, f"states are visually interchangeable: {inked}"


# ---------------------------------------------------------------------------
# The wizard end to end
# ---------------------------------------------------------------------------

#: Actions that walk a mock rig through all seven screens. ``reset`` on the
#: corner screen discards whatever the auto pass thought it saw -- with a mock
#: display nothing is projected, so any mark it finds is the mock camera's own
#: animation -- and the four ``record`` presses then pair each projected target
#: with its detected table corner.
FULL_RUN = [
    "next",
    "wait",
    "wait",
    "next",
    "next",
    "reset",
    "record",
    "record",
    "record",
    "record",
    "next",
    "next",
    "yes",
    "next",
]


@pytest.fixture()
def isolated_calibration(tmp_path, monkeypatch) -> object:
    """Redirect every calibration write into a temp directory."""
    import calibration_ui.report as report
    import projection.mapper as mapper

    monkeypatch.setattr(mapper, "CALIBRATION_FILE", tmp_path / "projector_calibration.json")
    monkeypatch.setattr(report, "CALIBRATION_DIR", tmp_path)
    return tmp_path


def _run(settings: Settings, script: list[str]) -> tuple[CalibrationApp, object]:
    console = Console(settings, script=list(script))
    app = CalibrationApp(settings, console=console)
    return app, app.run()


def test_a_full_run_produces_a_saved_calibration(
    settings: Settings, isolated_calibration
) -> None:
    app, result = _run(settings, FULL_RUN)

    assert result is not None
    assert result.is_calibrated
    assert result.homography is not None
    assert len(app.correspondences) == 4
    assert app.state.is_complete
    assert (isolated_calibration / "projector_calibration.json").is_file()


def test_a_full_run_writes_the_yaml_report(settings: Settings, isolated_calibration) -> None:
    """The spec asks for these three files by name."""
    _app, result = _run(settings, FULL_RUN)
    assert result is not None

    for name in (
        "camera_calibration.yaml",
        "projector_calibration.yaml",
        "calibration_timestamp.txt",
    ):
        assert (isolated_calibration / name).is_file(), name

    text = (isolated_calibration / "projector_calibration.yaml").read_text(encoding="utf-8")
    # The header exists so nobody edits the report expecting it to take effect.
    assert "does NOT load this file" in text


@pytest.mark.parametrize("stop_after", [0, 1, 4, 6, 10])
def test_cancelling_anywhere_saves_nothing(
    settings: Settings, isolated_calibration, stop_after: int
) -> None:
    """An abandoned wizard must leave the previous calibration in place."""
    script = FULL_RUN[:stop_after] + ["cancel", "cancel"]
    _app, result = _run(settings, script)

    assert result is None
    assert not (isolated_calibration / "projector_calibration.json").exists()
    assert not (isolated_calibration / "projector_calibration.yaml").exists()


def test_no_screen_ever_draws_on_the_camera_frame(
    settings: Settings, isolated_calibration, monkeypatch
) -> None:
    """The frame the wizard annotates is also the frame the detector reads.

    Every screen chains overlays with ``copy=False`` after the first call, which
    is what keeps the console responsive -- and which turns one misplaced
    ``copy=False`` on a chain's *first* call into a wizard banner drawn across
    the image detection is run on. This pins the invariant rather than the
    twenty-odd call sites that have to respect it.
    """
    from calibration_ui.calibration_app import CalibrationApp as App

    owned = _felt_frame()
    pristine = owned.copy()
    monkeypatch.setattr(App, "_capture", lambda self: owned)
    monkeypatch.setattr(App, "_capture_settled", lambda self: owned)

    _app, _result = _run(settings, FULL_RUN)

    assert np.array_equal(owned, pristine), "a screen drew on the camera frame"


def test_the_wizard_terminates_when_the_user_stops_answering(
    settings: Settings, isolated_calibration
) -> None:
    """Regression: the cancel confirmation used to be uncancellable.

    ``Escape`` dismissed the "are you sure" prompt instead of confirming it, so
    the only key meaning "get me out of here" could not, and an exhausted
    headless script span forever between the screen and its own confirmation.
    """
    _app, result = _run(settings, FULL_RUN[:6] + [])
    assert result is None


# ---------------------------------------------------------------------------
# Gates and geometry the wizard depends on
# ---------------------------------------------------------------------------


def test_the_final_score_is_the_worst_row_not_the_average() -> None:
    """One red makes the whole calibration POOR.

    Averaging lets a perfect corner fit hide a projection that misses the balls,
    and the corner fit is precisely the number that is near zero by
    construction.
    """
    assert _score_label([("a", "1", "info"), ("b", "2", "info")]) == "EXCELLENT"
    assert _score_label([("a", "1", "info"), ("b", "2", "warning")]) == "GOOD"
    assert _score_label([("a", "1", "info"), ("b", "2", "error")]) == "POOR"


def test_a_skewed_projection_cannot_be_saved_however_good_the_corner_fit(
    settings: Settings, isolated_calibration
) -> None:
    """Regression: the completion screen used to gate on corner RMSE alone.

    A four-point fit reports its own training error, which is zero whatever the
    projector is doing -- so a projection measured 37 degrees out of square and
    covering two thirds of the table scored EXCELLENT and offered Finish.
    """
    from calibration_ui.metrics import GridMetrics

    app = CalibrationApp(settings, console=Console(settings, script=[]))
    app.state.table_boundary = TableBoundary(
        Vec2(100, 100), Vec2(1800, 100), Vec2(1800, 900), Vec2(100, 900),
        Vec2(950, 500), 1700, 800, 0.95,
    )
    app.measurements.grid = GridMetrics(
        perpendicularity_deg=37.0, rotation_deg=13.3, coverage_x_pct=74.0, coverage_y_pct=68.0
    )

    rows = app._final_rows(app.assess_alignment(0.0))

    assert _score_label(rows) == "POOR"


def test_nudging_moves_the_projection_the_way_the_key_points(settings: Settings) -> None:
    app = CalibrationApp(settings, console=Console(settings, script=[]))
    point = Vec2(500.0, 400.0)

    app._nudge_projection("nudge_right")
    moved_right = app._apply_nudge(point)
    app._nudge_projection("nudge_down")
    moved_down = app._apply_nudge(point)

    assert moved_right.x > point.x
    assert moved_right.y == pytest.approx(point.y)
    assert moved_down.y > moved_right.y


def test_fine_tuning_keeps_the_keystone_correction(settings: Settings) -> None:
    """The nudge re-solves the homography instead of falling back to an affine.

    ``ProjectionMapper.nudge`` discards the homography by design, which on a
    keystoned rig throws away the correction the corner screen just earned. The
    wizard applies the nudge to the recorded correspondences and re-solves, so a
    user who taps an arrow key does not silently lose it.
    """
    from vision.calibration import compute_perspective_transform

    app = CalibrationApp(settings, console=Console(settings, script=[]))
    keystoned = TableBoundary(
        Vec2(200, 150), Vec2(1700, 90), Vec2(1780, 980), Vec2(140, 920),
        Vec2(955, 535), 1560, 830, 0.9,
    )
    app.state.table_boundary = keystoned
    app.camera_to_table, app.table_to_camera = compute_perspective_transform(keystoned, settings)
    for name, corner in zip(CORNER_NAMES, keystoned.corners(), strict=True):
        app._record_corner(name, corner)

    before = app._build_calibration()
    app._nudge_projection("nudge_left")
    app._nudge_projection("rotate_cw")
    after = app._build_calibration()

    assert before is not None and after is not None
    assert after.homography is not None, "nudging discarded the keystone correction"
    assert after.homography != before.homography
    # The affine fields stay as the human-legible summary of what was nudged.
    assert after.offset_x < 0.0
    assert after.rotation_deg > 0.0


def test_the_wizard_and_the_renderer_agree_on_where_targets_are_projected(
    settings: Settings,
) -> None:
    """A disagreement here is a calibration wrong by exactly the gap.

    Nothing on screen would show it: the mark is drawn at one pixel and recorded
    as another, and every later measurement is self-consistent about the wrong
    answer.
    """
    from projection.renderer import calibration_target_points

    app = CalibrationApp(settings, console=Console(settings, script=[]))
    assert [point for point, _label in app._target_points()] == [
        point for point, _label in calibration_target_points(settings)
    ]


def test_dragging_a_corner_past_another_keeps_the_order_clockwise() -> None:
    """Out-of-order corners solve to a mirrored homography with no clue why."""
    original = TableBoundary(
        Vec2(200, 200), Vec2(1700, 200), Vec2(1700, 900), Vec2(200, 900),
        Vec2(950, 550), 1500, 700, 0.9,
    )
    # Swap the two top corners, as a user dragging top-left past top-right does.
    scrambled = [Vec2(1700, 200), Vec2(200, 200), Vec2(1700, 900), Vec2(200, 900)]

    rebuilt = _rebuild_boundary(original, scrambled)

    assert rebuilt.top_left.x < rebuilt.top_right.x
    assert rebuilt.top_left.y < rebuilt.bottom_left.y
    assert rebuilt.bottom_right.x > rebuilt.bottom_left.x
