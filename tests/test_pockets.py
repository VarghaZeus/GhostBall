"""Tests for the Session 2 retrofit: pocket-based, colour-independent detection.

The claims this module makes are unusually checkable, so these tests check them
rather than checking that the code runs:

* **Any cloth colour.** The same table is rendered in five colours and detected
  in all of them, with the felt detector shown failing on the same frames. That
  contrast is the whole justification for the module existing.
* **Any table size.** Sizes from 6 ft to 10 ft, measured to within a stated
  tolerance rather than assumed from config.
* **Any camera height.** The one property the brief's fixed-reference formula
  cannot deliver, and the reason the scale comes from a ball. Asserted as: the
  table's pixel width may vary by more than 2x while the measurement does not
  move.
* **Nothing downstream breaks.** The boundary is the same type with the same
  fields, and the felt path still works where it always did.

Accuracy targets here are deliberately looser than the numbers actually
measured, because these are thresholds for "still working", not a record of the
current implementation's luck. The measured figures live in the README.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.config import BALL_DIAMETER_IN, Settings, nearest_standard_length_ft
from app.models import BallColor, PocketId, TableBoundary, Vec2
from tests.synthetic import FELT_COLOURS, BallSpec, render_table
from vision.calibration import detect_table_boundaries, detect_table_boundaries_by_felt
from vision.pockets import (
    HoughParams,
    PocketBlob,
    adaptive_cloth_mask,
    adopt_measured_table_size,
    classify_pockets_by_geometry,
    detect_pockets_loose,
    detect_pockets_refined,
    detect_table_boundaries_dynamic,
    dynamic_detect_table_size,
    get_dynamic_hough_params,
    resolve_scale,
)

#: Corner accuracy the whole system is specified against. The felt detector
#: measured 11.3 px worst case across its scenarios; this path should not be
#: meaningfully worse.
MAX_CORNER_ERROR_PX = 20.0

#: Table-size accuracy from a ball-derived scale. The dominant error is that a
#: ball's antialiased edge reads as part of the ball, making it about half a
#: pixel fat all round -- roughly 2% on a 23 px radius, and more as the ball gets
#: smaller in frame. 8% leaves room for a high camera without being so loose
#: that a real regression passes.
MAX_SIZE_ERROR_FRAC = 0.08


def _balls_for(settings: Settings) -> list[BallSpec]:
    """Five balls spread over the cloth, positioned relative to the table size."""
    layout = [
        (0.25, 0.50, BallColor.WHITE),
        (0.65, 0.50, BallColor.YELLOW),
        (0.74, 0.32, BallColor.BLUE),
        (0.79, 0.68, BallColor.RED),
        (0.40, 0.26, BallColor.PURPLE),
    ]
    return [
        BallSpec(u * settings.table.length_in, v * settings.table.width_in, colour)
        for u, v, colour in layout
    ]


def scene(
    length_ft: float = 7.0,
    felt: str = "green",
    margin: float = 0.10,
    *,
    balls: bool = True,
    **kwargs: object,
):
    """A rendered table of a given size and colour, plus its ground truth."""
    settings = Settings()
    settings.camera.use_mock = True
    settings.projector.use_mock = True
    settings.table.length_in = length_ft * 12.0
    settings.table.width_in = length_ft * 6.0
    frame, truth = render_table(
        settings,
        _balls_for(settings) if balls else [],
        margin=margin,
        felt_hsv=FELT_COLOURS[felt],
        **kwargs,  # type: ignore[arg-type]
    )
    return settings, frame, truth


def _corner_error(boundary: TableBoundary, truth) -> float:
    return max(
        boundary.corners()[i].distance_to(truth.corners[i]) for i in range(4)
    )


# ---------------------------------------------------------------------------
# Cloth colour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("felt", sorted(FELT_COLOURS))
def test_a_table_is_found_on_any_cloth_colour(felt: str) -> None:
    """The headline claim. A pocket is a hole whatever the cloth is dyed."""
    settings, frame, truth = scene(felt=felt, noise_sigma=2.0)

    boundary = detect_table_boundaries_dynamic(frame, settings)

    assert boundary is not None, f"no table found on {felt} cloth"
    assert boundary.detection_method == "pockets"
    assert _corner_error(boundary, truth) < MAX_CORNER_ERROR_PX


@pytest.mark.parametrize("felt", ["red", "blue", "burgundy"])
def test_the_felt_detector_fails_on_the_frames_the_pocket_one_handles(felt: str) -> None:
    """The justification for the retrofit, pinned so it cannot quietly go stale.

    If this ever starts passing, the felt thresholds have been widened to cover
    everything -- at which point they are also covering the green carpet and the
    plant behind the table, and the acceptance checks are doing all the work.
    """
    settings, frame, _truth = scene(felt=felt)
    assert detect_table_boundaries_by_felt(frame, settings) is None
    assert detect_table_boundaries_dynamic(frame, settings) is not None


def test_balls_are_still_detected_on_non_green_cloth() -> None:
    """Finding the table on red cloth is worthless if no ball is found on it.

    Ball detection works by inverting the felt mask, so a table the felt
    thresholds do not match reads as one enormous non-felt blob and yields zero
    balls -- a failure that looks like everything working until nothing is ever
    detected. The adaptive cloth fallback exists for this.
    """
    import time

    from vision.calibration import compute_perspective_transform
    from vision.detection import extract_game_state

    settings, frame, truth = scene(felt="red", noise_sigma=2.0)
    boundary = detect_table_boundaries_dynamic(frame, settings)
    assert boundary is not None
    camera_to_table, _ = compute_perspective_transform(boundary, settings)

    state = extract_game_state(frame, 0, time.perf_counter(), boundary, camera_to_table, settings)

    # Not all five: a red ball on red cloth is genuinely hard, and pretending
    # otherwise would make this test a description of one lucky render.
    assert len(state.balls) >= 3
    assert state.cue_ball is not None
    for ball in state.balls:
        assert ball.table_pos is not None
        nearest = min(ball.table_pos.distance_to(p) for p in truth.ball_positions_in)
        assert nearest < 1.0, "detected a ball nowhere near a real one"


# ---------------------------------------------------------------------------
# Table size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length_ft", [6.0, 7.0, 7.5, 8.0, 9.0, 10.0])
def test_any_table_size_is_measured_not_assumed(length_ft: float) -> None:
    """No hardcoded sizes: the same code measures a 6 ft and a 10 ft table."""
    settings, frame, truth = scene(length_ft=length_ft, noise_sigma=2.0)

    boundary = detect_table_boundaries_dynamic(frame, settings)

    assert boundary is not None
    assert boundary.is_measured
    assert boundary.scale_source == "ball"
    error = abs(boundary.length_ft - length_ft) / length_ft
    assert error < MAX_SIZE_ERROR_FRAC, f"measured {boundary.length_ft:.2f} ft, expected {length_ft}"

    # Both axes are measured separately, so check they agree with each other --
    # against the aspect the harness actually rendered, not against the 2:1 a
    # real table has. ``render_table`` fits the table to the frame on each axis
    # independently, so it draws a 16:9 table; asserting 2:1 here would be
    # testing the renderer's framing rather than the measurement.
    rendered_aspect = truth.corners[0].distance_to(truth.corners[1]) / truth.corners[0].distance_to(
        truth.corners[3]
    )
    assert boundary.length_ft / boundary.width_ft == pytest.approx(rendered_aspect, rel=0.06)


def test_a_non_standard_size_is_reported_as_itself() -> None:
    """An 8.2 ft table must not be rounded to 8 ft just because 8 has a name."""
    settings, frame, _truth = scene(length_ft=8.2, felt="blue", noise_sigma=2.0)

    boundary = detect_table_boundaries_dynamic(frame, settings)

    assert boundary is not None
    assert abs(boundary.length_ft - 8.2) / 8.2 < MAX_SIZE_ERROR_FRAC
    # Snapping is off by default precisely so this stays true; 8.2 is within 3%
    # of 8.0 and would be swallowed by it.
    assert nearest_standard_length_ft(8.2) == 8.0


def test_snapping_is_opt_in_and_lands_on_the_standard_size() -> None:
    settings, frame, _truth = scene(length_ft=7.5, noise_sigma=2.0)
    settings.vision.snap_to_standard_size = True

    boundary = detect_table_boundaries_dynamic(frame, settings)

    assert boundary is not None
    assert boundary.length_ft == pytest.approx(7.5, abs=0.01)
    assert "snap" in boundary.scale_source


# ---------------------------------------------------------------------------
# Camera height: the property the brief's formula cannot deliver
# ---------------------------------------------------------------------------


def test_the_measurement_survives_moving_the_camera() -> None:
    """The whole reason scale comes from a ball rather than a pixel reference.

    ``scale_factor = measured_px / 2000`` cannot work: a camera sees ``f*L/h``
    pixels, so doubling the table and doubling the height give an identical
    image and one number cannot be split into two unknowns. Measured against
    this harness, that formula reported 6.45, 5.11 and 2.96 ft for one unchanged
    6.33 ft table at three heights.

    A ball is a second known length in the same image, so the height cancels.
    """
    measurements: list[float] = []
    pixel_widths: list[float] = []
    for margin in (0.02, 0.10, 0.18, 0.28):
        settings, frame, _truth = scene(length_ft=7.5, margin=margin, noise_sigma=2.0)
        boundary = detect_table_boundaries_dynamic(frame, settings)
        assert boundary is not None, f"no table at margin {margin}"
        measurements.append(boundary.length_ft)
        pixel_widths.append(boundary.width_px)

    # The camera really did move: the table more than doubles in pixel width.
    assert max(pixel_widths) / min(pixel_widths) > 2.0
    # The table did not.
    assert max(measurements) - min(measurements) < 0.6
    for measured in measurements:
        assert abs(measured - 7.5) / 7.5 < MAX_SIZE_ERROR_FRAC


def test_the_fixed_reference_scale_ignores_the_image_entirely() -> None:
    """Pinning what ``scale_source: reference`` actually is, so nobody trusts it.

    It yields the same px/inch for every frame ever captured, which is exactly
    the assumption it encodes -- that the camera never moved from the height the
    reference was taken at. Kept because it is the only option on bare cloth,
    and warned about in the logs every time it is used.
    """
    settings = Settings()
    settings.vision.scale_source = "reference"
    quad = np.array([[0, 0], [1000, 0], [1000, 500], [0, 500]], dtype=np.float64)

    small, source_a = resolve_scale(np.zeros((10, 10, 3), np.uint8), quad, 1000.0, settings)
    large, source_b = resolve_scale(np.zeros((10, 10, 3), np.uint8), quad, 4000.0, settings)

    assert source_a == source_b == "reference"
    assert small == pytest.approx(large), "the reference scale should not depend on the image"
    expected = settings.vision.reference_table_width_px / (
        settings.vision.reference_table_length_ft * 12.0
    )
    assert small == pytest.approx(expected)


def test_the_configured_scale_reproduces_the_configured_table() -> None:
    """``config`` is circular by construction, and that is fine -- it is a fallback."""
    settings, frame, _truth = scene(length_ft=7.0, balls=False)
    settings.vision.scale_source = "config"

    boundary = detect_table_boundaries_dynamic(frame, settings)

    assert boundary is not None
    assert boundary.scale_source == "config"
    assert boundary.length_ft == pytest.approx(7.0, rel=0.01)


def test_bare_cloth_falls_back_rather_than_failing() -> None:
    """No balls means no ball-derived scale, but the table is still found."""
    settings, frame, truth = scene(balls=False)

    boundary = detect_table_boundaries_dynamic(frame, settings)

    assert boundary is not None
    assert boundary.scale_source == "config"
    assert _corner_error(boundary, truth) < MAX_CORNER_ERROR_PX


# ---------------------------------------------------------------------------
# Geometry robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rotation_deg", "perspective"),
    [(0.0, 0.0), (12.0, 0.0), (20.0, 0.0), (0.0, 0.12), (15.0, 0.10)],
)
def test_a_camera_that_is_not_square_to_the_table_still_works(
    rotation_deg: float, perspective: float
) -> None:
    """A camera bolted to a ceiling joist is never square to the table."""
    settings, frame, truth = scene(
        margin=0.20, rotation_deg=rotation_deg, perspective=perspective, noise_sigma=2.0
    )

    boundary = detect_table_boundaries_dynamic(frame, settings)

    assert boundary is not None
    assert _corner_error(boundary, truth) < MAX_CORNER_ERROR_PX


def test_uneven_lighting_does_not_hide_the_corner_pockets() -> None:
    """Regression: the contrast test used to be absolute, so vignetting broke it.

    Room light falls off toward the frame corners, which is where the corner
    pockets are. At 30% falloff their contrast measured 17 grey levels against
    an 18-level threshold and four of six pockets were rejected on a frame where
    every one was plainly visible. The test is now a ratio.
    """
    settings, frame, truth = scene(margin=0.14, vignette=0.35, noise_sigma=3.0)

    pockets = detect_pockets_loose(frame, settings)

    assert len(pockets) >= 5, f"vignetting cost us pockets: found {len(pockets)}"
    boundary = detect_table_boundaries_dynamic(frame, settings)
    assert boundary is not None
    assert _corner_error(boundary, truth) < MAX_CORNER_ERROR_PX


def test_the_projected_overlay_does_not_break_detection() -> None:
    """This system paints light on the thing it is looking at."""
    settings, frame, truth = scene(overlay_streak=True, noise_sigma=2.0)

    boundary = detect_table_boundaries_dynamic(frame, settings)

    assert boundary is not None
    assert _corner_error(boundary, truth) < MAX_CORNER_ERROR_PX


# ---------------------------------------------------------------------------
# The pipeline's parts
# ---------------------------------------------------------------------------


def test_the_loose_pass_finds_all_six_pockets() -> None:
    settings, frame, truth = scene(noise_sigma=2.0)

    pockets = detect_pockets_loose(frame, settings)

    assert len(pockets) == 6
    # Every corner should have a candidate almost exactly on it: the corner
    # pocket centre *is* the table corner, by the layout both this module and
    # vision.detection share.
    for corner in truth.corners:
        nearest = min(pockets, key=lambda p: p.center.distance_to(corner))
        assert nearest.center.distance_to(corner) < MAX_CORNER_ERROR_PX


def test_hough_parameters_scale_with_the_measured_table() -> None:
    """No hardcoding: a bigger table in frame gives proportionally bigger bounds."""
    small_settings, small_frame, _ = scene(length_ft=7.0, margin=0.26)
    big_settings, big_frame, _ = scene(length_ft=7.0, margin=0.02)

    small = dynamic_detect_table_size(
        small_frame, detect_pockets_loose(small_frame, small_settings), small_settings
    )
    big = dynamic_detect_table_size(
        big_frame, detect_pockets_loose(big_frame, big_settings), big_settings
    )
    assert small is not None and big is not None

    small_params = get_dynamic_hough_params(small, small_settings)
    big_params = get_dynamic_hough_params(big, big_settings)

    assert big_params.min_radius > small_params.min_radius * 1.5
    assert big_params.min_dist > small_params.min_dist * 1.5
    # The brief's three ratios, which are the point of the function.
    assert small_params.min_dist == pytest.approx(small.pocket_spacing_px * 0.95)
    assert small_params.min_radius == pytest.approx(small.pocket_radius_px * 0.8)
    assert small_params.max_radius == pytest.approx(small.pocket_radius_px * 1.3)


def test_the_refined_pass_rejects_what_the_loose_one_let_through() -> None:
    """The second pass is the point of measuring first, so it should be pickier."""
    settings, frame, _truth = scene(noise_sigma=3.0)
    loose = detect_pockets_loose(frame, settings)
    measurement = dynamic_detect_table_size(frame, loose, settings)
    assert measurement is not None

    params = get_dynamic_hough_params(measurement, settings)
    refined = detect_pockets_refined(frame, params, settings)

    assert len(refined) == 6
    for blob in refined:
        assert params.min_radius <= blob.radius_px <= params.max_radius


def test_pockets_are_classified_into_corners_and_sides() -> None:
    settings, frame, truth = scene(noise_sigma=2.0)
    pockets = detect_pockets_loose(frame, settings)

    corners, sides = classify_pockets_by_geometry(pockets)

    assert len(corners) == 4
    assert len(sides) == 2
    for corner in corners:
        nearest = min(corner.center.distance_to(c) for c in truth.corners)
        assert nearest < MAX_CORNER_ERROR_PX
    # Side pockets sit at the midpoints of the long rails, so they are much
    # closer to the table centre than any corner is.
    center = Vec2(
        sum(c.x for c in truth.corners) / 4.0, sum(c.y for c in truth.corners) / 4.0
    )
    assert max(s.center.distance_to(center) for s in sides) < min(
        c.center.distance_to(center) for c in corners
    )


def test_marks_that_are_not_a_table_are_rejected() -> None:
    """Four dark blobs in a line are not a pool table."""
    settings = Settings()
    blobs = [
        PocketBlob(center=Vec2(100 + 200 * i, 300.0), radius_px=20.0, circularity=0.9, contrast=0.6)
        for i in range(5)
    ]
    assert dynamic_detect_table_size(np.zeros((720, 1280, 3), np.uint8), blobs, settings) is None


def test_too_few_candidates_is_no_table_rather_than_a_guess() -> None:
    settings = Settings()
    blobs = [
        PocketBlob(center=Vec2(100.0, 100.0), radius_px=20.0, circularity=0.9, contrast=0.6),
        PocketBlob(center=Vec2(900.0, 110.0), radius_px=20.0, circularity=0.9, contrast=0.6),
    ]
    assert dynamic_detect_table_size(np.zeros((720, 1280, 3), np.uint8), blobs, settings) is None


def test_an_empty_frame_is_handled() -> None:
    settings = Settings()
    assert detect_pockets_loose(np.zeros((0, 0, 3), np.uint8), settings) == []
    assert detect_table_boundaries_dynamic(np.zeros((0, 0, 3), np.uint8), settings) is None


def test_the_measurement_dict_uses_the_long_axis_as_length() -> None:
    """The brief inverts the axes; a 7 ft table is 7 ft *long*, not 7 ft wide.

    Following it literally would report a 7 ft table as 7 ft wide and 14 ft
    long, and every table coordinate downstream would be transposed.
    """
    settings, frame, _truth = scene(length_ft=7.0, noise_sigma=2.0)
    pockets = detect_pockets_loose(frame, settings)
    measurement = dynamic_detect_table_size(frame, pockets, settings)
    assert measurement is not None

    data = measurement.as_dict()

    assert data["table_length_ft"] > data["table_width_ft"]
    assert data["table_length_px"] > data["table_width_px"]
    assert set(data) >= {
        "table_length_ft",
        "table_width_ft",
        "table_length_px",
        "table_width_px",
        "scale_factor",
        "pocket_spacing_px",
        "pocket_radius_px",
        "pixels_per_ft",
        "scale_source",
    }


# ---------------------------------------------------------------------------
# Adaptive cloth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("felt", ["green", "red", "blue", "burgundy"])
def test_the_adaptive_cloth_mask_covers_the_cloth_whatever_its_colour(felt: str) -> None:
    settings, frame, truth = scene(felt=felt)
    quad = np.array([[c.x, c.y] for c in truth.corners], dtype=np.float64)

    mask = adaptive_cloth_mask(frame, quad, settings)

    assert mask is not None
    from vision.pockets import _quad_mask

    interior = _quad_mask(frame.shape[:2], quad, inset=40.0)
    covered = np.count_nonzero(mask[interior > 0]) / np.count_nonzero(interior)
    assert covered > 0.75, f"only {covered:.0%} of {felt} cloth matched"
    # And it must not claim the surround, or every ball search would be empty.
    outside = np.count_nonzero(mask[interior == 0]) / np.count_nonzero(interior == 0)
    assert outside < 0.5


# ---------------------------------------------------------------------------
# Integration with the rest of the system
# ---------------------------------------------------------------------------


def test_the_dispatcher_honours_the_configured_method() -> None:
    settings, frame, _truth = scene(felt="red")

    settings.vision.table_detection_method = "felt"
    assert detect_table_boundaries(frame, settings) is None

    settings.vision.table_detection_method = "pockets"
    boundary = detect_table_boundaries(frame, settings)
    assert boundary is not None and boundary.detection_method == "pockets"


def test_auto_falls_back_to_felt_when_the_pockets_cannot_be_seen() -> None:
    """A tight crop or recessed returns can hide the pockets; the cloth remains."""
    settings, frame, _truth = scene(draw_pockets=False)
    assert settings.vision.table_detection_method == "auto"

    boundary = detect_table_boundaries(frame, settings)

    assert boundary is not None
    assert boundary.detection_method == "felt"
    assert not boundary.is_measured


def test_a_felt_boundary_reports_no_measurement_rather_than_zero() -> None:
    """Downstream must be able to tell "not measured" from "measured as zero"."""
    settings, frame, _truth = scene()

    boundary = detect_table_boundaries_by_felt(frame, settings)

    assert boundary is not None
    assert boundary.length_ft is None
    assert boundary.pixels_per_ft is None
    assert boundary.scale_source == ""
    assert not boundary.is_measured


def _measured(length_ft: float, source: str = "ball") -> TableBoundary:
    """A boundary carrying a measurement, for the adoption tests."""
    return TableBoundary(
        Vec2(0, 0), Vec2(100, 0), Vec2(100, 50), Vec2(0, 50), Vec2(50, 25),
        100.0, 50.0, 0.9,
        length_ft=length_ft, width_ft=length_ft / 2.0, pixels_per_ft=100.0 / length_ft,
        scale_source=source, detection_method="pockets",
    )


def test_adoption_is_off_by_default() -> None:
    """The measurement is not accurate enough to be trusted with this decision.

    A ball-derived size is +/-2% at best and +/-8% when the ball images small,
    while adjacent standard tables are 7% apart -- so it cannot reliably tell a
    7 ft table from a 7.5 ft one. Guessing wrong silently rescales every physics
    prediction, which is a worse outcome than leaving the user's config alone.
    """
    settings = Settings()
    assert settings.vision.adopt_measured_table_size is False
    assert adopt_measured_table_size(settings, _measured(9.0)) is False
    assert settings.table.length_in == pytest.approx(76.0)


def test_a_disagreement_is_always_reported_even_when_not_adopted() -> None:
    """Silence would leave the user with no way to discover a wrong preset."""
    from vision.pockets import report_size_disagreement

    settings = Settings()
    assert report_size_disagreement(settings, _measured(9.0)) is True
    assert settings.table.length_in == pytest.approx(76.0), "reporting must not mutate"
    # Within the noise band, there is nothing worth telling anybody.
    assert report_size_disagreement(settings, _measured(6.4)) is False


def test_adoption_replaces_the_configured_size_when_asked_to() -> None:
    """What makes "physics uses real-world dimensions" true rather than a slogan."""
    settings = Settings()
    settings.vision.adopt_measured_table_size = True
    settings.table.length_in = 76.0  # says 7 ft

    assert adopt_measured_table_size(settings, _measured(9.0)) is True
    assert settings.table.length_in == pytest.approx(108.0)
    assert settings.table.width_in == pytest.approx(54.0)


def test_adoption_ignores_a_scale_it_cannot_trust() -> None:
    """A config-derived size was computed *from* the config; adopting it is a loop.

    A reference-derived one is only as good as the camera height, which is the
    thing nobody can check. Neither is worth resizing the table for.
    """
    for source in ("config", "reference", ""):
        settings = Settings()
        settings.vision.adopt_measured_table_size = True
        assert adopt_measured_table_size(settings, _measured(9.0, source)) is False
        assert settings.table.length_in == pytest.approx(76.0)


def test_adoption_leaves_an_agreeing_measurement_alone() -> None:
    """No churn when the config was right, which is the common case."""
    settings = Settings()
    settings.vision.adopt_measured_table_size = True
    assert adopt_measured_table_size(settings, _measured(6.35)) is False
    assert settings.table.length_in == pytest.approx(76.0)


def test_settings_instances_do_not_share_one_table() -> None:
    """Regression: every Settings() aliased the same TableSize object.

    ``default_factory`` returned the shared ``TABLE_PRESETS`` entry rather than a
    copy, so adoption -- the first code to write to ``settings.table`` -- resized
    the preset for the whole process, and the next ``Settings()`` came back with
    somebody else's table.
    """
    from app.config import TABLE_PRESETS

    first, second = Settings(), Settings()
    first.table.length_in = 999.0
    assert second.table.length_in == pytest.approx(76.0)
    assert TABLE_PRESETS["7ft"].length_in == pytest.approx(76.0)


def test_a_measured_boundary_still_solves_the_usual_homography() -> None:
    """Sessions 3-7 consume a TableBoundary and must not notice the change."""
    from vision.calibration import camera_to_table_coords, compute_perspective_transform

    settings, frame, truth = scene(noise_sigma=2.0)
    boundary = detect_table_boundaries_dynamic(frame, settings)
    assert boundary is not None

    camera_to_table, _table_to_camera = compute_perspective_transform(boundary, settings)
    origin = camera_to_table_coords(boundary.top_left, camera_to_table)
    far = camera_to_table_coords(boundary.bottom_right, camera_to_table)

    assert origin.x == pytest.approx(0.0, abs=0.5)
    assert origin.y == pytest.approx(0.0, abs=0.5)
    assert far.x == pytest.approx(settings.table.length_in, abs=0.5)
    assert far.y == pytest.approx(settings.table.width_in, abs=0.5)


def test_ball_diameter_is_the_assumption_the_scale_rests_on() -> None:
    """Spelled out, because a mini table with small balls measures large.

    The code cannot distinguish a 2 in ball on a small table from a 2.25 in ball
    on a larger one -- that is the same ambiguity as camera height, one step
    down. The documented answer is to set ``scale_source: config``.
    """
    settings, frame, _truth = scene(length_ft=7.0, noise_sigma=2.0)
    boundary = detect_table_boundaries_dynamic(frame, settings)
    assert boundary is not None and boundary.pixels_per_ft is not None

    implied_ball_px = boundary.pixels_per_ft / 12.0 * BALL_DIAMETER_IN
    assert implied_ball_px > 1.0
    assert math.isclose(
        boundary.length_ft * 12.0,
        boundary.width_px / (boundary.pixels_per_ft / 12.0),
        rel_tol=1e-6,
    )


def test_hough_params_expose_a_complete_opencv_parameter_set() -> None:
    """A caller who does want cv2.HoughCircles should not have to re-derive these."""
    params = HoughParams(min_dist=40.0, min_radius=10.0, max_radius=20.0)
    as_dict = params.as_dict()
    assert set(as_dict) == {"minDist", "minRadius", "maxRadius", "dp", "param1", "param2"}
    assert as_dict["minDist"] == 40.0


def test_pocket_ids_are_assigned_to_the_right_corners() -> None:
    settings, frame, truth = scene(noise_sigma=2.0)
    pockets = detect_pockets_loose(frame, settings)
    measurement = dynamic_detect_table_size(frame, pockets, settings)
    assert measurement is not None

    assert set(measurement.pockets) == {
        PocketId.TOP_LEFT,
        PocketId.TOP_MIDDLE,
        PocketId.TOP_RIGHT,
        PocketId.BOTTOM_RIGHT,
        PocketId.BOTTOM_MIDDLE,
        PocketId.BOTTOM_LEFT,
    }
    top_left = measurement.pockets[PocketId.TOP_LEFT].center
    assert top_left.distance_to(truth.corners[0]) < MAX_CORNER_ERROR_PX
