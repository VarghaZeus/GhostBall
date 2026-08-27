"""Digital crop: geometry, the pocket guard, and the pixel-scale consequences.

Three things are worth testing here and they fail in different ways.

**The coordinate transform.** A crop introduces a second coordinate system, and
the conversion between them is a translation by the crop origin -- trivial, and
trivially skippable, because with no crop the origin is ``(0, 0)`` and omitting
the conversion is indistinguishable from applying it. Every default is uncropped,
so a missing conversion passes any test that does not deliberately crop first.

**The pocket guard.** A crop that loses a pocket does not degrade detection, it
removes the reference detection is built on. It has to be refused.

**The scale consequences.** Cropping changes what "a fraction of frame width"
means while leaving absolute pixel counts alone. Which quantities move and which
do not is the substance of the design, so it is pinned here rather than left as a
claim in a docstring.
"""

from __future__ import annotations

import json

import pytest

from app.models import Pocket, PocketId, TableBoundary, Vec2
from vision.crop import (
    MIN_CROP_PX,
    CropRect,
    fit_to_table,
    frame_to_sensor,
    pockets_outside,
    sensor_to_frame,
)

SENSOR = (2304, 1296)


def pocket(name: PocketId, x: float, y: float, radius: float = 30.0) -> Pocket:
    return Pocket(id=name, center_px=Vec2(x, y), radius_px=radius)


def boundary(x0=400.0, y0=300.0, x1=1900.0, y1=1000.0) -> TableBoundary:
    return TableBoundary(
        top_left=Vec2(x0, y0),
        top_right=Vec2(x1, y0),
        bottom_right=Vec2(x1, y1),
        bottom_left=Vec2(x0, y1),
        center=Vec2((x0 + x1) / 2, (y0 + y1) / 2),
        width_px=x1 - x0,
        height_px=y1 - y0,
        confidence=0.95,
    )


def six_pockets(x0=400.0, y0=300.0, x1=1900.0, y1=1000.0) -> list[Pocket]:
    mid = (x0 + x1) / 2
    return [
        pocket(PocketId.TOP_LEFT, x0, y0),
        pocket(PocketId.TOP_MIDDLE, mid, y0),
        pocket(PocketId.TOP_RIGHT, x1, y0),
        pocket(PocketId.BOTTOM_LEFT, x0, y1),
        pocket(PocketId.BOTTOM_MIDDLE, mid, y1),
        pocket(PocketId.BOTTOM_RIGHT, x1, y1),
    ]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestCropRect:
    def test_clamping_shrinks_before_it_moves(self) -> None:
        """A rectangle bigger than the frame becomes a valid one at the edge,
        not a valid size at an impossible origin."""
        rect = CropRect(x=100, y=100, width=9999, height=9999).clamped(*SENSOR)
        assert (rect.x, rect.y) == (0, 0)
        assert (rect.width, rect.height) == SENSOR

    def test_clamping_never_produces_an_empty_slice(self) -> None:
        """An empty crop would surface a long way from here, as "the camera
        stopped producing frames"."""
        rect = CropRect(x=5000, y=5000, width=0, height=0).clamped(*SENSOR)
        assert rect.width >= 1 and rect.height >= 1
        assert rect.x1 <= SENSOR[0] and rect.y1 <= SENSOR[1]

    def test_a_rectangle_hanging_off_the_edge_is_pulled_inside(self) -> None:
        rect = CropRect(x=2200, y=1200, width=400, height=300).clamped(*SENSOR)
        assert rect.x1 == SENSOR[0] and rect.y1 == SENSOR[1]
        assert (rect.width, rect.height) == (400, 300), "the size should survive"

    def test_zoom_keeps_the_centre_still(self) -> None:
        """Zooming about the origin would walk the table towards a corner and
        need a pan after every step to undo it."""
        rect = CropRect(x=600, y=300, width=1000, height=600)
        before = (rect.x + rect.width / 2, rect.y + rect.height / 2)
        zoomed = rect.zoomed(0.8, *SENSOR)
        after = (zoomed.x + zoomed.width / 2, zoomed.y + zoomed.height / 2)
        assert abs(before[0] - after[0]) <= 1
        assert abs(before[1] - after[1]) <= 1
        assert zoomed.width < rect.width

    def test_zoom_stops_at_a_floor(self) -> None:
        """Otherwise you can zoom into a corner and lose the very preview you
        are aiming with."""
        rect = CropRect.full(*SENSOR)
        for _ in range(60):
            rect = rect.zoomed(0.5, *SENSOR)
        assert rect.width >= MIN_CROP_PX and rect.height >= MIN_CROP_PX

    def test_zooming_out_from_full_frame_stays_full_frame(self) -> None:
        rect = CropRect.full(*SENSOR).zoomed(2.0, *SENSOR)
        assert rect.is_full(*SENSOR)

    def test_pan_preserves_the_size(self) -> None:
        rect = CropRect(x=600, y=300, width=1000, height=600)
        panned = rect.panned(150, -80, *SENSOR)
        assert (panned.width, panned.height) == (rect.width, rect.height)
        assert (panned.x, panned.y) == (750, 220)

    def test_pan_into_the_edge_stops_rather_than_shrinking(self) -> None:
        rect = CropRect(x=0, y=0, width=1000, height=600)
        panned = rect.panned(-500, -500, *SENSOR)
        assert (panned.x, panned.y) == (0, 0)
        assert (panned.width, panned.height) == (1000, 600)

    def test_is_full_is_exact(self) -> None:
        assert CropRect.full(*SENSOR).is_full(*SENSOR)
        assert not CropRect(x=1, y=0, width=SENSOR[0] - 1, height=SENSOR[1]).is_full(*SENSOR)


class TestCoordinateSpaces:
    """The translation that is invisible in the default configuration."""

    CROP = CropRect(x=300, y=200, width=1400, height=800)

    def test_frame_to_sensor_adds_the_origin(self) -> None:
        assert frame_to_sensor(10.0, 20.0, self.CROP) == (310.0, 220.0)

    def test_the_two_directions_are_inverses(self) -> None:
        for point in ((0.0, 0.0), (10.5, 20.25), (1399.0, 799.0)):
            there = frame_to_sensor(*point, self.CROP)
            assert sensor_to_frame(*there, self.CROP) == point

    def test_an_uncropped_frame_is_the_identity(self) -> None:
        """Which is exactly why a missing conversion goes unnoticed."""
        full = CropRect.full(*SENSOR)
        assert frame_to_sensor(42.0, 99.0, full) == (42.0, 99.0)


# ---------------------------------------------------------------------------
# The pocket constraint
# ---------------------------------------------------------------------------


class TestPocketGuard:
    def test_a_crop_containing_every_pocket_loses_nothing(self) -> None:
        full = CropRect.full(*SENSOR)
        assert pockets_outside(full, six_pockets(), full) == []

    def test_a_crop_that_cuts_a_pocket_names_it(self) -> None:
        """Only the pockets actually lost, so the refusal tells you which way to
        pan. Left edge at 800 loses the left pair; the top and bottom edges are
        deliberately clear so the other four survive."""
        full = CropRect.full(*SENSOR)
        proposed = CropRect(x=800, y=250, width=1200, height=850)
        lost = pockets_outside(proposed, six_pockets(), full)
        assert set(lost) == {"top_left", "bottom_left"}, lost

    def test_the_whole_pocket_must_survive_not_just_its_centre(self) -> None:
        """A crop edge through the middle of a pocket leaves a half-disc, which
        is not what the blob detector is looking for."""
        full = CropRect.full(*SENSOR)
        # Left edge exactly on the pocket centre: the centre is inside, the
        # pocket is not.
        proposed = CropRect(x=400, y=200, width=1600, height=900)
        assert "top_left" in pockets_outside(proposed, six_pockets(), full)

    def test_pockets_are_compared_in_the_right_space(self) -> None:
        """The regression this suite exists for. Pockets are frame-space, the
        proposed crop is sensor-space, and comparing them directly happens to
        work whenever the current crop starts at the origin.

        Here the current crop is offset, so a naive comparison gets it wrong: the
        pockets sit at sensor 700..2200, and a proposed crop of 650..2250 keeps
        all of them. Skip the conversion and they look like they are at 400..1900
        against the same proposal, which also passes -- so the assertion below is
        the *inverse* case, where the naive answer differs.
        """
        current = CropRect(x=300, y=200, width=1800, height=1000)
        pockets = six_pockets(x0=100.0, y0=100.0, x1=1500.0, y1=800.0)
        # In sensor space these span 400..1800 x 300..1000.
        proposed = CropRect(x=350, y=250, width=1500, height=800)
        assert pockets_outside(proposed, pockets, current) == []

        # Naive (no conversion) would have compared 100..1500 against x>=350 and
        # called top_left and bottom_left lost. Prove the conversion is what
        # saves it by shifting the proposal to where the naive answer is "safe"
        # and the true answer is not.
        naive_safe = CropRect(x=50, y=50, width=1500, height=800)
        assert pockets_outside(naive_safe, pockets, current) != []

    def test_no_detected_pockets_means_nothing_is_known_to_be_lost(self) -> None:
        """Not the same claim as "this crop is safe" -- the caller reports the
        detected count so the difference is visible."""
        full = CropRect.full(*SENSOR)
        assert pockets_outside(CropRect(x=0, y=0, width=200, height=200), [], full) == []


class TestFitToTable:
    def test_it_keeps_every_pocket(self) -> None:
        """The property that matters: fit-to-table must satisfy the pocket
        constraint by construction, not by a margin that happens to be big
        enough."""
        full = CropRect.full(*SENSOR)
        pockets = six_pockets()
        rect = fit_to_table(boundary(), pockets, full, *SENSOR)
        assert pockets_outside(rect, pockets, full) == []

    def test_it_is_tighter_than_the_full_frame(self) -> None:
        full = CropRect.full(*SENSOR)
        rect = fit_to_table(boundary(), six_pockets(), full, *SENSOR)
        assert rect.width < SENSOR[0]
        assert rect.width * rect.height < SENSOR[0] * SENSOR[1]

    def test_it_includes_the_pockets_not_just_the_cloth(self) -> None:
        """Pockets sit outside the playing surface. A boundary-only fit clips
        them, which is the bug this design avoids."""
        full = CropRect.full(*SENSOR)
        # Pockets deliberately well outside the cloth.
        pockets = six_pockets(x0=300.0, y0=200.0, x1=2000.0, y1=1100.0)
        rect = fit_to_table(boundary(), pockets, full, *SENSOR)
        assert rect.x <= 300 - 30, "the outlying pocket should have widened the fit"
        assert pockets_outside(rect, pockets, full) == []

    def test_it_works_with_no_pockets_detected(self) -> None:
        full = CropRect.full(*SENSOR)
        rect = fit_to_table(boundary(), [], full, *SENSOR)
        assert rect.width >= MIN_CROP_PX

    def test_it_returns_sensor_space_when_already_cropped(self) -> None:
        """Fitting while cropped must not compound the offset -- the classic
        symptom is the crop marching towards a corner on every tap."""
        current = CropRect(x=300, y=200, width=1800, height=1000)
        pockets = six_pockets(x0=100.0, y0=100.0, x1=1500.0, y1=800.0)
        rect = fit_to_table(boundary(100.0, 100.0, 1500.0, 800.0), pockets, current, *SENSOR)
        # The table is at sensor 400..1800, so the fit must bracket that.
        assert rect.x < 400 and rect.x1 > 1800
        assert pockets_outside(rect, pockets, current) == []

    def test_fitting_twice_is_stable(self) -> None:
        """Idempotence, in the sense that matters: fit, apply, fit again, and the
        framing should not creep."""
        full = CropRect.full(*SENSOR)
        pockets = six_pockets()
        first = fit_to_table(boundary(), pockets, full, *SENSOR)

        # Re-express the detections in the new crop's frame space, as the loop
        # would after re-detecting.
        moved = [
            pocket(p.id, *sensor_to_frame(p.center_px.x, p.center_px.y, first), p.radius_px)
            for p in pockets
        ]
        b = boundary()
        corners = [sensor_to_frame(c.x, c.y, first) for c in b.corners()]
        moved_boundary = boundary(corners[0][0], corners[0][1], corners[1][0], corners[2][1])

        second = fit_to_table(moved_boundary, moved, first, *SENSOR)
        assert abs(second.x - first.x) <= 2
        assert abs(second.width - first.width) <= 4


# ---------------------------------------------------------------------------
# Pixel-scale consequences
# ---------------------------------------------------------------------------


class TestScaleConsequences:
    """Which quantities a crop moves, and which it does not.

    This is the substance of the design decision, so it is pinned in tests
    rather than left as a claim in a docstring.
    """

    def test_crop_scale_is_one_when_not_cropping(self) -> None:
        """The safety property the whole change rests on: an uncropped rig must
        behave byte-identically to before."""
        from app.config import Settings

        assert Settings().camera.crop_scale == 1.0

    def test_crop_scale_is_the_width_ratio(self) -> None:
        from app.config import Settings

        s = Settings()
        s.camera.width, s.camera.height = 2304, 1296
        s.camera.crop.enabled = True
        s.camera.crop.width, s.camera.crop.height = 1152, 648
        assert s.camera.crop_scale == 2.0

    def test_rotation_transposes_the_sensor_size(self) -> None:
        """The crop is applied after rotation, so a crop validated against the
        pre-rotation size would be a valid rectangle covering the wrong region --
        the hardest kind of wrong to notice."""
        from app.config import Settings

        s = Settings()
        s.camera.width, s.camera.height = 2304, 1296
        assert s.camera.rotated_size == (2304, 1296)
        s.camera.rotation_deg = 90
        assert s.camera.rotated_size == (1296, 2304)
        s.camera.rotation_deg = 180
        assert s.camera.rotated_size == (2304, 1296)

    def test_a_fraction_of_width_setting_is_corrected_for_the_crop(self) -> None:
        """``pocket_radius_frac_range`` is the one genuinely scale-relative
        setting. A fraction of frame width is invariant under a *resize* and not
        under a *crop*, so without the correction a real pocket walks up through
        the configured ceiling as you zoom in, and detection stops finding six.
        """
        from app.config import Settings

        s = Settings()
        s.camera.width, s.camera.height = 2304, 1296
        low_frac, high_frac = s.vision.pocket_radius_frac_range

        # Uncropped: the ceiling in absolute px of the downscaled image.
        small_width = 640.0
        uncropped_max = high_frac * small_width * s.camera.crop_scale

        s.camera.crop.enabled = True
        s.camera.crop.width, s.camera.crop.height = 1152, 648
        cropped_max = high_frac * small_width * s.camera.crop_scale

        # The bound doubles with a 2x crop, which is exactly what keeps it at a
        # constant *absolute* size: a 2x crop downscaled to the same 640 px
        # magnifies everything in it by 2.
        assert cropped_max == pytest.approx(uncropped_max * 2.0)
        assert low_frac > 0

    def test_absolute_pixel_settings_are_deliberately_untouched(self) -> None:
        """A crop does not magnify, so a ball is the same number of pixels
        across. ``ball_radius_px_range`` therefore needs no correction -- and it
        is only the cold-start fallback anyway, since the radius is derived from
        the table homography once the table is found.
        """
        from app.config import Settings

        s = Settings()
        before = s.vision.ball_radius_px_range
        s.camera.crop.enabled = True
        s.camera.crop.width, s.camera.crop.height = 1152, 648
        assert s.vision.ball_radius_px_range == before


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestCropStore:
    def test_a_saved_crop_round_trips(self, tmp_path) -> None:
        from vision import crop_store

        path = tmp_path / "crop.json"
        rect = CropRect(x=300, y=200, width=1400, height=800)
        crop_store.save(rect, SENSOR, path)

        loaded, present = crop_store.load(SENSOR, path)
        assert present and loaded == rect

    def test_no_file_means_fall_back_to_the_yaml(self, tmp_path) -> None:
        from vision import crop_store

        loaded, present = crop_store.load(SENSOR, tmp_path / "absent.json")
        assert loaded is None and present is False

    def test_a_deliberate_full_frame_is_recorded_not_deleted(self, tmp_path) -> None:
        """"Somebody chose the full frame" and "nothing has been chosen" want
        different behaviour on the next boot: the first must not let a stale YAML
        crop quietly come back."""
        from vision import crop_store

        path = tmp_path / "crop.json"
        crop_store.save(None, SENSOR, path)

        loaded, present = crop_store.load(SENSOR, path)
        assert loaded is None
        assert present is True, "the choice has to be distinguishable from absence"

    def test_a_crop_from_a_different_resolution_is_rejected_not_rescaled(
        self, tmp_path, caplog
    ) -> None:
        """Rescaling would hand back a rectangle nobody chose: plausible, wrong,
        and framing the table slightly off with nothing pointing here."""
        import logging

        from vision import crop_store

        path = tmp_path / "crop.json"
        crop_store.save(CropRect(x=300, y=200, width=1400, height=800), (2304, 1296), path)

        with caplog.at_level(logging.WARNING):
            loaded, present = crop_store.load((1920, 1080), path)

        assert loaded is None
        assert present is True
        assert "no longer means the same region" in caplog.text

    def test_a_corrupt_file_is_ignored_rather_than_fatal(self, tmp_path, caplog) -> None:
        import logging

        from vision import crop_store

        path = tmp_path / "crop.json"
        path.write_text("{not json", encoding="utf-8")

        with caplog.at_level(logging.ERROR):
            loaded, present = crop_store.load(SENSOR, path)

        assert loaded is None and present is False
        assert "unreadable" in caplog.text

    def test_a_file_missing_fields_is_ignored(self, tmp_path) -> None:
        from vision import crop_store

        path = tmp_path / "crop.json"
        path.write_text(json.dumps({"enabled": True, "x": 0}), encoding="utf-8")
        loaded, present = crop_store.load(SENSOR, path)
        assert loaded is None and present is False

    def test_saving_is_atomic(self, tmp_path) -> None:
        """An interrupted save must leave the previous crop, not a truncated
        file that the next boot logs as corrupt."""
        from vision import crop_store

        path = tmp_path / "crop.json"
        crop_store.save(CropRect(x=1, y=2, width=300, height=400), SENSOR, path)
        crop_store.save(CropRect(x=5, y=6, width=700, height=800), SENSOR, path)

        assert not list(tmp_path.glob("*.tmp")), "the temporary file should be gone"
        loaded, _ = crop_store.load(SENSOR, path)
        assert loaded == CropRect(x=5, y=6, width=700, height=800)
