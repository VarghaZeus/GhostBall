"""Projector-assisted focus calibration.

The parts that can be tested without a lens are the parts most likely to be
subtly wrong, so they are tested hard:

* **Blob detection under defocus.** The whole design rests on finding the
  targets rather than computing where they should be, and it has to work at the
  wrong end of the focus range -- which is exactly when it is needed.
* **The analysis.** Each of the five failure modes has a specific physical
  cause and a specific instruction, and getting the wrong one sends somebody up
  a ladder for nothing. They are also *ordered*: reporting a tilted camera when
  the projector is off would be actively unhelpful.
* **Refusal.** A confident wrong number is worse than no number, so every path
  that cannot justify an answer must decline to give one.

The sweep itself is driven by a synthetic camera whose blur follows the lens,
which is enough to exercise the loop end to end -- including the exposure-drift
abort, which no real rig will reproduce on demand.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from projection.patterns import TestPattern, focus_target_centers, render_test_pattern
from vision.focus import FocusRange, TargetPeak
from vision.focus_calibration import (
    FocusDiagnosis,
    SweepOutcome,
    TargetRegion,
    analyse,
    bare_reference,
    detect_targets,
    focus_positions,
    measure_regions,
    sharpness_of,
    sweep_focus,
)

RANGE = FocusRange(0, 4095, 1, 0)


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.projector.width, s.projector.height = 1280, 720
    return s


@pytest.fixture
def pattern(settings) -> np.ndarray:
    """The target pattern as the projector would emit it, in BGR."""
    import cv2

    canvas = render_test_pattern(TestPattern.FOCUS_TARGETS, settings=settings)
    return cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGR)


def defocused(image: np.ndarray, sigma: float) -> np.ndarray:
    """The pattern as a camera at the wrong focus would see it."""
    import cv2

    if sigma <= 0:
        return image.copy()
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


# ---------------------------------------------------------------------------
# The pattern
# ---------------------------------------------------------------------------


class TestPatternGeometry:
    def test_five_targets_centre_plus_corners(self, settings) -> None:
        names = [n for n, _ in focus_target_centers(settings)]
        assert names == ["centre", "top_left", "top_right", "bottom_right", "bottom_left"]

    def test_corner_targets_are_inset_from_the_cushions(self, settings) -> None:
        """Three reasons, and all three bite at the edges: the projection is
        dimmest and most keystoned there, the target has to fit on the felt, and
        an overhead camera sees the rail partly occluding cloth near the
        cushion -- so a corner target can be clipped in the image while being
        projected perfectly."""
        centres = dict(focus_target_centers(settings))
        assert centres["top_left"].x >= 6.0
        assert centres["top_left"].y >= 6.0
        assert centres["bottom_right"].x <= settings.table.length_in - 6.0
        assert centres["bottom_right"].y <= settings.table.width_in - 6.0

    def test_the_pattern_is_pure_white_on_black(self, settings) -> None:
        """No labels, no outline, no theme colour. Every one of those is another
        thing with edges in it, and the measurement is a variance of edges --
        which is why the instructions live on the phone."""
        canvas = render_test_pattern(TestPattern.FOCUS_TARGETS, settings=settings)
        lit = canvas[canvas[:, :, 3] > 0]
        assert len(lit) > 0
        assert set(np.unique(lit[:, :3])) <= {0, 255}

    def test_it_lights_a_small_fraction_of_the_frame(self, settings) -> None:
        """Mostly black on purpose: contrast is the metric, and a projector that
        is flooding the cloth with light has nothing to be sharp against."""
        canvas = render_test_pattern(TestPattern.FOCUS_TARGETS, settings=settings)
        lit_fraction = (canvas[:, :, 3] > 0).mean()
        assert 0.01 < lit_fraction < 0.25


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestTargetDetection:
    @pytest.mark.parametrize("sigma", [0, 3, 9, 21, 35])
    def test_targets_are_found_at_every_defocus(self, pattern, sigma) -> None:
        """The property the whole approach depends on. At the far end of the
        lens range the image is badly blurred, and that is precisely when the
        targets have to be locatable -- a detector needing sharp edges would
        find nothing exactly when it matters."""
        regions = detect_targets(defocused(pattern, sigma))
        assert len(regions) == 5, f"found {len(regions)} at sigma {sigma}"

    def test_centroids_barely_move_under_defocus(self, pattern) -> None:
        """Defocus is a symmetric blur, so it spreads a blob without shifting
        it. That is what lets one set of boxes, fixed at the start, stay valid
        for the whole sweep."""
        sharp = {r.name: r.center_px for r in detect_targets(pattern)}
        blurred = {r.name: r.center_px for r in detect_targets(defocused(pattern, 21))}

        for name, (x, y) in sharp.items():
            bx, by = blurred[name]
            assert abs(bx - x) < 12 and abs(by - y) < 12, f"{name} moved"

    def test_regions_are_named_by_position(self, pattern) -> None:
        regions = {r.name: r.center_px for r in detect_targets(pattern)}
        assert set(regions) == {"centre", "top_left", "top_right", "bottom_left", "bottom_right"}
        assert regions["top_left"][0] < regions["top_right"][0]
        assert regions["top_left"][1] < regions["bottom_left"][1]

    def test_the_box_is_smaller_than_the_blob(self, pattern) -> None:
        """The outer checkers are the coarse band -- there so the blob can be
        found -- and the discriminating fine band is in the middle. Measuring
        the whole blob would dilute the fine band with squares that are resolved
        at every focus."""
        region = next(r for r in detect_targets(pattern) if r.name == "centre")
        x0, y0, x1, y1 = region.box
        assert (x1 - x0) * (y1 - y0) < region.area_px * 1.6

    def test_an_unlit_frame_finds_nothing(self) -> None:
        """The projector-is-off case, and it must be answered in two seconds
        rather than as a flat curve after a two-minute sweep."""
        assert detect_targets(np.zeros((480, 640, 3), dtype=np.uint8)) == []

    def test_small_bright_specks_are_ignored(self, pattern) -> None:
        """A reflection off a ball or a light fitting is not a target."""
        import cv2

        noisy = pattern.copy()
        for x in range(50, 300, 40):
            cv2.circle(noisy, (x, 40), 4, (255, 255, 255), -1)
        assert len(detect_targets(noisy)) == 5

    def test_long_thin_regions_are_rejected(self, pattern) -> None:
        """A rail highlight or the edge of the projection is a stripe, not a
        square."""
        import cv2

        striped = pattern.copy()
        cv2.rectangle(striped, (0, 700), (1279, 716), (255, 255, 255), -1)
        assert len(detect_targets(striped)) == 5


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------


class TestSharpness:
    def test_sharpness_falls_with_defocus(self, pattern) -> None:
        """Monotone over the range where there is still signal.

        Stopping at sigma 9 is not cherry-picking: past that the metric has
        collapsed to sensor-floor noise, where two readings differ by less than
        a percent in whichever direction. That the curve has a floor is the
        point of ``MIN_PROMINENCE`` -- a sweep living entirely down there is
        exactly the flat curve the analysis refuses to answer from.
        """
        regions = detect_targets(pattern)
        scores = [
            float(np.median(list(measure_regions(defocused(pattern, s), regions).values())))
            for s in (0, 1, 2, 3, 4)
        ]
        assert scores == sorted(scores, reverse=True), scores
        # Four orders of magnitude across that span, which is what makes the
        # peak findable at all.
        assert scores[-1] < scores[0] * 0.001

        # Past sigma ~5 the metric is on its noise floor and no longer ordered.
        # That floor is precisely why MIN_PROMINENCE exists: a sweep living
        # entirely down here is the flat curve the analysis refuses to answer
        # from, rather than a curve whose maximum means anything.
        floor = [
            float(np.median(list(measure_regions(defocused(pattern, s), regions).values())))
            for s in (5, 9, 15)
        ]
        assert max(floor) < scores[0] * 0.001

    def test_an_empty_patch_scores_zero(self) -> None:
        assert sharpness_of(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0

    def test_a_blank_patch_scores_near_zero(self) -> None:
        """Which is the whole problem with bare felt, and the reason the
        projector is involved at all."""
        assert sharpness_of(np.full((80, 80, 3), 90, dtype=np.uint8)) < 1.0


# ---------------------------------------------------------------------------
# A synthetic rig
# ---------------------------------------------------------------------------


class FakeFrame:
    def __init__(self, image):
        self.image = image
        self.timestamp = 0.0
        self.index = 0


class FakeRig:
    """A camera whose blur follows a lens, and a lens that moves.

    ``true_focus`` is where the synthetic lens is actually sharp; blur grows
    with distance from it, which is the one property the analysis depends on.
    """

    def __init__(self, pattern, true_focus=1400, scale=600.0, exposure_drift_at=None):
        self.pattern = pattern
        self.true_focus = true_focus
        self.scale = scale
        self.position = 0
        self.writes: list[int] = []
        self.exposure_drift_at = exposure_drift_at
        self.steps = 0

    # -- lens ---------------------------------------------------------------
    def read_focus(self, _device, opener=None):
        return self.position

    def write_focus(self, _device, value, opener=None):
        self.position = int(value)
        self.writes.append(int(value))

    # -- camera -------------------------------------------------------------
    def capture_frame(self):
        sigma = abs(self.position - self.true_focus) / self.scale
        return FakeFrame(defocused(self.pattern, sigma))

    def exposure_drifted(self, _status, tolerance=0.02):
        self.steps += 1
        if self.exposure_drift_at is not None and self.steps >= self.exposure_drift_at:
            return "ExposureTime moved from 8000 to 12000 during the measurement"
        return None


@pytest.fixture
def rig(pattern, monkeypatch):
    import vision.focus_calibration as fc

    fake = FakeRig(pattern)
    monkeypatch.setattr(fc, "read_focus", fake.read_focus)
    monkeypatch.setattr(
        fc, "approach_focus", lambda device, target, rng, opener=None: fake.write_focus(device, target)
    )
    return fake


class LockedExposure:
    locked = True
    detail = "exposure 8000us gain 1.5"
    baseline = {"ExposureTime": 8000}


class TestSweep:
    def test_a_clean_sweep_finds_the_true_focus(self, rig, pattern) -> None:
        regions = detect_targets(pattern)
        positions = focus_positions(RANGE, 256)

        outcome = sweep_focus(
            rig, "/dev/fake", regions, positions, RANGE,
            settle_seconds=0.0, frames=1, exposure_status=LockedExposure(),
        )
        outcome = analyse(outcome, RANGE)

        assert outcome.ok, outcome.diagnosis
        # Within one coarse step of where the synthetic lens is actually sharp.
        assert abs(outcome.best_focus - rig.true_focus) <= 256

    def test_every_target_is_measured_at_every_stop(self, rig, pattern) -> None:
        """All five come from the same frames, so the cost is per focus step
        rather than per target."""
        regions = detect_targets(pattern)
        positions = focus_positions(RANGE, 512)

        outcome = sweep_focus(
            rig, "/dev/fake", regions, positions, RANGE, settle_seconds=0.0, frames=1
        )
        assert set(outcome.curves) == {r.name for r in regions}
        for curve in outcome.curves.values():
            assert len(curve) == len(positions)

    def test_the_lens_always_arrives_from_below(self, rig, pattern) -> None:
        """A sweep that descended and a startup that ascends land in different
        places -- see vision.focus.approach_focus."""
        regions = detect_targets(pattern)
        sweep_focus(
            rig, "/dev/fake", regions, focus_positions(RANGE, 512), RANGE,
            settle_seconds=0.0, frames=1,
        )
        assert rig.writes == sorted(rig.writes)

    def test_exposure_drift_aborts_the_sweep(self, pattern, monkeypatch) -> None:
        """The failure most likely to silently corrupt the result. Sharpness
        scales with contrast, so an AE change reads as a focus change -- and
        continuing would produce a plausible curve that means nothing."""
        import vision.focus_calibration as fc

        fake = FakeRig(pattern, exposure_drift_at=3)
        monkeypatch.setattr(fc, "read_focus", fake.read_focus)
        monkeypatch.setattr(
            fc, "approach_focus",
            lambda device, target, rng, opener=None: fake.write_focus(device, target),
        )

        regions = detect_targets(pattern)
        outcome = sweep_focus(
            fake, "/dev/fake", regions, focus_positions(RANGE, 256), RANGE,
            settle_seconds=0.0, frames=1, exposure_status=LockedExposure(),
        )
        outcome = analyse(outcome, RANGE)

        assert not outcome.ok
        assert outcome.diagnosis.code == "ae_lock_failed"
        assert "discarded" in outcome.diagnosis.message

    def test_a_lens_that_never_moves_is_caught_and_named_precisely(
        self, pattern, monkeypatch
    ) -> None:
        """A lens pinned at one value is the cable case, and is now said to be
        that specifically -- "it reported the same value at every position" is
        checkable by the person reading it, where "the motor is not tracking"
        was a conclusion they had to take on trust."""
        import vision.focus_calibration as fc

        fake = FakeRig(pattern)
        monkeypatch.setattr(fc, "read_focus", lambda *a, **k: 0)  # never moves
        monkeypatch.setattr(
            fc, "approach_focus",
            lambda device, target, rng, opener=None: fake.write_focus(device, target),
        )

        regions = detect_targets(pattern)
        outcome = sweep_focus(
            fake, "/dev/fake", regions, focus_positions(RANGE, 512), RANGE,
            settle_seconds=0.0, frames=1,
        )
        outcome = analyse(outcome, RANGE)

        assert not outcome.ok
        assert outcome.diagnosis.code == "lens_not_moving"
        assert "ribbon" in outcome.diagnosis.message
        # The evidence, not just the verdict.
        assert "at every one of the" in outcome.diagnosis.message


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def outcome_from(curves: dict[str, dict[int, float]]) -> SweepOutcome:
    regions = [
        TargetRegion(name=name, center_px=(100.0 * i, 100.0 * i), box=(0, 0, 10, 10), area_px=900)
        for i, name in enumerate(curves, start=1)
    ]
    return SweepOutcome(curves=curves, regions=regions)


def peaked(peak_at: int, height: float = 1000.0, floor: float = 50.0) -> dict[int, float]:
    """A clean unimodal curve peaking at ``peak_at``."""
    return {
        p: floor + height * np.exp(-(((p - peak_at) / 400.0) ** 2))
        for p in range(0, 4096, 256)
    }


class TestDiagnosis:
    def test_a_clean_curve_yields_a_value(self) -> None:
        outcome = analyse(outcome_from({f"t{i}": peaked(1400) for i in range(5)}), RANGE)
        assert outcome.ok
        assert outcome.diagnosis is None

    def test_no_targets(self) -> None:
        outcome = analyse(outcome_from({}), RANGE)
        assert outcome.diagnosis.code == "no_targets"
        assert "projector" in outcome.diagnosis.message

    def test_a_flat_curve_is_refused(self) -> None:
        """Bare felt, or a lens that is not moving. The maximum of a flat curve
        is whichever sample caught the most sensor noise."""
        flat = {p: 100.0 + (p % 3) for p in range(0, 4096, 256)}
        outcome = analyse(outcome_from({f"t{i}": flat for i in range(5)}), RANGE)
        assert outcome.diagnosis.code == "flat_curve"
        assert not outcome.ok

    def test_a_shallow_peak_blames_ambient_light(self) -> None:
        shallow = peaked(1400, height=120.0, floor=100.0)
        outcome = analyse(outcome_from({f"t{i}": shallow for i in range(5)}), RANGE)
        assert outcome.diagnosis.code == "shallow_peak"
        assert "Dim the room" in outcome.diagnosis.message

    def test_a_peak_at_the_lens_limit_says_move_the_camera(self) -> None:
        at_max = {p: 50.0 + p / 4.0 for p in focus_positions(RANGE, 256)}
        outcome = analyse(outcome_from({f"t{i}": at_max for i in range(5)}), RANGE)
        assert outcome.diagnosis.code == "out_of_range"
        assert "further from" in outcome.diagnosis.message

    def test_a_peak_at_the_near_limit_says_the_other_direction(self) -> None:
        at_min = {p: 50.0 + (4095 - p) / 4.0 for p in focus_positions(RANGE, 256)}
        outcome = analyse(outcome_from({f"t{i}": at_min for i in range(5)}), RANGE)
        assert outcome.diagnosis.code == "out_of_range"
        assert "closer to" in outcome.diagnosis.message

    def test_a_peak_at_a_swept_edge_short_of_the_lens_limit_says_widen(self) -> None:
        """Different advice, and the difference is expensive to get wrong.

        A peak at the end of a hand-picked ``--start/--end`` window means the
        sweep was too narrow, not that the camera is at the wrong height --
        and telling someone to remount a ceiling camera when they simply need
        a wider sweep would cost them an afternoon.
        """
        # Steep enough to be a real rising edge rather than a flat band: a
        # gentle ramp across 400 counts is genuinely flat, and the analysis is
        # right to call that flat rather than out-of-range.
        narrow = {p: 50.0 + (p - 800) * 3.0 for p in focus_positions(RANGE, 64, start=800, end=1200)}
        outcome = analyse(outcome_from({f"t{i}": narrow for i in range(5)}), RANGE)
        assert outcome.diagnosis.code == "out_of_range"
        assert "Widen the sweep" in outcome.diagnosis.message
        assert "Move the camera" not in outcome.diagnosis.message

    def test_multiple_peaks_blame_vibration(self) -> None:
        twin = {}
        for p in range(0, 4096, 128):
            twin[p] = 50.0 + 900 * np.exp(-(((p - 900) / 200.0) ** 2)) + 880 * np.exp(
                -(((p - 2600) / 200.0) ** 2)
            )
        outcome = analyse(outcome_from({f"t{i}": twin for i in range(5)}), RANGE)
        assert outcome.diagnosis.code == "multiple_peaks"
        assert "vibration" in outcome.diagnosis.message

    def test_a_tilted_camera_still_yields_a_value(self) -> None:
        """Tilt is a warning alongside a valid focus, not instead of one. The
        lens still has a best position; the mount is separately wrong."""
        curves = {
            "centre": peaked(1400),
            "top_left": peaked(1100),
            "top_right": peaked(1150),
            "bottom_left": peaked(1700),
            "bottom_right": peaked(1750),
        }
        outcome = analyse(outcome_from(curves), RANGE, tilt_threshold=120)

        assert outcome.ok, "tilt must not suppress the answer"
        assert outcome.diagnosis.code == "camera_tilted"
        assert not outcome.diagnosis.fatal
        assert "Level the mount" in outcome.diagnosis.message
        assert outcome.tilt_spread >= 500

    def test_a_square_mount_reports_no_tilt(self) -> None:
        curves = {n: peaked(1400) for n in ("centre", "top_left", "top_right", "bottom_left")}
        outcome = analyse(outcome_from(curves), RANGE, tilt_threshold=120)
        assert outcome.ok
        assert outcome.diagnosis is None
        assert outcome.tilt_spread == 0

    def test_per_target_values_survive_a_bad_threshold(self) -> None:
        """The threshold is a placeholder until it is measured on the real rig,
        so the raw numbers are always reported and the real figure stays
        derivable from them."""
        curves = {"centre": peaked(1400), "top_left": peaked(1300), "top_right": peaked(1500)}
        outcome = analyse(outcome_from(curves), RANGE, tilt_threshold=1)
        assert len(outcome.peaks) == 3
        assert {p.name for p in outcome.peaks} == set(curves)
        assert all(p.peak_focus > 0 for p in outcome.peaks)

    def test_the_upstream_problem_is_reported_first(self) -> None:
        """A flat curve and a wild per-target spread happen together when the
        projector is off. Reporting the tilt would send someone up a ladder for
        a problem they do not have."""
        flat = {p: 100.0 + (p % 5) for p in range(0, 4096, 256)}
        outcome = analyse(outcome_from({f"t{i}": flat for i in range(5)}), RANGE, tilt_threshold=1)
        assert outcome.diagnosis.code == "flat_curve"


# ---------------------------------------------------------------------------
# The bare reference
# ---------------------------------------------------------------------------


class TestBareReference:
    def test_it_is_far_below_the_projected_peak(self, pattern) -> None:
        """The catch this whole second measurement exists for. Comparing a
        runtime bare-cloth reading against the projected-target peak would fire
        on every boot and look exactly like a camera that had been moved."""
        regions = detect_targets(pattern)
        peak = float(np.median(list(measure_regions(pattern, regions).values())))

        blank = FakeRig(np.full_like(pattern, 70))
        bare = bare_reference(blank, regions, frames=3)

        assert bare < peak / 10.0, f"bare {bare} vs peak {peak}"

    def test_it_survives_a_camera_returning_nothing(self) -> None:
        class Dead:
            def capture_frame(self):
                return None

        assert bare_reference(Dead(), [], frames=3) == 0.0


class TestFocusPositions:
    def test_both_ends_are_visited(self) -> None:
        positions = focus_positions(RANGE, 300)
        assert positions[0] == 0
        assert positions[-1] == 4095, "the far end must be measured, not skipped"

    def test_a_sub_range_is_respected(self) -> None:
        positions = focus_positions(RANGE, 64, start=1000, end=1500)
        assert positions[0] == 1000 and positions[-1] == 1500

    def test_reversed_bounds_are_tolerated(self) -> None:
        assert focus_positions(RANGE, 64, start=1500, end=1000)[0] == 1000


class TestDiagnosisShape:
    def test_a_diagnosis_names_an_action(self) -> None:
        """Every message has to tell somebody standing at a table what to do,
        not what was measured."""
        verbs = ("check", "move", "dim", "level", "re-run", "make sure", "is the", "wait")
        for outcome in (
            analyse(outcome_from({}), RANGE),
            analyse(outcome_from({"a": {p: 100.0 for p in range(0, 4096, 256)}}), RANGE),
        ):
            message = outcome.diagnosis.message.lower()
            assert any(v in message for v in verbs), message

    def test_peaks_carry_their_pixel_position(self) -> None:
        """So a tilt report can say which corner, and so the health check knows
        where to look later."""
        outcome = analyse(outcome_from({"centre": peaked(1400)}), RANGE)
        assert isinstance(outcome.peaks[0], TargetPeak)
        assert outcome.peaks[0].center_px == (100.0, 100.0)

    def test_a_fatal_diagnosis_suppresses_the_value(self) -> None:
        assert FocusDiagnosis(code="x", message="y").fatal is True


# ---------------------------------------------------------------------------
# Telling readback failures apart
# ---------------------------------------------------------------------------


class TestReadbackDiagnosis:
    """The message has to name the right cause, because it sends someone
    somewhere physical.

    This was one message on a bare mismatch count: "the motor is not tracking --
    check the camera ribbon". At least four distinct faults produce a disagreeing
    readback and only one of them is a cable, so that message was wrong more
    often than it was right -- and confidently enough that the real cause went
    unexamined while somebody reseated a connector that was fine.
    """

    DW9807 = FocusRange(0, 1023, 1, 480)

    def _diagnose(self, pairs, focus_range=None):
        from vision.focus_calibration import ReadbackSample, readback_diagnosis

        samples = [ReadbackSample(written=w, read=r) for w, r in pairs]
        return readback_diagnosis(samples, focus_range or self.DW9807)

    def test_a_clamped_out_of_range_request_does_not_blame_the_cable(self) -> None:
        """The case that prompted all of this. Positions computed against the old
        lens's 0-4095 get clamped by a 0-1023 driver, the readback disagrees for
        that reason alone, and reseating the ribbon cannot possibly help."""
        d = self._diagnose([(0, 0), (2048, 1023), (4095, 1023)])

        assert d.code == "focus_out_of_range"
        assert "0-1023" in d.message
        assert "software fault" in d.message
        assert "ribbon" not in d.message, "a clamp is not a cable problem"

    def test_a_lens_pinned_at_one_value_is_called_out_specifically(self) -> None:
        d = self._diagnose([(0, 480), (256, 480), (512, 480), (768, 480)])

        assert d.code == "lens_not_moving"
        assert "480" in d.message
        # Naming it as the power-on default is a strong hint the lens is simply
        # never being driven.
        assert "power-on default" in d.message
        assert "ribbon" in d.message

    def test_a_driver_that_quantises_is_not_reported_as_a_fault(self) -> None:
        """Nothing is broken: the positions asked for were not expressible."""
        coarse = FocusRange(0, 1000, 8, 0)
        d = self._diagnose([(100, 96), (300, 296)], coarse)

        assert d.code == "focus_quantised"
        assert "Nothing is faulty" in d.message
        assert "ribbon" not in d.message

    def test_a_lens_moving_but_not_to_order_names_both_causes(self) -> None:
        """In range, varying, and never the requested value. That is either a
        marginal cable or something else driving the same motor -- and the second
        is what a sensor with a libcamera AF algorithm does."""
        d = self._diagnose([(0, 12), (256, 300), (512, 130), (768, 800)])

        assert d.code == "lens_not_tracking"
        assert "not a clamp" in d.message
        assert "ribbon" in d.message
        assert "AF algorithm" in d.message

    def test_every_diagnosis_carries_the_numbers_it_was_based_on(self) -> None:
        """The verdict is a guess; the pairs are the observation. A reader who
        doubts the guess must still be able to see what happened."""
        for pairs in (
            [(0, 0), (4095, 1023)],
            [(0, 480), (512, 480)],
            [(0, 12), (256, 300)],
        ):
            message = self._diagnose(pairs).message
            assert "sent" in message and "read" in message, message

    def test_a_clean_sweep_is_not_a_diagnosis(self) -> None:
        d = self._diagnose([(0, 0), (256, 256), (512, 512)])
        assert not d.fatal
        assert d.message == ""

    def test_the_pair_list_is_truncated_so_the_message_stays_readable(self) -> None:
        """Thirty-two stops of evidence is not a message anyone reads."""
        d = self._diagnose([(p, p + 7) for p in range(0, 900, 32)])
        assert "and " in d.message and "more" in d.message
        assert d.message.count("sent") <= 5


class TestSweepDensityFollowsTheLens:
    """A stride is only meaningful relative to a span.

    The sweep stride was hardcoded at 128: 33 stops on the ak7375's 0-4095, and
    9 on a dw9807's 0-1023. Nine stops is far too coarse to locate a peak, so the
    fine pass starts from the least-bad of nine samples rather than from a real
    maximum -- a silent halving of calibration quality caused by changing camera.
    """

    def test_both_lenses_get_a_comparable_number_of_stops(self) -> None:
        from vision.focus_calibration import COARSE_SWEEP_SAMPLES, coarse_step, focus_positions

        for focus_range in (FocusRange(0, 4095, 1, 0), FocusRange(0, 1023, 1, 480)):
            stops = focus_positions(focus_range, coarse_step(focus_range))
            assert len(stops) >= COARSE_SWEEP_SAMPLES, (
                f"{focus_range.maximum}: only {len(stops)} stops"
            )

    def test_the_old_lens_keeps_the_stride_it_was_tuned_with(self) -> None:
        """~128 on the ak7375, so this generalisation does not quietly retune the
        lens the value was chosen against."""
        from vision.focus_calibration import coarse_step

        assert 120 <= coarse_step(FocusRange(0, 4095, 1, 0)) <= 128

    def test_the_new_lens_gets_a_much_finer_stride_than_the_old_constant(self) -> None:
        from vision.focus_calibration import coarse_step

        assert coarse_step(FocusRange(0, 1023, 1, 480)) < 128

    def test_the_stride_is_never_finer_than_the_driver_can_express(self) -> None:
        from vision.focus_calibration import coarse_step

        coarse = FocusRange(0, 100, 8, 0)
        assert coarse_step(coarse) >= coarse.step

    def test_positions_are_step_aligned_and_unique(self) -> None:
        """So a quantising driver cannot turn a position this code chose into a
        readback mismatch, and so snapping cannot produce duplicate stops."""
        from vision.focus_calibration import focus_positions

        coarse = FocusRange(0, 1000, 8, 0)
        stops = focus_positions(coarse, 20)
        assert all(p % 8 == 0 for p in stops), stops
        assert len(stops) == len(set(stops))
        assert stops == sorted(stops), "ascending order is load-bearing"
