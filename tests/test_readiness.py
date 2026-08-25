"""Readiness: is there a table to play on, and what does the projector say?

The failures worth testing here are all about *stability* rather than
correctness in the abstract. Getting the classification right on a single frame
is easy; not flapping between a game and a full-screen setup screen because
somebody reached across the table is the hard part, and it is what these pin.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.config import Settings
from app.models import SystemState
from app.readiness import Readiness, ReadinessTracker, table_detect_interval
from projection.onboarding import render_readiness_overlay, should_draw_readiness

NOW = 1000.0


@dataclass
class FakeBoundary:
    """Only the field readiness looks at. It reads confidence, not geometry."""

    confidence: float


@pytest.fixture
def tracker() -> ReadinessTracker:
    return ReadinessTracker(confirm_frames=5, min_confidence=0.45)


def feed(tracker, boundary, count=1, *, now=NOW, frame=True, last=NOW, checked=True):
    result = None
    for _ in range(count):
        result = tracker.observe(
            boundary, frame_arrived=frame, last_frame_at=last, table_checked=checked, now=now
        )
    return result


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_it_starts_in_starting(self, tracker) -> None:
        assert tracker.state is SystemState.STARTING

    def test_before_the_first_look_absence_means_nothing(self, tracker) -> None:
        """Not "no table" -- that is a finding. Announcing it before detection
        has run would flash a setup screen on every single start."""
        result = feed(tracker, None, count=20, checked=False)
        assert result.state is SystemState.STARTING

    def test_the_first_look_leaves_starting_immediately(self, tracker) -> None:
        """No hysteresis on the way out of STARTING: the first real observation
        is as good as the fifth, and lingering on a welcome screen after the
        table is already found is a worse first impression than a brief one."""
        assert feed(tracker, None, count=1).state is SystemState.NO_TABLE

    def test_a_confident_table_becomes_ready(self, tracker) -> None:
        assert feed(tracker, FakeBoundary(0.9), count=5).state is SystemState.READY

    def test_a_low_confidence_boundary_is_not_a_table(self, tracker) -> None:
        """Presence is the wrong test. Pointed at a ceiling, felt segmentation
        still returns *something* from whatever happens to be greenish, and a
        system that calls itself ready on that is worse than one that admits it
        cannot see a table."""
        assert feed(tracker, FakeBoundary(0.2), count=20).state is SystemState.NO_TABLE

    def test_a_silent_camera_is_not_a_missing_table(self, tracker) -> None:
        """They look identical from ``boundary is None`` and want opposite
        instructions -- one says check the ribbon, the other says move the
        mount."""
        result = feed(tracker, None, count=10, now=NOW + 60, last=NOW)
        assert result.state is SystemState.NO_CAMERA
        assert "ribbon" in result.detail

    def test_a_dead_camera_outranks_a_missing_table(self, tracker) -> None:
        """With no frames there is nothing to say about the table, so saying
        "no table" would send somebody to move a mount that is already right."""
        feed(tracker, None, count=10)
        assert tracker.state is SystemState.NO_TABLE

        result = feed(tracker, None, count=10, now=NOW + 60, last=NOW)
        assert result.state is SystemState.NO_CAMERA


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_one_bad_frame_does_not_end_a_game(self, tracker) -> None:
        """The whole reason for the confirmation count. A hand over the table
        must not throw a running game into a full-screen setup screen -- the
        recovery from that false positive is far more disruptive than the
        recovery from a missed frame."""
        feed(tracker, FakeBoundary(0.9), count=10)
        assert tracker.state is SystemState.READY

        feed(tracker, None, count=4)  # one short of the threshold
        assert tracker.state is SystemState.READY

    def test_a_sustained_loss_does_change_state(self, tracker) -> None:
        feed(tracker, FakeBoundary(0.9), count=10)
        assert feed(tracker, None, count=5).state is SystemState.NO_TABLE

    def test_flapping_input_never_settles_on_the_wrong_state(self, tracker) -> None:
        """Alternating observations reset the pending counter, so neither side
        ever reaches the threshold and the last committed state stands."""
        feed(tracker, FakeBoundary(0.9), count=10)
        for _ in range(20):
            feed(tracker, None, count=1)
            feed(tracker, FakeBoundary(0.9), count=1)
        assert tracker.state is SystemState.READY

    def test_recovery_also_needs_confirming(self, tracker) -> None:
        feed(tracker, None, count=10)
        feed(tracker, FakeBoundary(0.9), count=4)
        assert tracker.state is SystemState.NO_TABLE
        assert feed(tracker, FakeBoundary(0.9), count=1).state is SystemState.READY

    def test_reset_returns_to_starting(self, tracker) -> None:
        """Used on a camera reconnect: everything the tracker believed was about
        a camera that is no longer attached."""
        feed(tracker, FakeBoundary(0.9), count=10)
        tracker.reset()
        assert tracker.state is SystemState.STARTING
        assert feed(tracker, None, count=1, checked=False).state is SystemState.STARTING


# ---------------------------------------------------------------------------
# What it says
# ---------------------------------------------------------------------------


class TestCopy:
    @pytest.mark.parametrize(
        "state", [SystemState.STARTING, SystemState.NO_CAMERA, SystemState.NO_TABLE]
    )
    def test_every_non_ready_state_has_something_to_say(self, state) -> None:
        tracker = ReadinessTracker()
        tracker._state = state
        readiness = tracker.current(NOW)
        assert readiness.headline
        assert should_draw_readiness(state)

    def test_no_table_names_the_action(self, tracker) -> None:
        result = feed(tracker, None, count=10)
        assert "Mount the device" in result.detail

    def test_ready_says_nothing_on_the_felt(self) -> None:
        """The mode owns the cloth from that point, and a "Ready" banner under a
        game is exactly the interference the overlay design exists to avoid."""
        assert not should_draw_readiness(SystemState.READY)

    def test_readiness_serialises_for_the_api(self, tracker) -> None:
        payload = feed(tracker, FakeBoundary(0.9), count=10).as_dict()
        assert set(payload) == {
            "state", "headline", "detail", "since_seconds", "table_confidence", "playable"
        }
        assert payload["playable"] is True


# ---------------------------------------------------------------------------
# The projected screens
# ---------------------------------------------------------------------------


class TestOverlay:
    @pytest.fixture
    def settings(self) -> Settings:
        s = Settings()
        s.projector.width, s.projector.height = 640, 360
        return s

    @pytest.mark.parametrize(
        "state", [SystemState.STARTING, SystemState.NO_TABLE, SystemState.NO_CAMERA]
    )
    def test_each_screen_draws_something(self, settings, state) -> None:
        readiness = Readiness(state=state, headline="Headline", detail="Some instruction here.")
        canvas = render_readiness_overlay(readiness, settings, now=0.0)
        assert canvas.shape == (360, 640, 4)
        assert (canvas[:, :, 3] > 0).any()

    def test_it_needs_no_calibration(self, settings) -> None:
        """By definition these appear when the table has not been found, so
        there is no homography -- nothing here may depend on table geometry
        meaning anything."""
        readiness = Readiness(state=SystemState.NO_TABLE, headline="No table", detail="Mount it.")
        render_readiness_overlay(readiness, settings, now=0.0)  # no mapper argument exists

    def test_the_canvas_is_reused(self, settings) -> None:
        readiness = Readiness(state=SystemState.NO_TABLE, headline="No table", detail="Mount it.")
        first = render_readiness_overlay(readiness, settings, now=0.0)
        second = render_readiness_overlay(readiness, settings, canvas=first, now=0.1)
        assert second is first

    def test_a_long_instruction_is_clipped_to_two_lines(self, settings) -> None:
        """More than two lines will not be read from across a room; anything
        longer belongs on the phone."""
        from projection.onboarding import _wrap_to_width

        lines = _wrap_to_width(" ".join(["word"] * 200), 0.6, 400)
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Retry interval
# ---------------------------------------------------------------------------


class TestDetectInterval:
    def test_playing_uses_the_slow_interval(self) -> None:
        """The table has been found and does not move; re-checking is a slow
        guard against the camera being bumped."""
        assert table_detect_interval(SystemState.READY, 150) == 150

    def test_searching_looks_far_more_often(self) -> None:
        """Somebody is up a ladder and wants to know the moment it lands."""
        assert table_detect_interval(SystemState.NO_TABLE, 150) < 150

    def test_it_backs_off_as_attempts_pile_up(self) -> None:
        """A rig pointed at a ceiling searches forever, and detection costs
        ~100 ms on a Pi. Setup rates are worth paying for a minute, not a day."""
        intervals = [table_detect_interval(SystemState.NO_TABLE, 150, n) for n in range(6)]
        assert intervals == sorted(intervals)
        assert intervals[-1] == 150, "backoff must settle at the base interval, not past it"

    def test_a_dead_camera_stops_paying_for_detection(self) -> None:
        """No frames means nothing to detect in."""
        assert table_detect_interval(SystemState.NO_CAMERA, 150) == 150


# ---------------------------------------------------------------------------
# The gate, end to end
# ---------------------------------------------------------------------------


class TestReadinessGate:
    """The behaviour the integration fixture has to lower its threshold for."""

    def test_the_mode_does_not_draw_before_a_table_is_found(self) -> None:
        """Previously it started in freeplay and projected a scoreboard over an
        empty room -- which looks like working software and like broken
        detection at the same time, with no way to tell from the felt."""
        from app.main import VisionLoop
        from app.state import AppState

        settings = Settings()
        settings.camera.use_mock = settings.projector.use_mock = True
        settings.camera.width, settings.camera.height = 640, 360
        settings.projector.width, settings.projector.height = 640, 360
        settings.system.target_fps = 120
        settings.system.perf_log_interval_seconds = 1e6
        settings.system.health_log_interval_seconds = 1e6
        # High enough that a short run cannot reach READY.
        settings.system.readiness_confirm_frames = 500

        state = AppState(settings=settings)
        VisionLoop(state, max_frames=10).run()

        stages = state.tracker.snapshot().stage_ms
        assert "render" in stages, "the readiness screen never drew"
        assert "mode" not in stages, "the mode drew over a table that was not there"
        assert state.readiness.state is not SystemState.READY
