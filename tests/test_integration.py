"""Phase 8: the main loop, end to end.

Everything else in this suite tests a component against synthetic input. This
file tests the thing that has to hold all of them together for hours, so the
questions are different: not "is the answer right" but "does the run survive
what will happen to it". Concretely --

* a stage that throws is contained, counted and eventually switched off, and
  the loop keeps producing frames without it (:class:`TestStageContainment`);
* a camera that disappears comes back (:class:`TestCameraRecovery`);
* falling behind sheds work instead of accumulating latency
  (:class:`TestAdaptiveDegradation`);
* a wedged loop is detectable from outside (:class:`TestWatchdog`);
* shutdown releases the projector and the camera (:class:`TestShutdown`).

All of it runs against the mock camera and a discarded display, so the suite
needs no hardware and no table. Every test that runs the loop bounds it with
``max_frames``: an unbounded loop in a test suite is a hang waiting for a bad
day, and ``--frames`` is a shipped flag rather than a test hook, so this
exercises the real path.
"""

from __future__ import annotations

import csv
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import VisionLoop, create_app, parse_args
from app.state import AppState
from utils.performance import FrameProfiler, FrameRecord, LoopWatchdog, PerformanceTracker


@pytest.fixture
def settings() -> Settings:
    """Mock hardware, small frames, and no chatter.

    640x360 rather than the configured 1080p: the loop's behaviour under
    failure is resolution-independent, and a 1080p mock frame costs ~30 ms of
    synthesis per frame, which turns a 30-frame test into a second of wall
    clock for no added coverage.
    """
    s = Settings()
    s.camera.use_mock = True
    s.projector.use_mock = True
    s.camera.width, s.camera.height = 640, 360
    s.projector.width, s.projector.height = 640, 360
    # Loud enough to matter in a test would mean 30 log lines a second.
    s.system.perf_log_interval_seconds = 1e6
    s.system.health_log_interval_seconds = 1e6
    # The frame rate limiter would otherwise make a 30-frame test take a second
    # per test, and none of these assertions are about wall-clock pacing.
    s.system.target_fps = 120
    return s


@pytest.fixture
def state(settings: Settings) -> AppState:
    return AppState(settings=settings)


def run_loop(state: AppState, frames: int = 12, **kwargs) -> VisionLoop:
    """Run the loop to completion for a bounded number of frames."""
    loop = VisionLoop(state, max_frames=frames, **kwargs)
    loop.run()
    return loop


# ---------------------------------------------------------------------------
# The loop actually runs
# ---------------------------------------------------------------------------


class TestPipelineRuns:
    def test_loop_completes_the_requested_frames(self, state: AppState) -> None:
        run_loop(state, frames=10)
        assert state.frames_processed == 10
        assert state.tracker.total_frames == 10
        assert not state.is_running

    def test_every_stage_is_exercised(self, state: AppState) -> None:
        """The whole pipeline runs, not just the parts that cannot fail.

        Asserted through the timing map rather than by inspecting outputs,
        because a stage that silently stopped being called is precisely the
        integration bug this file exists to catch -- and it leaves every
        component's own unit tests passing.
        """
        run_loop(state, frames=12)
        stages = state.tracker.snapshot().stage_ms
        # 'table' runs on frame 0, 'physics' only while someone is aiming, and
        # 'render' only under a test-pattern override -- so those three are
        # covered by their own tests rather than asserted here.
        for stage in ("capture", "detect", "mode", "project"):
            assert stage in stages, f"stage {stage!r} never ran: {sorted(stages)}"

    def test_detection_and_table_geometry_reach_the_shared_state(
        self, state: AppState
    ) -> None:
        run_loop(state, frames=6)
        assert state.latest_frame is not None
        assert state.latest_game_state is not None
        # The mock camera draws a felt-coloured table, so the boundary and the
        # homography derived from it should both be populated.
        assert state.table_boundary is not None
        assert state.camera_to_table is not None

    def test_projector_receives_an_overlay(self, state: AppState) -> None:
        run_loop(state, frames=4)
        # The display is the mock, which discards frames -- so assert on the
        # loop's own record of what it sent rather than on the backend.
        assert state.latest_overlay is not None
        assert state.latest_overlay.shape == (
            state.settings.projector.height,
            state.settings.projector.width,
            4,
        ), "the projector must be fed an RGBA overlay at projector resolution"

    def test_health_summary_reflects_a_clean_run(self, state: AppState) -> None:
        run_loop(state, frames=8)
        health = state.health_summary()
        assert health["frames_processed"] == 8
        assert health["camera_reconnects"] == 0
        assert health["stage_errors"] == {}
        assert health["disabled_stages"] == []
        assert health["last_error"] is None

    def test_a_test_pattern_override_outranks_the_mode(self, state: AppState) -> None:
        """Someone aligning the projector must not get a trajectory line."""
        state.projection_override = "grid"
        run_loop(state, frames=4)
        stages = state.tracker.snapshot().stage_ms
        assert "render" in stages, "the pattern renderer never ran"
        assert "mode" not in stages, "the mode drew over an alignment pattern"

    def test_an_unknown_override_is_dropped_rather_than_fatal(
        self, state: AppState
    ) -> None:
        state.projection_override = "not-a-pattern"
        run_loop(state, frames=3)
        assert state.projection_override is None
        assert state.frames_processed == 3

    def test_a_blank_request_is_consumed_by_the_loop(self, state: AppState) -> None:
        """The web layer never touches the display, so the loop must clear it."""
        state.request_blank()
        run_loop(state, frames=2)
        assert state.blank_requested is False


# ---------------------------------------------------------------------------
# Stage failure containment
# ---------------------------------------------------------------------------


class TestStageContainment:
    def test_a_throwing_stage_does_not_stop_the_loop(self, state: AppState) -> None:
        loop = VisionLoop(state, max_frames=10)

        def boom(_frame):
            raise RuntimeError("detector exploded")

        loop._detect = boom
        loop.run()

        assert state.frames_processed == 10, "one broken stage ended the run"
        assert state.stage_errors["detect"] == 10
        assert "detect" in (state.last_error or "")

    def test_a_persistently_failing_stage_is_switched_off(
        self, state: AppState
    ) -> None:
        """Past the limit the stage is abandoned, and the loop carries on.

        The point of disabling rather than continuing to retry is cost: a stage
        that has failed on 30 consecutive frames will fail on the 31st, and
        paying for the attempt plus the traceback 30 times a second for the rest
        of a two-hour session is a meaningful fraction of the frame budget.
        """
        state.settings.system.stage_failure_limit = 5
        loop = VisionLoop(state, max_frames=12)
        loop._detect = lambda _frame: (_ for _ in ()).throw(RuntimeError("nope"))
        loop.run()

        assert "detect" in state.disabled_stages
        # Five failures to hit the limit, then no further attempts.
        assert state.stage_errors["detect"] == 5
        assert state.frames_processed == 12

    def test_an_intermittent_failure_never_disables_a_stage(
        self, state: AppState
    ) -> None:
        """One bad frame in three is noisy, not broken.

        The counter that decides is consecutive failures, and this is the case
        it exists for: disabling the detector for the rest of the session
        because of a flaky frame would be a far worse outcome than the flake.
        """
        state.settings.system.stage_failure_limit = 3
        calls = {"n": 0}
        real_detect = VisionLoop._detect

        def flaky(self_, frame):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise RuntimeError("transient")
            return real_detect(self_, frame)

        loop = VisionLoop(state, max_frames=15)
        loop._detect = lambda frame: flaky(loop, frame)
        loop.run()

        assert state.stage_errors["detect"] >= 4
        assert "detect" not in state.disabled_stages
        assert state.latest_game_state is not None, "good frames still produced state"

    def test_an_unimplemented_stage_is_not_counted_as_an_error(
        self, state: AppState
    ) -> None:
        """``NotImplementedError`` means "not built", which is not a failure.

        Worth separating because the two want opposite handling: an
        unimplemented stage should be mentioned once and then ignored forever,
        while a broken one should be counted, surfaced on the panel, and
        eventually disabled.
        """
        loop = VisionLoop(state, max_frames=6)
        loop._simulate = lambda _gs: (_ for _ in ()).throw(NotImplementedError("phase 5"))
        loop._detect = lambda _frame: (_ for _ in ()).throw(NotImplementedError("phase 3"))
        loop.run()

        assert state.stage_errors == {}
        assert state.disabled_stages == set()
        assert state.last_error is None
        assert state.frames_processed == 6

    def test_a_disabled_mode_keeps_the_last_overlay_on_the_felt(
        self, state: AppState
    ) -> None:
        """A frozen scoreboard beats an unlit table mid-game."""
        state.settings.system.stage_failure_limit = 2
        loop = VisionLoop(state, max_frames=8)

        real_update = state.mode_manager.update
        calls = {"n": 0}

        def failing_update(game_state, prediction):
            calls["n"] += 1
            if calls["n"] <= 2:
                return real_update(game_state, prediction)
            raise RuntimeError("renderer exploded")

        state.mode_manager.update = failing_update
        loop.run()

        assert "mode" in state.disabled_stages
        # Not just after disabling: the frames between the first failure and the
        # disable must hold the overlay too, or the table flickers black and
        # then freezes, which is the worst of both.
        assert state.latest_overlay is not None, "the projector went dark"


# ---------------------------------------------------------------------------
# Camera recovery
# ---------------------------------------------------------------------------


class TestCameraRecovery:
    def test_a_lost_camera_is_reopened_and_the_loop_continues(
        self, state: AppState
    ) -> None:

        state.settings.system.camera_recovery_seconds = 5.0
        loop = VisionLoop(state, max_frames=8)

        opened = {"n": 0}
        real_open_devices = loop._open_devices

        def open_devices():
            opened["n"] += 1
            ok = real_open_devices()
            if ok:
                _wrap_camera_to_fail_once(state)
            return ok

        loop._open_devices = open_devices
        loop.run()

        assert state.camera_reconnects == 1
        assert state.frames_processed == 8, "the loop did not resume after recovery"

    def test_recovery_gives_up_and_stops_rather_than_spinning(
        self, state: AppState, monkeypatch
    ) -> None:
        """An unplugged camera is not coming back; the loop must not spin on it.

        The reopen is what has to fail here, not just the first capture --
        recovery builds a *new* ``Camera``, so sabotaging the old object's
        ``capture_frame`` tests nothing but the fixture. Patching ``open`` is
        what actually models a device that is no longer on the bus.

        The budget is short (0.4 s) purely to keep the test quick; what is under
        test is that the budget is honoured at all, since a loop retrying
        forever burns a core and hides the failure behind a process that still
        looks alive.
        """
        import vision.camera as camera_module
        from vision.camera import CameraError

        state.settings.system.camera_recovery_seconds = 0.4
        loop = VisionLoop(state, max_frames=100)

        real_open_devices = loop._open_devices

        def open_devices():
            ok = real_open_devices()
            if ok:
                state.camera.capture_frame = lambda: (_ for _ in ()).throw(
                    CameraError("device gone")
                )
                # Only now, so the initial open above still succeeds.
                monkeypatch.setattr(
                    camera_module.Camera,
                    "open",
                    lambda self: (_ for _ in ()).throw(CameraError("no such device")),
                )
            return ok

        loop._open_devices = open_devices

        started = time.perf_counter()
        loop.run()
        elapsed = time.perf_counter() - started

        assert state.camera_reconnects == 0
        assert state.frames_processed == 0
        assert not state.is_running
        # Generous upper bound: the assertion is "bounded", not "fast".
        assert elapsed < 10.0, "recovery ran past its budget"

    def test_recovery_is_interrupted_by_a_stop_request(self, state: AppState) -> None:
        """Shutdown during recovery must be immediate, not one backoff later."""
        from vision.camera import CameraError

        state.settings.system.camera_recovery_seconds = 30.0
        loop = VisionLoop(state, max_frames=100)

        real_open_devices = loop._open_devices

        def open_devices():
            ok = real_open_devices()
            if ok:
                state.camera.capture_frame = lambda: (_ for _ in ()).throw(
                    CameraError("device gone")
                )
            return ok

        loop._open_devices = open_devices

        thread = loop.start()
        time.sleep(0.3)
        started = time.perf_counter()
        loop.stop(timeout=5.0)
        elapsed = time.perf_counter() - started

        assert not thread.is_alive()
        assert elapsed < 3.0, "stop waited for the full recovery budget"

    def test_tracks_are_dropped_across_a_reconnect(self, state: AppState) -> None:
        """Positions from before a dropout are not evidence about after it.

        Keeping them would produce a velocity the size of the table on the first
        frame back, which the state machine reads as a shot.
        """
        loop = VisionLoop(state, max_frames=6)
        resets = {"n": 0}
        real_reset = state.tracker_balls.reset
        state.tracker_balls.reset = lambda: (resets.__setitem__("n", resets["n"] + 1), real_reset())[1]

        real_open_devices = loop._open_devices

        def open_devices():
            ok = real_open_devices()
            if ok:
                _wrap_camera_to_fail_once(state)
            return ok

        loop._open_devices = open_devices
        loop.run()

        assert state.camera_reconnects == 1
        assert resets["n"] >= 1


def _wrap_camera_to_fail_once(state: AppState) -> None:
    """Make the next capture raise, then behave normally after a reopen.

    The wrapper is installed on the *camera object*, and recovery replaces that
    object wholesale -- which is exactly why one failure is all this produces.
    """
    from vision.camera import CameraError

    camera = state.camera
    real_capture = camera.capture_frame
    fired = {"done": False}

    def capture():
        if not fired["done"]:
            fired["done"] = True
            raise CameraError("simulated USB dropout")
        return real_capture()

    camera.capture_frame = capture


# ---------------------------------------------------------------------------
# Load shedding
# ---------------------------------------------------------------------------


class TestAdaptiveDegradation:
    def _tracker_at(self, state: AppState, frame_ms: float) -> None:
        """Fill the rolling window with frames of a given cost."""
        tracker = state.tracker
        for _ in range(tracker.window):
            tracker._frame_times.append(frame_ms)

    def test_sustained_overruns_shed_load(self, state: AppState) -> None:
        loop = VisionLoop(state)
        budget = state.tracker.frame_budget_ms
        self._tracker_at(state, budget * 3)

        loop._adapt()
        assert state.degradation_level == 1
        loop._adapt()
        assert state.degradation_level == 2

    def test_degradation_is_capped(self, state: AppState) -> None:
        """Shedding stops at level 2; below that there is nothing left to give
        up that would not change what the system does rather than how fast."""
        loop = VisionLoop(state)
        self._tracker_at(state, state.tracker.frame_budget_ms * 10)
        for _ in range(6):
            loop._adapt()
        assert state.degradation_level == 2

    def test_load_is_restored_when_there_is_room(self, state: AppState) -> None:
        loop = VisionLoop(state)
        state.degradation_level = 2
        self._tracker_at(state, state.tracker.frame_budget_ms * 0.5)

        loop._adapt()
        assert state.degradation_level == 1
        loop._adapt()
        assert state.degradation_level == 0

    def test_hysteresis_leaves_the_level_alone_in_the_dead_band(
        self, state: AppState
    ) -> None:
        """Between 1.1x and 1.5x of budget, nothing moves.

        Without the gap the loop oscillates between levels every couple of
        seconds, which is visible: the overlay changes character, recovers, and
        changes back.
        """
        loop = VisionLoop(state)
        state.degradation_level = 1
        self._tracker_at(state, state.tracker.frame_budget_ms * 1.3)

        loop._adapt()
        assert state.degradation_level == 1

    def test_degradation_can_be_switched_off(self, state: AppState) -> None:
        state.settings.system.adaptive_degradation = False
        loop = VisionLoop(state)
        self._tracker_at(state, state.tracker.frame_budget_ms * 5)

        loop._adapt()
        assert state.degradation_level == 0

    def test_table_detection_backs_off_with_the_level(self, state: AppState) -> None:
        loop = VisionLoop(state)
        base = loop._table_interval()
        state.degradation_level = 1
        assert loop._table_interval() == base * 2
        state.degradation_level = 2
        assert loop._table_interval() == base * 4

    def test_no_prediction_while_the_balls_are_moving(self, state: AppState) -> None:
        """The renderer draws no aiming line during a shot, so simulating one is
        pure waste at the moment the frame budget is tightest."""
        from app.models import SessionState

        loop = VisionLoop(state)
        state.mode_manager.session.state = SessionState.AIMING
        assert loop._should_simulate(0)

        state.mode_manager.session.state = SessionState.SHOT_IN_PROGRESS
        assert not loop._should_simulate(0)

    def test_prediction_halves_its_rate_under_degradation(
        self, state: AppState
    ) -> None:
        from app.models import SessionState

        loop = VisionLoop(state)
        state.mode_manager.session.state = SessionState.AIMING
        state.degradation_level = 1
        assert loop._should_simulate(0)
        assert not loop._should_simulate(1)

    def test_manual_cue_control_suppresses_prediction(self, state: AppState) -> None:
        loop = VisionLoop(state)
        state.auto_detect_cue = False
        assert not loop._should_simulate(0)


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class TestWatchdog:
    def test_nothing_is_stale_before_the_first_frame(self) -> None:
        """A loop that has not started yet is not a stalled one.

        Reporting a stall here would fire on every boot, and an alert that
        always fires is one nobody reads by the time it means something.
        """
        tracker = PerformanceTracker()
        dog = LoopWatchdog(tracker, stall_seconds=0.01)
        assert not dog.check()
        assert dog.stall_count == 0

    def test_a_stale_heartbeat_is_a_stall(self) -> None:
        tracker = PerformanceTracker()
        tracker.begin_frame()
        tracker.end_frame()
        dog = LoopWatchdog(tracker, stall_seconds=5.0)

        assert not dog.check()
        # Ten seconds in the future rather than sleeping: the clock is injectable
        # precisely so this test costs nothing.
        assert dog.check(now=time.perf_counter() + 10.0)
        assert dog.stalled
        assert dog.stall_count == 1

    def test_one_stall_counts_once_however_long_it_lasts(self) -> None:
        """The panel needs "this happened twice", not a number that climbs by
        one per poll while nothing new is wrong."""
        tracker = PerformanceTracker()
        tracker.begin_frame()
        tracker.end_frame()
        dog = LoopWatchdog(tracker, stall_seconds=1.0)

        base = time.perf_counter()
        for offset in (5.0, 6.0, 7.0, 8.0):
            dog.check(now=base + offset)
        assert dog.stall_count == 1

    def test_recovery_clears_the_flag(self) -> None:
        tracker = PerformanceTracker()
        tracker.begin_frame()
        tracker.end_frame()
        dog = LoopWatchdog(tracker, stall_seconds=1.0)

        dog.check(now=time.perf_counter() + 5.0)
        assert dog.stalled

        tracker.begin_frame()
        tracker.end_frame()
        assert not dog.check()
        assert not dog.stalled
        assert dog.stall_count == 1

    def test_the_loop_wires_the_watchdog_into_the_shared_state(
        self, state: AppState
    ) -> None:
        """A stall has to reach ``/api/status``, or it is invisible."""
        loop = VisionLoop(state)
        state.settings.system.watchdog_stall_seconds = 0.05
        dog = loop._start_watchdog()
        try:
            state.tracker.begin_frame()
            state.tracker.end_frame()
            time.sleep(0.05)
            dog.check(now=time.perf_counter() + 10.0)
            assert state.loop_stalled
            assert state.stall_count == 1
            assert "stalled" in (state.last_error or "")

            state.tracker.begin_frame()
            state.tracker.end_frame()
            dog.check()
            assert not state.loop_stalled
        finally:
            dog.stop()

    def test_the_watchdog_thread_stops(self, state: AppState) -> None:
        dog = LoopWatchdog(state.tracker, stall_seconds=1.0, poll_seconds=0.01).start()
        dog.stop(timeout=2.0)
        assert dog._thread is None


# ---------------------------------------------------------------------------
# Per-frame profiling
# ---------------------------------------------------------------------------


class TestFrameProfiler:
    def test_a_run_writes_one_row_per_frame(self, state: AppState, tmp_path) -> None:
        path = tmp_path / "trace.csv"
        run_loop(state, frames=7, profile_path=path)

        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 7
        assert [int(r["frame"]) for r in rows] == list(range(1, 8))
        for row in rows:
            assert float(row["total_ms"]) >= 0.0
            assert float(row["capture_ms"]) >= 0.0

    def test_a_stage_that_did_not_run_is_blank_rather_than_zero(
        self, state: AppState, tmp_path
    ) -> None:
        """"Skipped" and "took no time" are different facts, and conflating them
        is how a stage silently stops running without anyone noticing."""
        path = tmp_path / "trace.csv"
        run_loop(state, frames=6, profile_path=path)

        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        # 'table' runs on frame 0 only, so later rows must be blank there.
        assert rows[-1]["table_ms"] == ""

    def test_the_sink_is_detached_after_the_run(
        self, state: AppState, tmp_path
    ) -> None:
        """A closed file left wired into the tracker would raise on every frame
        of any subsequent run."""
        run_loop(state, frames=3, profile_path=tmp_path / "trace.csv")
        assert state.tracker.frame_sink is None

    def test_unknown_stages_land_in_the_extra_column(self, tmp_path) -> None:
        path = tmp_path / "trace.csv"
        with FrameProfiler(path) as profiler:
            profiler.record(
                FrameRecord(
                    index=1,
                    at=1.0,
                    total_ms=10.0,
                    latency_ms=12.0,
                    stages={"capture": 1.0, "invented": 4.0},
                )
            )
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(iter(csv.DictReader(handle)))
        assert float(row["extra_ms"]) == pytest.approx(4.0)

    def test_a_broken_sink_disables_itself_instead_of_the_run(self) -> None:
        tracker = PerformanceTracker()
        tracker.frame_sink = lambda _record: (_ for _ in ()).throw(OSError("disk full"))

        tracker.begin_frame()
        tracker.end_frame()

        assert tracker.frame_sink is None
        assert tracker.total_frames == 1

    def test_stage_times_are_per_frame_not_cumulative(self) -> None:
        tracker = PerformanceTracker()
        seen: list[FrameRecord] = []
        tracker.frame_sink = seen.append

        tracker.begin_frame()
        with tracker.stage("detect"):
            pass
        tracker.end_frame()

        tracker.begin_frame()
        with tracker.stage("project"):
            pass
        tracker.end_frame()

        assert "detect" in seen[0].stages and "project" not in seen[0].stages
        assert "project" in seen[1].stages and "detect" not in seen[1].stages


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_stop_ends_the_thread_promptly(self, state: AppState) -> None:
        loop = VisionLoop(state)  # unbounded: only stop() ends it
        thread = loop.start()
        # Wait for the loop to actually be running, rather than sleeping a fixed
        # interval and hoping -- a slow CI box would otherwise stop a loop that
        # had not started and pass for the wrong reason.
        deadline = time.perf_counter() + 5.0
        while not state.is_running and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert state.is_running

        loop.stop(timeout=5.0)
        assert not thread.is_alive()
        assert not state.is_running

    def test_shutdown_releases_the_camera_and_the_projector(
        self, state: AppState
    ) -> None:
        """Left open, the camera stays claimed and the projector stays lit --
        both of which look like a hung system to whoever is in the room."""
        run_loop(state, frames=3)
        assert not state.camera.is_open
        assert not state.display.is_open

    def test_the_projector_is_cleared_on_the_way_out(self, state: AppState) -> None:
        cleared = {"n": 0}
        loop = VisionLoop(state, max_frames=3)
        real_open = loop._open_devices

        def open_devices():
            ok = real_open()
            if ok:
                real_clear = state.display.clear
                state.display.clear = lambda: (
                    cleared.__setitem__("n", cleared["n"] + 1),
                    real_clear(),
                )[1]
            return ok

        loop._open_devices = open_devices
        loop.run()
        assert cleared["n"] >= 1, "the last overlay was left on the felt"

    def test_a_failure_to_open_leaves_the_loop_stopped_not_crashed(
        self, state: AppState
    ) -> None:
        loop = VisionLoop(state, max_frames=5)
        loop._open_devices = lambda: False
        loop.run()
        assert not state.is_running
        assert state.frames_processed == 0

    def test_stop_is_safe_before_the_loop_ever_started(self, state: AppState) -> None:
        VisionLoop(state).stop(timeout=0.1)
        assert state.stop_event.is_set()


# ---------------------------------------------------------------------------
# Web integration
# ---------------------------------------------------------------------------


class TestWebIntegration:
    def test_the_app_runs_the_loop_and_serves_status_together(
        self, state: AppState
    ) -> None:
        """The reason the loop is a thread rather than a coroutine: the panel
        has to stay responsive while the pipeline is saturating a core."""
        app = create_app(state, start_loop=True, max_frames=40)
        with TestClient(app) as client:
            deadline = time.perf_counter() + 10.0
            while state.frames_processed == 0 and time.perf_counter() < deadline:
                time.sleep(0.01)

            response = client.get("/api/status")
            assert response.status_code == 200
            body = response.json()
            assert body["health"]["frames_processed"] > 0
            assert body["health"]["calibration_source"] in ("file", "identity")

    def test_health_reports_degraded_when_the_loop_is_stalled(
        self, state: AppState
    ) -> None:
        """A monitor polling this must be able to tell a wedged loop from an
        idle one; from outside they look identical."""
        app = create_app(state, start_loop=False)
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

            state.loop_stalled = True
            # start_loop=False means the loop is not expected to run, so the
            # stall flag alone must not flip the verdict.
            assert client.get("/health").status_code == 200

        app = create_app(state, start_loop=True, max_frames=1)
        with TestClient(app) as client:
            state.is_running = True
            state.loop_stalled = True
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "degraded"

    def test_status_survives_a_loop_that_never_started(self, state: AppState) -> None:
        app = create_app(state, start_loop=False)
        with TestClient(app) as client:
            body = client.get("/api/status").json()
            assert body["running"] is False
            assert body["health"]["frames_processed"] == 0
            assert body["health"]["uptime_seconds"] >= 0.0

    def test_the_loop_stops_when_the_app_shuts_down(self, state: AppState) -> None:
        """Ctrl-C on uvicorn has to reach the vision thread, or the process
        exits with the projector still lit."""
        app = create_app(state, start_loop=True)
        with TestClient(app) as client:
            client.get("/health")
            deadline = time.perf_counter() + 10.0
            while not state.is_running and time.perf_counter() < deadline:
                time.sleep(0.01)
            assert state.is_running
        assert not state.is_running
        assert state.stop_event.is_set()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCommandLine:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.frames is None
        assert not args.mock and not args.headless and not args.no_loop
        assert args.profile is None

    def test_flags_parse(self, tmp_path) -> None:
        args = parse_args(
            ["--mock", "--headless", "--frames", "50", "--profile", str(tmp_path / "t.csv")]
        )
        assert args.mock and args.headless
        assert args.frames == 50
        assert args.profile.name == "t.csv"

    def test_headless_with_no_loop_is_rejected(self) -> None:
        """The combination asks for neither a loop nor a server."""
        from app.main import main

        assert main(["--headless", "--no-loop", "--mock"]) == 2

    def test_signal_handlers_install_without_a_main_thread(
        self, state: AppState
    ) -> None:
        """Installing handlers off the main thread raises in CPython, and that
        must degrade to "no handlers" rather than to a failure to start."""
        from app.main import _install_signal_handlers

        errors: list[BaseException] = []

        def target() -> None:
            try:
                _install_signal_handlers(state)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=5.0)
        assert not errors

    def test_a_signal_asks_the_loop_to_stop_rather_than_killing_it(
        self, state: AppState
    ) -> None:
        """SIGTERM is what systemd sends on ``stop`` and ``restart``.

        Without a handler the process dies between two frames with the projector
        still lit and the camera still claimed, and the next start finds the
        device busy. The handler is invoked directly here because delivering a
        real signal is not portable -- Windows has no meaningful SIGTERM -- and
        what needs testing is what the handler does, not that the OS can send
        one.
        """
        import signal as signal_module

        from app.main import _install_signal_handlers

        # Restored afterwards: leaving our handler installed would hijack Ctrl-C
        # for the rest of the pytest process.
        previous = {
            sig: signal_module.getsignal(sig)
            for sig in (signal_module.SIGINT, signal_module.SIGTERM)
        }
        try:
            _install_signal_handlers(state)
            handler = signal_module.getsignal(signal_module.SIGINT)
            assert callable(handler)
            assert handler is not previous[signal_module.SIGINT]

            assert not state.stop_event.is_set()
            handler(signal_module.SIGINT, None)
            assert state.stop_event.is_set()
            assert not state.is_running
        finally:
            for sig, original in previous.items():
                signal_module.signal(sig, original)

    def test_headless_runs_the_loop_without_a_server(self, state: AppState) -> None:
        """The projector-only install, and the way to measure the loop without
        the web server's CPU share in the numbers."""
        from app.main import run_headless

        assert run_headless(state, max_frames=5, profile_path=None) == 0
        assert state.frames_processed == 5
        assert not state.is_running
