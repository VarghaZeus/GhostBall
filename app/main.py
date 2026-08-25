"""Application entry point: FastAPI app plus the vision loop.

Two things run concurrently:

* The **vision loop**, in a dedicated ``threading.Thread``.
* The **web server**, on uvicorn's asyncio event loop.

A thread rather than an asyncio task, deliberately. The loop is CPU-bound --
OpenCV calls that hold the GIL for milliseconds at a time -- and running it as a
coroutine would starve the event loop between awaits, making the control panel
feel frozen exactly when the system is busiest. OpenCV releases the GIL inside
its native calls, so a real thread genuinely overlaps with the server.

Built for a session measured in hours
-------------------------------------
The stated target is not "runs" but "runs stably for hours", and that changes
what the loop has to do about failure. Over two hours at 30 FPS this executes
216,000 frames; anything with a one-in-ten-thousand failure rate happens twenty
times. So nothing here is allowed to end the run:

* **Stages are contained individually** (:meth:`VisionLoop._run_stage`). An
  exception in detection costs that frame's detection, not the game. Repeated
  failures are counted, logged with backoff, and eventually switch the stage off
  -- the loop keeps running without it, which is the difference between a
  missing overlay and a dark table.
* **The camera is expected to come and go** (:meth:`VisionLoop._recover_camera`).
  USB cameras drop off and reappear; the loop reopens with backoff rather than
  exiting, and counts it so a marginal cable is diagnosable after the fact.
* **Falling behind is a state, not a failure** (:meth:`VisionLoop._adapt`).
  When the frame budget is consistently blown the loop sheds optional work
  instead of accumulating latency, and picks it back up when there is room.
* **A wedged loop is detectable** (:class:`~utils.performance.LoopWatchdog`).
  A loop blocked in a driver call looks identical to an idle one from outside,
  which is why the heartbeat is watched from another thread.

Usage::

    python -m app.main                       # camera + projector + web panel
    python -m app.main --mock                # synthetic camera, no projector
    python -m app.main --headless            # vision loop only, no web server
    python -m app.main --no-loop             # web panel only
    python -m app.main --frames 100          # bounded run, for smoke tests
    python -m app.main --profile run.csv     # per-frame timing trace
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import PACKAGE_ROOT, load_settings
from app.state import AppState
from projection.onboarding import should_draw_readiness
from utils.logging import ChangeLogger, setup_logging

logger = logging.getLogger(__name__)
#: Table detection runs on an interval forever, and a rig pointed at a ceiling
#: rejects a low-confidence quad on every pass. Reported on change only.
_changes = ChangeLogger(logger)

STATIC_DIR = PACKAGE_ROOT / "web" / "static"

#: Stages the loop cannot run without. Everything else can be switched off after
#: repeated failure; these two are the loop itself, so a persistent failure in
#: them stops the run rather than degrading it.
ESSENTIAL_STAGES = frozenset({"capture", "project"})

#: After the first failure in a stage, log every Nth one. A stage failing on
#: every frame at 30 FPS would otherwise write 1800 tracebacks a minute and make
#: the log useless for finding anything else.
STAGE_LOG_EVERY = 300

#: Backoff between camera reopen attempts, in seconds. Starts short because most
#: USB dropouts recover immediately, and caps rather than growing without bound
#: because a camera that comes back after ten minutes should be picked up.
CAMERA_RETRY_BACKOFF = (0.25, 0.5, 1.0, 2.0, 5.0)

#: How often to reconsider the degradation level, in frames. Two seconds at 30
#: FPS -- long enough that one slow frame does not trigger it, short enough to
#: react within a shot.
ADAPT_INTERVAL_FRAMES = 60

#: SoC temperature at which to start warning. A Pi 5 throttles around 80-85 C,
#: and a throttled Pi shows up as an unexplained FPS drop.
THERMAL_WARN_C = 75.0


class VisionLoop:
    """The capture -> detect -> simulate -> render -> project pipeline."""

    def __init__(
        self,
        state: AppState,
        max_frames: int | None = None,
        profile_path: Path | str | None = None,
    ) -> None:
        self.state = state
        self.max_frames = max_frames
        self.profile_path = Path(profile_path) if profile_path else None
        self._thread: threading.Thread | None = None
        self._reported: set[str] = set()
        #: Consecutive failures per stage, reset by a success. The *consecutive*
        #: count is what decides whether to switch a stage off: a stage that
        #: fails on one frame in a thousand is noisy, not broken, and disabling
        #: it would be an overreaction that costs the overlay for the rest of
        #: the session.
        self._consecutive: dict[str, int] = {}
        #: Reused canvas for projected test patterns. Owned by the loop thread.
        self._pattern_canvas = None
        #: Reused canvas for the readiness screens.
        self._readiness_canvas = None
        #: Whether table detection has run at all. Until it has, "no table" is
        #: not a finding, just an absence of one.
        self._table_checked = False
        #: Base re-detection interval in frames. The table does not move, but
        #: the camera can be bumped, so this is a slow re-check rather than a
        #: one-shot. Scaled up under load; see :meth:`_table_interval`.
        self._table_detect_interval = 150
        #: Whether a measured table size has already replaced the configured
        #: one. See :meth:`_maybe_adopt_table_size` for why this only happens
        #: once per run.
        self._table_size_adopted = False
        self._watchdog = None
        self._profiler = None

    # -- stage plumbing -----------------------------------------------------

    def _skip(self, stage: str, exc: BaseException) -> None:
        """Note that a stage is unimplemented, logging only the first occurrence."""
        if stage not in self._reported:
            self._reported.add(stage)
            logger.warning("stage '%s' unavailable, skipping: %s", stage, exc)

    def _run_stage(self, stage: str, fn, *args, **kwargs) -> tuple[bool, object]:
        """Run one pipeline stage: timed, contained and skippable.

        The single place failure policy lives, so every stage gets the same
        treatment and adding a stage cannot accidentally opt out of it.

        Three outcomes, and the distinction between the first two is the point:

        * ``NotImplementedError`` -- the stage does not exist yet. Logged once,
          then silent. Not an error and not counted as one.
        * Any other exception -- the stage is broken. Counted, logged with
          backoff, and after enough *consecutive* failures switched off for the
          rest of the run.
        * Success -- the consecutive counter resets.

        Returns:
            ``(ran, result)``. ``ran`` is ``False`` for a skipped, disabled or
            failed stage, so callers can tell "no result" from "result is None".
        """
        state = self.state
        if stage in state.disabled_stages:
            return False, None

        try:
            with state.tracker.stage(stage):
                result = fn(*args, **kwargs)
        except NotImplementedError as exc:
            self._skip(stage, exc)
            return False, None
        except Exception as exc:  # noqa: BLE001 - containment is the whole job
            self._note_failure(stage, exc)
            return False, None

        self._consecutive[stage] = 0
        return True, result

    def _note_failure(self, stage: str, exc: BaseException) -> None:
        """Count, log and possibly disable a failing stage."""
        state = self.state
        total = state.note_stage_error(stage, exc)
        run = self._consecutive[stage] = self._consecutive.get(stage, 0) + 1

        if run == 1 or run % STAGE_LOG_EVERY == 0:
            # exc_info on the first failure only: the traceback is what tells
            # you where the bug is, and the thousandth copy of it does not.
            logger.error(
                "stage '%s' failed (%d consecutive, %d total): %s",
                stage,
                run,
                total,
                exc,
                exc_info=(run == 1),
            )

        limit = state.settings.system.stage_failure_limit
        if run >= limit and stage not in ESSENTIAL_STAGES:
            state.disabled_stages.add(stage)
            logger.error(
                "disabling stage '%s' after %d consecutive failures; "
                "the loop continues without it",
                stage,
                run,
            )

    # -- stage bodies -------------------------------------------------------

    def _render_pattern(self, state: AppState):
        """Render the test pattern the control panel asked for.

        Returns ``None`` and drops the override on a bad pattern name, rather
        than raising. The name comes from the API, which validates it -- but a
        pattern that has been removed from the enum since the request was made
        would otherwise take the whole loop down, and a projector stuck on a
        black screen with the panel still claiming a pattern is up is a much
        worse failure than the pattern quietly not appearing.
        """
        from projection.draw import ensure_canvas
        from projection.patterns import TestPattern, render_test_pattern

        try:
            pattern = TestPattern(state.projection_override)
        except ValueError:
            logger.warning("dropping unknown projection override %r", state.projection_override)
            state.projection_override = None
            return None

        # One reused buffer. A static pattern still has to be re-sent every
        # frame -- an OpenCV full-screen window that is not being fed stops
        # repainting -- so this runs at 30 FPS like anything else.
        self._pattern_canvas = ensure_canvas(self._pattern_canvas, state.settings)
        return render_test_pattern(
            pattern, state.mapper, state.settings, canvas=self._pattern_canvas
        )

    def _end_wizard(self, wizard) -> None:
        """Tear down a finished wizard and hand the table back to play.

        The tracks and the shot machine are reset, not merely resumed: somebody
        has been leaning over the table for several minutes, and the positions
        the tracker still believes in are from before that.
        """
        state = self.state
        state.wizard = None
        state.wizard_lock.release()
        state.tracker_balls.reset()
        state.mode_manager.reset_state_machine()
        state.readiness.reset()
        self._table_checked = False
        logger.info(
            "wizard %s (%s)", "finished" if wizard.finished else "cancelled", wizard.flow.value
        )

    def _render_readiness(self, state: AppState):
        """Draw the welcome / no-table / no-camera screen."""
        from projection.onboarding import render_readiness_overlay

        self._readiness_canvas = render_readiness_overlay(
            state.readiness.current(), state.settings, canvas=self._readiness_canvas
        )
        return self._readiness_canvas

    def _detect_table(self, frame) -> None:
        """Refresh the cached table boundary, homography and pockets.

        Pockets are derived here rather than per frame because they are a
        function of the boundary and nothing else -- caching them with it saves
        re-deriving six positions thirty times a second for no new information.
        """
        from vision.calibration import (
            CalibrationError,
            compute_perspective_transform,
            detect_table_boundaries,
        )
        from vision.detection import detect_pocket_openings

        state = self.state
        boundary = detect_table_boundaries(frame, state.settings)
        if boundary is None:
            # Keep the previous boundary. One bad frame must not blank the
            # overlay, and a table that "disappears" because someone leaned over
            # it is the common case.
            logger.debug("table not detected this pass; keeping cached boundary")
            return

        # Confidence is checked *here*, before the boundary is cached, so that
        # everything downstream inherits one answer to "is there a table".
        #
        # It used to be checked only by the readiness tracker, which left three
        # different notions of found in the system at once: the loop cached any
        # boundary at all, the panel reported presence, and readiness applied a
        # threshold. A ceiling scored 41%, so readiness said no table while the
        # panel said found.
        #
        # It is also the expensive disagreement. With a boundary cached,
        # `extract_game_state` runs ball detection, cue detection and pocket
        # refinement inside it -- measured at 5x the cost of the no-table path.
        # A false table does not merely mislead, it triples the frame time.
        threshold = state.settings.vision.table_min_confidence
        if boundary.confidence < threshold:
            _changes.report(
                "low_confidence_table",
                round(boundary.confidence, 2),
                logging.INFO,
                "ignoring a table detected at %.0f%% confidence (need %.0f%%); "
                "if this is a real table, check the whole cloth is in view",
                boundary.confidence * 100,
                threshold * 100,
            )
            return
        _changes.recovered(
            "low_confidence_table", logging.INFO, "table detection is confident again"
        )

        self._maybe_adopt_table_size(boundary)

        try:
            c2t, t2c = compute_perspective_transform(boundary, state.settings)
        except CalibrationError as exc:
            # A degenerate quad passed the acceptance checks but will not solve.
            # Keep whatever was working before rather than losing calibration.
            logger.warning("table found but homography failed (%s); keeping previous", exc)
            return

        moved = state.table_boundary is not None and (
            boundary.center.distance_to(state.table_boundary.center) > 5.0
        )
        state.table_boundary = boundary
        state.camera_to_table = c2t
        state.table_to_camera = t2c
        state.pockets = detect_pocket_openings(frame, boundary, state.settings)

        if moved:
            # The camera was bumped or the table re-found somewhere else, so
            # existing tracks refer to stale positions. Keeping them would
            # produce enormous phantom velocities and fire a spurious shot.
            logger.info("table moved in frame; resetting ball tracks")
            state.tracker_balls.reset()

    def _maybe_adopt_table_size(self, boundary) -> None:
        """Report -- and optionally act on -- a measured table size. Once per run.

        Pocket detection measures the table in feet; physics, the renderer and
        the projector mapper all take their geometry from ``settings.table``. If
        those disagree -- an 8 ft table under a config that says 7 ft -- every
        prediction is scaled wrong by 15%.

        The default is to say so and change nothing, because the measurement is
        not accurate enough to be trusted with the decision: see
        ``vision.adopt_measured_table_size``. Turning that on makes this resize
        the table instead.

        **Once**, either way. Table detection runs on an interval, so acting on
        every pass would let the table change size mid-shot on a frame where
        somebody leaned over a pocket and the measurement wobbled. The first
        measured boundary wins; restart to re-measure.
        """
        from vision.pockets import adopt_measured_table_size, report_size_disagreement

        if self._table_size_adopted:
            return
        # Latch on the first *measured* boundary rather than the first boundary
        # of any kind, so a felt-detected frame early on does not use up the one
        # chance without having measured anything.
        if not boundary.is_measured:
            return
        self._table_size_adopted = True

        report_size_disagreement(self.state.settings, boundary)
        if adopt_measured_table_size(self.state.settings, boundary):
            # Everything derived from the table's dimensions is now stale.
            self.state.prediction_cache.clear()

    def _detect(self, frame):
        """Run the vision pipeline over one frame."""
        from vision.detection import extract_game_state

        state = self.state
        return extract_game_state(
            frame.image,
            frame.index,
            frame.timestamp,
            boundary=state.table_boundary,
            camera_to_table=state.camera_to_table,
            settings=state.settings,
            tracker=state.tracker_balls,
            pockets=state.pockets or None,
        )

    def _simulate(self, game_state):
        """Predict where the current aim sends the balls.

        Returns the prediction, or ``None`` when nobody is aiming -- in which
        case the caller drops the stale one rather than leaving the previous
        aiming line to be drawn over a live shot.
        """
        from physics.simulator import estimate_shot_from_cue

        state = self.state
        if game_state.cue_stick is None:
            return None
        return estimate_shot_from_cue(
            game_state,
            game_state.cue_stick,
            settings=state.settings,
            cache=state.prediction_cache,
        )

    def _project(self, overlay) -> None:
        """Push this frame to the projector, or clear it.

        The web layer cannot touch the display -- OpenCV windows belong to the
        thread that made them -- so a blank request is consumed here. Consumed
        even when there is an overlay to draw, or a stale request would blank a
        later frame.
        """
        state = self.state
        blank_now = state.blank_requested
        state.blank_requested = False

        if overlay is not None and not blank_now:
            state.display.send_frame(overlay)
            state.latest_overlay = overlay
        else:
            # Explicitly clear rather than leaving the last overlay frozen on
            # the felt.
            state.display.clear()
            state.latest_overlay = None

    # -- load shedding ------------------------------------------------------

    def _table_interval(self) -> int:
        """Frames between table re-detections.

        Two independent pressures, pulling opposite ways.

        Under load, detection is the most expensive stage that does not have to
        run every frame, so it is the first thing to give up.

        While *searching*, it should run far more often than while playing.
        Somebody is up a ladder aiming the camera and wants to know the moment
        it lands; detection costs ~100 ms on a Pi, which is a real stall, but
        paid every couple of seconds during setup it buys feedback exactly when
        it is worth having. Once a table is found it does not move, and the
        base interval is a slow guard against the camera being bumped.

        See :func:`app.readiness.table_detect_interval` for the backoff that
        keeps a permanently table-less rig from paying setup rates forever.
        """
        from app.readiness import table_detect_interval

        state = self.state
        base = self._table_detect_interval * (1 << state.degradation_level)
        return table_detect_interval(state.readiness.state, base, state.table_attempts)

    def _should_simulate(self, frame_index: int) -> bool:
        """Whether to recompute the shot prediction this frame.

        Two independent reasons to skip. Under degradation, every other frame --
        the prediction is smoothed across frames anyway, so halving its rate
        costs a little responsiveness and nothing else. And always, once the
        balls are moving: the renderer deliberately draws no aiming line during
        a shot, so simulating one is pure waste at the exact moment the frame
        budget is tightest.
        """
        from app.models import SessionState

        state = self.state
        if not state.auto_detect_cue:
            return False
        if state.mode_manager.session.state not in (SessionState.IDLE, SessionState.AIMING):
            return False
        if state.degradation_level >= 1 and frame_index % 2:
            return False
        return True

    def _adapt(self) -> None:
        """Raise or lower the degradation level from recent frame times.

        Hysteresis on purpose: shedding work at 1.5x budget and restoring it at
        1.1x leaves a gap so the loop cannot oscillate between levels every two
        seconds, which would be visible as the overlay changing character.

        p95 rather than the mean, because the mean is exactly the statistic that
        hides the problem -- a loop that hits budget on nine frames in ten and
        takes 200 ms on the tenth averages fine and looks terrible.
        """
        state = self.state
        if not state.settings.system.adaptive_degradation:
            return

        budget = state.tracker.frame_budget_ms
        p95 = state.tracker.percentile(95)
        if p95 <= 0:
            return

        level = state.degradation_level
        if p95 > budget * 1.5 and level < 2:
            state.degradation_level = level + 1
            logger.warning(
                "shedding load: degradation level %d -> %d (p95 %.1f ms vs %.1f ms budget)",
                level,
                level + 1,
                p95,
                budget,
            )
        elif p95 < budget * 1.1 and level > 0:
            state.degradation_level = level - 1
            logger.info(
                "restoring load: degradation level %d -> %d (p95 %.1f ms)", level, level - 1, p95
            )

    # -- camera recovery ----------------------------------------------------

    def _recover_camera(self, exc: BaseException) -> bool:
        """Close and reopen the camera after a loss. Returns whether it worked.

        A USB camera that browns out, or a Pi camera whose ISP is reconfigured,
        raises and then works again seconds later. Exiting the loop for that
        would mean a dark table and an SSH session for something that fixes
        itself, so this retries with backoff up to a configured time budget.

        The budget is a real limit rather than infinite patience: a camera that
        has been unplugged is not coming back on its own, and a loop spinning on
        reopen attempts forever burns a core and hides the problem behind a
        healthy-looking process.
        """
        state = self.state
        budget = state.settings.system.camera_recovery_seconds
        logger.error("camera lost (%s); attempting recovery for up to %.1fs", exc, budget)

        from vision.camera import Camera, CameraError

        deadline = time.perf_counter() + budget
        attempt = 0
        while time.perf_counter() < deadline and not state.stop_event.is_set():
            delay = CAMERA_RETRY_BACKOFF[min(attempt, len(CAMERA_RETRY_BACKOFF) - 1)]
            # Interruptible: waiting on the stop event rather than sleeping means
            # a shutdown during recovery is immediate instead of taking up to
            # five seconds per attempt.
            if state.stop_event.wait(delay):
                return False
            attempt += 1

            with contextlib.suppress(Exception):
                if state.camera is not None:
                    state.camera.close()
            try:
                state.camera = Camera(state.settings.camera).open()
            except (CameraError, RuntimeError, OSError) as retry_exc:
                logger.warning("camera reopen attempt %d failed: %s", attempt, retry_exc)
                continue

            state.camera_reconnects += 1
            # Tracks are anchored to positions from before the dropout, and the
            # table may well have been disturbed by whatever caused it. Starting
            # clean costs one shot's worth of history and avoids phantom
            # velocities the size of the table.
            state.tracker_balls.reset()
            # Everything the readiness tracker believed was about a camera that
            # is no longer attached.
            state.readiness.reset()
            self._table_checked = False
            logger.warning(
                "camera recovered on attempt %d via %s (reconnect #%d)",
                attempt,
                state.camera.backend_name,
                state.camera_reconnects,
            )
            return True

        logger.error("camera did not recover within %.0fs; stopping the vision loop", budget)
        return False

    # -- lifecycle ----------------------------------------------------------

    def _open_devices(self) -> bool:
        """Bring up the camera and projector. Returns whether the loop can run."""
        from projection.display import Display
        from vision.camera import Camera, CameraError

        state = self.state
        try:
            state.camera = Camera(state.settings.camera).open()
            state.display = Display(state.settings.projector).open()
        except (CameraError, RuntimeError) as exc:
            # Nothing to do without a capture source; the web panel stays up so
            # the failure is visible without SSH.
            logger.error("cannot start vision loop: %s", exc)
            return False
        return True

    def _log_startup(self) -> None:
        """One banner with everything needed to interpret the rest of the log.

        Written as a block rather than scattered through initialisation because
        the first question about any log from a two-hour session is "what was
        this actually running", and the answer should not require reading
        thirty lines of interleaved module output.
        """
        state = self.state
        settings = state.settings
        logger.info("--- GhostBall starting ---")
        logger.info(
            "camera:      %s %dx%d @ %d FPS%s",
            state.camera.backend_name,
            settings.camera.width,
            settings.camera.height,
            settings.camera.fps,
            "  [MOCK]" if state.camera.is_mock else "",
        )
        logger.info(
            "projector:   %s %dx%d%s",
            state.display.backend_name,
            settings.projector.width,
            settings.projector.height,
            "  [MOCK]" if state.display.is_mock else "",
        )
        focus = state.camera.focus
        if not settings.camera.focus_enabled:
            logger.info("focus:       disabled in config")
        elif state.camera.is_mock:
            logger.info("focus:       n/a (mock camera)")
        elif focus.ok:
            logger.info("focus:       %s (from %s)", focus.detail, focus.source)
        elif not focus.calibrated:
            # Not an error -- nothing is broken, the rig has simply never been
            # told where to focus. Worth a warning because the symptom (soft
            # picture, poor detection) reads as badly tuned thresholds.
            logger.warning("focus:       NOT CALIBRATED -- %s", focus.detail)
        else:
            # Calibrated and it did not take. That is a fault.
            logger.error("focus:       %s", focus.detail)
            logger.error("             Re-run: python -m tools.focus_sweep")

        logger.info(
            "table:       %s %.1f x %.1f in",
            settings.table_preset,
            settings.table.length_in,
            settings.table.width_in,
        )
        logger.info(
            "mode:        %s   theme: %s   target: %d FPS",
            state.mode_manager.session.mode.value,
            settings.render.theme,
            settings.system.target_fps,
        )

        if state.calibration_source == "file":
            logger.info(
                "calibration: loaded from file (RMSE %.2f px)",
                state.mapper.calibration.rmse_px,
            )
        else:
            # Loud, because the failure it causes does not look like a
            # calibration failure. An identity mapping projects a perfectly
            # plausible overlay onto the wrong part of the table, and the
            # natural reading of that is "the detection is wrong".
            logger.warning(
                "calibration: NONE -- projecting through an identity mapping. "
                "Overlays will not line up with the felt. "
                "Run the calibration wizard (python -m calibration_ui.calibration_app)."
            )
        if state.pending_stages:
            logger.info("pending:     %s", ", ".join(sorted(state.pending_stages)))

    def _start_watchdog(self):
        """Start the stall watchdog, wired to the health flags on ``AppState``."""
        from utils.performance import LoopWatchdog

        state = self.state

        def on_stall(idle_seconds: float) -> None:
            state.loop_stalled = True
            state.stall_count += 1
            state.last_error = f"loop stalled for {idle_seconds:.1f}s"

        def on_recover() -> None:
            state.loop_stalled = False

        watchdog = LoopWatchdog(
            state.tracker,
            stall_seconds=state.settings.system.watchdog_stall_seconds,
            on_stall=on_stall,
            on_recover=on_recover,
        )
        return watchdog.start()

    def _start_profiler(self):
        """Attach a per-frame CSV trace, if one was asked for."""
        if self.profile_path is None:
            return None
        from utils.performance import FrameProfiler

        return FrameProfiler(self.profile_path).attach(self.state.tracker)

    def _log_health(self) -> None:
        """Periodic host-health line: CPU, memory and SoC temperature.

        Separate from the perf summary because the two explain different
        things. The perf line says the loop is slow; this one says why -- a
        throttling SoC and a busy CPU produce identical frame times and want
        opposite responses.
        """
        from utils.performance import get_system_metrics

        state = self.state
        metrics = get_system_metrics()
        temp = metrics["temp_c"]
        cpu = metrics["cpu_pct"]
        mem = metrics["mem_pct"]

        hot = temp is not None and temp >= THERMAL_WARN_C
        logger.log(
            logging.WARNING if hot else logging.INFO,
            "health uptime=%.0fs frames=%d cpu=%s%% mem=%s%% temp=%s%s reconnects=%d degrade=%d",
            state.uptime_seconds(),
            state.frames_processed,
            "?" if cpu is None else f"{cpu:.0f}",
            "?" if mem is None else f"{mem:.0f}",
            "?" if temp is None else f"{temp:.1f}C",
            "  THERMAL THROTTLING LIKELY" if hot else "",
            state.camera_reconnects,
            state.degradation_level,
        )

    def _close_devices(self) -> None:
        """Release the projector and camera, leaving the felt clean."""
        state = self.state
        with contextlib.suppress(Exception):
            if state.display is not None:
                state.display.clear()
                state.display.close()
        with contextlib.suppress(Exception):
            if state.camera is not None:
                state.camera.close()

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        """Run until stopped. Blocks -- call via :meth:`start` for the threaded form."""
        state = self.state
        if not self._open_devices():
            return

        self._log_startup()
        self._watchdog = self._start_watchdog()
        self._profiler = self._start_profiler()
        state.is_running = True

        try:
            self._loop()
        except Exception:  # noqa: BLE001
            # Should be unreachable -- every stage is contained -- so if it
            # happens the traceback is the whole story and belongs in the log
            # rather than in a thread excepthook nobody reads.
            logger.exception("vision loop died unexpectedly")
        finally:
            state.is_running = False
            if self._watchdog is not None:
                self._watchdog.stop()
                self._watchdog = None
            self._close_devices()
            if self._profiler is not None:
                self._profiler.close()
                state.tracker.frame_sink = None
                self._profiler = None
            # From the shared state rather than a local, so the count is
            # still right on the unreachable path where _loop() raised.
            logger.info(
                "vision loop stopped after %d frames (%.0fs, %d camera reconnects)",
                state.frames_processed,
                state.uptime_seconds(),
                state.camera_reconnects,
            )
            state.tracker.log_summary(state.settings.system.latency_warn_ms)

    def _loop(self) -> int:
        """The frame loop proper. Returns the number of frames processed.

        Progress is also published to ``state.frames_processed`` per frame,
        which is what the panel and the shutdown log read -- a return value is
        no use to either while the loop is still running.
        """
        from utils.performance import RateLimiter
        from vision.camera import CameraError

        state = self.state
        tracker = state.tracker
        limiter = RateLimiter(state.settings.system.target_fps)
        frames_done = 0
        last_perf_log = time.perf_counter()
        last_health_log = last_perf_log

        while not state.stop_event.is_set():
            if self.max_frames is not None and frames_done >= self.max_frames:
                logger.info("reached --frames limit of %d", self.max_frames)
                break

            tracker.begin_frame()

            # 1. Capture -----------------------------------------------------
            # Not routed through _run_stage: a lost camera is recoverable in a
            # way a failed render is not, and the recovery has to happen here
            # rather than being counted and ignored.
            try:
                with tracker.stage("capture"):
                    frame = state.camera.capture_frame()
            except CameraError as exc:
                if not self._recover_camera(exc):
                    break
                limiter = RateLimiter(state.settings.system.target_fps)
                continue
            except Exception as exc:  # noqa: BLE001
                self._note_failure("capture", exc)
                if state.stop_event.wait(0.1):
                    break
                continue

            if frame is None:
                continue  # dropped grab; not worth a frame's accounting
            state.latest_frame = frame.image

            # 2. Table detection, on an interval ------------------------------
            if frame.index % self._table_interval() == 0:
                had_table = state.table_boundary is not None
                self._run_stage("table", self._detect_table, frame.image)
                self._table_checked = True
                if state.table_boundary is None or not had_table:
                    state.table_attempts = 0 if state.table_boundary else state.table_attempts + 1

            # 2b. Readiness ---------------------------------------------------
            # Every frame, not just on detection passes: the boundary is cached,
            # so the condition is observable continuously, and hysteresis
            # counted in detection passes would take minutes rather than
            # seconds to settle.
            state.readiness.observe(
                state.table_boundary,
                frame_arrived=True,
                last_frame_at=tracker.last_frame_at,
                table_checked=self._table_checked,
            )

            # 3. Detection ----------------------------------------------------
            ran, game_state = self._run_stage("detect", self._detect, frame)
            if ran:
                state.latest_game_state = game_state
            game_state = state.latest_game_state

            # 4. Physics ------------------------------------------------------
            if game_state is not None and self._should_simulate(frame.index):
                ran, prediction = self._run_stage("physics", self._simulate, game_state)
                if ran:
                    state.latest_prediction = prediction
                    if prediction is not None:
                        state.last_shot_confidence = prediction.confidence

            # 5. Mode logic and rendering -------------------------------------
            overlay = None
            wizard = state.wizard
            if wizard is not None and not (wizard.finished or wizard.cancelled):
                # The wizard owns the projector, and the mode stage is skipped
                # entirely rather than merely outranked -- so no mode logic runs,
                # no scoring can advance, and a cue waved over the table during
                # setup cannot fire a shot.
                ran, overlay = self._run_stage("wizard", wizard.update, frame)
                if not ran:
                    overlay = state.latest_overlay
            elif state.projection_override is not None:
                # A test pattern from the control panel outranks the mode.
                # Someone is standing behind the projector aligning it, and a
                # trajectory line appearing over the alignment grid would be
                # actively unhelpful.
                _, overlay = self._run_stage("render", self._render_pattern, state)
            elif should_draw_readiness(state.readiness.state):
                # Outranks the mode. There is no table, so a scoreboard would be
                # drawn over an empty room -- which looks like working software
                # and like broken detection at the same time, with no way to
                # tell from the felt which it is.
                _, overlay = self._run_stage("render", self._render_readiness, state)
            elif game_state is not None:
                ran, output = self._run_stage(
                    "mode", state.mode_manager.update, game_state, state.latest_prediction
                )
                if ran:
                    overlay = output.overlay
                else:
                    # Hold the last good overlay rather than going dark, on any
                    # failure and not just once the stage has been disabled.
                    # The frames between "started failing" and "failed enough
                    # times to disable" are a second long, and blanking the
                    # table for a second before freezing it is the worst of both
                    # -- a flicker, and then a freeze anyway.
                    #
                    # Stale for a frame is a scoreboard that did not update.
                    # Blank for a frame is the table going dark mid-shot, which
                    # players read as the system crashing.
                    overlay = state.latest_overlay

            if wizard is not None and (wizard.finished or wizard.cancelled):
                self._end_wizard(wizard)

            # 6. Projector output ---------------------------------------------
            self._run_stage("project", self._project, overlay)

            frame_ms = tracker.end_frame(capture_timestamp=frame.timestamp)
            frames_done += 1
            state.frames_processed = frames_done

            # 7. Periodic reporting and load shedding -------------------------
            now = time.perf_counter()
            if now - last_perf_log >= state.settings.system.perf_log_interval_seconds:
                tracker.log_summary(state.settings.system.latency_warn_ms)
                last_perf_log = now
            if now - last_health_log >= state.settings.system.health_log_interval_seconds:
                self._log_health()
                last_health_log = now
            if frames_done % ADAPT_INTERVAL_FRAMES == 0:
                self._adapt()

            overrun = limiter.sleep()
            if overrun > 0.005:
                logger.debug(
                    "frame %d overran budget by %.1f ms (processing %.1f ms)",
                    frame.index,
                    overrun * 1000.0,
                    frame_ms,
                )

        return frames_done

    def start(self) -> threading.Thread:
        """Run the loop on a daemon thread."""
        self._thread = threading.Thread(target=self.run, name="vision-loop", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the loop and wait for it to finish."""
        self.state.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                # Daemon thread, so the process still exits -- but a stuck loop
                # means a blocking call somewhere and is worth flagging.
                logger.warning("vision loop did not stop within %.1fs", timeout)
            else:
                self._thread = None


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app(
    state: AppState | None = None,
    start_loop: bool = True,
    max_frames: int | None = None,
    profile_path: Path | str | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    A factory rather than a module-level singleton so tests can construct an app
    with ``start_loop=False`` and no hardware.

    Args:
        state: Pre-built shared state. A fresh one is created if omitted.
        start_loop: Whether to run the vision loop alongside the server.
        max_frames: Stop the loop after this many frames. For smoke tests.
        profile_path: Write a per-frame timing trace here.
    """
    app_state = state or AppState()
    loop = VisionLoop(app_state, max_frames=max_frames, profile_path=profile_path)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        """Start the loop on boot, stop it on shutdown.

        The modern replacement for ``@app.on_event("startup")``, which the spec's
        pseudocode used and which is deprecated in current FastAPI.
        """
        if start_loop:
            loop.start()
        try:
            yield
        finally:
            # Runs on SIGINT/SIGTERM too: uvicorn traps both and unwinds the
            # lifespan, which is what makes Ctrl-C leave the projector black and
            # the camera released rather than relying on process teardown.
            loop.stop()

    app = FastAPI(
        title="GhostBall",
        description="Projection-mapped AR pool on a Raspberry Pi 5",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.app_state = app_state
    app.state.vision_loop = loop

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_state.settings.web.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from web.api import router

    app.include_router(router)

    @app.get("/health")
    async def health() -> JSONResponse:
        """Liveness probe, and the one endpoint safe to poll from a monitor.

        Distinguishes "server up" from "loop running", which matters because the
        server intentionally stays up when the loop cannot start -- that is how
        the failure becomes visible without SSH. ``ok`` folds in the watchdog,
        so a wedged-but-alive loop reports unhealthy rather than reporting the
        30 FPS it was managing before it wedged.
        """
        healthy = (not start_loop) or (app_state.is_running and not app_state.loop_stalled)
        return JSONResponse(
            {
                "status": "ok" if healthy else "degraded",
                "loop_running": app_state.is_running,
                "pending_stages": sorted(app_state.pending_stages),
                **app_state.health_summary(),
            },
            status_code=200 if healthy else 503,
        )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            """Serve the control panel."""
            return FileResponse(str(STATIC_DIR / "index.html"))
    else:
        logger.warning("no static directory at %s; control panel UI unavailable", STATIC_DIR)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """The application's command-line surface.

    Exposed separately from :func:`parse_args` so ``launcher.py`` can wrap it as
    an ``argparse`` parent and add its own flags, rather than defining a second
    copy of these options that would drift out of step with this one.

    Args:
        add_help: Set ``False`` when using this as a ``parents=`` entry, or
            argparse registers ``-h`` twice and raises.
    """
    parser = argparse.ArgumentParser(description="GhostBall", add_help=add_help)
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use the synthetic camera and discard projector output",
    )
    parser.add_argument(
        "--no-loop", action="store_true", help="serve the web panel without the vision loop"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the vision loop without the web server",
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="stop the loop after N frames"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        metavar="CSV",
        help="write a per-frame timing trace to this file",
    )
    parser.add_argument("--host", default=None, help="override the bind host")
    parser.add_argument("--port", type=int, default=None, help="override the bind port")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the application's own arguments."""
    return build_parser().parse_args(argv)


def _install_signal_handlers(state: AppState) -> None:
    """Make SIGINT and SIGTERM ask the loop to stop, rather than killing it.

    Only needed on the headless path. Under uvicorn the server traps both and
    unwinds the lifespan, which stops the loop; installing handlers there as
    well would fight it for the signal.

    SIGTERM is the one that matters in practice -- it is what systemd sends on
    ``stop`` and ``restart``, and without a handler the process dies between two
    frames with the projector still lit and the camera still claimed.
    """

    def handle(signum: int, _frame: object) -> None:
        logger.info("received %s; shutting down", signal.Signals(signum).name)
        state.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError, AttributeError):
            # ValueError when not on the main thread; AttributeError for
            # platforms without SIGTERM. Neither is worth failing to start over.
            signal.signal(sig, handle)


def run_headless(state: AppState, max_frames: int | None, profile_path: Path | None) -> int:
    """Run the vision loop in the foreground with no web server.

    For a projector-only install, and for measuring the loop without the
    server's CPU share in the numbers. The loop runs on *this* thread rather
    than a spawned one so that Ctrl-C lands where the work is.
    """
    _install_signal_handlers(state)
    loop = VisionLoop(state, max_frames=max_frames, profile_path=profile_path)
    loop.run()
    return 0


def run(args: argparse.Namespace) -> int:
    """Start the application from already-parsed arguments.

    Split out of :func:`main` so a wrapper that has already parsed a superset of
    these flags -- ``launcher.py`` -- can hand the namespace straight over
    instead of reconstructing an ``argv`` for this to parse a second time.
    Extra attributes on the namespace are ignored.
    """
    settings = load_settings(args.config)

    if args.log_level:
        settings.system.log_level = args.log_level
    setup_logging(settings.system.log_level, settings.system.log_to_file)

    if args.mock:
        # Forced together on purpose: a synthetic camera with a real projector
        # would paint predictions for a table that is not there.
        settings.camera.use_mock = True
        settings.projector.use_mock = True
        logger.info("mock mode: synthetic camera, projector output discarded")

    state = AppState(settings=settings)

    if args.headless:
        if args.no_loop:
            logger.error("--headless and --no-loop together would run nothing")
            return 2
        return run_headless(state, args.frames, args.profile)

    host = args.host or settings.web.host
    port = args.port or settings.web.port
    app = create_app(
        state,
        start_loop=not args.no_loop,
        max_frames=args.frames,
        profile_path=args.profile,
    )

    import uvicorn

    logger.info("control panel at http://%s:%d/", host, port)
    try:
        uvicorn.run(app, host=host, port=port, log_config=None)
    except KeyboardInterrupt:
        logger.info("interrupted")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console entry point."""
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
