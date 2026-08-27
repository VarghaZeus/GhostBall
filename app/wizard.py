"""The setup wizard, driven from the projector and the phone at once.

One state machine, two surfaces. The projector shows patterns and nothing else;
the phone carries every word of instruction, every number, and every button.
That split is not a convenience -- it is a measurement requirement. The camera
focus step scores the *variance of edges* inside the projected targets, so a
line of instructional text on the cloth would be measured along with them. The
felt has to stay pure pattern, which means the instructions have to live
somewhere else, which means the phone.

Frame-driven, not a blocking loop
---------------------------------
The old console wizard was ``while True: frame = capture()``, which cannot be
driven by a request handler. Here each step is an object the vision loop ticks
once per frame with :meth:`Wizard.update`, and the web layer only ever reads
:meth:`Wizard.view` or posts a :meth:`Wizard.act`. Long operations -- the focus
sweep -- run as phases inside a step, advancing a little per frame, so nothing
blocks and progress is visible on the phone while it happens.

Three flows, not one
--------------------
A bumped box should not mean a seven-step restart. Each flow re-measures one
thing and writes one artifact:

* ``FULL`` -- first install, or a move big enough that nothing can be trusted.
* ``FOCUS`` -- camera focus only. The box moved vertically or the lens shifted.
* ``TABLE`` -- table position and projector alignment. The box slid or turned.

What happens when the two are coupled -- a move large enough to invalidate both
-- is deliberately *not* decided here. It is measured, in
:mod:`app.calibration_status`, from the sharpness reference and the recorded
table corners, and reported to the user with a suggestion. Cascading a deletion
from one flow to the other would throw away good calibrations for a nudge.

Corners are found, not tapped
-----------------------------
Corner mapping projects four markers and locates them in the camera image,
rather than asking someone to tap them on a preview. Two reasons: tapping a
1080p preview accurately on a phone is miserable, and the machinery to find a
bright blob under any focus already exists for the focus targets. The
correspondence is then camera px -> table inches (via the table homography)
paired with the projector px we know exactly because we drew them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np

from app.calibration_status import WizardFlow
from app.models import Vec2

logger = logging.getLogger(__name__)

__all__ = ["Wizard", "WizardFlow", "StepId", "FLOW_STEPS"]


class StepId(str, Enum):
    WELCOME = "welcome"
    DETECT_TABLE = "detect_table"
    PROJECTOR_WARMUP = "projector_warmup"
    CAMERA_FOCUS = "camera_focus"
    CORNER_MAPPING = "corner_mapping"
    VERIFY = "verify"
    DONE = "done"


#: Which steps each flow runs. The lists are the entire difference between the
#: flows -- every step is written to work in any of them, so a flow is a
#: subsequence rather than a mode.
FLOW_STEPS: dict[WizardFlow, list[StepId]] = {
    WizardFlow.FULL: [
        StepId.WELCOME,
        StepId.DETECT_TABLE,
        StepId.PROJECTOR_WARMUP,
        StepId.CAMERA_FOCUS,
        StepId.CORNER_MAPPING,
        StepId.VERIFY,
        StepId.DONE,
    ],
    # Focus needs the projector on and its own focus ring set, because the
    # camera cannot resolve detail the projector never drew -- hence the warm-up
    # step is in this flow too, short as it is.
    WizardFlow.FOCUS: [StepId.PROJECTOR_WARMUP, StepId.CAMERA_FOCUS, StepId.DONE],
    WizardFlow.TABLE: [
        StepId.DETECT_TABLE,
        StepId.CORNER_MAPPING,
        StepId.VERIFY,
        StepId.DONE,
    ],
}

#: Table positions, in inches, where the alignment markers are projected. Inset
#: from the corners for the same reasons the focus targets are: the projection
#: is dimmest and most keystoned at the edges, and an overhead camera sees the
#: rail occluding cloth near the cushion.
MARKER_INSET_IN = 8.0


@dataclass
class StepResult:
    """What a step has established so far, for the phone to display."""

    rows: list[tuple[str, str, str]] = field(default_factory=list)  # label, value, severity
    message: str = ""
    severity: str = "info"
    busy: bool = False
    progress: float | None = None
    can_advance: bool = False


class Wizard:
    """The wizard's whole state. Lives on ``AppState``; ticked by the loop."""

    def __init__(self, state, flow: WizardFlow = WizardFlow.FULL) -> None:
        self.state = state
        self.flow = flow
        self.steps = list(FLOW_STEPS[flow])
        self.index = 0
        #: Bumped on every change so a polling client can tell it is behind
        #: without diffing the payload.
        self.version = 1
        self.started_at = time.perf_counter()
        self.finished = False
        self.cancelled = False
        self.result_message = ""

        self._canvas: np.ndarray | None = None
        self._handlers = {
            StepId.WELCOME: _Welcome(),
            StepId.DETECT_TABLE: _DetectTable(),
            StepId.PROJECTOR_WARMUP: _ProjectorWarmup(),
            StepId.CAMERA_FOCUS: _CameraFocus(),
            StepId.CORNER_MAPPING: _CornerMapping(),
            StepId.VERIFY: _Verify(),
            StepId.DONE: _Done(),
        }
        self._results: dict[StepId, StepResult] = {}
        self._enter()

    # -- navigation ---------------------------------------------------------

    @property
    def step(self) -> StepId:
        return self.steps[min(self.index, len(self.steps) - 1)]

    @property
    def handler(self):
        return self._handlers[self.step]

    def _enter(self) -> None:
        self._results[self.step] = StepResult()
        self.handler.enter(self)
        self.version += 1
        logger.info("wizard %s: step %s", self.flow.value, self.step.value)

    def advance(self, delta: int = 1) -> None:
        target = max(0, min(len(self.steps) - 1, self.index + delta))
        if target == self.index:
            return
        self.handler.leave(self)
        self.index = target
        self._enter()

    # -- per-frame ----------------------------------------------------------

    def update(self, frame) -> np.ndarray | None:
        """One tick from the vision loop. Returns the overlay to project.

        Every step's work happens here, on the loop thread, because that is the
        thread that owns the camera and the display. The web layer never does
        anything but read state and record intent.
        """
        if self.finished or self.cancelled:
            return None
        try:
            return self.handler.update(self, frame)
        except Exception as exc:  # noqa: BLE001 - a broken step must not end the run
            logger.exception("wizard step %s failed", self.step.value)
            self.result(message=f"This step hit an error: {exc}", severity="error")
            return None

    # -- actions ------------------------------------------------------------

    def act(self, action: str, from_step: str | None = None) -> tuple[bool, str]:
        """Apply an action from either surface. Returns ``(ok, message)``.

        ``from_step`` is the step the client believed it was on. A mismatch is
        rejected rather than applied, which is what stops two phones tapping
        Next on the same step from advancing twice and skipping one. Cheap,
        stateless, and it also covers a phone acting on a screen somebody else
        already moved past.
        """
        if from_step is not None and from_step != self.step.value:
            return False, (
                f"That was sent from the '{from_step}' step and this is now "
                f"'{self.step.value}'. The screen has been refreshed."
            )

        if action == "cancel":
            self.cancelled = True
            self.version += 1
            return True, "Setup cancelled."
        if action == "back":
            self.advance(-1)
            return True, ""
        if action == "next":
            if not self.result().can_advance:
                return False, "This step is not finished yet."
            self.advance(1)
            return True, ""

        ok, message = self.handler.act(self, action)
        self.version += 1
        return ok, message

    # -- results ------------------------------------------------------------

    def result(self, **updates) -> StepResult:
        """Read, or update, the current step's result block."""
        current = self._results.setdefault(self.step, StepResult())
        if updates:
            for key, value in updates.items():
                setattr(current, key, value)
            self.version += 1
        return current

    # -- serialisation ------------------------------------------------------

    def view(self) -> dict[str, object]:
        """Everything the phone renders. The only thing the API exposes."""
        handler = self.handler
        result = self.result()
        return {
            "active": not (self.finished or self.cancelled),
            "flow": self.flow.value,
            "version": self.version,
            "step": self.step.value,
            "step_number": self.index + 1,
            "step_count": len(self.steps),
            "title": handler.title,
            "instruction": handler.instruction,
            "rows": [{"label": r[0], "value": r[1], "severity": r[2]} for r in result.rows],
            "message": result.message or self.result_message,
            "severity": result.severity,
            "busy": result.busy,
            "progress": result.progress,
            "actions": handler.actions(self),
            "finished": self.finished,
            "cancelled": self.cancelled,
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 1),
        }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


class _Step:
    """Base class. Every hook is optional."""

    title = ""
    instruction = ""

    def enter(self, wizard: Wizard) -> None:
        wizard.result(can_advance=True)

    def leave(self, wizard: Wizard) -> None:
        return None

    def update(self, wizard: Wizard, frame):
        return None

    def act(self, wizard: Wizard, action: str) -> tuple[bool, str]:
        return False, f"'{action}' does nothing on this step."

    def actions(self, wizard: Wizard) -> list[dict]:
        return [
            {"id": "back", "label": "Back", "enabled": wizard.index > 0, "primary": False},
            {
                "id": "next",
                "label": "Next",
                "enabled": wizard.result().can_advance,
                "primary": True,
            },
            {"id": "cancel", "label": "Cancel", "enabled": True, "primary": False},
        ]


class _Welcome(_Step):
    title = "Set up GhostBall"
    instruction = (
        "This takes a few minutes. You will need to be at the table and at the "
        "projector. Everything is shown on this phone -- the projector only shows "
        "patterns."
    )

    def update(self, wizard, frame):
        from app.models import SystemState
        from app.readiness import Readiness
        from projection.onboarding import render_readiness_overlay

        wizard._canvas = render_readiness_overlay(
            Readiness(state=SystemState.STARTING, headline="GhostBall", detail="Setup in progress"),
            wizard.state.settings,
            canvas=wizard._canvas,
        )
        return wizard._canvas


class _DetectTable(_Step):
    title = "Find the table"
    instruction = (
        "Clear the balls off the cloth and stand back so nothing is over the table. "
        "This finds the cushions on its own."
    )

    def enter(self, wizard):
        wizard.result(can_advance=False, busy=True, message="Looking for the table...")

    def update(self, wizard, frame):
        from vision.calibration import detect_table_boundaries

        state = wizard.state
        boundary = detect_table_boundaries(frame.image, state.settings)
        threshold = state.settings.vision.table_min_confidence

        if boundary is not None and boundary.confidence >= threshold:
            state.table_boundary = boundary
            self._solve(wizard, boundary)
            wizard.result(
                busy=False,
                can_advance=True,
                severity="ok",
                message="Table found.",
                rows=[
                    ("Confidence", f"{boundary.confidence * 100:.0f}%", "ok"),
                    ("Size in frame", f"{boundary.width_px:.0f} x {boundary.height_px:.0f} px", "info"),
                    ("Method", boundary.detection_method or "felt", "info"),
                ],
            )
        else:
            found = boundary.confidence if boundary else 0.0
            wizard.result(
                busy=True,
                can_advance=False,
                severity="warn",
                message=(
                    "No table yet. Check the whole cloth is in view and nothing is "
                    "leaning over it."
                ),
                rows=[("Best so far", f"{found * 100:.0f}% (need {threshold * 100:.0f}%)", "warn")],
            )
        return None  # nothing projected: a pattern would confuse felt detection

    def _solve(self, wizard, boundary) -> None:
        from vision.calibration import CalibrationError, compute_perspective_transform

        state = wizard.state
        try:
            c2t, t2c = compute_perspective_transform(boundary, state.settings)
        except CalibrationError as exc:
            wizard.result(severity="warn", message=f"Table found but not solvable: {exc}")
            return
        state.camera_to_table, state.table_to_camera = c2t, t2c


class _ProjectorWarmup(_Step):
    title = "Aim and focus the projector"
    # The one instruction about the projector's own optics. There is no app
    # control for it and there should not be: the projector has a remote, it is
    # a ten-second job by eye, and a software control for a hardware focus ring
    # would be a worse version of a thing that already works.
    instruction = (
        "Two things at the projector, both by hand:\n"
        "1. Move it until the white rectangle covers the whole cloth.\n"
        "2. Focus it with its own remote until the edges are crisp.\n\n"
        "Focus matters more than it looks: the next step measures how sharply the "
        "camera sees a projected pattern, and it cannot resolve detail the "
        "projector never drew."
    )

    def update(self, wizard, frame):
        from projection.patterns import TestPattern, render_test_pattern

        wizard._canvas = render_test_pattern(
            TestPattern.FULL_TABLE,
            wizard.state.mapper,
            wizard.state.settings,
            canvas=_cleared(wizard),
        )
        return wizard._canvas


class _CameraFocus(_Step):
    title = "Focus the camera"
    instruction = (
        "Five checkerboards are being projected onto the cloth. Tap Start, then "
        "stand clear -- nothing should touch the table or the mount while it runs."
    )

    def enter(self, wizard):
        wizard.result(can_advance=False, message="Ready when you are.")
        self.phase = "idle"
        self.positions: list[int] = []
        self.at = 0
        #: Frames to throw away after moving the lens, then frames to measure.
        #: At 15-30 FPS this is roughly a third of a second per position, which
        #: is both enough for the motor to settle and enough samples for the
        #: median to mean something.
        self.discard_frames = 3
        self.sample_frames = 5
        self.samples: list[dict[str, float]] = []
        self.settled = 0
        #: (written, read) per position, so a disagreement can be diagnosed
        #: rather than just counted. See vision.focus_calibration.
        self.readbacks = []
        self.regions: list = []
        self.outcome = None
        self.lens = None
        self.focus_range = None
        self.exposure = None
        self._lock = None

    def leave(self, wizard):
        self._release(wizard)

    def act(self, wizard, action):
        if action == "start":
            self.phase = "locating"
            wizard.result(busy=True, message="Looking for the targets...", progress=0.0)
            return True, ""
        if action == "retry":
            self.enter(wizard)
            return True, "Ready to try again."
        return super().act(wizard, action)

    def actions(self, wizard):
        base = super().actions(wizard)
        if self.phase == "idle":
            base.insert(0, {"id": "start", "label": "Start", "enabled": True, "primary": True})
        elif self.phase == "done":
            base.insert(0, {"id": "retry", "label": "Run again", "enabled": True, "primary": False})
        return base

    def update(self, wizard, frame):
        from projection.patterns import TestPattern, render_test_pattern

        canvas = render_test_pattern(
            TestPattern.FOCUS_TARGETS, wizard.state.mapper, wizard.state.settings,
            canvas=_cleared(wizard),
        )
        wizard._canvas = canvas

        if self.phase == "idle":
            return canvas
        if self.phase == "locating":
            self._locate(wizard, frame)
        elif self.phase == "sweeping":
            self._sweep_one(wizard, frame)
        return canvas

    # -- phases ---------------------------------------------------------

    def _locate(self, wizard, frame):
        from vision.focus import FocusError, find_lens_subdev, query_focus_range
        from vision.focus_calibration import coarse_step, detect_targets, focus_positions

        settings = wizard.state.settings
        self.lens = find_lens_subdev(settings.camera.lens_driver)
        if self.lens is None:
            self._fail(wizard, "No focus motor found, so the lens cannot be driven.")
            return
        try:
            self.focus_range = query_focus_range(self.lens.path)
        except FocusError as exc:
            self._fail(wizard, str(exc))
            return

        self.regions = detect_targets(frame.image)
        if len(self.regions) < 5:
            # Answered in seconds rather than as a flat curve after two minutes.
            self._fail(
                wizard,
                f"Only {len(self.regions)} of 5 targets are visible. Is the projector "
                "on and aimed at the cloth, with the room dim enough to see the "
                "pattern clearly?",
            )
            return

        # Derived from the range the driver reported, not a fixed stride. A
        # hardcoded 128 was 33 stops on the ak7375's 0-4095 and is 9 on a
        # dw9807's 0-1023 -- the same number meaning a coarse pass on one lens
        # and a useless one on the next.
        self.positions = focus_positions(self.focus_range, coarse_step(self.focus_range))
        self.at = 0
        self.curves = {r.name: {} for r in self.regions}
        self._lock = wizard.state.camera.exposure_lock()
        self.exposure = self._lock.__enter__()
        self.phase = "sweeping"
        per_position = self.discard_frames + self.sample_frames
        seconds = len(self.positions) * per_position / max(
            1, wizard.state.settings.system.target_fps
        )
        wizard.result(
            busy=True,
            message=f"Sweeping {len(self.positions)} focus positions, about "
            f"{max(1, round(seconds))}s...",
            progress=0.0,
            rows=[("Targets", f"{len(self.regions)} found", "ok"),
                  ("Exposure", "locked" if self.exposure.locked else "NOT locked",
                   "ok" if self.exposure.locked else "warn")],
        )

    def _sweep_one(self, wizard, frame):
        """Advance the sweep by one frame.

        Each focus position gets several frames, not one, and the first few are
        thrown away. Both halves matter:

        * **Discarding.** A frame handed back immediately after a control change
          was already in the ISP pipeline before it, so it shows the *previous*
          lens position. Measuring it shifts the whole curve by one stop, which
          is a systematic error that looks like a plausible answer.
        * **Several samples, combined by median.** Sensor noise moves the
          variance of the Laplacian by a few percent frame to frame, and near
          the peak neighbouring positions differ by less than that -- so a
          single sample picks the winner by noise. The median is used rather
          than the mean because one frame caught mid-anything is a large
          outlier, and a mean would let it move the peak.

        The CLI sweep has always done this. The wizard measured one frame per
        position, which made it faster and less accurate than the tool it was
        meant to replace.

        Spread across loop ticks rather than blocking: the phone keeps updating
        and the panel keeps being served while it runs.
        """
        from vision.focus import approach_focus, read_focus
        from vision.focus_calibration import ReadbackSample, measure_regions

        if self.at >= len(self.positions):
            self._finish_sweep(wizard)
            return

        position = self.positions[self.at]
        if not self.samples and self.settled == 0:
            approach_focus(self.lens.path, position, self.focus_range)

        if self.settled < self.discard_frames:
            self.settled += 1
            return

        self.samples.append(measure_regions(frame.image, self.regions))
        if len(self.samples) < self.sample_frames:
            return

        for region in self.regions:
            values = [sample[region.name] for sample in self.samples]
            self.curves[region.name][position] = float(np.median(values))
        self.readbacks.append(
            ReadbackSample(written=position, read=read_focus(self.lens.path))
        )

        self.samples = []
        self.settled = 0
        self.at += 1
        wizard.result(
            progress=self.at / len(self.positions),
            message=f"Sweeping... {self.at} of {len(self.positions)}",
        )

    def _finish_sweep(self, wizard):
        from vision.focus_calibration import SweepOutcome, analyse

        self._release(wizard)
        outcome = SweepOutcome(
            curves=self.curves, regions=self.regions, readbacks=list(self.readbacks)
        )
        drift = wizard.state.camera.exposure_drifted(self.exposure) if self.exposure else None
        if drift:
            outcome.diagnosis = None  # analyse() will not overwrite a set diagnosis
            self._fail(wizard, f"Exposure moved during the sweep ({drift}); result discarded.")
            return

        outcome = analyse(outcome, self.focus_range)
        self.outcome = outcome
        self.phase = "done"

        rows = [
            (peak.name, f"{peak.peak_focus} ({peak.prominence:.1f}x)", "info")
            for peak in outcome.peaks
        ]
        if not outcome.ok:
            wizard.result(
                busy=False, can_advance=False, severity="error", progress=None,
                rows=rows, message=outcome.diagnosis.message if outcome.diagnosis else "No result.",
            )
            return

        self._save(wizard, outcome)
        severity = "warn" if outcome.diagnosis else "ok"
        message = (
            outcome.diagnosis.message
            if outcome.diagnosis
            else f"Focus set to {outcome.best_focus} and saved."
        )
        rows.append(("Spread across targets", f"{outcome.tilt_spread} counts",
                     "warn" if outcome.diagnosis else "ok"))
        wizard.result(
            busy=False, can_advance=True, severity=severity, progress=None,
            rows=rows, message=message,
        )

    def _save(self, wizard, outcome):
        from vision.focus import FocusCalibration, apply_focus, save_focus_calibration
        from vision.focus_calibration import bare_reference

        state = wizard.state
        apply_focus(outcome.best_focus, device=self.lens.path)
        # Bare cloth, targets off, at the chosen focus. The only reference the
        # runtime health check can compare against -- the peak above was taken
        # on projected checkerboards and is an order of magnitude larger.
        state.display.clear()
        bare = bare_reference(state.camera, self.regions, settings=state.settings)
        save_focus_calibration(
            FocusCalibration(
                focus_absolute=outcome.best_focus,
                peak_sharpness=outcome.best_sharpness,
                bare_table_sharpness=bare,
                per_target=tuple(outcome.peaks),
                tilt_spread=outcome.tilt_spread,
                tilt_note=outcome.tilt_note,
                camera_resolution=f"{state.settings.camera.width}x{state.settings.camera.height}",
                lens_name=self.lens.name,
                created_at=_now(),
            )
        )

    def _release(self, wizard):
        if self._lock is not None:
            try:
                self._lock.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.exception("releasing the exposure lock failed")
            self._lock = None

    def _fail(self, wizard, message):
        self._release(wizard)
        self.phase = "done"
        wizard.result(busy=False, can_advance=False, severity="error", progress=None,
                      message=message)


class _CornerMapping(_Step):
    title = "Line the projection up with the table"
    instruction = (
        "Four markers are being projected near the corners of the cloth. Stand "
        "clear and tap Align -- the camera finds them itself, so there is nothing "
        "to tap on a preview."
    )

    def enter(self, wizard):
        wizard.result(can_advance=False, message="Ready when you are.")
        self.phase = "idle"
        self.calibration = None

    def act(self, wizard, action):
        if action == "align":
            self.phase = "aligning"
            wizard.result(busy=True, message="Finding the markers...")
            return True, ""
        if action == "retry":
            self.enter(wizard)
            return True, ""
        return super().act(wizard, action)

    def actions(self, wizard):
        base = super().actions(wizard)
        if self.phase == "idle":
            base.insert(0, {"id": "align", "label": "Align", "enabled": True, "primary": True})
        elif self.phase == "done":
            base.insert(0, {"id": "retry", "label": "Try again", "enabled": True, "primary": False})
        return base

    def update(self, wizard, frame):
        wizard._canvas = self._render_markers(wizard)
        if self.phase == "aligning":
            self._align(wizard, frame)
        return wizard._canvas

    def _marker_points(self, wizard) -> list[Vec2]:
        settings = wizard.state.settings
        length, width = settings.table.length_in, settings.table.width_in
        inset = MARKER_INSET_IN
        return [
            Vec2(inset, inset),
            Vec2(length - inset, inset),
            Vec2(length - inset, width - inset),
            Vec2(inset, width - inset),
        ]

    def _render_markers(self, wizard):
        from projection import draw

        canvas = _cleared(wizard)
        mapper = wizard.state.mapper
        for point in self._marker_points(wizard):
            centre = mapper.table_to_projector(point)
            # A filled disc, not a cross or a ring. It has to survive being
            # found as a bright blob under whatever focus the camera is at, and
            # a disc is the shape that degrades most gracefully to one.
            draw.draw_circle(canvas, centre, 26, (255, 255, 255), alpha=255, filled=True)
        return canvas

    def _align(self, wizard, frame):
        from projection.mapper import ProjectionMapper, solve_projector_homography
        from vision.calibration import CalibrationError, camera_to_table_coords
        from vision.focus_calibration import detect_targets

        state = wizard.state
        if state.camera_to_table is None:
            self._fail(wizard, "The table has not been found, so markers cannot be placed.")
            return

        found = detect_targets(frame.image, expected=4)
        if len(found) < 4:
            self._fail(
                wizard,
                f"Only {len(found)} of 4 markers were visible. Check nothing is over the "
                "table and the projection covers the whole cloth.",
            )
            return

        # Order the detections the same way the projected points are ordered --
        # clockwise from top-left in the camera's view.
        ordered = _clockwise(found)
        table_points, projector_points = [], []
        for region, projected in zip(ordered, self._marker_points(wizard), strict=True):
            try:
                table_pos = camera_to_table_coords(
                    Vec2(*region.center_px), state.camera_to_table
                )
            except CalibrationError:
                self._fail(wizard, "A marker fell outside the table plane.")
                return
            table_points.append(table_pos)
            projector_points.append(state.mapper.table_to_projector(projected))

        calibration = solve_projector_homography(
            table_points,
            projector_points,
            state.settings.projector.width,
            state.settings.projector.height,
        )
        calibration.table_corners_px = [
            [float(c.x), float(c.y)] for c in state.table_boundary.corners()
        ] if state.table_boundary else None
        # The crop those corners were measured under. They are frame-space
        # coordinates, so without recording the origin they were relative to, a
        # later re-framing is indistinguishable from the box being knocked.
        calibration.camera_crop = state.current_crop()
        calibration.created_at = _now()

        self.calibration = calibration
        state.mapper = ProjectionMapper(calibration)
        state.mode_manager.mapper = state.mapper
        self.phase = "done"

        good = calibration.rmse_px <= 20
        wizard.result(
            busy=False,
            can_advance=good,
            severity="ok" if good else "warn",
            message=(
                f"Aligned to {calibration.rmse_px:.1f} px."
                if good
                else f"Alignment is poor ({calibration.rmse_px:.1f} px). Check the "
                "projection covers the whole cloth squarely, then try again."
            ),
            rows=[
                ("Error", f"{calibration.rmse_px:.1f} px", "ok" if good else "warn"),
                ("Markers", f"{len(found)} of 4", "ok"),
            ],
        )

    def _fail(self, wizard, message):
        self.phase = "done"
        wizard.result(busy=False, can_advance=False, severity="error", message=message)


class _Verify(_Step):
    title = "Check it lines up"
    instruction = (
        "The outline being projected should sit exactly on the cushions. If it "
        "does, you are done. If it is off, go back and align again."
    )

    def update(self, wizard, frame):
        from projection.patterns import TestPattern, render_test_pattern

        wizard._canvas = render_test_pattern(
            TestPattern.FULL_TABLE, wizard.state.mapper, wizard.state.settings,
            canvas=_cleared(wizard),
        )
        return wizard._canvas


class _Done(_Step):
    title = "Finished"
    instruction = "Everything has been saved. The table is ready to play."

    def enter(self, wizard):
        wizard.result(can_advance=False, severity="ok")
        self._save(wizard)

    def _save(self, wizard) -> None:
        from projection.mapper import save_calibration

        mapping = wizard._handlers[StepId.CORNER_MAPPING]
        calibration = getattr(mapping, "calibration", None)
        if calibration is not None:
            save_calibration(calibration)
            wizard.state.calibration_source = "file"
        wizard.finished = True
        wizard.result_message = "Saved."
        wizard.version += 1

    def actions(self, wizard):
        return [{"id": "cancel", "label": "Close", "enabled": True, "primary": True}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleared(wizard):
    from projection.draw import ensure_canvas

    wizard._canvas = ensure_canvas(wizard._canvas, wizard.state.settings)
    return wizard._canvas


def _clockwise(regions):
    """Order detected blobs clockwise from top-left, as the markers are drawn."""
    xs = [r.center_px[0] for r in regions]
    ys = [r.center_px[1] for r in regions]
    mid_x, mid_y = sum(xs) / len(xs), sum(ys) / len(ys)
    top = sorted([r for r in regions if r.center_px[1] < mid_y], key=lambda r: r.center_px[0])
    bottom = sorted(
        [r for r in regions if r.center_px[1] >= mid_y], key=lambda r: r.center_px[0], reverse=True
    )
    ordered = top + bottom
    if len(ordered) != len(regions):  # degenerate split; fall back to angle order
        import math

        ordered = sorted(
            regions, key=lambda r: math.atan2(r.center_px[1] - mid_y, r.center_px[0] - mid_x)
        )
    return ordered


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
