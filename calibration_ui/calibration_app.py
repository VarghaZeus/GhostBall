"""Full-screen calibration wizard.

Phase 6. Seven screens that take a projector pointed roughly at a pool table
and end with a transform accurate enough to draw an aiming line along.

Why this is a separate application rather than a mode: it needs to drive the
projector directly, without the game's calibration applied, and it needs to
block the main pipeline while the user works. Running it as a mode would mean
the vision loop is simultaneously trying to use the transform being replaced.

The seven screens come from ``ar_pool_calibration_ui_prompt.md``:

1. Welcome and hardware check
2. Auto-detect table
3. Projector warm-up
4. Corner mapping (the critical step)
5. Fine-tune alignment
6. Test projection
7. Complete

The design constraint the spec is emphatic about, and which drives every
decision here: the user is not a programmer, is standing at a pool table in a
dim room, and should be done in well under ten minutes. Every message is a
physical instruction ("twist the projector two degrees clockwise"), never a
number. The numbers are still shown -- underneath, smaller, for whoever is
tuning the rig.

Two surfaces, and the difference is the point
---------------------------------------------
The **projector** shows marks on the felt (:mod:`projection.renderer`). The
**console** shows the camera's view of those marks with annotations
(:mod:`calibration_ui.overlay_renderer`, driven by
:class:`calibration_ui.console.Console`). Calibration is the act of making the
two agree, so the wizard has to be able to show them separately.

How a corner actually gets recorded
-----------------------------------
A correspondence is ``(where a mark is in camera px, which projector pixel
produced it)``. Three ways to establish one, all writing the same pair:

``Auto``
    Blank the projector, capture; project all four marks, capture; difference
    the two frames and take the four centroids. One button, four corners, no
    human judgement. This is the default and it is what the user should use.
``Click``
    The user taps the mark on the console's camera view. The fallback for when
    detection picks up a reflection, or the room is bright enough that the
    difference image is marginal.
``Nudge and record``
    Arrow keys walk the active mark across the felt until it sits on the
    cushion nose, then Record pairs it with the *detected* table corner. The
    only mode that needs no camera sight of the projected light at all, so it
    is the last resort that always works.

The spec describes only the third. The first two are additions: the third takes
around forty keypresses to place four corners and the first takes one button,
and a wizard measured against a ten-minute budget should not spend it on
keypresses. All three are offered because each fails in a different place.

Where the accuracy actually comes from
--------------------------------------
Two things, and neither is obvious:

**Ask for the cushion nose.** Table coordinates are defined at the inside of
the cushion, so a user who lines marks up with the pocket jaw or the rail edge
introduces a constant offset that fine-tuning will chase forever and never fix.
The on-screen text names the cushion nose on every screen that asks.

**Do not trust the corner RMSE.** Four correspondences give an exact fit, so the
reported reprojection error is structurally near zero whether or not the
transform is right anywhere else on the table. That is why screen 6 exists: it
measures the composed chain against a ball in the middle of the felt, which is
the only number here that can actually be wrong.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import BALL_RADIUS_IN, Settings, get_settings
from app.models import (
    AlignmentError,
    Ball,
    CalibrationState,
    ProjectorCalibration,
    Severity,
    TableBoundary,
    Vec2,
)
from calibration_ui import overlay_renderer as ui
from calibration_ui.console import Button, Console
from calibration_ui.metrics import (
    CORNER_LABELS,
    CORNER_NAMES,
    GridMetrics,
    ProjectionCheck,
    assign_marks_to_corners,
    compute_alignment_error,
    compute_grid_metrics,
    locate_projected_marks,
    projection_error_in,
    solve_projector_to_camera,
)

logger = logging.getLogger(__name__)

TOTAL_STEPS = 7

#: Alignment RMSE thresholds in projector px. 20 px is the spec's stated target
#: and so the pass mark; 5 px is about the practical floor given detection noise.
RMSE_EXCELLENT = 5.0
RMSE_ACCEPTABLE = 20.0

#: End-to-end error thresholds in inches, for the test-projection screen. Half
#: an inch is a quarter of a ball and invisible in play; an inch and a half is
#: most of a ball and the aiming line visibly misses.
TRAJECTORY_EXCELLENT_IN = 0.5
TRAJECTORY_ACCEPTABLE_IN = 1.5

#: Table detection confidence bands, from the spec's screen 2.
CONFIDENCE_GOOD = 0.85
CONFIDENCE_MARGINAL = 0.70

#: Seconds between projector repaints for animated content. The projector is
#: repainted on a change rather than every console frame, because
#: ``Display.show`` runs its own ``waitKey`` and each one is a chance to eat a
#: keystroke meant for the console. Eight hertz is enough for a pulse to read as
#: a pulse.
_PROJECTION_INTERVAL_S = 0.12

#: Seconds between automatic re-measurements of the projected marks while the
#: corner-mapping screen is in auto mode. Each measurement blanks the projector
#: for a few frames, which is a visible blink -- so this is a compromise between
#: feedback that tracks the user's hands and a projection that looks broken.
_REMEASURE_INTERVAL_S = 1.2

#: Frames discarded after changing what is projected, before capturing. The
#: camera pipeline is several frames deep, so capturing immediately returns an
#: image of the *previous* projection -- which would make the difference image
#: pure noise and the marks unfindable.
_SETTLE_FRAMES = 3

#: One press of an arrow key, as a fraction of the projector frame. Fine enough
#: to land inside the 20 px target, coarse enough to cross the table without
#: wearing out a keyboard.
_TARGET_NUDGE_FRACTION = 0.004
_PROJECTION_NUDGE_FRACTION = 0.005
_SCALE_NUDGE = 0.004
_ROTATION_NUDGE_DEG = 0.25

#: Radius of the test screen's per-ball ring, in ball radii. Big enough to clear
#: the ball and land on cloth, small enough that two adjacent balls' rings do not
#: merge into one blob and defeat the matching.
_TEST_RING_RADII = 2.2


@dataclass(slots=True)
class _Measurements:
    """Everything the wizard has measured, gathered for the final report.

    Separate from :class:`~app.models.CalibrationState`, which is the shared
    domain object and carries only what other layers need. These are the
    wizard's own working notes: which of them exist depends on which screens the
    user actually ran, so every one is optional and the report says so when one
    is missing rather than inventing a zero.
    """

    grid: GridMetrics | None = None
    trajectory_error_in: float | None = None
    alignment: AlignmentError | None = None
    #: Where each projected corner mark was last seen, camera px.
    observed_marks: dict[str, Vec2] = field(default_factory=dict)
    #: The test screen's ball-to-ring matching, or ``None`` if it never ran.
    projection_check: ProjectionCheck | None = None


class CalibrationApp:
    """Drives the seven-screen wizard."""

    def __init__(self, settings: Settings | None = None, console: Console | None = None) -> None:
        self.settings = settings or get_settings()
        self.state = CalibrationState()
        #: corner name -> (camera_px, projector_px), the correspondences the
        #: homography is solved from.
        self.correspondences: dict[str, tuple[Vec2, Vec2]] = {}
        self.camera: object | None = None
        self.display: object | None = None
        #: Injected for tests and headless runs; a real one is built in
        #: :meth:`_open_hardware` otherwise.
        self.console = console

        self.measurements = _Measurements()
        self.camera_to_table: np.ndarray | None = None
        self.table_to_camera: np.ndarray | None = None
        self.projector_to_camera: np.ndarray | None = None
        self.calibration: ProjectorCalibration | None = None
        self.mapper: object | None = None  # projection.mapper.ProjectionMapper

        #: Per-corner adjustments to where each target is projected, in
        #: projector px, from the arrow keys on the corner-mapping screen.
        self._target_offsets: dict[str, Vec2] = {}
        #: Whole-projection fine-tune, applied to every correspondence before
        #: the homography is solved. See :meth:`_apply_nudge`.
        self._nudge_offset = Vec2(0.0, 0.0)
        self._nudge_scale = 1.0
        self._nudge_rotation_deg = 0.0

        self._last_frame: np.ndarray | None = None
        self._camera_lost = False
        self._frame_index = 0
        self._last_projection_at = 0.0
        self._projection_key: object = None

    # -- flow control -------------------------------------------------------

    def run(self) -> ProjectorCalibration | None:
        """Run the wizard to completion.

        Returns:
            The solved calibration, or ``None`` if the user cancelled. Callers
            must handle ``None`` by keeping the previous calibration -- an
            abandoned wizard should never leave the table uncalibrated.
        """
        screens = {
            1: self.screen_welcome,
            2: self.screen_auto_detect_table,
            3: self.screen_projector_warmup,
            4: self.screen_corner_mapping,
            5: self.screen_fine_tune,
            6: self.screen_test_projection,
            7: self.screen_complete,
        }

        try:
            self._open_hardware()
        except Exception as exc:  # noqa: BLE001 - any hardware failure is fatal here
            # No camera and no projector means there is nothing to calibrate and
            # no way to tell the user so on screen. Say it on the console the
            # operator launched this from.
            logger.error("cannot start the calibration wizard: %s", exc)
            self._close_hardware()
            return None

        try:
            while 1 <= self.state.step <= TOTAL_STEPS:
                screen = screens[self.state.step]
                logger.info(
                    "calibration step %d/%d: %s", self.state.step, TOTAL_STEPS, screen.__name__
                )
                advance = screen()
                if advance is None:
                    logger.info("calibration cancelled by user at step %d", self.state.step)
                    return None
                # Screens return the step delta, so a screen can send the user
                # back (-1) when a check fails -- e.g. corner mapping rejecting a
                # bad table detection sends them to step 2 to redo it.
                self.state.step += advance
            self.state.is_complete = True
            return self._solve()
        except KeyboardInterrupt:
            logger.info("calibration interrupted at step %d", self.state.step)
            return None
        finally:
            self._close_hardware()

    def _open_hardware(self) -> None:
        """Open the camera, a full-screen projector window and the console."""
        from projection.display import Display
        from vision.camera import Camera

        self.camera = Camera(self.settings.camera).open()
        self.display = Display(self.settings.projector).open()
        if self.console is None:
            self.console = Console(self.settings)
        self.console.open()

    def _close_hardware(self) -> None:
        """Release everything, blanking the projection first.

        Order matters: blank before closing, or the last overlay stays frozen on
        the felt until something else claims the output.
        """
        if self.console is not None:
            self.console.close()
        if self.display is not None:
            self.display.clear()
            self.display.close()
            self.display = None
        if self.camera is not None:
            self.camera.close()
            self.camera = None

    def _solve(self) -> ProjectorCalibration | None:
        """Solve, persist and report the calibration.

        Called once, after screen 7 has been passed. Saving here rather than in
        the screen keeps the rule that a cancelled wizard writes nothing.
        """
        from projection.mapper import save_calibration

        calibration = self._build_calibration()
        if calibration is None:
            logger.error(
                "cannot solve a calibration from %d correspondence(s) and %s table homography",
                len(self.correspondences),
                "a" if self.camera_to_table is not None else "no",
            )
            return None

        save_calibration(calibration)
        self._write_report(calibration)
        self.calibration = calibration
        return calibration

    def _write_report(self, calibration: ProjectorCalibration) -> None:
        """Write the YAML report next to the JSON. Never fatal.

        A report is a convenience; the JSON the application loads has already
        been written by the time this runs. Failing the whole wizard because a
        secondary file could not be written would mean re-doing a calibration
        that in fact succeeded.
        """
        from calibration_ui.report import write_calibration_report

        try:
            paths = write_calibration_report(
                calibration,
                self.state.table_boundary,
                self.camera_to_table,
                self.measurements.grid,
                self.measurements.trajectory_error_in,
                self.settings,
            )
            logger.info("calibration report: %s", ", ".join(str(p) for p in paths.values()))
        except OSError as exc:
            logger.warning("could not write the calibration report: %s", exc)

    def _build_calibration(self) -> ProjectorCalibration | None:
        """Solve the table -> projector transform without persisting it.

        Split out from :meth:`_solve` because screens 5, 6 and 7 all need the
        transform in order to project through it and measure it, and none of
        them should be writing to disk -- the user has not finished yet and may
        still cancel.
        """
        from projection.mapper import solve_projector_homography
        from vision.calibration import camera_to_table_coords

        if len(self.correspondences) < 4:
            logger.debug("only %d correspondences recorded; need 4", len(self.correspondences))
            return None
        if self.camera_to_table is None:
            logger.debug("no table homography; cannot map camera points into table space")
            return None

        table_points = [
            camera_to_table_coords(camera_px, self.camera_to_table)
            for camera_px, _projector_px in self.correspondences.values()
        ]
        projector_points = [
            self._apply_nudge(projector_px)
            for _camera_px, projector_px in self.correspondences.values()
        ]

        try:
            calibration = solve_projector_homography(
                table_points,
                projector_points,
                self.settings.projector.width,
                self.settings.projector.height,
            )
        except ValueError as exc:
            logger.error("calibration solve failed: %s", exc)
            return None

        # The affine fields are the human-legible summary of the fine-tune
        # nudges, not the transform -- ``ProjectionMapper`` uses the homography
        # whenever there is one. Recording them here is what lets the report and
        # the web panel say "the projection was shifted 40 px right" without
        # decomposing a 3x3.
        calibration.offset_x = self._nudge_offset.x
        calibration.offset_y = self._nudge_offset.y
        calibration.scale_x = self._nudge_scale
        calibration.scale_y = self._nudge_scale
        calibration.rotation_deg = self._nudge_rotation_deg
        return calibration

    def _apply_nudge(self, point: Vec2) -> Vec2:
        """Apply the fine-tune adjustment to one projector-space point.

        Scale and rotate about the frame centre, then translate -- the same
        order as :meth:`projection.mapper.ProjectionMapper._affine_matrix`, and
        for the same reason: rotating before scaling would make a nudge shear
        the projection, which is not what an arrow key should do.

        Applied to the recorded correspondences before solving, rather than
        composed onto the finished matrix. That is what keeps the result a
        genuine homography: the wizard's fine-tune preserves the keystone
        correction instead of discarding it, which is the one thing the affine
        nudge path in ``ProjectionMapper.nudge`` cannot do.
        """
        if (
            self._nudge_offset == Vec2(0.0, 0.0)
            and self._nudge_scale == 1.0
            and self._nudge_rotation_deg == 0.0
        ):
            return point

        center_x = self.settings.projector.width / 2.0
        center_y = self.settings.projector.height / 2.0
        dx, dy = point.x - center_x, point.y - center_y
        theta = math.radians(self._nudge_rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return Vec2(
            center_x + (dx * cos_t - dy * sin_t) * self._nudge_scale + self._nudge_offset.x,
            center_y + (dx * sin_t + dy * cos_t) * self._nudge_scale + self._nudge_offset.y,
        )

    def assess_alignment(self, rmse_px: float) -> AlignmentError:
        """Turn an RMSE into advice a non-technical user can act on.

        The wizard's whole voice: the number is meaningless to the user, and the
        severity is what decides whether they are allowed to finish.
        """
        if rmse_px <= RMSE_EXCELLENT:
            return AlignmentError(
                total_rmse=rmse_px,
                message="Alignment looks excellent.",
                severity="info",
            )
        if rmse_px <= RMSE_ACCEPTABLE:
            return AlignmentError(
                total_rmse=rmse_px,
                message="Alignment is good enough to play.",
                severity="info",
            )
        return AlignmentError(
            total_rmse=rmse_px,
            message=(
                "The projection does not line up with the table. "
                "Check that the projector is square to the table and try again."
            ),
            severity="error",
        )

    # -- camera and projector plumbing --------------------------------------

    def _capture(self) -> np.ndarray:
        """Grab a frame, falling back to the last good one.

        Never returns ``None``. A screen with no image to draw on has no way to
        tell the user what went wrong, so a lost camera yields a black frame and
        the screens report the loss on top of it. Sustained failure is recorded
        in ``_camera_lost`` and surfaced in the pre-flight checklist.
        """
        from vision.camera import CameraError

        if self.camera is not None:
            try:
                frame = self.camera.capture_frame()
            except CameraError as exc:
                if not self._camera_lost:
                    logger.error("camera lost during calibration: %s", exc)
                self._camera_lost = True
                frame = None
            if frame is not None:
                self._camera_lost = False
                self._last_frame = frame.image
                self._frame_index = frame.index

        if self._last_frame is None:
            self._last_frame = np.zeros(
                (self.settings.camera.height, self.settings.camera.width, 3), dtype=np.uint8
            )
        return self._last_frame

    def _capture_settled(self) -> np.ndarray:
        """Capture after the projector's change has reached the sensor.

        See :data:`_SETTLE_FRAMES`: the pipeline is several frames deep, so the
        first frame after a repaint still shows the old projection.
        """
        for _ in range(_SETTLE_FRAMES):
            self._capture()
        return self._capture()

    def _send(self, overlay_rgba: np.ndarray) -> None:
        """Push one RGBA overlay to the projector."""
        if self.display is not None:
            self.display.send_frame(overlay_rgba)

    def _project_step(
        self,
        *,
        alignment: AlignmentError | None = None,
        targets: list[tuple[Vec2, str]] | None = None,
        force: bool = False,
    ) -> None:
        """Repaint the projector for the current step, on a throttle.

        Throttled because every repaint runs a ``waitKey`` inside
        ``Display.show``, and each of those is a chance to consume a key the
        console was waiting for. ``force`` bypasses it for the repaints that
        must not be dropped -- a step change, or the frame that has to be on the
        felt before the next capture.
        """
        from projection.renderer import render_calibration_overlay

        now = time.perf_counter()
        if not force and now - self._last_projection_at < _PROJECTION_INTERVAL_S:
            return
        self._last_projection_at = now

        overlay = render_calibration_overlay(
            self.state,
            self.mapper,
            alignment,
            self.settings,
            targets=targets,
            now=now,
        )
        self._send(overlay)

    def _project_blank(self) -> None:
        """Show black. Physically, this projects nothing onto the felt."""
        if self.display is not None:
            self.display.clear()

    def _project_white(self) -> None:
        """Fill the output with white, for the warm-up screen's fit test."""
        from projection import draw

        canvas = draw.new_canvas(self.settings)
        canvas[:, :, :] = 255
        self._send(canvas)

    # -- console plumbing ---------------------------------------------------

    def _show(self, view: np.ndarray, buttons: list[Button]) -> str | None:
        """Display one console frame and return whatever the user did."""
        assert self.console is not None
        self.console.show(view, buttons)
        return self.console.poll()

    def _confirm_cancel(self) -> bool:
        """Ask before throwing away recorded work.

        Only asked once anything has been recorded. On screen 1 there is nothing
        to lose and a confirmation is just an extra tap between the user and the
        exit.
        """
        if not self.correspondences:
            return True
        assert self.console is not None
        buttons = [
            Button("Keep going", "no", "n", primary=True),
            Button("Quit without saving", "yes", "y"),
        ]
        while True:
            view = ui.draw_banner_message(
                self._capture(),
                f"Quit now and the {len(self.correspondences)} corners you placed are lost.",
                "warning",
            )
            action = self._show(view, buttons)
            if action in ("yes", "cancel"):
                # A second Escape confirms rather than dismissing. Dismissing on
                # Escape leaves the only key that means "get me out of here"
                # unable to, and turns an exhausted headless script into an
                # infinite loop -- both screens keep asking, nothing ever
                # answers.
                return True
            if action == "no":
                return False

    # -- corner bookkeeping -------------------------------------------------

    def _target_points(self) -> list[tuple[Vec2, str]]:
        """Where the four corner targets are currently projected.

        The base layout comes from :func:`projection.renderer.calibration_target_points`
        so the app and the renderer cannot disagree about it; the per-corner
        arrow-key offsets are added on top.
        """
        from projection.renderer import calibration_target_points

        points = []
        for (base, label), name in zip(calibration_target_points(self.settings), CORNER_NAMES, strict=True):
            offset = self._target_offsets.get(name, Vec2(0.0, 0.0))
            points.append((base + offset, label))
        return points

    def _target_point(self, name: str) -> Vec2:
        """The current projector pixel of one corner's target."""
        index = CORNER_NAMES.index(name)
        return self._target_points()[index][0]

    def _active_corner(self) -> str | None:
        """The next corner to place, or ``None`` when all four are recorded."""
        return next((name for name in CORNER_NAMES if name not in self.correspondences), None)

    def _record_corner(self, name: str, camera_px: Vec2) -> None:
        """Pair a camera observation with the projector pixel that produced it."""
        self.correspondences[name] = (camera_px, self._target_point(name))
        logger.info(
            "recorded corner %s: camera (%.1f, %.1f) <- projector (%.1f, %.1f)",
            name,
            camera_px.x,
            camera_px.y,
            self._target_point(name).x,
            self._target_point(name).y,
        )
        self._refresh_alignment()

    def _refresh_alignment(self) -> None:
        """Recompute the live error and the per-corner errors the renderer uses.

        ``state.corner_errors`` is load-bearing beyond display:
        ``render_calibration_overlay`` picks the pulsing target by its length, so
        it has to hold exactly one entry per recorded corner, in
        :data:`CORNER_NAMES` order.
        """
        from vision.calibration import pixels_per_inch

        boundary = self.state.table_boundary
        observed = {name: camera_px for name, (camera_px, _p) in self.correspondences.items()}
        self.measurements.observed_marks = observed

        if boundary is None:
            self.state.corner_errors = [0.0 for name in CORNER_NAMES if name in observed]
            return

        detected = dict(zip(CORNER_NAMES, boundary.corners(), strict=True))
        self.state.corner_errors = [
            detected[name].distance_to(observed[name]) for name in CORNER_NAMES if name in observed
        ]
        self.measurements.alignment = compute_alignment_error(
            detected, observed, pixels_per_inch(boundary, self.settings)
        )

    def _measure_marks(self) -> dict[str, Vec2]:
        """Project the four targets, find them in the camera, and record them.

        The blink is deliberate and unavoidable: the difference image needs a
        frame of the table with the projector dark, and there is no way to get
        one without going dark. Doing it this way rather than looking for bright
        pixels is what makes the measurement survive a lit room -- see
        :func:`calibration_ui.metrics.locate_projected_marks`.
        """
        boundary = self.state.table_boundary
        if boundary is None:
            logger.warning("cannot measure projected marks before the table is found")
            return {}

        targets = self._target_points()
        self._project_blank()
        dark = self._capture_settled()
        self._project_step(targets=targets, force=True)
        lit = self._capture_settled()

        marks = locate_projected_marks(dark, lit)
        found = assign_marks_to_corners(marks, boundary)
        for name, camera_px in found.items():
            self.correspondences[name] = (camera_px, self._target_point(name))
        self._refresh_alignment()

        logger.info("auto-measured %d of 4 corners", len(found))
        return found

    # -- screens ------------------------------------------------------------
    #
    # Each returns the step delta to apply: +1 to continue, a negative number to
    # send the user back, or None to cancel the wizard.

    def screen_welcome(self) -> int | None:
        """Screen 1: confirm the camera and projector are working.

        A live camera thumbnail and a projected message, side by side -- if
        either is missing the user finds out here rather than three screens in,
        which is the difference between "the camera is not plugged in" and
        "calibration does not work".

        Most of the checklist is unticked on purpose. The wizard cannot verify
        that a camera is pointed straight down or that the lighting is even, and
        a tick it did not earn would be worse than no tick: the user would stop
        looking at the one thing that is actually wrong.
        """
        self._project_step(force=True)
        buttons = [
            Button("I've checked everything", "next", "n", primary=True),
            Button("Retry hardware", "retry", "r"),
            Button("Cancel", "cancel", "q"),
        ]

        while True:
            frame = self._capture()
            has_frame = not self._camera_lost and self._last_frame is not None
            items: list[tuple[str, bool | None]] = [
                ("Camera mounted overhead, looking straight down", None),
                ("Camera is delivering frames", has_frame and not self._camera_lost),
                ("Camera is real hardware, not the simulator", self._camera_is_real()),
                ("Projector output is live", self._display_is_real()),
                ("Projector aimed at the table (you can adjust it later)", None),
                ("Balls on the table -- optional, but it helps detection", None),
                ("Even lighting, no hard shadows across the cloth", None),
            ]

            view = ui.render_step_instructions(
                frame,
                self.state,
                "Check the setup below, then continue.",
                self.settings,
            )
            view = ui.draw_checklist(view, "PRE-FLIGHT CHECKLIST", items, copy=False)
            if not self._camera_is_real() or not self._display_is_real():
                view = ui.draw_notice(
                    view,
                    "Running on simulated hardware -- this calibration will not be usable.",
                    "warning",
                    copy=False,
                )

            # Continue stays available even on simulated hardware: a developer
            # walking the wizard without a table is a real use, and the warning
            # above already says the result is worthless.
            action = self._show(view, buttons)
            if action == "next":
                return 1
            if action == "retry":
                self._retry_hardware()
            elif action == "cancel":
                return None

    def _camera_is_real(self) -> bool:
        return self.camera is not None and not getattr(self.camera, "is_mock", True)

    def _display_is_real(self) -> bool:
        return self.display is not None and not getattr(self.display, "is_mock", True)

    def _retry_hardware(self) -> None:
        """Close and reopen the camera and projector.

        For the common case of the user plugging in the thing they forgot. Any
        failure leaves the wizard on the welcome screen with the checklist
        showing what is still missing, which is exactly where they want to be.
        """
        logger.info("retrying hardware at the user's request")
        from projection.display import Display
        from vision.camera import Camera

        try:
            if self.camera is not None:
                self.camera.close()
            if self.display is not None:
                self.display.close()
            self.camera = Camera(self.settings.camera).open()
            self.display = Display(self.settings.projector).open()
            self._camera_lost = False
            self._project_step(force=True)
        except Exception as exc:  # noqa: BLE001 - report and stay on this screen
            logger.error("hardware retry failed: %s", exc)

    def screen_auto_detect_table(self) -> int | None:
        """Screen 2: find the table and let the user confirm or correct it.

        Auto-detection succeeds most of the time; the important part is the
        manual fallback, because the failure cases (unusual felt colour, a
        strong shadow across a rail) are exactly the ones a user cannot fix by
        retrying.

        Correction is two taps rather than a drag: tap a corner handle to pick
        it up, tap where it belongs to put it down. A drag needs press, move and
        release events tracked across frames, and on a touchscreen it competes
        with the scroll gesture -- whereas two taps work identically with a
        mouse, a finger and a scripted test.
        """
        from vision.calibration import (
            CalibrationError,
            compute_perspective_transform,
            detect_table_boundaries,
        )

        self._project_step(force=True)
        boundary: TableBoundary | None = self.state.table_boundary
        auto = boundary is None
        selected: str | None = None
        problem: str | None = None

        while True:
            frame = self._capture()
            if auto:
                found = detect_table_boundaries(frame, self.settings)
                if found is not None:
                    boundary = found

            confidence = boundary.confidence if boundary is not None else 0.0
            instruction, severity = _detection_advice(confidence, selected)

            view = ui.render_step_instructions(frame, self.state, instruction, self.settings)
            if boundary is not None:
                view = ui.draw_table_outline(view, boundary, show_handles=not auto, copy=False)
                view = ui.draw_alignment_grid(view, boundary, settings=self.settings, copy=False)
                if selected is not None:
                    view = ui.draw_corner_target(
                        view,
                        _corner_of(boundary, selected),
                        f"MOVING {CORNER_LABELS[selected]}",
                        is_active=True,
                        copy=False,
                    )
            view = ui.draw_confidence_bar(
                view,
                "Table detection confidence",
                confidence,
                (CONFIDENCE_MARGINAL, CONFIDENCE_GOOD),
                copy=False,
            )
            if problem is not None:
                view = ui.draw_banner_message(view, problem, "error", copy=False)
            elif confidence < CONFIDENCE_MARGINAL:
                view = ui.draw_checklist(
                    view,
                    "CANNOT SEE THE TABLE CLEARLY -- TRY:",
                    [
                        ("Reduce shadows: move a lamp, or turn the room light up", None),
                        ("Clean the camera lens", None),
                        ("Check the whole table is in frame", None),
                        ("Re-tune the felt colour: python -m tools.camera_preview --mask", None),
                    ],
                    copy=False,
                )

            buttons = [
                Button("Continue", "next", "n", enabled=boundary is not None, primary=True),
                Button("Re-detect", "retry", "r"),
                Button("Manual Adjust" if auto else "Auto Corners", "manual", "m"),
                Button("Back", "back", "b"),
                Button("Cancel", "cancel", "q"),
            ]
            action = self._show(view, buttons)
            problem = None

            if action == "cancel":
                if self._confirm_cancel():
                    return None
                continue
            if action == "back":
                return -1
            if action == "retry":
                auto, selected, boundary = True, None, None
                continue
            if action == "manual":
                auto = not auto
                selected = None
                logger.info("table corners: %s", "auto-detecting" if auto else "manual adjust")
                continue
            if action == "click" and not auto and boundary is not None:
                point = self.console.take_click() if self.console else None
                if point is not None:
                    boundary, selected = _apply_corner_tap(boundary, selected, point, frame.shape)
                continue
            if action == "next" and boundary is not None:
                try:
                    self.camera_to_table, self.table_to_camera = compute_perspective_transform(
                        boundary, self.settings
                    )
                except CalibrationError as exc:
                    # The corners do not form a usable quad. Almost always two
                    # handles dropped on top of each other, so say that rather
                    # than "the homography is singular".
                    logger.error("table homography rejected: %s", exc)
                    problem = "Those corners do not form a table. Re-detect, or move them apart."
                    continue
                self.state.table_boundary = boundary
                self._refresh_alignment()
                logger.info(
                    "table accepted at %.0f%% confidence, %.0fx%.0f px",
                    confidence * 100,
                    boundary.width_px,
                    boundary.height_px,
                )
                return 1

    def screen_projector_warmup(self) -> int | None:
        """Screen 3: fit the projection to the table while the lamp settles.

        Two jobs at once, which is why the screen does not feel like waiting.
        The countdown is not busywork -- projector colour temperature drifts for
        the first minute or two, and a transform solved before it settles is
        visibly wrong once the lamp stabilises. Meanwhile the white rectangle
        gives the user the one thing they should be doing with that time: making
        the projection cover the whole table.

        Skipping is allowed. Two minutes is a long time to be told to wait by a
        machine, and a user who skips gets a warning rather than a refusal.
        """
        total = float(self.settings.projector.warmup_seconds)
        started = time.perf_counter()
        self._project_white()
        skipped = False

        while True:
            frame = self._capture()
            remaining = max(0.0, total - (time.perf_counter() - started))
            ready = remaining <= 0.0 or skipped

            view = ui.render_step_instructions(
                frame,
                self.state,
                "Adjust the projector so the white rectangle covers the whole table.",
                self.settings,
            )
            if ready:
                view = ui.draw_banner_message(
                    view,
                    "Lamp is warm. Continue when the rectangle covers the cloth.",
                    "info",
                    copy=False,
                )
            else:
                view = ui.draw_countdown(
                    view, remaining, total, "Projector lamp warming up -- keep adjusting",
                    copy=False,
                )

            buttons = [
                Button("Ready", "next", "n", enabled=ready, primary=ready),
                Button("Skip warm-up", "skip", "s", enabled=not ready, primary=not ready),
                Button("Back", "back", "b"),
                Button("Cancel", "cancel", "q"),
            ]
            action = self._show(view, buttons)

            if action == "cancel":
                if self._confirm_cancel():
                    return None
            elif action == "back":
                return -1
            elif action == "skip":
                logger.warning(
                    "warm-up skipped with %.0fs remaining; colour and geometry may drift "
                    "and the calibration may need redoing once the lamp settles",
                    remaining,
                )
                skipped = True
            elif action == "next" and ready:
                return 1

    def screen_corner_mapping(self) -> int | None:
        """Screen 4: the critical step -- map projector pixels to table corners.

        Everything downstream is only as good as this. See the module docstring
        for the three ways a corner can be recorded and why all three exist.

        The projected target is a fine crosshair, not a filled dot: a dot's
        centre is ambiguous by several pixels once it is a few inches across on
        cloth, and that ambiguity lands directly in the alignment RMSE.

        Auto mode re-measures on a timer rather than on demand, so the error
        readout tracks the projector while the user has both hands on it. That
        is what makes this screen a feedback loop instead of a form.
        """
        if self.state.table_boundary is None:
            logger.warning("reached corner mapping with no table; sending the user back to step 2")
            return -2

        auto = True
        last_measure = 0.0
        self._project_step(targets=self._target_points(), force=True)

        while True:
            now = time.perf_counter()
            if auto and now - last_measure >= _REMEASURE_INTERVAL_S:
                self._measure_marks()
                last_measure = time.perf_counter()

            frame = self._capture()
            active = self._active_corner()
            alignment = self.measurements.alignment
            recorded = len(self.correspondences)

            view = ui.render_step_instructions(
                frame, self.state, _mapping_instruction(auto, active, recorded), self.settings
            )
            view = ui.draw_table_outline(view, self.state.table_boundary, show_handles=False, copy=False)
            view = self._draw_corner_states(view, active)
            if recorded:
                seen = [name for name in CORNER_NAMES if name in self.measurements.observed_marks]
                view = ui.draw_projected_vs_detected(
                    view,
                    [self.measurements.observed_marks[name] for name in seen],
                    [_corner_of(self.state.table_boundary, name) for name in seen],
                    copy=False,
                )
            if alignment is not None:
                view = ui.draw_alignment_feedback(view, alignment, self.settings, copy=False)

            self._project_step(targets=self._target_points(), alignment=alignment)

            buttons = [
                Button("Continue", "next", "n", enabled=recorded >= 4, primary=recorded >= 4),
                Button("Auto-Adjust", "auto", "a", primary=recorded < 4),
                Button("Manual Adjust" if auto else "Auto Mode", "manual", "m"),
                Button("Redo Last", "retry", "r", enabled=recorded > 0),
                Button("Reset", "reset", "x", enabled=recorded > 0),
                Button("Record Here", "record", "e", enabled=not auto and active is not None),
                Button("Back", "back", "b"),
                Button("Cancel", "cancel", "q"),
            ]
            action = self._show(view, buttons)

            if action == "cancel":
                if self._confirm_cancel():
                    return None
            elif action == "back":
                return -1
            elif action == "auto":
                auto = True
                self._measure_marks()
                last_measure = time.perf_counter()
            elif action == "manual":
                auto = not auto
                logger.info("corner mapping mode: %s", "auto" if auto else "manual")
            elif action == "retry":
                self._forget_last_corner()
                auto = False
            elif action == "reset":
                logger.info("corner mapping reset by the user")
                self.correspondences.clear()
                self._target_offsets.clear()
                self._refresh_alignment()
                auto = False
            elif action == "record" and active is not None:
                # Pair the nudged target with the *detected* corner: the user has
                # just told us, by moving it there, that this projector pixel
                # lands on that cushion nose.
                self._record_corner(active, _corner_of(self.state.table_boundary, active))
                auto = False
            elif action == "click" and not auto and active is not None:
                point = self.console.take_click() if self.console else None
                if point is not None:
                    self._record_corner(active, point)
            elif action in _NUDGE_ACTIONS and active is not None:
                self._nudge_target(active, action)
                auto = False
                self._project_step(targets=self._target_points(), force=True)
            elif action == "next" and recorded >= 4:
                self._rebuild_mapper()
                if self.mapper is None:
                    logger.error("four corners recorded but the transform will not solve")
                    continue
                return 1

    def _draw_corner_states(self, view: np.ndarray, active: str | None) -> np.ndarray:
        """Mark every corner target on the preview in its current state."""
        for name in CORNER_NAMES:
            observed = self.measurements.observed_marks.get(name)
            if observed is not None:
                view = ui.draw_corner_target(
                    view, observed, CORNER_LABELS[name], is_recorded=True,
                    copy=False,
                )
            elif name == active and self.state.table_boundary is not None:
                # Not yet found. Point at where it belongs, so the user knows
                # which corner the wizard is waiting on.
                view = ui.draw_corner_target(
                    view,
                    _corner_of(self.state.table_boundary, name),
                    CORNER_LABELS[name],
                    is_active=True,
                    copy=False,
                )
        return view

    def _forget_last_corner(self) -> None:
        """Drop the last recorded corner in :data:`CORNER_NAMES` order.

        Canonical order rather than placement order, and for manual placement
        the two are the same thing -- corners are always offered for placement in
        that order, so the last one recorded is the last one present. After an
        auto pass, which records all four at once, "last" is simply the bottom
        left; a user who wants a specific corner back uses Reset.
        """
        for name in reversed(CORNER_NAMES):
            if name in self.correspondences:
                del self.correspondences[name]
                logger.info("cleared corner %s", name)
                break
        self._refresh_alignment()

    def _nudge_target(self, name: str, action: str) -> None:
        """Walk one projected target across the felt with the arrow keys."""
        step = self.settings.projector.width * _TARGET_NUDGE_FRACTION
        delta = _NUDGE_ACTIONS[action]
        current = self._target_offsets.get(name, Vec2(0.0, 0.0))
        self._target_offsets[name] = current + delta.scaled(step)
        logger.debug("target %s nudged to offset %s", name, self._target_offsets[name])

    def _rebuild_mapper(self) -> None:
        """Re-solve the calibration and the projector->camera transform.

        Both, together, because they are the two halves of every measurement
        from here on: the calibration is what the wizard projects *through*, and
        the projector->camera transform is how it works out where that light
        physically landed.
        """
        from projection.mapper import ProjectionMapper

        calibration = self._build_calibration()
        if calibration is None:
            self.mapper = None
            return
        try:
            self.mapper = ProjectionMapper(calibration)
        except ValueError as exc:
            logger.error("solved calibration is unusable: %s", exc)
            self.mapper = None
            return
        self.calibration = calibration
        self.projector_to_camera = solve_projector_to_camera(
            {
                name: (camera_px, self._apply_nudge(projector_px))
                for name, (camera_px, projector_px) in self.correspondences.items()
            }
        )

    def screen_fine_tune(self) -> int | None:
        """Screen 5: nudge the whole projection, with live squareness metrics.

        The grid over the felt is the feedback; the numbers next to it are the
        proof. Perpendicularity and coverage are computed from the two solved
        homographies rather than by looking for the projected grid lines in the
        image -- same answer, nothing to fail, and it stays correct when the
        user turns the grid off to see the cloth.

        A note on the stubbed design this replaces: the original plan was for
        nudging to fall back to an affine transform and warn the user that the
        keystone correction was being discarded. It does not need to. The nudge
        is applied to the recorded correspondences and the homography is
        re-solved, so keystone survives and there is nothing to warn about.
        """
        if self.mapper is None:
            self._rebuild_mapper()
        if self.mapper is None:
            logger.warning("no usable transform; sending the user back to corner mapping")
            return -1

        while True:
            frame = self._capture()
            self._refresh_grid_metrics()
            grid = self.measurements.grid

            view = ui.render_step_instructions(
                frame,
                self.state,
                "Arrow keys move the projection. + and - zoom, [ and ] rotate.",
                self.settings,
            )
            if self.state.table_boundary is not None:
                view = ui.draw_table_outline(view, self.state.table_boundary, show_handles=False, copy=False)
            view = ui.draw_metric_rows(view, "PROJECTION QUALITY", _grid_rows(grid), copy=False)
            if self.measurements.alignment is not None:
                view = ui.draw_alignment_feedback(
                    view, self.measurements.alignment, self.settings, copy=False
                )

            self._project_step()

            buttons = [
                Button("Continue", "next", "n", primary=True),
                Button("Left", "nudge_left"),
                Button("Right", "nudge_right"),
                Button("Up", "nudge_up"),
                Button("Down", "nudge_down"),
                Button("Reset Nudge", "reset", "x"),
                Button("Back", "back", "b"),
                Button("Cancel", "cancel", "q"),
            ]
            action = self._show(view, buttons)

            if action == "cancel":
                if self._confirm_cancel():
                    return None
            elif action == "back":
                return -1
            elif action == "reset":
                self._nudge_offset = Vec2(0.0, 0.0)
                self._nudge_scale = 1.0
                self._nudge_rotation_deg = 0.0
                logger.info("fine-tune nudge reset")
                self._rebuild_mapper()
                self._project_step(force=True)
            elif action in _NUDGE_ACTIONS or action in ("scale_up", "scale_down", "rotate_cw", "rotate_ccw"):
                self._nudge_projection(action)
                self._rebuild_mapper()
                self._project_step(force=True)
            elif action == "next":
                if grid is not None:
                    self.state.grid_rotation_error = grid.rotation_deg
                return 1

    def _nudge_projection(self, action: str) -> None:
        """Apply one fine-tune press to the whole projection."""
        if action in _NUDGE_ACTIONS:
            step = self.settings.projector.width * _PROJECTION_NUDGE_FRACTION
            self._nudge_offset = self._nudge_offset + _NUDGE_ACTIONS[action].scaled(step)
        elif action == "scale_up":
            self._nudge_scale += _SCALE_NUDGE
        elif action == "scale_down":
            self._nudge_scale = max(0.1, self._nudge_scale - _SCALE_NUDGE)
        elif action == "rotate_cw":
            self._nudge_rotation_deg += _ROTATION_NUDGE_DEG
        elif action == "rotate_ccw":
            self._nudge_rotation_deg -= _ROTATION_NUDGE_DEG
        self.state.projector_offset = self._nudge_offset
        self.state.projector_scale = Vec2(self._nudge_scale, self._nudge_scale)
        self.state.projector_rotation = self._nudge_rotation_deg
        logger.debug(
            "fine-tune: offset %s scale %.4f rotation %.2f deg",
            self._nudge_offset,
            self._nudge_scale,
            self._nudge_rotation_deg,
        )

    def _refresh_grid_metrics(self) -> None:
        """Recompute the squareness and coverage numbers, if possible."""
        if self.projector_to_camera is None or self.camera_to_table is None:
            self.measurements.grid = None
            return
        self.measurements.grid = compute_grid_metrics(
            self.projector_to_camera, self.camera_to_table, self.settings
        )

    def screen_test_projection(self) -> int | None:
        """Screen 6: project onto real balls and measure where the light lands.

        The only end-to-end check in the wizard, and the one that catches what
        the corner RMSE cannot: a transform can fit its own four corners
        perfectly and still be wrong in the middle of the table.

        The measurement is made against **photographed light**, not against
        arithmetic. A ring is projected at each detected ball's table position,
        the felt is photographed with the projector dark and then lit, and the
        rings are found by differencing. How far each ring landed from its ball
        is the error of the entire physical chain.

        The obvious cheaper version -- compose the calibration with the measured
        projector-to-camera homography and see where a ball maps to -- returns
        exactly zero for any input whatsoever, because both matrices are solved
        from the same four correspondences and four points determine a
        homography. It was written that way first and measured nothing. See
        :func:`calibration_ui.metrics.projection_error_in`.

        The user is asked to confirm visually as well, because a number cannot
        catch a projection that is landing on the floor.
        """
        if self.mapper is None:
            self._rebuild_mapper()
        if self.mapper is None:
            logger.warning("no usable transform; sending the user back to corner mapping")
            return -2

        answered_no = False
        balls: list[Ball] = []
        last_measure = 0.0

        while True:
            # Measured on a timer, not every frame: each measurement blinks the
            # projector, and a wizard that strobes while the user leans over the
            # table to place a ball is worse than one that updates every second.
            if time.perf_counter() - last_measure >= _REMEASURE_INTERVAL_S:
                balls = self._detect_balls(self._capture())
                check = self._measure_projection_error(balls)
                measured = check is not None and math.isfinite(check.mean_error_in)
                error_in = check.mean_error_in if measured and check is not None else None
                self.measurements.trajectory_error_in = error_in
                self.state.trajectory_test_error = 0.0 if error_in is None else error_in
                last_measure = time.perf_counter()

            frame = self._capture()
            error_in = self.measurements.trajectory_error_in

            view = ui.render_step_instructions(
                frame,
                self.state,
                _test_instruction(balls, error_in),
                self.settings,
            )
            view = self._draw_observed_marks(view)
            view = ui.draw_metric_rows(
                view, "END-TO-END CHECK", _trajectory_rows(self.measurements.projection_check),
                copy=False,
            )
            if answered_no:
                view = ui.draw_banner_message(
                    view, "Send it back to fine-tune, or re-place the corners.", "warning",
                    copy=False,
                )

            buttons = [
                Button("Yes, looks right", "yes", "y", primary=True),
                Button("No, re-adjust", "no", "o"),
                Button("Skip this test", "skip", "s"),
                Button("Back", "back", "b"),
                Button("Cancel", "cancel", "q"),
            ]
            action = self._show(view, buttons)

            if action == "cancel":
                if self._confirm_cancel():
                    return None
            elif action == "back":
                return -1
            elif action == "no":
                answered_no = True
            elif action == "skip":
                # A user with no balls to hand should not be stuck here, but the
                # report has to say the check did not run rather than imply it
                # passed.
                logger.info("end-to-end test skipped by the user")
                self.measurements.trajectory_error_in = None
                self.measurements.projection_check = None
                return 1
            elif action == "yes":
                return 1

    def _detect_balls(self, frame: np.ndarray) -> list[Ball]:
        """Find balls, with table positions filled in. Empty list on any failure."""
        from vision.detection import extract_game_state

        if self.state.table_boundary is None or self.camera_to_table is None:
            return []
        state = extract_game_state(
            frame,
            self._frame_index,
            time.perf_counter(),
            self.state.table_boundary,
            self.camera_to_table,
            self.settings,
        )
        return [ball for ball in state.balls if ball.table_pos is not None]

    def _measure_projection_error(self, balls: list[Ball]) -> ProjectionCheck | None:
        """Project a ring at every ball, photograph the felt, and measure the gap.

        Returns ``None`` when the check could not be made at all -- no balls, no
        transform, or no projected light reaching the camera. ``None`` is not
        zero, and the report says "not measured" rather than implying a pass.
        """
        from vision.calibration import pixels_per_inch

        self.measurements.projection_check = None
        if not balls or self.mapper is None or self.state.table_boundary is None:
            return None

        # Dark reference first, then the rings, same as corner mapping. The
        # projector has to actually be showing each frame before it is captured,
        # which is why the sends here are unthrottled.
        self._project_blank()
        dark = self._capture_settled()
        self._project_test_pattern(balls)
        lit = self._capture_settled()

        marks = locate_projected_marks(dark, lit, max_marks=max(8, len(balls) * 2))

        px_per_inch = pixels_per_inch(self.state.table_boundary, self.settings)
        # A ring more than three ball-radii from a ball is not that ball's ring.
        # Sized off the ball rather than fixed in px so it means the same thing
        # at any camera mounting height.
        check = projection_error_in(
            [ball.center_px for ball in balls],
            [mark.center for mark in marks],
            px_per_inch,
            max_match_px=3.0 * BALL_RADIUS_IN * px_per_inch,
        )
        self.measurements.projection_check = check
        return check

    def _project_test_pattern(self, balls: list[Ball]) -> None:
        """Ring every detected ball and draw an aiming line between two of them.

        The ring is drawn at a couple of ball radii so it lands on the *felt*
        around the ball rather than on the ball itself. That matters for the
        measurement: a mark painted onto a glossy sphere has its apparent
        centroid pulled around by the curvature, whereas a ring on flat cloth
        has its centroid exactly where it was projected.

        Sent unthrottled, because the very next thing the caller does is
        photograph the felt -- a dropped repaint here would be measured as a
        projection that missed every ball.
        """
        from projection import draw
        from projection.themes import resolve_theme

        if self.mapper is None:
            return

        theme = resolve_theme(self.settings)
        canvas = draw.new_canvas(self.settings)
        ppi = self.mapper.pixels_per_inch()

        for ball in balls:
            if ball.table_pos is None:
                continue
            draw.draw_ring(
                canvas,
                self.mapper.table_to_projector(ball.table_pos),
                ppi * BALL_RADIUS_IN * _TEST_RING_RADII,
                theme.ghost_ball,
                thickness=3,
                alpha=240,
            )

        positioned = [ball.table_pos for ball in balls if ball.table_pos is not None]
        if len(positioned) >= 2:
            draw.draw_polyline(
                canvas,
                self.mapper.table_to_projector_batch([positioned[0], positioned[1]]),
                theme.cue_path,
                thickness=4,
                alpha=250,
            )
        self._last_projection_at = time.perf_counter()
        self._send(canvas)

    def _draw_observed_marks(self, view: np.ndarray) -> np.ndarray:
        """Show where each ring actually landed against where its ball is.

        The same residual view as corner mapping, one screen later and in the
        middle of the table -- which is the part of the felt that no corner
        correspondence constrains, and therefore the only place this view can
        show something the corner screen did not already know.

        Drawn strictly from the matched pairs. Zipping the raw mark list against
        the raw ball list instead draws arrows between a ball and whichever ring
        happens to share its index, which produced 800-pixel arrows across the
        table next to a reported error of a fifth of an inch.
        """
        check = self.measurements.projection_check
        if check is None or not check.pairs:
            return view
        return ui.draw_projected_vs_detected(
            view, [mark for _ball, mark in check.pairs], [ball for ball, _mark in check.pairs],
            copy=False,
        )

    def screen_complete(self) -> int | None:
        """Screen 7: report the result, and only then offer to save.

        If any check failed there is no Finish button. Saving a bad calibration
        means every prediction the system makes afterwards is wrong and the user
        has no way to know why -- so the wizard would rather send them back
        around than hand them a broken table.

        The gate is the *whole* scorecard, not the corner RMSE. Gating on RMSE
        alone was the first implementation and it passed a projection measured
        at 37 degrees out of square covering two thirds of the table, because a
        four-point fit reports its own training error and that is always near
        zero. The number that can actually fail has to be one of the ones on the
        gate.
        """
        self._rebuild_mapper()
        calibration = self.calibration
        rmse = calibration.rmse_px if calibration is not None else float("inf")
        rows = self._final_rows(self.assess_alignment(rmse))
        verdict = _worst_verdict(rows, rmse, self.assess_alignment(rmse))
        acceptable = calibration is not None and _score_label(rows) != "POOR"

        self._project_step(alignment=verdict, force=True)

        while True:
            frame = self._capture()
            view = ui.render_step_instructions(
                frame,
                self.state,
                "Calibration complete." if acceptable else "Not good enough to save yet.",
                self.settings,
            )
            view = ui.draw_metric_rows(view, f"FINAL SCORE: {_score_label(rows)}", rows, copy=False)
            view = ui.draw_alignment_feedback(view, verdict, self.settings, copy=False)

            buttons = [
                Button("Finish & Save", "next", "n", enabled=acceptable, primary=acceptable),
                Button("Redo Corners", "retry", "r", primary=not acceptable),
                Button("Back", "back", "b"),
                Button("Cancel", "cancel", "q"),
            ]
            action = self._show(view, buttons)

            if action == "cancel":
                if self._confirm_cancel():
                    return None
            elif action == "back":
                return -1
            elif action == "retry":
                return 4 - self.state.step  # back to corner mapping
            elif action == "next" and acceptable:
                logger.info(
                    "calibration accepted: RMSE %.1f px, trajectory error %s",
                    rmse,
                    _format_inches(self.measurements.trajectory_error_in),
                )
                return 1

    def _final_rows(self, verdict: AlignmentError) -> list[tuple[str, str, Severity]]:
        """The completion screen's scorecard."""
        boundary = self.state.table_boundary
        grid = self.measurements.grid
        error_in = self.measurements.trajectory_error_in

        rows: list[tuple[str, str, Severity]] = []
        if boundary is not None:
            rows.append(
                (
                    "Table detection",
                    f"{boundary.confidence * 100:.0f}%",
                    "info" if boundary.confidence >= CONFIDENCE_GOOD else "warning",
                )
            )
        rows.append(("Corner alignment", f"{verdict.total_rmse:.1f} px", verdict.severity))
        if grid is not None:
            rows.append(
                (
                    "Grid squareness",
                    f"{grid.perpendicularity_deg:.1f} deg",
                    _band(abs(grid.perpendicularity_deg - 90.0), *_SQUARENESS_BANDS),
                )
            )
            rows.append(
                (
                    "Projection rotation",
                    f"{grid.rotation_deg:+.1f} deg",
                    _band(abs(grid.rotation_deg), *_ROTATION_BANDS),
                )
            )
            rows.append(
                (
                    "Table coverage",
                    f"{grid.coverage_x_pct:.0f}% x {grid.coverage_y_pct:.0f}%",
                    _coverage_severity(min(grid.coverage_x_pct, grid.coverage_y_pct)),
                )
            )
        check = self.measurements.projection_check
        if check is not None:
            rows.append(
                (
                    "Rings landed on their ball",
                    f"{check.matched} of {check.balls_checked}",
                    "info" if check.all_matched else "warning",
                )
            )
        rows.append(
            (
                "End-to-end accuracy",
                _format_inches(error_in),
                _trajectory_severity(error_in),
            )
        )
        return rows


# ---------------------------------------------------------------------------
# Screen text and small pure helpers
# ---------------------------------------------------------------------------

#: Arrow-key actions and the direction each moves things, in projector px.
_NUDGE_ACTIONS: dict[str, Vec2] = {
    "nudge_left": Vec2(-1.0, 0.0),
    "nudge_right": Vec2(1.0, 0.0),
    "nudge_up": Vec2(0.0, -1.0),
    "nudge_down": Vec2(0.0, 1.0),
}


def _corner_of(boundary: TableBoundary, name: str) -> Vec2:
    """One named corner of a boundary, camera px."""
    return boundary.corners()[CORNER_NAMES.index(name)]


def _detection_advice(confidence: float, selected: str | None) -> tuple[str, Severity]:
    """What to tell the user about the table detection, per the spec's bands."""
    if selected is not None:
        return f"Tap where the {CORNER_LABELS[selected].lower()} cushion nose is.", "warning"
    if confidence >= CONFIDENCE_GOOD:
        return "Looking good -- the outline should sit on the cushion noses.", "info"
    if confidence >= CONFIDENCE_MARGINAL:
        return "Almost there. Hold still, or nudge the corners by hand.", "warning"
    return "Cannot see the table clearly.", "error"


def _apply_corner_tap(
    boundary: TableBoundary, selected: str | None, point: Vec2, frame_shape: tuple[int, ...]
) -> tuple[TableBoundary, str | None]:
    """Handle one tap on the camera view during manual corner correction.

    Two taps per correction: the first picks up the nearest handle, the second
    puts it down. The pick-up radius is generous -- a tenth of the frame -- since
    the alternative on a touchscreen is a user tapping four times before they
    grab anything.
    """
    if selected is None:
        nearest = min(CORNER_NAMES, key=lambda name: _corner_of(boundary, name).distance_to(point))
        if _corner_of(boundary, nearest).distance_to(point) > frame_shape[1] * 0.1:
            logger.debug("tap at %s was not near a corner handle; ignoring", point)
            return boundary, None
        logger.info("picked up the %s corner", nearest)
        return boundary, nearest

    corners = list(boundary.corners())
    corners[CORNER_NAMES.index(selected)] = point
    logger.info("moved the %s corner to (%.0f, %.0f)", selected, point.x, point.y)
    return _rebuild_boundary(boundary, corners), None


def _rebuild_boundary(boundary: TableBoundary, corners: list[Vec2]) -> TableBoundary:
    """Rebuild a boundary around edited corners, keeping the clockwise order.

    Re-ordering matters: a user can drag the top-left handle past the top-right
    one, and a boundary whose ``corners()`` are no longer clockwise solves to a
    mirrored homography -- which looks like the projector is throwing the image
    backwards and gives no clue why.
    """
    from vision.calibration import order_corners

    array = np.array([[corner.x, corner.y] for corner in corners], dtype=np.float32)
    ordered = order_corners(array)
    points = [Vec2(float(x), float(y)) for x, y in ordered]
    center = Vec2(
        sum(point.x for point in points) / 4.0, sum(point.y for point in points) / 4.0
    )
    width_px = (points[0].distance_to(points[1]) + points[3].distance_to(points[2])) / 2.0
    height_px = (points[0].distance_to(points[3]) + points[1].distance_to(points[2])) / 2.0
    return TableBoundary(
        top_left=points[0],
        top_right=points[1],
        bottom_right=points[2],
        bottom_left=points[3],
        center=center,
        width_px=width_px,
        height_px=height_px,
        # A hand-placed corner is exactly as good as the user's eyesight, which
        # is better than a marginal detection and not something to score. 1.0
        # would claim a certainty nothing measured.
        confidence=max(boundary.confidence, CONFIDENCE_GOOD),
    )


def _mapping_instruction(auto: bool, active: str | None, recorded: int) -> str:
    """One sentence for the corner-mapping screen, matching the current mode."""
    if recorded >= 4:
        return "All four corners placed. Follow the arrows to shrink the error."
    if auto:
        return "Finding the projected marks... aim the projector at the table."
    if active is None:
        return "All four corners placed."
    corner = CORNER_LABELS[active].lower()
    return f"Tap the {corner} mark, or arrow it onto the {corner} cushion nose."


def _test_instruction(balls: list[Ball], error_in: float | None) -> str:
    """One sentence for the test-projection screen."""
    if not balls:
        return "Put a ball or two on the table, or skip this test."
    if error_in is None:
        return "Checking the projection against the balls..."
    return "Do the rings sit on the balls?"


#: ``(good_below, warn_below)`` for each graded check, in its own units.
#: Three bands rather than the pass/fail on :class:`GridMetrics` because the
#: completion screen gates on a red row: with only pass and warn, a projection
#: measured 37 degrees out of square scores GOOD and gets saved.
_SQUARENESS_BANDS = (2.0, 6.0)  # degrees away from 90
_ROTATION_BANDS = (2.0, 6.0)  # degrees away from the rails
#: Coverage runs the other way -- higher is better -- so it has its own helper.
_COVERAGE_GOOD_PCT = 95.0
_COVERAGE_WARN_PCT = 85.0


def _band(value: float, good_below: float, warn_below: float) -> Severity:
    """Grade a smaller-is-better measurement into a severity."""
    if value <= good_below:
        return "info"
    return "warning" if value <= warn_below else "error"


def _coverage_severity(pct: float) -> Severity:
    """Grade a larger-is-better percentage into a severity."""
    if pct >= _COVERAGE_GOOD_PCT:
        return "info"
    return "warning" if pct >= _COVERAGE_WARN_PCT else "error"


def _grid_rows(grid: GridMetrics | None) -> list[tuple[str, str, Severity]]:
    """Scorecard rows for the fine-tune screen.

    Same thresholds as the completion screen's, so a row the user sees as green
    here cannot turn red two screens later with nothing changed in between.
    """
    if grid is None:
        return [("Projection geometry", "not measured", "warning")]
    return [
        (
            "Perpendicularity (90 is perfect)",
            f"{grid.perpendicularity_deg:.1f} deg",
            _band(abs(grid.perpendicularity_deg - 90.0), *_SQUARENESS_BANDS),
        ),
        (
            "Rotation against the rails",
            f"{grid.rotation_deg:+.1f} deg",
            _band(abs(grid.rotation_deg), *_ROTATION_BANDS),
        ),
        (
            "Coverage along the table",
            f"{grid.coverage_x_pct:.0f}%",
            _coverage_severity(grid.coverage_x_pct),
        ),
        (
            "Coverage across the table",
            f"{grid.coverage_y_pct:.0f}%",
            _coverage_severity(grid.coverage_y_pct),
        ),
    ]


def _trajectory_rows(check: ProjectionCheck | None) -> list[tuple[str, str, Severity]]:
    """Scorecard rows for the test-projection screen.

    "Rings landed on" sits next to the error, and it is the more important of
    the two: a tiny error over one ball out of six means the projection missed
    five of them, which a mean alone would report as a pass.
    """
    if check is None:
        return [("Projection check", "not measured", "warning")]
    error_in = check.mean_error_in if math.isfinite(check.mean_error_in) else None
    return [
        ("Balls found", str(check.balls_checked), "info" if check.balls_checked else "warning"),
        (
            "Rings landed on",
            f"{check.matched} of {check.balls_checked}",
            "info" if check.all_matched else "warning",
        ),
        ("Projection error mid-table", _format_inches(error_in), _trajectory_severity(error_in)),
    ]


def _trajectory_severity(error_in: float | None) -> Severity:
    """How bad an end-to-end error is, in playing terms."""
    if error_in is None:
        return "warning"
    if error_in <= TRAJECTORY_EXCELLENT_IN:
        return "info"
    if error_in <= TRAJECTORY_ACCEPTABLE_IN:
        return "warning"
    return "error"


def _format_inches(error_in: float | None) -> str:
    """An inch measurement, or an honest admission that there is not one."""
    return "not measured" if error_in is None else f"{error_in:.2f} in"


def _worst_verdict(
    rows: list[tuple[str, str, Severity]], rmse: float, fallback: AlignmentError
) -> AlignmentError:
    """The message for the completion screen, naming the worst failing check.

    "Alignment looks excellent" over a scorecard with a red row on it is a lie
    the user has no reason to doubt, and they will go and play on a table whose
    projection misses by two inches. So the headline names the row that failed;
    the scorecard next to it says by how much.
    """
    failing = [(label, severity) for label, _value, severity in rows if severity == "error"]
    if failing:
        label, _severity = failing[0]
        return AlignmentError(
            total_rmse=rmse,
            message=f"{label} is out of tolerance. Go back and adjust, then re-check.",
            severity="error",
        )
    warning = [(label, severity) for label, _value, severity in rows if severity == "warning"]
    if warning:
        label, _severity = warning[0]
        return AlignmentError(
            total_rmse=rmse,
            message=f"Good enough to play, but {label.lower()} could be better.",
            severity="warning",
        )
    return fallback


def _score_label(rows: list[tuple[str, str, Severity]]) -> str:
    """A one-word verdict over every scorecard row.

    Deliberately harsh: one red makes the whole thing POOR. The alternative is
    averaging, which lets a perfect corner fit hide a projection that misses the
    balls by two inches -- and the corner fit is the one number that is near
    zero by construction.
    """
    severities = [severity for _label, _value, severity in rows]
    if "error" in severities:
        return "POOR"
    if "warning" in severities:
        return "GOOD"
    return "EXCELLENT"


def run_calibration_app(
    settings: Settings | None = None, console: Console | None = None
) -> ProjectorCalibration | None:
    """Run the wizard. ``python -m calibration_ui.calibration_app``"""
    from utils.logging import setup_logging

    settings = settings or get_settings()
    setup_logging(settings.system.log_level, settings.system.log_to_file)
    return CalibrationApp(settings, console=console).run()


def main(argv: list[str] | None = None) -> int:
    """Command line entry point, matching the flags the other tools take."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="python -m calibration_ui.calibration_app",
        description="Interactive projector-to-table calibration wizard.",
    )
    parser.add_argument("--config", type=Path, help="use a different config.yaml")
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "synthetic camera and discarded projector output. Walks the screens "
            "with no hardware; the calibration it produces is meaningless and "
            "the welcome screen says so."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="log every measurement and input event"
    )
    args = parser.parse_args(argv)

    from app.config import load_settings

    settings = load_settings(args.config) if args.config else get_settings()
    if args.mock:
        settings.camera.use_mock = True
        settings.projector.use_mock = True
    if args.debug:
        settings.system.log_level = "DEBUG"

    calibration = run_calibration_app(settings)
    if calibration is None:
        # Non-zero on cancel as well as on failure: a script that runs the
        # wizard and then starts the app needs to know the transform did not
        # change, and "the user walked away" is not a success.
        print("Calibration was not saved.")
        return 1
    print(
        f"Calibration saved: corner RMSE {calibration.rmse_px:.1f} px, "
        f"created {calibration.created_at}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
