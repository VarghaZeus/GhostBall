"""REST endpoints for the control panel.

Implemented against :class:`app.state.AppState`, so every route works today and
returns real values -- FPS, backend names, calibration status, settings. Routes
that need an unimplemented pipeline stage return HTTP 503 with the name of the
stage rather than 500, which distinguishes "not built yet" from "broken" for
anyone poking at the API during the build-out.

Threading note: these handlers run on uvicorn's event loop while the vision loop
runs in its own thread. They therefore only ever *read* shared state, or mutate
it through :class:`~app.state.AppState`, which serialises writes. No handler
touches a camera or a display directly.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from app.models import GameModeName
from web.schemas import (
    ActionResponse,
    BrightnessRequest,
    CalibrationFinalizeResponse,
    CalibrationPointRequest,
    CalibrationStatusResponse,
    ChallengeRequest,
    DetectionCountsResponse,
    DifficultyRequest,
    DrillRequest,
    FocusResponse,
    HealthResponse,
    ModeRequest,
    NudgeRequest,
    PatternRequest,
    PatternResponse,
    PerfResponse,
    PlayerResponse,
    SessionResponse,
    SettingsRequest,
    SettingsResponse,
    StatusResponse,
    SystemResponse,
    TrainingResultResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ar-pool"])


def _state(request: Request):
    """Pull the shared :class:`~app.state.AppState` off the FastAPI app.

    Stored on ``app.state`` rather than in a module global so that tests can
    build an app with a fresh state, and so a second app instance in one process
    does not silently share the first one's camera.
    """
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise HTTPException(status_code=503, detail="application state not initialised")
    return state


def _require(stage: str, state) -> None:
    """503 if a pipeline stage the route depends on is not implemented yet.

    503 rather than 501, because from the panel's perspective this is a
    temporarily unavailable capability it should keep polling for, not a
    permanent protocol gap.
    """
    if stage in state.pending_stages:
        raise HTTPException(
            status_code=503,
            detail=f"pipeline stage '{stage}' is not implemented yet",
        )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request) -> StatusResponse:
    """Everything the panel polls, in one call.

    Deliberately one endpoint rather than several: the panel polls this a few
    times a second, and on a Pi that is already CPU-bound the difference between
    one request and six is worth having.
    """
    state = _state(request)
    from utils.performance import get_system_metrics

    perf = state.tracker.snapshot()
    metrics = get_system_metrics()
    detections = state.detection_summary()

    return StatusResponse(
        running=state.is_running,
        current_mode=state.mode_manager.session.mode,
        session_state=state.mode_manager.session.state,
        is_calibrated=state.mapper.calibration.is_calibrated,
        calibration_rmse_px=state.mapper.calibration.rmse_px,
        last_shot_confidence=state.last_shot_confidence,
        performance=PerfResponse(
            fps=perf.fps,
            frame_ms_avg=perf.frame_ms_avg,
            frame_ms_p95=perf.frame_ms_p95,
            latency_ms=perf.latency_ms,
            dropped_frames=perf.dropped_frames,
            total_frames=perf.total_frames,
            stage_ms=perf.stage_ms,
            frame_age_ms=(
                None if (age := state.frame_age_ms()) is None else round(age, 1)
            ),
        ),
        detections=DetectionCountsResponse(**detections),
        system=SystemResponse(
            cpu_pct=metrics["cpu_pct"],
            mem_pct=metrics["mem_pct"],
            temp_c=metrics["temp_c"],
            camera_backend=state.camera.backend_name if state.camera else "none",
            display_backend=state.display.backend_name if state.display else "none",
            using_mock_camera=bool(state.camera and state.camera.is_mock),
            using_mock_display=bool(state.display and state.display.is_mock),
            camera_resolution=f"{state.settings.camera.width}x{state.settings.camera.height}",
            camera_target_fps=state.settings.camera.fps,
            projector_resolution=(
                f"{state.settings.projector.width}x{state.settings.projector.height}"
            ),
            projection_override=state.projection_override,
        ),
        health=HealthResponse(**state.health_summary()),
        focus=FocusResponse(**state.focus_summary()),
        pending_stages=sorted(state.pending_stages),
    )


@router.get("/session", response_model=SessionResponse)
async def get_session(request: Request) -> SessionResponse:
    """Current scoreboard."""
    import time

    state = _state(request)
    session = state.mode_manager.session
    current = session.current_player
    return SessionResponse(
        mode=session.mode,
        state=session.state,
        players=[
            PlayerResponse(
                name=p.name,
                score=p.score,
                shots_taken=p.shots_taken,
                accuracy_pct=round(p.accuracy_pct, 1),
                is_eliminated=p.is_eliminated,
            )
            for p in session.players
        ],
        current_player=current.name if current else None,
        combo_count=session.combo_count,
        elapsed_seconds=(
            round(time.perf_counter() - session.started_at, 1) if session.started_at else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Mode and settings
# ---------------------------------------------------------------------------


@router.post("/mode", response_model=ActionResponse)
async def set_mode(request: Request, body: ModeRequest) -> ActionResponse:
    """Switch game mode, optionally starting a new game with the given players.

    Refuses a mode that does not exist yet rather than accepting it. The manager
    falls back to freeplay for an unimplemented mode, which is the right
    behaviour down there -- a bad value must not take the game down -- but
    answering 200 "mode set to Freeplay" to a request for King of the Hill tells
    the user their tap worked when it did nothing.
    """
    from modes.mode_manager import implemented_modes

    state = _state(request)
    available = implemented_modes()
    if body.mode not in available:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{body.mode.value.replace('_', ' ')} is not built yet; "
                f"available modes: {', '.join(m.value for m in available)}"
            ),
        )

    mode = state.mode_manager.load_mode(body.mode)
    if body.players and mode.is_competitive:
        state.mode_manager.start_game(body.players)
        # start_game rebuilds the player list, so the mode has to be told again
        # -- King of the Hill starts its turn clock in on_enter, and a clock
        # started before the players existed counts down for nobody.
        mode.on_enter(state.mode_manager.session)
    # Ask the loop to blank rather than blanking here: leaving the previous
    # mode's overlay on the felt through the switch looks like the system hung,
    # but the projector belongs to the loop's thread. See the note on
    # AppState.blank_requested.
    state.request_blank()
    return ActionResponse(message=f"mode set to {mode.display_name}")


@router.post("/mode/difficulty", response_model=ActionResponse)
async def set_difficulty(request: Request, body: DifficultyRequest) -> ActionResponse:
    """Set the difficulty of the active mode.

    409 rather than 400 when the mode has no difficulty: the request is
    well-formed and the user is not wrong to ask, the table is simply in a state
    where it means nothing. The message names the mode that does have one.
    """
    state = _state(request)
    mode = state.mode_manager.mode
    setter = getattr(mode, "set_difficulty", None)
    if setter is None:
        raise HTTPException(
            status_code=409,
            detail=f"{mode.display_name} has no difficulty setting; king of the hill does",
        )
    chosen = setter(body.difficulty)
    return ActionResponse(message=f"difficulty set to {chosen.value}")


@router.post("/mode/challenge", response_model=ActionResponse)
async def select_challenge(request: Request, body: ChallengeRequest) -> ActionResponse:
    """Jump to a trick shot challenge by position."""
    from modes.trick_shots import TrickShotsMode

    state = _state(request)
    mode = state.mode_manager.mode
    if not isinstance(mode, TrickShotsMode):
        raise HTTPException(
            status_code=409,
            detail=f"{mode.display_name} has no challenges; switch to trick shots first",
        )
    challenge = mode.select(body.index)
    if challenge is None:
        raise HTTPException(status_code=503, detail="no challenges could be loaded")
    state.request_blank()
    return ActionResponse(message=f"{challenge.name}: {challenge.description}")


@router.get("/mode/challenges", response_model=list[dict])
async def list_challenges(request: Request) -> list[dict]:
    """Every trick shot with the best result so far, for the panel's list."""
    from modes.trick_shots import TrickShotsMode, challenge_summary

    state = _state(request)
    mode = state.mode_manager.mode
    if not isinstance(mode, TrickShotsMode):
        return []
    return challenge_summary(mode)


@router.get("/settings", response_model=SettingsResponse)
async def get_settings_route(request: Request) -> SettingsResponse:
    """Live settings, so the panel renders real slider positions on load."""
    from modes.mode_manager import implemented_modes
    from projection.themes import theme_names

    state = _state(request)
    settings = state.settings
    return SettingsResponse(
        brightness=settings.projector.brightness_pct,
        overlay_alpha=settings.projector.overlay_alpha_pct,
        trajectory_smoothing=settings.render.trajectory_smoothing_pct,
        physics_accuracy=settings.physics.accuracy,
        theme=settings.render.theme,
        target_fps=settings.system.target_fps,
        table_preset=settings.table_preset,
        auto_detect_cue=state.auto_detect_cue,
        available_themes=theme_names(),
        available_modes=implemented_modes(),
    )


@router.post("/settings", response_model=SettingsResponse)
async def update_settings(request: Request, body: SettingsRequest) -> SettingsResponse:
    """Apply a partial settings update.

    Mutates the live settings object in place, which is what makes changes take
    effect on the very next frame without a restart. Not persisted to
    ``config.yaml`` -- a slider dragged mid-game should not permanently rewrite
    the user's tuned config.
    """
    state = _state(request)
    settings = state.settings

    if body.brightness is not None:
        settings.projector.brightness_pct = body.brightness
    if body.overlay_alpha is not None:
        settings.projector.overlay_alpha_pct = body.overlay_alpha
    if body.trajectory_smoothing is not None:
        settings.render.trajectory_smoothing_pct = body.trajectory_smoothing
    if body.physics_accuracy is not None:
        settings.physics.accuracy = body.physics_accuracy
    if body.theme is not None:
        # Validated here rather than left to the renderer's fallback. The
        # renderer does fall back to ``classic`` on an unknown name, which is the
        # right behaviour for a hand-edited config file -- but a panel that
        # accepts a theme, returns 200, and then shows a different theme is a
        # confusing way to report a typo.
        from projection.themes import theme_names

        if body.theme not in theme_names():
            raise HTTPException(
                status_code=400,
                detail=f"unknown theme {body.theme!r}; expected one of {theme_names()}",
            )
        settings.render.theme = body.theme
    if body.auto_detect_cue is not None:
        state.auto_detect_cue = body.auto_detect_cue

    logger.info("settings updated via API: %s", body.model_dump(exclude_none=True))
    return await get_settings_route(request)


@router.post("/projector/brightness", response_model=ActionResponse)
async def set_brightness(request: Request, body: BrightnessRequest) -> ActionResponse:
    """Set overlay output brightness.

    This scales the rendered overlay, not the projector lamp. Lamp control would
    need HDMI-CEC, which the GoodDee does not reliably expose -- and scaling the
    overlay is actually the better control, since it dims the graphics without
    changing how the felt looks.
    """
    state = _state(request)
    state.settings.projector.brightness_pct = body.value
    return ActionResponse(message=f"overlay brightness set to {body.value}%")


@router.get("/projector/patterns", response_model=PatternResponse)
async def list_patterns(request: Request) -> PatternResponse:
    """Available test patterns and which one is currently projected."""
    from projection.patterns import TestPattern

    state = _state(request)
    return PatternResponse(
        active=state.projection_override,
        available=[p.value for p in TestPattern],
    )


@router.post("/projector/pattern", response_model=PatternResponse)
async def set_pattern(request: Request, body: PatternRequest) -> PatternResponse:
    """Project a test pattern, or ``pattern: null`` to hand the projector back.

    This is the half of ``tools/projection_test.py`` that cannot be a CLI. The
    person aligning a projector is standing behind it with both hands on it, and
    cannot also be at a keyboard -- so the patterns have to be reachable from a
    phone.

    The pattern is *recorded*, not drawn: the projector belongs to the vision
    loop's thread, which picks the request up on its next pass. See the note on
    :attr:`app.state.AppState.projection_override`.
    """
    from projection.patterns import TestPattern

    state = _state(request)
    available = [p.value for p in TestPattern]

    if body.pattern is None:
        state.request_blank()
        return PatternResponse(active=None, available=available, message="projection cleared")

    if body.pattern not in available:
        raise HTTPException(
            status_code=400,
            detail=f"unknown pattern {body.pattern!r}; expected one of {available}",
        )

    state.projection_override = body.pattern
    # Not blanking first: the loop overwrites the whole frame each pass, so a
    # blank in between would only add a black flash.
    logger.info("projecting test pattern %s", body.pattern)
    return PatternResponse(
        active=body.pattern,
        available=available,
        message=f"projecting {body.pattern.replace('_', ' ')} pattern",
    )


@router.post("/reset", response_model=ActionResponse)
async def reset(request: Request) -> ActionResponse:
    """Clear game state and blank the projection."""
    state = _state(request)
    state.mode_manager.reset()
    state.request_blank()
    state.tracker.reset()
    return ActionResponse(message="session reset")


# ---------------------------------------------------------------------------
# Camera preview
# ---------------------------------------------------------------------------


def _overlay_in_camera_space(state, height: int, width: int):
    """Warp the projected overlay from projector space into camera space.

    Without this step a "preview" would be meaningless. The overlay is drawn in
    *projector* pixels and the frame is in *camera* pixels; the two are the same
    size by default, so blending them directly produces an image that looks
    plausible and is wrong -- the trajectory would appear wherever the projector
    happens to put it in its own frame, not where it actually lands on the felt.

    The composition is ``projector -> table -> camera``: the mapper's inverse
    followed by the table homography the detector solved. Returns ``None`` when
    either is unavailable, which is the normal state before the table has been
    found.
    """
    import cv2

    overlay = state.latest_overlay
    if overlay is None or state.table_to_camera is None:
        return None
    warp = state.table_to_camera @ state.mapper.inverse_matrix
    return cv2.warpPerspective(
        overlay,
        warp,
        (width, height),
        # Anything outside the projector's frame maps to fully transparent
        # rather than to black, so the felt shows through in the preview exactly
        # as it does in reality.
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


@router.get(
    "/preview.jpg",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}, 503: {"description": "no frame yet"}},
)
async def preview(
    request: Request,
    width: int = Query(640, ge=160, le=3840, description="output width in px"),
    overlay: bool = Query(False, description="blend in what is being projected"),
) -> Response:
    """A JPEG of what the camera currently sees.

    The only window into a headless Pi under a pool table, and the view the
    calibration wizard is built around: with ``overlay=1`` it shows the
    projection superimposed on the camera's view, which is the comparison the
    whole alignment process consists of.

    Downscaled *before* anything else happens. At full 1080p the encode alone is
    ~15 ms and the warp another ~10, all of it on the event loop thread
    competing with the vision loop for the same cores; at 640 px the whole
    endpoint is a couple of milliseconds. The panel displays this in a card a
    few hundred pixels wide, so the resolution buys nothing.
    """
    import cv2

    state = _state(request)
    frame = state.latest_frame
    if frame is None:
        # 503, not 404: the resource is not missing, it is not ready. The panel
        # polls this, and a stream of 404s in the console buries real errors.
        raise HTTPException(status_code=503, detail="no frame captured yet")

    if overlay:
        # Warp at full resolution and downscale after: warping the small image
        # would sample a transform built for full-resolution pixels.
        warped = _overlay_in_camera_space(state, frame.shape[0], frame.shape[1])
        if warped is not None:
            from projection.renderer import blend_overlay

            frame = blend_overlay(frame, warped, state.settings.overlay_alpha)

    if width < frame.shape[1]:
        scale = width / frame.shape[1]
        frame = cv2.resize(
            frame, (width, max(1, int(round(frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA
        )

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encoding failed")

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        # The panel re-requests this every poll. Without no-store, Safari in
        # particular serves the first frame forever and the preview looks frozen.
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@router.get("/calibration/status", response_model=CalibrationStatusResponse)
async def calibration_status(request: Request) -> CalibrationStatusResponse:
    """Calibration state and a plain-language alignment verdict."""
    state = _state(request)
    cal = state.mapper.calibration

    if not cal.is_calibrated:
        quality = "uncalibrated"
    elif cal.rmse_px <= 5.0:
        quality = "excellent"
    elif cal.rmse_px <= 20.0:
        # 20 px is the spec's stated target, so it is the pass/fail line.
        quality = "good"
    else:
        quality = "poor"

    return CalibrationStatusResponse(
        is_calibrated=cal.is_calibrated,
        table_detected=state.table_boundary is not None,
        rmse_px=round(cal.rmse_px, 2),
        alignment_quality=quality,
        corners_recorded=len(state.calibration_points),
        created_at=cal.created_at,
    )


@router.post("/calibration/corner/{corner}", response_model=ActionResponse)
async def record_calibration_corner(
    request: Request, corner: str, body: CalibrationPointRequest
) -> ActionResponse:
    """Record one camera-px / projector-px correspondence.

    Keyed by corner name so re-recording a corner replaces it rather than
    appending -- a user who nudges the top-left target three times should end up
    with one point, not three near-duplicates that skew the least-squares fit.
    """
    state = _state(request)
    valid = {"top_left", "top_right", "bottom_right", "bottom_left", "center"}
    if corner not in valid:
        raise HTTPException(
            status_code=400, detail=f"unknown corner {corner!r}; expected one of {sorted(valid)}"
        )

    state.calibration_points[corner] = (body.camera_px, body.projector_px)
    logger.info(
        "recorded calibration corner %s: camera %s -> projector %s",
        corner,
        body.camera_px,
        body.projector_px,
    )
    return ActionResponse(
        message=f"{corner} recorded ({len(state.calibration_points)} of 4 minimum)"
    )


@router.post("/calibration/nudge", response_model=ActionResponse)
async def nudge_calibration(request: Request, body: NudgeRequest) -> ActionResponse:
    """Apply a fine-tune adjustment to the live calibration."""
    state = _state(request)
    state.mapper.nudge(
        dx=body.dx, dy=body.dy, dscale=body.dscale, drotation=body.drotation
    )
    return ActionResponse(message="calibration nudged")


@router.post("/calibration/finalize", response_model=CalibrationFinalizeResponse)
async def finalize_calibration(request: Request) -> CalibrationFinalizeResponse:
    """Solve the transform from the recorded corners and save it.

    Returns ``success=False`` with an explanatory message on too few points or a
    failed solve, rather than raising. The calibration UI shows this text
    directly to a non-technical user, so a stack trace would be useless to them.
    """
    state = _state(request)
    from projection.mapper import ProjectionMapper, save_calibration, solve_projector_homography
    from vision.calibration import camera_to_table_coords

    if len(state.calibration_points) < 4:
        return CalibrationFinalizeResponse(
            success=False,
            message=f"need at least 4 corners, have {len(state.calibration_points)}",
        )

    if state.camera_to_table is None:
        return CalibrationFinalizeResponse(
            success=False,
            message="table not detected yet, so camera points cannot be mapped to the table",
        )

    from app.models import Vec2

    table_points: list[Vec2] = []
    projector_points: list[Vec2] = []
    try:
        for camera_px, projector_px in state.calibration_points.values():
            # The correspondence the user gave is camera->projector, but the
            # mapper's input space is table inches -- so each camera point goes
            # through the table homography first.
            table_points.append(
                camera_to_table_coords(Vec2(*camera_px), state.camera_to_table)
            )
            projector_points.append(Vec2(*projector_px))

        calibration = solve_projector_homography(
            table_points,
            projector_points,
            state.settings.projector.width,
            state.settings.projector.height,
        )
    except (ValueError, RuntimeError) as exc:
        logger.error("calibration solve failed: %s", exc)
        return CalibrationFinalizeResponse(success=False, message=str(exc))

    state.mapper = ProjectionMapper(calibration)
    save_calibration(calibration)

    within_target = calibration.rmse_px <= 20.0
    return CalibrationFinalizeResponse(
        success=True,
        rmse_error_pixels=round(calibration.rmse_px, 2),
        message=(
            f"calibrated, {calibration.rmse_px:.1f} px RMSE"
            + ("" if within_target else " -- above the 20 px target, consider recalibrating")
        ),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@router.post("/training/start_drill", response_model=ActionResponse)
async def start_drill(request: Request, body: DrillRequest) -> ActionResponse:
    """Begin a training drill.

    Capability is checked before preconditions. The other order reads more
    naturally and gives worse answers: with the drill machinery unbuilt, telling
    the user "no frame captured yet" sends them off to check the camera for a
    problem that is not there.
    """
    state = _state(request)
    _require("modes", state)
    _require("detection", state)

    from modes.training import TrainingMode

    if state.mode_manager.session.mode is not GameModeName.TRAINING:
        state.mode_manager.load_mode(GameModeName.TRAINING)

    mode = state.mode_manager.mode
    assert isinstance(mode, TrainingMode)
    if state.latest_game_state is None:
        raise HTTPException(status_code=503, detail="no frame captured yet")

    from modes.training import DrillUnavailable

    try:
        drill = mode.start_drill(body.drill_type, state.latest_game_state)
    except DrillUnavailable as exc:
        # The layout does not support this drill -- no clear pot, or no cue ball
        # on the cloth. 409 rather than 503: nothing is broken or unbuilt, the
        # request simply conflicts with the state of the table, and the message
        # says what to move. 500 here would send someone reading logs for a
        # fault that is a ball in the wrong place.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NotImplementedError as exc:
        # Backstop for a stub *inside* an otherwise-available stage. Kept now
        # that the modes are built, because the same trap applies to the next
        # unfinished method: 500 would say "broken" about something merely
        # unwritten.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ActionResponse(message=drill.instruction)


@router.get("/training/result", response_model=TrainingResultResponse)
async def training_result(request: Request) -> TrainingResultResponse:
    """The most recent drill result and running stats.

    Returns ``has_result=False`` rather than 404 before the first attempt --
    the panel polls this continuously, and a stream of 404s in the console makes
    real errors hard to spot.
    """
    state = _state(request)
    from modes.training import TrainingMode

    mode = state.mode_manager.mode
    if not isinstance(mode, TrainingMode) or mode.last_result is None:
        return TrainingResultResponse(has_result=False)

    result = mode.last_result
    return TrainingResultResponse(
        has_result=True,
        success=result.success,
        accuracy_pct=result.accuracy_pct,
        stars=result.stars,
        feedback=result.feedback,
        next_instruction=result.next_instruction,
        attempts=mode.stats.attempts,
        success_rate_pct=round(mode.stats.success_rate_pct, 1),
    )
