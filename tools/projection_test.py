"""Projector alignment, animation preview and render profiling.

The counterpart to ``tools.camera_preview``: that one answers "is the camera
working?", this one answers "is the projector aligned, and can we afford to draw
this?". Both are meant to be the first thing run on new hardware, before any
game logic exists to blame.

Three jobs, and the reason they are one tool is that they all need the same
scaffolding -- settings, a mapper, a display and a canvas -- and separating them
would triplicate it:

**Alignment.** ``--pattern`` projects a static test pattern. This is what you
run while physically moving the projector.

**Animation.** ``--demo`` animates a real simulated shot, or the effect system,
or the game UI. Watching an aiming line crawl on the actual felt is the only way
to judge whether the dash rhythm and the smoothing feel right; both are
subjective and neither survives being decided from a still image.

**Profiling.** ``--profile`` times every render function at the configured
resolution and prints the frame-budget arithmetic. The render stage shares a
33 ms frame with capture, detection and physics, so the useful output is not
milliseconds but *what fraction of the budget is left*.

Works headless: with ``--headless`` the frames are written as PNGs to
``data/snapshots`` instead of going to a projector, which is how this gets used
over SSH on a Pi with nothing plugged into HDMI.

Usage::

    python -m tools.projection_test --pattern grid            # align the projector
    python -m tools.projection_test --pattern corners --theme neon
    python -m tools.projection_test --demo trajectory         # animated aiming line
    python -m tools.projection_test --demo effects            # trails, bursts, pots
    python -m tools.projection_test --demo ui                 # scoreboard and timer
    python -m tools.projection_test --profile                 # timing table, no output
    python -m tools.projection_test --pattern grid --headless  # write a PNG instead

Keys (windowed mode) are handled by the display's own OpenCV window; use Ctrl-C
to stop, or pass ``--seconds``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, Settings, load_settings  # noqa: E402
from app.models import (  # noqa: E402
    Ball,
    BallColor,
    BallKind,
    GameModeName,
    GameSession,
    GameState,
    Player,
    SessionState,
    Vec2,
)
from projection import draw  # noqa: E402
from projection.display import Display  # noqa: E402
from projection.effects import EffectContext, EffectSystem  # noqa: E402
from projection.mapper import ProjectionMapper, identity_calibration, load_calibration  # noqa: E402
from projection.patterns import TestPattern, render_test_pattern  # noqa: E402
from projection.renderer import (  # noqa: E402
    TrajectorySmoother,
    render_ball_trails,
    render_game_ui,
    render_trajectory_overlay,
)
from projection.themes import resolve_theme, theme_names  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.performance import PerformanceTracker, RateLimiter  # noqa: E402

logger = logging.getLogger("projection_test")

SNAPSHOT_DIR = DATA_DIR / "snapshots"

DEMOS = ("trajectory", "effects", "ui")


# ---------------------------------------------------------------------------
# Scene fixtures
# ---------------------------------------------------------------------------


def demo_game_state(settings: Settings, timestamp: float = 0.0) -> GameState:
    """A plausible mid-game table: cue ball plus four object balls.

    Positions are hand-picked rather than random so the same run twice looks the
    same -- a profiling number that moves because the balls moved is not a
    profiling number. The spread is deliberately awkward (a cut onto a ball near
    a cushion, with two others in the path) because that is the case where the
    overlay gets crowded and the layout has to hold up.
    """
    length, width = settings.table.length_in, settings.table.width_in
    layout = [
        ("cue", BallColor.WHITE, BallKind.CUE, None, Vec2(length * 0.25, width * 0.5)),
        ("ball_03", BallColor.RED, BallKind.SOLID, 3, Vec2(length * 0.62, width * 0.34)),
        ("ball_09", BallColor.YELLOW, BallKind.STRIPE, 9, Vec2(length * 0.74, width * 0.62)),
        ("ball_08", BallColor.BLACK, BallKind.EIGHT, 8, Vec2(length * 0.88, width * 0.20)),
        ("ball_02", BallColor.BLUE, BallKind.SOLID, 2, Vec2(length * 0.55, width * 0.78)),
    ]
    balls = [
        Ball(
            id=ball_id,
            center_px=Vec2(0.0, 0.0),
            radius_px=13.0,
            color=color,
            kind=kind,
            number=number,
            table_pos=position,
            confidence=0.9,
        )
        for ball_id, color, kind, number, position in layout
    ]
    return GameState(
        timestamp=timestamp,
        frame_index=0,
        balls=balls,
        cue_ball=balls[0],
        confidence=0.9,
    )


def demo_session() -> GameSession:
    """A three-player session with a combo running, for the UI demo."""
    return GameSession(
        mode=GameModeName.KING_OF_THE_HILL,
        state=SessionState.AIMING,
        players=[Player("ALICE", 42, 12, 7), Player("BOB", 31, 10, 4), Player("CY", 55, 14, 9)],
        current_player_index=2,
        combo_count=3,
    )


def demo_prediction(settings: Settings, game_state: GameState, angle_deg: float):
    """Simulate a shot from the demo layout at a given aim angle.

    The fan, so the projected demo shows what the real overlay shows -- power
    ticks and a split aim/consequence path. Previously this passed
    ``default_power``, which free-rolls the cue ball some thirteen table lengths
    and made the demo trajectory wrap the table several times over.
    """
    from physics.simulator import simulate_shot_fan

    assert game_state.cue_ball is not None and game_state.cue_ball.table_pos is not None
    return simulate_shot_fan(
        game_state.cue_ball.table_pos,
        angle_deg,
        game_state.object_balls(),
        settings,
    )


def build_mapper(settings: Settings, use_saved: bool) -> ProjectionMapper:
    """A mapper for the run, from the saved calibration or the identity fallback."""
    if use_saved:
        calibration = load_calibration()
        if calibration is not None:
            logger.info(
                "using saved calibration (RMSE %.2f px, calibrated=%s)",
                calibration.rmse_px,
                calibration.is_calibrated,
            )
            return ProjectionMapper(calibration)
        logger.warning("no saved calibration found; falling back to identity")
    else:
        logger.info("ignoring any saved calibration (--identity)")
    return ProjectionMapper(identity_calibration(settings))


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def profile(settings: Settings, mapper: ProjectionMapper, iterations: int = 40) -> int:
    """Time every render path and report against the frame budget.

    Reports the median as well as the mean. The mean is what a sustained frame
    rate depends on, but a render that is usually 4 ms and occasionally 30 ms
    drops a frame every time it spikes, and only the spread shows that.

    Each case gets its own reused canvas, so the numbers are the cost of
    *drawing* -- allocation is measured separately as its own row, because it is
    avoidable and worth knowing the size of.
    """
    theme = resolve_theme(settings)
    game_state = demo_game_state(settings)
    prediction = demo_prediction(settings, game_state, -12.0)
    session = demo_session()
    smoother = TrajectorySmoother(settings)

    effects = EffectSystem(settings)
    ctx = EffectContext.build(mapper, settings, theme)
    # A realistic effect load: a pot in progress, three collisions and a popup.
    effects.spawn_pocket(Vec2(settings.table.length_in / 2.0, 0.0), points=10, now=0.0)
    for i in range(3):
        effects.spawn_collision(Vec2(30.0 + i * 8, 12.0 + i * 5), outgoing_deg=30.0 * i, now=0.0)
    for i in range(24):
        # Wind trails forward so the trail draw cost is the steady-state one.
        moving = demo_game_state(settings, timestamp=i / 30.0)
        assert moving.cue_ball is not None
        moving.cue_ball.table_pos = Vec2(20.0 + i * 1.5, 19.0 + i * 0.4)
        effects.observe(moving, now=i / 30.0)

    canvas = draw.new_canvas(settings)
    cases: list[tuple[str, object]] = [
        ("canvas clear", lambda t: draw.reset_canvas(canvas)),
        (
            "trajectory overlay",
            lambda t: render_trajectory_overlay(
                prediction, game_state, mapper, settings, canvas, theme=theme, smoother=smoother, now=t
            ),
        ),
        (
            "game ui + effects",
            lambda t: render_game_ui(
                session, mapper, settings, canvas, theme=theme, effects=effects,
                seconds_remaining=8.5, feedback_text="NICE SHOT", now=t,
            ),
        ),
        ("ball trails", lambda t: effects.render(draw.reset_canvas(canvas), ctx, now=t)),
        (
            "test pattern (grid)",
            lambda t: render_test_pattern(
                TestPattern.GRID, mapper, settings, canvas=draw.reset_canvas(canvas), theme=theme
            ),
        ),
        ("new_canvas (alloc)", lambda t: draw.new_canvas(settings)),
    ]

    budget_ms = 1000.0 / settings.system.target_fps
    print(
        f"\nrender profile  {settings.projector.width}x{settings.projector.height}"
        f"  theme={theme.name}  glow={theme.glow}"
        f"  budget={budget_ms:.1f} ms/frame at {settings.system.target_fps} FPS\n"
    )
    print(f"{'stage':<24}{'mean':>9}{'median':>9}{'max':>9}{'% budget':>10}")
    print("-" * 61)

    total_mean = 0.0
    for label, call in cases:
        call(0.0)  # warm up: first call pays for OpenCV's lazy internals
        samples = []
        for i in range(iterations):
            start = time.perf_counter()
            call(i / 30.0)
            samples.append((time.perf_counter() - start) * 1000.0)
        samples.sort()
        mean = sum(samples) / len(samples)
        median = samples[len(samples) // 2]
        if label != "new_canvas (alloc)":
            total_mean += mean
        print(
            f"{label:<24}{mean:>8.2f}ms{median:>8.2f}ms{samples[-1]:>8.2f}ms"
            f"{100.0 * mean / budget_ms:>9.1f}%"
        )

    print("-" * 61)
    # The sum is an upper bound, not a prediction: no single frame runs every
    # one of these. A frame draws a trajectory *or* trails, plus the UI.
    print(f"{'sum of all stages':<24}{total_mean:>8.2f}ms{'':>18}{100.0 * total_mean / budget_ms:>9.1f}%")
    print(
        "\nNo frame runs every stage: a typical frame is one overlay plus the UI.\n"
        "If the render stage is over budget, the order to try is: turn off the\n"
        "theme's glow, drop show_impact_angles, then lower the projector\n"
        "resolution -- cost here scales with pixel count, not with scene detail."
    )
    return 0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_frame(overlay: np.ndarray, label: str) -> Path:
    """Write an overlay to ``data/snapshots`` as a PNG.

    PNG rather than JPEG, unlike ``camera_preview``: this is line art on a
    transparent background, and JPEG would both lose the alpha channel and put
    ringing artefacts around every thin bright line -- exactly the thing being
    inspected.
    """
    import cv2

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = SNAPSHOT_DIR / f"projection-{label}-{stamp}.png"
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
    logger.info("wrote %s", path)
    return path


def run_pattern(
    settings: Settings,
    mapper: ProjectionMapper,
    pattern: TestPattern,
    display: Display | None,
    seconds: float | None,
) -> int:
    """Project one static test pattern, or write it to disk."""
    theme = resolve_theme(settings)
    overlay = render_test_pattern(pattern, mapper, settings, theme=theme)

    if display is None:
        save_frame(overlay, pattern.value)
        return 0

    # Re-sent every frame rather than once. An OpenCV full-screen window that is
    # not being fed stops repainting and appears frozen, and on some window
    # managers is dropped entirely -- so a "static" pattern still needs a loop.
    limiter = RateLimiter(settings.system.target_fps)
    deadline = None if seconds is None else time.perf_counter() + seconds
    logger.info("projecting %s pattern; Ctrl-C to stop", pattern.value)
    while deadline is None or time.perf_counter() < deadline:
        display.send_frame(overlay)
        limiter.sleep()
    return 0


def run_demo(
    settings: Settings,
    mapper: ProjectionMapper,
    demo: str,
    display: Display | None,
    seconds: float | None,
) -> int:
    """Animate one of the render paths."""
    theme = resolve_theme(settings)
    tracker = PerformanceTracker(target_fps=settings.system.target_fps)
    game_state = demo_game_state(settings)
    session = demo_session()
    smoother = TrajectorySmoother(settings)
    effects = EffectSystem(settings)
    canvas = draw.new_canvas(settings)

    start = time.perf_counter()
    deadline = None if seconds is None else start + seconds
    limiter = RateLimiter(settings.system.target_fps)
    frames = 0
    logger.info("running %s demo; Ctrl-C to stop", demo)

    while deadline is None or time.perf_counter() < deadline:
        now = time.perf_counter()
        elapsed = now - start
        tracker.begin_frame()

        with tracker.stage("render"):
            if demo == "trajectory":
                # Sweep the aim through a 40 degree arc. A sweep rather than a
                # fixed angle because the thing being judged is how the line
                # behaves *while moving* -- smoothing and dash crawl both look
                # fine on a still frame.
                angle = -20.0 + 20.0 * math.sin(elapsed * 0.6)
                prediction = demo_prediction(settings, game_state, angle)
                canvas = render_trajectory_overlay(
                    prediction, game_state, mapper, settings, canvas,
                    theme=theme, smoother=smoother, now=now,
                )
            elif demo == "effects":
                canvas = _animate_effects(settings, mapper, effects, game_state, canvas, theme, elapsed, now)
            else:
                session.combo_count = 2 + int(elapsed / 3.0) % 3
                canvas = render_game_ui(
                    session, mapper, settings, canvas, theme=theme, effects=effects,
                    seconds_remaining=max(0.0, 30.0 - elapsed), feedback_text="NICE SHOT", now=now,
                )

        with tracker.stage("project"):
            if display is not None:
                display.send_frame(canvas)

        tracker.end_frame()
        frames += 1
        if display is None and frames == 30:
            # Headless: one representative frame a second in, by which point the
            # trails have filled and the effects are mid-flight.
            save_frame(canvas, demo)
            break
        limiter.sleep()

    tracker.log_summary(settings.system.latency_warn_ms)
    snapshot = tracker.snapshot()
    print(
        f"\n{demo}: {frames} frames, {snapshot.fps:.1f} FPS achievable, "
        f"render {snapshot.stage_ms.get('render', 0.0):.2f} ms/frame "
        f"(p95 frame {snapshot.frame_ms_p95:.2f} ms)"
    )
    return 0


def _animate_effects(
    settings: Settings,
    mapper: ProjectionMapper,
    effects: EffectSystem,
    game_state: GameState,
    canvas: np.ndarray,
    theme,
    elapsed: float,
    now: float,
) -> np.ndarray:
    """Drive a ball across the table so trails, bursts and pots all fire.

    The ball is walked along a path that bounces off the cushions, which makes
    the auto-detected collision bursts fire from real direction changes rather
    than being spawned on a timer -- so this exercises
    :meth:`~projection.effects.EffectSystem.observe` and not just the drawing.
    Every four seconds it is dropped into a pocket to trigger the celebration.
    """
    length, width = settings.table.length_in, settings.table.width_in
    assert game_state.cue_ball is not None

    # Triangle waves: linear travel with a sharp turn at each cushion.
    period_x, period_y = 5.0, 3.1
    tx = (elapsed % period_x) / period_x
    ty = (elapsed % period_y) / period_y
    x = length * (2.0 * tx if tx < 0.5 else 2.0 * (1.0 - tx))
    y = width * (2.0 * ty if ty < 0.5 else 2.0 * (1.0 - ty))
    game_state.cue_ball.table_pos = Vec2(
        min(max(x, 2.0), length - 2.0), min(max(y, 2.0), width - 2.0)
    )
    game_state.timestamp = now

    if int(elapsed) % 4 == 0 and elapsed % 4.0 < 0.05:
        effects.spawn_pocket(Vec2(length / 2.0, 0.0), points=10, now=now)

    return render_ball_trails(game_state, effects, mapper, settings, canvas, theme=theme, now=now)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Projector test patterns, animation previews and render profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    parser.add_argument(
        "--pattern",
        choices=[p.value for p in TestPattern],
        help="project a static alignment pattern",
    )
    parser.add_argument("--demo", choices=DEMOS, help="animate a render path")
    parser.add_argument(
        "--profile", action="store_true", help="time every render function and exit"
    )
    parser.add_argument("--theme", choices=theme_names(), help="override the configured theme")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="ignore the saved calibration and stretch the table to the frame",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="write a PNG to data/snapshots instead of opening the projector",
    )
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    parser.add_argument(
        "--iterations", type=int, default=40, help="samples per stage when profiling"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    setup_logging(args.log_level, log_to_file=False)

    if args.theme:
        settings.render.theme = args.theme
    if args.headless:
        settings.projector.use_mock = True

    if not (args.pattern or args.demo or args.profile):
        # Defaulting to the grid rather than printing usage: someone running this
        # with no flags is standing at a table trying to line up a projector, and
        # the grid is what they want.
        logger.info("no mode given; showing the grid pattern (see --help)")
        args.pattern = TestPattern.GRID.value

    mapper = build_mapper(settings, use_saved=not args.identity)

    if args.profile:
        return profile(settings, mapper, iterations=args.iterations)

    display: Display | None = None
    try:
        if not args.headless:
            display = Display(settings.projector).open()
            logger.info("display backend: %s", display.backend_name)
        if args.pattern:
            return run_pattern(settings, mapper, TestPattern(args.pattern), display, args.seconds)
        return run_demo(settings, mapper, args.demo, display, args.seconds)
    except KeyboardInterrupt:
        logger.info("interrupted")
        return 0
    finally:
        if display is not None:
            # Leave the felt clean; a frozen overlay is worse than nothing.
            display.clear()
            display.close()


if __name__ == "__main__":
    raise SystemExit(main())
