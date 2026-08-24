"""Game mode registry and the shot state machine.

Phase 7.1. The :class:`GameMode` base class and the ``idle -> aiming ->
shot_in_progress -> settling`` state machine are implemented, because they are
pure logic over a :class:`~app.models.GameState` and every mode depends on them
behaving identically. The individual modes are stubs.

The state machine is the piece most worth getting right and testing. Its job is
to answer one question -- "has a shot been taken?" -- from noisy per-frame
detections, and every mode's scoring hangs off that answer. Two failure modes it
is explicitly built to avoid:

* **Flapping** between aiming and shot_in_progress when the cue is detected
  intermittently. Hence the confirmation frame counts rather than instant
  transitions.
* **Ending a shot early** during the moment when every ball happens to be moving
  slowly. Hence the settle timer, which requires stillness to persist.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from app.config import Settings, get_settings
from app.models import (
    Ball,
    CueStick,
    GameModeName,
    GameSession,
    GameState,
    ModeOutput,
    Player,
    SessionState,
    ShotPrediction,
)
from modes.rendering import ModeRenderer

logger = logging.getLogger(__name__)

#: Consecutive frames a condition must hold before the state changes. At 30 FPS
#: two frames is 66 ms -- enough to reject a single-frame detection glitch,
#: short enough to feel instant.
CONFIRM_FRAMES = 2

#: Seconds the table must stay still before a shot counts as finished. Balls
#: creeping to a stop can dip below the movement threshold and rise back above
#: it, so a single still frame is not sufficient evidence.
SETTLE_SECONDS = 1.0


class GameMode(ABC):
    """Base class for all game modes.

    A mode is handed the current observed state and the shot prediction, and
    returns what to draw plus any feedback. It does *not* own the state machine
    -- :class:`ModeManager` does -- so that scoring rules and shot detection can
    be tested and changed independently.
    """

    name: GameModeName = GameModeName.FREEPLAY
    display_name: str = "Freeplay"
    #: Whether the mode needs turn tracking and a scoreboard. Freeplay does not.
    is_competitive: bool = False

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        #: Table -> projector transform, injected by :class:`ModeManager`.
        #: ``None`` until the manager is wired up, which is the normal state in
        #: unit tests -- every mode must return an empty output rather than
        #: raise when it cannot draw.
        self.mapper: object | None = None
        #: Overlay scaffolding: reused canvas, trajectory smoother and the
        #: effect system. One per mode instance, so switching modes drops the
        #: previous mode's effects with it.
        self.renderer = ModeRenderer(self.settings)

    def on_enter(self, session: GameSession) -> None:
        """Called once when the mode becomes active. Set up mode-specific state."""
        logger.info("entering mode %s", self.display_name)

    def on_exit(self, session: GameSession) -> None:
        """Called once when leaving the mode. Release anything mode-specific."""
        logger.info("leaving mode %s", self.display_name)

    @abstractmethod
    def update(
        self,
        game_state: GameState,
        prediction: ShotPrediction | None,
        session: GameSession,
    ) -> ModeOutput:
        """Produce this frame's overlay and feedback.

        Must be cheap and must not block -- it runs inside the frame budget.
        Anything expensive belongs in :meth:`on_shot_complete`, which fires once
        per shot rather than 30 times a second.
        """

    def on_shot_complete(  # noqa: B027 - intentional no-op, not an abstract method
        self, game_state: GameState, session: GameSession, pocketed: list[Ball]
    ) -> None:
        """Called once when the table settles after a shot.

        This is where scoring, turn advancement and foul detection belong. The
        default implementation does nothing, which is correct for freeplay.

        ``pocketed`` holds the *balls* as they were last seen, not just their
        ids, because every scoring rule needs to know what went down: a solid,
        a stripe, the 8, or the cue ball. By the time this fires those balls are
        off the table and out of ``game_state``, so the manager snapshots them
        at the strike -- see :meth:`ModeManager._newly_pocketed`.
        """


class ModeManager:
    """Owns the active mode and the shot state machine."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.session = GameSession(mode=self.settings.system.startup_mode)
        self._mode: GameMode | None = None
        self._confirm_count = 0
        self._still_since: float | None = None
        #: Balls on the table at the moment of the strike, by id. Kept as whole
        #: objects rather than ids so that scoring can ask what a pocketed ball
        #: *was* after it has left the frame.
        self._balls_before_shot: dict[str, Ball] = {}
        #: Ball tracker supplying frame-to-frame motion, injected by the vision
        #: loop. The manager does not construct one: detection is what populates
        #: it, so ownership belongs with the loop that runs detection.
        self.tracker: object | None = None
        self._mapper: object | None = None
        self.load_mode(self.settings.system.startup_mode)

    @property
    def mapper(self) -> object | None:
        """Table -> projector transform, injected by the vision loop.

        Set on the manager and propagated to whichever mode is active, so a mode
        loaded later still gets one. Modes cannot build their own: the loop owns
        the calibration and it can be replaced at runtime by the wizard.
        """
        return self._mapper

    @mapper.setter
    def mapper(self, value: object | None) -> None:
        self._mapper = value
        if self._mode is not None:
            self._mode.mapper = value

    # -- mode registry ------------------------------------------------------

    def load_mode(self, name: GameModeName | str) -> GameMode:
        """Activate a mode by name.

        Unknown or unimplemented modes fall back to freeplay with a warning
        rather than raising -- a bad value from the API should not take the game
        down mid-session. Callers that need to *tell the user* the mode they
        asked for does not exist should check :func:`mode_registry` first, or
        compare ``session.mode`` against what they requested; the fallback is
        silent by design at this layer.
        """
        mode_name = GameModeName(name) if isinstance(name, str) else name
        mode_cls = mode_registry().get(mode_name)
        if mode_cls is None:
            from modes.freeplay import FreeplayMode

            logger.warning("mode %s is not implemented yet; staying in freeplay", mode_name.value)
            mode_cls = FreeplayMode
            mode_name = GameModeName.FREEPLAY

        if self._mode is not None:
            self._mode.on_exit(self.session)

        self._mode = mode_cls(self.settings)
        self._mode.mapper = self._mapper
        self.session.mode = mode_name
        self.session.state = SessionState.IDLE
        self.reset_state_machine()
        self._mode.on_enter(self.session)
        return self._mode

    @property
    def mode(self) -> GameMode:
        assert self._mode is not None, "ModeManager always holds a mode"
        return self._mode

    def reset_state_machine(self) -> None:
        """Clear transition bookkeeping. Called on mode change and on reset."""
        self._confirm_count = 0
        self._still_since = None
        self._balls_before_shot = {}

    # -- state machine ------------------------------------------------------

    def update_state(self, game_state: GameState, now: float | None = None) -> SessionState:
        """Advance the shot state machine one frame and return the new state.

        Deliberately separate from :meth:`update` so it can be tested against a
        synthetic sequence of ``GameState`` objects with no rendering involved.

        Args:
            game_state: This frame's observations.
            now: Injectable clock for tests. Defaults to ``time.perf_counter()``.
        """
        now = time.perf_counter() if now is None else now
        state = self.session.state
        cue = game_state.cue_stick
        balls_moving = self._any_ball_moving(game_state)

        if state is SessionState.GAME_OVER:
            # Terminal until someone resets. Without this the machine walks
            # straight back to idle and the winning overlay disappears a frame
            # after it arrives.
            return state

        if state is SessionState.IDLE:
            if cue is not None and cue.visible:
                self._confirm_count += 1
                if self._confirm_count >= CONFIRM_FRAMES:
                    self._transition(SessionState.AIMING)
            else:
                self._confirm_count = 0

        elif state is SessionState.AIMING:
            if self._is_strike(cue) or balls_moving:
                # Snapshot the ball set now: comparing against it after the shot
                # is how pocketed balls are identified, and it must be taken
                # before anything leaves the table.
                self._balls_before_shot = {
                    b.id: b for b in game_state.balls if not b.pocketed
                }
                self._transition(SessionState.SHOT_IN_PROGRESS)
            elif cue is None or not cue.visible:
                self._confirm_count += 1
                if self._confirm_count >= CONFIRM_FRAMES * 3:
                    # The player walked away rather than shot. A longer
                    # confirmation than usual, since dropping out of aiming
                    # yanks the overlay away mid-aim if it fires too eagerly.
                    self._transition(SessionState.IDLE)
            else:
                self._confirm_count = 0

        elif state is SessionState.SHOT_IN_PROGRESS:
            if balls_moving:
                self._still_since = None
            else:
                if self._still_since is None:
                    self._still_since = now
                elif now - self._still_since >= SETTLE_SECONDS:
                    self._transition(SessionState.SETTLING)

        elif state is SessionState.SETTLING:
            # Terminal-per-shot: scoring runs, then the mode decides where next.
            pocketed = self._newly_pocketed(game_state)
            self.mode.on_shot_complete(game_state, self.session, pocketed)
            # The hook may have ended the game -- an 8 ball, or a target score
            # reached. Only move on if it did not: transitioning unconditionally
            # overwrote GAME_OVER with IDLE and the win was never visible.
            if self.session.state is SessionState.SETTLING:
                self._transition(SessionState.IDLE)

        return self.session.state

    def _transition(self, new_state: SessionState) -> None:
        if new_state is not self.session.state:
            logger.debug("state %s -> %s", self.session.state.value, new_state.value)
            self.session.state = new_state
        self._confirm_count = 0
        if new_state is not SessionState.SHOT_IN_PROGRESS:
            self._still_since = None

    def _is_strike(self, cue: CueStick | None) -> bool:
        """Whether cue-tip speed indicates the ball has been struck."""
        if cue is None:
            return False
        return cue.velocity >= self.settings.vision.strike_velocity_threshold

    def _any_ball_moving(self, game_state: GameState) -> bool:
        """Whether any ball is in motion.

        Delegates to the :class:`~vision.detection.BallTracker`, which owns the
        frame-to-frame deltas. Reports ``False`` when no tracker has been
        attached, which keeps the state machine at rest rather than firing
        spurious shot events -- the safe direction to fail, since a missed shot
        is recoverable and a phantom one corrupts the score.
        """
        if self.tracker is None:
            return False
        return self.tracker.any_moving(self.settings.vision.ball_stopped_threshold)

    def _newly_pocketed(self, game_state: GameState) -> list[Ball]:
        """Balls present before the shot and absent now, as they last looked.

        Set difference rather than trusting a ``pocketed`` flag, because a ball
        occluded by a hand also goes missing, and the settle timer is what makes
        this reliable -- by the time it fires, hands are usually clear.

        Returns the snapshot objects, not anything from ``game_state``: the
        whole point is that these balls are no longer in it.
        """
        current = {b.id for b in game_state.balls if not b.pocketed}
        return [
            ball
            for ball_id, ball in sorted(self._balls_before_shot.items())
            if ball_id not in current
        ]

    # -- per-frame entry point ---------------------------------------------

    def update(
        self, game_state: GameState, prediction: ShotPrediction | None
    ) -> ModeOutput:
        """Advance the state machine, then let the active mode render.

        This is what ``app.main`` calls once per frame.
        """
        self.update_state(game_state)
        output = self.mode.update(game_state, prediction, self.session)
        output.session = self.session
        return output

    # -- session control ----------------------------------------------------

    def start_game(self, player_names: list[str]) -> GameSession:
        """Begin a new competitive game."""
        self.session.players = [Player(name=n) for n in player_names]
        self.session.current_player_index = 0
        self.session.started_at = time.perf_counter()
        self.session.state = SessionState.IDLE
        self.reset_state_machine()
        logger.info("started %s with %s", self.session.mode.value, ", ".join(player_names))
        return self.session

    def reset(self) -> GameSession:
        """Clear game state, keeping the active mode."""
        self.session = GameSession(mode=self.session.mode)
        self.reset_state_machine()
        logger.info("session reset (mode %s)", self.session.mode.value)
        return self.session


def mode_registry() -> dict[GameModeName, type[GameMode]]:
    """Which modes actually exist, by name.

    Imports are inside the function to avoid a circular import: the mode modules
    import :class:`GameMode` from here.

    A function rather than a module constant so that the control panel can ask
    what is available and grey out the rest. Offering a button that silently
    falls back to freeplay is worse than offering no button -- the user taps it,
    sees a success message, and gets a different mode.
    """
    from modes.classic import ClassicMode
    from modes.freeplay import FreeplayMode
    from modes.king_of_the_hill import KingOfTheHillMode
    from modes.training import TrainingMode
    from modes.trick_shots import TrickShotsMode

    return {
        GameModeName.FREEPLAY: FreeplayMode,
        GameModeName.CLASSIC: ClassicMode,
        GameModeName.KING_OF_THE_HILL: KingOfTheHillMode,
        GameModeName.TRICK_SHOTS: TrickShotsMode,
        GameModeName.TRAINING: TrainingMode,
        # `knockout` is specified in ar_pool_games_and_animations.md and is not
        # built: it is King of the Hill's turn machinery with a bracket on top,
        # and shipping four modes that work beats five where one is a sketch.
    }


def implemented_modes() -> list[GameModeName]:
    """Names of the modes that can actually be loaded."""
    return list(mode_registry())


def load_mode(name: GameModeName | str, settings: Settings | None = None) -> GameMode:
    """Instantiate a mode without a manager. Mainly for tests."""
    return ModeManager(settings).load_mode(name)
