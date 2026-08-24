"""Freeplay mode: continuous shot prediction, no rules or scoring.

Phase 7.2. The simplest mode and therefore the one to bring up first -- it
exercises the whole pipeline (capture, detect, simulate, render, project)
without any game logic in the way. If freeplay looks right, everything else is
rules on top.
"""

from __future__ import annotations

import logging

from app.models import GameModeName, GameSession, GameState, ModeOutput, ShotPrediction
from modes.mode_manager import GameMode

logger = logging.getLogger(__name__)


class FreeplayMode(GameMode):
    """Show the predicted trajectory whenever someone is aiming.

    Behaviour by state:

    ``AIMING``
        Render the prediction. This is the whole mode.
    ``SHOT_IN_PROGRESS``
        Render nothing. The overlay must be off the felt while the balls are
        actually moving -- a stale aiming line under a rolling ball reads as a
        bug, and the spec's "zero interference with natural play" rules out
        projecting onto live play.
    ``SETTLING`` / ``IDLE``
        Nothing to draw.

    All four cases are :class:`~modes.rendering.ModeRenderer`'s defaults, so
    this mode is the composer with no arguments -- which is the point. Anything
    freeplay had to say for itself would be something the other modes would have
    to repeat.
    """

    name = GameModeName.FREEPLAY
    display_name = "Freeplay"
    is_competitive = False

    def on_enter(self, session: GameSession) -> None:
        super().on_enter(session)
        self.renderer.reset()

    def update(
        self,
        game_state: GameState,
        prediction: ShotPrediction | None,
        session: GameSession,
    ) -> ModeOutput:
        """Render the aiming overlay, or nothing.

        Returns an empty :class:`~app.models.ModeOutput` rather than raising
        when there is nothing to draw -- an un-aimed table is the normal resting
        case, not an error, and it happens for most of the frames in a session.
        """
        overlay = self.renderer.compose(game_state, prediction, session, self.mapper)
        return ModeOutput(overlay=overlay, next_action="Line up a shot")
