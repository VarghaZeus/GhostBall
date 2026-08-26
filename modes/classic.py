"""Classic mode: eight-ball with AR guidance.

Phase 7.2. Standard pool rules, standard scoring, and the trajectory overlay on
top. The baseline the other modes are variations of.

Rules implemented, and the ones deliberately not
------------------------------------------------
Implemented: an open table until the first legal pot, group assignment from
that pot, potting your own group keeps you at the table, a scratch or potting
the opponent's ball hands over the turn, and the 8 wins if your group is clear
and loses if it is not.

Not implemented: called pockets, ball-in-hand placement, two-shot carry, and
push-outs. Every one of those needs input the system does not have -- a player
declaring a pocket, or picking the cue ball up and putting it down. The rules
here are exactly the ones that can be *observed* from overhead, and the mode
says nothing about the ones it cannot see rather than guessing and being wrong
in front of somebody who knows the real rules.
"""

from __future__ import annotations

import logging

import numpy as np

from app.models import (
    Ball,
    GameModeName,
    GameSession,
    GameState,
    ModeOutput,
    Player,
    SessionState,
    ShotPrediction,
)
from modes.mode_manager import GameMode
from modes.rendering import highlight_ball
from modes.scoring import (
    FOUL_PENALTY,
    POINTS_PER_BALL,
    BallGroup,
    classify_shot,
    group_of,
    object_balls_on_table,
)

logger = logging.getLogger(__name__)


class ClassicMode(GameMode):
    """Eight-ball for one or two players, with the aiming line."""

    name = GameModeName.CLASSIC
    display_name = "Classic"
    is_competitive = True

    def __init__(self, settings: object | None = None) -> None:
        super().__init__(settings)  # type: ignore[arg-type]
        #: Group assigned to each player index. Empty until the first legal pot.
        self.groups: dict[int, BallGroup] = {}
        self.winner: Player | None = None
        self._feedback = "Break when ready"

    # -- lifecycle ----------------------------------------------------------

    def on_enter(self, session: GameSession) -> None:
        super().on_enter(session)
        self.renderer.reset()
        self.groups.clear()
        self.winner = None
        self._feedback = "Break when ready"
        if not session.players:
            # Solo practice with scoring. Better than refusing to run: somebody
            # who taps "Classic" without entering names wants to play, and an
            # empty scoreboard over a working aiming line is a fine answer.
            session.players = [Player(name="Player 1")]
            logger.info("classic started with no players; created a solo scoreboard")

    # -- rules --------------------------------------------------------------

    def group_for(self, player_index: int) -> BallGroup:
        """The group a player is on, or ``OPEN`` before the table is split."""
        return self.groups.get(player_index, BallGroup.OPEN)

    def _assign_groups(self, session: GameSession, potted: list[Ball]) -> None:
        """Split the table from the first legal pot.

        Uses the majority group of what went down, because a break can drop one
        of each and the alternative -- taking the first in an arbitrarily
        ordered list -- makes the assignment depend on detection order.
        """
        if self.groups or not potted:
            return
        solids = sum(1 for ball in potted if group_of(ball) is BallGroup.SOLIDS)
        stripes = sum(1 for ball in potted if group_of(ball) is BallGroup.STRIPES)
        if solids == stripes:
            # Genuinely ambiguous, so leave the table open rather than pick.
            logger.info("break potted %d of each; table stays open", solids)
            return

        mine = BallGroup.SOLIDS if solids > stripes else BallGroup.STRIPES
        index = session.current_player_index % max(1, len(session.players))
        self.groups[index] = mine
        for other in range(len(session.players)):
            if other != index:
                self.groups[other] = mine.opposing()
        logger.info("table split: player %d is on %s", index + 1, mine.value)

    def legal_target_ids(
        self, game_state: GameState, session: GameSession
    ) -> list[str] | None:
        """The balls the shooter is on: their group, the whole rack, or the 8.

        Three cases, in the order the rules create them. Before the table is
        split every object ball is fair game. After it, the shooter's own group.
        Once that group is off the table the 8 is the only legal ball -- and it
        matters that this narrows rather than falling back to everything, because
        a position that leaves a clear shot on the opponent's last stripe is not
        a good leave, it is a foul waiting to happen.
        """
        if not session.players:
            return None
        index = session.current_player_index % len(session.players)
        group = self.group_for(index)
        if group is BallGroup.OPEN:
            # Everything except the 8. An open table means either group is fair
            # game, not that the 8 is -- hitting it first is a foul at any point
            # before your own group is cleared, so a leave whose only shot is on
            # the 8 is not a good leave.
            return [b.id for b in object_balls_on_table(game_state)]
        if self._group_cleared(game_state, group):
            return [
                b.id
                for b in game_state.object_balls()
                if b.table_pos is not None and group_of(b) is BallGroup.EIGHT
            ]
        return [
            b.id
            for b in game_state.object_balls()
            if b.table_pos is not None and group_of(b) is group
        ]

    def _group_cleared(self, game_state: GameState, group: BallGroup) -> bool:
        """Whether a player's group is off the table and the 8 is legal for them.

        Judged from what is *visible*, which is the honest source: a ball hidden
        under a hand at the moment of the check would read as potted. The settle
        timer makes that unlikely, and the cost of being wrong here is offering
        the 8 a shot early rather than corrupting a score.
        """
        if not group.is_playable:
            return False
        return not any(group_of(ball) is group for ball in game_state.object_balls())

    def on_shot_complete(
        self, game_state: GameState, session: GameSession, pocketed: list[Ball]
    ) -> None:
        """Score the shot, assign groups, and decide whose turn it is next."""
        if self.winner is not None:
            return

        index = session.current_player_index % max(1, len(session.players))
        player = session.players[index]
        player.shots_taken += 1

        outcome = classify_shot(pocketed, self.group_for(index))

        if outcome.eight_potted:
            self._resolve_eight(game_state, session, player, index, outcome.scratched)
            return

        if not self.groups:
            self._assign_groups(session, outcome.own)

        if outcome.scored:
            player.shots_made += 1
            player.score += POINTS_PER_BALL * len(outcome.own)
            session.balls_pocketed_this_turn += len(outcome.own)
            session.combo_count += len(outcome.own)
            self._celebrate(outcome.own, POINTS_PER_BALL)

        if outcome.is_foul:
            self._penalise(session, index)
            self._feedback = "Scratch -- ball in hand" if outcome.scratched else "Wrong ball"
            session.advance_player()
        elif outcome.scored:
            # Potted legally: stay at the table. This is the rule that makes a
            # break run feel like one.
            self._feedback = f"+{POINTS_PER_BALL * len(outcome.own)}  shoot again"
        else:
            self._feedback = "Miss"
            session.advance_player()

        logger.info(
            "classic shot: %s (own=%d opponent=%d scratch=%s) -> %s",
            outcome.summary,
            len(outcome.own),
            len(outcome.opponent),
            outcome.scratched,
            session.players[session.current_player_index % len(session.players)].name,
        )

    def _resolve_eight(
        self,
        game_state: GameState,
        session: GameSession,
        player: Player,
        index: int,
        scratched: bool,
    ) -> None:
        """The 8 went down. Win if the group was clear and the cue ball stayed up."""
        cleared = self._group_cleared(game_state, self.group_for(index))
        if cleared and not scratched:
            self.winner = player
            player.score += POINTS_PER_BALL * 2
            self._feedback = f"{player.name} wins"
            session.state = SessionState.GAME_OVER
            logger.info("classic: %s wins on the 8", player.name)
            return

        # Early or on a scratch: the other player takes it.
        others = [p for i, p in enumerate(session.players) if i != index]
        self.winner = others[0] if others else player
        self._feedback = (
            f"8 ball early -- {self.winner.name} wins"
            if not scratched
            else f"8 and scratch -- {self.winner.name} wins"
        )
        session.state = SessionState.GAME_OVER
        logger.info("classic: %s loses on the 8 (cleared=%s scratch=%s)", player.name, cleared, scratched)

    def _penalise(self, session: GameSession, index: int) -> None:
        """Hand the foul penalty to everybody else, and break the combo."""
        for other, opponent in enumerate(session.players):
            if other != index:
                opponent.score += FOUL_PENALTY
        session.combo_count = 0

    def _celebrate(self, potted: list[Ball], points: int) -> None:
        """Spawn a score popup where each ball went down."""
        for ball in potted:
            if ball.table_pos is not None:
                self.renderer.effects.spawn_score(ball.table_pos, f"+{points}")

    # -- rendering ----------------------------------------------------------

    def update(
        self,
        game_state: GameState,
        prediction: ShotPrediction | None,
        session: GameSession,
    ) -> ModeOutput:
        """Aiming line, plus a ring on the balls this player may legally pot."""
        index = session.current_player_index % max(1, len(session.players))
        group = self.group_for(index)
        advice = ""
        if session.state is SessionState.AIMING:
            # Only while aiming. Scoring five resting places is the most
            # expensive thing this mode does per frame, and during a shot the
            # renderer draws no aiming line for it to annotate.
            advice = self.recommend_power(game_state, prediction, session)

        def decorate(canvas: np.ndarray, ctx: object) -> None:
            # Only while aiming: rings around every legal ball during a shot
            # would be six more things moving on the cloth.
            if session.state is not SessionState.AIMING or self.winner is not None:
                return
            for ball in game_state.object_balls():
                if ball.table_pos is None:
                    continue
                if group.is_playable and group_of(ball) is not group:
                    continue
                highlight_ball(canvas, ctx, ball.table_pos)  # type: ignore[arg-type]

        overlay = self.renderer.compose(
            game_state,
            prediction,
            session,
            self.mapper,
            feedback=advice or self._feedback,
            decorate=decorate,
        )
        return ModeOutput(
            overlay=overlay,
            feedback_text=self._feedback,
            next_action=self._next_action(session, group),
        )

    def _next_action(self, session: GameSession, group: BallGroup) -> str:
        """One line for the control panel, not the projector."""
        if self.winner is not None:
            return f"{self.winner.name} wins -- reset to play again"
        player = session.current_player
        who = player.name if player else "Player"
        if not group.is_playable:
            return f"{who}: table open"
        return f"{who}: on {group.value}"
