"""King of the Hill: keep potting, keep the table.

Phase 7.2. Turn-based arcade scoring for two to four players. Pot a ball and you
stay at the table with time added; miss and the next player is up with a fresh
clock. First to :data:`TARGET_SCORE` wins.

The clock is the mode
---------------------
Everything here hangs off a countdown, which makes this the one mode whose state
can change without a shot being taken -- a turn can simply run out. That is
handled in :meth:`KingOfTheHillMode.update`, the only per-frame hook a mode has,
rather than in a timer thread: the vision loop is already a clock ticking thirty
times a second, and a second one would need locks around the session for no
benefit.

Difficulty, and what it can honestly change
--------------------------------------------
The spec asks difficulty to change "number of balls on table or pocket
complexity". The system cannot remove balls from a real table, so it changes
which ball it *asks* for: easy nominates the ball with the straightest pot
available, hard nominates the hardest one on the table, and the points follow.
That is a real difficulty knob -- the shot genuinely gets harder -- and it needs
no cooperation from the furniture.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.config import BALL_RADIUS_IN
from app.models import (
    Ball,
    GameModeName,
    GameSession,
    GameState,
    ModeOutput,
    Player,
    SessionState,
    ShotPrediction,
    Vec2,
)
from modes.mode_manager import GameMode
from modes.rendering import highlight_ball
from modes.scoring import BallGroup, group_of, nearest_pocket, object_balls_on_table
from physics.models import TableGeometry

logger = logging.getLogger(__name__)

#: Points to win.
TARGET_SCORE = 100

#: Seconds per turn, per the spec.
TURN_SECONDS = 90.0

#: Seconds added for a successful pot. Rewards a run without letting one player
#: hold the table indefinitely -- the bonus is well under the time a shot takes.
TIME_BONUS_SECONDS = 5.0

#: Base points for a pot, before the difficulty and combo multipliers.
BASE_POINTS = 10

#: Combo multiplier is capped here. Uncapped, a lucky run on an easy layout puts
#: the game out of reach on one turn, and the mode stops being a contest.
MAX_COMBO_MULTIPLIER = 5


class Difficulty(str, Enum):
    """How hard a shot the mode asks for, and what it pays."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @property
    def multiplier(self) -> float:
        """Score multiplier. Harder targets are worth more, or nobody picks them."""
        return {"easy": 1.0, "medium": 1.5, "hard": 2.5}[self.value]

    @property
    def rank(self) -> float:
        """Where in the sorted difficulty list to take the target from.

        0.0 takes the easiest available pot, 1.0 the hardest. Medium takes the
        middle rather than a fixed index, so it stays medium whether there are
        three balls on the table or thirteen.
        """
        return {"easy": 0.0, "medium": 0.5, "hard": 1.0}[self.value]


@dataclass(slots=True)
class ShotDifficulty:
    """How hard one particular pot is, and why.

    Kept as its components rather than a single number so the reason is
    inspectable: a long straight pot and a short thin cut can score the same and
    are different shots to be asked for.
    """

    ball: Ball
    pocket: Vec2
    pocket_index: int
    #: Inches the object ball must travel.
    distance_in: float
    #: Degrees between the cue-to-ball line and the ball-to-pocket line. Zero is
    #: a straight pot; past about 70 the shot is close to impossible.
    cut_angle_deg: float

    @property
    def score(self) -> float:
        """A single ordering value, higher is harder.

        Cut angle dominates distance, weighted roughly 2:1, which matches how
        players talk about pots: a three-foot straight-in is routine and a
        one-foot cut at seventy degrees is not.
        """
        return self.cut_angle_deg / 45.0 * 2.0 + self.distance_in / 40.0


class KingOfTheHillMode(GameMode):
    """Turn-based, timed, combo-scored table control."""

    name = GameModeName.KING_OF_THE_HILL
    display_name = "King of the Hill"
    is_competitive = True

    def __init__(self, settings: object | None = None) -> None:
        super().__init__(settings)  # type: ignore[arg-type]
        self.difficulty = Difficulty.MEDIUM
        self.winner: Player | None = None
        self.target: ShotDifficulty | None = None
        self._turn_ends_at: float = 0.0
        self._feedback = "Pot a ball to stay at the table"
        self._geometry = TableGeometry.from_settings(self.settings)

    # -- lifecycle ----------------------------------------------------------

    def on_enter(self, session: GameSession) -> None:
        super().on_enter(session)
        self.renderer.reset()
        self.winner = None
        self.target = None
        if len(session.players) < 2:
            # The mode is a contest for the table; one player has already won
            # it. Two placeholders beat refusing to start.
            existing = [p.name for p in session.players]
            names = existing + [f"Player {i}" for i in range(len(existing) + 1, 3)]
            session.players = [Player(name=n) for n in names]
            logger.info("king of the hill needs 2+ players; created %s", ", ".join(names))
        self.start_turn(session)

    def set_difficulty(self, difficulty: Difficulty | str) -> Difficulty:
        """Change the difficulty. Takes effect on the next target selection."""
        self.difficulty = Difficulty(difficulty)
        logger.info("king of the hill difficulty: %s", self.difficulty.value)
        return self.difficulty

    # -- turns --------------------------------------------------------------

    def start_turn(self, session: GameSession, now: float | None = None) -> None:
        """Reset the clock and the combo for whoever is up."""
        now = time.perf_counter() if now is None else now
        self._turn_ends_at = now + TURN_SECONDS
        session.combo_count = 0
        session.balls_pocketed_this_turn = 0
        player = session.current_player
        self._feedback = f"{player.name if player else 'Player'} -- {TURN_SECONDS:.0f}s"

    def seconds_remaining(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        return max(0.0, self._turn_ends_at - now)

    def _end_turn(self, session: GameSession, reason: str, now: float | None = None) -> None:
        """Hand over to the next player."""
        session.advance_player()
        self.start_turn(session, now)
        player = session.current_player
        self._feedback = f"{reason} -- {player.name if player else 'next'} up"

    # -- target selection ---------------------------------------------------

    def choose_target(self, game_state: GameState) -> ShotDifficulty | None:
        """Nominate the ball to pot, at the current difficulty.

        Ranks every (ball, pocket) pair by how hard the pot is and picks by
        :attr:`Difficulty.rank`. Pairs whose path is blocked by another ball are
        dropped first -- asking for a pot that cannot be made is the one thing
        that would make the mode feel broken rather than hard.
        """
        cue = game_state.cue_ball
        if cue is None or cue.table_pos is None:
            return None

        candidates: list[ShotDifficulty] = []
        others = object_balls_on_table(game_state)
        for ball in others:
            assert ball.table_pos is not None
            for index, pocket in enumerate(self._geometry.pocket_centers()):
                shot = self._rate(cue.table_pos, ball, pocket, index)
                if shot is None:
                    continue
                if self._path_blocked(cue.table_pos, ball, pocket, others):
                    continue
                candidates.append(shot)

        if not candidates:
            return None
        candidates.sort(key=lambda s: s.score)
        # Best pot *per ball*, so a difficulty pick chooses between balls rather
        # than between six pockets for the same ball -- the hardest pocket for
        # an easy ball is not the "hard" shot the player expects to be shown.
        best_per_ball: dict[str, ShotDifficulty] = {}
        for shot in candidates:
            best_per_ball.setdefault(shot.ball.id, shot)
        ranked = sorted(best_per_ball.values(), key=lambda s: s.score)
        position = int(round(self.difficulty.rank * (len(ranked) - 1)))
        return ranked[position]

    @staticmethod
    def _rate(
        cue_pos: Vec2, ball: Ball, pocket: Vec2, pocket_index: int
    ) -> ShotDifficulty | None:
        """Distance and cut angle for one (ball, pocket) pair.

        Static because it is pure geometry, and shared: training mode ranks the
        same way, and two modes disagreeing about which pot is easy would show
        up the moment somebody switched between them.
        """
        assert ball.table_pos is not None
        to_pocket = pocket - ball.table_pos
        to_ball = ball.table_pos - cue_pos
        if to_pocket.length() < 1e-6 or to_ball.length() < 1e-6:
            return None

        angle = math.degrees(
            math.acos(
                max(
                    -1.0,
                    min(
                        1.0,
                        (to_ball.x * to_pocket.x + to_ball.y * to_pocket.y)
                        / (to_ball.length() * to_pocket.length()),
                    ),
                )
            )
        )
        # Past 85 degrees the object ball cannot be sent toward the pocket at
        # all -- the cue ball would have to pass through it.
        if angle >= 85.0:
            return None
        return ShotDifficulty(
            ball=ball,
            pocket=pocket,
            pocket_index=pocket_index,
            distance_in=to_pocket.length(),
            cut_angle_deg=angle,
        )

    @staticmethod
    def _path_blocked(
        cue_pos: Vec2, target: Ball, pocket: Vec2, others: list[Ball]
    ) -> bool:
        """Whether another ball sits on either leg of the shot.

        Two segments, cue-to-target and target-to-pocket, each checked by
        point-to-segment distance against two ball radii. Crude -- it ignores
        that the cue ball only has to *reach* the target, not stop there -- and
        crude is right: the alternative is running the simulator six times per
        ball per frame.
        """
        assert target.table_pos is not None
        for other in others:
            if other.id == target.id or other.table_pos is None:
                continue
            for start, end in ((cue_pos, target.table_pos), (target.table_pos, pocket)):
                if _segment_distance(other.table_pos, start, end) < BALL_RADIUS_IN * 2.0:
                    return True
        return False

    # -- scoring ------------------------------------------------------------

    def on_shot_complete(
        self, game_state: GameState, session: GameSession, pocketed: list[Ball]
    ) -> None:
        """Award points and decide whether the shooter keeps the table."""
        if self.winner is not None:
            return

        player = session.current_player
        if player is None:
            return
        player.shots_taken += 1

        scratched = any(group_of(ball) is BallGroup.CUE for ball in pocketed)
        scored = [ball for ball in pocketed if group_of(ball) is not BallGroup.CUE]

        if scratched:
            # A scratch ends the turn whatever else went down. Keeping the table
            # after putting the cue ball in a pocket would be the one rule
            # everybody at the table would object to.
            session.combo_count = 0
            self._end_turn(session, "Scratch")
            return

        if not scored:
            self._end_turn(session, "Miss")
            return

        player.shots_made += 1
        for ball in scored:
            session.combo_count += 1
            multiplier = min(session.combo_count, MAX_COMBO_MULTIPLIER)
            points = int(round(BASE_POINTS * self.difficulty.multiplier * multiplier))
            player.score += points
            session.balls_pocketed_this_turn += 1
            if ball.table_pos is not None:
                self.renderer.effects.spawn_score(
                    ball.table_pos, f"+{points}" + (f" x{multiplier}" if multiplier > 1 else "")
                )

        self._turn_ends_at += TIME_BONUS_SECONDS
        self._feedback = (
            f"+{TIME_BONUS_SECONDS:.0f}s  combo x{min(session.combo_count, MAX_COMBO_MULTIPLIER)}"
            if session.combo_count > 1
            else f"+{TIME_BONUS_SECONDS:.0f}s  shoot again"
        )

        if player.score >= TARGET_SCORE:
            self.winner = player
            session.state = SessionState.GAME_OVER
            self._feedback = f"{player.name} wins with {player.score}"
            logger.info("king of the hill: %s wins on %d", player.name, player.score)

    # -- rendering ----------------------------------------------------------

    def update(
        self,
        game_state: GameState,
        prediction: ShotPrediction | None,
        session: GameSession,
    ) -> ModeOutput:
        """Draw the aiming line, the nominated ball and the clock."""
        now = time.perf_counter()
        remaining = self.seconds_remaining(now)

        if self.winner is None and remaining <= 0.0 and session.state in (
            SessionState.IDLE,
            SessionState.AIMING,
        ):
            # Only between shots. Cutting a turn off while the balls are still
            # rolling would void a pot the player already made.
            self._end_turn(session, "Time", now)
            remaining = self.seconds_remaining(now)

        if session.state is SessionState.AIMING and self.winner is None:
            self.target = self.choose_target(game_state)

        target = self.target

        def decorate(canvas: np.ndarray, ctx: object) -> None:
            if target is None or self.winner is not None:
                return
            if session.state is not SessionState.AIMING:
                return
            assert target.ball.table_pos is not None
            highlight_ball(
                canvas,
                ctx,  # type: ignore[arg-type]
                target.ball.table_pos,
                label=f"{self.difficulty.value.upper()}",
                now=now,
            )

        overlay = self.renderer.compose(
            game_state,
            prediction,
            session,
            self.mapper,
            feedback=self._feedback,
            seconds_remaining=None if self.winner else remaining,
            leaderboard=True,
            decorate=decorate,
            now=now,
        )
        return ModeOutput(
            overlay=overlay,
            feedback_text=self._feedback,
            next_action=self._next_action(session, remaining),
        )

    def _next_action(self, session: GameSession, remaining: float) -> str:
        if self.winner is not None:
            return f"{self.winner.name} wins -- reset to play again"
        player = session.current_player
        who = player.name if player else "Player"
        if self.target is None:
            return f"{who}: {remaining:.0f}s left"
        _index, pocket = nearest_pocket(self.target.pocket, self._geometry)
        del pocket
        return (
            f"{who}: {remaining:.0f}s left, "
            f"{self.difficulty.value} target at {self.target.cut_angle_deg:.0f} deg cut"
        )


def _segment_distance(point: Vec2, start: Vec2, end: Vec2) -> float:
    """Shortest distance from a point to a line segment, in table inches."""
    span = end - start
    length_sq = span.x * span.x + span.y * span.y
    if length_sq < 1e-9:
        return point.distance_to(start)
    t = ((point.x - start.x) * span.x + (point.y - start.y) * span.y) / length_sq
    t = max(0.0, min(1.0, t))
    return point.distance_to(Vec2(start.x + span.x * t, start.y + span.y * t))
