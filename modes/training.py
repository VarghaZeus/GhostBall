"""Training mode: guided drills with shot evaluation.

Phase 7.3. Drill definitions and the scoring curve are implemented -- they are
pure data and pure arithmetic, they are what the API's ``/api/training/result``
returns, and they are worth being able to test without a table. Drill selection
and rendering are stubs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from app.models import (
    Ball,
    DrillType,
    GameModeName,
    GameSession,
    GameState,
    ModeOutput,
    PocketId,
    SessionState,
    ShotPrediction,
    Vec2,
)
from modes.mode_manager import GameMode
from modes.scoring import (
    POCKET_NAMES,
    object_balls_on_table,
    path_blocked,
    rate_pot,
)
from physics.models import TableGeometry
from physics.simulator import aim_angle_for_pocket, simulate_shot_fan
from projection.renderer import render_training_overlay

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Drill:
    """One drill instance: a concrete shot the player is asked to make."""

    drill_type: DrillType
    instruction: str  # projected verbatim, so keep it short
    target_ball_id: str | None = None
    target_pocket: PocketId | None = None
    #: For position drills: where the cue ball should finish, and how close
    #: counts. A radius rather than a point, because "near the centre spot" is
    #: the actual skill being taught.
    target_zone_center: Vec2 | None = None
    target_zone_radius_in: float = 8.0
    #: Rails the ball must contact first, for bank drills.
    required_rails: int = 0

    #: Which power level to ask for: an index into
    #: ``settings.physics.power_buckets``. ``None`` prescribes nothing and the
    #: player is shown all five levels to choose from, as in freeplay.
    #:
    #: Prescribing power is what lets the drill draw a cue-ball resting place at
    #: all. Direction after contact is geometry and always known; distance
    #: depends on how hard the shot is struck, so it is only honest to mark a
    #: resting spot when the drill is the thing that decided the power.
    power_bucket: int | None = None

    #: Where to strike the cue ball, in ball radii from centre -- ``x`` positive
    #: right, ``y`` positive top. ``None`` draws no tip diagram; ``Vec2(0, 0)``
    #: draws one marked centre-ball, which is a different instruction: a drill
    #: can deliberately teach a stun shot, and the diagram is how it says so.
    #:
    #: Prescribed, never measured. The drill defines the english, so this needs
    #: no tip tracking and no cue analysis -- but it does have to reach the
    #: simulator, or the drawn prediction would contradict the diagram drawn
    #: beside it.
    tip_offset: Vec2 | None = None


@dataclass(slots=True)
class DrillResult:
    """Outcome of one attempt, as served by ``GET /api/training/result``."""

    success: bool
    accuracy_pct: float
    stars: int  # 1-3
    feedback: str
    distance_error_in: float = 0.0
    angle_error_deg: float = 0.0
    next_instruction: str = ""


@dataclass(slots=True)
class DrillStats:
    """Running tally across a session, for the progress display."""

    attempts: int = 0
    successes: int = 0
    total_accuracy: float = 0.0
    per_drill: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate_pct(self) -> float:
        return 100.0 * self.successes / self.attempts if self.attempts else 0.0

    @property
    def mean_accuracy_pct(self) -> float:
        return self.total_accuracy / self.attempts if self.attempts else 0.0


#: Tolerances that define success per drill type. These are the numbers to tune
#: after watching real players -- too tight and the mode is demoralising, too
#: loose and it teaches nothing.
DRILL_TOLERANCES: dict[DrillType, dict[str, float]] = {
    DrillType.POTTING: {"distance_in": 3.0, "angle_deg": 5.0},
    DrillType.POSITION: {"distance_in": 8.0, "angle_deg": 15.0},
    DrillType.BANK_SHOT: {"distance_in": 5.0, "angle_deg": 8.0},
    DrillType.SAFETY: {"distance_in": 10.0, "angle_deg": 20.0},
}


def score_attempt(
    drill: Drill,
    distance_error_in: float,
    angle_error_deg: float,
    pocketed: bool,
) -> DrillResult:
    """Turn a measured error into an accuracy percentage, stars and feedback.

    The curve is exponential decay rather than linear, on purpose. Linear
    scoring against a 3-inch tolerance gives 50% for a 1.5-inch miss, which
    feels punishing for what is a decent shot; exponential decay keeps near
    misses in the high scores and separates good from great where the player can
    actually perceive the difference.

    Distance is weighted above angle at 70/30: for a potting drill, ending up in
    the right place matters more than the path taken to get there.
    """
    tolerance = DRILL_TOLERANCES[drill.drill_type]
    # exp(-x) reaches ~0.37 at one tolerance unit, so a miss exactly at the
    # tolerance boundary scores ~37% rather than 0 -- it was close, and the
    # score should say so.
    distance_score = math.exp(-distance_error_in / max(tolerance["distance_in"], 1e-6))
    angle_score = math.exp(-angle_error_deg / max(tolerance["angle_deg"], 1e-6))
    accuracy = 100.0 * (0.7 * distance_score + 0.3 * angle_score)

    within_tolerance = (
        distance_error_in <= tolerance["distance_in"]
        and angle_error_deg <= tolerance["angle_deg"]
    )
    # Potting drills have an objective outcome, so the ball going in overrides
    # the accuracy curve. For position drills there is nothing to pot, and
    # tolerance is the only measure available.
    success = pocketed if drill.drill_type is DrillType.POTTING else within_tolerance

    if accuracy >= 85.0 and success:
        stars, feedback = 3, "Perfect!"
    elif accuracy >= 60.0 and success:
        stars, feedback = 2, "Good shot."
    elif success:
        stars, feedback = 1, "In, but scrappy."
    elif accuracy >= 60.0:
        stars, feedback = 1, f"Close -- missed by {distance_error_in:.1f} in."
    else:
        stars, feedback = 1, f"Missed by {distance_error_in:.1f} in. Try again."

    return DrillResult(
        success=success,
        accuracy_pct=round(accuracy, 1),
        stars=stars,
        feedback=feedback,
        distance_error_in=round(distance_error_in, 2),
        angle_error_deg=round(angle_error_deg, 2),
    )


class TrainingMode(GameMode):
    """Guided drills with per-shot evaluation."""

    name = GameModeName.TRAINING
    display_name = "Training"
    is_competitive = False

    def __init__(self, settings: object | None = None) -> None:
        super().__init__(settings)  # type: ignore[arg-type]
        self.current_drill: Drill | None = None
        self.last_result: DrillResult | None = None
        self.stats = DrillStats()
        #: Prediction for the drill's ideal shot, held so the renderer can draw
        #: it alongside the player's current aim.
        self.target_prediction: ShotPrediction | None = None
        #: The prediction that was live when the player struck. Bank drills need
        #: it to count rails, which cannot be observed between frames.
        self._shot_prediction: ShotPrediction | None = None

    def start_drill(self, drill_type: DrillType, game_state: GameState) -> Drill:
        """Pick a concrete drill for the current ball layout.

        Layout-aware, not scripted: asking for a ball that is not on the table,
        or a pot that is physically blocked, makes the mode useless. Every
        (ball, pocket) pair is checked for a clear path before one is chosen,
        and the *easiest* clear pot is taken rather than a random one -- a drill
        is practice, and practice starts from a shot the player can make.

        Raises:
            DrillUnavailable: When the layout supports no drill of this type.
                Raised rather than returning ``None`` because every caller has
                to tell the user something, and an exception carries the reason.
        """
        cue = game_state.cue_ball
        if cue is None or cue.table_pos is None:
            raise DrillUnavailable("Put the cue ball on the table")

        geometry = TableGeometry.from_settings(self.settings)
        candidates = _clear_pots(cue.table_pos, game_state, geometry)
        if not candidates and drill_type is not DrillType.POSITION:
            raise DrillUnavailable("No clear pot from here -- move the balls and retry")

        if drill_type is DrillType.POSITION:
            drill = self._position_drill(cue, candidates, geometry)
        elif drill_type is DrillType.BANK_SHOT:
            drill = self._bank_drill(candidates)
        elif drill_type is DrillType.SAFETY:
            drill = self._safety_drill(candidates)
        else:
            ball, pocket_index, pocket = candidates[0]
            drill = Drill(
                drill_type=DrillType.POTTING,
                instruction=f"Pot it in the {POCKET_NAMES[pocket_index].replace('_', ' ')}",
                target_ball_id=ball.id,
                target_pocket=_pocket_id(pocket_index),
                power_bucket=_MEDIUM,
                # Centre ball, explicitly. A potting drill is about the line, and
                # adding english to it would teach two things at once and let a
                # player pot by accident off unintended spin. Prescribed rather
                # than left as None so the diagram appears and says "no spin" --
                # which for a beginner is the instruction, not the absence of one.
                tip_offset=Vec2(0.0, 0.0),
            )

        self.current_drill = drill
        self.last_result = None
        self.target_prediction = self._ideal_shot(cue, drill, game_state)
        logger.info("training drill: %s -- %s", drill.drill_type.value, drill.instruction)
        return drill

    def _position_drill(
        self,
        cue: Ball,
        candidates: list[tuple[Ball, int, Vec2]],
        geometry: TableGeometry,
    ) -> Drill:
        """Pot a ball and leave the cue ball in a zone.

        The zone is the table centre, which is the honest default: it is the
        position that leaves the most next shots available, which is what
        position play is for.
        """
        center = Vec2(geometry.length_in / 2.0, geometry.width_in / 2.0)
        if candidates:
            ball, pocket_index, _pocket = candidates[0]
            tip = _spin_towards_zone(cue, ball, center)
            return Drill(
                drill_type=DrillType.POSITION,
                instruction=(
                    "Pot it with draw, cue ball to centre"
                    if tip.y < 0.0
                    else "Pot it with follow, cue ball to centre"
                ),
                target_ball_id=ball.id,
                target_pocket=_pocket_id(pocket_index),
                target_zone_center=center,
                # No power prescribed, deliberately. Choosing the pace *is* the
                # skill a position drill teaches -- prescribing it would do the
                # student's thinking for them and leave nothing to learn but the
                # line. So this is the drill that gets a recommendation instead:
                # all five levels stay on the cloth with the one that leaves
                # centre table highlighted, which is advice a player can see the
                # reasoning behind and overrule.
                tip_offset=tip,
            )
        return Drill(
            drill_type=DrillType.POSITION,
            instruction="Bring the cue ball back to centre table",
            target_zone_center=center,
        )

    def _bank_drill(self, candidates: list[tuple[Ball, int, Vec2]]) -> Drill:
        """Pot off a cushion. Takes the hardest clear pot, since banks are hard."""
        ball, pocket_index, _pocket = candidates[-1]
        return Drill(
            drill_type=DrillType.BANK_SHOT,
            instruction="Bank it off one rail and pot it",
            target_ball_id=ball.id,
            target_pocket=_pocket_id(pocket_index),
            required_rails=1,
            # Banks need pace: the ball has a rail's worth of extra distance to
            # cover and loses energy at the cushion, and a bank hit softly dies
            # short of the pocket.
            power_bucket=_STRONG,
            tip_offset=Vec2(0.0, 0.0),
        )

    def _safety_drill(self, candidates: list[tuple[Ball, int, Vec2]]) -> Drill:
        """Leave the opponent nothing. Scored on the cue ball, not on a pot."""
        ball, _pocket_index, _pocket = candidates[0]
        return Drill(
            drill_type=DrillType.SAFETY,
            instruction="Soft draw -- leave the cue ball safe",
            target_ball_id=ball.id,
            # Soft, with draw. A safety is about *not* moving the cue ball far,
            # and draw off a soft hit is the standard way to kill it near the
            # contact instead of letting it drift into the open.
            power_bucket=_VERY_SOFT,
            tip_offset=Vec2(0.0, -0.35),
        )

    def _ideal_shot(
        self, cue: Ball, drill: Drill, game_state: GameState
    ) -> ShotPrediction | None:
        """Simulate the shot the drill is asking for, to draw alongside the aim.

        The comparison *is* the teaching: a dashed ideal line next to the
        player's solid live one shows the error before the shot is taken, which
        is the only moment the information is useful.

        A drill with a target ball but no pocket -- a safety -- still gets a
        line, aimed full at the ball. Without this it got none, because there was
        no pocket to solve an aim for. That left the safety drill prescribing a
        soft draw and then drawing nothing to show what one looks like, which is
        the least useful place to withhold the demonstration: killing the cue
        ball near the contact is the entire skill being taught.
        """
        if drill.target_ball_id is None:
            return None
        target = game_state.ball_by_id(drill.target_ball_id)
        if target is None or target.table_pos is None or cue.table_pos is None:
            return None

        if drill.target_pocket is None:
            offset = target.table_pos - cue.table_pos
            angle = math.degrees(math.atan2(offset.y, offset.x))
        else:
            geometry = TableGeometry.from_settings(self.settings)
            pocket = geometry.pocket_centers()[_pocket_index(drill.target_pocket)]
            angle = aim_angle_for_pocket(cue.table_pos, target.table_pos, pocket)
        # The fan rather than a single shot, and never ``default_power``: on the
        # power scale that value free-rolls the cue ball some thirteen table
        # lengths, so the "ideal" line it used to draw ran around the table
        # several times over. The drill's own level is the only power here that
        # means anything.
        return simulate_shot_fan(
            cue.table_pos,
            angle,
            game_state.object_balls(),
            self.settings,
            tip_offset=drill.tip_offset,
            prescribed_bucket=drill.power_bucket,
        )

    def legal_target_ids(
        self, game_state: GameState, session: GameSession
    ) -> list[str] | None:
        """The drill's target ball, which is the only one that counts here.

        Narrower than any competitive mode's answer, and correctly so: a drill
        asking for a specific pot is not satisfied by leaving a good shot on some
        other ball. That makes the recommendation in training a genuinely
        different judgement from the one in classic -- "which pace leaves me on
        *this* ball" rather than "on anything I am allowed to hit".

        ``None`` when no drill is running, so nothing is recommended against a
        goal that does not exist yet.
        """
        drill = self.current_drill
        if drill is None or drill.target_ball_id is None:
            return None
        target = game_state.ball_by_id(drill.target_ball_id)
        if target is None or target.table_pos is None:
            # The drill's ball is gone -- potted, or lost by the detector. Either
            # way there is nothing to recommend a pace for.
            return []
        return [target.id]

    def evaluate(self, game_state: GameState, pocketed: list[Ball]) -> DrillResult:
        """Measure the attempt against the active drill.

        Measurement differs by drill type: potting compares the target ball's
        fate, position compares the *cue* ball's resting place against the
        target zone, and bank counts rail contacts before the pot. Once
        measured, the errors go to :func:`score_attempt` -- keeping measurement
        and scoring separate is what lets the curve be tuned without touching
        geometry.
        """
        drill = self.current_drill
        if drill is None:
            return DrillResult(False, 0.0, 1, "No drill running")

        potted_ids = {ball.id for ball in pocketed}
        target_potted = drill.target_ball_id in potted_ids if drill.target_ball_id else False

        if drill.drill_type is DrillType.POSITION:
            distance_error, angle_error = self._position_error(game_state, drill)
        elif drill.drill_type is DrillType.BANK_SHOT:
            distance_error, angle_error = self._bank_error(game_state, drill, target_potted)
        elif drill.drill_type is DrillType.SAFETY:
            distance_error, angle_error = self._safety_error(game_state)
        else:
            distance_error, angle_error = self._potting_error(game_state, drill, target_potted)

        result = score_attempt(drill, distance_error, angle_error, target_potted)
        result.next_instruction = _coach_tip(result, self.stats)
        return result

    def _potting_error(
        self, game_state: GameState, drill: Drill, potted: bool
    ) -> tuple[float, float]:
        """How far the target ball finished from the pocket it was aimed at."""
        if potted:
            return 0.0, 0.0
        target = (
            game_state.ball_by_id(drill.target_ball_id) if drill.target_ball_id else None
        )
        if target is None or target.table_pos is None or drill.target_pocket is None:
            # The ball is neither potted nor visible. Score it as a clear miss
            # rather than guessing -- an invented small error would read as a
            # near miss and teach the wrong lesson.
            return _MISS_DISTANCE_IN, _MISS_ANGLE_DEG
        geometry = TableGeometry.from_settings(self.settings)
        pocket = geometry.pocket_centers()[_pocket_index(drill.target_pocket)]
        return target.table_pos.distance_to(pocket), 0.0

    def _position_error(self, game_state: GameState, drill: Drill) -> tuple[float, float]:
        """How far the cue ball finished from the target zone."""
        cue = game_state.cue_ball
        if cue is None or cue.table_pos is None or drill.target_zone_center is None:
            return _MISS_DISTANCE_IN, _MISS_ANGLE_DEG
        distance = cue.table_pos.distance_to(drill.target_zone_center)
        # Inside the zone is a hit, not a near miss: the drill asks for an area.
        return max(0.0, distance - drill.target_zone_radius_in), 0.0

    def _bank_error(
        self, game_state: GameState, drill: Drill, potted: bool
    ) -> tuple[float, float]:
        """Potting error, plus a penalty if the required rails were not used.

        Rail count comes from the prediction that was live at the strike, for
        the same reason trick shots use it: the flight happens between frames
        and cannot be observed directly.
        """
        distance, angle = self._potting_error(game_state, drill, potted)
        rails = 0
        if self._shot_prediction is not None:
            rails = sum(1 for i in self._shot_prediction.impact_points if i.is_cushion)
        if rails < drill.required_rails:
            # A pot without the bank is not the drill. Penalise in angle, which
            # is the axis the scoring curve weights lower -- the shot was still
            # made, it was just the wrong shot.
            angle += _MISS_ANGLE_DEG
        return distance, angle

    def _safety_error(self, game_state: GameState) -> tuple[float, float]:
        """How safe the cue ball ended up: distance from the nearest object ball.

        Inverted, because a safety wants the cue ball *far* from everything. A
        cue ball 30 inches from the nearest ball scores zero error; one touching
        another ball scores the full miss.
        """
        cue = game_state.cue_ball
        others = [b for b in game_state.object_balls() if b.table_pos is not None]
        if cue is None or cue.table_pos is None or not others:
            return _MISS_DISTANCE_IN, 0.0
        nearest = min(cue.table_pos.distance_to(b.table_pos) for b in others)  # type: ignore[arg-type]
        return max(0.0, _SAFE_DISTANCE_IN - nearest), 0.0

    def update(
        self,
        game_state: GameState,
        prediction: ShotPrediction | None,
        session: GameSession,
    ) -> ModeOutput:
        """Draw the drill target, the player's aim and any feedback."""
        if session.state is SessionState.AIMING and prediction is not None:
            self._shot_prediction = prediction

        advice = ""
        if session.state is SessionState.AIMING:
            # The position drill is the one this fires on: it deliberately
            # prescribes no power, so there is a level to recommend. Potting and
            # bank drills prescribe one, and `recommend_power` stays quiet rather
            # than arguing with the drill.
            advice = self.recommend_power(game_state, self.target_prediction, session)

        feedback = advice or self._feedback_line()
        if self.mapper is None:
            return ModeOutput(feedback_text=feedback, next_action=self._next_action())

        canvas = self.renderer.compose(
            game_state,
            None,  # the aim is drawn by the training overlay, not the generic one
            session,
            self.mapper,
            feedback=feedback,
        )
        if canvas is not None:
            render_training_overlay(
                game_state,
                self.target_prediction,
                prediction if session.state is SessionState.AIMING else None,
                self.mapper,  # type: ignore[arg-type]
                feedback_text="",
                settings=self.settings,
                canvas=canvas,
                clear=False,
                tip_offset=(
                    self.current_drill.tip_offset if self.current_drill else None
                ),
            )
        return ModeOutput(
            overlay=canvas, feedback_text=feedback, next_action=self._next_action()
        )

    def _feedback_line(self) -> str:
        """One projected line: the last result if there is one, else the task."""
        if self.last_result is not None:
            stars = "*" * self.last_result.stars
            return f"{stars}  {self.last_result.feedback}"
        if self.current_drill is not None:
            return self.current_drill.instruction
        return "Pick a drill to begin"

    def _next_action(self) -> str:
        """A longer line for the control panel, which has room for it."""
        if self.current_drill is None:
            return "POST /api/training/start to pick a drill"
        return (
            f"{self.current_drill.instruction} "
            f"({self.stats.successes}/{self.stats.attempts}, "
            f"{self.stats.success_rate_pct:.0f}%)"
        )

    def on_shot_complete(
        self, game_state: GameState, session: GameSession, pocketed: list[Ball]
    ) -> None:
        """Score the attempt and update the running stats."""
        if self.current_drill is None:
            return

        result = self.evaluate(game_state, pocketed)
        self.last_result = result
        self.stats.attempts += 1
        self.stats.total_accuracy += result.accuracy_pct
        if result.success:
            self.stats.successes += 1
        key = self.current_drill.drill_type.value
        self.stats.per_drill[key] = self.stats.per_drill.get(key, 0) + 1
        self._shot_prediction = None

        if session.players:
            player = session.players[session.current_player_index % len(session.players)]
            player.shots_taken += 1
            if result.success:
                player.shots_made += 1
                player.score += result.stars * 10

        cue = game_state.cue_ball
        if cue is not None and cue.table_pos is not None:
            self.renderer.effects.spawn_score(
                cue.table_pos, f"{result.accuracy_pct:.0f}%"
            )
        logger.info(
            "drill %s: %s %.0f%% (%d star) -- %s",
            self.current_drill.drill_type.value,
            "hit" if result.success else "miss",
            result.accuracy_pct,
            result.stars,
            result.feedback,
        )


class DrillUnavailable(RuntimeError):
    """No drill of the requested type fits the balls currently on the table."""


#: Error attributed to a shot that missed entirely. Well past every tolerance in
#: :data:`DRILL_TOLERANCES`, so the scoring curve bottoms out rather than
#: producing a flattering number for a shot that went nowhere near.
_MISS_DISTANCE_IN = 40.0
_MISS_ANGLE_DEG = 45.0

#: Distance from the nearest ball at which a safety counts as fully successful.
_SAFE_DISTANCE_IN = 30.0

#: Power levels drills ask for, as indices into ``physics.power_buckets``.
#:
#: Named rather than written as bare integers at each call site: a drill saying
#: ``power_bucket=3`` is unreadable, and if the configured levels are ever
#: reordered these are the one place to look. They are indices and not distances
#: because the drill is asking for *the level the player is shown* -- the tick
#: labelled "strong" -- and a distance would drift away from whatever that tick
#: currently means after a friction retune.
#: ``_SOFT`` and ``_VERY_HARD`` are unused by the current drills and kept for
#: the same reason the others are named: these are the five levels, and defining
#: three of five would make the next drill author guess at the numbering.
_VERY_SOFT, _SOFT, _MEDIUM, _STRONG, _VERY_HARD = 0, 1, 2, 3, 4


def _spin_towards_zone(cue: Ball, target: Ball, zone: Vec2) -> Vec2:
    """Draw or follow, whichever sends the cue ball toward the target zone.

    A sign test, not position play: it asks only whether the zone is behind the
    contact point or beyond it, along the line of the shot. Behind means draw,
    beyond means follow. That is the first thing a coach would say about the
    shot, and it makes the prescribed spin meaningful rather than always
    centre-ball.

    What it deliberately does not do is *solve* for the tip offset that lands the
    cue ball in the zone. That needs scoring candidate positions rather than
    reading a sign, and getting it half right would be worse than being plainly
    approximate -- a drill claiming to have solved the shot has to be right.
    """
    if cue.table_pos is None or target.table_pos is None:
        return Vec2(0.0, 0.0)
    along = target.table_pos - cue.table_pos
    length = along.length()
    if length < 1e-6:
        return Vec2(0.0, 0.0)
    unit = along.scaled(1.0 / length)
    to_zone = zone - target.table_pos
    # Positive means the zone lies further along the shot line than the object
    # ball, so the cue ball has to carry on through it.
    forward = to_zone.x * unit.x + to_zone.y * unit.y
    return Vec2(0.0, 0.35 if forward > 0.0 else -0.35)


def _pocket_id(index: int) -> PocketId:
    """``PocketId`` for an index into ``TableGeometry.pocket_centers()``."""
    return PocketId(POCKET_NAMES[index % len(POCKET_NAMES)])


def _pocket_index(pocket: PocketId) -> int:
    """The inverse of :func:`_pocket_id`."""
    return POCKET_NAMES.index(pocket.value)


def _clear_pots(
    cue_pos: Vec2, game_state: GameState, geometry: TableGeometry
) -> list[tuple[Ball, int, Vec2]]:
    """Every unobstructed pot available, easiest first.

    Shares the difficulty model with King of the Hill rather than inventing a
    second one: two modes disagreeing about which shot is easy would be visible
    the moment somebody switched between them. It lives in
    :mod:`modes.scoring`, so this no longer needs a function-local import to
    reach around a cycle.
    """
    others = object_balls_on_table(game_state)
    scored: list[tuple[float, Ball, int, Vec2]] = []
    for ball in others:
        for index, pocket in enumerate(geometry.pocket_centers()):
            shot = rate_pot(cue_pos, ball, pocket, index)
            if shot is None or path_blocked(cue_pos, ball, pocket, others):
                continue
            scored.append((shot.score, ball, index, pocket))
    scored.sort(key=lambda item: item[0])
    return [(ball, index, pocket) for _score, ball, index, pocket in scored]


def _coach_tip(result: DrillResult, stats: DrillStats) -> str:
    """A short piece of advice, chosen from what the numbers actually say.

    Kept to statements the data supports. "Work on your bridge" is the sort of
    thing a coach says and this system has no way to observe, so it is not in
    here -- advice that cannot be grounded is noise the player learns to ignore.
    """
    if stats.attempts >= 3 and stats.success_rate_pct >= 80.0:
        return "Consistent -- try a harder drill"
    if result.success and result.accuracy_pct < 60.0:
        return "In, but scrappy. Slow the cue down"
    if not result.success and result.distance_error_in < 4.0:
        return "Very close. Same shot again"
    if not result.success:
        return "Line up the ghost ball before you strike"
    return "Good. Keep the rhythm"
