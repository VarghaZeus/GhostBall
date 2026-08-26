"""Ball groups, foul detection and pocket attribution, shared by the modes.

Phase 7. Every competitive mode has to answer the same three questions after a
shot -- what went down, whose it was, and was it legal -- and the answers are
pure functions of a list of balls. Keeping them here means Classic and King of
the Hill cannot drift apart on what counts as a foul, and it means the rules
can be tested without a table, a projector or a state machine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from enum import Enum

from app.config import BALL_RADIUS_IN
from app.models import Ball, BallKind, GameState, PowerTick, ShotPrediction, Vec2
from physics.models import TableGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "BallGroup",
    "ShotOutcome",
    "ShotDifficulty",
    "LeaveQuality",
    "PositionAssessment",
    "BucketRecommender",
    "group_of",
    "classify_shot",
    "nearest_pocket",
    "pocket_name",
    "rate_pot",
    "path_blocked",
    "assess_leave",
    "leave_advice",
]

#: Points for potting a ball you were entitled to pot. The spec's figure for
#: Classic; King of the Hill overrides it with its own flat rate.
POINTS_PER_BALL = 25

#: Points the opponent gets for a foul. Potting the cue ball is the common one.
FOUL_PENALTY = 2


class BallGroup(str, Enum):
    """Which half of the rack a ball belongs to.

    ``OPEN`` is a state of the *table*, not of a ball: before anyone has potted
    legally, neither group is assigned and any ball is fair game.
    """

    OPEN = "open"
    SOLIDS = "solids"
    STRIPES = "stripes"
    EIGHT = "eight"
    CUE = "cue"

    @property
    def is_playable(self) -> bool:
        """Whether this is a group a player can be assigned to."""
        return self in (BallGroup.SOLIDS, BallGroup.STRIPES)

    def opposing(self) -> BallGroup:
        """The other group, for assigning the second player."""
        if self is BallGroup.SOLIDS:
            return BallGroup.STRIPES
        if self is BallGroup.STRIPES:
            return BallGroup.SOLIDS
        return BallGroup.OPEN


def group_of(ball: Ball) -> BallGroup:
    """Which group a ball belongs to.

    Reads ``kind`` first and falls back to ``number``, because they come from
    different places and disagree more often than is comfortable: ``kind`` is
    inferred from the ball's stripe pattern, which a projected overlay across
    the ball can confuse, while ``number`` is only set when the rack number was
    legible. Trusting either alone loses balls the other could have classified.
    """
    if ball.kind is BallKind.CUE:
        return BallGroup.CUE
    if ball.kind is BallKind.EIGHT:
        return BallGroup.EIGHT
    if ball.kind is BallKind.SOLID:
        return BallGroup.SOLIDS
    if ball.kind is BallKind.STRIPE:
        return BallGroup.STRIPES

    number = ball.number
    if number is None:
        return BallGroup.OPEN
    if number == 8:
        return BallGroup.EIGHT
    return BallGroup.SOLIDS if 1 <= number <= 7 else BallGroup.STRIPES


@dataclass(slots=True)
class ShotOutcome:
    """What one shot did, in rules terms.

    Deliberately a description rather than a decision: it says the cue ball went
    down and three stripes went with it, and leaves "so whose turn is it" to the
    mode, because Classic and King of the Hill answer that differently from the
    same facts.
    """

    pocketed: list[Ball]
    #: Cue ball potted -- a foul in every mode here.
    scratched: bool = False
    #: The 8 went down. Whether that wins or loses is the mode's business.
    eight_potted: bool = False
    #: Balls of the player's own group.
    own: list[Ball] = None  # type: ignore[assignment]
    #: Balls of the opponent's group. Potting one is a foul in Classic.
    opponent: list[Ball] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.own is None:
            self.own = []
        if self.opponent is None:
            self.opponent = []

    @property
    def is_foul(self) -> bool:
        return self.scratched or bool(self.opponent)

    @property
    def scored(self) -> bool:
        """Whether the player potted something they were entitled to."""
        return bool(self.own) and not self.scratched

    @property
    def summary(self) -> str:
        """One short line for the projected feedback."""
        if self.scratched:
            return "Scratch"
        if self.eight_potted:
            return "8 ball"
        if not self.pocketed:
            return "Miss"
        if self.opponent and not self.own:
            return "Wrong ball"
        return f"{len(self.own)} potted" if len(self.own) != 1 else "Potted"


def classify_shot(pocketed: list[Ball], player_group: BallGroup) -> ShotOutcome:
    """Sort what went down into the categories the rules care about.

    ``player_group`` of :attr:`BallGroup.OPEN` means the table is still open, in
    which case nothing counts as the opponent's -- which is what makes the first
    legal pot assign the groups rather than being a foul.
    """
    outcome = ShotOutcome(pocketed=list(pocketed))
    for ball in pocketed:
        group = group_of(ball)
        if group is BallGroup.CUE:
            outcome.scratched = True
        elif group is BallGroup.EIGHT:
            outcome.eight_potted = True
        elif not player_group.is_playable:
            # Open table: every object ball is the player's, which is how the
            # group gets assigned.
            outcome.own.append(ball)
        elif group is player_group:
            outcome.own.append(ball)
        elif group is BallGroup.OPEN:
            # Unclassifiable ball -- the detector could not tell stripe from
            # solid. Credit it rather than penalise: the player did pot
            # something, and a foul call on our own uncertainty is the one
            # outcome they cannot argue with or learn from.
            logger.debug("ball %s has no readable group; crediting it", ball.id)
            outcome.own.append(ball)
        else:
            outcome.opponent.append(ball)
    return outcome


def nearest_pocket(position: Vec2, geometry: TableGeometry) -> tuple[int, Vec2]:
    """Index and centre of the pocket closest to a table position.

    Used to attribute a potted ball to a pocket, which trick shots need in order
    to check the ball went where the challenge asked. The ball's last observed
    position is a frame or two before it dropped, so this is an inference --
    accurate enough at pocket scale, and the only option, since a ball in a
    pocket is not visible to be measured.
    """
    centers = geometry.pocket_centers()
    index = min(range(len(centers)), key=lambda i: position.distance_to(centers[i]))
    return index, centers[index]


#: Pocket names in the order :meth:`TableGeometry.pocket_centers` returns them,
#: matching :class:`app.models.PocketId`'s spelling so a challenge file can name
#: a pocket the same way the rest of the system does.
POCKET_NAMES = (
    "top_left",
    "top_middle",
    "top_right",
    "bottom_right",
    "bottom_middle",
    "bottom_left",
)


def pocket_name(index: int) -> str:
    """Human-readable pocket name for an index into ``pocket_centers()``."""
    return POCKET_NAMES[index % len(POCKET_NAMES)]


def object_balls_on_table(game_state: GameState) -> list[Ball]:
    """Every ball still in play that is not the cue ball.

    A convenience the modes all reached for independently; ``GameState`` has
    ``object_balls`` but includes the 8, and target selection usually should
    not.
    """
    return [
        ball
        for ball in game_state.object_balls()
        if ball.table_pos is not None and group_of(ball) is not BallGroup.EIGHT
    ]

# ---------------------------------------------------------------------------
# Shot difficulty
# ---------------------------------------------------------------------------
#
# Lives here rather than in a mode because three callers now need it -- King of
# the Hill picks a target by it, training ranks drills by it, and position
# scoring below asks it whether a leave has a pot in it. Two modes disagreeing
# about which pot is easy would be visible the moment somebody switched between
# them; three would be worse.
#
# It used to be static methods on ``KingOfTheHillMode``, reached from training
# through a function-local import to dodge the resulting cycle. Position scoring
# is in this module and this module is imported *by* KOTH, so that arrangement
# could not stretch to a third caller.


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


def rate_pot(
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


def path_blocked(
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


def _segment_distance(point: Vec2, start: Vec2, end: Vec2) -> float:
    """Shortest distance from a point to a line segment, in table inches."""
    span = end - start
    length_sq = span.x * span.x + span.y * span.y
    if length_sq < 1e-9:
        return point.distance_to(start)
    t = ((point.x - start.x) * span.x + (point.y - start.y) * span.y) / length_sq
    t = max(0.0, min(1.0, t))
    return point.distance_to(Vec2(start.x + span.x * t, start.y + span.y * t))


# ---------------------------------------------------------------------------
# Position scoring
# ---------------------------------------------------------------------------

#: How close to a cushion counts as frozen on the rail, in ball radii.
#:
#: Not a geometric threshold -- the ball fits anywhere -- but a cueing one. With
#: the cue ball nearer than about this, the bridge hand has no room behind it and
#: the range of usable cue elevations and angles collapses. Two radii is roughly
#: where a player starts having to think about it rather than just play the shot.
RAIL_FREEZE_RADII = 2.0

#: Penalty added to a leave's difficulty for being frozen on a rail.
#:
#: Sized against :attr:`ShotDifficulty.score`, where 2.0 is a 45-degree cut and
#: 1.0 is a 40-inch pot. So this says a rail-frozen straight-in is about as
#: unattractive as a comfortable 30-degree cut from open cloth -- which is
#: roughly the trade a player would make.
RAIL_FREEZE_PENALTY = 1.3


class LeaveQuality(str, Enum):
    """How playable a cue-ball resting place is.

    Four outcomes rather than one number because they want different words on
    the cloth. Collapsing them to a score would make "you cannot reach a legal
    ball" and "you can reach one but the pot is a thin cut" differ by a
    magnitude, when they differ in kind -- and the first is the one worth
    interrupting a player to say.

    Ordered worst to best, so ``max`` picks the better leave and comparisons
    read the way they sound.
    """

    #: Cue ball went down. Nothing else about the position matters.
    SCRATCH = "scratch"
    #: No clear line to any ball this player may hit. A snooker.
    SNOOKERED = "snookered"
    #: Can hit something legal, but no clear pot from there.
    NO_POT = "no_pot"
    #: A clear pot exists. The only outcome worth recommending.
    PLAYABLE = "playable"

    @property
    def rank(self) -> int:
        return _LEAVE_RANK[self]

    @property
    def is_playable(self) -> bool:
        return self is LeaveQuality.PLAYABLE


_LEAVE_RANK: dict[LeaveQuality, int] = {
    LeaveQuality.SCRATCH: 0,
    LeaveQuality.SNOOKERED: 1,
    LeaveQuality.NO_POT: 2,
    LeaveQuality.PLAYABLE: 3,
}


@dataclass(frozen=True, slots=True)
class PositionAssessment:
    """What a cue-ball resting place leaves the player.

    Carries the reason alongside the verdict, for the same reason
    :class:`ShotDifficulty` keeps its components: "no good leave" and "no good
    leave *because you are snookered*" are different things to tell somebody,
    and the second is the one that changes what they do next.
    """

    quality: LeaveQuality
    #: The easiest pot available from here, if any.
    best_pot: ShotDifficulty | None = None
    #: Difficulty of that pot plus positional penalties, lower is better.
    #: ``inf`` for anything not :attr:`LeaveQuality.PLAYABLE`, so an unplayable
    #: leave can never sort above a playable one by arithmetic accident.
    cost: float = math.inf
    #: Whether the cue ball finished within cueing distance of a cushion.
    on_rail: bool = False


def assess_leave(
    cue_pos: Vec2,
    legal_balls: list[Ball],
    all_balls: list[Ball],
    geometry: TableGeometry,
    scratched: bool = False,
) -> PositionAssessment:
    """Judge what a cue-ball position leaves for the next shot.

    Args:
        cue_pos: Where the cue ball came to rest, table inches.
        legal_balls: Balls this player is allowed to hit next. An empty list
            means the caller could not determine it, which is not the same as
            there being nothing legal -- callers that cannot answer should not
            be asking, so this returns ``NO_POT`` rather than inventing a
            verdict.
        all_balls: Everything on the table, for obstruction checks.
        geometry: For the rail-proximity test.
        scratched: Whether the cue ball was potted getting here.

    The three geometric questions are answered with the same difficulty model
    the modes already use to pick targets -- :func:`rate_pot` for whether the
    angle is sensible or desperate, :func:`path_blocked` for whether the line is
    clear. Reusing it is the point: a leave this calls playable has to be a shot
    King of the Hill would be willing to set as a target.
    """
    if scratched:
        # Nothing else is worth computing. A scratch is not a bad position, it
        # is the absence of one -- the cue ball is in hand and every geometric
        # question below is about a table state that will not exist.
        return PositionAssessment(quality=LeaveQuality.SCRATCH)

    on_rail = _is_on_rail(cue_pos, geometry)
    if not legal_balls:
        return PositionAssessment(quality=LeaveQuality.NO_POT, on_rail=on_rail)

    # Can a legal ball be *hit* at all? Asked separately from whether it can be
    # potted, because the two failures are different sizes. No clear pot means
    # play safe; no clear contact means you are snookered and may well be
    # conceding a foul.
    reachable = [
        ball
        for ball in legal_balls
        if ball.table_pos is not None
        and not _contact_blocked(cue_pos, ball, all_balls)
    ]
    if not reachable:
        return PositionAssessment(quality=LeaveQuality.SNOOKERED, on_rail=on_rail)

    best: ShotDifficulty | None = None
    for ball in reachable:
        for index, pocket in enumerate(geometry.pocket_centers()):
            shot = rate_pot(cue_pos, ball, pocket, index)
            if shot is None or path_blocked(cue_pos, ball, pocket, all_balls):
                continue
            if best is None or shot.score < best.score:
                best = shot

    if best is None:
        return PositionAssessment(quality=LeaveQuality.NO_POT, on_rail=on_rail)

    cost = best.score + (RAIL_FREEZE_PENALTY if on_rail else 0.0)
    return PositionAssessment(
        quality=LeaveQuality.PLAYABLE, best_pot=best, cost=cost, on_rail=on_rail
    )


def _is_on_rail(cue_pos: Vec2, geometry: TableGeometry) -> bool:
    """Whether the cue ball is too close to a cushion to cue comfortably."""
    limit = BALL_RADIUS_IN * RAIL_FREEZE_RADII
    return (
        cue_pos.x - geometry.x_min < limit
        or geometry.x_max - cue_pos.x < limit
        or cue_pos.y - geometry.y_min < limit
        or geometry.y_max - cue_pos.y < limit
    )


def _contact_blocked(cue_pos: Vec2, target: Ball, others: list[Ball]) -> bool:
    """Whether another ball sits between the cue ball and this one.

    Only the cue-to-target leg, unlike :func:`path_blocked`, which also checks
    the route to a pocket. That difference is what separates being snookered
    from merely having no pot: here the question is whether the ball can be
    struck at all.
    """
    assert target.table_pos is not None
    for other in others:
        if other.id == target.id or other.table_pos is None:
            continue
        if _segment_distance(other.table_pos, cue_pos, target.table_pos) < BALL_RADIUS_IN * 2.0:
            return True
    return False


# ---------------------------------------------------------------------------
# Recommending a power level
# ---------------------------------------------------------------------------

#: How much better a challenger must score than the incumbent recommendation
#: before the highlight moves, in :attr:`ShotDifficulty.score` units.
#:
#: The reason there is a margin at all: two adjacent levels often leave near
#: equally good positions, and detection noise walks the measured aim across the
#: prediction cache's quantisation cells from frame to frame. Recommending
#: whichever won this frame makes the highlight flap MEDIUM / STRONG / MEDIUM on
#: the cloth several times a second. A recommendation that flickers is worse
#: than a stable slightly-suboptimal one -- it is unreadable, and it advertises
#: that the system cannot make up its mind, which is a good reason not to trust
#: the rest of the overlay either.
#:
#: Same discipline as the readiness state machine and the loop's load shedding:
#: the incumbent holds unless the challenger beats it by a real margin, not just
#: by winning a frame. 0.35 is about a five-degree difference in cut angle on
#: the resulting pot -- below what a player would notice, and well above the
#: frame-to-frame jitter.
RECOMMEND_MARGIN = 0.35


class BucketRecommender:
    """Picks which power level to highlight, and holds it steady.

    Stateful, and owned by a mode instance rather than being a free function,
    for two reasons that both come down to time: the hysteresis needs to
    remember what it recommended last frame, and the memo needs somewhere to
    live across frames.

    The memo is keyed on the resting positions themselves rather than on the aim.
    Those positions come from the prediction cache, which quantises the aim
    already, so identical positions mean the aim has not moved materially and the
    answer cannot have changed. Keying on them inherits that quantisation
    instead of introducing a second threshold that could drift out of step with
    the first -- and it stays correct if the prediction cache is ever retuned.
    """

    def __init__(self, margin: float = RECOMMEND_MARGIN) -> None:
        self.margin = margin
        #: Label of the level currently highlighted, or ``None`` for no
        #: recommendation. Held by label rather than by index so it survives a
        #: config reload that reorders the levels.
        self._incumbent: str | None = None
        self._memo_key: tuple | None = None
        self._memo: list[PositionAssessment] | None = None

    def reset(self) -> None:
        """Forget the incumbent. Called when the shot or the rules change.

        Hysteresis across a shot boundary would be wrong rather than merely
        stale: it would hold a recommendation made for a table layout that no
        longer exists.
        """
        self._incumbent = None
        self._memo_key = None
        self._memo = None

    def annotate(
        self,
        prediction: ShotPrediction,
        legal_balls: list[Ball],
        all_balls: list[Ball],
        geometry: TableGeometry,
    ) -> list[PositionAssessment]:
        """Mark the recommended tick on ``prediction``, in place.

        Returns the per-level assessments, in the same order as the ticks, so a
        caller can say *why* there is no recommendation rather than only that
        there is none.

        Recommends nothing, and clears any incumbent, when a level is already
        prescribed. A drill saying "medium" beside a highlight saying "strong"
        is the overlay contradicting itself, and the drill is the one with
        authority -- it knows what it is teaching.
        """
        ticks = prediction.power_ticks
        if not ticks:
            return []
        if any(tick.prescribed for tick in ticks):
            self._incumbent = None
            return []

        assessments = self._assess(ticks, legal_balls, all_balls, geometry)
        winner = self._choose(ticks, assessments)
        self._incumbent = winner
        prediction.power_ticks = [
            replace(tick, recommended=tick.label == winner) if winner else tick
            for tick in ticks
        ]
        return assessments

    def _assess(
        self,
        ticks: list[PowerTick],
        legal_balls: list[Ball],
        all_balls: list[Ball],
        geometry: TableGeometry,
    ) -> list[PositionAssessment]:
        """Score every level, reusing last frame's answer where nothing moved."""
        key = (
            tuple((round(t.position.x, 2), round(t.position.y, 2), t.scratched) for t in ticks),
            tuple(sorted(b.id for b in legal_balls)),
            tuple(
                sorted(
                    (round(b.table_pos.x, 1), round(b.table_pos.y, 1))
                    for b in all_balls
                    if b.table_pos is not None
                )
            ),
        )
        if key == self._memo_key and self._memo is not None:
            return self._memo

        assessments = [
            assess_leave(
                tick.position, legal_balls, all_balls, geometry, scratched=tick.scratched
            )
            for tick in ticks
        ]
        self._memo_key = key
        self._memo = assessments
        return assessments

    def _choose(
        self, ticks: list[PowerTick], assessments: list[PositionAssessment]
    ) -> str | None:
        """The level to highlight, applying hysteresis against the incumbent.

        ``None`` when no level leaves anything playable. Deliberately not "the
        least bad one": picking quietly from four unplayable options presents a
        guess as advice, which is the confidently-wrong failure this whole
        mechanism exists to avoid. The caller says *no good leave, play safe*
        instead, which is real information.
        """
        playable = [
            (tick.label, assessment.cost)
            for tick, assessment in zip(ticks, assessments, strict=True)
            if assessment.quality.is_playable
        ]
        if not playable:
            return None

        best_label, best_cost = min(playable, key=lambda pair: pair[1])
        if self._incumbent is None:
            return best_label

        held = next((cost for label, cost in playable if label == self._incumbent), None)
        if held is None:
            # The incumbent stopped being playable, so there is nothing to hold.
            return best_label
        # Lower cost is better, so the challenger has to come in *below* the
        # incumbent by the margin -- not merely below it.
        return best_label if best_cost < held - self.margin else self._incumbent


def leave_advice(assessments: list[PositionAssessment]) -> str:
    """One short line for when no level leaves a playable position.

    Says which failure it is. "No good leave" tells a player to stop looking for
    a pot; "snookered at every pace" tells them the shot itself is wrong and to
    play for the foul or a safety instead, which is a different decision.

    Empty string when something *is* playable -- there is a highlighted tick in
    that case and it says everything needed.
    """
    if not assessments or any(a.quality.is_playable for a in assessments):
        return ""
    worst = min(a.quality.rank for a in assessments)
    if worst == LeaveQuality.SCRATCH.rank and all(
        a.quality is LeaveQuality.SCRATCH for a in assessments
    ):
        return "scratches at every pace -- play elsewhere"
    if all(a.quality is LeaveQuality.SNOOKERED for a in assessments):
        return "snookered at every pace -- play safe"
    return "no good leave from here -- play safe"
