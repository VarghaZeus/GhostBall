"""Ball groups, foul detection and pocket attribution, shared by the modes.

Phase 7. Every competitive mode has to answer the same three questions after a
shot -- what went down, whose it was, and was it legal -- and the answers are
pure functions of a list of balls. Keeping them here means Classic and King of
the Hill cannot drift apart on what counts as a foul, and it means the rules
can be tested without a table, a projector or a state machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from app.models import Ball, BallKind, GameState, Vec2
from physics.models import TableGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "BallGroup",
    "ShotOutcome",
    "group_of",
    "classify_shot",
    "nearest_pocket",
    "pocket_name",
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
