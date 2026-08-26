"""Trick shot challenges: preset layouts, projected placement, star ratings.

Phase 7.2. Ten challenges from ``data/challenges.json``, progressive difficulty,
one to three stars each.

The problem this mode has and the others do not
-----------------------------------------------
Every other mode reads the table and reacts. This one needs the balls in
*specific places* and cannot put them there -- the system is a camera and a
projector, not a hand. So the mode has a setup phase: it projects a ring on the
cloth for every ball the challenge needs, watches until the balls are on those
rings, and only then arms the shot. Without that phase the mode would be a set
of instructions nobody could follow accurately enough for the evaluation to
mean anything.

Placement tolerance is :data:`PLACEMENT_TOLERANCE_IN`, which is deliberately
loose. The alternative -- demanding the exact inch -- turns a game into a
fiddly chore, and the challenges are designed so that an inch either way does
not change which shot is being asked for.

Judging a shot you can only see the ends of
-------------------------------------------
The camera sees the table before the strike and after it settles; the flight in
between happens faster than table detection runs. So rail counts come from the
*prediction that was live at the moment of the strike* -- the simulator's model
of the shot the player actually took. That is an inference, and it is the same
one the aiming line already makes; if it is wrong the player can see it is
wrong, because the line was drawn on the cloth in front of them.

What is measured directly: which balls left the table, and which pocket the
target was nearest when it did.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import DATA_DIR, Settings, get_settings
from app.models import (
    Ball,
    GameModeName,
    GameSession,
    GameState,
    ModeOutput,
    SessionState,
    ShotPrediction,
    Vec2,
)
from modes.mode_manager import GameMode
from modes.rendering import draw_placement_marker
from modes.scoring import POCKET_NAMES, BallGroup, group_of, nearest_pocket
from physics.models import TableGeometry

logger = logging.getLogger(__name__)

CHALLENGE_FILE = DATA_DIR / "challenges.json"

#: Inches a ball may be from its mark and still count as placed. Two inches is
#: most of a ball's width and well inside what changes the shot; tighter than
#: this and setting up a challenge takes longer than shooting it.
PLACEMENT_TOLERANCE_IN = 2.0

#: The table the challenge coordinates were authored on. Layouts are scaled from
#: this to whatever table is actually in front of the camera, so one file works
#: on a 6 ft table and a 9 ft one.
AUTHORED_LENGTH_IN = 76.0
AUTHORED_WIDTH_IN = 38.0


@dataclass(slots=True)
class ChallengeBall:
    """One ball a challenge needs placed, in table inches."""

    role: str  # cue | target | blocker
    position: Vec2


@dataclass(slots=True)
class Challenge:
    """One trick shot."""

    id: int
    name: str
    description: str
    difficulty: int
    balls: list[ChallengeBall]
    target_pocket: str
    #: Cushions the shot must involve for full marks.
    min_rails: int = 0
    #: Balls that must go down for full marks. 1 unless it is a combo.
    min_potted: int = 1

    #: Power level the shot is set up for: an index into
    #: ``settings.physics.power_buckets``. ``None`` leaves it to the player and
    #: all five levels are shown.
    #:
    #: A trick shot usually only works at one pace, so prescribing it is part of
    #: the challenge rather than a hint -- and it is what lets the overlay mark a
    #: cue-ball resting place instead of five candidates.
    power_bucket: int | None = None

    #: Tip contact the shot needs, in ball radii from centre: ``x`` positive
    #: right, ``y`` positive top. ``None`` draws no diagram.
    #:
    #: Many trick shots are *defined* by their english -- a draw shot is not the
    #: same challenge played with follow -- so this is part of the shot's
    #: specification, not decoration on top of it.
    tip_offset: Vec2 | None = None

    @property
    def cue_position(self) -> Vec2 | None:
        return next((b.position for b in self.balls if b.role == "cue"), None)

    @property
    def target_position(self) -> Vec2 | None:
        return next((b.position for b in self.balls if b.role == "target"), None)

    def scaled_to(self, settings: Settings) -> Challenge:
        """This challenge on the table that is actually here.

        Proportional on each axis independently, which is exact for regulation
        tables because they are all 2:1 -- the scale factor is the same either
        way and the shot's geometry is preserved.
        """
        sx = settings.table.length_in / AUTHORED_LENGTH_IN
        sy = settings.table.width_in / AUTHORED_WIDTH_IN
        if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
            return self
        return Challenge(
            id=self.id,
            name=self.name,
            description=self.description,
            difficulty=self.difficulty,
            balls=[
                ChallengeBall(role=b.role, position=Vec2(b.position.x * sx, b.position.y * sy))
                for b in self.balls
            ],
            target_pocket=self.target_pocket,
            min_rails=self.min_rails,
            min_potted=self.min_potted,
            # Not scaled. A power *level* names a fraction of the table, so it
            # already means the same shot on a bigger one; and a tip offset is in
            # ball radii, which do not change with the table.
            power_bucket=self.power_bucket,
            tip_offset=self.tip_offset,
        )


@dataclass(slots=True)
class ChallengeResult:
    """How an attempt went."""

    challenge_id: int
    success: bool
    stars: int
    potted: int
    rails: int
    pocket: str
    feedback: str


def _optional_int(value: object) -> int | None:
    """An optional integer field, absent when the JSON omits or nulls it."""
    return None if value is None else int(value)  # type: ignore[arg-type]


def _optional_tip_offset(value: object) -> Vec2 | None:
    """A ``tip_offset`` from JSON, as ``{"x": .., "y": ..}`` or ``[x, y]``.

    Both spellings accepted because the file is hand-edited and the object form
    is far more readable for a value whose axes are not interchangeable -- an
    author who transposes ``[0.3, -0.4]`` gets right-hand english where they
    wanted draw, and nothing in the file would look wrong.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return Vec2(float(value.get("x", 0.0)), float(value.get("y", 0.0)))
    x, y = value  # type: ignore[misc]
    return Vec2(float(x), float(y))


def load_challenges(path: Path | None = None) -> list[Challenge]:
    """Read the challenge file.

    A missing or malformed file yields an empty list and a loud log rather than
    an exception: the mode is one of five and taking the whole application down
    because a data file was edited badly is out of proportion. The mode reports
    "no challenges loaded" on the projector, which is the actionable version of
    the same information.
    """
    source = path or CHALLENGE_FILE
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("cannot read challenges from %s: %s", source, exc)
        return []

    challenges: list[Challenge] = []
    for entry in data.get("challenges", []):
        try:
            challenges.append(
                Challenge(
                    id=int(entry["id"]),
                    name=str(entry["name"]),
                    description=str(entry["description"]),
                    difficulty=int(entry.get("difficulty", 1)),
                    balls=[
                        ChallengeBall(
                            role=str(ball.get("role", "target")),
                            position=Vec2(float(ball["x"]), float(ball["y"])),
                        )
                        for ball in entry["balls"]
                    ],
                    target_pocket=str(entry.get("target_pocket", "top_right")),
                    min_rails=int(entry.get("min_rails", 0)),
                    min_potted=int(entry.get("min_potted", 1)),
                    power_bucket=_optional_int(entry.get("power_bucket")),
                    tip_offset=_optional_tip_offset(entry.get("tip_offset")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            # Skip the bad one and keep the rest: one malformed challenge should
            # not cost the player the other nine.
            logger.error("skipping malformed challenge %r: %s", entry.get("id", "?"), exc)

    if challenges:
        logger.info("loaded %d trick shot challenges from %s", len(challenges), source)
    return challenges


class TrickShotsMode(GameMode):
    """Set up a preset layout, shoot it, get graded."""

    name = GameModeName.TRICK_SHOTS
    display_name = "Trick Shots"
    is_competitive = False

    def __init__(self, settings: object | None = None) -> None:
        super().__init__(settings)  # type: ignore[arg-type]
        self.challenges = load_challenges()
        self.index = 0
        self.last_result: ChallengeResult | None = None
        self.results: dict[int, ChallengeResult] = {}
        #: Set once the balls are on their marks; cleared on advance.
        self.armed = False
        #: The prediction that was live when the player struck, kept so rail
        #: contacts can be counted after the fact.
        self._shot_prediction: ShotPrediction | None = None
        self._geometry = TableGeometry.from_settings(self.settings)
        self._feedback = ""

    # -- lifecycle ----------------------------------------------------------

    def on_enter(self, session: GameSession) -> None:
        super().on_enter(session)
        self.renderer.reset()
        self.armed = False
        self._shot_prediction = None
        self._feedback = self.current.description if self.current else "No challenges loaded"

    @property
    def current(self) -> Challenge | None:
        """The active challenge, scaled to this table."""
        if not self.challenges:
            return None
        settings = self.settings if isinstance(self.settings, Settings) else get_settings()
        return self.challenges[self.index % len(self.challenges)].scaled_to(settings)

    def select(self, index: int) -> Challenge | None:
        """Jump to a challenge by position, wrapping."""
        if not self.challenges:
            return None
        self.index = index % len(self.challenges)
        self.armed = False
        self.last_result = None
        self._shot_prediction = None
        current = self.current
        self._feedback = current.description if current else ""
        logger.info(
            "trick shot %d/%d: %s",
            self.index + 1,
            len(self.challenges),
            current.name if current else "?",
        )
        return current

    def advance(self) -> Challenge | None:
        """Move to the next challenge."""
        return self.select(self.index + 1)

    # -- placement ----------------------------------------------------------

    def placement_status(self, game_state: GameState) -> list[tuple[ChallengeBall, bool]]:
        """Each required ball and whether something is sitting on its mark.

        Matching is by proximity alone, not by colour: which physical ball is
        the "target" does not matter to any of these challenges, and demanding a
        specific one would mean the player hunting through the rack for the 4.
        The cue ball is the exception and is matched by kind, since every
        challenge starts from it.
        """
        challenge = self.current
        if challenge is None:
            return []

        placed = [b for b in game_state.balls if b.table_pos is not None and not b.pocketed]
        status: list[tuple[ChallengeBall, bool]] = []
        claimed: set[str] = set()
        for required in challenge.balls:
            candidates = [
                ball
                for ball in placed
                if ball.id not in claimed
                and (
                    (required.role == "cue") == (group_of(ball) is BallGroup.CUE)
                )
            ]
            match = min(
                candidates,
                key=lambda b: b.table_pos.distance_to(required.position),  # type: ignore[union-attr]
                default=None,
            )
            ok = (
                match is not None
                and match.table_pos is not None
                and match.table_pos.distance_to(required.position) <= PLACEMENT_TOLERANCE_IN
            )
            if ok and match is not None:
                claimed.add(match.id)
            status.append((required, ok))
        return status

    def is_ready(self, game_state: GameState) -> bool:
        """Whether every required ball is on its mark."""
        status = self.placement_status(game_state)
        return bool(status) and all(ok for _required, ok in status)

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, pocketed: list[Ball]) -> ChallengeResult:
        """Grade an attempt against the active challenge.

        Three stars needs everything the challenge asked for: the right number
        of balls down, in the named pocket, off the required cushions. Two is
        the pot in the wrong pocket or without the rails -- a made shot, but not
        the shot. One is the target moved and stayed up. There is no zero: the
        player took a shot at a hard thing, and a zero teaches nothing a one
        does not.
        """
        challenge = self.current
        if challenge is None:
            return ChallengeResult(-1, False, 1, 0, 0, "", "No challenge loaded")

        objects = [ball for ball in pocketed if group_of(ball) is not BallGroup.CUE]
        scratched = len(objects) != len(pocketed)
        rails = self._count_rails()
        pocket = ""
        if objects:
            last = objects[0]
            if last.table_pos is not None:
                index, _center = nearest_pocket(last.table_pos, self._geometry)
                pocket = POCKET_NAMES[index]

        right_pocket = pocket == challenge.target_pocket
        enough_balls = len(objects) >= challenge.min_potted
        enough_rails = rails >= challenge.min_rails
        success = enough_balls and not scratched

        if success and right_pocket and enough_rails:
            stars, feedback = 3, "Perfect!"
        elif success:
            missing = []
            if not right_pocket:
                missing.append("wrong pocket")
            if not enough_rails:
                plural = "rail" if challenge.min_rails == 1 else "rails"
                missing.append(f"needed {challenge.min_rails} {plural}")
            stars = 2
            feedback = "Made it -- " + ", ".join(missing) if missing else "Made it"
        elif scratched:
            stars, feedback = 1, "Scratch"
        elif objects:
            stars, feedback = 2, "Potted, but not enough"
        else:
            stars, feedback = 1, "Missed -- try again"

        result = ChallengeResult(
            challenge_id=challenge.id,
            success=success and right_pocket and enough_rails,
            stars=stars,
            potted=len(objects),
            rails=rails,
            pocket=pocket,
            feedback=feedback,
        )
        # Keep the best attempt, so replaying a challenge cannot cost stars.
        previous = self.results.get(challenge.id)
        if previous is None or result.stars > previous.stars:
            self.results[challenge.id] = result
        return result

    def _count_rails(self) -> int:
        """Cushion contacts in the shot the player took, per the live prediction."""
        if self._shot_prediction is None:
            return 0
        return sum(1 for impact in self._shot_prediction.impact_points if impact.is_cushion)

    def on_shot_complete(
        self, game_state: GameState, session: GameSession, pocketed: list[Ball]
    ) -> None:
        """Grade the attempt and report it."""
        if self.current is None or not self.armed:
            return
        result = self.evaluate(pocketed)
        self.last_result = result
        self.armed = False
        self._shot_prediction = None
        session.combo_count = 0
        self._feedback = f"{'*' * result.stars}  {result.feedback}"
        if session.players:
            player = session.players[session.current_player_index % len(session.players)]
            player.shots_taken += 1
            if result.success:
                player.shots_made += 1
                player.score += result.stars * 10
        logger.info(
            "trick shot %s: %d star(s), %d potted, %d rail(s), pocket=%s",
            self.current.name,
            result.stars,
            result.potted,
            result.rails,
            result.pocket or "none",
        )

    @property
    def total_stars(self) -> int:
        """Stars earned across every challenge attempted."""
        return sum(result.stars for result in self.results.values())

    # -- rendering ----------------------------------------------------------

    def update(
        self,
        game_state: GameState,
        prediction: ShotPrediction | None,
        session: GameSession,
    ) -> ModeOutput:
        """Project the layout to set up, then the aiming line to shoot."""
        challenge = self.current
        if challenge is None:
            overlay = self.renderer.compose(
                game_state, None, session, self.mapper, feedback="No challenges loaded"
            )
            return ModeOutput(overlay=overlay, feedback_text="No challenges loaded")

        status = self.placement_status(game_state)
        ready = bool(status) and all(ok for _r, ok in status)
        if ready and not self.armed:
            self.armed = True
            self._feedback = challenge.description
            logger.info("trick shot %s armed", challenge.name)
        elif not ready and self.armed and session.state is SessionState.IDLE:
            # The layout was disturbed before the shot. Disarming stops the
            # attempt being graded against a layout that is no longer there.
            self.armed = False
            self._feedback = "Re-place the balls"

        # Latch the prediction while aiming: after the strike there is nothing
        # left to read the rail count from.
        if session.state is SessionState.AIMING and prediction is not None:
            self._shot_prediction = prediction

        def decorate(canvas: np.ndarray, ctx: object) -> None:
            if self.armed and session.state is not SessionState.IDLE:
                # Placement marks would sit under the balls during the shot.
                return
            for required, ok in status:
                draw_placement_marker(
                    canvas,
                    ctx,  # type: ignore[arg-type]
                    required.position,
                    satisfied=ok,
                    label=required.role.upper() if not ok else "",
                )

        header = f"#{self.index + 1}/{len(self.challenges)}  {challenge.name}"
        overlay = self.renderer.compose(
            game_state,
            prediction,
            session,
            self.mapper,
            feedback=f"{header} -- {self._feedback}",
            show_prediction=self.armed,
            decorate=decorate,
        )
        return ModeOutput(
            overlay=overlay,
            feedback_text=self._feedback,
            next_action=(
                challenge.description if self.armed else "Place the balls on the marks"
            ),
        )


def star_bar(stars: int) -> str:
    """Stars as text, for the control panel and the logs."""
    return "*" * max(0, min(3, stars)) + "." * (3 - max(0, min(3, stars)))


def challenge_summary(mode: TrickShotsMode) -> list[dict[str, object]]:
    """Every challenge with its best result, for ``/api/status`` and the panel."""
    summary: list[dict[str, object]] = []
    for position, challenge in enumerate(mode.challenges):
        result = mode.results.get(challenge.id)
        summary.append(
            {
                "index": position,
                "id": challenge.id,
                "name": challenge.name,
                "difficulty": challenge.difficulty,
                "stars": result.stars if result else 0,
                "completed": bool(result and result.success),
            }
        )
    return summary
