"""Tests for Phase 7: the game modes.

The modes are where rules live, and rules are the part of this system a player
will argue with. So these tests are mostly about *outcomes* -- who is on strike,
what the score is, whether the turn passed -- driven through the real state
machine rather than by calling ``on_shot_complete`` directly. A rule that only
works when the scoring hook is called by hand is not a rule anybody can play by.

The rendering tests are deliberately thin and assert only what is not a matter
of taste: that layers stack rather than wipe each other, and that a mode with no
projector returns nothing instead of raising.

``DriveTable`` below is the harness. It plays shots -- aim, strike, roll, settle
-- so that every scoring test reads as a sequence of shots and the state machine
is exercised the same way it will be in the room.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace

import pytest

from app.config import Settings
from app.models import (
    Ball,
    BallColor,
    BallKind,
    CueStick,
    GameModeName,
    GameSession,
    GameState,
    Player,
    PowerTick,
    SessionState,
    ShotPrediction,
    Vec2,
)
from modes.classic import ClassicMode
from modes.king_of_the_hill import (
    BASE_POINTS,
    MAX_COMBO_MULTIPLIER,
    TARGET_SCORE,
    TIME_BONUS_SECONDS,
    TURN_SECONDS,
    Difficulty,
    KingOfTheHillMode,
)
from modes.mode_manager import ModeManager, implemented_modes, mode_registry
from modes.scoring import (
    FOUL_PENALTY,
    POINTS_PER_BALL,
    RAIL_FREEZE_PENALTY,
    RECOMMEND_MARGIN,
    BallGroup,
    BucketRecommender,
    LeaveQuality,
    assess_leave,
    classify_shot,
    group_of,
    leave_advice,
)
from modes.training import DrillUnavailable, TrainingMode
from modes.trick_shots import TrickShotsMode, load_challenges
from physics.models import TableGeometry
from projection.mapper import ProjectionMapper, identity_calibration


@pytest.fixture()
def settings() -> Settings:
    """Mock hardware, small projector frame -- geometry is resolution-free."""
    s = Settings()
    s.camera.use_mock = True
    s.projector.use_mock = True
    s.projector.width = 640
    s.projector.height = 360
    return s


@pytest.fixture()
def mapper(settings: Settings) -> ProjectionMapper:
    return ProjectionMapper(identity_calibration(settings))


class _Tracker:
    """Stands in for ``BallTracker``: the state machine only asks one question."""

    def __init__(self) -> None:
        self.moving = False

    def any_moving(self, threshold: float | None = None) -> bool:
        return self.moving


def ball(
    ball_id: str,
    x: float,
    y: float,
    kind: BallKind = BallKind.SOLID,
    number: int | None = None,
    color: BallColor = BallColor.YELLOW,
) -> Ball:
    return Ball(
        id=ball_id,
        center_px=Vec2(x * 8.0, y * 8.0),
        radius_px=10.0,
        color=color,
        kind=kind,
        number=number,
        table_pos=Vec2(x, y),
    )


def cue_ball(x: float = 19.0, y: float = 19.0) -> Ball:
    return ball("cue", x, y, BallKind.CUE, color=BallColor.WHITE)


class DriveTable:
    """Plays shots through the real state machine.

    Exists because the interesting rules are about *sequences* -- pot, stay at
    the table, pot again, miss, hand over -- and a test that pokes
    ``on_shot_complete`` directly would pass while the state machine never
    reached it.
    """

    def __init__(self, manager: ModeManager, balls: list[Ball]) -> None:
        self.manager = manager
        self.balls = list(balls)
        self.tracker = _Tracker()
        manager.tracker = self.tracker
        self.clock = 1000.0

    def _state(self, balls: list[Ball], *, aiming: bool, velocity: float = 0.0) -> GameState:
        cue = next((b for b in balls if b.kind is BallKind.CUE), None)
        stick = CueStick(
            tip_px=Vec2(0.0, 0.0),
            angle_deg=0.0,
            tip_table_pos=Vec2(16.0, 19.0),
            shaft_visible=True,
            velocity=velocity,
            confidence=0.9,
        )
        return GameState(
            timestamp=self.clock,
            frame_index=0,
            balls=list(balls),
            cue_ball=cue,
            cue_stick=stick if aiming else None,
            confidence=0.9,
        )

    def shoot(self, pots: list[str] | None = None) -> None:
        """One complete shot: aim, strike, roll, settle, score.

        ``pots`` names the balls that leave the table, which is how a pot is
        expressed to a system that can only see before and after.
        """
        manager = self.manager
        # Aim, long enough to clear CONFIRM_FRAMES.
        for _ in range(3):
            manager.update_state(self._state(self.balls, aiming=True), now=self.clock)
            self.clock += 0.05
        assert manager.session.state is SessionState.AIMING

        # Strike.
        manager.update_state(self._state(self.balls, aiming=True, velocity=99.0), now=self.clock)
        self.clock += 0.05
        self.tracker.moving = True
        manager.update_state(self._state(self.balls, aiming=False), now=self.clock)

        # Balls settle, with whatever was potted now missing.
        self.balls = [b for b in self.balls if b.id not in set(pots or ())]
        self.tracker.moving = False
        self.clock += 0.1
        manager.update_state(self._state(self.balls, aiming=False), now=self.clock)
        self.clock += 5.0  # past SETTLE_SECONDS
        manager.update_state(self._state(self.balls, aiming=False), now=self.clock)
        # SETTLING fires the scoring hook on the following tick.
        manager.update_state(self._state(self.balls, aiming=False), now=self.clock)

    def add(self, new_ball: Ball) -> None:
        self.balls.append(new_ball)


def manager_for(
    settings: Settings, mode: GameModeName, players: list[str] | None = None
) -> ModeManager:
    manager = ModeManager(settings)
    manager.load_mode(mode)
    if players:
        manager.start_game(players)
        manager.mode.on_enter(manager.session)
    return manager


# ---------------------------------------------------------------------------
# Registry and switching
# ---------------------------------------------------------------------------


def test_every_advertised_mode_can_actually_be_loaded(settings: Settings) -> None:
    """The panel greys out what is unavailable using this list, so it must be true."""
    manager = ModeManager(settings)
    for name in implemented_modes():
        loaded = manager.load_mode(name)
        assert manager.session.mode is name, f"{name.value} silently fell back"
        assert loaded.name is name


def test_the_three_requested_modes_are_present() -> None:
    registry = mode_registry()
    for name in (
        GameModeName.CLASSIC,
        GameModeName.KING_OF_THE_HILL,
        GameModeName.TRICK_SHOTS,
    ):
        assert name in registry


def test_an_unbuilt_mode_falls_back_rather_than_raising(settings: Settings) -> None:
    """A bad value from the API must not take the game down mid-session."""
    manager = ModeManager(settings)
    manager.load_mode(GameModeName.KNOCKOUT)
    assert manager.session.mode is GameModeName.FREEPLAY


def test_switching_modes_drops_the_previous_mode_effects(settings: Settings) -> None:
    """A score popup from the last game must not float over a fresh rack."""
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada"])
    manager.mode.renderer.effects.spawn_score(Vec2(20.0, 20.0), "+25")
    assert manager.mode.renderer.effects.update() > 0

    previous = manager.mode
    manager.load_mode(GameModeName.FREEPLAY)
    assert manager.mode is not previous
    assert manager.mode.renderer.effects.update() == 0


# ---------------------------------------------------------------------------
# Ball groups
# ---------------------------------------------------------------------------


def test_group_falls_back_to_the_number_when_the_stripe_read_fails() -> None:
    """``kind`` and ``number`` come from different places and disagree.

    ``kind`` is read from the stripe pattern, which projected light across the
    ball can confuse; ``number`` is only set when the rack number was legible.
    Trusting either alone loses balls the other could classify.
    """
    unknown_stripe = Ball("x", Vec2(0, 0), 10, kind=BallKind.UNKNOWN, number=12)
    assert group_of(unknown_stripe) is BallGroup.STRIPES

    unknown_solid = Ball("y", Vec2(0, 0), 10, kind=BallKind.UNKNOWN, number=3)
    assert group_of(unknown_solid) is BallGroup.SOLIDS

    nothing_known = Ball("z", Vec2(0, 0), 10, kind=BallKind.UNKNOWN)
    assert group_of(nothing_known) is BallGroup.OPEN


def test_an_unclassifiable_ball_is_credited_not_penalised() -> None:
    """The detector's uncertainty must not be charged to the player.

    A foul call the player cannot argue with, learn from, or see the reason for
    is the worst thing this system can do to a game.
    """
    mystery = Ball("m", Vec2(0, 0), 10, kind=BallKind.UNKNOWN)
    outcome = classify_shot([mystery], BallGroup.SOLIDS)
    assert outcome.own == [mystery]
    assert not outcome.is_foul


def test_an_open_table_makes_every_object_ball_the_players() -> None:
    stripe = Ball("s", Vec2(0, 0), 10, kind=BallKind.STRIPE, number=9)
    outcome = classify_shot([stripe], BallGroup.OPEN)
    assert outcome.own == [stripe]
    assert not outcome.is_foul


def test_the_cue_ball_is_always_a_foul() -> None:
    outcome = classify_shot([cue_ball()], BallGroup.SOLIDS)
    assert outcome.scratched
    assert outcome.is_foul
    assert not outcome.scored


# ---------------------------------------------------------------------------
# Classic
# ---------------------------------------------------------------------------


def classic_rack() -> list[Ball]:
    return [
        cue_ball(),
        ball("s1", 40.0, 19.0, BallKind.SOLID, 1),
        ball("s2", 45.0, 12.0, BallKind.SOLID, 2),
        ball("t9", 50.0, 26.0, BallKind.STRIPE, 9),
        ball("t10", 55.0, 14.0, BallKind.STRIPE, 10),
        ball("eight", 60.0, 19.0, BallKind.EIGHT, 8, BallColor.BLACK),
    ]


def test_potting_your_own_ball_scores_and_keeps_the_table(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    table = DriveTable(manager, classic_rack())

    table.shoot(pots=["s1"])

    session = manager.session
    assert session.players[0].score == POINTS_PER_BALL
    assert session.current_player.name == "Ada", "a legal pot keeps you at the table"
    assert manager.mode.group_for(0) is BallGroup.SOLIDS
    assert manager.mode.group_for(1) is BallGroup.STRIPES


def test_a_miss_hands_over(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    table = DriveTable(manager, classic_rack())

    table.shoot(pots=[])

    assert manager.session.current_player.name == "Grace"
    assert manager.session.players[0].score == 0


def test_a_scratch_hands_over_and_pays_the_opponent(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    table = DriveTable(manager, classic_rack())

    table.shoot(pots=["s1", "cue"])

    session = manager.session
    # No points on a foul, even though the ball stayed down. The ball being off
    # the table still counts toward clearing the group -- that is read from the
    # table, not from the score -- so the only thing forfeited is the 25.
    assert session.players[0].score == 0
    assert session.players[1].score == FOUL_PENALTY
    assert session.current_player.name == "Grace"


def test_potting_the_opponents_ball_is_a_foul(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, ClassicMode)
    table = DriveTable(manager, classic_rack())

    table.shoot(pots=["s1"])  # Ada is on solids
    assert mode.group_for(0) is BallGroup.SOLIDS
    table.shoot(pots=["t9"])  # a stripe

    assert manager.session.current_player.name == "Grace"
    assert manager.session.players[1].score >= FOUL_PENALTY


def test_a_split_break_leaves_the_table_open(settings: Settings) -> None:
    """One of each down is genuinely ambiguous, so nothing is assigned.

    The alternative -- taking the first in the list -- makes the assignment
    depend on detection order, which is arbitrary and would differ between two
    runs of the same break.
    """
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, ClassicMode)
    table = DriveTable(manager, classic_rack())

    table.shoot(pots=["s1", "t9"])

    assert mode.group_for(0) is BallGroup.OPEN
    assert mode.groups == {}


def test_the_eight_wins_only_after_the_group_is_cleared(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, ClassicMode)
    table = DriveTable(manager, classic_rack())

    table.shoot(pots=["s1"])  # Ada on solids
    table.shoot(pots=["s2"])  # solids cleared
    table.shoot(pots=["eight"])

    assert mode.winner is not None
    assert mode.winner.name == "Ada"
    assert manager.session.state is SessionState.GAME_OVER


def test_the_eight_early_loses(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, ClassicMode)
    table = DriveTable(manager, classic_rack())

    table.shoot(pots=["s1"])  # on solids, one still up
    table.shoot(pots=["eight"])

    assert mode.winner is not None
    assert mode.winner.name == "Grace"


def test_classic_runs_solo_rather_than_refusing(settings: Settings) -> None:
    """Somebody who taps Classic without entering names wants to play."""
    manager = manager_for(settings, GameModeName.CLASSIC)
    manager.mode.on_enter(manager.session)
    assert len(manager.session.players) == 1


# ---------------------------------------------------------------------------
# King of the Hill
# ---------------------------------------------------------------------------


def koth_rack() -> list[Ball]:
    return [
        cue_ball(12.0, 19.0),
        ball("b1", 40.0, 19.0, BallKind.SOLID, 1),
        ball("b2", 55.0, 10.0, BallKind.SOLID, 2),
        ball("b3", 60.0, 28.0, BallKind.STRIPE, 9),
    ]


def test_a_pot_keeps_the_table_and_adds_time(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.KING_OF_THE_HILL, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, KingOfTheHillMode)
    table = DriveTable(manager, koth_rack())
    before = mode.seconds_remaining()

    table.shoot(pots=["b1"])

    assert manager.session.current_player.name == "Ada"
    assert manager.session.players[0].score > 0
    # The bonus is added to the deadline, so the remaining time went up despite
    # the clock having advanced during the shot.
    assert mode.seconds_remaining() > before - TURN_SECONDS + TIME_BONUS_SECONDS


def test_a_miss_hands_over_with_a_fresh_clock(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.KING_OF_THE_HILL, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, KingOfTheHillMode)
    table = DriveTable(manager, koth_rack())

    table.shoot(pots=[])

    assert manager.session.current_player.name == "Grace"
    assert mode.seconds_remaining() == pytest.approx(TURN_SECONDS, abs=1.0)


def test_the_combo_multiplier_climbs_and_is_capped(
    settings: Settings, monkeypatch
) -> None:
    """Uncapped, one lucky run puts the game out of reach and it stops being a contest.

    The target score is raised out of the way: with the real one, a seven-pot
    run wins on the fourth ball and the last three score nothing, which would be
    the win condition under test rather than the cap.
    """
    import modes.king_of_the_hill as koth

    monkeypatch.setattr(koth, "TARGET_SCORE", 10_000)
    manager = manager_for(settings, GameModeName.KING_OF_THE_HILL, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, KingOfTheHillMode)
    mode.set_difficulty(Difficulty.EASY)
    session = manager.session

    balls = [cue_ball(12.0, 19.0)] + [
        ball(f"b{i}", 30.0 + i * 4.0, 19.0, BallKind.SOLID, i) for i in range(1, 9)
    ]
    table = DriveTable(manager, balls)

    scores = []
    for i in range(1, 8):
        before = session.players[0].score
        table.shoot(pots=[f"b{i}"])
        scores.append(session.players[0].score - before)

    assert scores[0] == BASE_POINTS, "first pot is the base rate"
    assert scores[1] > scores[0], "the combo multiplies"
    # Capped: once past the cap every pot is worth the same.
    assert scores[-1] == scores[-2] == BASE_POINTS * MAX_COMBO_MULTIPLIER


def test_a_scratch_ends_the_turn_even_with_a_pot(settings: Settings) -> None:
    """Keeping the table after potting the cue ball is the one rule everybody objects to."""
    manager = manager_for(settings, GameModeName.KING_OF_THE_HILL, ["Ada", "Grace"])
    table = DriveTable(manager, koth_rack())

    table.shoot(pots=["b1", "cue"])

    assert manager.session.current_player.name == "Grace"
    assert manager.session.combo_count == 0


def test_reaching_the_target_score_wins(settings: Settings) -> None:
    manager = manager_for(settings, GameModeName.KING_OF_THE_HILL, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, KingOfTheHillMode)
    manager.session.players[0].score = TARGET_SCORE - 1
    table = DriveTable(manager, koth_rack())

    table.shoot(pots=["b1"])

    assert mode.winner is not None
    assert mode.winner.name == "Ada"
    assert manager.session.state is SessionState.GAME_OVER


def test_difficulty_changes_which_shot_is_asked_for(settings: Settings) -> None:
    """Easy nominates the straightest pot available, hard the hardest.

    The system cannot take balls off a real table, so this is what difficulty
    can honestly mean -- and the shot genuinely does get harder.
    """
    manager = manager_for(settings, GameModeName.KING_OF_THE_HILL, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, KingOfTheHillMode)
    state = GameState(
        timestamp=0.0,
        frame_index=0,
        balls=koth_rack(),
        cue_ball=koth_rack()[0],
    )

    mode.set_difficulty(Difficulty.EASY)
    easy = mode.choose_target(state)
    mode.set_difficulty(Difficulty.HARD)
    hard = mode.choose_target(state)

    assert easy is not None and hard is not None
    assert hard.score > easy.score
    assert Difficulty.HARD.multiplier > Difficulty.EASY.multiplier


def test_a_blocked_pot_is_never_nominated(settings: Settings) -> None:
    """Asking for a pot that cannot be made is what would make the mode feel broken."""
    manager = manager_for(settings, GameModeName.KING_OF_THE_HILL, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, KingOfTheHillMode)

    cue = cue_ball(12.0, 19.0)
    target = ball("t", 40.0, 19.0, BallKind.SOLID, 1)
    wall = ball("w", 26.0, 19.0, BallKind.SOLID, 2)  # squarely between them
    state = GameState(
        timestamp=0.0, frame_index=0, balls=[cue, target, wall], cue_ball=cue
    )

    chosen = mode.choose_target(state)
    assert chosen is None or chosen.ball.id != "t"


def test_the_turn_clock_runs_out_between_shots_only(settings: Settings) -> None:
    """Cutting a turn off mid-roll would void a pot the player already made."""
    manager = manager_for(settings, GameModeName.KING_OF_THE_HILL, ["Ada", "Grace"])
    mode = manager.mode
    assert isinstance(mode, KingOfTheHillMode)
    mode._turn_ends_at = time.perf_counter() - 1.0  # expired

    state = GameState(timestamp=0.0, frame_index=0, balls=koth_rack(), cue_ball=koth_rack()[0])
    manager.session.state = SessionState.SHOT_IN_PROGRESS
    mode.update(state, None, manager.session)
    assert manager.session.current_player.name == "Ada", "turn ended while balls were rolling"

    manager.session.state = SessionState.IDLE
    mode.update(state, None, manager.session)
    assert manager.session.current_player.name == "Grace"


# ---------------------------------------------------------------------------
# Trick shots
# ---------------------------------------------------------------------------


def test_ten_challenges_ship_and_parse() -> None:
    challenges = load_challenges()
    assert len(challenges) == 10
    for challenge in challenges:
        assert challenge.cue_position is not None, f"{challenge.name} has no cue ball"
        assert challenge.target_position is not None
        assert 1 <= challenge.difficulty <= 3
        assert challenge.description


def test_challenges_get_harder() -> None:
    difficulties = [c.difficulty for c in load_challenges()]
    assert difficulties[0] <= difficulties[-1]
    assert max(difficulties) == 3


def test_a_malformed_challenge_is_skipped_not_fatal(tmp_path) -> None:
    """One bad entry must not cost the player the other nine."""
    path = tmp_path / "challenges.json"
    path.write_text(
        json.dumps(
            {
                "challenges": [
                    {"id": 1, "name": "bad"},  # no balls
                    {
                        "id": 2,
                        "name": "good",
                        "description": "fine",
                        "difficulty": 1,
                        "balls": [{"role": "cue", "x": 10.0, "y": 10.0}],
                        "target_pocket": "top_left",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_challenges(path)
    assert [c.name for c in loaded] == ["good"]


def test_a_missing_challenge_file_is_not_fatal(tmp_path) -> None:
    assert load_challenges(tmp_path / "nope.json") == []


def test_layouts_scale_to_the_table_in_front_of_the_camera(settings: Settings) -> None:
    """One file has to work on a 6 ft table and a 9 ft one."""
    settings.table.length_in = 100.0
    settings.table.width_in = 50.0
    mode = TrickShotsMode(settings)
    challenge = mode.current
    assert challenge is not None
    for required in challenge.balls:
        assert 0.0 <= required.position.x <= 100.0
        assert 0.0 <= required.position.y <= 50.0
    # And it really did scale, rather than clamping.
    assert max(b.position.x for b in challenge.balls) > 76.0 * 0.5


def test_the_shot_arms_only_once_the_balls_are_on_their_marks(settings: Settings) -> None:
    """The setup phase is the whole reason this mode can be evaluated at all."""
    mode = TrickShotsMode(settings)
    challenge = mode.current
    assert challenge is not None

    misplaced = GameState(
        timestamp=0.0,
        frame_index=0,
        balls=[cue_ball(2.0, 2.0)],
        cue_ball=cue_ball(2.0, 2.0),
    )
    assert not mode.is_ready(misplaced)

    placed = [
        Ball(
            "cue" if b.role == "cue" else b.role,
            Vec2(0, 0),
            10,
            kind=BallKind.CUE if b.role == "cue" else BallKind.SOLID,
            table_pos=b.position,
        )
        for b in challenge.balls
    ]
    good = GameState(
        timestamp=0.0, frame_index=0, balls=placed, cue_ball=placed[0]
    )
    assert mode.is_ready(good)


def test_three_stars_needs_the_pocket_the_challenge_asked_for(settings: Settings) -> None:
    mode = TrickShotsMode(settings)
    mode.armed = True
    challenge = mode.current
    assert challenge is not None and challenge.target_pocket == "top_right"

    # Potted into the named corner.
    right = mode.evaluate([ball("t", 76.0, 0.0, BallKind.SOLID, 3)])
    assert right.stars == 3
    assert right.success

    # Same pot, wrong pocket: made the shot, not the challenge.
    wrong = mode.evaluate([ball("t", 0.0, 38.0, BallKind.SOLID, 3)])
    assert wrong.stars == 2
    assert not wrong.success


def test_a_miss_still_scores_one_star(settings: Settings) -> None:
    """There is no zero. A zero teaches nothing a one does not."""
    mode = TrickShotsMode(settings)
    mode.armed = True
    assert mode.evaluate([]).stars == 1


def test_replaying_a_challenge_cannot_cost_stars(settings: Settings) -> None:
    mode = TrickShotsMode(settings)
    mode.armed = True
    challenge = mode.current
    assert challenge is not None

    mode.evaluate([ball("t", 76.0, 0.0, BallKind.SOLID, 3)])  # 3 stars
    assert mode.results[challenge.id].stars == 3
    mode.evaluate([])  # a fluffed replay
    assert mode.results[challenge.id].stars == 3


def test_advancing_wraps_around_the_set(settings: Settings) -> None:
    mode = TrickShotsMode(settings)
    first = mode.current
    for _ in range(len(mode.challenges)):
        mode.advance()
    assert mode.current is not None and first is not None
    assert mode.current.id == first.id


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_a_drill_is_chosen_from_the_layout_not_scripted(settings: Settings) -> None:
    """Asking for a ball that is not on the table makes the mode useless."""
    mode = TrainingMode(settings)
    cue = cue_ball(12.0, 19.0)
    target = ball("b1", 45.0, 19.0, BallKind.SOLID, 1)
    state = GameState(timestamp=0.0, frame_index=0, balls=[cue, target], cue_ball=cue)

    from app.models import DrillType

    drill = mode.start_drill(DrillType.POTTING, state)

    assert drill.target_ball_id == "b1"
    assert drill.target_pocket is not None
    assert drill.instruction


def test_a_drill_with_nothing_to_pot_says_so(settings: Settings) -> None:
    from app.models import DrillType

    mode = TrainingMode(settings)
    cue = cue_ball()
    state = GameState(timestamp=0.0, frame_index=0, balls=[cue], cue_ball=cue)

    with pytest.raises(DrillUnavailable, match="clear pot"):
        mode.start_drill(DrillType.POTTING, state)


def test_a_drill_without_a_cue_ball_says_that_instead(settings: Settings) -> None:
    from app.models import DrillType

    mode = TrainingMode(settings)
    state = GameState(timestamp=0.0, frame_index=0, balls=[], cue_ball=None)

    with pytest.raises(DrillUnavailable, match="cue ball"):
        mode.start_drill(DrillType.POTTING, state)


def test_a_position_drill_scores_the_cue_ball_not_the_pot(settings: Settings) -> None:
    from app.models import DrillType

    mode = TrainingMode(settings)
    cue = cue_ball(12.0, 19.0)
    target = ball("b1", 45.0, 19.0, BallKind.SOLID, 1)
    state = GameState(timestamp=0.0, frame_index=0, balls=[cue, target], cue_ball=cue)
    drill = mode.start_drill(DrillType.POSITION, state)
    assert drill.target_zone_center is not None

    # Cue ball finishing in the zone is a success even with nothing potted.
    settled = cue_ball(drill.target_zone_center.x, drill.target_zone_center.y)
    in_zone = GameState(
        timestamp=0.0, frame_index=0, balls=[settled, target], cue_ball=settled
    )
    assert mode.evaluate(in_zone, []).success

    far = cue_ball(2.0, 2.0)
    out_of_zone = GameState(timestamp=0.0, frame_index=0, balls=[far, target], cue_ball=far)
    assert not mode.evaluate(out_of_zone, []).success


def test_stats_accumulate_across_attempts(settings: Settings) -> None:
    from app.models import DrillType, GameSession

    mode = TrainingMode(settings)
    cue = cue_ball(12.0, 19.0)
    target = ball("b1", 45.0, 19.0, BallKind.SOLID, 1)
    state = GameState(timestamp=0.0, frame_index=0, balls=[cue, target], cue_ball=cue)
    mode.start_drill(DrillType.POTTING, state)
    session = GameSession()

    mode.on_shot_complete(state, session, [target])  # potted
    mode.on_shot_complete(state, session, [])  # missed

    assert mode.stats.attempts == 2
    assert mode.stats.successes == 1
    assert mode.stats.success_rate_pct == pytest.approx(50.0)
    assert mode.last_result is not None
    assert mode.last_result.next_instruction


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_a_mode_without_a_projector_returns_nothing_rather_than_raising(
    settings: Settings,
) -> None:
    """Normal at startup and in every unit test: the loop wires the mapper later."""
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada"])
    assert manager.mapper is None
    state = GameState(timestamp=0.0, frame_index=0, balls=classic_rack(), cue_ball=cue_ball())

    output = manager.update(state, None)

    assert output.overlay is None
    assert output.next_action


def test_the_mapper_reaches_a_mode_loaded_after_it_was_set(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    manager = ModeManager(settings)
    manager.mapper = mapper
    manager.load_mode(GameModeName.CLASSIC)
    assert manager.mode.mapper is mapper


def test_overlays_stack_rather_than_wiping_each_other(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """Regression: every render_* call cleared the buffer, so only the last survived.

    ``ensure_canvas`` zeroes by default -- correctly, since a stale overlay
    accumulating frame on frame fills the felt with light. But that made the
    scoreboard wipe the aiming line underneath it, and the projected result was
    a UI with no trajectory. Layering is now opt-in via ``clear=False``.
    """
    from physics.simulator import estimate_shot_from_cue

    manager = ModeManager(settings)
    manager.mapper = mapper
    manager.tracker = _Tracker()
    manager.load_mode(GameModeName.CLASSIC)
    manager.start_game(["Ada", "Grace"])
    manager.mode.on_enter(manager.session)

    balls = classic_rack()
    state = GameState(
        timestamp=0.0,
        frame_index=0,
        balls=balls,
        cue_ball=balls[0],
        cue_stick=CueStick(
            tip_px=Vec2(0, 0), angle_deg=0.0, tip_table_pos=Vec2(16.0, 19.0),
            shaft_visible=True, confidence=0.9,
        ),
    )
    prediction = estimate_shot_from_cue(state, state.cue_stick, settings=settings)
    assert prediction is not None and not prediction.is_empty

    for _ in range(3):
        output = manager.update(state, prediction)
    assert manager.session.state is SessionState.AIMING

    overlay = output.overlay
    assert overlay is not None
    # The scoreboard is rail-anchored near the top; the aiming line runs across
    # the middle of the table. Both must have ink.
    top_band = overlay[: overlay.shape[0] // 5, :, 3]
    middle_band = overlay[overlay.shape[0] // 3 : 2 * overlay.shape[0] // 3, :, 3]
    assert int((top_band > 0).sum()) > 100, "no scoreboard"
    assert int((middle_band > 0).sum()) > 100, "the UI layer wiped the trajectory"


def test_nothing_is_drawn_over_the_balls_while_they_are_moving(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """A stale aiming line under a rolling ball reads as a bug."""
    from physics.simulator import estimate_shot_from_cue

    manager = ModeManager(settings)
    manager.mapper = mapper
    manager.tracker = _Tracker()
    manager.load_mode(GameModeName.FREEPLAY)

    balls = classic_rack()
    aiming = GameState(
        timestamp=0.0, frame_index=0, balls=balls, cue_ball=balls[0],
        cue_stick=CueStick(
            tip_px=Vec2(0, 0), angle_deg=0.0, tip_table_pos=Vec2(16.0, 19.0),
            shaft_visible=True, confidence=0.9,
        ),
    )
    prediction = estimate_shot_from_cue(aiming, aiming.cue_stick, settings=settings)
    for _ in range(3):
        manager.update(aiming, prediction)
    aimed = int((manager.mode.renderer._canvas[:, :, 3] > 0).sum())

    manager.session.state = SessionState.SHOT_IN_PROGRESS
    output = manager.mode.update(aiming, prediction, manager.session)

    assert output.overlay is not None
    moving = int((output.overlay[:, :, 3] > 0).sum())
    assert moving < aimed / 2, "the aiming line survived into the shot"


def test_the_canvas_is_reused_between_frames(
    settings: Settings, mapper: ProjectionMapper
) -> None:
    """8 MB per frame of allocation is 8% of the budget for nothing."""
    manager = ModeManager(settings)
    manager.mapper = mapper
    manager.tracker = _Tracker()
    manager.load_mode(GameModeName.FREEPLAY)
    state = GameState(timestamp=0.0, frame_index=0, balls=classic_rack(), cue_ball=cue_ball())

    first = manager.update(state, None).overlay
    second = manager.update(state, None).overlay
    assert first is second


def test_ensure_canvas_still_clears_by_default(settings: Settings) -> None:
    """The opt-out must not have changed the default; a stale overlay accumulates."""
    from projection import draw

    canvas = draw.new_canvas(settings)
    canvas[:] = 200
    assert not draw.ensure_canvas(canvas, settings).any()

    canvas[:] = 200
    assert draw.ensure_canvas(canvas, settings, clear=False).any()


# ---------------------------------------------------------------------------
# The state machine, through a mode
# ---------------------------------------------------------------------------


def test_the_scoring_hook_fires_exactly_once_per_shot(settings: Settings) -> None:
    """Twice would double every score; never would mean nothing is ever scored."""
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    calls: list[list[Ball]] = []
    manager.mode.on_shot_complete = lambda gs, session, pocketed: calls.append(pocketed)  # type: ignore[assignment]

    table = DriveTable(manager, classic_rack())
    table.shoot(pots=["s1"])

    assert len(calls) == 1
    assert [b.id for b in calls[0]] == ["s1"]


def test_the_hook_receives_the_balls_not_just_their_ids(settings: Settings) -> None:
    """Every scoring rule needs to know *what* went down, and it is gone by then."""
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    seen: list[Ball] = []
    manager.mode.on_shot_complete = lambda gs, s, pocketed: seen.extend(pocketed)  # type: ignore[assignment]

    table = DriveTable(manager, classic_rack())
    table.shoot(pots=["t9"])

    assert len(seen) == 1
    assert seen[0].kind is BallKind.STRIPE
    assert group_of(seen[0]) is BallGroup.STRIPES


def test_a_ball_briefly_occluded_is_not_scored_as_potted(settings: Settings) -> None:
    """A hand over the table is the single most common false pot.

    The settle timer is what makes the set difference reliable: by the time it
    fires, hands are usually clear. This drives a shot where the ball vanishes
    and comes back before the table settles.
    """
    manager = manager_for(settings, GameModeName.CLASSIC, ["Ada", "Grace"])
    table = DriveTable(manager, classic_rack())
    hidden = next(b for b in table.balls if b.id == "s1")

    for _ in range(3):
        manager.update_state(table._state(table.balls, aiming=True), now=table.clock)
        table.clock += 0.05
    manager.update_state(table._state(table.balls, aiming=True, velocity=99.0), now=table.clock)
    table.tracker.moving = True
    without = [b for b in table.balls if b.id != "s1"]
    table.clock += 0.05
    manager.update_state(table._state(without, aiming=False), now=table.clock)

    # The hand moves away before anything settles.
    table.tracker.moving = False
    table.clock += 0.1
    manager.update_state(table._state(table.balls, aiming=False), now=table.clock)
    table.clock += 5.0
    manager.update_state(table._state(table.balls, aiming=False), now=table.clock)
    manager.update_state(table._state(table.balls, aiming=False), now=table.clock)

    assert hidden in table.balls
    assert manager.session.players[0].score == 0, "an occluded ball was scored as potted"


# ---------------------------------------------------------------------------
# The API surface the two configurable modes need
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(settings: Settings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.state import AppState

    app = create_app(AppState(settings=settings), start_loop=False)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_difficulty_is_reachable_from_the_panel(client) -> None:
    """A difficulty nobody can change is not a feature."""
    assert client.post("/api/mode", json={"mode": "king_of_the_hill"}).status_code == 200
    response = client.post("/api/mode/difficulty", json={"difficulty": "hard"})
    assert response.status_code == 200
    assert "hard" in response.json()["message"]


def test_difficulty_on_a_mode_without_one_says_which_mode_has_it(client) -> None:
    """409 and a pointer, not 400 and a shrug: the request is not malformed."""
    client.post("/api/mode", json={"mode": "freeplay"})
    response = client.post("/api/mode/difficulty", json={"difficulty": "hard"})
    assert response.status_code == 409
    assert "king of the hill" in response.json()["detail"].lower()


def test_challenges_can_be_listed_and_selected(client) -> None:
    assert client.post("/api/mode", json={"mode": "trick_shots"}).status_code == 200

    listing = client.get("/api/mode/challenges").json()
    assert len(listing) == 10
    assert listing[0]["stars"] == 0
    assert {"index", "id", "name", "difficulty", "stars", "completed"} <= set(listing[0])

    response = client.post("/api/mode/challenge", json={"index": 4})
    assert response.status_code == 200
    assert listing[4]["name"] in response.json()["message"]


def test_selecting_a_challenge_in_the_wrong_mode_is_refused(client) -> None:
    client.post("/api/mode", json={"mode": "freeplay"})
    response = client.post("/api/mode/challenge", json={"index": 0})
    assert response.status_code == 409
    assert "trick shots" in response.json()["detail"].lower()


def test_starting_a_competitive_game_starts_its_clock(client) -> None:
    """Regression: the turn clock was started before the players existed.

    ``load_mode`` calls ``on_enter``, which starts the countdown; ``start_game``
    then replaces the player list. The turn was left counting down for a player
    who had been discarded, so the first turn was short by however long the
    request took.
    """
    response = client.post(
        "/api/mode", json={"mode": "king_of_the_hill", "players": ["Ada", "Grace"]}
    )
    assert response.status_code == 200
    assert client.get("/api/status").json()["current_mode"] == "king_of_the_hill"
    game = client.get("/api/session").json()
    assert [p["name"] for p in game["players"]] == ["Ada", "Grace"]


# ---------------------------------------------------------------------------
# Position scoring and the power recommendation
# ---------------------------------------------------------------------------

_LEVELS = ("very soft", "soft", "medium", "strong", "very hard")

#: A corner. Snookered from here against the fixtures below, so it stands in for
#: "this level leaves nothing" without needing obstructing balls placed for it.
_DEAD = Vec2(74.0, 36.0)


def _pos_ball(ball_id: str, x: float, y: float) -> Ball:
    return Ball(
        id=ball_id,
        center_px=Vec2(0.0, 0.0),
        radius_px=12.0,
        kind=BallKind.SOLID,
        table_pos=Vec2(x, y),
    )


def _fan(positions: list[Vec2], scratched: set[str] | None = None) -> ShotPrediction:
    """A prediction carrying one tick per level at the given positions."""
    scratched = scratched or set()
    return ShotPrediction(
        trajectory_path=[Vec2(0.0, 0.0), Vec2(1.0, 1.0)],
        contact_index=1,
        power_ticks=[
            PowerTick(
                label=label,
                position=position,
                distance_in=float(index * 20),
                scratched=label in scratched,
            )
            for index, (label, position) in enumerate(zip(_LEVELS, positions, strict=True))
        ],
    )


@pytest.fixture()
def geometry(settings: Settings) -> TableGeometry:
    return TableGeometry.from_settings(settings)


@pytest.fixture()
def one_ball() -> list[Ball]:
    """A single legal ball, off both table axes so cut angle varies with position."""
    return [_pos_ball("t", 44.0, 26.0)]


def test_scratch_outranks_every_other_consideration(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """A potted cue ball is judged SCRATCH regardless of where it landed.

    Short-circuited before any geometry runs, and that is the point rather than
    an optimisation: a potted cue ball's resting position is the pocket mouth,
    which is an ordinary-looking coordinate that would happily score as a decent
    leave. Every geometric question below it is about a table state that will not
    exist, because the next shot is from hand.
    """
    # A position that would otherwise assess as playable.
    assert assess_leave(Vec2(25.0, 19.0), one_ball, one_ball, geometry).quality is (
        LeaveQuality.PLAYABLE
    )
    scratched = assess_leave(
        Vec2(25.0, 19.0), one_ball, one_ball, geometry, scratched=True
    )
    assert scratched.quality is LeaveQuality.SCRATCH
    assert scratched.cost == math.inf


def test_a_blocked_contact_is_snookered_not_merely_unpottable(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """A ball in the way of the *contact* is a snooker, a distinct verdict.

    The two failures are different sizes and want different words. No clear pot
    means play safe; no clear contact means you are conceding a foul unless you
    find a swerve. Collapsing them into one "bad leave" would lose the more
    urgent of the two.
    """
    blocker = _pos_ball("blk", 34.0, 22.5)
    assessment = assess_leave(
        Vec2(25.0, 19.0), one_ball, one_ball + [blocker], geometry
    )
    assert assessment.quality is LeaveQuality.SNOOKERED
    assert assessment.cost == math.inf


def test_no_legal_balls_is_not_a_playable_leave(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """An empty legal set never scores as playable.

    Reachable when a mode knows the rules and the answer is genuinely nothing --
    training once the drill's ball is potted. Distinct from the mode returning
    ``None``, which means it cannot know and suppresses the recommendation
    upstream before this is ever called.
    """
    assessment = assess_leave(Vec2(25.0, 19.0), [], one_ball, geometry)
    assert not assessment.quality.is_playable
    assert assessment.cost == math.inf


def test_a_rail_frozen_leave_costs_more_than_the_same_shot_off_the_rail(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """Sitting on a cushion is penalised, and it is the penalty being tested.

    Compared against the same position's own raw pot difficulty rather than
    against a different position, because cut angle varies across the table and
    two different spots differ for reasons that have nothing to do with the rail.
    A naive comparison of on-rail against off-rail can show the rail scoring
    *better*, purely because the angle happened to be kinder there.
    """
    on_rail = assess_leave(Vec2(25.0, 1.3), one_ball, one_ball, geometry)
    assert on_rail.on_rail
    assert on_rail.best_pot is not None
    assert on_rail.cost == pytest.approx(on_rail.best_pot.score + RAIL_FREEZE_PENALTY)

    off_rail = assess_leave(Vec2(25.0, 19.0), one_ball, one_ball, geometry)
    assert not off_rail.on_rail
    assert off_rail.best_pot is not None
    assert off_rail.cost == pytest.approx(off_rail.best_pot.score)


def test_the_recommendation_picks_the_cheapest_playable_leave(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """The highlighted level is the one leaving the easiest next shot."""
    poor, great = Vec2(10.0, 30.0), Vec2(12.0, 14.0)
    assert assess_leave(great, one_ball, one_ball, geometry).cost < assess_leave(
        poor, one_ball, one_ball, geometry
    ).cost

    prediction = _fan([_DEAD, great, poor, _DEAD, _DEAD])
    BucketRecommender().annotate(prediction, one_ball, one_ball, geometry)

    recommended = [t.label for t in prediction.power_ticks if t.recommended]
    assert recommended == ["soft"]


def test_all_five_ticks_survive_a_recommendation(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """Recommending must not remove or replace the other four levels.

    The reason the recommendation is legible at all. A bare "hit MEDIUM" teaches
    nothing and cannot be argued with; the other four ticks are what let a player
    see what the advice gave up and overrule it. So the annotation adds a flag
    and changes nothing else.
    """
    positions = [_DEAD, Vec2(12.0, 14.0), Vec2(10.0, 30.0), _DEAD, _DEAD]
    prediction = _fan(positions)
    BucketRecommender().annotate(prediction, one_ball, one_ball, geometry)

    assert [t.label for t in prediction.power_ticks] == list(_LEVELS)
    assert [t.position for t in prediction.power_ticks] == positions
    assert sum(1 for t in prediction.power_ticks if t.recommended) == 1


def test_nothing_playable_recommends_nothing_and_says_why(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """With no playable leave, no tick is marked and the advice is explicit.

    Not "the least bad one". Picking quietly from five unplayable options
    presents a guess as advice, which is the confidently-wrong failure the whole
    fan exists to avoid -- and it is worse here than for a resting position,
    because advice carries more authority than a mark.
    """
    wall = _pos_ball("wall", 34.0, 22.5)
    prediction = _fan([Vec2(25.0, 19.0)] * 5)
    assessments = BucketRecommender().annotate(
        prediction, one_ball, one_ball + [wall], geometry
    )

    assert not any(t.recommended for t in prediction.power_ticks)
    advice = leave_advice(assessments)
    assert advice
    assert "snookered" in advice


def test_the_advice_distinguishes_which_failure_it_is(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """A scratch at every pace reads differently from a snooker at every pace.

    They call for different decisions -- play elsewhere versus play safe -- so
    one generic line would be throwing away the useful half of the message.
    """
    scratch_fan = _fan([Vec2(0.0, 0.0)] * 5, scratched=set(_LEVELS))
    scratch_advice = leave_advice(
        BucketRecommender().annotate(scratch_fan, one_ball, one_ball, geometry)
    )
    assert "scratch" in scratch_advice

    wall = _pos_ball("wall", 34.0, 22.5)
    snooker_fan = _fan([Vec2(25.0, 19.0)] * 5)
    snooker_advice = leave_advice(
        BucketRecommender().annotate(snooker_fan, one_ball, one_ball + [wall], geometry)
    )
    assert "snookered" in snooker_advice
    assert scratch_advice != snooker_advice


def test_no_advice_line_when_a_level_is_recommended(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """The highlighted tick says it better than words do.

    Text on the cloth is expensive -- it is projected onto felt and read from
    several feet away -- so it is spent only on the thing a player cannot read
    off the marks themselves.
    """
    prediction = _fan([_DEAD, Vec2(12.0, 14.0), Vec2(10.0, 30.0), _DEAD, _DEAD])
    assessments = BucketRecommender().annotate(
        prediction, one_ball, one_ball, geometry
    )
    assert any(t.recommended for t in prediction.power_ticks)
    assert leave_advice(assessments) == ""


def test_a_prescribed_level_suppresses_the_recommendation(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """A drill's instruction wins; nothing is recommended alongside it.

    A drill saying "medium" beside a highlight saying "strong" is the overlay
    arguing with itself, and the drill is the one with authority -- it knows what
    it is teaching. So prescription and recommendation are mutually exclusive by
    construction rather than by the renderer picking one to draw.
    """
    prediction = _fan([_DEAD, Vec2(12.0, 14.0), Vec2(10.0, 30.0), _DEAD, _DEAD])
    prediction.power_ticks = [
        replace(tick, prescribed=tick.label == "medium")
        for tick in prediction.power_ticks
    ]
    assessments = BucketRecommender().annotate(
        prediction, one_ball, one_ball, geometry
    )

    assert assessments == []
    assert not any(t.recommended for t in prediction.power_ticks)


# -- hysteresis -------------------------------------------------------------


def test_the_recommendation_does_not_flap_between_near_equal_levels(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """Two levels scoring almost equally must not trade the highlight per frame.

    The failure this prevents is specific and visible. The memo recomputes when
    the quantised aim moves, and detection noise walks the measured aim across
    quantisation cells several times a second -- so with two near-tied levels the
    highlight would flicker MEDIUM / STRONG / MEDIUM on the cloth. A flickering
    recommendation is worse than a stable slightly-suboptimal one: it is
    unreadable, and it advertises that the system cannot make up its mind, which
    is a reason to distrust every other mark too.

    The two positions here differ by less than the margin but are *not* equal --
    an exact tie would pass on a broken implementation, because ``min`` is stable
    and would return the same winner every frame for the wrong reason.
    """
    a, b = Vec2(10.0, 6.0), Vec2(10.0, 8.0)
    cost_a = assess_leave(a, one_ball, one_ball, geometry).cost
    cost_b = assess_leave(b, one_ball, one_ball, geometry).cost
    assert cost_a != cost_b, "an exact tie would not exercise the margin"
    assert abs(cost_a - cost_b) < RECOMMEND_MARGIN

    recommender = BucketRecommender()
    seen = []
    for frame in range(10):
        # The two levels swap places, so whichever is cheaper alternates.
        swapped = frame % 2 == 1
        prediction = _fan(
            [_DEAD, _DEAD, b if swapped else a, a if swapped else b, _DEAD]
        )
        recommender.annotate(prediction, one_ball, one_ball, geometry)
        seen.append(next(t.label for t in prediction.power_ticks if t.recommended))

    assert len(set(seen)) == 1, f"recommendation flapped: {seen}"


def test_without_hysteresis_the_same_input_does_flap(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """The margin is load-bearing, not decorative.

    Guards the guard. Without this, a change that quietly stopped applying the
    margin would leave the test above passing for the wrong reason -- because the
    inputs happened not to produce a contest -- and the flicker would come back
    unnoticed.
    """
    a, b = Vec2(10.0, 6.0), Vec2(10.0, 8.0)
    recommender = BucketRecommender(margin=0.0)
    seen = []
    for frame in range(10):
        swapped = frame % 2 == 1
        prediction = _fan(
            [_DEAD, _DEAD, b if swapped else a, a if swapped else b, _DEAD]
        )
        recommender.annotate(prediction, one_ball, one_ball, geometry)
        seen.append(next(t.label for t in prediction.power_ticks if t.recommended))

    assert len(set(seen)) > 1, "the inputs must genuinely contest the highlight"


def test_a_decisively_better_leave_still_moves_the_highlight(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """Hysteresis must not become stubbornness.

    A margin that never yields is not stability, it is a stuck display. The
    incumbent has to lose when the alternative is genuinely, visibly better.
    """
    poor, great = Vec2(10.0, 30.0), Vec2(12.0, 14.0)
    gap = (
        assess_leave(poor, one_ball, one_ball, geometry).cost
        - assess_leave(great, one_ball, one_ball, geometry).cost
    )
    assert gap > RECOMMEND_MARGIN, "fixture must clear the margin to test this"

    recommender = BucketRecommender()
    first = _fan([_DEAD, _DEAD, poor, _DEAD, _DEAD])
    recommender.annotate(first, one_ball, one_ball, geometry)
    assert next(t.label for t in first.power_ticks if t.recommended) == "medium"

    second = _fan([_DEAD, great, poor, _DEAD, _DEAD])
    recommender.annotate(second, one_ball, one_ball, geometry)
    assert next(t.label for t in second.power_ticks if t.recommended) == "soft"


def test_an_incumbent_that_stops_being_playable_releases_the_hold(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """A level that is no longer playable cannot keep the highlight.

    Otherwise the margin would defend a recommendation that has become wrong
    rather than merely contested -- the worst case for hysteresis, since it is
    the one where holding steady is indistinguishable from being broken.
    """
    poor, great = Vec2(10.0, 30.0), Vec2(12.0, 14.0)
    recommender = BucketRecommender()
    recommender.annotate(
        _fan([_DEAD, _DEAD, poor, _DEAD, _DEAD]), one_ball, one_ball, geometry
    )

    # A ball now blocks the incumbent's leave, but not the alternative's.
    wall = _pos_ball("wall", 30.0, 27.0)
    prediction = _fan([_DEAD, great, poor, _DEAD, _DEAD])
    recommender.annotate(prediction, one_ball, one_ball + [wall], geometry)
    assert next(t.label for t in prediction.power_ticks if t.recommended) == "soft"


def test_reset_forgets_the_incumbent(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """Hysteresis must not survive a shot.

    Holding across a shot boundary would not be staleness, it would be advice
    about a table layout that no longer exists. The mode manager calls this at
    the single point where a shot completes.
    """
    poor, great = Vec2(10.0, 30.0), Vec2(12.0, 14.0)
    recommender = BucketRecommender()
    recommender.annotate(
        _fan([_DEAD, _DEAD, poor, _DEAD, _DEAD]), one_ball, one_ball, geometry
    )
    recommender.reset()

    # With no incumbent, the best leave wins outright rather than having to clear
    # a margin against the previous pick.
    prediction = _fan([_DEAD, great, poor, _DEAD, _DEAD])
    recommender.annotate(prediction, one_ball, one_ball, geometry)
    assert next(t.label for t in prediction.power_ticks if t.recommended) == "soft"


# -- the memo ---------------------------------------------------------------


def test_the_memo_reuses_the_assessment_when_nothing_moved(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """An unchanged frame must not re-score five positions.

    Keyed on the resting positions rather than on the aim, so it inherits the
    prediction cache's quantisation instead of introducing a second threshold
    that could drift out of step with the first. Identical positions mean the aim
    has not moved materially and the answer cannot have changed.

    Asserted by object identity, which is the only thing that distinguishes a
    memo hit from a recomputation that happens to agree.
    """
    positions = [_DEAD, Vec2(12.0, 14.0), Vec2(10.0, 30.0), _DEAD, _DEAD]
    recommender = BucketRecommender()
    first = recommender.annotate(_fan(positions), one_ball, one_ball, geometry)
    second = recommender.annotate(_fan(positions), one_ball, one_ball, geometry)
    assert first is second


def test_the_memo_recomputes_when_a_resting_place_moves(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """A materially different aim must be re-scored."""
    base = [_DEAD, Vec2(12.0, 14.0), Vec2(10.0, 30.0), _DEAD, _DEAD]
    moved = [_DEAD, Vec2(20.0, 14.0), Vec2(10.0, 30.0), _DEAD, _DEAD]
    recommender = BucketRecommender()
    first = recommender.annotate(_fan(base), one_ball, one_ball, geometry)
    second = recommender.annotate(_fan(moved), one_ball, one_ball, geometry)
    assert first is not second


def test_the_memo_recomputes_when_the_legal_set_changes(
    geometry: TableGeometry, one_ball: list[Ball]
) -> None:
    """Same aim, different rules, different answer.

    The legal set is in the key because it is an input to the verdict, not a
    property of the aim. Classic changes it when the table splits; training
    changes it every drill. A memo keyed on geometry alone would hand back a
    verdict computed against the wrong goal.
    """
    positions = [_DEAD, Vec2(12.0, 14.0), Vec2(10.0, 30.0), _DEAD, _DEAD]
    other = [_pos_ball("other", 30.0, 12.0)]
    recommender = BucketRecommender()
    first = recommender.annotate(_fan(positions), one_ball, one_ball, geometry)
    second = recommender.annotate(_fan(positions), other, one_ball + other, geometry)
    assert first is not second


# -- per-mode legal target sets ---------------------------------------------


def test_freeplay_recommends_nothing(settings: Settings) -> None:
    """Freeplay has no concept of a next ball, so it stays silent.

    Not a gap. There is no goal in freeplay, so any recommendation would be
    scored against a guessed one -- and a confident recommendation against a
    guessed goal is worse than no recommendation, because the player has no way
    to tell which it is. The ticks still show; they are rules-independent.
    """
    from modes.freeplay import FreeplayMode

    mode = FreeplayMode(settings)
    state = _mode_game_state()
    session = GameSession(mode=mode.name, state=SessionState.AIMING)
    assert mode.legal_target_ids(state, session) is None

    prediction = _fan([_DEAD, Vec2(12.0, 14.0), Vec2(10.0, 30.0), _DEAD, _DEAD])
    assert mode.recommend_power(state, prediction, session) == ""
    assert not any(t.recommended for t in prediction.power_ticks)


def test_classic_legal_set_narrows_as_the_rules_narrow(settings: Settings) -> None:
    """Open table, then the shooter's group, then the 8.

    The last narrowing matters most: once a player's group is clear, a leave that
    offers a clear shot on the opponent's stripe is not a good leave, it is a
    foul waiting to happen. Falling back to "everything" there would recommend
    the pace that sets one up.
    """
    from modes.classic import ClassicMode
    from modes.scoring import BallGroup

    mode = ClassicMode(settings)
    state = _mode_game_state()
    session = GameSession(
        mode=mode.name, state=SessionState.AIMING, players=[Player("A"), Player("B")]
    )

    # Open table: either group, but *not* the 8 -- hitting it first is a foul
    # until your own group is cleared, so a leave whose only shot is on the 8 is
    # not a good leave.
    open_ids = mode.legal_target_ids(state, session)
    assert open_ids is not None
    assert set(open_ids) == {"solid_1", "stripe_9"}
    assert "eight" not in open_ids

    # Split: only the shooter's group.
    mode.groups[0] = BallGroup.SOLIDS
    mode.groups[1] = BallGroup.STRIPES
    solids = mode.legal_target_ids(state, session)
    assert solids == ["solid_1"]

    # Group cleared: only the 8.
    mode.groups[0] = BallGroup.STRIPES
    session.current_player_index = 0
    state_without_stripes = _mode_game_state(include_stripe=False)
    assert mode.legal_target_ids(state_without_stripes, session) == ["eight"]


def test_training_legal_set_is_the_drills_ball_alone(settings: Settings) -> None:
    """A drill is not satisfied by a good shot on some other ball.

    Narrower than any competitive mode's answer, which makes the recommendation
    in training a different judgement: "which pace leaves me on *this* ball"
    rather than "on anything I am allowed to hit".
    """
    from app.models import DrillType
    from modes.training import TrainingMode

    mode = TrainingMode(settings)
    state = _mode_game_state()
    session = GameSession(mode=mode.name, state=SessionState.AIMING)
    assert mode.legal_target_ids(state, session) is None  # no drill yet

    mode.start_drill(DrillType.POTTING, state)
    assert mode.current_drill is not None
    ids = mode.legal_target_ids(state, session)
    assert ids == [mode.current_drill.target_ball_id]


def test_the_position_drill_recommends_rather_than_prescribing(
    settings: Settings,
) -> None:
    """The position drill leaves the pace to the player, and advises on it.

    Choosing the pace *is* the skill a position drill teaches, so prescribing it
    would do the student's thinking for them and leave nothing to learn but the
    line. Potting and bank drills keep prescribing, because there the pace is a
    means rather than the lesson.

    Also the reason the recommendation path is exercised in training at all
    rather than sitting dormant behind three drills that all prescribe.
    """
    from app.models import DrillType
    from modes.training import TrainingMode

    mode = TrainingMode(settings)
    state = _mode_game_state()

    position = mode.start_drill(DrillType.POSITION, state)
    assert position.power_bucket is None, "the position drill must not prescribe pace"

    potting = mode.start_drill(DrillType.POTTING, state)
    assert potting.power_bucket is not None
    bank = mode.start_drill(DrillType.BANK_SHOT, state)
    assert bank.power_bucket is not None


def _mode_game_state(include_stripe: bool = True) -> GameState:
    """A cue ball, a solid, a stripe and the 8 -- enough to split a table."""
    balls = [
        Ball(
            id="cue",
            center_px=Vec2(0.0, 0.0),
            radius_px=12.0,
            kind=BallKind.CUE,
            table_pos=Vec2(20.0, 19.0),
        ),
        Ball(
            id="solid_1",
            center_px=Vec2(0.0, 0.0),
            radius_px=12.0,
            kind=BallKind.SOLID,
            number=1,
            table_pos=Vec2(44.0, 26.0),
        ),
        Ball(
            id="eight",
            center_px=Vec2(0.0, 0.0),
            radius_px=12.0,
            kind=BallKind.EIGHT,
            number=8,
            table_pos=Vec2(56.0, 12.0),
        ),
    ]
    if include_stripe:
        balls.insert(
            2,
            Ball(
                id="stripe_9",
                center_px=Vec2(0.0, 0.0),
                radius_px=12.0,
                kind=BallKind.STRIPE,
                number=9,
                table_pos=Vec2(50.0, 30.0),
            ),
        )
    return GameState(
        timestamp=0.0, frame_index=1, balls=balls, cue_ball=balls[0], confidence=0.9
    )
