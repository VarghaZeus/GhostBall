"""The setup wizard, its three flows, and calibration staleness.

Two things are being pinned here.

**The flows are subsequences, not modes.** A bumped box should not mean a
seven-step restart, so each flow runs a subset of the same steps and writes one
artifact. If a step ever starts behaving differently depending on which flow it
is in, the whole design has quietly become three wizards.

**Coupling is measured, never assumed.** The camera and projector are one unit,
so a move can invalidate both calibrations at once -- but re-running focus must
not throw away a good homography for a vertical nudge. The rule is that each
calibration records what the world looked like when it was taken, staleness is a
comparison against that, and a stale calibration is *reported*, never deleted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.calibration_status import (
    CORNER_DRIFT_FRACTION,
    WizardFlow,
    assess,
    corner_drift_px,
)
from app.config import Settings
from app.main import create_app
from app.models import ProjectorCalibration, TableBoundary, Vec2
from app.state import AppState
from app.wizard import FLOW_STEPS, StepId, Wizard
from vision.focus import FocusCalibration


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.camera.use_mock = s.projector.use_mock = True
    s.camera.width, s.camera.height = 640, 360
    s.projector.width, s.projector.height = 640, 360
    return s


@pytest.fixture
def state(settings) -> AppState:
    return AppState(settings=settings)


def boundary(dx: float = 0.0, confidence: float = 0.9) -> TableBoundary:
    return TableBoundary(
        top_left=Vec2(100 + dx, 100),
        top_right=Vec2(900 + dx, 100),
        bottom_right=Vec2(900 + dx, 500),
        bottom_left=Vec2(100 + dx, 500),
        center=Vec2(500 + dx, 300),
        width_px=800.0,
        height_px=400.0,
        confidence=confidence,
    )


def projector_calibration(dx: float = 0.0) -> ProjectorCalibration:
    return ProjectorCalibration(
        projector_width=1920,
        projector_height=1080,
        rmse_px=4.2,
        is_calibrated=True,
        table_corners_px=[[c.x, c.y] for c in boundary(dx).corners()],
        created_at="2026-08-25T10:00:00+00:00",
    )


def focus_calibration() -> FocusCalibration:
    return FocusCalibration(
        focus_absolute=1400,
        peak_sharpness=800.0,
        bare_table_sharpness=40.0,
        created_at="2026-08-25T10:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


class TestFlows:
    def test_every_flow_ends_at_done(self) -> None:
        for flow, steps in FLOW_STEPS.items():
            assert steps[-1] is StepId.DONE, flow

    def test_partial_flows_are_strict_subsequences_of_the_full_one(self) -> None:
        """The entire difference between flows is which steps they visit. If a
        partial flow ever contains a step the full one does not, or reorders
        them, these have stopped being one wizard."""
        full = FLOW_STEPS[WizardFlow.FULL]
        for flow in (WizardFlow.FOCUS, WizardFlow.TABLE):
            steps = FLOW_STEPS[flow]
            positions = [full.index(step) for step in steps]
            assert positions == sorted(positions), f"{flow} reorders the steps"

    def test_the_focus_flow_skips_alignment(self, state) -> None:
        """Re-running focus must not touch a good homography."""
        steps = FLOW_STEPS[WizardFlow.FOCUS]
        assert StepId.CAMERA_FOCUS in steps
        assert StepId.CORNER_MAPPING not in steps

    def test_the_table_flow_skips_focus(self, state) -> None:
        steps = FLOW_STEPS[WizardFlow.TABLE]
        assert StepId.CORNER_MAPPING in steps
        assert StepId.CAMERA_FOCUS not in steps

    def test_the_focus_flow_still_warms_the_projector(self) -> None:
        """The camera cannot resolve detail the projector never drew, so the
        focus flow needs the projector on and its own ring set even though it
        writes nothing about the projector."""
        assert StepId.PROJECTOR_WARMUP in FLOW_STEPS[WizardFlow.FOCUS]

    def test_a_partial_flow_is_genuinely_shorter(self) -> None:
        assert len(FLOW_STEPS[WizardFlow.FOCUS]) < len(FLOW_STEPS[WizardFlow.FULL])
        assert len(FLOW_STEPS[WizardFlow.TABLE]) < len(FLOW_STEPS[WizardFlow.FULL])


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------


class TestWizardMachine:
    def test_it_starts_on_the_first_step_of_its_flow(self, state) -> None:
        wizard = Wizard(state, WizardFlow.FOCUS)
        assert wizard.step is StepId.PROJECTOR_WARMUP
        assert wizard.view()["step_count"] == len(FLOW_STEPS[WizardFlow.FOCUS])

    def test_next_advances(self, state) -> None:
        wizard = Wizard(state, WizardFlow.FOCUS)
        ok, _ = wizard.act("next", from_step="projector_warmup")
        assert ok
        assert wizard.step is StepId.CAMERA_FOCUS

    def test_an_action_from_a_stale_step_is_rejected(self, state) -> None:
        """The double-advance guard. Two people tapping Next on the same step
        would otherwise skip one, which is not hypothetical when two people are
        setting up a table."""
        wizard = Wizard(state, WizardFlow.FOCUS)
        wizard.act("next", from_step="projector_warmup")

        ok, message = wizard.act("next", from_step="projector_warmup")
        assert not ok
        assert "projector_warmup" in message and "camera_focus" in message
        assert wizard.step is StepId.CAMERA_FOCUS, "the rejected tap must not advance"

    def test_an_action_with_no_step_is_allowed(self, state) -> None:
        """The guard is opt-in, so a client that does not track steps still
        works -- it simply gives up the protection."""
        wizard = Wizard(state, WizardFlow.FOCUS)
        assert wizard.act("next", from_step=None)[0]

    def test_next_is_refused_while_a_step_is_unfinished(self, state) -> None:
        wizard = Wizard(state, WizardFlow.TABLE)
        assert wizard.step is StepId.DETECT_TABLE
        ok, message = wizard.act("next", from_step="detect_table")
        assert not ok
        assert "not finished" in message

    def test_back_works_and_does_not_run_off_the_start(self, state) -> None:
        wizard = Wizard(state, WizardFlow.FOCUS)
        wizard.act("back", from_step="projector_warmup")
        assert wizard.index == 0

    def test_cancel_ends_it(self, state) -> None:
        wizard = Wizard(state, WizardFlow.FULL)
        assert wizard.act("cancel", from_step="welcome")[0]
        assert wizard.cancelled
        assert wizard.view()["active"] is False

    def test_the_version_moves_on_every_change(self, state) -> None:
        """So a poller can tell it is behind without diffing the payload."""
        wizard = Wizard(state, WizardFlow.FOCUS)
        before = wizard.view()["version"]
        wizard.act("next", from_step="projector_warmup")
        assert wizard.view()["version"] > before

    def test_a_step_that_throws_does_not_end_the_wizard(self, state) -> None:
        """Setup takes minutes and involves a ladder. Losing it to one bad frame
        would be a poor trade for a simpler error path."""
        wizard = Wizard(state, WizardFlow.FULL)
        wizard.handler.update = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))

        assert wizard.update(object()) is None
        assert not wizard.finished and not wizard.cancelled
        assert "boom" in wizard.view()["message"]

    def test_the_view_is_complete_enough_to_render(self, state) -> None:
        """The phone carries every word; nothing instructional is on the felt."""
        view = Wizard(state, WizardFlow.FULL).view()
        for key in ("step", "step_number", "step_count", "title", "instruction", "actions"):
            assert key in view, key
        assert view["title"] and view["instruction"]
        assert any(a["id"] == "cancel" for a in view["actions"])

    def test_the_projector_step_names_the_remote_and_offers_no_control(self, state) -> None:
        """Projector focus is a ten-second job with the remote. A software
        control for a hardware focus ring would be a worse version of a thing
        that already works -- but it still has to be *said*, because the camera
        cannot resolve what the projector never drew."""
        wizard = Wizard(state, WizardFlow.FOCUS)
        view = wizard.view()
        assert "remote" in view["instruction"].lower()
        assert not [a for a in view["actions"] if "focus" in a["id"].lower()]


# ---------------------------------------------------------------------------
# Staleness and coupling
# ---------------------------------------------------------------------------


class TestCornerDrift:
    def test_identical_corners_have_no_drift(self) -> None:
        corners = [[c.x, c.y] for c in boundary().corners()]
        assert corner_drift_px(corners, corners) == pytest.approx(0.0)

    def test_a_shift_shows_up_as_that_shift(self) -> None:
        recorded = [[c.x, c.y] for c in boundary().corners()]
        current = [[c.x, c.y] for c in boundary(60).corners()]
        assert corner_drift_px(recorded, current) == pytest.approx(60.0)

    def test_missing_data_is_not_an_error(self) -> None:
        """Calibrations written before the fingerprint existed simply cannot be
        checked, which must degrade to "no opinion" rather than "stale"."""
        assert corner_drift_px(None, [[1, 2]]) is None
        assert corner_drift_px([[1, 2]], None) is None

    def test_it_uses_the_mean_not_the_worst_corner(self) -> None:
        """One corner can be occluded or mis-fitted by a hand on the rail, and a
        max would report a bump every time somebody leaned on the table."""
        recorded = [[0, 0], [0, 0], [0, 0], [0, 0]]
        current = [[0, 0], [0, 0], [0, 0], [400, 0]]
        assert corner_drift_px(recorded, current) == pytest.approx(100.0)


class TestStaleness:
    def test_nothing_calibrated_suggests_the_full_run(self) -> None:
        status = assess(None, None)
        assert status.suggested is WizardFlow.FULL
        assert not status.as_dict()["all_ok"]

    def test_a_healthy_rig_suggests_nothing(self) -> None:
        status = assess(
            projector_calibration(), focus_calibration(),
            boundary=boundary(), live_sharpness=38.0,
        )
        assert status.suggested is None
        assert status.as_dict()["all_ok"]

    def test_a_sideways_shift_only_invalidates_alignment(self) -> None:
        """Focus depends on distance, which a lateral move does not change.
        Re-running focus here would be wasted time."""
        status = assess(
            projector_calibration(), focus_calibration(),
            boundary=boundary(60), live_sharpness=38.0,
        )
        assert status.suggested is WizardFlow.TABLE
        items = {i.key: i for i in status.items}
        assert items["projector"].stale
        assert not items["focus"].stale

    def test_a_soft_picture_only_invalidates_focus(self) -> None:
        status = assess(
            projector_calibration(), focus_calibration(),
            boundary=boundary(), live_sharpness=10.0,
        )
        assert status.suggested is WizardFlow.FOCUS
        items = {i.key: i for i in status.items}
        assert items["focus"].stale
        assert not items["projector"].stale

    def test_both_stale_together_asks_for_the_full_run(self) -> None:
        """The coupling case. The two are joined through the mount, so fixing
        one leaves the other mismatched -- and the mismatch is invisible from
        the felt, which is what makes it worth saying out loud."""
        status = assess(
            projector_calibration(), focus_calibration(),
            boundary=boundary(200), live_sharpness=5.0,
        )
        assert status.suggested is WizardFlow.FULL
        assert "moved" in status.headline
        assert "mismatched" in status.headline

    def test_stale_is_never_deleted(self) -> None:
        """A calibration flagged stale still works and still gets used. Deleting
        a possibly-good one on a heuristic would be a worse failure than leaving
        a possibly-stale one and saying so."""
        status = assess(
            projector_calibration(), focus_calibration(),
            boundary=boundary(200), live_sharpness=5.0,
        )
        for item in status.items:
            assert item.calibrated, "staleness must not clear the calibrated flag"

    def test_every_item_names_the_flow_that_fixes_it(self) -> None:
        """What makes the status actionable: a button next to the problem."""
        status = assess(None, None)
        for item in status.items:
            assert item.fixed_by in tuple(WizardFlow)
            assert item.detail, f"{item.key} says nothing about itself"

    def test_a_missing_live_reading_gives_no_opinion_on_focus(self) -> None:
        """Before any measurement there is nothing to compare against, and
        guessing would flag every fresh boot."""
        status = assess(projector_calibration(), focus_calibration(), boundary=boundary())
        assert not [i for i in status.items if i.stale]

    def test_the_drift_threshold_scales_with_the_frame(self) -> None:
        """A fixed pixel count would mean something different at every camera
        resolution."""
        small = assess(
            projector_calibration(), focus_calibration(),
            boundary=boundary(30), frame_width=640,
        )
        large = assess(
            projector_calibration(), focus_calibration(),
            boundary=boundary(30), frame_width=4000,
        )
        assert {i.key: i.stale for i in small.items}["projector"]
        assert not {i.key: i.stale for i in large.items}["projector"]
        assert CORNER_DRIFT_FRACTION > 0


# ---------------------------------------------------------------------------
# Over the API
# ---------------------------------------------------------------------------


@pytest.fixture
def client(state):
    with TestClient(create_app(state, start_loop=False)) as test_client:
        yield test_client


class TestWizardApi:
    def test_the_overview_is_available_before_anything_is_calibrated(self, client) -> None:
        body = client.get("/api/calibration/overview").json()
        assert body["items"]
        assert body["suggested"] == "full"
        assert body["headline"]

    def test_no_wizard_reports_inactive_rather_than_erroring(self, client) -> None:
        assert client.get("/api/wizard").json()["active"] is False

    @pytest.mark.parametrize("flow", ["full", "focus", "table"])
    def test_each_flow_starts_from_the_panel(self, client, flow) -> None:
        """Both entry points are the Setup tab. Not SSH, not a second process."""
        response = client.post("/api/wizard/start", json={"flow": flow})
        assert response.status_code == 200
        body = response.json()
        assert body["active"] and body["flow"] == flow
        assert body["step_number"] == 1

    def test_an_unknown_flow_is_rejected(self, client) -> None:
        assert client.post("/api/wizard/start", json={"flow": "sideways"}).status_code == 422

    def test_a_game_in_progress_refuses_with_a_reason(self, client, state) -> None:
        """"Someone is mid-game" and "another phone has it" send you to
        different places, so they cannot be the same 409."""
        state.mode_manager.start_game(["Sam", "Ali"])

        response = client.post("/api/wizard/start", json={"flow": "full"})
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["reason"] == "game_in_progress"
        assert "Reset session" in detail["message"]
        assert detail["can_force"] is False, "a live game must not be forceable"

    def test_an_action_returns_the_new_state_immediately(self, client) -> None:
        """What makes Next feel instant: the tapping phone renders from this
        response rather than waiting for the next poll."""
        client.post("/api/wizard/start", json={"flow": "focus"})
        body = client.post(
            "/api/wizard/action", json={"action": "next", "from_step": "projector_warmup"}
        ).json()
        assert body["step"] == "camera_focus"

    def test_a_stale_action_is_refused_and_returns_current_state(self, client) -> None:
        """409 rather than 400: the request was well formed and arrived against
        a state that had moved on, which the client recovers from by
        re-rendering."""
        client.post("/api/wizard/start", json={"flow": "focus"})
        client.post("/api/wizard/action", json={"action": "next", "from_step": "projector_warmup"})

        response = client.post(
            "/api/wizard/action", json={"action": "next", "from_step": "projector_warmup"}
        )
        assert response.status_code == 409
        assert response.json()["detail"]["step"] == "camera_focus"

    def test_acting_with_no_wizard_is_a_conflict(self, client) -> None:
        assert client.post("/api/wizard/action", json={"action": "next"}).status_code == 409

    def test_the_wizard_survives_a_reload(self, client) -> None:
        """Stateless resume: the phone holds nothing, so a browser refresh is
        just another GET."""
        client.post("/api/wizard/start", json={"flow": "table"})
        first = client.get("/api/wizard").json()
        second = client.get("/api/wizard").json()
        assert first["step"] == second["step"] == "detect_table"
        assert second["active"]

    def test_cancelling_releases_it(self, client, state) -> None:
        client.post("/api/wizard/start", json={"flow": "full"})
        client.post("/api/wizard/action", json={"action": "cancel", "from_step": "welcome"})
        assert state.wizard.cancelled


class TestForceSemantics:
    """``force`` is not a master key. It honours what each refusal says about
    itself, because the two refusals protect very different things."""

    def test_a_busy_table_can_be_forced_past(self, client, state) -> None:
        """No score to lose, and detection can get stuck reporting movement --
        a draught on a light, a reflection. An unforceable refusal you cannot
        clear is a trap."""
        from app.models import SessionState

        state.mode_manager.session.state = SessionState.SHOT_IN_PROGRESS

        refused = client.post("/api/wizard/start", json={"flow": "full"})
        assert refused.status_code == 409
        assert refused.json()["detail"]["can_force"] is True

        forced = client.post("/api/wizard/start", json={"flow": "full", "force": True})
        assert forced.status_code == 200
        assert forced.json()["active"]

    def test_a_seated_game_cannot_be_forced_past(self, client, state) -> None:
        """The one thing force must not do. Losing somebody's game to a stray
        tap on another phone is worse than making them tap Reset."""
        state.mode_manager.start_game(["Sam", "Ali"])

        forced = client.post("/api/wizard/start", json={"flow": "full", "force": True})
        assert forced.status_code == 409
        assert forced.json()["detail"]["reason"] == "game_in_progress"
        assert state.wizard is None
