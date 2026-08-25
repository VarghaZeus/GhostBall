"""The control panel's JavaScript, actually executed.

Why this file exists: a change renamed ``paintStatus`` to ``renderStatus``
without updating its caller, and it shipped. The suite was green, and the file
passed ``node --check`` -- because a dangling identifier is *valid syntax*. It
fails at runtime with a ReferenceError, inside a ``try`` that reported it as
"Lost contact with the Pi", so the panel displayed nothing and blamed the
network.

Nothing that only parses the file can catch that. These tests run the panel's
script against a stub DOM and a stub ``fetch`` (see ``panel_harness.js``) and
assert that fields end up populated -- which is the only assertion that would
have failed.

Node is required, and its absence is a **failure**, not a skip. A silent skip is
how the original bug would have survived a second time: the suite goes green on
a machine that never ran these, which is precisely the machine the panel is
served from. Set ``AR_POOL_SKIP_JS_TESTS=1`` to opt out deliberately -- an
explicit choice in an environment variable is a different thing from an
accident.

jsdom was the alternative and was rejected: it needs ``npm install`` on a
machine whose LAN may have no internet, so it would not run in the place that
matters either.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.config import PACKAGE_ROOT

HARNESS = Path(__file__).parent / "panel_harness.js"
INDEX_HTML = PACKAGE_ROOT / "web" / "static" / "index.html"

#: Opt out explicitly. Anything else that leaves node unavailable is a failure.
SKIP_ENV = "AR_POOL_SKIP_JS_TESTS"

pytestmark = pytest.mark.skipif(
    os.environ.get(SKIP_ENV) == "1",
    reason=f"{SKIP_ENV}=1 -- the panel's JavaScript is not being executed",
)


def require_node() -> str:
    """The node binary, or a failure explaining what to install.

    A hard failure rather than a skip. These tests exist because a green suite
    once shipped a completely broken panel; a suite that goes green by not
    running them reproduces exactly that.
    """
    node = shutil.which("node")
    if node is None:
        pytest.fail(
            "node is required to execute the control panel's JavaScript, and is not "
            "on PATH. Install it (Raspberry Pi OS: `sudo apt install -y nodejs`), or "
            f"set {SKIP_ENV}=1 to skip these deliberately."
        )
    return node


#: A plausible ``GET /api/status`` body. Values are distinctive rather than
#: realistic -- 17 FPS and 43 ms are easy to find in an assertion failure, where
#: 30 and 100 could have come from anywhere.
STATUS = {
    "running": True,
    "current_mode": "freeplay",
    "session_state": "idle",
    "is_calibrated": True,
    "calibration_rmse_px": 4.25,
    "last_shot_confidence": 0.75,
    "performance": {
        "fps": 17.0,
        "frame_ms_avg": 43.0,
        "frame_ms_p95": 58.5,
        "latency_ms": 43.0,
        "dropped_frames": 12,
        "total_frames": 900,
        "stage_ms": {"capture": 31.4, "detect": 10.8},
        "frame_age_ms": 66.0,
    },
    "detections": {
        "balls": 9,
        "cue_ball_visible": True,
        "cue_stick_visible": False,
        "pockets": 6,
        "table_detected": True,
        "confidence": 0.82,
    },
    "system": {
        "cpu_pct": 31.0,
        "mem_pct": 7.0,
        "temp_c": 56.8,
        "camera_backend": "picamera2",
        "display_backend": "opencv",
        "using_mock_camera": False,
        "using_mock_display": False,
        "camera_resolution": "1920x1080",
        "camera_target_fps": 30,
        "projector_resolution": "1920x1080",
        "projection_override": None,
    },
    "health": {
        "uptime_seconds": 1323.0,
        "started_at": "2026-08-24T19:00:00+00:00",
        "frames_processed": 19432,
        "camera_reconnects": 0,
        "stage_errors": {},
        "disabled_stages": [],
        "degradation_level": 2,
        "loop_stalled": False,
        "stall_count": 0,
        "calibration_source": "file",
        "last_error": None,
    },
    "pending_stages": ["hailo"],
}

SETTINGS = {
    "brightness": 80,
    "overlay_alpha": 65,
    "trajectory_smoothing": 60,
    "physics_accuracy": "balanced",
    "theme": "classic",
    "available_themes": ["classic", "neon"],
    "available_modes": ["freeplay", "classic"],
    "auto_detect_cue": True,
}

CALIBRATION = {
    "is_calibrated": True,
    "rmse_px": 4.25,
    "quality": "good",
    "corners_recorded": 4,
    "created_at": "2026-08-24",
    "message": "",
}


def run_panel(
    tmp_path: Path, responses: dict, tab_hash: str = "", storage: dict | None = None
) -> dict:
    """Execute the panel against the given API responses; return what it drew.

    ``tab_hash`` and ``storage`` decide which tab comes up, which is how tab
    behaviour gets exercised without synthesising clicks.
    """
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps({"responses": responses, "hash": tab_hash, "storage": storage or {}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [require_node(), str(HARNESS), str(INDEX_HTML), str(scenario)],
        capture_output=True,
        text=True,
        # Explicit, because `text=True` alone decodes with the *locale* encoding
        # -- cp1252 on a Windows dev box -- and the panel's placeholder em-dash
        # comes back as mojibake. The Pi's UTF-8 locale hides this entirely,
        # which is the worst kind of platform difference to leave in a test.
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"harness failed:\n{result.stderr}"
    assert result.stdout.strip(), f"harness produced no output:\n{result.stderr}"
    return json.loads(result.stdout)


def healthy_responses(**overrides) -> dict:
    responses = {
        "/status": STATUS,
        "/settings": SETTINGS,
        "/calibration/status": CALIBRATION,
        "/projector/patterns": {"available": ["grid", "corners"], "active": None},
        "/training/result": {"has_result": False},
    }
    responses.update(overrides)
    return responses


@pytest.fixture
def panel(tmp_path):
    return run_panel(tmp_path, healthy_responses())


# ---------------------------------------------------------------------------
# The regression this file exists for
# ---------------------------------------------------------------------------


class TestStatusIsPainted:
    def test_the_script_runs_without_throwing(self, panel) -> None:
        assert panel["errors"] == []

    def test_the_status_endpoint_is_actually_polled(self, panel) -> None:
        assert "/status" in panel["requested"]

    def test_the_metric_strip_is_populated(self, panel) -> None:
        """The assertion that would have caught the rename.

        With ``paintStatus`` undefined these all stay at their placeholder
        em-dash, because the ReferenceError is swallowed by the poll's catch.
        """
        assert panel["texts"]["mFps"] == "17"
        assert panel["texts"]["mLatency"] == "43"
        assert panel["texts"]["mBalls"] == "9"
        assert panel["texts"]["mCalib"] == "4.3"

    def test_the_status_card_is_populated(self, panel) -> None:
        assert panel["texts"]["running"] == "running"
        assert panel["texts"]["mode"] == "freeplay"
        assert panel["texts"]["sstate"] == "idle"
        assert panel["texts"]["p95"] == "58.5 ms"
        assert panel["texts"]["dropped"] == "12 / 900"
        assert "capture 31.4" in panel["texts"]["stages"]

    def test_the_detections_card_is_populated(self, panel) -> None:
        assert panel["texts"]["dTable"] == "found"
        assert panel["texts"]["dBalls"] == "9"
        assert panel["texts"]["dCueBall"] == "visible"
        assert panel["texts"]["dCue"] == "not visible"
        assert panel["texts"]["dPockets"] == "6 / 6"

    def test_the_system_card_is_populated(self, panel) -> None:
        assert panel["texts"]["cpu"] == "31%"
        assert panel["texts"]["temp"] == "56.8 C"
        assert "picamera2" in panel["texts"]["camBackend"]

    def test_the_health_card_is_populated(self, panel) -> None:
        assert panel["texts"]["hUptime"] == "22.1 min"
        assert panel["texts"]["hFrames"] in ("19,432", "19432")
        assert panel["texts"]["hCalSrc"] == "saved calibration"
        assert panel["texts"]["hDegrade"] == "level 2"

    def test_no_field_is_left_at_its_placeholder(self, panel) -> None:
        """A blanket check, so a *new* card that never gets painted is caught
        without anyone remembering to add an assertion for it."""
        painted_by_status = [
            "running", "frameAge", "p95", "dropped", "mode", "sstate", "shotConf",
            "dTable", "dBalls", "dCueBall", "dCue", "dPockets", "dConf",
            "cpu", "mem", "temp", "camBackend", "dispBackend", "projOverride", "pending",
            "hUptime", "hFrames", "hCalSrc", "hReconn", "hStalls", "hDegrade",
            "hErrors", "hDisabled", "hLastErr",
        ]
        stale = [name for name in painted_by_status if panel["texts"][name] in ("", "—")]
        assert not stale, f"never painted: {stale}"

    def test_the_connection_reads_connected(self, panel) -> None:
        assert panel["texts"]["conn"] == "connected"

    def test_no_banner_is_shown_on_a_healthy_poll(self, panel) -> None:
        assert not panel["shown"]["errBanner"]
        assert not panel["shown"]["renderBanner"]
        assert not panel["shown"]["healthBanner"]
        assert not panel["shown"]["mockBanner"]


# ---------------------------------------------------------------------------
# The two failure kinds report differently
# ---------------------------------------------------------------------------


class TestFailuresAreDistinguished:
    def test_a_fetch_failure_says_the_pi_is_unreachable(self, tmp_path) -> None:
        panel = run_panel(
            tmp_path, healthy_responses(**{"/status": {"__throw": "Failed to fetch"}})
        )
        assert panel["texts"]["conn"] == "disconnected"
        assert panel["shown"]["errBanner"]
        assert "Lost contact" in panel["texts"]["errBanner"]
        assert not panel["shown"]["renderBanner"]

    def test_a_render_failure_does_not_claim_disconnected(self, tmp_path) -> None:
        """The whole point of the split.

        A panel that says "disconnected" while its own camera preview is
        updating three inches away sends you to debug the wrong machine -- which
        is exactly what happened.
        """
        broken = dict(STATUS)
        broken["performance"] = None  # paintStatus reads p.fps and throws

        panel = run_panel(tmp_path, healthy_responses(**{"/status": broken}))

        assert panel["shown"]["renderBanner"]
        assert "bug in the control panel" in panel["texts"]["renderBanner"]
        assert panel["texts"]["conn"] != "disconnected"
        assert not panel["shown"]["errBanner"], "a painter bug is not a network failure"

    def test_a_render_failure_reaches_the_console_every_time(self, tmp_path) -> None:
        """Banner latched for the user, console unlatched for whoever is
        debugging -- the stack is the useful part and there is no cost to it."""
        broken = dict(STATUS)
        broken["performance"] = None

        panel = run_panel(tmp_path, healthy_responses(**{"/status": broken}))
        assert any("paintStatus failed" in line for line in panel["consoleErrors"])

    def test_an_http_error_is_a_fetch_failure_not_a_render_one(self, tmp_path) -> None:
        """503 is the deliberate "stage not built" signal, so it must land on the
        connection banner and not be mistaken for a panel bug."""
        panel = run_panel(
            tmp_path,
            healthy_responses(
                **{"/status": {"__status": 503, "body": {"detail": "stage 'hailo' pending"}}}
            ),
        )
        assert panel["shown"]["errBanner"]
        assert "hailo" in panel["texts"]["errBanner"]
        assert not panel["shown"]["renderBanner"]


# ---------------------------------------------------------------------------
# Conditions the panel is supposed to shout about
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_mock_mode_is_announced(self, tmp_path) -> None:
        mocked = json.loads(json.dumps(STATUS))
        mocked["system"]["using_mock_camera"] = True

        panel = run_panel(tmp_path, healthy_responses(**{"/status": mocked}))
        assert panel["shown"]["mockBanner"]
        assert "synthetic" in panel["texts"]["mockBanner"]

    def test_a_stalled_loop_raises_the_health_banner(self, tmp_path) -> None:
        stalled = json.loads(json.dumps(STATUS))
        stalled["health"]["loop_stalled"] = True
        stalled["health"]["stall_count"] = 3

        panel = run_panel(tmp_path, healthy_responses(**{"/status": stalled}))
        assert panel["shown"]["healthBanner"]
        assert "STALLED NOW (3)" in panel["texts"]["hStalls"]

    def test_a_disabled_stage_raises_the_health_banner(self, tmp_path) -> None:
        degraded = json.loads(json.dumps(STATUS))
        degraded["health"]["disabled_stages"] = ["mode"]

        panel = run_panel(tmp_path, healthy_responses(**{"/status": degraded}))
        assert panel["shown"]["healthBanner"]
        assert panel["texts"]["hDisabled"] == "mode"

    def test_an_uncalibrated_rig_is_flagged(self, tmp_path) -> None:
        """An identity mapping projects a plausible overlay onto the wrong part
        of the table, which reads as a detection bug -- so it has to be said."""
        uncalibrated = json.loads(json.dumps(STATUS))
        uncalibrated["is_calibrated"] = False
        uncalibrated["health"]["calibration_source"] = "identity"

        panel = run_panel(tmp_path, healthy_responses(**{"/status": uncalibrated}))
        assert panel["texts"]["mCalib"] == "off"
        assert panel["texts"]["hCalSrc"] == "none (identity)"


# ---------------------------------------------------------------------------
# The harness's own assumptions
# ---------------------------------------------------------------------------


class TestHarness:
    def test_every_id_the_script_asks_for_exists_in_the_markup(self, panel) -> None:
        """``getElementById`` throws in the harness for an unknown id, so a
        mistyped id anywhere on a painted path fails the run rather than
        silently doing nothing."""
        assert panel["errors"] == []
        assert not [line for line in panel["consoleErrors"] if "no such element" in line]

    def test_the_harness_notices_a_panel_that_never_paints(self, tmp_path) -> None:
        """Guards the guard: break the call site the way the real bug did and
        confirm these tests would go red."""
        broken_html = tmp_path / "broken.html"
        broken_html.write_text(
            INDEX_HTML.read_text(encoding="utf-8").replace(
                "function paintStatus(s) {", "function renamedByAccident(s) {"
            ),
            encoding="utf-8",
        )

        scenario = tmp_path / "scenario.json"
        scenario.write_text(json.dumps({"responses": healthy_responses()}), encoding="utf-8")
        result = subprocess.run(
            [require_node(), str(HARNESS), str(broken_html), str(scenario)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        drawn = json.loads(result.stdout)

        assert drawn["texts"]["mFps"] == "—", "the placeholder should still be there"
        assert drawn["shown"]["renderBanner"], "and it should be reported as a panel bug"
        assert drawn["texts"]["conn"] != "disconnected"


class TestTabsRender:
    """The tab bar as it actually ends up, not as the markup describes it.

    The bug these exist for: ``selectTab`` used ``querySelectorAll("[data-tab]")``,
    which also matches the tab *buttons* -- so selecting a tab hid the other
    three buttons and left seven cards unreachable. The suite stayed green
    because every tab test read the source HTML with a regex, and the one
    harness-based test checked painted text, which happens whether or not an
    element is visible.

    Worse: the harness stubbed ``querySelectorAll`` to return ``[]``, so the
    line containing the bug never executed. Same shape as ``node --check`` --
    the file ran, the broken path did not.
    """

    def test_all_four_tabs_are_visible(self, panel) -> None:
        """The regression itself. Three of these were hidden."""
        assert [t["tab"] for t in panel["tabs"]] == ["play", "setup", "tune", "diagnostics"]
        hidden = [t["tab"] for t in panel["tabs"] if t["hidden"]]
        assert not hidden, f"tab buttons hidden: {hidden}"

    def test_exactly_one_tab_is_selected(self, panel) -> None:
        assert [t["selected"] for t in panel["tabs"]].count(True) == 1

    def test_selecting_a_tab_never_hides_a_tab_button(self, tmp_path) -> None:
        """The invariant rather than the symptom: whatever else selection does,
        the bar has to survive it."""
        for tab in ("play", "setup", "tune", "diagnostics"):
            drawn = run_panel(tmp_path, healthy_responses(), tab_hash=tab)
            assert not [t for t in drawn["tabs"] if t["hidden"]], f"hidden after selecting {tab}"

    @pytest.mark.parametrize(
        "tab,expected",
        [
            ("play", {"Status", "Detections", "Mode"}),
            ("setup", {"Setup & calibration", "Camera", "Projection", "Calibration"}),
            ("tune", {"Settings", "Training"}),
            ("diagnostics", {"System", "Health"}),
        ],
    )
    def test_each_tab_shows_its_own_cards_and_only_those(self, tmp_path, tab, expected) -> None:
        drawn = run_panel(tmp_path, healthy_responses(), tab_hash=tab)
        visible = {s["title"] for s in drawn["sections"] if not s["hidden"]}
        assert visible == expected

    def test_every_card_is_reachable_from_exactly_one_tab(self, tmp_path) -> None:
        """A blanket check over the rendered DOM, so a card that lands on no tab
        -- or on two -- is caught without anyone maintaining a list."""
        seen: dict[str, list[str]] = {}
        for tab in ("play", "setup", "tune", "diagnostics"):
            drawn = run_panel(tmp_path, healthy_responses(), tab_hash=tab)
            for section in drawn["sections"]:
                if not section["hidden"]:
                    seen.setdefault(section["title"], []).append(tab)

        every = {s["title"] for s in run_panel(tmp_path, healthy_responses())["sections"]}
        unreachable = every - set(seen)
        assert not unreachable, f"cards on no tab: {sorted(unreachable)}"
        duplicated = {title: tabs for title, tabs in seen.items() if len(tabs) > 1}
        assert not duplicated, f"cards on several tabs: {duplicated}"

    def test_a_hash_selects_the_tab(self, tmp_path) -> None:
        drawn = run_panel(tmp_path, healthy_responses(), tab_hash="diagnostics")
        assert [t["tab"] for t in drawn["tabs"] if t["selected"]] == ["diagnostics"]

    def test_the_stored_tab_is_restored(self, tmp_path) -> None:
        drawn = run_panel(tmp_path, healthy_responses(), storage={"ghostball.tab": "tune"})
        assert [t["tab"] for t in drawn["tabs"] if t["selected"]] == ["tune"]

    def test_an_explicit_hash_beats_the_stored_tab(self, tmp_path) -> None:
        """A link somebody followed is a stronger statement than whatever they
        happened to be looking at last time."""
        drawn = run_panel(
            tmp_path, healthy_responses(), tab_hash="setup", storage={"ghostball.tab": "tune"}
        )
        assert [t["tab"] for t in drawn["tabs"] if t["selected"]] == ["setup"]

    def test_a_nonsense_hash_falls_back_rather_than_showing_nothing(self, tmp_path) -> None:
        drawn = run_panel(tmp_path, healthy_responses(), tab_hash="not-a-tab")
        assert [t["tab"] for t in drawn["tabs"] if t["selected"]] == ["play"]
        assert [s for s in drawn["sections"] if not s["hidden"]]

    def test_a_problem_marks_its_tab(self, tmp_path) -> None:
        """So a stall on Diagnostics is discoverable from Play, without opening
        all four tabs to go looking for it."""
        stalled = json.loads(json.dumps(STATUS))
        stalled["health"]["loop_stalled"] = True
        drawn = run_panel(tmp_path, healthy_responses(**{"/status": stalled}))
        assert "diagnostics" in [t["tab"] for t in drawn["tabs"] if t["alert"]]


class TestCollapsibleCards:
    def test_cards_start_expanded(self, panel) -> None:
        assert not [s for s in panel["sections"] if s["collapsed"]]

    def test_a_stored_collapse_is_honoured(self, tmp_path) -> None:
        drawn = run_panel(
            tmp_path,
            healthy_responses(),
            storage={"ghostball.collapsed": json.dumps(["Detections"])},
        )
        assert {s["title"] for s in drawn["sections"] if s["collapsed"]} == {"Detections"}

    def test_collapsing_hides_the_body_not_the_card(self, tmp_path) -> None:
        """A collapsed card is still on its tab and still painted -- the status
        poll fills every card in one call regardless of what is folded away."""
        drawn = run_panel(
            tmp_path,
            healthy_responses(),
            storage={"ghostball.collapsed": json.dumps(["Detections"])},
        )
        card = next(s for s in drawn["sections"] if s["title"] == "Detections")
        assert card["collapsed"] and not card["hidden"]
        assert drawn["texts"]["dBalls"] == "9"


class TestTabsMarkup:
    """Structural checks. Kept, but no longer the only ones: these are exactly
    what passed while the panel was broken."""

    def test_every_card_is_assigned_to_a_tab(self) -> None:
        """A card with no tab is invisible: it belongs to no group, so nothing
        ever unhides it."""
        import re

        html = INDEX_HTML.read_text(encoding="utf-8")
        cards = re.findall(r'<section class="card"([^>]*)>', html)
        assert cards, "no cards found"
        untabbed = [c for c in cards if "data-tab=" not in c]
        assert not untabbed, f"{len(untabbed)} card(s) with no tab"

    def test_the_tabs_named_in_the_script_are_the_ones_used_in_the_markup(self) -> None:
        import re

        html = INDEX_HTML.read_text(encoding="utf-8")
        declared = set(re.findall(r'\["(\w+)", "[^"]+"\],', html.split("const TABS")[1][:300]))
        used = set(re.findall(r'data-tab="(\w+)"', html))
        assert used <= declared, f"cards reference tabs that do not exist: {used - declared}"
        assert declared <= used, f"tabs with no cards behind them: {declared - used}"

    def test_banners_and_metrics_stay_outside_the_tabs(self) -> None:
        """A stall warning on a tab you are not looking at is worse than no
        warning, and the hero metrics are the at-a-glance view."""
        html = INDEX_HTML.read_text(encoding="utf-8")
        grid_start = html.index('<div class="grid">')
        for element_id in ("errBanner", "healthBanner", "readyBanner", "mFps", "mLatency"):
            assert html.index(f'id="{element_id}"') < grid_start, f"{element_id} is inside the tabs"

    def test_the_preview_is_gated_on_its_tab_being_visible(self, panel) -> None:
        """The one real cost of hiding cards: the preview warps, resizes and
        JPEG-encodes on the cores the vision loop needs, and paying that for a
        card nobody is looking at is pure waste."""
        assert "/preview.jpg" not in " ".join(panel["requested"]), (
            "the preview was fetched while the Play tab was selected"
        )

    def test_a_card_from_another_tab_is_still_painted(self, panel) -> None:
        """Hidden is not unmounted. The status poll fills every card in one
        call, so switching tabs shows current data rather than dashes.

        Note this passed while the panel was broken -- painting happens whether
        or not an element is visible, which is why it was never sufficient on
        its own.
        """
        # 'cpu' lives on Diagnostics; the default tab is Play.
        assert panel["texts"]["cpu"] == "31%"


class TestReadinessBanner:
    def test_a_missing_table_is_announced(self, tmp_path) -> None:
        body = json.loads(json.dumps(STATUS))
        body["readiness"] = {
            "state": "no_table",
            "headline": "No pool table detected",
            "detail": "Mount the device above the middle of the table.",
            "since_seconds": 12.0,
            "table_confidence": 0.0,
            "playable": False,
        }
        panel = run_panel(tmp_path, healthy_responses(**{"/status": body}))
        assert panel["shown"]["readyBanner"]
        assert "No pool table detected" in panel["texts"]["readyBanner"]
        assert "Mount the device" in panel["texts"]["readyBanner"]

    def test_a_ready_system_says_nothing(self, tmp_path) -> None:
        body = json.loads(json.dumps(STATUS))
        body["readiness"] = {
            "state": "ready", "headline": "Ready", "detail": "",
            "since_seconds": 300.0, "table_confidence": 0.92, "playable": True,
        }
        panel = run_panel(tmp_path, healthy_responses(**{"/status": body}))
        assert not panel["shown"]["readyBanner"]


class TestStageTimes:
    def test_an_interval_stage_shows_its_per_frame_cost(self, tmp_path) -> None:
        """Table detection measured 98.8 ms per invocation and looked like the
        most expensive stage in the pipeline, while running once every 600
        frames for an amortised 0.16 ms. The list has to make that visible."""
        body = json.loads(json.dumps(STATUS))
        body["performance"]["stage_ms"] = {"capture": 31.4, "table": 98.8}
        body["performance"]["stage_coverage"] = {"capture": 1.0, "table": 0.0017}
        body["performance"]["stage_amortised_ms"] = {"capture": 31.4, "table": 0.16}

        panel = run_panel(tmp_path, healthy_responses(**{"/status": body}))
        stages = panel["texts"]["stages"]
        assert "table 98.8 (0.16/f)" in stages
        assert "capture 31.4" in stages and "capture 31.4 (" not in stages
