"""Tests for the control panel's API surface.

Complements the API tests in ``test_scaffold.py``, which cover the routes that
existed with the scaffold. This module covers Phase 5: the projection-request
mechanism, the camera preview, and the two classes of honesty bug the panel is
most prone to --

* **Reporting success for something that did not happen.** Switching to an
  unbuilt mode, or a setting that was silently ignored. The panel is the only
  feedback channel on a headless Pi, so a lie here is expensive.
* **Answering 500 for something merely unbuilt.** The API's stated contract is
  503 with the name of the stage, and 500 means "file a bug".

Threading is the other theme. The web layer and the vision loop are different
threads, and the projector belongs to the loop's -- so several tests here assert
that a handler *records a request* rather than acting, which is easy to
regress by "simplifying" the indirection away.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.models import GameModeName


@pytest.fixture()
def settings() -> Settings:
    """Mock hardware and a small projector frame, so nothing touches a device."""
    s = Settings()
    s.camera.use_mock = True
    s.projector.use_mock = True
    s.projector.width = 640
    s.projector.height = 360
    return s


@pytest.fixture()
def state(settings: Settings):
    from app.state import AppState

    return AppState(settings=settings)


@pytest.fixture()
def client(state):
    """A client with the vision loop disabled, so no hardware is touched."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    # raise_server_exceptions=False so an unhandled exception surfaces as the
    # 500 the browser would actually see, rather than propagating into the test
    # and being mistaken for a test-harness failure.
    app = create_app(state, start_loop=False)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture()
def frame_state(state):
    """State that has seen one frame, without running a loop.

    Several routes behave differently before and after the first frame, and the
    interesting bugs live on the *after* side -- which a loop-less client would
    never reach.
    """
    import time

    from app.models import Ball, BallColor, BallKind, GameState, Vec2

    balls = [
        Ball(
            id="cue",
            center_px=Vec2(400.0, 300.0),
            radius_px=12.0,
            color=BallColor.WHITE,
            kind=BallKind.CUE,
            table_pos=Vec2(19.0, 19.0),
        )
    ]
    state.latest_frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    state.latest_game_state = GameState(
        timestamp=time.perf_counter(),
        frame_index=1,
        balls=balls,
        cue_ball=balls[0],
        confidence=0.9,
    )
    return state


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status_reports_frame_age_not_just_fps(client, frame_state) -> None:
    """Frame age is the liveness signal the panel can trust.

    FPS is a mean over a rolling window, so a loop that died a minute ago still
    reports 30 until the window drains. A frame age cannot be misread, which is
    why it is a separate field rather than something the panel infers.
    """
    before = client.get("/api/status").json()["performance"]
    assert before["frame_age_ms"] is None or before["frame_age_ms"] >= 0.0

    body = client.get("/api/status").json()["performance"]
    assert body["frame_age_ms"] is not None
    assert 0.0 <= body["frame_age_ms"] < 60_000.0


def test_status_reports_frame_age_none_before_the_first_frame(client) -> None:
    """``None``, not zero. Zero would read as "a frame just arrived"."""
    assert client.get("/api/status").json()["performance"]["frame_age_ms"] is None


def test_status_reports_configured_geometry(client, settings: Settings) -> None:
    """Resolutions are shown because they explain most "detection is bad" reports.

    Thresholds are tuned at a particular capture resolution; a camera that came
    up at a different one produces plausible-looking but wrong detection, and
    nothing else in the panel would reveal it.
    """
    system = client.get("/api/status").json()["system"]
    assert system["camera_resolution"] == f"{settings.camera.width}x{settings.camera.height}"
    assert system["projector_resolution"] == "640x360"
    assert system["camera_target_fps"] == settings.camera.fps


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------


def test_switching_to_an_unbuilt_mode_is_refused(client) -> None:
    """503 and a real explanation, not 200 and a quiet substitution.

    The mode manager falls back to freeplay for an unimplemented mode, which is
    correct down there -- a bad value must not take the game down mid-session.
    But answering 200 "mode set to Knockout" to a request for it tells the user
    their tap worked when nothing happened.

    ``knockout`` is the last mode with no implementation, so it is what this
    tests. If it is ever built, point this at whatever is unbuilt then rather
    than deleting it -- the refusal path is what keeps the panel honest.
    """
    response = client.post("/api/mode", json={"mode": "knockout"})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "knockout" in detail.lower()
    # The message has to say what *is* available, or the user has nowhere to go.
    assert "freeplay" in detail

    # And the switch must not have happened.
    assert client.get("/api/status").json()["current_mode"] == GameModeName.FREEPLAY.value


def test_switching_to_a_built_mode_works(client) -> None:
    assert client.post("/api/mode", json={"mode": "training"}).status_code == 200
    assert client.get("/api/status").json()["current_mode"] == "training"


def test_settings_advertises_what_can_be_selected(client) -> None:
    """The panel builds its selectors from the server, not from hardcoded lists.

    A list in the HTML goes stale the moment a theme or mode is added in Python,
    and the panel is the only place a user can discover the names.
    """
    body = client.get("/api/settings").json()
    assert "classic" in body["available_themes"]
    assert body["available_modes"] == [
        "freeplay",
        "classic",
        "king_of_the_hill",
        "trick_shots",
        "training",
    ]
    # Everything advertised must actually be selectable.
    for mode in body["available_modes"]:
        assert client.post("/api/mode", json={"mode": mode}).status_code == 200
    for theme in body["available_themes"]:
        assert client.post("/api/settings", json={"theme": theme}).status_code == 200


def test_mode_switch_asks_the_loop_to_blank(client, state) -> None:
    """The handler records a request; it must not drive the projector itself.

    A full-screen OpenCV window belongs to the thread that created it -- the
    vision loop. A handler calling ``send_frame`` from the event loop is
    undefined behaviour ranging from an ignored repaint to a segfault, so this
    indirection is load-bearing rather than ceremony.
    """
    state.blank_requested = False
    client.post("/api/mode", json={"mode": "training"})
    assert state.blank_requested is True


def test_reset_asks_the_loop_to_blank(client, state) -> None:
    state.blank_requested = False
    assert client.post("/api/reset").json()["success"] is True
    assert state.blank_requested is True


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_auto_detect_cue_round_trips(client, state) -> None:
    """A toggle the panel shows must be readable back, or it cannot render state."""
    client.post("/api/settings", json={"auto_detect_cue": False})
    assert state.auto_detect_cue is False
    assert client.get("/api/settings").json()["auto_detect_cue"] is False
    client.post("/api/settings", json={"auto_detect_cue": True})
    assert client.get("/api/settings").json()["auto_detect_cue"] is True


def test_unknown_theme_is_rejected_rather_than_silently_defaulted(client) -> None:
    """400, because the renderer's fallback would make a typo invisible.

    The renderer does fall back to ``classic`` for an unknown theme -- right for
    a hand-edited config file. Accepting it here, returning 200, and then showing
    a different theme is a confusing way to report a typo.
    """
    response = client.post("/api/settings", json={"theme": "sparkles"})
    assert response.status_code == 400
    assert "sparkles" in response.json()["detail"]
    assert client.get("/api/settings").json()["theme"] == "classic"


# ---------------------------------------------------------------------------
# Projected test patterns
# ---------------------------------------------------------------------------


def test_pattern_request_is_recorded_not_drawn(client, state) -> None:
    """Same threading constraint as blanking: the handler only leaves a note."""
    body = client.post("/api/projector/pattern", json={"pattern": "grid"}).json()
    assert body["active"] == "grid"
    assert state.projection_override == "grid"
    # Visible in status, so the panel can show what the projector is doing even
    # if the request came from another phone.
    assert client.get("/api/status").json()["system"]["projection_override"] == "grid"


def test_clearing_a_pattern_also_blanks(client, state) -> None:
    """"Clear the projection" has to mean it.

    Dropping the override without blanking would leave the last pattern frame
    frozen on the felt until something else drew; blanking without dropping the
    override would blank for one frame and then bring the pattern back.
    """
    client.post("/api/projector/pattern", json={"pattern": "corners"})
    state.blank_requested = False
    client.post("/api/projector/pattern", json={"pattern": None})
    assert state.projection_override is None
    assert state.blank_requested is True


def test_unknown_pattern_is_rejected_with_the_valid_names(client) -> None:
    response = client.post("/api/projector/pattern", json={"pattern": "spirograph"})
    assert response.status_code == 400
    assert "grid" in response.json()["detail"]


def test_pattern_listing_matches_what_is_accepted(client) -> None:
    """Everything advertised must be projectable, or the panel offers dead buttons."""
    available = client.get("/api/projector/patterns").json()["available"]
    assert available
    for name in available:
        assert client.post("/api/projector/pattern", json={"pattern": name}).status_code == 200


def test_loop_renders_the_requested_pattern(state, settings: Settings) -> None:
    """The other half of the contract: the loop must honour the recorded request.

    Tested against the loop's render helper directly rather than by running a
    thread -- the request/consume split is the thing under test, and a real loop
    would make it a timing race.
    """
    from app.main import VisionLoop

    loop = VisionLoop(state)
    state.projection_override = "grid"
    overlay = loop._render_pattern(state)
    assert overlay is not None
    assert overlay.shape == (settings.projector.height, settings.projector.width, 4)
    assert int((overlay[:, :, 3] > 0).sum()) > 100

    # A second call reuses the buffer rather than allocating a new 8 MB canvas
    # every frame -- a static pattern still has to be re-sent 30 times a second,
    # because an unfed OpenCV window stops repainting.
    again = loop._render_pattern(state)
    assert again is overlay


def test_loop_drops_an_unrenderable_override(state) -> None:
    """A bad override must not take the loop down.

    The API validates the name, so this only happens if a pattern is removed
    while one is selected -- but the failure mode matters: an exception here
    kills the vision loop, leaving a black projector and a panel still cheerfully
    reporting that a pattern is up.
    """
    from app.main import VisionLoop

    loop = VisionLoop(state)
    state.projection_override = "no_such_pattern"
    assert loop._render_pattern(state) is None
    assert state.projection_override is None


# ---------------------------------------------------------------------------
# Camera preview
# ---------------------------------------------------------------------------


def test_preview_is_503_before_the_first_frame(client) -> None:
    """503, not 404: the resource is not missing, it is not ready yet.

    The panel polls this, and a stream of 404s in the browser console buries the
    errors that matter.
    """
    response = client.get("/api/preview.jpg")
    assert response.status_code == 503
    assert "frame" in response.json()["detail"].lower()


def test_preview_returns_a_decodable_jpeg_at_the_requested_width(client, frame_state) -> None:
    import cv2

    response = client.get("/api/preview.jpg?width=320")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # Without no-store, mobile Safari serves the first frame forever and the
    # preview looks frozen.
    assert response.headers["cache-control"] == "no-store"

    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[1] == 320
    # Aspect ratio preserved: 640x360 downscaled to 320 wide is 180 tall.
    assert image.shape[0] == 180


def test_preview_width_is_bounded(client, frame_state) -> None:
    """A caller must not be able to ask for a 20000 px encode on a Pi."""
    assert client.get("/api/preview.jpg?width=5").status_code == 422
    assert client.get("/api/preview.jpg?width=99999").status_code == 422


def test_preview_overlay_warps_into_camera_space(client, frame_state, settings: Settings) -> None:
    """The overlay must be warped, not pasted.

    Projector space and camera space are different spaces that happen to be the
    same *size*, so blending directly produces an image that looks plausible and
    is wrong -- the trajectory would appear wherever it falls in the projector's
    own frame rather than where it lands on the felt. The test pins the
    consequence: with a table homography that is not the identity, the warped
    result differs from the raw overlay.
    """
    import cv2

    from projection import draw

    # A full-white overlay, so any geometry difference shows up as coverage.
    overlay = draw.new_canvas(settings)
    overlay[:, :, :] = 255
    frame_state.latest_overlay = overlay
    # Table homography that maps the table into the middle of the camera frame,
    # so the warp has something to do.
    frame_state.table_to_camera = np.array(
        [[6.0, 0.0, 80.0], [0.0, 6.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    plain = client.get("/api/preview.jpg?width=320&overlay=0")
    with_overlay = client.get("/api/preview.jpg?width=320&overlay=1")
    assert with_overlay.status_code == 200
    assert plain.content != with_overlay.content

    blended = cv2.imdecode(np.frombuffer(with_overlay.content, np.uint8), cv2.IMREAD_COLOR)
    # Brightened where the projection lands and untouched elsewhere. A pasted
    # (unwarped) overlay would have covered the whole frame uniformly.
    #
    # The threshold is 150, not 250: the blend applies the global overlay alpha,
    # so a white overlay over the frame's grey 60 lands near 187 rather than at
    # 255. Asserting near-white here would be asserting that the opacity slider
    # does nothing.
    bright = (blended > 150).all(axis=2)
    assert bright.any(), "no projected region in the preview"
    assert not bright.all(), "overlay covered the whole frame; it was not warped"


def test_preview_overlay_is_harmless_without_a_table(client, frame_state, settings: Settings) -> None:
    """Before the table is found there is no way to warp, so the overlay is skipped.

    Skipped rather than pasted unwarped: an overlay in the wrong space is worse
    than no overlay, because it looks like a calibration problem.
    """
    from projection import draw

    frame_state.latest_overlay = draw.new_canvas(settings)
    frame_state.latest_overlay[:, :, :] = 255
    frame_state.table_to_camera = None

    plain = client.get("/api/preview.jpg?width=320&overlay=0")
    requested = client.get("/api/preview.jpg?width=320&overlay=1")
    assert requested.status_code == 200
    assert plain.content == requested.content


# ---------------------------------------------------------------------------
# Unbuilt stages
# ---------------------------------------------------------------------------


def test_start_drill_never_answers_500(client, frame_state) -> None:
    """Regression: this returned 500 as soon as there was a frame to work with.

    Originally that was an unbuilt ``start_drill`` raising ``NotImplementedError``
    through FastAPI. Now the drill machinery is built and the same route has a
    second way to fail for a reason that is not a bug: this fixture's table has
    a cue ball and nothing to pot, so no potting drill exists on it.

    Both must stay off 500, which is the API's signal for "this is a fault, file
    it". A layout with no available drill answers 409 and names what to move.
    """
    response = client.post("/api/training/start_drill", json={"drill_type": "potting"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail and "clear pot" in detail.lower()


def test_start_drill_succeeds_once_there_is_something_to_pot(client, state) -> None:
    """The happy path, so the 409 above is a real branch and not the only one."""
    import time

    from app.models import Ball, BallColor, BallKind, GameState, Vec2

    cue = Ball(
        id="cue", center_px=Vec2(400.0, 300.0), radius_px=12.0,
        color=BallColor.WHITE, kind=BallKind.CUE, table_pos=Vec2(19.0, 19.0),
    )
    target = Ball(
        id="b1", center_px=Vec2(700.0, 300.0), radius_px=12.0,
        color=BallColor.YELLOW, kind=BallKind.SOLID, number=1,
        table_pos=Vec2(50.0, 19.0),
    )
    state.latest_game_state = GameState(
        timestamp=time.perf_counter(),
        frame_index=1,
        balls=[cue, target],
        cue_ball=cue,
    )
    response = client.post("/api/training/start_drill", json={"drill_type": "potting"})
    assert response.status_code == 200
    assert "pot it" in response.json()["message"].lower()


def test_training_result_is_empty_rather_than_404(client) -> None:
    """The panel polls this continuously before the first attempt."""
    body = client.get("/api/training/result").json()
    assert body["has_result"] is False


# ---------------------------------------------------------------------------
# Panel asset
# ---------------------------------------------------------------------------


def test_panel_is_served_and_self_contained(client) -> None:
    """One file, no CDN. The Pi is on a LAN that may have no internet at all.

    A panel that silently depends on a font or a framework from the network is a
    panel that renders as unstyled text in the one environment it has to work
    in.
    """
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "GhostBall" in html
    for pattern in ("<script src", "<link rel=\"stylesheet\"", "https://", "http://"):
        assert pattern not in html, f"panel references external resource: {pattern}"


def test_panel_only_calls_endpoints_that_exist(client) -> None:
    """Every path the panel fetches must be a real route.

    The panel is plain JavaScript with no build step and no type checking, so a
    renamed endpoint fails silently at runtime in a browser nobody is watching.
    This is the cheapest possible guard against that.
    """
    import re

    html = client.get("/").text
    # Paths passed to the panel's api() helper, which prefixes "/api".
    called = set(re.findall(r'api\("(/[^"?]*)', html))
    # Plus the preview, which is an <img> src rather than a fetch.
    called |= {m for m in re.findall(r'"(/api/[a-z._]+)\?', html)}

    routes = {
        route.path
        for route in client.app.routes
        if getattr(route, "path", "").startswith("/api")
    }
    for path in called:
        full = path if path.startswith("/api") else "/api" + path
        assert full in routes, f"panel calls {full}, which is not a route"


# ---------------------------------------------------------------------------
# Rebooting the host
# ---------------------------------------------------------------------------


class TestRebootTheHost:
    """``POST /api/system/reboot``.

    Every test here replaces :func:`app.power.spawn_reboot` before anything
    else happens, via an autouse fixture rather than per-test decoration --
    forgetting it in one test would reboot the machine running the suite, and
    that is not a failure mode to leave to diligence.

    Belt and braces: the endpoint independently refuses on a host with mock
    hardware, which every fixture in this module has, and on a host that is not
    Linux. So a test that *did* forget the patch would still have to get past
    both of those before doing any damage.
    """

    @pytest.fixture(autouse=True)
    def no_real_reboots(self, monkeypatch):
        """Records reboot attempts instead of making them.

        Patches the one function that runs the command, so nothing in this class
        can reach a real ``sudo reboot`` however the route is wired up.
        """
        from app import power

        fired: list[str] = []

        def fake_reboot() -> tuple[bool, str]:
            fired.append("reboot")
            return True, ""

        monkeypatch.setattr(power, "reboot_now", fake_reboot)
        return fired

    @pytest.fixture()
    def rig(self, monkeypatch):
        """A client that looks like the Pi: real hardware settings, Linux, sudo.

        No device is opened -- ``AppState`` builds no camera and the loop is not
        started -- so this is only about what the *guards* see. The two host
        checks are patched through their own tiny accessors precisely so a test
        can do this without mutating ``sys.platform`` for the whole process.
        """
        from fastapi.testclient import TestClient

        from app.config import Settings
        from app.main import create_app
        from app.state import AppState

        s = Settings()
        s.camera.use_mock = False
        s.projector.use_mock = False
        s.projector.width = 640
        s.projector.height = 360

        from app import power

        monkeypatch.setattr(power, "_linux", lambda: True)
        monkeypatch.setattr(power, "_sudo_path", lambda: "/usr/bin/sudo")

        app = create_app(AppState(settings=s), start_loop=False)
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client

    def test_mock_hardware_is_refused_rather_than_rebooted(
        self, client, no_real_reboots
    ) -> None:
        """The guard that keeps a developer's machine alive. A synthetic camera
        and a discarded projector mean this process is not the rig, and
        rebooting the laptop somebody is writing this on is a far worse outcome
        than a button that declines to work there."""
        res = client.post("/api/system/reboot")

        assert res.status_code == 409
        message = res.json()["detail"]["message"]
        assert "mock hardware" in message
        assert no_real_reboots == []

    def test_a_non_linux_host_is_refused_and_says_so(self, monkeypatch) -> None:
        """``sudo reboot`` is not a command Windows has, and "it failed" is a
        much worse answer than "this is the wrong kind of host"."""
        from app import power
        from app.config import Settings

        s = Settings()
        s.camera.use_mock = False
        s.projector.use_mock = False
        monkeypatch.setattr(power, "_linux", lambda: False)

        refusal = power.reboot_refusal(s)
        assert refusal and "Linux command" in refusal

    def test_a_linux_host_with_no_sudo_is_refused(self, monkeypatch) -> None:
        from app import power
        from app.config import Settings

        s = Settings()
        s.camera.use_mock = False
        s.projector.use_mock = False
        monkeypatch.setattr(power, "_linux", lambda: True)
        monkeypatch.setattr(power, "_sudo_path", lambda: None)

        refusal = power.reboot_refusal(s)
        assert refusal and "sudo" in refusal

    def test_a_missing_sudoers_entry_is_reported_not_silently_swallowed(
        self, rig, monkeypatch, no_real_reboots
    ) -> None:
        """The failure this endpoint exists to be honest about. Without a
        passwordless sudoers entry, ``sudo reboot`` from a service with no TTY
        fails in milliseconds -- so a panel that fired and reported success
        would tell somebody the Pi is rebooting while it carries on running.
        """
        from app import power

        monkeypatch.setattr(
            power, "reboot_now",
            lambda: (False, "sudo: a password is required"),
        )

        res = rig.post("/api/system/reboot")

        assert res.status_code == 403
        message = res.json()["detail"]["message"]
        # sudo's own words, because they are already written for a human.
        assert "a password is required" in message
        # And the one-line fix, because being told the permission is missing is
        # much less use than being told the command that grants it.
        assert "sudoers.d" in message
        # Said not to have happened, rather than acknowledged and dropped.
        assert "not issued" in message

    def test_it_reboots_the_rig(self, rig, no_real_reboots) -> None:
        res = rig.post("/api/system/reboot")

        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        # The operator is told to expect the disconnect, because it is about to
        # happen and the panel's other name for it is "Lost contact with the Pi".
        assert "lose contact" in body["message"]
        # Exactly once. Two calls would be two reboots.
        assert no_real_reboots == ["reboot"]

    def test_the_route_is_sync_so_it_does_not_stall_the_event_loop(self) -> None:
        """The reboot is issued synchronously so its exit status can be reported
        honestly rather than guessed at. A blocking subprocess call inside a
        coroutine would freeze uvicorn's event loop for the duration -- including
        the status poll of every other phone looking at the panel. FastAPI runs a
        plain ``def`` route in a threadpool, which is the behaviour wanted.
        """
        import inspect

        from web.api import reboot_host

        assert not inspect.iscoroutinefunction(reboot_host), (
            "the route is async again, so power.reboot_now() now blocks the event loop"
        )

    def test_the_command_is_argv_not_a_shell_string(self) -> None:
        """No shell means nothing here can become an injection if somebody later
        wants to pass a delay or a wall message."""
        from app import power

        assert isinstance(power.REBOOT_COMMAND, tuple)
        assert power.REBOOT_COMMAND == ("sudo", "-n", "reboot")
        assert not any(" " in part for part in power.REBOOT_COMMAND)

    def test_it_never_waits_for_a_password(self) -> None:
        """``-n`` is the flag that matters. Without it sudo sits on a prompt
        nobody can see: the request hangs for the whole timeout and then reports
        something vague, instead of failing in milliseconds with the one sentence
        that leads to the fix."""
        from app import power

        assert "-n" in power.REBOOT_COMMAND


class TestRebootCommand:
    """:func:`app.power.reboot_now` itself.

    A separate class because :class:`TestRebootTheHost` stubs ``reboot_now`` out
    wholesale -- which is right for testing the route, and useless for testing
    the function. Safety here comes from replacing ``subprocess.run`` instead,
    with an autouse default that fails rather than runs, so a test that forgets
    to stub it cannot reboot the machine running the suite.
    """

    @pytest.fixture(autouse=True)
    def no_real_subprocesses(self, monkeypatch):
        import subprocess

        def refuse(*args, **_kwargs):
            raise AssertionError(f"this test did not stub subprocess.run: {args!r}")

        monkeypatch.setattr(subprocess, "run", refuse)

    def test_a_nonzero_exit_is_a_failure_carrying_sudos_own_words(
        self, monkeypatch
    ) -> None:
        """Why the command is run for real instead of asked about. The previous
        version ran ``sudo -n -l reboot`` beforehand, which resolves the command
        against the caller's PATH -- so it could answer "not permitted" on a rig
        where ``sudo reboot`` works fine, and the button then refused to work on
        a perfectly good machine.
        """
        import subprocess

        from app import power

        class Result:
            returncode = 1
            stdout = ""
            stderr = "sudo: a password is required\n"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())

        ok, detail = power.reboot_now()
        assert ok is False
        assert detail == "sudo: a password is required"

    def test_a_zero_exit_is_success(self, monkeypatch) -> None:
        import subprocess

        from app import power

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())

        assert power.reboot_now()[0] is True

    def test_a_hang_is_reported_rather_than_waited_out_forever(
        self, monkeypatch
    ) -> None:
        """And the message leaves room for the reboot having worked: a command
        that stops answering because the host is going down looks exactly like
        one that is stuck."""
        import subprocess

        from app import power

        def timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd="sudo -n reboot", timeout=15.0)

        monkeypatch.setattr(subprocess, "run", timeout)

        ok, detail = power.reboot_now()
        assert ok is False
        assert "already going down" in detail

    def test_a_missing_sudo_is_reported_not_a_traceback(self, monkeypatch) -> None:
        import subprocess

        from app import power

        def missing(*_args, **_kwargs):
            raise FileNotFoundError(2, "No such file or directory", "sudo")

        monkeypatch.setattr(subprocess, "run", missing)

        ok, detail = power.reboot_now()
        assert ok is False
        assert "not installed" in detail

    def test_the_endpoint_is_a_post(self, client) -> None:
        """A GET that reboots the Pi is one link preview away from disaster."""
        assert client.get("/api/system/reboot").status_code == 405
