"""The launcher and its preflight checks.

What matters here is not that the checks pass on this machine -- they will,
which proves nothing -- but that each one *fails correctly*: with the right
severity, and carrying a fix. A preflight that reports "camera not found" and
stops has told the person under the table nothing they did not already know, so
every failing check is asserted to carry an instruction.

The other half is the contract with ``app.main``: the launcher borrows that
parser rather than declaring a second copy, and hands the parsed namespace
straight over. Both halves of that are pinned here, because the failure if
either drifts is a flag that silently does nothing.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

import launcher
from launcher import FAIL, OK, WARN, Check


@pytest.fixture
def args():
    """A parsed namespace with the launcher's defaults."""
    return launcher.build_launcher_parser().parse_args([])


class RunSpy:
    """Stands in for ``app.main.run``, recording the namespace handed to it."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, args) -> int:
        self.calls.append(args)
        return 0

    @property
    def args(self):
        assert self.calls, "app.main.run() was never called"
        return self.calls[-1]


@pytest.fixture
def run_spy(monkeypatch) -> RunSpy:
    spy = RunSpy()
    monkeypatch.setattr("app.main.run", spy)
    return spy


# ---------------------------------------------------------------------------
# Argument surface
# ---------------------------------------------------------------------------


class TestArguments:
    def test_the_application_flags_are_all_accepted(self) -> None:
        """Borrowed from ``app.main``, not redeclared -- so a flag added there
        works here the same day, without anyone remembering to mirror it."""
        parsed = launcher.build_launcher_parser().parse_args(
            ["--mock", "--headless", "--frames", "10", "--port", "9999"]
        )
        assert parsed.mock and parsed.headless
        assert parsed.frames == 10
        assert parsed.port == 9999

    def test_launcher_flags_default_off(self, args) -> None:
        assert not args.check
        assert not args.force
        assert not args.skip_checks

    def test_the_namespace_is_what_app_main_expects(self, args) -> None:
        """``run(args)`` reads these by name; a rename in either file breaks the
        handover, and it would break at startup rather than at import."""
        for attribute in (
            "config",
            "mock",
            "no_loop",
            "headless",
            "frames",
            "profile",
            "host",
            "port",
            "log_level",
        ):
            assert hasattr(args, attribute), f"app.main.run() reads args.{attribute}"

    def test_contradictory_flags_are_rejected(self) -> None:
        with pytest.raises(SystemExit):
            launcher.main(["--check", "--skip-checks"])


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


class TestChecks:
    def test_python_version_passes_on_a_supported_interpreter(self) -> None:
        assert launcher.check_python().status == OK

    def test_dependencies_are_present_in_a_working_install(self) -> None:
        checks = launcher.check_dependencies()
        assert not [c for c in checks if c.failed]

    def test_a_missing_dependency_fails_with_the_pip_name(self, monkeypatch) -> None:
        """``cv2`` and ``opencv-python`` are not the same string, and telling
        someone to ``pip install cv2`` sends them to a different package."""
        real_import = __import__

        def fake_import(name, *rest):
            if name == "cv2":
                raise ImportError("libGL.so.1: cannot open shared object file")
            return real_import(name, *rest)

        monkeypatch.setattr("builtins.__import__", fake_import)
        checks = launcher.check_dependencies()
        failures = [c for c in checks if c.failed]

        assert failures, "a missing dependency must fail, not warn"
        assert any("opencv-python" in c.fix for c in failures)
        # The detail is the ImportError itself: a wheel that installed cleanly
        # and cannot load libGL is the case pip metadata would call healthy.
        assert any("libGL" in c.detail for c in failures)

    def test_several_missing_dependencies_collapse_to_one_instruction(
        self, monkeypatch
    ) -> None:
        real_import = __import__

        def fake_import(name, *rest):
            if name in ("cv2", "fastapi"):
                raise ImportError("nope")
            return real_import(name, *rest)

        monkeypatch.setattr("builtins.__import__", fake_import)
        checks = launcher.check_dependencies()
        assert any(c.fix == "pip install -r requirements.txt" for c in checks)

    def test_an_optional_dependency_only_warns(self, monkeypatch) -> None:
        real_import = __import__

        def fake_import(name, *rest):
            if name == "psutil":
                raise ImportError("nope")
            return real_import(name, *rest)

        monkeypatch.setattr("builtins.__import__", fake_import)
        checks = launcher.check_dependencies()
        psutil_checks = [c for c in checks if "psutil" in c.name]
        assert psutil_checks and psutil_checks[0].status == WARN
        assert not [c for c in checks if c.failed]

    def test_the_shipped_config_loads(self) -> None:
        check, settings = launcher.check_config(None)
        assert check.status == OK
        assert settings is not None

    def test_a_malformed_config_fails_rather_than_falling_back(
        self, tmp_path
    ) -> None:
        """Silently running on default HSV ranges when someone has spent an
        evening tuning them against their own felt is worse than not starting."""
        bad = tmp_path / "config.yaml"
        bad.write_text("table_preset: 27ft\n", encoding="utf-8")

        check, settings = launcher.check_config(bad)
        assert check.status == FAIL
        assert settings is None
        assert check.fix, "a failure with no fix is just an error message"

    def test_a_missing_config_warns_and_still_yields_settings(
        self, tmp_path
    ) -> None:
        check, settings = launcher.check_config(tmp_path / "absent.yaml")
        assert check.status == WARN
        assert settings is not None, "every field has a working default"

    def test_data_dirs_are_created(self) -> None:
        from app.config import CALIBRATION_DIR, LOG_DIR

        assert launcher.check_data_dirs().status == OK
        assert LOG_DIR.is_dir() and CALIBRATION_DIR.is_dir()

    def test_a_free_port_passes(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        assert launcher.check_port("127.0.0.1", free_port).status == OK

    def test_a_held_port_fails_and_says_what_to_do(self) -> None:
        """The failure this prevents is misleading: uvicorn's "address already
        in use" usually means a previous copy of *this* app is still running,
        so the projector is already lit and every other symptom misleads."""
        with socket.socket() as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]

            check = launcher.check_port("127.0.0.1", port)

        assert check.status == FAIL
        assert f"--port {port + 1}" in check.fix

    def test_a_wildcard_bind_is_probed_on_loopback(self) -> None:
        """``0.0.0.0`` is not a connectable address, but bind-testing it is the
        right way to ask whether the port is free."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        assert launcher.check_port("0.0.0.0", port).status == OK


class TestHardwareChecks:
    def test_mock_camera_is_reported_as_a_warning(self) -> None:
        """A synthetic camera nobody noticed looks exactly like a detection bug."""
        from app.config import Settings

        settings = Settings()
        settings.camera.use_mock = True
        check = launcher.check_camera(settings)
        assert check.status == WARN
        assert "mock" in check.detail

    def test_mock_display_is_reported_as_a_warning(self) -> None:
        from app.config import Settings

        settings = Settings()
        settings.projector.use_mock = True
        assert launcher.check_display(settings).status == WARN

    def test_the_camera_check_does_not_open_the_device(self, monkeypatch) -> None:
        """Opening and closing here leaves a window in which some V4L2 drivers
        report the camera busy -- so the check would occasionally cause the
        failure it exists to find."""
        import vision.camera as camera_module
        from app.config import Settings

        opened = {"n": 0}
        monkeypatch.setattr(
            camera_module.Camera,
            "open",
            lambda self: opened.__setitem__("n", opened["n"] + 1),
        )
        launcher.check_camera(Settings())
        assert opened["n"] == 0

    def test_a_headless_linux_session_warns_about_the_display(
        self, monkeypatch
    ) -> None:
        """With no DISPLAY the projector window cannot open, the display layer
        discards frames, and the system runs perfectly while projecting
        nothing."""
        from app.config import Settings

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

        check = launcher.check_display(Settings())
        assert check.status == WARN
        assert "DISPLAY" in check.fix

    def test_missing_calibration_warns_loudly_with_the_wizard_command(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr("projection.mapper.load_calibration", lambda *a, **k: None)
        check = launcher.check_calibration()
        assert check.status == WARN
        assert "calibration_app" in check.fix

    def test_a_solved_calibration_passes(self, monkeypatch) -> None:
        from app.models import ProjectorCalibration

        monkeypatch.setattr(
            "projection.mapper.load_calibration",
            lambda *a, **k: ProjectorCalibration(
                projector_width=1920,
                projector_height=1080,
                rmse_px=3.2,
                is_calibrated=True,
                created_at="2026-01-02T03:04:05",
            ),
        )
        check = launcher.check_calibration()
        assert check.status == OK
        assert "3.2 px" in check.detail

    def test_a_poor_calibration_warns_rather_than_passing(self, monkeypatch) -> None:
        from app.models import ProjectorCalibration

        monkeypatch.setattr(
            "projection.mapper.load_calibration",
            lambda *a, **k: ProjectorCalibration(
                projector_width=1920,
                projector_height=1080,
                rmse_px=45.0,
                is_calibrated=True,
            ),
        )
        check = launcher.check_calibration()
        assert check.status == WARN
        assert check.fix


# ---------------------------------------------------------------------------
# Preflight as a whole
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_a_clean_machine_produces_no_failures(self, args) -> None:
        args.mock = True
        checks, settings = launcher.preflight(args)
        assert settings is not None
        assert not [c for c in checks if c.failed]

    def test_every_failing_check_carries_a_fix(self, args) -> None:
        """The contract the whole report rests on."""
        args.config = Path("does-not-exist-and-is-not-yaml")
        checks, _ = launcher.preflight(args)
        for check in checks:
            if check.failed:
                assert check.fix, f"check {check.name!r} failed without saying what to do"

    def test_preflight_stops_at_a_failure_that_invalidates_the_rest(
        self, args, tmp_path
    ) -> None:
        """A bad config makes every later check meaningless -- they all read
        settings -- so it returns rather than printing a cascade with one
        cause."""
        bad = tmp_path / "config.yaml"
        bad.write_text("table_preset: nonsense\n", encoding="utf-8")
        args.config = bad

        checks, settings = launcher.preflight(args)
        assert settings is None
        assert checks[-1].name == "config" and checks[-1].failed

    def test_mock_skips_the_calibration_check(self, args) -> None:
        """Under --mock nothing is projected onto a real table, so an absent
        calibration is not worth a warning."""
        args.mock = True
        checks, _ = launcher.preflight(args)
        assert not [c for c in checks if c.name == "calibration"]

    def test_no_loop_skips_the_hardware_checks(self, args) -> None:
        args.no_loop = True
        checks, _ = launcher.preflight(args)
        names = {c.name for c in checks}
        assert "camera" not in names and "display" not in names
        assert "web port" in names, "the panel still needs its port"

    def test_headless_skips_the_port_check(self, args) -> None:
        args.headless = True
        args.mock = True
        checks, _ = launcher.preflight(args)
        names = {c.name for c in checks}
        assert "web port" not in names
        assert "camera" in names

    def test_mock_is_reflected_in_the_report(self, args) -> None:
        """--mock forces both after the config loads, so the report has to say
        so or it contradicts what is about to happen."""
        args.mock = True
        checks, settings = launcher.preflight(args)
        assert settings.camera.use_mock and settings.projector.use_mock
        camera = next(c for c in checks if c.name == "camera")
        assert camera.status == WARN

    def test_the_report_prints_without_a_terminal(self, args, capsys) -> None:
        checks, _ = launcher.preflight(args)
        launcher.print_report(checks)
        out = capsys.readouterr().out
        assert "Preflight" in out
        for check in checks:
            assert check.name in out

    def test_the_report_has_no_escape_codes_when_piped(self, args, capsys) -> None:
        """It gets piped into journalctl, where colour is noise."""
        launcher.print_report([Check("thing", FAIL, "broke", "do this")])
        assert "\033[" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_check_only_runs_nothing(self, run_spy, capsys) -> None:
        assert launcher.main(["--check", "--mock"]) == 0
        assert not run_spy.calls
        assert "Nothing was started" in capsys.readouterr().out

    def test_a_failed_check_blocks_the_start(self, monkeypatch, run_spy, capsys) -> None:
        monkeypatch.setattr(
            launcher,
            "preflight",
            lambda _args: ([Check("thing", FAIL, "broke", "do this")], None),
        )

        assert launcher.main(["--mock"]) == 1
        assert not run_spy.calls
        assert "--force" in capsys.readouterr().out

    def test_force_starts_anyway(self, monkeypatch, run_spy, capsys) -> None:
        monkeypatch.setattr(
            launcher,
            "preflight",
            lambda _args: ([Check("thing", FAIL, "broke", "do this")], _settings()),
        )

        assert launcher.main(["--mock", "--force"]) == 0
        assert run_spy.args.mock is True
        assert "starting anyway" in capsys.readouterr().out

    def test_skip_checks_hands_over_immediately(self, monkeypatch, run_spy, capsys) -> None:
        monkeypatch.setattr(
            launcher, "preflight", lambda _a: pytest.fail("preflight ran under --skip-checks")
        )

        assert launcher.main(["--skip-checks", "--mock"]) == 0
        assert run_spy.args.mock is True
        assert capsys.readouterr().out == "", "no report when the checks are skipped"

    def test_the_parsed_namespace_reaches_the_application(self, run_spy) -> None:
        """The launcher parses once and hands the namespace over, rather than
        rebuilding an argv for app.main to parse a second time."""
        launcher.main(["--skip-checks", "--mock", "--frames", "7", "--port", "8123"])
        assert run_spy.args.frames == 7
        assert run_spy.args.port == 8123

    def test_the_panel_url_is_printed(self, run_spy, capsys) -> None:
        """localhost is useless from the phone the panel is designed for."""
        launcher.main(["--mock", "--port", "8123", "--frames", "1"])
        assert "http://localhost:8123/" in capsys.readouterr().out


class TestPathBootstrap:
    def test_the_package_dir_is_first_on_sys_path(self, monkeypatch) -> None:
        """Prepended, not appended: ``app`` is a common enough name that another
        one on the path would otherwise win and the import would succeed with
        the wrong module."""
        monkeypatch.setattr(sys, "path", ["/somewhere/else", str(launcher.PACKAGE_DIR)])
        launcher._bootstrap_path()
        assert sys.path[0] == str(launcher.PACKAGE_DIR)
        assert sys.path.count(str(launcher.PACKAGE_DIR)) == 1

    def test_bootstrap_is_idempotent(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "path", [str(launcher.PACKAGE_DIR), "/elsewhere"])
        before = list(sys.path)
        launcher._bootstrap_path()
        assert sys.path == before

    def test_the_launcher_sits_beside_the_packages_it_imports(self) -> None:
        """If this file is ever moved, the path bootstrap silently points
        somewhere useless -- so pin the assumption."""
        assert (launcher.PACKAGE_DIR / "app" / "main.py").is_file()
        assert (launcher.PACKAGE_DIR / "config.yaml").is_file()


def _settings():
    from app.config import Settings

    return Settings()
