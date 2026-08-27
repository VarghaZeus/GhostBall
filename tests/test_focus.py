"""Lens focus over V4L2, and the log-once helper.

Focus cannot be tested against real hardware here, so what *is* pinned is
everything that would silently do the wrong thing on the Pi:

* the ioctl request numbers and struct layouts, against the published constants
  -- a wrong request number fails as EINVAL from a device that is otherwise
  working perfectly, which is a miserable thing to debug from a hex literal;
* discovery by driver name, against a fake sysfs tree -- because the subdev
  index is not stable across reboots and a hardcoded path would drive whatever
  happened to land at that number;
* the readback check, which is the only thing that can tell "the lens moved"
  from "the write was accepted and ignored". That is the failure a half-seated
  ribbon produces, and it is why ``apply_focus`` verifies rather than trusting
  a successful write.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from pathlib import Path

import pytest

from utils.logging import ChangeLogger
from vision import focus as focus_module
from vision.focus import (
    V4L2_CID_FOCUS_ABSOLUTE,
    VIDIOC_G_CTRL,
    VIDIOC_QUERYCTRL,
    VIDIOC_S_CTRL,
    FocusRange,
    FocusStatus,
    apply_focus,
    find_lens_subdev,
    query_focus_range,
    read_focus,
    write_focus,
)

# ---------------------------------------------------------------------------
# The kernel ABI
# ---------------------------------------------------------------------------


class TestIoctlEncoding:
    def test_request_numbers_match_the_published_constants(self) -> None:
        """From ``linux/videodev2.h``. If these drift, every call returns EINVAL
        against hardware that is working."""
        assert VIDIOC_G_CTRL == 0xC008561B
        assert VIDIOC_S_CTRL == 0xC008561C
        assert VIDIOC_QUERYCTRL == 0xC0445624

    def test_the_control_id_is_the_camera_class_base_plus_ten(self) -> None:
        assert V4L2_CID_FOCUS_ABSOLUTE == 0x009A090A

    def test_struct_sizes_match_the_kernel(self) -> None:
        """``v4l2_control`` is 8 bytes and ``v4l2_queryctrl`` 68. A mismatch
        here corrupts the request number, since the size is encoded into it."""
        assert struct.calcsize(focus_module._CTRL_FORMAT) == 8
        assert struct.calcsize(focus_module._QUERYCTRL_FORMAT) == 68

    def test_the_size_is_encoded_into_the_request(self) -> None:
        assert (VIDIOC_S_CTRL >> 16) & 0x3FFF == 8
        assert (VIDIOC_QUERYCTRL >> 16) & 0x3FFF == 68


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def make_sysfs(tmp_path: Path, devices: dict[str, str]) -> Path:
    """A fake ``/sys/class/video4linux`` tree."""
    root = tmp_path / "sysfs"
    root.mkdir()
    for node, name in devices.items():
        entry = root / node
        entry.mkdir()
        (entry / "name").write_text(name + "\n", encoding="utf-8")
    return root


class TestDiscovery:
    #: What the Pi 5 actually shows with the Arducam attached.
    REAL_TREE = {
        "video0": "rp1-cfe-csi2_ch0",
        "video19": "pispbe-input",
        "v4l-subdev0": "imx519 10-001a",
        "v4l-subdev3": "ak7375 10-000c",
    }

    def test_the_lens_is_found_by_driver_name(self, tmp_path) -> None:
        lens = find_lens_subdev(
            "ak7375", sysfs_root=make_sysfs(tmp_path, self.REAL_TREE), dev_root=Path("/dev")
        )
        assert lens is not None
        assert lens.path == Path("/dev/v4l-subdev3")
        assert lens.name == "ak7375 10-000c"

    def test_a_moved_subdev_index_is_still_found(self, tmp_path) -> None:
        """The reason discovery exists at all. Device enumeration order is not
        stable across reboots, and a hardcoded ``/dev/v4l-subdev3`` would drive
        the image sensor here rather than the lens."""
        shuffled = {
            "v4l-subdev0": "ak7375 10-000c",
            "v4l-subdev3": "imx519 10-001a",
        }
        lens = find_lens_subdev(
            "ak7375", sysfs_root=make_sysfs(tmp_path, shuffled), dev_root=Path("/dev")
        )
        assert lens is not None
        assert lens.path == Path("/dev/v4l-subdev0")

    def test_matching_is_case_insensitive_and_partial(self, tmp_path) -> None:
        """The I2C bus and address suffix vary with how the camera is wired, so
        only the driver name itself can be matched on."""
        lens = find_lens_subdev(
            "AK7375", sysfs_root=make_sysfs(tmp_path, self.REAL_TREE), dev_root=Path("/dev")
        )
        assert lens is not None

    def test_no_lens_returns_none_rather_than_raising(self, tmp_path) -> None:
        """A dev box has no V4L2 at all, and importing this module must not be
        conditional on hardware."""
        tree = make_sysfs(tmp_path, {"video0": "some-usb-webcam"})
        assert find_lens_subdev("ak7375", sysfs_root=tree, dev_root=Path("/dev")) is None

    def test_a_missing_sysfs_returns_none(self, tmp_path) -> None:
        assert find_lens_subdev("ak7375", sysfs_root=tmp_path / "nope") is None

    def test_discovery_is_deterministic_with_duplicates(self, tmp_path) -> None:
        """Two identical lenses must resolve to the same one every boot, or the
        focus value lands on a different camera each time."""
        tree = make_sysfs(
            tmp_path, {"v4l-subdev5": "ak7375 10-000c", "v4l-subdev1": "ak7375 11-000c"}
        )
        first = find_lens_subdev("ak7375", sysfs_root=tree, dev_root=Path("/dev"))
        second = find_lens_subdev("ak7375", sysfs_root=tree, dev_root=Path("/dev"))
        assert first == second
        assert first.path.name == "v4l-subdev1"


# ---------------------------------------------------------------------------
# A fake lens
# ---------------------------------------------------------------------------


class FakeLens:
    """A V4L2 subdev that behaves like an ak7375 -- or misbehaves on request."""

    def __init__(self, minimum=0, maximum=4095, position=0, sticks=True, fail_on=()):
        self.range = FocusRange(minimum, maximum, 1, 0)
        self.position = position
        #: When ``False``, writes are accepted and discarded -- which is exactly
        #: how a lens on a half-seated ribbon presents.
        self.sticks = sticks
        self.fail_on = set(fail_on)
        self.writes: list[int] = []
        #: ``perf_counter`` at each write. Needed because the ordering of the
        #: backoff and the target is not the property that matters -- the gap
        #: between them is. See ``TestApproachDirection``.
        self.write_times: list[float] = []

    def opener(self, device):
        class _Handle:
            def __enter__(self):
                return 7  # a plausible fd; the fake ioctl ignores it

            def __exit__(self, *exc):
                return None

        return _Handle()

    def ioctl(self, fd, request, payload):
        if request in self.fail_on:
            raise OSError(22, "Invalid argument")
        if request == VIDIOC_QUERYCTRL:
            return struct.pack(
                focus_module._QUERYCTRL_FORMAT,
                V4L2_CID_FOCUS_ABSOLUTE, 2, b"Focus, Absolute",
                self.range.minimum, self.range.maximum, self.range.step, self.range.default,
                0, 0, 0,
            )
        control_id, value = struct.unpack(focus_module._CTRL_FORMAT, payload)
        if request == VIDIOC_S_CTRL:
            self.writes.append(value)
            self.write_times.append(time.perf_counter())
            if self.sticks:
                self.position = value
            return payload
        if request == VIDIOC_G_CTRL:
            return struct.pack(focus_module._CTRL_FORMAT, control_id, self.position)
        raise AssertionError(f"unexpected ioctl {request:#x}")


@pytest.fixture
def lens(monkeypatch):
    fake = FakeLens()
    monkeypatch.setattr(focus_module, "_ioctl", fake.ioctl)
    return fake


DEVICE = Path("/dev/v4l-subdev-fake")


class TestControlAccess:
    def test_the_range_is_read_from_the_driver(self, lens) -> None:
        """Queried rather than assumed: 0-4095 is this motor's range, not a
        property of V4L2, and clamping against a hardcoded span would
        mis-drive a different lens."""
        found = query_focus_range(DEVICE, opener=lens.opener)
        assert (found.minimum, found.maximum) == (0, 4095)

    def test_write_then_read_round_trips(self, lens) -> None:
        write_focus(DEVICE, 512, opener=lens.opener)
        assert read_focus(DEVICE, opener=lens.opener) == 512

    def test_an_ioctl_failure_becomes_a_focus_error(self, lens) -> None:
        lens.fail_on = {VIDIOC_S_CTRL}
        with pytest.raises(focus_module.FocusError, match="cannot set focus_absolute"):
            write_focus(DEVICE, 512, opener=lens.opener)

    def test_a_permissions_failure_names_the_video_group(self, monkeypatch) -> None:
        """The commonest real cause, and not one you would guess from EACCES."""

        def denied(fd, request, payload):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(focus_module, "_ioctl", denied)
        fake = FakeLens()
        with pytest.raises(focus_module.FocusError, match="video"):
            write_focus(DEVICE, 512, opener=fake.opener)


# ---------------------------------------------------------------------------
# apply_focus -- the operation with the readback
# ---------------------------------------------------------------------------


class TestApplyFocus:
    def test_a_confirmed_set_reports_ok(self, lens) -> None:
        status = apply_focus(512, device=DEVICE, opener=lens.opener)
        assert status.ok
        assert status.available
        assert (status.requested, status.actual) == (512, 512)
        assert lens.writes == [512]

    def test_a_lens_that_ignores_the_write_is_not_ok(self, monkeypatch, caplog) -> None:
        """The failure this whole design exists for.

        The write succeeds at the syscall level and the motor does not move, so
        anything that trusts a clean return reports success while every frame
        stays soft.
        """
        fake = FakeLens(sticks=False, position=0)
        monkeypatch.setattr(focus_module, "_ioctl", fake.ioctl)

        with caplog.at_level(logging.ERROR):
            status = apply_focus(512, device=DEVICE, opener=fake.opener)

        assert status.available, "the device was found and accepted the write"
        assert not status.ok
        assert status.mismatch
        assert (status.requested, status.actual) == (512, 0)
        assert "ribbon" in status.detail, "the detail must name the likely cause"
        assert any(record.levelno >= logging.ERROR for record in caplog.records)

    def test_an_out_of_range_value_is_clamped_not_rejected(self, monkeypatch) -> None:
        fake = FakeLens(minimum=0, maximum=1023)
        monkeypatch.setattr(focus_module, "_ioctl", fake.ioctl)

        status = apply_focus(4095, device=DEVICE, opener=fake.opener)
        assert status.ok
        assert status.actual == 1023

    def test_a_missing_lens_reports_unavailable_without_raising(self, monkeypatch) -> None:
        """A camera that will not focus is a degraded system; refusing to start
        would be a dead one."""
        status = apply_focus(512, finder=lambda _name: None)
        assert not status.available
        assert not status.ok
        assert "no V4L2 subdev" in status.detail

    def test_a_missing_lens_lists_what_was_present(self, monkeypatch) -> None:
        """When the lens is absent the next question is "what *is* there" --
        which separates a camera that never probed from one that probed without
        its focus motor."""
        monkeypatch.setattr(
            focus_module, "list_v4l_devices", lambda: [("video0", "rp1-cfe-csi2_ch0")]
        )
        status = apply_focus(512, finder=lambda _name: None)
        assert "rp1-cfe-csi2_ch0" in status.detail

    def test_a_dead_device_reports_unavailable(self, monkeypatch) -> None:
        fake = FakeLens(fail_on={VIDIOC_QUERYCTRL})
        monkeypatch.setattr(focus_module, "_ioctl", fake.ioctl)

        status = apply_focus(512, device=DEVICE, opener=fake.opener)
        assert not status.available
        assert not status.ok

    def test_the_status_serialises_for_the_api(self, lens) -> None:
        payload = apply_focus(512, device=DEVICE, opener=lens.opener).as_dict()
        assert set(payload) == {
            "available", "device", "lens_name", "requested", "actual", "ok", "detail",
            "source", "calibrated",
        }

    def test_an_untried_status_is_not_a_failure_shape(self) -> None:
        """Before the camera opens there is nothing to report, and that must not
        render as a focus fault on the panel."""
        status = FocusStatus()
        assert not status.available and not status.ok and not status.mismatch


# ---------------------------------------------------------------------------
# Log-once
# ---------------------------------------------------------------------------


class TestChangeLogger:
    @pytest.fixture
    def changes(self):
        return ChangeLogger(logging.getLogger("test.changes"))

    def test_a_repeated_condition_logs_once(self, changes, caplog) -> None:
        """The whole point. At 30 FPS an ungated warning is 1,800 lines a
        minute, which does not merely waste an SD card -- it buries every other
        line in the log."""
        with caplog.at_level(logging.WARNING, logger="test.changes"):
            for _ in range(100):
                changes.report("k", "bad", logging.WARNING, "it is bad")
        assert len(caplog.records) == 1

    def test_a_changed_condition_logs_again(self, changes, caplog) -> None:
        """Repetition is what gets suppressed, not information."""
        with caplog.at_level(logging.WARNING, logger="test.changes"):
            changes.report("k", "error A", logging.WARNING, "A")
            changes.report("k", "error A", logging.WARNING, "A")
            changes.report("k", "error B", logging.WARNING, "B")
        assert [r.getMessage() for r in caplog.records] == ["A", "B"]

    def test_recovery_is_reported_once(self, changes, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="test.changes"):
            changes.report("k", "bad", logging.WARNING, "broke")
            assert changes.recovered("k", logging.INFO, "fixed")
            assert not changes.recovered("k", logging.INFO, "fixed")
        assert [r.getMessage() for r in caplog.records] == ["broke", "fixed"]

    def test_recovery_is_silent_when_nothing_was_wrong(self, changes, caplog) -> None:
        """Otherwise every healthy frame from startup announces a recovery from
        a problem that never happened."""
        with caplog.at_level(logging.INFO, logger="test.changes"):
            for _ in range(50):
                changes.recovered("k", logging.INFO, "fixed")
        assert caplog.records == []

    def test_a_condition_can_recur_after_recovering(self, changes, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="test.changes"):
            changes.report("k", "bad", logging.WARNING, "broke")
            changes.recovered("k", logging.INFO, "fixed")
            changes.report("k", "bad", logging.WARNING, "broke")
        assert len(caplog.records) == 3

    def test_none_is_a_usable_state(self, changes, caplog) -> None:
        """``None`` is a legitimate state value, which is why the sentinel for
        "never seen" has to be something else."""
        with caplog.at_level(logging.INFO, logger="test.changes"):
            assert changes.report("k", None, logging.INFO, "first")
            assert not changes.report("k", None, logging.INFO, "first")
        assert len(caplog.records) == 1

    def test_keys_are_independent(self, changes, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="test.changes"):
            changes.report("a", True, logging.WARNING, "a")
            changes.report("b", True, logging.WARNING, "b")
        assert len(caplog.records) == 2

    def test_clear_resets(self, changes, caplog) -> None:
        """Module-level instances outlive a run; without this the second run in
        a process silently logs nothing."""
        with caplog.at_level(logging.WARNING, logger="test.changes"):
            changes.report("k", "bad", logging.WARNING, "broke")
            changes.clear()
            changes.report("k", "bad", logging.WARNING, "broke")
        assert len(caplog.records) == 2


class TestPerFrameCallSites:
    """The sites that were actually spamming, checked end to end."""

    def test_the_felt_coverage_notice_is_reported_once(self, caplog) -> None:
        """Reported at 15 lines a second on the real rig. Keyed on a constant
        rather than the coverage figure, which jitters frame to frame and would
        re-log regardless."""
        import numpy as np

        from vision import detection
        from vision.detection import _cloth_mask

        detection._changes.clear()
        settings = detection.get_settings()

        # A frame with no felt at all, so coverage is 0 and the adaptive path is
        # taken on every call.
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        frame[:, :, 2] = 200  # solid red: nothing the felt thresholds will match

        from app.models import TableBoundary, Vec2

        boundary = TableBoundary(
            top_left=Vec2(10, 10),
            top_right=Vec2(190, 10),
            bottom_right=Vec2(190, 110),
            bottom_left=Vec2(10, 110),
            center=Vec2(100, 60),
            width_px=180.0,
            height_px=100.0,
            confidence=0.9,
        )

        with caplog.at_level(logging.INFO, logger="vision.detection"):
            for _ in range(30):
                _cloth_mask(frame, boundary, 1.0, settings)

        spam = [r for r in caplog.records if "cover only" in r.getMessage()]
        assert len(spam) <= 1, f"logged {len(spam)} times for 30 frames"
        detection._changes.clear()

    def test_simulator_skipped_balls_is_reported_once(self, caplog) -> None:
        """``simulate_shot`` runs on every frame the player is aiming, so
        anything logged inside it is logged 30 times a second."""
        from app.models import Ball, Vec2
        from physics import simulator

        simulator._changes.clear()
        # No table_pos: exactly the state an un-homographied table produces, and
        # what the warning is about.
        orphan = Ball(id="1", center_px=Vec2(5, 5), radius_px=10)

        with caplog.at_level(logging.WARNING, logger="physics.simulator"):
            for _ in range(20):
                simulator.simulate_shot(Vec2(20.0, 20.0), 0.0, 50.0, [orphan])

        spam = [r for r in caplog.records if "no table position" in r.getMessage()]
        assert len(spam) <= 1, f"logged {len(spam)} times for 20 simulated shots"
        simulator._changes.clear()


# ---------------------------------------------------------------------------
# Backlash
# ---------------------------------------------------------------------------


class TestApproachDirection:
    """A voice coil lands in a slightly different place depending on which way
    it travelled. Every move here arrives from below, so that the sweep and the
    cold-boot apply agree -- otherwise the rig is measurably softer at boot than
    at calibration, with nothing in any log pointing at the cause."""

    def test_moving_up_is_a_single_move(self, lens) -> None:
        lens.position = 100
        focus_module.approach_focus(DEVICE, 1400, FocusRange(0, 4095, 1, 0), opener=lens.opener)
        assert lens.writes == [1400]

    def test_moving_down_backs_off_first_and_arrives_from_below(self, lens) -> None:
        lens.position = 2000
        span = FocusRange(0, 4095, 1, 0)
        focus_module.approach_focus(DEVICE, 1400, span, opener=lens.opener)
        assert lens.writes == [1400 - focus_module.backlash_margin(span), 1400]
        assert lens.writes[0] < lens.writes[1], "the final move must be upward"

    def test_the_backoff_is_given_time_to_actually_happen(self, lens) -> None:
        """The two writes must be separated in *time*, not merely ordered.

        The bug this catches passed every ordering assertion above. A focus write
        returns once the kernel has latched the value over I2C, which is long
        before the lens has physically travelled -- so issuing the backoff and
        the target back to back lets the voice coil retarget in flight. The lens
        never reaches the backoff point and still arrives at the target from
        above, which is precisely the failure the backoff exists to prevent.

        Silent, too: ``lens.writes`` reads ``[1336, 1400]`` either way, the logs
        say "backing off", and the only evidence is a rig that focuses slightly
        differently than it measured.

        It fires on the path that matters most. A sweep finishes with the lens at
        the top of its range, then applies the chosen value -- so the backoff
        branch runs, and the calibration ends up measured from below and applied
        from above on the very run that saves it.

        Run at the real default rather than a shortened one, because a settle
        that is only exercised at a test value is a settle nobody has checked.
        """
        lens.position = 2000
        focus_module.approach_focus(DEVICE, 1400, FocusRange(0, 4095, 1, 0), opener=lens.opener)

        assert len(lens.write_times) == 2
        gap = lens.write_times[1] - lens.write_times[0]
        # Most of the settle rather than all of it: `time.sleep` guarantees a
        # lower bound, but asserting the exact figure would make this fail on a
        # coarse-grained clock for no reason.
        assert gap >= focus_module.BACKLASH_SETTLE_SECONDS * 0.8, (
            f"backoff and target were {gap * 1000:.1f} ms apart; the lens cannot "
            "have reached the backoff point"
        )

    def test_a_plain_upward_move_does_not_wait(self, lens) -> None:
        """No backoff, no settle. The wait is the cost of the extra move only.

        Worth pinning because this is the common path -- every cold boot and
        every ascending step of a sweep -- and slowing it down by a tenth of a
        second per move would add seconds to a calibration for nothing.
        """
        lens.position = 100
        focus_module.approach_focus(
            DEVICE, 1400, FocusRange(0, 4095, 1, 0), opener=lens.opener
        )
        assert len(lens.write_times) == 1

    def test_the_backoff_is_clamped_to_the_lens_range(self, lens) -> None:
        """Near zero there is no room to back off below the minimum."""
        lens.position = 40
        focus_module.approach_focus(DEVICE, 10, FocusRange(0, 4095, 1, 0), opener=lens.opener)
        assert lens.writes == [0, 10]

    def test_apply_focus_uses_the_approach(self, lens) -> None:
        lens.position = 3000
        status = apply_focus(1400, device=DEVICE, opener=lens.opener)
        assert status.ok
        assert lens.writes[-1] == 1400
        assert lens.writes[0] < 1400, "arrived from above without backing off"


# ---------------------------------------------------------------------------
# The calibration file
# ---------------------------------------------------------------------------


class TestFocusCalibrationFile:
    def test_round_trips(self, tmp_path) -> None:
        calibration = focus_module.FocusCalibration(
            focus_absolute=1400,
            peak_sharpness=812.5,
            bare_table_sharpness=41.2,
            per_target=(
                focus_module.TargetPeak("centre", (960.0, 540.0), 1400, 812.5, 6.1),
                focus_module.TargetPeak("top_left", (300.0, 200.0), 1390, 740.0, 5.2),
            ),
            tilt_spread=10,
            camera_resolution="1920x1080",
            lens_name="ak7375 10-000c",
            created_at="2026-08-25T10:00:00+00:00",
        )
        path = tmp_path / "focus.json"
        focus_module.save_focus_calibration(calibration, path)
        assert focus_module.load_focus_calibration(path) == calibration

    def test_a_missing_file_is_not_an_error(self, tmp_path) -> None:
        assert focus_module.load_focus_calibration(tmp_path / "absent.json") is None

    def test_a_corrupt_file_reads_as_uncalibrated(self, tmp_path, caplog) -> None:
        """Failing to boot because a calibration file has a typo in it would be
        far worse than booting uncalibrated -- which is a state the panel
        already reports, so nothing is hidden by degrading this way."""
        path = tmp_path / "focus.json"
        path.write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.ERROR, logger="vision.focus"):
            assert focus_module.load_focus_calibration(path) is None
        assert any("unreadable" in r.getMessage() for r in caplog.records)

    def test_a_file_from_the_other_approach_direction_warns(self, tmp_path, caplog) -> None:
        """The stored value has the backlash offset baked into it, so a change
        to the approach direction has to be visible rather than silently
        shifting every saved value."""
        path = tmp_path / "focus.json"
        path.write_text(
            json.dumps({"focus_absolute": 1400, "approach": "descending"}), encoding="utf-8"
        )
        with caplog.at_level(logging.WARNING, logger="vision.focus"):
            calibration = focus_module.load_focus_calibration(path)
        assert calibration is not None
        assert any("backlash" in r.getMessage() for r in caplog.records)

    def test_the_approach_direction_is_recorded_by_default(self) -> None:
        assert focus_module.FocusCalibration(focus_absolute=1).approach == "ascending"


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


class TestResolveFocusValue:
    def test_nothing_saved_means_uncalibrated_not_a_default(self, tmp_path) -> None:
        """The reason the config key is nullable. A guessed focus produces a rig
        that is soft for reasons nobody can see; "not calibrated" points at the
        fix."""
        from app.config import CameraSettings

        value, source = focus_module.resolve_focus_value(
            CameraSettings(), tmp_path / "absent.json"
        )
        assert value is None
        assert source == "none"

    def test_the_calibration_file_is_used_when_present(self, tmp_path) -> None:
        from app.config import CameraSettings

        path = tmp_path / "focus.json"
        focus_module.save_focus_calibration(
            focus_module.FocusCalibration(focus_absolute=1400), path
        )
        assert focus_module.resolve_focus_value(CameraSettings(), path) == (1400, "file")

    def test_a_config_override_beats_the_file(self, tmp_path) -> None:
        """Someone typing a number into a config file has made a more specific
        statement than a calibration run from last month."""
        from app.config import CameraSettings

        path = tmp_path / "focus.json"
        focus_module.save_focus_calibration(
            focus_module.FocusCalibration(focus_absolute=1400), path
        )
        settings = CameraSettings(focus_absolute=900)
        assert focus_module.resolve_focus_value(settings, path) == (900, "config")

    def test_the_shipped_config_does_not_pin_a_focus_value(self) -> None:
        """If config.yaml carried a number, first run would be indistinguishable
        from calibrated-to-that-number, and nothing could decide whether to ask
        for a calibration."""
        from app.config import load_settings

        assert load_settings().camera.focus_absolute is None


class TestUncalibratedIsNotAFault:
    def test_status_reports_uncalibrated_distinctly(self) -> None:
        """``ok=False`` covers both "the lens would not move" and "we never told
        it where to go". Those want different words, so the panel gets a
        different flag for each."""
        status = FocusStatus(detail="no focus calibration for this rig")
        assert not status.calibrated
        assert not status.ok
        assert status.as_dict()["source"] == "none"

    def test_a_calibrated_status_says_where_it_came_from(self, lens) -> None:
        status = apply_focus(1400, device=DEVICE, opener=lens.opener, source="file")
        assert status.calibrated
        assert status.as_dict()["source"] == "file"


class TestBackendWiring:
    """The startup path, which is what makes every boot correct."""

    def _backend(self):
        from vision.camera import Picamera2Backend

        return Picamera2Backend()

    def test_an_uncalibrated_rig_does_not_guess_a_value(self, tmp_path, monkeypatch) -> None:
        """The lens stays where it powered up, which is visibly soft and points
        at the real fix. A plausible-looking guess would leave a permanently
        mediocre picture with no symptom to chase."""
        from app.config import CameraSettings

        monkeypatch.setattr(
            focus_module, "focus_calibration_path", lambda: tmp_path / "absent.json"
        )
        applied = []
        monkeypatch.setattr(
            focus_module, "apply_focus", lambda *a, **k: applied.append(a) or FocusStatus()
        )

        backend = self._backend()
        backend._apply_focus(CameraSettings())

        assert not applied, "nothing should have been written to the lens"
        assert not backend.focus_status().calibrated
        assert "focus_sweep" in backend.focus_status().detail, "must say how to fix it"

    def test_a_saved_value_is_applied_at_startup(self, tmp_path, monkeypatch) -> None:
        """Mandatory every boot: the VCM resets to 0 on power cycle, so a saved
        value that is not re-applied means running unfocused."""
        from app.config import CameraSettings

        path = tmp_path / "focus.json"
        focus_module.save_focus_calibration(
            focus_module.FocusCalibration(focus_absolute=1400), path
        )
        monkeypatch.setattr(focus_module, "focus_calibration_path", lambda: path)

        seen = {}

        def fake_apply(value, **kwargs):
            seen["value"] = value
            seen["source"] = kwargs.get("source")
            return FocusStatus(available=True, requested=value, actual=value, ok=True, **{})

        monkeypatch.setattr(focus_module, "apply_focus", fake_apply)

        backend = self._backend()
        backend._apply_focus(CameraSettings())

        assert seen["value"] == 1400
        assert seen["source"] == "file"

    def test_focus_can_be_switched_off_for_a_fixed_focus_lens(self) -> None:
        from app.config import CameraSettings

        backend = self._backend()
        backend._apply_focus(CameraSettings(focus_enabled=False))
        assert "disabled" in backend.focus_status().detail


# ---------------------------------------------------------------------------
# Range-relative constants
# ---------------------------------------------------------------------------


class TestNothingAssumesTheOldLensRange:
    """0-4095 is the ak7375's range, not a universal one.

    Swapping to a dw9807 (0-1023, Camera Module 3) silently changed the meaning
    of every number that had been chosen against the old lens. Nothing reported
    it, because each number was still *a* valid number -- just no longer the one
    anybody had reasoned about.
    """

    AK7375 = FocusRange(0, 4095, 1, 0)
    DW9807 = FocusRange(0, 1023, 1, 480)

    def test_the_backlash_margin_is_the_same_fraction_of_travel_on_both(self) -> None:
        """It was a flat 64 counts: 1.5% of the ak7375's travel and 6.3% of the
        dw9807's -- four times the intended back-off on the new lens."""
        for focus_range in (self.AK7375, self.DW9807):
            fraction = focus_module.backlash_margin(focus_range) / focus_range.span
            assert abs(fraction - focus_module.BACKLASH_FRACTION) < 0.01

    def test_the_ak7375_still_gets_exactly_the_margin_it_always_had(self) -> None:
        """The old lens must be bit-for-bit unaffected. This value was measured
        against that motor, so a refactor that moved it would be changing tuning
        under the guise of generalising."""
        assert focus_module.backlash_margin(self.AK7375) == 64

    def test_a_tiny_range_still_gets_a_usable_margin(self) -> None:
        """Below a few counts a back-off is smaller than the motor's own settling
        error and stops meaning anything."""
        assert focus_module.backlash_margin(FocusRange(0, 15, 1, 0)) >= (
            focus_module.BACKLASH_MIN_COUNTS
        )

    def test_the_backoff_scales_with_the_lens(self, lens) -> None:
        """End to end, not just the helper: a descending move on the new lens
        must back off by the new lens's margin."""
        lens.range = self.DW9807
        lens.position = 900
        focus_module.approach_focus(DEVICE, 500, self.DW9807, opener=lens.opener)
        assert lens.writes == [500 - focus_module.backlash_margin(self.DW9807), 500]


class TestFocusRangeHelpers:
    def test_contains_distinguishes_out_of_range_from_merely_clamped(self) -> None:
        """The distinction the whole readback diagnosis rests on. ``clamp``
        returns a different number for an out-of-range value and for a value that
        was already fine at the boundary, and cannot tell the caller which
        happened."""
        r = FocusRange(0, 1023, 1, 480)
        assert r.contains(1023) and r.contains(0)
        assert not r.contains(1024)
        assert not r.contains(-1)
        # Identical results from clamp, opposite meanings.
        assert r.clamp(1023) == r.clamp(1024) == 1023

    def test_snap_is_a_no_op_at_step_one(self) -> None:
        """Both lenses seen so far report step=1, so this must cost nothing."""
        r = FocusRange(0, 1023, 1, 480)
        for value in (0, 1, 480, 1023):
            assert r.snap(value) == value

    def test_snap_aligns_to_a_coarse_step_grid(self) -> None:
        """A driver that quantises makes the readback legitimately differ from
        the request, which reads exactly like a motor that will not track. Removed
        at source rather than diagnosed after the fact."""
        r = FocusRange(0, 1000, 8, 0)
        assert r.snap(100) == 96
        assert r.snap(7) == 0

    def test_snap_never_leaves_the_range(self) -> None:
        r = FocusRange(0, 1000, 8, 0)
        assert r.snap(5000) <= 1000
        assert r.snap(-5) == 0

    def test_span_is_the_scale_everything_else_is_relative_to(self) -> None:
        assert FocusRange(0, 1023, 1, 480).span == 1023
        assert FocusRange(0, 4095, 1, 0).span == 4095


class TestApplyFocusRespectsTheDriversRange:
    """Clamping is already covered above; what is new is *saying which kind of
    adjustment happened*. An out-of-range value and a value off the step grid
    point at different fixes -- a stale configured number versus a driver that
    quantises -- and one "adjusted" message for both loses the actionable half.
    """

    def test_an_out_of_range_request_says_so(self, monkeypatch, caplog) -> None:
        import logging

        fake = FakeLens(minimum=0, maximum=1023)
        monkeypatch.setattr(focus_module, "_ioctl", fake.ioctl)

        with caplog.at_level(logging.WARNING):
            status = apply_focus(4000, device=DEVICE, opener=fake.opener)

        # A stale value from the old lens must not present as a broken lens.
        assert status.ok and status.actual == 1023
        assert "outside the lens range 0-1023" in caplog.text

    def test_an_off_grid_request_blames_the_step_not_the_range(
        self, monkeypatch, caplog
    ) -> None:
        import logging

        fake = FakeLens(minimum=0, maximum=1000)
        fake.range = FocusRange(0, 1000, 8, 0)
        monkeypatch.setattr(focus_module, "_ioctl", fake.ioctl)

        with caplog.at_level(logging.WARNING):
            status = apply_focus(100, device=DEVICE, opener=fake.opener)

        assert status.ok and status.actual == 96
        assert "step grid" in caplog.text
        assert "outside the lens range" not in caplog.text


# ---------------------------------------------------------------------------
# libcamera's autofocus, where the sensor has one
# ---------------------------------------------------------------------------


class TestLibcameraAfIsReleased:
    """The hole in this module's founding premise.

    :mod:`vision.focus` exists because the IMX519's stock tuning file binds no AF
    algorithm, so libcamera silently drops ``AfMode``/``LensPosition`` and cannot
    move the lens. That is true of the IMX519 and is *not* a property of
    libcamera. On a sensor whose tuning does bind ``rpi.af`` -- the IMX708 of
    Camera Module 3 -- libcamera owns the very VCM this code writes to, and while
    the camera streams it will drive it.

    The symptom is nasty precisely because it clears the obvious test: every V4L2
    write is accepted and the readback disagrees at *every* position, while a
    manual ``v4l2-ctl`` write with nothing streaming sticks perfectly. So the
    cable tests fine and the old diagnosis blamed the cable anyway.
    """

    class FakeCam:
        def __init__(self, controls, refuse=False):
            self.camera_controls = controls
            self.refuse = refuse
            self.set_calls = []

        def set_controls(self, controls):
            if self.refuse:
                raise RuntimeError("control not supported")
            self.set_calls.append(controls)

    def _backend(self, cam):
        from vision.camera import Picamera2Backend

        backend = Picamera2Backend()
        backend._cam = cam
        return backend

    def test_af_is_put_in_manual_mode_when_the_sensor_has_it(self) -> None:
        cam = self.FakeCam({"AfMode": (0, 2, 0), "ExposureTime": (1, 2, 1)})
        self._backend(cam)._release_libcamera_af()

        # 0 is AfModeEnum.Manual.
        assert cam.set_calls == [{"AfMode": 0}]

    def test_a_sensor_with_no_af_is_left_alone(self) -> None:
        """The IMX519 case. There is nothing to disable and trying would log a
        misleading warning about a control that was never honoured anyway."""
        cam = self.FakeCam({"ExposureTime": (1, 2, 1)})
        self._backend(cam)._release_libcamera_af()

        assert cam.set_calls == []

    def test_a_refused_af_control_warns_and_does_not_raise(self, caplog) -> None:
        """A camera that will not focus is a degraded system; refusing to start
        would be a dead one. But it must say so, because the readbacks are about
        to start disagreeing and this is the reason."""
        import logging

        cam = self.FakeCam({"AfMode": (0, 2, 0)}, refuse=True)
        with caplog.at_level(logging.WARNING):
            self._backend(cam)._release_libcamera_af()

        assert "AfMode=Manual was refused" in caplog.text
        assert "disagree at every position" in caplog.text

    def test_unreadable_controls_do_not_raise(self) -> None:
        class Broken:
            @property
            def camera_controls(self):
                raise RuntimeError("not configured yet")

        self._backend(Broken())._release_libcamera_af()  # must not raise

    def test_focus_is_applied_after_the_stream_starts(self) -> None:
        """Ordering, and it matters. libcamera writes the lens once at start-up on
        any sensor with AF bound, so a V4L2 write issued before ``start()`` is
        overwritten a moment later -- and the log still says it succeeded.

        Asserted on the source because the alternative is a full picamera2 stub,
        and what needs pinning is a two-line ordering decision that a future
        tidy-up would otherwise silently reverse.
        """
        import inspect

        from vision.camera import Picamera2Backend

        source = inspect.getsource(Picamera2Backend.start)
        assert source.index("_release_libcamera_af") < source.index("_cam.start()"), (
            "AF must be released before the stream starts"
        )
        assert source.index("_cam.start()") < source.index("_apply_focus"), (
            "the V4L2 focus write must come after start(), or libcamera overwrites it"
        )


class TestASavedValueIsTiedToTheLensItWasMeasuredOn:
    """Motor units are not portable, and nothing about a stored number says so.

    ``focus.json`` holds a raw ``focus_absolute``. 1800 is most of the way out on
    an ak7375 (0-4095) and off the end of a dw9807 (0-1023), where it clamps to
    1023 -- a lens jammed at one extreme, a picture that is soft, and a panel
    reporting "calibrated" from a value that genuinely came from a file. Nothing
    in that chain points at the camera swap.
    """

    def _saved(self, tmp_path, lens_name, value=1800):
        import json

        path = tmp_path / "focus.json"
        path.write_text(
            json.dumps(
                {
                    "focus_absolute": value,
                    "lens_name": lens_name,
                    "approach": focus_module.APPROACH_DIRECTION,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_a_value_from_a_different_lens_is_flagged(self, tmp_path, caplog) -> None:
        import logging

        from app.config import CameraSettings

        path = self._saved(tmp_path, "ak7375 10-000c")
        settings = CameraSettings(lens_driver="dw9807")

        with caplog.at_level(logging.WARNING):
            value, source = focus_module.resolve_focus_value(settings, path)

        # Still returned: refusing to focus at all is worse than focusing on a
        # suspect number, and apply_focus clamps it safely.
        assert (value, source) == (1800, "file")
        assert "ak7375" in caplog.text and "dw9807" in caplog.text
        assert "re-run the focus calibration" in caplog.text.lower()

    def test_a_matching_lens_says_nothing(self, tmp_path, caplog) -> None:
        import logging

        from app.config import CameraSettings

        path = self._saved(tmp_path, "dw9807 10-000c", value=480)
        with caplog.at_level(logging.WARNING):
            value, _ = focus_module.resolve_focus_value(
                CameraSettings(lens_driver="dw9807"), path
            )

        assert value == 480
        assert "re-run" not in caplog.text.lower()

    def test_a_file_predating_the_field_is_not_flagged(self, tmp_path, caplog) -> None:
        """An empty ``lens_name`` is unknown, not mismatched, and warning on it
        would fire on every rig with an older calibration file."""
        import logging

        from app.config import CameraSettings

        path = self._saved(tmp_path, "")
        with caplog.at_level(logging.WARNING):
            focus_module.resolve_focus_value(CameraSettings(lens_driver="dw9807"), path)

        assert "re-run" not in caplog.text.lower()
