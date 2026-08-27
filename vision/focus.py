"""Lens focus, driven straight at the VCM subdev over V4L2.

Not through libcamera, and that is the whole point of this module.

The Arducam 16MP (IMX519) has an ak7375 voice-coil focus motor. On a Pi 5 with
the stock Raspberry Pi libcamera, the tuning file that gets loaded --
``/usr/share/libcamera/ipa/rpi/pisp/imx519.json`` -- has no ``rpi.af`` block, so
the IPA has no autofocus algorithm bound. Setting ``AfMode`` or ``LensPosition``
through picamera2 then does this::

    WARN IPARPI ipa_base.cpp:812  Could not set AF_MODE - no AF algorithm
    WARN IPARPI ipa_base.cpp:1424 Could not set LENS_POSITION - no AF algorithm

Note what that is: a warning *inside libcamera*, on its own log stream. The
Python call does not raise. `set_controls` returns normally, the control is
dropped on the floor, and every layer above reports success while the lens sits
wherever it powered up. A `try/except` around it catches nothing, which is
exactly how this went unnoticed.

The motor itself is fine and is bound by the kernel independently of libcamera's
opinion about it::

    /dev/v4l-subdev3 -> ak7375 10-000c
    focus_absolute: min=0 max=4095 step=1

That range is the ak7375's, not a universal one, and nothing here may assume it.
A dw9807 (Camera Module 3 / IMX708) on the same rig reports::

    /dev/v4l-subdev3 -> dw9807 10-000c
    focus_absolute: min=0 max=1023 step=1 default=480

Every bound, stride and margin in this module is therefore derived from
:func:`query_focus_range` rather than written down. Swapping the camera used to
change the meaning of numbers that had been chosen against the old lens -- the
back-off margin went from 1.5% of travel to 6.3% of it, and the sweep's stride
from 33 stops to 9 -- with nothing reporting that anything had changed.

**A sensor whose tuning file does bind an AF algorithm is a different case.**
The reasoning above says libcamera will not touch the lens, and on the IMX519 it
cannot. Where an ``rpi.af`` block *is* present, libcamera's AF owns this same
VCM, and while the camera is streaming it will drive it -- overwriting whatever
this module wrote, between the write and the readback. That presents as a
readback mismatch at *every* position while a manual ``v4l2-ctl`` write with
nothing streaming sticks perfectly. See :meth:`vision.camera.Camera._apply_focus`,
which puts AF in manual mode first for exactly this reason.

So this module talks to that subdev with ``VIDIOC_S_CTRL``, and then reads the
value back with ``VIDIOC_G_CTRL``. The readback is not belt-and-braces: it is
the only honest confirmation available, because the failure mode being designed
against -- a half-seated ribbon -- presents as a device node that opens and
accepts writes that do not stick.

The subdev index is resolved by name at startup, never hardcoded. ``v4l-subdev3``
today is ``v4l-subdev2`` after a reboot that enumerates devices in a different
order, and a hardcoded path would silently drive the wrong device -- or, worse,
succeed against something that is not a lens.

``ctypes``/``fcntl`` rather than shelling out to ``v4l2-ctl``: no dependency on
v4l2-utils being installed, no output parsing, and a real ``OSError`` with an
errno when something is wrong.

Backlash
--------
A voice coil has hysteresis: driving to 1400 from below leaves the lens in a
slightly different place than arriving at 1400 from above. The spring and the
coil do not perfectly cancel, and the difference is small but real -- enough to
matter at the depth of field this rig runs at.

So every move in this module **approaches its target from below**, and
:func:`approach_focus` is the only function that should be used to get there.
Ascending is chosen rather than descending because it is the direction that
comes for free on a cold boot: the VCM powers up at 0, so applying a saved value
at startup is already an upward move. A sweep that ascended and a startup that
descended would land in different places, and the symptom -- "calibration said
1400 but it looks softer at boot" -- points nowhere near the cause.
"""

from __future__ import annotations

import errno
import logging
import struct
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "APPROACH_DIRECTION",
    "BACKLASH_FRACTION",
    "BACKLASH_MIN_COUNTS",
    "BACKLASH_SETTLE_SECONDS",
    "FocusCalibration",
    "FocusError",
    "FocusRange",
    "FocusStatus",
    "LensDevice",
    "TargetPeak",
    "apply_focus",
    "approach_focus",
    "backlash_margin",
    "find_lens_subdev",
    "load_focus_calibration",
    "query_focus_range",
    "read_focus",
    "resolve_focus_value",
    "save_focus_calibration",
    "write_focus",
]

#: Where the kernel exposes V4L2 devices and their driver names.
SYSFS_V4L = Path("/sys/class/video4linux")

#: The focus motor's driver name, as it appears in sysfs. A fragment rather than
#: the full ``ak7375 10-000c`` because the I2C bus and address in the suffix vary
#: with how the camera is wired.
DEFAULT_LENS_NAME = "ak7375"

#: How far below a target to back off before approaching it, as a fraction of
#: the control's full span. Only used when the lens is already above the target
#: -- see :func:`approach_focus`.
#:
#: A fraction rather than a count, because backlash is a property of the
#: mechanism and scales with the range the driver uses to express it. This was
#: a flat 64 counts, chosen on an ak7375 whose range is 0-4095 -- 1.5% of
#: travel. The same 64 on a dw9807 (0-1023) is 6.3%, four times the intended
#: back-off, which wastes travel near the bottom of the range and makes the
#: clamp at the minimum bite far sooner than intended. 1/64 reproduces the
#: original 64 counts exactly on the ak7375.
BACKLASH_FRACTION = 1.0 / 64.0

#: Floor for the above. Below a handful of counts a back-off is smaller than the
#: motor's own settling error and stops meaning anything.
BACKLASH_MIN_COUNTS = 8

#: Seconds to wait after the backoff write before driving to the target.
#:
#: Not optional, and not a politeness. ``FOCUS_ABSOLUTE`` returns as soon as the
#: kernel has latched the value over I2C, which is long before the lens has
#: physically gone anywhere -- a VCM takes tens of milliseconds to travel and
#: settle. Issue the two writes back to back and the coil simply retargets in
#: flight: the lens never reaches the backoff point, so it still arrives at the
#: target from above, which is the entire failure the backoff exists to prevent.
#:
#: 0.1 s is several times the ak7375's travel time over a 64-count move and a
#: fraction of the 0.35 s the sweep already spends settling before each capture,
#: so it costs nothing anyone will notice on the one move per calibration that
#: needs it.
BACKLASH_SETTLE_SECONDS = 0.1

#: The direction every move arrives from. Recorded in the calibration file so a
#: future change here is visible in old data rather than silently invalidating
#: it.
APPROACH_DIRECTION = "ascending"

#: ``V4L2_CID_CAMERA_CLASS_BASE + 10``. See ``linux/v4l2-controls.h``.
V4L2_CID_FOCUS_ABSOLUTE = 0x009A0900 + 10

#: ``struct v4l2_control { __u32 id; __s32 value; }``. ``=`` for native order and
#: standard sizes with no alignment padding -- both fields are 4-byte, so the
#: struct is exactly 8 bytes and matches the kernel's layout.
_CTRL_FORMAT = "=Ii"
_CTRL_SIZE = struct.calcsize(_CTRL_FORMAT)

#: ``struct v4l2_queryctrl``: id, type, name[32], min, max, step, default, flags,
#: reserved[2] -- 68 bytes.
_QUERYCTRL_FORMAT = "=II32siiiiI2I"
_QUERYCTRL_SIZE = struct.calcsize(_QUERYCTRL_FORMAT)


def _iowr(type_char: str, number: int, size: int) -> int:
    """Encode an ``_IOWR`` ioctl request number, the way ``linux/ioctl.h`` does.

    Computed rather than pasted as a magic hex constant so the derivation is
    visible and testable: the direction bits, size and type all have to be
    right, and a wrong request number fails as ``EINVAL`` from a device that is
    otherwise working perfectly -- an unpleasant thing to debug from a hex
    literal.
    """
    read_write = 3  # _IOC_READ | _IOC_WRITE
    return (read_write << 30) | (size << 16) | (ord(type_char) << 8) | number


VIDIOC_G_CTRL = _iowr("V", 27, _CTRL_SIZE)
VIDIOC_S_CTRL = _iowr("V", 28, _CTRL_SIZE)
VIDIOC_QUERYCTRL = _iowr("V", 36, _QUERYCTRL_SIZE)


class FocusError(RuntimeError):
    """Raised when the focus motor cannot be found or driven."""


@dataclass(frozen=True, slots=True)
class LensDevice:
    """A focus-motor subdev, resolved by driver name."""

    path: Path
    #: The full name from sysfs, e.g. ``ak7375 10-000c``. Logged verbatim,
    #: because the I2C address in it is what tells you which camera port it is.
    name: str


@dataclass(frozen=True, slots=True)
class FocusRange:
    """The control's advertised limits, read from the driver."""

    minimum: int
    maximum: int
    step: int
    default: int

    @property
    def span(self) -> int:
        """Counts from end to end. The scale everything else is relative to."""
        return self.maximum - self.minimum

    def contains(self, value: int) -> bool:
        """Whether the driver would accept ``value`` without clamping it.

        Kept separate from :meth:`clamp` so a caller can tell *that* a value was
        out of range, not merely receive a different number back. The two are
        the same test, and conflating them is how an out-of-range request became
        indistinguishable from a lens that would not move.
        """
        return self.minimum <= int(value) <= self.maximum

    def clamp(self, value: int) -> int:
        return max(self.minimum, min(self.maximum, int(value)))

    def snap(self, value: int) -> int:
        """Clamp, then align to the driver's step grid.

        Both lenses seen so far report ``step=1``, which makes the alignment a
        no-op -- but a driver advertising a coarser step rounds whatever it is
        given, and the readback then legitimately differs from the request. That
        reads exactly like a motor that will not track, so it is removed at
        source rather than diagnosed later.
        """
        clamped = self.clamp(value)
        if self.step <= 1:
            return clamped
        offset = clamped - self.minimum
        aligned = self.minimum + (offset // self.step) * self.step
        return min(self.maximum, aligned)


@dataclass(frozen=True, slots=True)
class FocusStatus:
    """The outcome of trying to set focus. Surfaced through ``/api/status``.

    ``ok`` means the value was written *and read back unchanged*. Anything else
    is a real fault, and the distinction between ``available=False`` (no motor
    found) and ``ok=False`` (motor found, would not take the value) is the one
    that matters when diagnosing: the first is a driver or wiring problem, the
    second is a lens that is not responding to a bus that otherwise works.
    """

    available: bool = False
    device: str | None = None
    lens_name: str | None = None
    requested: int | None = None
    actual: int | None = None
    ok: bool = False
    detail: str = "focus control not attempted"
    #: Where the value came from: ``file`` (a calibration run), ``config`` (a
    #: hand-set override) or ``none`` (never calibrated). Surfaced because an
    #: uncalibrated rig is not broken -- it has simply never been told where to
    #: focus -- and those want different words on the panel.
    source: str = "none"

    @property
    def calibrated(self) -> bool:
        """Whether a focus value has ever been established for this rig."""
        return self.source != "none"

    @property
    def mismatch(self) -> bool:
        """Whether the lens accepted the write but reports a different value."""
        return (
            self.available
            and self.requested is not None
            and self.actual is not None
            and self.requested != self.actual
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "device": self.device,
            "lens_name": self.lens_name,
            "requested": self.requested,
            "actual": self.actual,
            "ok": self.ok,
            "detail": self.detail,
            "source": self.source,
            "calibrated": self.calibrated,
        }


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------


def find_lens_subdev(
    name_fragment: str = DEFAULT_LENS_NAME,
    sysfs_root: Path | None = None,
    dev_root: Path | None = None,
) -> LensDevice | None:
    """Locate the focus-motor subdev by driver name.

    Walks ``/sys/class/video4linux/*/name`` and matches on the driver string, so
    the answer follows the hardware rather than an enumeration order that is not
    stable across reboots.

    ``sysfs_root`` and ``dev_root`` are injectable so the walk can be tested
    against a fake tree on a machine with no V4L2 at all.

    Returns:
        The device, or ``None`` when no matching subdev exists -- which is the
        normal case on a dev box and must not raise.
    """
    sysfs_root = SYSFS_V4L if sysfs_root is None else sysfs_root
    dev_root = Path("/dev") if dev_root is None else dev_root

    if not sysfs_root.is_dir():
        return None

    fragment = name_fragment.lower()
    # Sorted so that a machine with two identical lenses picks the same one
    # every boot rather than whichever the filesystem happened to yield first.
    for entry in sorted(sysfs_root.iterdir()):
        name_file = entry / "name"
        try:
            name = name_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if fragment not in name.lower():
            continue
        node = dev_root / entry.name
        return LensDevice(path=node, name=name)
    return None


def list_v4l_devices(sysfs_root: Path | None = None) -> list[tuple[str, str]]:
    """Every V4L2 node and its driver name, for diagnostics.

    What you want in the log when :func:`find_lens_subdev` comes back empty: the
    question is then "what *is* there", and the answer distinguishes a camera
    that did not probe at all from one that probed without its focus motor.
    """
    sysfs_root = SYSFS_V4L if sysfs_root is None else sysfs_root
    if not sysfs_root.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for entry in sorted(sysfs_root.iterdir()):
        try:
            found.append((entry.name, (entry / "name").read_text(encoding="utf-8").strip()))
        except OSError:
            continue
    return found


# ---------------------------------------------------------------------------
# Control access
# ---------------------------------------------------------------------------


def _ioctl(fd: int, request: int, payload: bytes) -> bytes:
    """One ioctl round-trip, returning the kernel's written-back buffer."""
    import fcntl

    return fcntl.ioctl(fd, request, payload, True)


def query_focus_range(device: Path, opener=None) -> FocusRange:
    """Read the driver's advertised focus limits.

    Queried rather than assumed. 0-4095 is what the ak7375 reports here, but the
    range is a property of the motor and its driver, and clamping a configured
    value against a hardcoded span would silently mis-drive a different lens.
    """
    opener = _open_device if opener is None else opener
    payload = struct.pack(
        _QUERYCTRL_FORMAT, V4L2_CID_FOCUS_ABSOLUTE, 0, b"", 0, 0, 0, 0, 0, 0, 0
    )
    with opener(device) as fd:
        try:
            result = _ioctl(fd, VIDIOC_QUERYCTRL, payload)
        except OSError as exc:
            raise FocusError(f"{device}: cannot query focus_absolute ({exc})") from exc

    _, _, _, minimum, maximum, step, default, *_ = struct.unpack(_QUERYCTRL_FORMAT, result)
    return FocusRange(minimum=minimum, maximum=maximum, step=max(1, step), default=default)


def read_focus(device: Path, opener=None) -> int:
    """Current ``focus_absolute`` from the lens."""
    opener = _open_device if opener is None else opener
    payload = struct.pack(_CTRL_FORMAT, V4L2_CID_FOCUS_ABSOLUTE, 0)
    with opener(device) as fd:
        try:
            result = _ioctl(fd, VIDIOC_G_CTRL, payload)
        except OSError as exc:
            raise FocusError(f"{device}: cannot read focus_absolute ({exc})") from exc
    _, value = struct.unpack(_CTRL_FORMAT, result)
    return int(value)


def write_focus(device: Path, value: int, opener=None) -> None:
    """Set ``focus_absolute``. Does not verify -- see :func:`apply_focus`."""
    opener = _open_device if opener is None else opener
    payload = struct.pack(_CTRL_FORMAT, V4L2_CID_FOCUS_ABSOLUTE, int(value))
    with opener(device) as fd:
        try:
            _ioctl(fd, VIDIOC_S_CTRL, payload)
        except OSError as exc:
            hint = ""
            if exc.errno == errno.EACCES:
                hint = " -- add the user to the 'video' group"
            elif exc.errno == errno.EINVAL:
                hint = " -- the value may be outside the driver's range"
            raise FocusError(f"{device}: cannot set focus_absolute={value} ({exc}){hint}") from exc


def backlash_margin(focus_range: FocusRange) -> int:
    """How far to undershoot before approaching a target from below.

    Derived from the range the driver advertises rather than fixed, because the
    count is only meaningful relative to the span it is expressed in. See
    :data:`BACKLASH_FRACTION`.
    """
    return max(BACKLASH_MIN_COUNTS, round(focus_range.span * BACKLASH_FRACTION))


def approach_focus(
    device: Path,
    target: int,
    focus_range: FocusRange,
    opener=None,
    settle_seconds: float = BACKLASH_SETTLE_SECONDS,
) -> None:
    """Drive the lens to ``target``, always arriving from below.

    The only function that should move this lens. See the module docstring for
    why the direction is fixed: a voice coil lands in a slightly different place
    depending on which way it travelled, and a sweep that ascends paired with a
    startup that descends produces a rig that is measurably softer at boot than
    it was at calibration, with nothing in any log to explain it.

    Arriving from below costs one extra move only when the lens is currently
    above the target -- on a cold boot it is at 0 and this is a plain move.

    **The backoff has to be given time to happen.** The write returns once the
    kernel has latched the value, not once the lens has moved, so issuing the
    backoff and the target back to back lets the coil retarget in flight: the
    lens never reaches the backoff point and still arrives from above, which
    silently defeats the whole function. ``settle_seconds`` is the wait between
    them, and it is a parameter only so a test can vary it -- callers should
    leave it alone.

    Args:
        settle_seconds: Wait after the backoff write. Defaults to
            :data:`BACKLASH_SETTLE_SECONDS`. Ignored when no backoff is needed,
            which is the common case and stays instant.
    """
    opener = _open_device if opener is None else opener
    current = read_focus(device, opener=opener)
    if current > target:
        # Undershoot first, so the final move is upward like every other one.
        backoff = max(focus_range.minimum, target - backlash_margin(focus_range))
        logger.debug("focus: backing off to %d before approaching %d from below", backoff, target)
        write_focus(device, backoff, opener=opener)
        # Let the lens actually get there before retargeting. Without this the
        # backoff is a value the kernel saw and the lens never visited.
        if settle_seconds > 0.0:
            time.sleep(settle_seconds)
    write_focus(device, target, opener=opener)


class _open_device:
    """Context manager yielding a raw fd for a V4L2 node.

    A class rather than ``contextlib.contextmanager`` so it can be substituted
    wholesale in tests with something backed by a fake ioctl.
    """

    def __init__(self, device: Path) -> None:
        self.device = device
        self._fd: int | None = None

    def __enter__(self) -> int:
        import os

        try:
            # O_RDWR: querying and reading are reads, but the same handle has to
            # take S_CTRL, and reopening per operation would triple the syscalls
            # in the sweep's inner loop.
            self._fd = os.open(str(self.device), os.O_RDWR)
        except OSError as exc:
            raise FocusError(f"cannot open {self.device} ({exc})") from exc
        return self._fd

    def __exit__(self, *exc_info: object) -> None:
        import os

        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# ---------------------------------------------------------------------------
# The operation the rest of the system uses
# ---------------------------------------------------------------------------


def apply_focus(
    value: int,
    device: Path | None = None,
    name_fragment: str = DEFAULT_LENS_NAME,
    opener=None,
    finder=None,
    source: str = "none",
) -> FocusStatus:
    """Set the lens position and confirm it took. Never raises.

    The confirmation is the reason this is not just :func:`write_focus`. Writing
    a control that the driver accepts and the motor ignores is indistinguishable
    from success at the syscall level, and that is precisely how a half-seated
    ribbon presents -- the I2C device answers, the write returns 0, and the lens
    does not move. Reading the value back is the cheapest thing that can tell
    those apart.

    Returns a :class:`FocusStatus` rather than raising, because a camera that
    will not focus should still run: an out-of-focus table is a degraded system,
    a refusal to start is a dead one.
    """
    finder = find_lens_subdev if finder is None else finder

    if device is None:
        lens = finder(name_fragment)
        if lens is None:
            present = ", ".join(f"{node} ({name})" for node, name in list_v4l_devices()) or "none"
            detail = (
                f"no V4L2 subdev matching {name_fragment!r}; focus cannot be set. "
                f"Present: {present}"
            )
            logger.error("focus: %s", detail)
            return FocusStatus(available=False, requested=value, detail=detail, source=source)
    else:
        lens = LensDevice(path=device, name=str(device))

    try:
        focus_range = query_focus_range(lens.path, opener=opener)
    except FocusError as exc:
        logger.error("focus: %s", exc)
        return FocusStatus(
            available=False, device=str(lens.path), lens_name=lens.name,
            requested=value, detail=str(exc), source=source,
        )

    target = focus_range.snap(value)
    if target != value:
        # Two different things, and they point at different fixes: a value the
        # lens cannot reach is a stale configured number, a value off the step
        # grid is a driver that quantises. Reporting them as one "adjusted"
        # message loses the part that says what to change.
        reason = (
            f"outside the lens range {focus_range.minimum}-{focus_range.maximum}"
            if not focus_range.contains(value)
            else f"not on the driver's {focus_range.step}-count step grid"
        )
        logger.warning("focus: requested %d is %s; using %d", value, reason, target)

    try:
        approach_focus(lens.path, target, focus_range, opener=opener)
        actual = read_focus(lens.path, opener=opener)
    except FocusError as exc:
        logger.error("focus: %s", exc)
        return FocusStatus(
            available=True, device=str(lens.path), lens_name=lens.name,
            requested=target, detail=str(exc), source=source,
        )

    if actual != target:
        # A real error, not a warning. The system will run and every frame will
        # be soft, and there is no other symptom that points here.
        detail = (
            f"lens {lens.name} accepted focus_absolute={target} but reads back {actual}. "
            "The lens is not responding to the focus motor -- check the camera ribbon "
            "is fully seated at both ends."
        )
        logger.error("focus: %s", detail)
        return FocusStatus(
            available=True, device=str(lens.path), lens_name=lens.name,
            requested=target, actual=actual, ok=False, detail=detail, source=source,
        )

    detail = f"focus_absolute={actual} on {lens.name} (range {focus_range.minimum}-{focus_range.maximum})"
    logger.info("focus: %s", detail)
    return FocusStatus(
        available=True, device=str(lens.path), lens_name=lens.name,
        requested=target, actual=actual, ok=True, detail=detail, source=source,
    )


# ---------------------------------------------------------------------------
# Persisted calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetPeak:
    """Where one projected target was sharpest.

    Five of these -- centre plus four inset corners -- are what turn a single
    focus number into a statement about the *mount*. One peak says where to put
    the lens; five say whether the camera is square to the cloth.
    """

    name: str
    #: Target centre in camera px, from blob detection rather than from any
    #: homography -- see :mod:`vision.focus_calibration`.
    center_px: tuple[float, float]
    peak_focus: int
    peak_sharpness: float
    #: Peak height over the curve's median. Below ~1.5 the curve is flat enough
    #: that its maximum is noise, which is why it is kept rather than discarded
    #: once the decision is made.
    prominence: float


@dataclass(frozen=True, slots=True)
class FocusCalibration:
    """What a focus calibration run established, persisted between boots.

    A file rather than ``config.yaml``, and the reason is ownership: this is
    written by software, and ``config.yaml`` is written by a person and full of
    their comments. Machine-written values in a hand-edited file get clobbered
    in one direction or the other.

    It also makes "never calibrated" representable. With the value living in
    config it always had a default, so first run and calibrated-to-that-value
    were indistinguishable -- and "should this rig be swept?" has no answer
    without that distinction.
    """

    focus_absolute: int
    #: Peak sharpness on the *projected targets*. Comparable only against
    #: another measurement taken the same way, with the targets up.
    peak_sharpness: float = 0.0
    #: Sharpness of the bare table interior at ``focus_absolute``, targets off.
    #:
    #: The runtime health check's only usable reference. Checking a bare-felt
    #: measurement against ``peak_sharpness`` would compare numbers an order of
    #: magnitude apart and would fire on every boot.
    bare_table_sharpness: float = 0.0
    per_target: tuple[TargetPeak, ...] = ()
    #: Spread of the per-target peaks, in focus counts. Large means the sensor
    #: plane is not parallel to the cloth.
    tilt_spread: int = 0
    tilt_note: str = ""
    #: Which way the lens was travelling when it arrived. Recorded so a change
    #: to :data:`APPROACH_DIRECTION` invalidates old files visibly, instead of
    #: silently shifting every stored value by the backlash.
    approach: str = APPROACH_DIRECTION
    #: Capture geometry at calibration time. Sharpness is not comparable across
    #: resolutions, so the health check has to know whether it still matches.
    camera_resolution: str = ""
    lens_name: str = ""
    created_at: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "focus_absolute": self.focus_absolute,
            "peak_sharpness": round(self.peak_sharpness, 3),
            "bare_table_sharpness": round(self.bare_table_sharpness, 3),
            "per_target": [
                {
                    "name": t.name,
                    "center_px": list(t.center_px),
                    "peak_focus": t.peak_focus,
                    "peak_sharpness": round(t.peak_sharpness, 3),
                    "prominence": round(t.prominence, 3),
                }
                for t in self.per_target
            ],
            "tilt_spread": self.tilt_spread,
            "tilt_note": self.tilt_note,
            "approach": self.approach,
            "camera_resolution": self.camera_resolution,
            "lens_name": self.lens_name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FocusCalibration:
        targets = tuple(
            TargetPeak(
                name=str(entry["name"]),
                center_px=(float(entry["center_px"][0]), float(entry["center_px"][1])),
                peak_focus=int(entry["peak_focus"]),
                peak_sharpness=float(entry.get("peak_sharpness", 0.0)),
                prominence=float(entry.get("prominence", 0.0)),
            )
            for entry in data.get("per_target", [])
        )
        return cls(
            focus_absolute=int(data["focus_absolute"]),
            peak_sharpness=float(data.get("peak_sharpness", 0.0)),
            bare_table_sharpness=float(data.get("bare_table_sharpness", 0.0)),
            per_target=targets,
            tilt_spread=int(data.get("tilt_spread", 0)),
            tilt_note=str(data.get("tilt_note", "")),
            approach=str(data.get("approach", APPROACH_DIRECTION)),
            camera_resolution=str(data.get("camera_resolution", "")),
            lens_name=str(data.get("lens_name", "")),
            created_at=str(data.get("created_at", "")),
        )


def focus_calibration_path() -> Path:
    """Where the focus calibration lives. Beside the projector's."""
    from app.config import CALIBRATION_DIR

    return CALIBRATION_DIR / "focus.json"


def load_focus_calibration(path: Path | None = None) -> FocusCalibration | None:
    """Read the saved focus calibration, or ``None`` if there is none.

    A corrupt file is logged and treated as absent rather than raised. Failing
    to boot because a calibration file has a typo in it would be far worse than
    booting uncalibrated -- and booting uncalibrated is already a state the
    panel reports, so nothing is hidden by degrading this way.
    """
    import json

    source = path or focus_calibration_path()
    if not source.is_file():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        calibration = FocusCalibration.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error(
            "focus calibration %s is unreadable (%s); treating the rig as uncalibrated",
            source,
            exc,
        )
        return None

    if calibration.approach != APPROACH_DIRECTION:
        # Not fatal, but the stored value was reached travelling the other way,
        # so the backlash offset is baked into it.
        logger.warning(
            "focus calibration was taken approaching %s but this build approaches %s; "
            "the saved value may be off by the lens backlash. Re-run focus calibration.",
            calibration.approach,
            APPROACH_DIRECTION,
        )
    return calibration


def save_focus_calibration(calibration: FocusCalibration, path: Path | None = None) -> Path:
    """Write the focus calibration, creating the directory if needed."""
    import json

    target = path or focus_calibration_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(calibration.as_dict(), indent=2), encoding="utf-8")
    logger.info(
        "saved focus calibration to %s (focus_absolute=%d, %d target(s))",
        target,
        calibration.focus_absolute,
        len(calibration.per_target),
    )
    return target


def resolve_focus_value(settings, path: Path | None = None) -> tuple[int | None, str]:
    """Decide which focus value to apply, and say where it came from.

    Precedence, highest first:

    1. ``camera.focus_absolute`` in config -- a deliberate hand-set override.
       It wins because someone typing a number into a config file has made a
       more specific statement than a calibration run from last month did.
    2. ``data/calibration/focus.json`` -- the calibrated value.
    3. Nothing. Returns ``None``, and the caller reports the rig as
       uncalibrated rather than guessing.

    There is deliberately no default number. A guessed focus produces a rig that
    is soft for reasons nobody can see; "not calibrated" on the panel points
    straight at the fix.
    """
    override = getattr(settings, "focus_absolute", None)
    if override is not None:
        return int(override), "config"

    calibration = load_focus_calibration(path)
    if calibration is not None:
        # A stored value is in the *calibrated lens's* raw units, and those units
        # are not portable. 1800 means most of the way out on an ak7375 (0-4095)
        # and is off the end of a dw9807 (0-1023), where it clamps to 1023 and
        # produces a lens jammed at one extreme -- soft, with a panel that says
        # "calibrated" and a value that came from a file, so nothing points here.
        # Cannot be an error: the file may predate this field, and refusing to
        # focus at all is worse than focusing on a suspect number.
        configured = str(getattr(settings, "lens_driver", "") or "")
        stored = calibration.lens_name or ""
        if configured and stored and configured.lower() not in stored.lower():
            logger.warning(
                "focus: %s was calibrated on lens %r but this rig is configured for %r. "
                "focus_absolute=%d is in the old lens's units and may be meaningless "
                "here -- re-run the focus calibration.",
                path or "the focus calibration",
                stored, configured, calibration.focus_absolute,
            )
        return calibration.focus_absolute, "file"
    return None, "none"
