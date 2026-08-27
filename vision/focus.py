"""Lens focus. Two control paths, chosen by what the sensor actually offers.

Which one a rig uses is not configurable, because it is not a preference -- it
is a property of the sensor's libcamera tuning file, and getting it wrong
produces a lens that cannot be driven at all.

Path 1: V4L2 counts (no AF algorithm bound)
-------------------------------------------
The Arducam 16MP (IMX519) has an ak7375 voice-coil motor. On a Pi 5 with the
stock Raspberry Pi libcamera, the tuning file that gets loaded --
``/usr/share/libcamera/ipa/rpi/pisp/imx519.json`` -- has no ``rpi.af`` block, so
the IPA has no autofocus algorithm bound. Setting ``AfMode`` or ``LensPosition``
through picamera2 then does this::

    WARN IPARPI ipa_base.cpp:812  Could not set AF_MODE - no AF algorithm
    WARN IPARPI ipa_base.cpp:1424 Could not set LENS_POSITION - no AF algorithm

Note what that is: a warning *inside libcamera*, on its own log stream. The
Python call does not raise. ``set_controls`` returns normally, the control is
dropped on the floor, and every layer above reports success while the lens sits
wherever it powered up. A ``try/except`` around it catches nothing, which is
exactly how this went unnoticed.

The motor itself is fine and is bound by the kernel independently of libcamera's
opinion about it::

    /dev/v4l-subdev3 -> ak7375 10-000c
    focus_absolute: min=0 max=4095 step=1

So for this sensor the answer is to bypass libcamera entirely and write
``focus_absolute`` at the subdev. That is :class:`V4L2Focus`.

Path 2: libcamera dioptres (AF algorithm bound)
-----------------------------------------------
**The conclusion above does not generalise, and assuming it did cost a long
diagnosis.** The IMX708 (Camera Module 3, dw9807 motor) *does* have ``rpi.af`` in
its tuning file. libcamera therefore owns the same VCM, and it wins::

    service running:  v4l2-ctl --set-ctrl focus_absolute=800  ->  reads back 477
    service stopped:  v4l2-ctl --set-ctrl focus_absolute=800  ->  reads back 800

Every write accepted, the lens never moving, and 477 is not the driver's 480
default -- it is wherever AF parked the lens at stream start. The V4L2 write does
reach the driver; libcamera's AF simply moves the lens back before anything can
observe it. So each obvious test exonerates the cable, and the sweep reported
"the motor is not tracking, check the ribbon" at all 34 positions.

For such a sensor the lens has to be driven *through* libcamera:
``AfMode=Manual`` to stop AF, then ``LensPosition``. That is
:class:`LibcameraFocus`.

Units, and why they are never converted
---------------------------------------
The two paths do not share a number system. ``focus_absolute`` is an integer in
the driver's own counts; ``LensPosition`` is a float in dioptres (reciprocal
metres, 0.0 being infinity). The mapping between them is a property of the
individual lens and is not published, so there is no conversion to make -- and
the ranges overlap enough that a misread value is *plausible* rather than
obviously broken. 477 counts read as dioptres asks for focus 2 mm from the lens;
1.8 dioptres read as counts is very nearly infinity. Both are numbers the system
would accept, apply, and report as calibrated.

So the unit travels with every stored or compared value --
:attr:`FocusRange.kind`, :attr:`FocusStatus.kind`,
:attr:`FocusCalibration.kind` -- and a mismatch is **refused**, never
reinterpreted. An existing ``focus.json`` with no recorded unit is read as
counts, which is what it must be: it could only have been written by the V4L2
path.

Ordering
--------
On any AF-bound sensor libcamera writes the lens once at stream start. Both
``AfMode=Manual`` and the position therefore have to be set **after**
``start()`` -- see :meth:`vision.camera.Picamera2Backend.start`. Setting them
before is silently undone a moment later while every log line reports success,
which is the shape of the original bug.

Ranges are queried, never assumed
---------------------------------
Every bound, stride and margin here is derived from the control's advertised
range. The ak7375 is 0-4095, the dw9807 0-1023, a dioptre range 0-32; a number
chosen against one is meaningless against the others. Swapping the camera used
to silently change what the constants meant -- the back-off margin went from
1.5% of travel to 6.3%, and the sweep stride from 33 stops to 9 -- with nothing
reporting that anything had changed.

Readback is mandatory on both paths
-----------------------------------
The V4L2 path reads back with ``VIDIOC_G_CTRL``; the libcamera path reads
``LensPosition`` out of capture metadata. Neither is belt-and-braces: a write
that is accepted and ignored is indistinguishable from success at the call site,
and that is how *both* failure modes present -- a half-seated ribbon on one path,
autofocus fighting you on the other.

The comparison differs, though, and that difference is in
:attr:`FocusRange.tolerance`. Counts are compared exactly: an integer the driver
either latched or did not. Dioptres are compared within a tolerance, because
libcamera maps the request onto the VCM's own discrete steps and reports where it
actually went -- an exact comparison there would call every write a failure.

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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "APPROACH_DIRECTION",
    "FOCUS_COUNTS",
    "FOCUS_DIOPTRES",
    "BACKLASH_FRACTION",
    "BACKLASH_MIN_COUNTS",
    "BACKLASH_SETTLE_SECONDS",
    "FocusCalibration",
    "FocusController",
    "FocusError",
    "FocusRange",
    "FocusStatus",
    "LensDevice",
    "LibcameraFocus",
    "V4L2Focus",
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

#: The two ways a lens gets driven here, and they are not interchangeable.
#:
#: ``counts`` is V4L2 ``focus_absolute``: an integer in the driver's own units,
#: range advertised by the driver, meaning nothing outside it. ``dioptres`` is
#: libcamera ``LensPosition``: a float, reciprocal metres, where 0.0 is infinity
#: and 2.0 is half a metre.
#:
#: A number in one system is not a number in the other, and there is no
#: conversion -- the mapping from counts to dioptres is a property of the
#: specific lens and is not published. So the unit travels with every value
#: that gets stored or compared, and a mismatch is refused rather than
#: reinterpreted. 477 counts read as dioptres would be a request to focus 2 mm
#: from the lens; 1.8 dioptres read as counts is very nearly infinity. Both are
#: plausible numbers, and both would present as a lens that will not focus.
FOCUS_COUNTS = "counts"
FOCUS_DIOPTRES = "dioptres"

#: Readback tolerance for dioptres, as a fraction of the control's span, with a
#: floor. Unlike ``focus_absolute``, ``LensPosition`` is not what comes back:
#: libcamera quantises the request onto the VCM's own steps, so an exact
#: comparison would report every single write as a failure. This is loose enough
#: to absorb that and tight enough that a lens which has not moved still fails.
DIOPTRE_TOLERANCE_FRACTION = 0.02
DIOPTRE_TOLERANCE_FLOOR = 0.05

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
    """The control's advertised limits, read from the driver, *with its unit*.

    ``kind`` is load-bearing rather than decorative. It decides whether values
    are integers on a step grid or floats on a continuum, whether a readback is
    compared exactly or within a tolerance, and whether a stored calibration may
    be loaded at all. Defaulting it to :data:`FOCUS_COUNTS` keeps every existing
    V4L2 construction -- and every calibration file already on disk -- meaning
    exactly what it meant before.
    """

    minimum: float
    maximum: float
    step: float
    default: float
    kind: str = FOCUS_COUNTS

    @property
    def continuous(self) -> bool:
        """Whether values are floats on a continuum rather than integer counts."""
        return self.kind == FOCUS_DIOPTRES

    @property
    def span(self) -> float:
        """End to end. The scale every relative quantity here is a fraction of."""
        return self.maximum - self.minimum

    @property
    def tolerance(self) -> float:
        """How far a readback may sit from the request and still count as arrived.

        Zero for counts: ``focus_absolute`` is an integer the driver either
        latched or did not, so anything other than equality is a real fault and
        a tolerance would only hide one.

        Non-zero for dioptres, because there the readback is *expected* to
        differ -- libcamera maps the requested position onto the VCM's discrete
        steps and reports where it actually went. See
        :data:`DIOPTRE_TOLERANCE_FRACTION`.
        """
        if not self.continuous:
            return 0.0
        return max(DIOPTRE_TOLERANCE_FLOOR, abs(self.span) * DIOPTRE_TOLERANCE_FRACTION)

    def agrees(self, requested: float, actual: float) -> bool:
        """Whether a readback means the lens arrived where it was sent.

        The one place the exact-versus-tolerant comparison lives, so the two
        unit systems cannot end up being compared by two different rules in two
        different callers.
        """
        return abs(float(requested) - float(actual)) <= self.tolerance

    def format(self, value: float) -> str:
        """A value with its unit, for a log line or the panel.

        Counts are integers and dioptres want two decimals and a unit word --
        printing ``1.7999999999`` at somebody, or ``477.0``, is a small thing
        that makes a diagnostic read as untrustworthy.
        """
        if self.continuous:
            return f"{float(value):.2f} dioptres"
        return str(int(round(float(value))))

    def contains(self, value: float) -> bool:
        """Whether the driver would accept ``value`` without clamping it.

        Kept separate from :meth:`clamp` so a caller can tell *that* a value was
        out of range, not merely receive a different number back. The two are
        the same test, and conflating them is how an out-of-range request became
        indistinguishable from a lens that would not move.
        """
        return self.minimum <= float(value) <= self.maximum

    def clamp(self, value: float) -> float:
        clamped = max(self.minimum, min(self.maximum, float(value)))
        return clamped if self.continuous else int(round(clamped))

    def snap(self, value: float) -> float:
        """Clamp, then align to the driver's step grid.

        A continuous control has no grid to align to, so this is just a clamp --
        the quantisation libcamera applies is its own business and is absorbed by
        :attr:`tolerance` instead, since we cannot predict where it will land.

        For counts: both lenses seen so far report ``step=1``, which makes the
        alignment a no-op -- but a driver advertising a coarser step rounds
        whatever it is given, and the readback then legitimately differs from the
        request. That reads exactly like a motor that will not track, so it is
        removed at source rather than diagnosed later.
        """
        clamped = self.clamp(value)
        if self.continuous or self.step <= 1:
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
    requested: float | None = None
    actual: float | None = None
    #: Which control was driven: :data:`FOCUS_COUNTS` or :data:`FOCUS_DIOPTRES`.
    #: Reported because ``requested=1.8`` and ``requested=477`` are both valid
    #: and mean nothing without it.
    kind: str = FOCUS_COUNTS
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


# ---------------------------------------------------------------------------
# Focus controllers
# ---------------------------------------------------------------------------


class FocusController(ABC):
    """One way of driving this rig's lens, with its unit attached.

    Two implementations, chosen by what the sensor actually offers rather than by
    configuration -- see :meth:`vision.camera.Picamera2Backend.focus_controller`.

    * :class:`V4L2Focus` writes ``focus_absolute`` straight at the VCM subdev.
      Correct when libcamera has no AF algorithm bound and therefore never
      touches the lens.
    * :class:`LibcameraFocus` sets ``AfMode=Manual`` and ``LensPosition`` through
      picamera2. Necessary when libcamera *does* have AF bound, because then it
      owns the same motor and wins every argument: on an IMX708 the V4L2 write
      is accepted, the readback never budges from wherever AF parked the lens,
      and a manual ``v4l2-ctl`` write with the service stopped works perfectly --
      so every obvious test exonerates the cable.

    The interface is deliberately narrow: a range, a read, a write, and a name.
    Everything above it -- the backlash approach, the sweep, the readback
    diagnosis -- is unit-agnostic and works through this.
    """

    #: :data:`FOCUS_COUNTS` or :data:`FOCUS_DIOPTRES`.
    kind: str = FOCUS_COUNTS

    @abstractmethod
    def range(self) -> FocusRange:
        """The control's limits. Cached by implementations; querying costs I/O."""

    @abstractmethod
    def read(self) -> float:
        """Where the lens reports it is. Raises :class:`FocusError` on failure."""

    @abstractmethod
    def write(self, value: float) -> None:
        """Request a position. Does not verify -- see :func:`apply_focus`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identity, for logs and the panel."""

    def prepare(self) -> None:
        """Take exclusive control of the lens. Called once before any write.

        Deliberately not abstract: doing nothing is the correct *and complete*
        implementation for V4L2, where nothing else is driving the motor. For
        libcamera this is where autofocus gets switched off, and it has to happen
        after the stream starts -- see :meth:`LibcameraFocus.prepare`.
        """
        return None


class V4L2Focus(FocusController):
    """``focus_absolute`` on a VCM subdev, in the driver's raw counts."""

    kind = FOCUS_COUNTS

    def __init__(self, lens: LensDevice, opener=None) -> None:
        self.lens = lens
        self._opener = opener
        self._range: FocusRange | None = None

    @classmethod
    def find(cls, name_fragment: str = DEFAULT_LENS_NAME, opener=None) -> V4L2Focus | None:
        """Locate the motor by driver name, or ``None`` if it is not there.

        ``None`` rather than raising: no motor is the normal case on a dev box
        and on a fixed-focus lens, and neither is an error.
        """
        lens = find_lens_subdev(name_fragment)
        return None if lens is None else cls(lens, opener=opener)

    def range(self) -> FocusRange:
        if self._range is None:
            self._range = query_focus_range(self.lens.path, opener=self._opener)
        return self._range

    def read(self) -> float:
        return read_focus(self.lens.path, opener=self._opener)

    def write(self, value: float) -> None:
        write_focus(self.lens.path, int(round(value)), opener=self._opener)

    @property
    def name(self) -> str:
        return self.lens.name


class LibcameraFocus(FocusController):
    """``LensPosition`` through picamera2, in dioptres.

    The path for any sensor whose tuning file binds an AF algorithm. Note the
    inversion of this module's original premise: :mod:`vision.focus` exists
    because setting ``LensPosition`` on an IMX519 is silently dropped, and the
    conclusion drawn from that -- drive V4L2 instead -- is right for the IMX519
    and exactly wrong for an IMX708, where libcamera holds the lens against you.
    """

    kind = FOCUS_DIOPTRES

    def __init__(self, camera, settle_frames: int = 2) -> None:
        #: A live ``Picamera2``. Held, not owned -- the camera backend closes it.
        self._cam = camera
        #: Frames to let pass before trusting a readback. ``LensPosition`` in
        #: metadata describes the frame it came with, so a value read
        #: immediately after a write describes the lens *before* it moved -- the
        #: same one-frame lag that made the sweep measure the previous stop.
        self._settle_frames = settle_frames
        self._range: FocusRange | None = None
        self._prepared = False

    def range(self) -> FocusRange:
        if self._range is None:
            limits = (self._cam.camera_controls or {}).get("LensPosition")
            if not limits:
                raise FocusError(
                    "this camera reports no LensPosition control, so libcamera "
                    "cannot drive the lens"
                )
            minimum, maximum, default = limits[0], limits[1], limits[2]
            if default is None:
                # libcamera may report no default. Infinity is the honest
                # fallback: it is what the control means at 0.0 and it is where
                # a lens with no opinion should sit.
                default = minimum
            self._range = FocusRange(
                minimum=float(minimum),
                maximum=float(maximum),
                # No grid. Continuous controls are clamped, never aligned.
                step=0.0,
                default=float(default),
                kind=FOCUS_DIOPTRES,
            )
        return self._range

    def prepare(self) -> None:
        """Switch libcamera's autofocus off so it stops driving the lens.

        **Must be called after the stream has started.** libcamera writes the
        lens once at stream start on any AF-bound sensor, so a manual position
        set beforehand is overwritten a moment later while every log line
        reports success. That is the ordering bug this whole path exists to fix,
        and doing the AF switch here rather than at configure time is what keeps
        it fixed.
        """
        if self._prepared:
            return
        try:
            # 0 is AfModeEnum.Manual. The integer rather than the enum import so
            # this does not depend on a picamera2 version exposing it.
            self._cam.set_controls({"AfMode": 0})
        except (RuntimeError, KeyError, ValueError) as exc:
            raise FocusError(
                f"could not set AfMode=Manual ({exc}); libcamera will keep driving "
                "the lens and no requested position will stick"
            ) from exc
        self._prepared = True
        logger.info("focus: AfMode=Manual -- libcamera has released the lens")

    def read(self) -> float:
        """``LensPosition`` from capture metadata.

        Reads several frames and takes the last, because the value describes the
        frame it arrived with: the first metadata after a write still reports
        where the lens was before it moved.
        """
        value = None
        for _ in range(max(1, self._settle_frames)):
            try:
                metadata = self._cam.capture_metadata()
            except (RuntimeError, OSError) as exc:
                raise FocusError(f"could not read capture metadata ({exc})") from exc
            if metadata is not None and "LensPosition" in metadata:
                value = metadata["LensPosition"]
        if value is None:
            raise FocusError(
                "capture metadata carries no LensPosition, so the lens position "
                "cannot be confirmed"
            )
        return float(value)

    def write(self, value: float) -> None:
        self.prepare()
        try:
            self._cam.set_controls({"LensPosition": float(value)})
        except (RuntimeError, KeyError, ValueError) as exc:
            raise FocusError(f"could not set LensPosition={value:.3f} ({exc})") from exc

    @property
    def name(self) -> str:
        model = ""
        try:
            model = str((self._cam.camera_properties or {}).get("Model", ""))
        except (AttributeError, RuntimeError):
            pass
        return f"libcamera LensPosition{f' on {model}' if model else ''}"


def backlash_margin(focus_range: FocusRange) -> float:
    """How far to undershoot before approaching a target from below.

    Derived from the range the driver advertises rather than fixed, because a
    number of anything is only meaningful relative to the span it is expressed
    in. See :data:`BACKLASH_FRACTION`.

    The floor applies to counts only, and that exception is the point:
    :data:`BACKLASH_MIN_COUNTS` is 8 *counts*, a sensible floor on a 0-4095
    control and a nonsensical one on a 0-32 dioptre control, where it would
    demand a quarter of the lens's whole travel as a back-off. A continuous
    control needs no floor -- there is no integer quantisation for a small margin
    to be swallowed by.
    """
    margin = abs(focus_range.span) * BACKLASH_FRACTION
    if focus_range.continuous:
        return margin
    return max(BACKLASH_MIN_COUNTS, round(margin))


def approach_focus(
    controller: FocusController,
    target: float,
    settle_seconds: float = BACKLASH_SETTLE_SECONDS,
) -> None:
    """Drive the lens to ``target``, always arriving from below.

    The only function that should move this lens. See the module docstring for
    why the direction is fixed: a voice coil lands in a slightly different place
    depending on which way it travelled, and a sweep that ascends paired with a
    startup that descends produces a rig that is measurably softer at boot than
    it was at calibration, with nothing in any log to explain it.

    Unit-agnostic. Both control schemes drive the same kind of motor, so the
    backlash is just as real in dioptres as in counts, and the back-off is a
    fraction of whatever span the controller reports rather than a number of
    anything.

    Arriving from below costs one extra move only when the lens is currently
    above the target -- on a cold boot it is at one end and this is a plain move.

    **The backoff has to be given time to happen.** The write returns once the
    request has been latched, not once the lens has moved, so issuing the backoff
    and the target back to back lets the coil retarget in flight: the lens never
    reaches the backoff point and still arrives from above, which silently
    defeats the whole function.

    Args:
        controller: The lens. Its ``range()`` supplies the clamp and the span.
        target: Position in the controller's own units.
        settle_seconds: Wait after the backoff write. A parameter only so a test
            can vary it; callers should leave it alone.
    """
    focus_range = controller.range()
    current = controller.read()
    if current > target:
        # Undershoot first, so the final move is upward like every other one.
        backoff = max(focus_range.minimum, target - backlash_margin(focus_range))
        logger.debug(
            "focus: backing off to %s before approaching %s from below",
            focus_range.format(backoff), focus_range.format(target),
        )
        controller.write(backoff)
        # Let the lens actually get there before retargeting. Without this the
        # backoff is a value the driver saw and the lens never visited.
        if settle_seconds > 0.0:
            time.sleep(settle_seconds)
    controller.write(target)


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
    value: float,
    controller: FocusController | None = None,
    device: Path | None = None,
    name_fragment: str = DEFAULT_LENS_NAME,
    opener=None,
    source: str = "none",
) -> FocusStatus:
    """Set the lens position and confirm it took. Never raises.

    The confirmation is the reason this is not just a write. A control that the
    driver accepts and the motor ignores is indistinguishable from success at the
    syscall level -- that is how a half-seated ribbon presents, and it is also
    how libcamera holding the lens presents. Reading the value back is the
    cheapest thing that can tell either apart from working.

    Confirmation is *tolerant on a continuous control and exact on a discrete
    one*, which is :attr:`FocusRange.tolerance`'s whole job. ``focus_absolute``
    is an integer the driver either latched or did not. ``LensPosition`` is a
    request libcamera maps onto the VCM's own steps, so it comes back slightly
    different by design and an exact comparison would call every write a failure.

    Args:
        value: Target, in ``controller``'s units.
        controller: The lens. When omitted, a V4L2 controller is built from
            ``device`` or by searching for ``name_fragment`` -- which keeps every
            existing caller and the whole IMX519 path working unchanged.

    Returns a :class:`FocusStatus` rather than raising, because a camera that
    will not focus should still run: an out-of-focus table is a degraded system,
    a refusal to start is a dead one.
    """
    if controller is None:
        if device is not None:
            controller = V4L2Focus(LensDevice(path=device, name=str(device)), opener=opener)
        else:
            controller = V4L2Focus.find(name_fragment, opener=opener)
            if controller is None:
                present = ", ".join(f"{node} ({name})" for node, name in list_v4l_devices()) or "none"
                detail = (
                    f"no V4L2 subdev matching {name_fragment!r}; focus cannot be set. "
                    f"Present: {present}"
                )
                logger.error("focus: %s", detail)
                return FocusStatus(available=False, requested=value, detail=detail, source=source)

    identity = controller.name

    try:
        focus_range = controller.range()
    except FocusError as exc:
        logger.error("focus: %s", exc)
        return FocusStatus(
            available=False, device=identity, lens_name=identity,
            requested=value, kind=controller.kind, detail=str(exc), source=source,
        )

    target = focus_range.snap(value)
    if target != value:
        # Two different things, and they point at different fixes: a value the
        # lens cannot reach is a stale configured number, a value off the step
        # grid is a driver that quantises. Reporting them as one "adjusted"
        # message loses the part that says what to change.
        reason = (
            f"outside the lens range {focus_range.format(focus_range.minimum)}"
            f"-{focus_range.format(focus_range.maximum)}"
            if not focus_range.contains(value)
            else f"not on the driver's {focus_range.step}-count step grid"
        )
        logger.warning(
            "focus: requested %s is %s; using %s",
            focus_range.format(value), reason, focus_range.format(target),
        )

    try:
        controller.prepare()
        approach_focus(controller, target)
        actual = controller.read()
    except FocusError as exc:
        logger.error("focus: %s", exc)
        return FocusStatus(
            available=True, device=identity, lens_name=identity,
            requested=target, kind=controller.kind, detail=str(exc), source=source,
        )

    if not focus_range.agrees(target, actual):
        # A real error, not a warning. The system will run and every frame will
        # be soft, and there is no other symptom that points here.
        detail = (
            f"lens {identity} accepted {focus_range.format(target)} but reads back "
            f"{focus_range.format(actual)}. "
        ) + (
            "libcamera is not honouring LensPosition -- check AfMode is Manual and "
            "that nothing else has the camera open."
            if controller.kind == FOCUS_DIOPTRES
            else "The lens is not responding to the focus motor -- check the camera "
            "ribbon is fully seated at both ends."
        )
        logger.error("focus: %s", detail)
        return FocusStatus(
            available=True, device=identity, lens_name=identity,
            requested=target, actual=actual, kind=controller.kind,
            ok=False, detail=detail, source=source,
        )

    detail = (
        f"{focus_range.format(actual)} on {identity} "
        f"(range {focus_range.format(focus_range.minimum)}"
        f"-{focus_range.format(focus_range.maximum)})"
    )
    logger.info("focus: %s", detail)
    return FocusStatus(
        available=True, device=identity, lens_name=identity,
        requested=target, actual=actual, kind=controller.kind,
        ok=True, detail=detail, source=source,
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
    #: Where this target was sharpest, in the controller's units. **Float**, not
    #: int: a dioptre peak is fractional -- the useful band for an overhead pool
    #: rig is roughly 0.4 to 1.0 dioptres -- and rounding it to an integer
    #: discards the entire answer. It was ``int``, which is harmless on the
    #: counts path where positions are whole numbers anyway, and on the dioptre
    #: path turned every peak into 0 or 1.
    peak_focus: float
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

    focus_absolute: float
    #: Which control this value drives, and therefore what the number *means*:
    #: :data:`FOCUS_COUNTS` or :data:`FOCUS_DIOPTRES`.
    #:
    #: Defaults to counts, because every file written before this existed came
    #: from the V4L2 path. That default is why an IMX519 rig's calibration stays
    #: valid across this change without being touched.
    #:
    #: There is no conversion between the two, so a mismatch is refused by
    #: :func:`load_focus_calibration` rather than reinterpreted. Reinterpreting
    #: would be the worst available option: 477 counts read as dioptres asks for
    #: focus 2 mm from the lens, 1.8 dioptres read as counts is very nearly
    #: infinity, and both are numbers the system would accept and act on.
    kind: str = FOCUS_COUNTS
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
            "kind": self.kind,
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
                peak_focus=(
                    float(entry["peak_focus"])
                    if str(data.get("kind", FOCUS_COUNTS)) == FOCUS_DIOPTRES
                    else int(entry["peak_focus"])
                ),
                peak_sharpness=float(entry.get("peak_sharpness", 0.0)),
                prominence=float(entry.get("prominence", 0.0)),
            )
            for entry in data.get("per_target", [])
        )
        # Keyed on the unit, because ``int()`` is right for counts and destroys
        # a dioptre value: 1.75 comes back as 1, which is very nearly infinity
        # and is a number the system would happily apply.
        kind = str(data.get("kind", FOCUS_COUNTS))
        raw = data["focus_absolute"]
        return cls(
            focus_absolute=float(raw) if kind == FOCUS_DIOPTRES else int(raw),
            peak_sharpness=float(data.get("peak_sharpness", 0.0)),
            bare_table_sharpness=float(data.get("bare_table_sharpness", 0.0)),
            per_target=targets,
            tilt_spread=int(data.get("tilt_spread", 0)),
            tilt_note=str(data.get("tilt_note", "")),
            # Absent means counts: written before the libcamera path existed,
            # which could only have been the V4L2 one.
            kind=kind,
            approach=str(data.get("approach", APPROACH_DIRECTION)),
            camera_resolution=str(data.get("camera_resolution", "")),
            lens_name=str(data.get("lens_name", "")),
            created_at=str(data.get("created_at", "")),
        )


def focus_calibration_path() -> Path:
    """Where the focus calibration lives. Beside the projector's."""
    from app.config import CALIBRATION_DIR

    return CALIBRATION_DIR / "focus.json"


def load_focus_calibration(
    path: Path | None = None, expected_kind: str | None = None
) -> FocusCalibration | None:
    """Read the saved focus calibration, or ``None`` if there is none.

    A corrupt file is logged and treated as absent rather than raised. Failing
    to boot because a calibration file has a typo in it would be far worse than
    booting uncalibrated -- and booting uncalibrated is already a state the
    panel reports, so nothing is hidden by degrading this way.

    ``expected_kind`` is the unit this rig can actually drive. A file in the
    other unit is **refused, not converted**: there is no published mapping
    between raw counts and dioptres, so any conversion would be invention. The
    two numbers overlap in range too, which is what makes this dangerous rather
    than merely wrong -- a counts value of 477 is a perfectly acceptable dioptre
    request and vice versa, so without this check swapping the camera would
    produce a confidently applied, badly wrong focus with a panel reporting
    "calibrated".
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

    if expected_kind is not None and calibration.kind != expected_kind:
        logger.warning(
            "focus calibration %s is in %s but this camera is driven in %s. The two "
            "are not convertible, so the saved value is being ignored rather than "
            "reinterpreted -- re-run the focus calibration.",
            source, calibration.kind, expected_kind,
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
        "saved focus calibration to %s (focus=%s %s, %d target(s))",
        target,
        calibration.focus_absolute,
        calibration.kind,
        len(calibration.per_target),
    )
    return target


def resolve_focus_value(
    settings, path: Path | None = None, kind: str = FOCUS_COUNTS
) -> tuple[float | None, str]:
    """Decide which focus value to apply, and say where it came from.

    Precedence, highest first:

    1. The hand-set override in config for *this rig's unit* --
       ``camera.focus_absolute`` for counts, ``camera.focus_dioptres`` for
       dioptres. It wins because someone typing a number into a config file has
       made a more specific statement than a calibration run from last month did.
    2. ``data/calibration/focus.json`` -- the calibrated value, and only if it
       was written in the same unit.
    3. Nothing. Returns ``None``, and the caller reports the rig as
       uncalibrated rather than guessing.

    Two separate config keys rather than one, because one key would have to mean
    counts on one camera and dioptres on another, and a number whose unit depends
    on which camera is plugged in is exactly the ambiguity this change exists to
    remove. Having both also means a rig that gets swapped back and forth keeps
    each camera's value.

    There is deliberately no default number. A guessed focus produces a rig that
    is soft for reasons nobody can see; "not calibrated" on the panel points
    straight at the fix.
    """
    key = "focus_dioptres" if kind == FOCUS_DIOPTRES else "focus_absolute"
    override = getattr(settings, key, None)
    if override is not None:
        return (float(override) if kind == FOCUS_DIOPTRES else int(override)), "config"

    calibration = load_focus_calibration(path, expected_kind=kind)
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
