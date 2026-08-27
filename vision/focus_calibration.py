"""Projector-assisted focus calibration.

Pool felt is nearly featureless. A contrast-based focus sweep against bare cloth
gives a curve with no peak in it, and the "best" position that falls out is
whichever stop caught the most sensor noise. The projector fixes this exactly:
it can put guaranteed high-frequency edges at precisely the plane we care about,
the felt, and nowhere else.

Four things this does that a naive sweep does not.

**It finds the targets rather than computing where they should be.**
The obvious route is table -> projector through the mapper, but that needs a
solved projector calibration, which does not exist yet at this point in the
wizard and which itself wants a focused camera. Circular. So the targets are
*detected* in the camera image as bright blobs. That needs no calibration at
all, works at any focus -- defocus blurs a blob symmetrically and barely moves
its centroid -- and answers "is the pattern actually visible?" in two seconds
instead of as a flat curve after two minutes.

**It measures inside the targets only.** Room clutter at other distances --
floor, cushions, whatever is on the wall behind -- has its own focus optimum,
and including it adds a second peak that competes with the real one.

**It locks exposure, and verifies the lock.** See
:meth:`vision.camera.Camera.exposure_lock`. Sharpness scales with contrast, so
an AE change reads as a focus change; and AE genuinely does move during a sweep,
in a direction correlated with focus. Requesting the lock is not enough --
``set_controls`` accepting a request is a different claim from the ISP honouring
it, which is the same trap the libcamera focus path fell into.

**It reports five peaks, not one.** One peak says where to put the lens. Five
say whether the sensor plane is parallel to the cloth, which is a mount problem
that will wreck the homography later and is invisible from a single reading.

Throughout: a confident wrong number is worse than no number. Every path that
cannot justify an answer refuses to give one, and says which of the known
failures it looks like.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings, get_settings
from vision.focus import FOCUS_DIOPTRES, FocusRange, TargetPeak, approach_focus

logger = logging.getLogger(__name__)

__all__ = [
    "COARSE_SWEEP_SAMPLES",
    "FocusDiagnosis",
    "ReadbackSample",
    "SweepOutcome",
    "TargetRegion",
    "analyse",
    "coarse_step",
    "detect_targets",
    "focus_positions",
    "measure_regions",
    "readback_diagnosis",
    "sweep_focus",
]

#: Size of the measurement box around each detected target, as a fraction of the
#: target's detected extent. Under 1.0 on purpose: the outer checkers are the
#: coarse band, present so the blob can be *found*, while the discriminating
#: fine band is in the middle. Measuring the whole blob dilutes the fine band's
#: contribution with squares that are resolved at every focus.
REGION_SHRINK = 0.7

#: Minimum blob area in camera px for a candidate to be a target rather than a
#: reflection off a ball or a light fitting.
MIN_TARGET_AREA_PX = 400

#: A peak must stand this far above the curve's median to count as a peak.
#: Below it the curve is flat and its maximum is noise.
#:
#: The checks that use this run in a deliberate order -- see :func:`analyse`.
#: Structural explanations (nothing visible, lens not moving, optimum outside
#: the lens range) come before statistical ones (weak peak, several peaks),
#: because a structural cause makes the statistics a symptom rather than the
#: story.
MIN_PROMINENCE = 1.5

#: Below this the curve has a peak but a weak one -- the usual cause is ambient
#: light washing the projected pattern out.
SHALLOW_PROMINENCE = 3.0

#: Per-target peaks spreading more than this many focus counts means the sensor
#: plane is not parallel to the cloth.
#:
#: TODO: placeholder. This wants measuring on the real rig -- deliberately
#: tilt the mount by a known amount and read the spread. Until then the report
#: always prints the raw per-target values, so a badly chosen threshold here
#: still leaves the data visible and the real number derivable from it.
DEFAULT_TILT_THRESHOLD = 120

#: How many stops a coarse sweep aims to visit, across whatever range the driver
#: advertises.
#:
#: A count, not a stride, because a stride is only meaningful relative to a span.
#: The stride was hardcoded at 128, chosen on an ak7375 whose range is 0-4095 --
#: 33 stops, a reasonable coarse pass. The same 128 on a dw9807 (0-1023) is 9
#: stops for the entire sweep, which is far too coarse to locate a peak: the
#: fine pass is left starting from a "peak" that is really just the least-bad of
#: nine samples. Deriving from the span keeps the sweep's *resolution* fixed
#: instead of its arithmetic.
COARSE_SWEEP_SAMPLES = 32


@dataclass(frozen=True, slots=True)
class TargetRegion:
    """One projected target, located in the camera image."""

    name: str
    center_px: tuple[float, float]
    #: Measurement box as ``(x0, y0, x1, y1)`` in camera px.
    box: tuple[int, int, int, int]
    area_px: int

    def crop(self, image: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        return image[y0:y1, x0:x1]


@dataclass(frozen=True, slots=True)
class ReadbackSample:
    """One position the lens was sent to, and the value it reported afterwards."""

    written: float
    read: float
    #: How far apart the two may sit and still count as arrived. Zero for raw
    #: counts, non-zero for dioptres -- libcamera quantises the request onto the
    #: VCM's own steps, so an exact comparison there would report every write as
    #: a failure. Supplied by :attr:`vision.focus.FocusRange.tolerance`, so the
    #: rule lives in one place rather than being re-decided per caller.
    tolerance: float = 0.0

    @property
    def agrees(self) -> bool:
        return abs(float(self.written) - float(self.read)) <= self.tolerance


@dataclass(slots=True)
class SweepOutcome:
    """Everything a sweep produced, whether or not it succeeded."""

    #: Per target: focus position -> sharpness.
    curves: dict[str, dict[int, float]] = field(default_factory=dict)
    peaks: list[TargetPeak] = field(default_factory=list)
    regions: list[TargetRegion] = field(default_factory=list)
    #: Best overall focus, or ``None`` when the result was refused.
    best_focus: int | None = None
    best_sharpness: float = 0.0
    tilt_spread: int = 0
    tilt_note: str = ""
    diagnosis: FocusDiagnosis | None = None
    exposure_detail: str = ""
    #: Every position the lens was sent to, with where it said it went.
    #:
    #: This was a bare mismatch count, and a count cannot be diagnosed. "The
    #: readback disagreed 9 times" is true of a driver clamping an out-of-range
    #: request, of a motor that is not moving at all, of a driver that rounds to
    #: a coarse step, and of something else driving the same lens concurrently --
    #: four different problems, one of which is a cable and three of which are
    #: not. The numbers are what tell them apart, so the numbers are kept.
    readbacks: list[ReadbackSample] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.best_focus is not None

    @property
    def readback_drift(self) -> int:
        """How many positions read back as something other than what was sent."""
        return sum(1 for sample in self.readbacks if not sample.agrees)


@dataclass(frozen=True, slots=True)
class FocusDiagnosis:
    """A refusal, phrased as the physical thing to go and do about it.

    Keyed by ``code`` so the wizard can branch on it, but the ``message`` is
    what a person standing at the table reads, and it names an action rather
    than a measurement.
    """

    code: str
    message: str
    #: False when the sweep found a usable answer anyway -- tilt is reported as
    #: a warning alongside a valid focus value, not instead of one.
    fatal: bool = True


# ---------------------------------------------------------------------------
# Finding the targets
# ---------------------------------------------------------------------------


def detect_targets(
    image: np.ndarray,
    expected: int = 5,
    names: list[str] | None = None,
    min_area_px: int = MIN_TARGET_AREA_PX,
) -> list[TargetRegion]:
    """Locate the projected targets as bright blobs in a camera frame.

    Threshold, take connected components, keep the largest. Deliberately crude,
    because it has to work on a heavily defocused frame at the wrong end of the
    lens range -- anything relying on corners or edges would find nothing there,
    which is precisely when it is needed.

    Otsu rather than a fixed threshold: the projector's brightness on cloth
    varies with the room, the lamp's age and how far the throw is, and a fixed
    level would need retuning per install.

    Names are assigned by position, not by matching against where the pattern
    put them -- the point is to avoid depending on a projector calibration.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Blur first: at best focus the checkerboard is many small blobs rather than
    # one, and the job here is to find the *target*, not its squares.
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=9)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area_px:
            continue
        # Targets are square-ish. A long thin region is a rail highlight or the
        # edge of the projection, not a target.
        aspect = w / h if h else 999.0
        if not 0.4 <= aspect <= 2.5:
            continue
        candidates.append((area, index, (x, y, w, h), centroids[index]))

    candidates.sort(reverse=True, key=lambda c: c[0])
    chosen = candidates[:expected]
    if not chosen:
        return []

    regions = [
        _region_from_blob(image.shape, box, centroid, area)
        for area, _index, box, centroid in chosen
    ]
    return _name_regions(regions, names)


def _region_from_blob(shape, box, centroid, area) -> TargetRegion:
    x, y, w, h = box
    cx, cy = float(centroid[0]), float(centroid[1])
    half_w = w * REGION_SHRINK / 2.0
    half_h = h * REGION_SHRINK / 2.0
    height, width = shape[:2]
    return TargetRegion(
        name="",
        center_px=(cx, cy),
        box=(
            max(0, int(cx - half_w)),
            max(0, int(cy - half_h)),
            min(width, int(cx + half_w)),
            min(height, int(cy + half_h)),
        ),
        area_px=int(area),
    )


def _name_regions(regions: list[TargetRegion], names: list[str] | None) -> list[TargetRegion]:
    """Name each region by where it sits, so reports are readable.

    Positional naming rather than matching against the pattern's table
    coordinates: the whole design avoids needing a projector calibration here,
    and "top_left" from the camera's point of view is what a person looking at
    the report wants anyway.
    """
    if not regions:
        return []
    xs = [r.center_px[0] for r in regions]
    ys = [r.center_px[1] for r in regions]
    mid_x, mid_y = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0

    named: list[TargetRegion] = []
    for region in regions:
        cx, cy = region.center_px
        # The centre target is the one nearest the middle of the whole set.
        near_x = abs(cx - mid_x) < (max(xs) - min(xs)) * 0.2
        near_y = abs(cy - mid_y) < (max(ys) - min(ys)) * 0.2
        if near_x and near_y:
            name = "centre"
        else:
            name = ("top" if cy < mid_y else "bottom") + ("_left" if cx < mid_x else "_right")
        named.append(
            TargetRegion(
                name=name, center_px=region.center_px, box=region.box, area_px=region.area_px
            )
        )

    # Duplicate names mean the layout was not what we assumed; fall back to
    # numbering rather than reporting two regions with the same label.
    if len({r.name for r in named}) != len(named):
        named = [
            TargetRegion(name=f"target_{i + 1}", center_px=r.center_px, box=r.box, area_px=r.area_px)
            for i, r in enumerate(sorted(named, key=lambda r: (r.center_px[1], r.center_px[0])))
        ]
    if names:
        named = [
            TargetRegion(name=n, center_px=r.center_px, box=r.box, area_px=r.area_px)
            for n, r in zip(names, named, strict=False)
        ]
    return named


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def sharpness_of(patch: np.ndarray) -> float:
    """Variance of the Laplacian over a patch.

    Comparable only between measurements of the *same* patch under the *same*
    exposure. That is why the exposure lock exists and why the region boxes are
    fixed once at the start rather than re-detected per step.
    """
    import cv2

    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_regions(image: np.ndarray, regions: list[TargetRegion]) -> dict[str, float]:
    """Sharpness inside each region."""
    return {region.name: sharpness_of(region.crop(image)) for region in regions}


def sweep_focus(
    camera,
    controller,
    regions: list[TargetRegion],
    positions: list[float],
    focus_range: FocusRange,
    *,
    settle_seconds: float = 0.35,
    frames: int = 3,
    exposure_status=None,
    on_step=None,
) -> SweepOutcome:
    """Step the lens across ``positions``, measuring every region at each stop.

    All five regions come from the same frames, so the cost is per focus step
    rather than per target.

    Args:
        exposure_status: Result of :meth:`vision.camera.Camera.exposure_lock`.
            When given, drift is checked at every step and the sweep is aborted
            the moment the lock slips -- continuing would produce a plausible
            curve that means nothing.
        on_step: Progress callback ``(index, total, position, per_region)``.
    """
    outcome = SweepOutcome(regions=list(regions))
    outcome.curves = {region.name: {} for region in regions}
    if exposure_status is not None:
        outcome.exposure_detail = exposure_status.detail

    for index, position in enumerate(positions, start=1):
        # Always from below -- see vision.focus.approach_focus. The sweep has to
        # arrive the same way startup does, or the calibrated value is soft at
        # boot with nothing to explain why.
        approach_focus(controller, position)
        time.sleep(settle_seconds)

        # Discard: these were already in the ISP pipeline before the control
        # changed, so they show the previous focus.
        for _ in range(frames):
            camera.capture_frame()

        per_frame: list[dict[str, float]] = []
        for _ in range(frames):
            frame = camera.capture_frame()
            if frame is not None:
                per_frame.append(measure_regions(frame.image, regions))

        if not per_frame:
            logger.warning("no frames captured at focus_absolute=%d", position)
            continue

        # Median across frames: one frame caught mid-anything is a large
        # outlier, and a mean would let it move the peak.
        measured = {
            region.name: float(np.median([m[region.name] for m in per_frame]))
            for region in regions
        }
        for name, value in measured.items():
            outcome.curves[name][position] = value

        outcome.readbacks.append(
            ReadbackSample(
                written=position,
                read=controller.read(),
                tolerance=focus_range.tolerance,
            )
        )

        if exposure_status is not None:
            drift = camera.exposure_drifted(exposure_status)
            if drift:
                outcome.diagnosis = FocusDiagnosis(
                    code="ae_lock_failed",
                    message=(
                        f"Exposure moved during the sweep ({drift}). Every reading after "
                        "that point is measuring brightness, not focus, so the result is "
                        "discarded. Re-run it; if it happens again the camera is not "
                        "honouring the exposure lock."
                    ),
                )
                return outcome

        if on_step is not None:
            on_step(index, len(positions), position, measured)

    return outcome


# ---------------------------------------------------------------------------
# Making sense of it
# ---------------------------------------------------------------------------


def _peak_of(name: str, curve: dict[int, float], center_px) -> TargetPeak | None:
    if not curve:
        return None
    position = max(curve, key=lambda p: curve[p])
    values = list(curve.values())
    median = float(np.median(values)) or 1e-9
    return TargetPeak(
        name=name,
        center_px=(float(center_px[0]), float(center_px[1])),
        peak_focus=int(position),
        peak_sharpness=float(curve[position]),
        prominence=float(curve[position] / median),
    )


def _count_local_maxima(curve: dict[int, float]) -> int:
    """Local maxima that are more than noise.

    A clean focus curve is unimodal. Two real humps mean either something else
    in the box is in focus at a different distance, or the lens was shaken --
    and either way the single "best" position is not trustworthy.
    """
    ordered = [curve[p] for p in sorted(curve)]
    if len(ordered) < 5:
        return 1
    threshold = max(ordered) * 0.5
    maxima = 0
    for i in range(1, len(ordered) - 1):
        if ordered[i] >= threshold and ordered[i] > ordered[i - 1] and ordered[i] >= ordered[i + 1]:
            maxima += 1
    return max(1, maxima)


def _summarise(
    samples: list[ReadbackSample], focus_range: FocusRange, limit: int = 4
) -> str:
    """``sent 128 read 480`` pairs, truncated. The evidence, in the message.

    Naming the numbers is the whole point. Every diagnosis below is a guess at a
    cause; the pairs are the observation the guess was made from, so a reader who
    disagrees with the guess can still see what happened.
    """
    shown = ", ".join(
        f"sent {focus_range.format(s.written)} read {focus_range.format(s.read)}"
        for s in samples[:limit]
    )
    if len(samples) > limit:
        shown += f", and {len(samples) - limit} more"
    return shown


def readback_diagnosis(
    samples: list[ReadbackSample], focus_range: FocusRange
) -> FocusDiagnosis:
    """Why the lens reported positions other than the ones it was sent.

    This used to be one message naming the ribbon cable, fired on a bare count
    of mismatches. That is wrong more often than it is right, and expensively so:
    it sends someone to reseat a cable that is fine, and it does it with enough
    confidence that the actual cause goes unexamined. At least four distinct
    faults produce a disagreeing readback, and only one of them is a cable:

    * **The value never fit.** The driver clamped an out-of-range request, so of
      course the readback differs. A software fault -- whatever computed the
      position used the wrong bounds -- and no amount of reseating helps.
    * **The lens never moved.** Every position reads back identical. Either
      nothing is driving the motor (the cable case), or the value being read is
      not the one being written.
    * **The driver quantises.** Readbacks land on the step grid rather than
      where they were sent. Not a fault at all; the request was simply not
      expressible.
    * **Something else is driving the same lens.** Readbacks vary, are in range,
      and are not the requested values. On a sensor whose libcamera tuning binds
      an AF algorithm, that algorithm owns the same VCM this module writes to,
      and it wins -- see :mod:`vision.focus`.

    Ordered from most to least specific, so the narrowest explanation that fits
    the evidence is the one reported.
    """
    mismatched = [sample for sample in samples if not sample.agrees]
    if not mismatched:
        return FocusDiagnosis(code="lens_tracking", message="", fatal=False)

    # 1. Out of range: the request could never have been honoured.
    out_of_range = [s for s in mismatched if not focus_range.contains(s.written)]
    if out_of_range:
        return FocusDiagnosis(
            code="focus_out_of_range",
            message=(
                f"{len(out_of_range)} of the focus positions were outside the lens range "
                f"{focus_range.format(focus_range.minimum)}-"
                f"{focus_range.format(focus_range.maximum)}, so the driver clamped them "
                "and the readback disagrees for that reason alone "
                f"({_summarise(out_of_range, focus_range)}). "
                "The motor and the cable are not implicated. This is a software fault: "
                "whatever chose these positions used the wrong bounds for this lens."
            ),
        )

    # 2. Every reading identical: the lens is parked.
    readings = {sample.read for sample in samples}
    if len(readings) == 1 and len(samples) > 1:
        stuck = next(iter(readings))
        at_default = " -- its power-on default" if stuck == focus_range.default else ""
        control = "LensPosition" if focus_range.kind == FOCUS_DIOPTRES else "focus_absolute"
        # Different unit, different culprit. On the libcamera path a cable cannot
        # be the explanation -- the writes are going through the same process
        # that owns the sensor -- so pointing at one would send somebody to a
        # screwdriver for a software problem. That mistake is the reason this
        # function exists.
        cause = (
            "libcamera is still driving the lens: AfMode is not Manual, or something "
            "else has the camera open and is running autofocus against you."
            if focus_range.kind == FOCUS_DIOPTRES
            else "Check the camera ribbon is fully seated at both ends; if a manual write "
            "with v4l2-ctl does stick, then the write is reaching a different device than "
            "the one being read."
        )
        return FocusDiagnosis(
            code="lens_not_moving",
            message=(
                f"The lens reported {control}={focus_range.format(stuck)} at every one of "
                f"the {len(samples)} positions it was sent to{at_default}. It is not moving "
                f"at all, rather than moving imprecisely. {cause}"
            ),
        )

    # 3. Readbacks on the step grid: the driver rounded, nothing is broken.
    if focus_range.step > 1 and all(s.read == focus_range.snap(s.written) for s in mismatched):
        return FocusDiagnosis(
            code="focus_quantised",
            message=(
                f"The driver rounded every focus request to its {focus_range.step}-count step "
                f"grid ({_summarise(mismatched, focus_range)}). Nothing is faulty -- the positions asked for "
                "were simply not expressible. Sweep on step-aligned positions."
            ),
        )

    # 4. Moving, in range, but not to order.
    #
    # Unit-aware, like case 2 and for the same reason: the plausible causes are
    # different, and naming the wrong one costs a trip to the rig with a
    # screwdriver for what is a software problem, or the reverse.
    if focus_range.kind == FOCUS_DIOPTRES:
        cause = (
            "Most likely something is still driving autofocus: check AfMode is Manual "
            "and that no other process has the camera open. A cable cannot explain this "
            "one -- the writes are going through the same process that owns the sensor."
        )
    else:
        cause = (
            "Two causes worth separating: the ribbon is marginal, or something else is "
            "driving the same motor -- if this sensor's libcamera tuning binds an AF "
            "algorithm, it owns this VCM too and will overwrite every position while the "
            "camera streams. A manual v4l2-ctl write that sticks while nothing is "
            "streaming points at the second, not the first."
        )
    return FocusDiagnosis(
        code="lens_not_tracking",
        message=(
            f"The lens moved but not to where it was sent, at {len(mismatched)} of "
            f"{len(samples)} positions ({_summarise(mismatched, focus_range)}). Every value "
            f"asked for was within the lens range {focus_range.format(focus_range.minimum)}-"
            f"{focus_range.format(focus_range.maximum)}, so this is not a clamp. {cause}"
        ),
    )


def analyse(
    outcome: SweepOutcome,
    focus_range: FocusRange,
    tilt_threshold: int = DEFAULT_TILT_THRESHOLD,
) -> SweepOutcome:
    """Turn curves into a decision, or into the reason there isn't one.

    Order matters. The checks run from "nothing was measured" outward to "the
    answer is fine but the mount is crooked", so the first thing reported is
    always the most upstream problem -- telling someone their camera is tilted
    when the projector is off would send them up a ladder for nothing.
    """
    if outcome.diagnosis is not None:  # already refused, e.g. exposure drift
        return outcome

    if not outcome.curves or not any(outcome.curves.values()):
        outcome.diagnosis = FocusDiagnosis(
            code="no_targets",
            message=(
                "No focus targets were visible. Is the projector on, awake, and aimed at "
                "the table? The checkerboards should be clearly visible on the cloth "
                "before this runs."
            ),
        )
        return outcome

    if outcome.readback_drift:
        outcome.diagnosis = readback_diagnosis(outcome.readbacks, focus_range)
        return outcome

    peaks = [
        peak
        for region in outcome.regions
        if (peak := _peak_of(region.name, outcome.curves[region.name], region.center_px))
    ]
    outcome.peaks = peaks
    if not peaks:
        outcome.diagnosis = FocusDiagnosis(
            code="no_targets", message="No usable measurements were taken."
        )
        return outcome

    best = max(peaks, key=lambda p: p.peak_sharpness)
    prominence = max(p.prominence for p in peaks)

    if prominence < MIN_PROMINENCE:
        outcome.diagnosis = FocusDiagnosis(
            code="flat_curve",
            message=(
                f"The sharpness curve is flat (best is only {prominence:.2f}x the median), "
                "so there is no focus peak to find. Either the targets are not actually "
                "landing on the table, or the lens is not moving. Check the projection "
                "first."
            ),
        )
        return outcome

    # Before the shallow-peak check, deliberately. A curve whose true optimum
    # lies outside what was swept runs monotonically into the edge, and such a
    # curve is often only modestly prominent -- so testing prominence first
    # would diagnose a real out-of-range mount as "dim the room", which is
    # advice that cannot possibly help. A peak sitting on an edge is a
    # structural fact about the curve and a more specific claim, so it is asked
    # first.
    #
    # Against the *swept* endpoints, not the lens limits. A peak at the end of
    # the measured window means the optimum is outside the window, and that is
    # true whether the window was the whole lens range or a hand-picked
    # ``--start/--end``. Which of those it was decides the advice: one means
    # move the camera, the other means widen the sweep, and telling somebody to
    # remount a camera when they simply swept too narrow a band would be an
    # expensive mistake.
    swept = sorted({p for curve in outcome.curves.values() for p in curve})
    at_edge = [p for p in peaks if swept and p.peak_focus in (swept[0], swept[-1])]
    if at_edge:
        peak_focus = at_edge[0].peak_focus
        near_end = peak_focus == swept[-1]
        at_lens_limit = peak_focus in (focus_range.minimum, focus_range.maximum)

        if at_lens_limit:
            direction = "further from" if near_end else "closer to"
            edge_word = "closest" if near_end else "furthest"
            message = (
                f"Best focus is at the {edge_word} the lens can go ({peak_focus}), so true "
                f"focus is outside its range. Move the camera {direction} the table and "
                "run this again."
            )
        else:
            message = (
                f"Best focus is at the edge of the range that was swept ({peak_focus}), so "
                "the real optimum was never measured. Widen the sweep -- the lens can go "
                f"from {focus_range.minimum} to {focus_range.maximum}."
            )
        outcome.diagnosis = FocusDiagnosis(code="out_of_range", message=message)
        return outcome

    noisy = [p.name for p in peaks if _count_local_maxima(outcome.curves[p.name]) > 1]
    if len(noisy) > len(peaks) / 2:
        outcome.diagnosis = FocusDiagnosis(
            code="multiple_peaks",
            message=(
                f"{len(noisy)} of {len(peaks)} targets show more than one focus peak. That "
                "is usually vibration during the sweep, or an exposure lock that did not "
                "hold. Make sure nothing is touching the table or the mount, and re-run."
            ),
        )
        return outcome

    # The residual case: a single interior peak that is simply weak. With the
    # structural explanations ruled out above, ambient light washing the pattern
    # out is by far the likeliest cause.
    if prominence < SHALLOW_PROMINENCE:
        outcome.diagnosis = FocusDiagnosis(
            code="shallow_peak",
            message=(
                f"There is a peak but a weak one ({prominence:.1f}x the median), which "
                "usually means ambient light is washing the pattern out. Dim the room and "
                "run it again."
            ),
        )
        return outcome

    # A usable answer. Everything below is reported alongside it, not instead.
    outcome.best_focus = best.peak_focus
    outcome.best_sharpness = best.peak_sharpness

    focuses = [p.peak_focus for p in peaks]
    outcome.tilt_spread = int(max(focuses) - min(focuses))
    if len(peaks) >= 3 and outcome.tilt_spread > tilt_threshold:
        near = min(peaks, key=lambda p: p.peak_focus)
        far = max(peaks, key=lambda p: p.peak_focus)
        outcome.tilt_note = (
            f"The camera is tilted: {far.name} focuses {outcome.tilt_spread} counts "
            f"differently from {near.name}, so the sensor is not parallel to the cloth. "
            "Level the mount before calibrating the projector -- a tilt this size will "
            "distort the table homography too."
        )
        outcome.diagnosis = FocusDiagnosis(
            code="camera_tilted", message=outcome.tilt_note, fatal=False
        )
    return outcome


def focus_positions(
    focus_range: FocusRange, step: int, start: int | None = None, end: int | None = None
) -> list[int]:
    """Focus stops to visit, inclusive of both ends."""
    low = focus_range.minimum if start is None else focus_range.snap(start)
    high = focus_range.maximum if end is None else focus_range.snap(end)
    if low > high:
        low, high = high, low

    # Generated by index rather than with range(), because a dioptre sweep steps
    # by fractions -- 0.31 of a dioptre, say -- and range() is integers only.
    # Integer ranges come out bit-identical: with low=0, high=4095, step=127 the
    # arithmetic below produces exactly what range(0, 4096, 127) did.
    span = high - low
    step = max(step, 1e-9) if focus_range.continuous else max(1, int(step))
    count = max(1, int(math.floor(span / step)))
    positions = [low + index * step for index in range(count + 1)]
    if positions[-1] < high:
        positions.append(high)

    # Snapped, so a driver that quantises cannot turn a position this function
    # chose into a readback mismatch. A no-op on a continuous control and at
    # step=1, which is both lenses seen so far -- but the sweep asserts that what
    # it wrote is what it asked for, and that assertion has to be one the driver
    # can actually satisfy.
    snapped = [focus_range.snap(p) for p in positions]
    # dict.fromkeys rather than set(): order is the sweep order, and ascending
    # order is load-bearing (see vision.focus.approach_focus).
    return list(dict.fromkeys(snapped))


def coarse_step(focus_range: FocusRange) -> float:
    """Stride for a coarse sweep over ``focus_range``, in its own units.

    See :data:`COARSE_SWEEP_SAMPLES`. Derived from the span so the sweep's
    *resolution* is what stays fixed across lenses rather than its arithmetic --
    which matters twice over now, since a dioptre range is 0-32 where a counts
    range is 0-4095, and a stride chosen for either is absurd on the other.

    Never finer than the driver's own step, which would produce duplicates.
    """
    if focus_range.continuous:
        return abs(focus_range.span) / COARSE_SWEEP_SAMPLES
    return max(focus_range.step, 1, int(abs(focus_range.span) // COARSE_SWEEP_SAMPLES))


def bare_reference(
    camera, regions: list[TargetRegion], frames: int = 5, settings: Settings | None = None
) -> float:
    """Sharpness of the same regions with the targets off.

    The runtime health check's only usable reference. It cannot compare against
    the peak from this sweep -- that was measured on high-contrast projected
    checkerboards and sits an order of magnitude above anything bare cloth
    produces, so the comparison would fire on every boot.

    The caller must have blanked the projector before calling this. Nothing here
    can verify that, which is worth stating: a "bare" reference taken with the
    targets still up would be quietly useless.
    """
    settings = settings or get_settings()
    scores: list[float] = []
    for _ in range(frames):
        frame = camera.capture_frame()
        if frame is not None:
            measured = measure_regions(frame.image, regions)
            if measured:
                scores.append(float(np.median(list(measured.values()))))
    return float(np.median(scores)) if scores else 0.0
