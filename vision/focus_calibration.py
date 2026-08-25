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
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings, get_settings
from vision.focus import FocusRange, TargetPeak, approach_focus, read_focus

logger = logging.getLogger(__name__)

__all__ = [
    "FocusDiagnosis",
    "SweepOutcome",
    "TargetRegion",
    "analyse",
    "detect_targets",
    "measure_regions",
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
    readback_drift: int = 0

    @property
    def ok(self) -> bool:
        return self.best_focus is not None


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
    device,
    regions: list[TargetRegion],
    positions: list[int],
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
        approach_focus(device, position, focus_range)
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

        if read_focus(device) != position:
            outcome.readback_drift += 1

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
        outcome.diagnosis = FocusDiagnosis(
            code="lens_not_tracking",
            message=(
                f"The lens did not go where it was sent at {outcome.readback_drift} of the "
                "focus positions. The motor is not tracking -- check the camera ribbon is "
                "fully seated at both ends."
            ),
        )
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
    low = focus_range.minimum if start is None else focus_range.clamp(start)
    high = focus_range.maximum if end is None else focus_range.clamp(end)
    if low > high:
        low, high = high, low
    positions = list(range(low, high + 1, max(1, step)))
    if positions[-1] != high:
        positions.append(high)
    return positions


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
