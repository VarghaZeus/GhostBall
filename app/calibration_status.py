"""What is calibrated, what has gone stale, and which flow fixes it.

The camera and the projector are bolted into one box, and that single fact is
what makes this module necessary. Two calibrations depend on where that box is:

* **Camera focus** depends on the distance from the lens to the cloth.
* **Projector alignment** depends on the box's whole pose -- distance, position
  and rotation relative to the table.

So they are *coupled through the mount*, and re-running one does not fix the
other. Move the box up 10 cm: focus is wrong and the projection has grown. Slide
it sideways: focus is fine and the projection has shifted. Re-running only focus
after the first case leaves a stale homography that nothing complains about, and
the symptom -- overlays that miss by a couple of inches -- looks exactly like a
detection problem.

The design decision this module encodes: **measure the staleness, do not guess
at it, and never cascade a deletion.**

Guessing would mean rules like "a focus run invalidates the projector
calibration", which is wrong half the time and throws away a good calibration
for a lateral nudge. Instead each calibration records what the world looked like
when it was taken, and staleness is a *comparison* against that:

* Focus records the sharpness of bare cloth at the chosen lens position. A live
  measurement well below it means the lens is no longer focused there.
* The projector calibration records **where the table's corners sat in the
  camera image** when it was solved. That is a direct proxy for the box's pose:
  if the camera's view of the table has moved, the box has moved, and since the
  projector is rigidly attached to the camera, its alignment moved with it.

Both are measurements the system already takes. Neither needs the user to
remember what they did.

Stale is not deleted. A calibration flagged stale still works and still gets
used -- it is simply reported, with the flow that would fix it, and the user
decides. Deleting a possibly-good calibration on a heuristic would be a worse
failure than leaving a possibly-stale one in place and saying so.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = [
    "CalibrationItem",
    "CalibrationStatus",
    "WizardFlow",
    "assess",
    "corner_drift_px",
]

#: How far the table's corners may drift in the camera image before the
#: projector alignment is presumed stale, as a fraction of the frame width.
#:
#: 2% of a 1920 px frame is ~38 px, which at a typical overhead scale is a
#: little over an inch and a half on the cloth -- comfortably more than
#: detection jitter, comfortably less than a bump anyone would notice.
#:
#: TODO: placeholder pending a measurement on the real rig. The raw drift is
#: always reported alongside the verdict, so a badly chosen threshold still
#: leaves the number visible and the right one derivable.
CORNER_DRIFT_FRACTION = 0.02

#: How far below its calibrated reference the live sharpness may fall before
#: focus is presumed stale. Deliberately loose: the reference is bare cloth
#: under whatever the room lighting was that day, and ambient light moves this
#: number as much as focus does. It is a prompt to check, not a verdict.
SHARPNESS_STALE_RATIO = 0.5


class WizardFlow(str, Enum):
    """Which run fixes a given problem.

    The whole point of having three is that a bumped box should not mean a
    seven-step restart. Each flow re-measures one thing and writes one artifact.
    """

    #: Everything, in order. First install, or a move large enough that nothing
    #: can be trusted.
    FULL = "full"
    #: Camera focus only. The box moved vertically, or the lens was disturbed.
    FOCUS = "focus"
    #: Table position and projector alignment. The box shifted or rotated.
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class CalibrationItem:
    """One calibrated thing, and whether it can still be believed."""

    key: str
    label: str
    calibrated: bool
    #: ``True`` when it exists but the world appears to have changed under it.
    stale: bool = False
    #: One line for the panel, naming what was measured rather than a verdict.
    detail: str = ""
    #: The flow that re-measures this. What the panel's button runs.
    fixed_by: WizardFlow = WizardFlow.FULL
    measured_at: str = ""

    @property
    def ok(self) -> bool:
        return self.calibrated and not self.stale

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "calibrated": self.calibrated,
            "stale": self.stale,
            "ok": self.ok,
            "detail": self.detail,
            "fixed_by": self.fixed_by.value,
            "measured_at": self.measured_at,
        }


@dataclass(frozen=True, slots=True)
class CalibrationStatus:
    """Everything the Setup tab needs to show, and what to offer.

    Exists so that "not calibrated" appears in exactly one place with a button
    next to it, rather than in three places with no way to act on any of them.
    """

    items: list[CalibrationItem] = field(default_factory=list)
    #: The flow to offer most prominently, or ``None`` when nothing needs doing.
    suggested: WizardFlow | None = None
    headline: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "items": [item.as_dict() for item in self.items],
            "suggested": self.suggested.value if self.suggested else None,
            "headline": self.headline,
            "all_ok": all(item.ok for item in self.items) if self.items else False,
        }


def _crop_offset(recorded_crop, current_crop) -> tuple[float | None, float]:
    """Translation from a recorded crop's frame space into the current one.

    ``(None, 0.0)`` means the question cannot be answered: a calibration solved
    before crops were recorded, read on a rig that is now cropping. The recorded
    corners are relative to an origin nobody wrote down, so any comparison is
    guesswork -- and guessing produces a "the box has moved" claim out of a
    re-framing, which is worse than declining to check.
    """
    def origin(crop):
        if crop is None:
            return None
        try:
            return float(crop[0]), float(crop[1])
        except (TypeError, ValueError, IndexError):
            return None

    was, now = origin(recorded_crop), origin(current_crop)
    if was is None and now is None:
        # Neither side crops, or neither knows. Nothing to correct for.
        return 0.0, 0.0
    if was is None or now is None:
        return None, 0.0
    return was[0] - now[0], was[1] - now[1]


def corner_drift_px(recorded, current, recorded_crop=None, current_crop=None) -> float | None:
    """Mean distance between two sets of table corners, in camera px.

    The box-pose proxy. Both sets are the table as *the camera sees it*, so a
    change means the camera moved relative to the table -- almost. There is one
    other way for it to change, and it is not the table walking about: the
    digital crop. Frame-space coordinates are relative to the crop origin, so
    re-framing shifts all four corners at once, by a lot, and the honest reading
    of that is "the box moved" unless the crop is accounted for.

    So the two crops are taken as arguments and the recorded corners are
    translated into the current crop's frame before comparing. Both default to
    ``None`` for callers that have no crop information, and when the crops are
    unknown *and* could differ, this returns ``None`` -- no answer, rather than
    a confident wrong one.

    Mean rather than max: one corner can be occluded or mis-fitted by a hand on
    the rail, and a max would report a bump every time somebody leaned on it.

    Returns ``None`` when either set is missing or malformed, which is the
    normal case for a calibration written before this was recorded.
    """
    if not recorded or not current or len(recorded) != len(current):
        return None

    offset_x, offset_y = _crop_offset(recorded_crop, current_crop)
    if offset_x is None:
        return None
    recorded = [[float(c[0]) + offset_x, float(c[1]) + offset_y] for c in recorded]
    try:
        distances = [
            math.dist((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
            for a, b in zip(recorded, current, strict=True)
        ]
    except (TypeError, ValueError, IndexError):
        return None
    return sum(distances) / len(distances)


def _corners_of(boundary) -> list[list[float]] | None:
    if boundary is None:
        return None
    try:
        return [[float(c.x), float(c.y)] for c in boundary.corners()]
    except (AttributeError, TypeError):
        return None


def assess(
    projector_calibration,
    focus_calibration,
    *,
    boundary=None,
    live_sharpness: float | None = None,
    frame_width: int = 1920,
    drift_fraction: float = CORNER_DRIFT_FRACTION,
    focus_status: dict | None = None,
    current_crop=None,
) -> CalibrationStatus:
    """Judge both calibrations against what the system can currently see.

    Args:
        projector_calibration: The loaded :class:`ProjectorCalibration`, or
            ``None``.
        focus_calibration: The loaded :class:`~vision.focus.FocusCalibration`,
            or ``None``.
        boundary: The table as currently detected, for the drift comparison.
            ``None`` simply means no drift check is possible this pass.
        live_sharpness: A current bare-cloth sharpness reading, if one has been
            taken. Compared against the focus calibration's stored reference --
            never against its projected-target peak, which is an order of
            magnitude larger and would report every rig as broken.
        focus_status: ``AppState.focus_summary()``. Passed in rather than
            recomputed so that this and ``/api/status`` cannot disagree about
            the lens, which is precisely what they were doing.
        current_crop: The digital crop in force, as ``[x, y, width, height]`` in
            sensor px, or ``None``. Needed because ``boundary`` is in frame space
            and the calibration's recorded corners were too -- under a different
            crop. Without it a re-framing reads as the box having been knocked.
    """
    items: list[CalibrationItem] = []

    # -- focus --------------------------------------------------------------
    if focus_calibration is None:
        items.append(
            CalibrationItem(
                key="focus",
                label="Camera focus",
                calibrated=False,
                detail="Never calibrated. The lens sits where it powered up, so the "
                "picture is soft and detection will be poor.",
                fixed_by=WizardFlow.FOCUS,
            )
        )
    else:
        stale, detail = _focus_staleness(focus_calibration, live_sharpness)
        # A saved calibration the lens is not actually sitting at is stale in
        # the way that matters: the number on disk describes a state the
        # hardware is not in.
        if focus_status is not None and not focus_status.get("ok", True):
            stale = True
            detail = str(focus_status.get("detail") or detail)
        items.append(
            CalibrationItem(
                key="focus",
                label="Camera focus",
                calibrated=True,
                stale=stale,
                detail=detail,
                fixed_by=WizardFlow.FOCUS,
                measured_at=focus_calibration.created_at,
            )
        )

    # -- projector alignment ------------------------------------------------
    calibrated = bool(projector_calibration and projector_calibration.is_calibrated)
    if not calibrated:
        items.append(
            CalibrationItem(
                key="projector",
                label="Table alignment",
                calibrated=False,
                detail="Never aligned. Overlays are drawn through an identity mapping, "
                "so they will not line up with the felt.",
                fixed_by=WizardFlow.TABLE,
            )
        )
    else:
        stale, detail = _alignment_staleness(
            projector_calibration, boundary, frame_width, drift_fraction, current_crop
        )
        items.append(
            CalibrationItem(
                key="projector",
                label="Table alignment",
                calibrated=True,
                stale=stale,
                detail=detail,
                fixed_by=WizardFlow.TABLE,
                measured_at=projector_calibration.created_at,
            )
        )

    return _summarise(items)


def _focus_staleness(calibration, live_sharpness: float | None) -> tuple[bool, str]:
    reference = calibration.bare_table_sharpness
    base = f"Set to {calibration.focus_absolute}"
    if calibration.tilt_note:
        base += " (camera is tilted -- see the calibration report)"

    if live_sharpness is None or reference <= 0:
        return False, f"{base}."
    ratio = live_sharpness / reference
    if ratio < SHARPNESS_STALE_RATIO:
        return True, (
            f"{base}, but the picture now measures {ratio * 100:.0f}% as sharp as when it "
            "was calibrated. The box may have been moved or knocked."
        )
    return False, f"{base}. Sharpness is {ratio * 100:.0f}% of its calibrated reading."


def _alignment_staleness(
    calibration, boundary, frame_width: int, drift_fraction: float, current_crop=None
) -> tuple[bool, str]:
    base = f"Solved to {calibration.rmse_px:.1f} px"
    current = _corners_of(boundary)
    drift = corner_drift_px(
        getattr(calibration, "table_corners_px", None),
        current,
        recorded_crop=getattr(calibration, "camera_crop", None),
        current_crop=current_crop,
    )

    if drift is None:
        # Either nothing recorded to compare against, or the crop moved and the
        # recorded corners predate crops -- see corner_drift_px. Both mean no
        # answer; neither means "no drift".
        return False, f"{base}."

    limit = frame_width * drift_fraction
    if drift > limit:
        # Stated as an observation with its number, not as a diagnosis. The
        # threshold is a placeholder; the measurement is not.
        return True, (
            f"{base}, but the table now sits {drift:.0f} px from where it was when this "
            f"was solved (limit {limit:.0f}). The box has moved, so the projection will "
            "be off by roughly the same amount."
        )
    return False, f"{base}. The table is within {drift:.0f} px of where it was solved."


def _summarise(items: list[CalibrationItem]) -> CalibrationStatus:
    """Pick what to offer, and say it in one line.

    The ordering rule is that a *move* is reported before a *gap*: if both
    calibrations exist and both look stale, the box was bumped and a full run is
    the honest answer, because the two are coupled through the mount and fixing
    one leaves the other mismatched.
    """
    missing = [i for i in items if not i.calibrated]
    stale = [i for i in items if i.calibrated and i.stale]

    if not missing and not stale:
        return CalibrationStatus(items=items, suggested=None, headline="Calibrated and ready.")

    if len(missing) == len(items):
        return CalibrationStatus(
            items=items,
            suggested=WizardFlow.FULL,
            headline="Not set up yet. Run the full setup to get started.",
        )

    if len(stale) == len(items):
        # Both moved together, which is what a bumped box looks like. Offering
        # one of them would fix half the problem and leave a mismatch that the
        # user has no way to see.
        return CalibrationStatus(
            items=items,
            suggested=WizardFlow.FULL,
            headline=(
                "Both focus and alignment look stale, which usually means the box was "
                "moved. Run the full setup -- fixing only one would leave the other "
                "mismatched."
            ),
        )

    needs = missing + stale
    first = needs[0]
    verb = "calibrated" if not first.calibrated else "re-checked"
    return CalibrationStatus(
        items=items,
        suggested=first.fixed_by,
        headline=f"{first.label} needs to be {verb}.",
    )
