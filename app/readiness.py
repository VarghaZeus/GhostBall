"""Is this thing ready to play, and if not, what should the person do?

Separate from :class:`~app.models.SessionState`, deliberately. That enum
describes a *shot* -- idle, aiming, shot in progress, settling -- it is owned by
the mode manager and driven by cue and ball motion. "There is no table in front
of the camera" is not a phase of a shot. Folding it in would force every mode's
scoring rules to guard against a state that has nothing to do with them, and
would put the mode manager in charge of a question about the hardware.

So readiness is its own small machine, owned by the vision loop (which is what
already knows about frames and table detection), living on ``AppState``, read by
both the renderer and the panel.

Three things it is careful about.

**Hysteresis.** A hand over the table, or one frame where the felt mask leaked,
must not throw a running game into a full-screen setup screen. The recovery from
a false positive here is far more disruptive than the recovery from a false
negative, so transitions require the condition to hold for a run of frames --
the same discipline the shot machine uses, for a stronger reason.

**Confidence, not presence.** ``table_boundary is not None`` is the wrong test.
Pointed at a ceiling, felt segmentation will occasionally hand back a
low-confidence quad from whatever happens to be greenish, and a system that
declares itself ready on that is worse than one that admits it cannot see a
table.

**A dead camera is not an absent table.** They look identical from
``table_boundary is None`` and they want completely different instructions --
one says check the ribbon, the other says point me at a pool table. The frame
clock separates them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.models import SystemState

logger = logging.getLogger(__name__)

__all__ = ["Readiness", "ReadinessTracker", "SystemState"]

#: Consecutive frames a condition must hold before the state changes. At 15-30
#: FPS this is one to two seconds -- long enough to ride out an arm reaching
#: across the table, short enough that mounting the camera gives feedback while
#: you are still on the ladder.
CONFIRM_FRAMES = 30

#: Seconds without a frame before the camera is presumed gone. Well above any
#: plausible hiccup; the vision loop's own watchdog handles genuinely wedged
#: loops on a shorter fuse.
CAMERA_SILENT_SECONDS = 3.0

#: Below this, a detected boundary is not trusted. Pointed at anything that is
#: not a pool table, the felt detector still returns *something* now and then.
MIN_TABLE_CONFIDENCE = 0.45


@dataclass(frozen=True, slots=True)
class Readiness:
    """The current state, and what to say about it.

    ``headline`` and ``detail`` are projected on the cloth and shown on the
    panel verbatim, so they are written as instructions to a person standing at
    a table rather than as descriptions of an internal state.
    """

    state: SystemState = SystemState.STARTING
    headline: str = "Starting up"
    detail: str = ""
    #: How long the current state has been in force, seconds.
    since_seconds: float = 0.0
    #: Confidence of the last accepted table detection, for the panel.
    table_confidence: float = 0.0

    @property
    def playable(self) -> bool:
        return self.state is SystemState.READY

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "headline": self.headline,
            "detail": self.detail,
            "since_seconds": round(self.since_seconds, 1),
            "table_confidence": round(self.table_confidence, 3),
            "playable": self.playable,
        }


#: What each state says. Kept together so the wording can be read as a set --
#: these are the only words most users will ever see from this system, and they
#: should sound like one voice.
_COPY: dict[SystemState, tuple[str, str]] = {
    SystemState.STARTING: (
        "GhostBall",
        "Starting up",
    ),
    SystemState.NO_CAMERA: (
        "No camera",
        "The camera has stopped sending frames. Check the ribbon cable is seated "
        "at both ends, then restart.",
    ),
    SystemState.NO_TABLE: (
        "No pool table detected",
        "Mount the device above the middle of the table, pointing straight down, "
        "so the whole cloth is in view.",
    ),
    SystemState.READY: ("Ready", ""),
}


class ReadinessTracker:
    """Turns per-frame observations into a stable readiness state.

    One :meth:`observe` call per frame from the vision loop. Cheap on purpose:
    it runs inside the frame budget and does nothing but compare a couple of
    numbers.
    """

    def __init__(
        self,
        confirm_frames: int = CONFIRM_FRAMES,
        min_confidence: float = MIN_TABLE_CONFIDENCE,
        camera_silent_seconds: float = CAMERA_SILENT_SECONDS,
    ) -> None:
        self.confirm_frames = confirm_frames
        self.min_confidence = min_confidence
        self.camera_silent_seconds = camera_silent_seconds

        self._state = SystemState.STARTING
        self._since = time.perf_counter()
        self._pending: SystemState | None = None
        self._pending_count = 0
        self._confidence = 0.0
        #: Whether table detection has run at all yet. Until it has, "no table"
        #: is not a finding -- it is simply not knowing, and announcing it would
        #: flash a setup screen on every start.
        self._has_looked = False

    # -- observation --------------------------------------------------------

    def observe(
        self,
        boundary,
        *,
        frame_arrived: bool,
        last_frame_at: float | None,
        table_checked: bool = False,
        now: float | None = None,
    ) -> Readiness:
        """Fold one frame into the state. Returns the current readiness.

        Args:
            boundary: The cached :class:`~app.models.TableBoundary`, or ``None``.
            frame_arrived: Whether this pass had a frame to work with.
            last_frame_at: ``perf_counter`` of the most recent frame.
            table_checked: Whether table detection has ever run. Before it has,
                the absence of a boundary means nothing.
            now: Injectable clock, so the whole machine can be tested without
                sleeping.
        """
        now = time.perf_counter() if now is None else now
        if table_checked:
            self._has_looked = True

        confidence = float(getattr(boundary, "confidence", 0.0) or 0.0) if boundary else 0.0
        if boundary is not None and confidence >= self.min_confidence:
            self._confidence = confidence

        observed = self._classify(
            boundary, confidence, frame_arrived=frame_arrived, last_frame_at=last_frame_at, now=now
        )
        self._advance(observed, now)
        return self.current(now)

    def _classify(self, boundary, confidence, *, frame_arrived, last_frame_at, now) -> SystemState:
        """What this single frame suggests, before hysteresis."""
        # A dead camera outranks everything: with no frames there is nothing to
        # say about the table, and saying "no table detected" to someone whose
        # camera has fallen off the bus would send them to move a mount that is
        # already in the right place.
        if last_frame_at is None:
            return SystemState.STARTING if not frame_arrived else SystemState.NO_TABLE
        if now - last_frame_at > self.camera_silent_seconds:
            return SystemState.NO_CAMERA

        if not self._has_looked:
            return SystemState.STARTING
        if boundary is None or confidence < self.min_confidence:
            return SystemState.NO_TABLE
        return SystemState.READY

    def _advance(self, observed: SystemState, now: float) -> None:
        """Apply hysteresis, then commit."""
        if observed is self._state:
            self._pending = None
            self._pending_count = 0
            return

        if observed is not self._pending:
            self._pending = observed
            self._pending_count = 0
        self._pending_count += 1
        # Leaving STARTING is not gated: the first real observation is as good
        # as the thirtieth, and making the welcome screen linger for a second
        # after the table is already found would be a worse first impression
        # than showing it briefly.
        needed = 1 if self._state is SystemState.STARTING else self.confirm_frames
        if self._pending_count >= needed:
            logger.info(
                "readiness %s -> %s", self._state.value, observed.value
            )
            self._state = observed
            self._since = now
            self._pending = None
            self._pending_count = 0

    # -- reading ------------------------------------------------------------

    def current(self, now: float | None = None) -> Readiness:
        now = time.perf_counter() if now is None else now
        headline, detail = _COPY[self._state]
        return Readiness(
            state=self._state,
            headline=headline,
            detail=detail,
            since_seconds=max(0.0, now - self._since),
            table_confidence=self._confidence,
        )

    @property
    def state(self) -> SystemState:
        return self._state

    def reset(self) -> None:
        """Back to STARTING. For a camera reconnect, where everything the
        tracker believed was about a camera that is no longer attached."""
        self._state = SystemState.STARTING
        self._since = time.perf_counter()
        self._pending = None
        self._pending_count = 0
        self._has_looked = False


def table_detect_interval(state: SystemState, base_interval: int, attempts: int = 0) -> int:
    """How many frames to wait before looking for the table again.

    Not a constant, because the right answer differs by an order of magnitude
    between the two situations:

    * **Playing.** The table has been found and does not move. Re-checking is a
      slow guard against the camera being bumped, and the base interval is
      already generous.
    * **Looking.** Somebody is up a ladder aiming the camera and wants to know
      the moment it lands. Detection costs ~100 ms on a Pi, which is a real
      stall -- but paid every couple of seconds during setup, in exchange for
      feedback while they are still holding the bracket, it is clearly worth it.

    With a backoff, because the searching case is also what a rig pointed at a
    ceiling does forever, and burning a twentieth of a core on that for hours is
    not free. Doubles up to the base interval, so a genuinely table-less system
    settles at the same cost as a playing one.
    """
    if state is SystemState.READY:
        return base_interval
    if state is SystemState.NO_CAMERA:
        # No frames means nothing to detect in; do not spend anything on it.
        return base_interval
    fast = max(1, base_interval // 5)
    return min(base_interval, fast * (2 ** min(attempts, 4)))
