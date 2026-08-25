"""Exclusive access to the projector, and why a request for it was refused.

Calibration takes over the projector and suspends play. Two things follow from
that, and this module exists for the second one.

The first is obvious: entry has to be refused while a game is running.

The second is the whole point. **A refusal has to say which refusal it is.**
"Someone is mid-game" and "calibration is already running on another phone"
send the user to completely different places -- one to the table, one to find
whoever is holding the other phone -- and a bare 409 sends them to neither.
Worse, when two people are setting up a table and both get an unexplained
refusal, each concludes the other's phone is broken. So the holder is recorded
with an address and a start time, and the refusal names them.

Takeover is allowed, deliberately. A phone that goes flat or walks out of range
would otherwise hold the table hostage until the idle timer expires, and a
setup crew of two hitting that has no way forward except waiting. The previous
holder finds out on its next action, which is the right moment: it is told what
happened rather than silently failing.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = ["ExclusiveLock", "LockHolder", "RefusalReason", "Refusal"]

#: How long a holder can go silent before the lock is released. Long enough that
#: a phone locking its screen mid-step does not lose the wizard, short enough
#: that a phone carried out of the room does not strand the table.
DEFAULT_IDLE_TIMEOUT = 600.0


class RefusalReason(str, Enum):
    """Why exclusive access was refused. String-valued so it crosses the API.

    The panel branches on this rather than on the message text, so the wording
    can be improved without breaking the client.
    """

    #: A game is set up with players and a score worth protecting.
    GAME_IN_PROGRESS = "game_in_progress"
    #: Balls are moving, or somebody is mid-shot.
    TABLE_BUSY = "table_busy"
    #: Somebody else already has it.
    ALREADY_HELD = "already_held"
    #: Nothing is running, so there is nothing to take over.
    LOOP_NOT_RUNNING = "loop_not_running"


@dataclass(frozen=True, slots=True)
class LockHolder:
    """Who holds the lock, in terms a person can act on."""

    #: Best available address for the holder -- usually the client IP. This is
    #: what makes "already running elsewhere" actionable rather than infuriating.
    address: str = "unknown"
    #: Optional self-chosen name, so a household with three phones on DHCP can
    #: tell them apart without reading a lease table.
    label: str = ""
    acquired_at: float = field(default_factory=time.perf_counter)
    last_seen_at: float = field(default_factory=time.perf_counter)

    def describe(self) -> str:
        return f"{self.label} ({self.address})" if self.label else self.address

    def held_for(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        return max(0.0, now - self.acquired_at)

    def idle_for(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        return max(0.0, now - self.last_seen_at)


@dataclass(frozen=True, slots=True)
class Refusal:
    """A refusal, phrased for the person who has to do something about it."""

    reason: RefusalReason
    message: str
    #: Who is holding it, when that is the reason. ``None`` otherwise.
    holder: str | None = None
    holder_seconds: float | None = None
    #: Whether the caller could retry with ``force=True``. False for the cases
    #: where forcing would take a live game away from whoever is playing it.
    can_force: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "message": self.message,
            "holder": self.holder,
            "holder_seconds": (
                None if self.holder_seconds is None else round(self.holder_seconds, 1)
            ),
            "can_force": self.can_force,
        }


def _humanise(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)} seconds ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes ago"
    return f"{seconds / 3600:.1f} hours ago"


class ExclusiveLock:
    """Single-holder lock over the projector, with an explained refusal.

    Guarded by a real mutex despite the rest of ``AppState`` being lock-free.
    The lock-free pattern there works because the vision loop is the only writer
    of each field; this one is written by request handlers on the server's
    thread pool, so two phones tapping Start at the same instant genuinely race,
    and the outcome of that race is which of them owns the table.
    """

    def __init__(self, idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> None:
        self.idle_timeout = idle_timeout
        self._holder: LockHolder | None = None
        self._mutex = threading.Lock()
        #: Set when a holder is displaced, so its next action can be told what
        #: happened instead of failing with a bare "you do not hold this".
        self._displaced: dict[str, str] = {}

    # -- inspection ---------------------------------------------------------

    @property
    def holder(self) -> LockHolder | None:
        """Current holder, or ``None``. Expires an idle one as a side effect."""
        with self._mutex:
            self._expire_locked()
            return self._holder

    @property
    def is_held(self) -> bool:
        return self.holder is not None

    def _expire_locked(self, now: float | None = None) -> None:
        """Release a holder that has stopped talking to us. Caller holds the mutex."""
        if self._holder is None:
            return
        if self._holder.idle_for(now) > self.idle_timeout:
            logger.warning(
                "releasing calibration lock: %s has been silent for %.0fs",
                self._holder.describe(),
                self._holder.idle_for(now),
            )
            self._displaced[self._holder.address] = "released after going idle"
            self._holder = None

    # -- acquisition --------------------------------------------------------

    def acquire(
        self,
        address: str = "unknown",
        label: str = "",
        force: bool = False,
        blockers: list[Refusal] | None = None,
    ) -> Refusal | None:
        """Take the lock. Returns ``None`` on success, or why it was refused.

        Args:
            address: Where the request came from, for the refusal message.
            label: Optional friendly name for the holder.
            force: Take it from the current holder, and push past any blocker
                that declares itself forceable. A blocker with
                ``can_force=False`` -- a seated game -- is never overridden:
                losing somebody's game to a stray tap on another phone is worse
                than making them tap Reset.
            blockers: Conditions the caller has already evaluated, such as a
                game in progress. Checked before the lock itself, because
                "there is a game on" is a more useful thing to be told than
                "someone else is calibrating", even when both are true.
        """
        for blocker in blockers or []:
            if force and blocker.can_force:
                logger.warning("%s forced past: %s", address, blocker.reason.value)
                continue
            return blocker

        with self._mutex:
            self._expire_locked()

            if self._holder is not None and self._holder.address != address:
                if not force:
                    held = self._holder
                    return Refusal(
                        reason=RefusalReason.ALREADY_HELD,
                        message=(
                            f"Calibration is already running on {held.describe()}, started "
                            f"{_humanise(held.held_for())}. Continue there, or take over "
                            "from here."
                        ),
                        holder=held.describe(),
                        holder_seconds=held.held_for(),
                        can_force=True,
                    )
                logger.warning(
                    "calibration lock taken from %s by %s", self._holder.describe(), address
                )
                self._displaced[self._holder.address] = f"taken over by {address}"

            now = time.perf_counter()
            self._holder = LockHolder(
                address=address, label=label, acquired_at=now, last_seen_at=now
            )
            self._displaced.pop(address, None)
            logger.info("calibration lock acquired by %s", self._holder.describe())
            return None

    def touch(self, address: str) -> bool:
        """Record that ``address`` is still there. Returns whether it still holds.

        Called on every read as well as every action -- a phone showing the
        wizard is still using it even when nobody is tapping, and expiring it
        mid-step because the user was up a ladder would be its own bug.
        """
        with self._mutex:
            self._expire_locked()
            if self._holder is None or self._holder.address != address:
                return False
            self._holder = LockHolder(
                address=self._holder.address,
                label=self._holder.label,
                acquired_at=self._holder.acquired_at,
                last_seen_at=time.perf_counter(),
            )
            return True

    def release(self, address: str | None = None) -> bool:
        """Give the lock up. ``None`` releases whoever holds it.

        Returns whether anything was released, so a double release -- the wizard
        finishing and the client also cancelling -- is not an error.
        """
        with self._mutex:
            if self._holder is None:
                return False
            if address is not None and self._holder.address != address:
                return False
            logger.info("calibration lock released by %s", self._holder.describe())
            self._holder = None
            return True

    def displacement_notice(self, address: str) -> str | None:
        """Why ``address`` no longer holds the lock, if it was displaced.

        Consumed once. The point is that a phone whose action fails learns it
        was taken over rather than seeing a generic error and concluding the Pi
        broke -- which is the same failure mode as an unexplained 409, one level
        down.
        """
        with self._mutex:
            return self._displaced.pop(address, None)


def game_in_progress_refusal(session) -> Refusal | None:
    """Refuse calibration when there is a game worth protecting.

    Two distinct conditions with two distinct instructions, which is the reason
    this returns a typed refusal rather than a bool: a seated game needs
    resetting from the panel, while a table still settling just needs a moment.
    """
    from app.models import SessionState

    if session.players:
        names = ", ".join(p.name for p in session.players)
        return Refusal(
            reason=RefusalReason.GAME_IN_PROGRESS,
            message=(
                f"A game of {session.mode.value.replace('_', ' ')} is set up ({names}). "
                "Calibration takes over the projector, so finish it or tap Reset session "
                "on the Mode card first."
            ),
            # Deliberately not forceable. Losing somebody's game to a stray tap
            # on another phone is worse than making them tap Reset.
            can_force=False,
        )

    if session.state not in (SessionState.IDLE, SessionState.GAME_OVER):
        return Refusal(
            reason=RefusalReason.TABLE_BUSY,
            message=(
                f"The table is busy ({session.state.value.replace('_', ' ')}). "
                "Wait for the balls to stop, or start anyway."
            ),
            # Forceable, unlike a seated game. There is no score to lose here --
            # and detection can get stuck reporting movement (a draught on a
            # light, a reflection), which would otherwise leave no way into the
            # wizard at all. An unforceable refusal you cannot clear is a trap.
            can_force=True,
        )
    return None
