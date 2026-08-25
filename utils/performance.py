"""FPS and latency instrumentation.

The two hard numbers in the spec are 30 FPS and sub-100 ms camera-to-projector
latency, and neither is meaningful as an average over a whole session -- a run
that averages 30 FPS while stuttering to 8 FPS on every shot is a failure. So
this module keeps a rolling window and reports percentiles, and
:class:`StageTimer` attributes the time to a named pipeline stage so a
regression can be traced to the stage that caused it.

All timing uses :func:`time.perf_counter`. It is monotonic, so NTP stepping the
clock mid-game cannot show up as a negative frame time, and it is
high-resolution on every platform -- unlike :func:`time.monotonic`, whose
granularity on Windows is ~15.6 ms. At a 33 ms frame budget that quantises
every measurement to 0, 15.6 or 31.2 ms, which makes frame timing worthless for
development on a non-Linux box.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Rolling window length. At 30 FPS, 90 frames is the last 3 seconds -- long
#: enough to be stable, short enough that the panel reacts to a stall.
DEFAULT_WINDOW = 90


@dataclass(slots=True)
class FrameRecord:
    """One frame's timings, handed to :attr:`PerformanceTracker.frame_sink`.

    Separate from :class:`PerfSnapshot` because the two answer different
    questions. A snapshot is the rolling average -- what the panel shows and
    what the periodic log line reports. A record is a single frame, and exists
    so :class:`FrameProfiler` can write a trace you can open in a spreadsheet
    and find the one frame in nine thousand that took 400 ms.
    """

    index: int
    #: ``time.perf_counter()`` at the end of the frame.
    at: float
    total_ms: float
    #: End-to-end camera-to-projector latency, when a capture timestamp was
    #: given to :meth:`PerformanceTracker.end_frame`.
    latency_ms: float
    #: Stage name -> ms, for this frame only. Stages that did not run this
    #: frame are absent rather than zero -- "the renderer was skipped" and "the
    #: renderer took no time" are different facts.
    stages: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PerfSnapshot:
    """Point-in-time performance summary, as served by ``GET /api/status``."""

    fps: float = 0.0
    frame_ms_avg: float = 0.0
    frame_ms_p95: float = 0.0
    frame_ms_max: float = 0.0
    latency_ms: float = 0.0
    dropped_frames: int = 0
    total_frames: int = 0
    #: Mean ms **per invocation** of each stage. Not per frame: stages that run
    #: on an interval are timed only on the frames they ran.
    stage_ms: dict[str, float] = field(default_factory=dict)
    #: Fraction of recent frames each stage ran on, 0.0-1.0.
    stage_coverage: dict[str, float] = field(default_factory=dict)
    #: Mean ms **per frame**, i.e. ``stage_ms * stage_coverage``. The number
    #: that actually determines frame rate, and the one to compare stages by.
    #:
    #: Both are kept because they answer different questions and conflating
    #: them is actively misleading. Table detection measured 98.8 ms per
    #: invocation and looked like the most expensive stage in the pipeline --
    #: while running once every 600 frames, for an amortised 0.16 ms, roughly
    #: a four-hundredth of what capture costs on every single frame.
    stage_amortised_ms: dict[str, float] = field(default_factory=dict)


class PerformanceTracker:
    """Rolling-window frame timing.

    Usage per frame::

        tracker.begin_frame()
        with tracker.stage("detect"):
            ...
        tracker.end_frame()
    """

    def __init__(self, window: int = DEFAULT_WINDOW, target_fps: int = 30) -> None:
        self.window = window
        self.target_fps = target_fps
        #: Budget per frame in ms. Exceeding it means we cannot hold target FPS.
        self.frame_budget_ms = 1000.0 / target_fps
        self._frame_times: deque[float] = deque(maxlen=window)
        self._stage_times: dict[str, deque[float]] = {}
        self._frame_start: float | None = None
        self._latest_latency_ms = 0.0
        self.total_frames = 0
        self.dropped_frames = 0
        #: For each stage, the frame numbers it ran on. Needed because a
        #: stage's own deque of durations says nothing about how often it ran:
        #: a stage invoked once per 600 frames still fills its deque, it just
        #: takes 54,000 frames to do it.
        #:
        #: Per-frame stage times, cleared by :meth:`begin_frame`. Kept
        #: separately from the rolling deques rather than derived from them: a
        #: deque's last element is not necessarily *this* frame's, since a
        #: stage that ran two frames ago is still at the end of its own deque.
        self._frame_stages: dict[str, float] = {}
        self._stage_frames: dict[str, deque[int]] = {}
        #: ``perf_counter`` at the last :meth:`end_frame`, or ``None`` before
        #: the first. The heartbeat a watchdog reads; see :class:`LoopWatchdog`.
        self._last_frame_at: float | None = None
        #: Optional per-frame callback, set by :class:`FrameProfiler`. Called
        #: inside :meth:`end_frame`, so it runs on the loop thread and inside
        #: the frame budget -- it must be cheap. ``None`` by default: tracing
        #: every frame costs real time, so it is opt-in.
        self.frame_sink: Callable[[FrameRecord], None] | None = None

    # -- frame accounting ---------------------------------------------------

    def begin_frame(self) -> None:
        """Mark the start of a frame's processing."""
        self._frame_start = time.perf_counter()
        self._frame_stages = {}

    def end_frame(self, capture_timestamp: float | None = None) -> float:
        """Close out the frame and return its processing time in ms.

        Args:
            capture_timestamp: ``time.perf_counter()`` reading from when the frame
                was grabbed off the sensor. When given, end-to-end latency is
                measured from there rather than from :meth:`begin_frame`, which
                is the number that actually matters -- it includes time the
                frame spent queued before we started work on it.
        """
        if self._frame_start is None:
            logger.debug("end_frame() without begin_frame(); ignoring")
            return 0.0

        now = time.perf_counter()
        frame_ms = (now - self._frame_start) * 1000.0
        self._frame_times.append(frame_ms)
        self.total_frames += 1

        if capture_timestamp is not None:
            self._latest_latency_ms = (now - capture_timestamp) * 1000.0
        else:
            self._latest_latency_ms = frame_ms

        if frame_ms > self.frame_budget_ms:
            self.dropped_frames += 1

        self._frame_start = None
        self._last_frame_at = now

        if self.frame_sink is not None:
            record = FrameRecord(
                index=self.total_frames,
                at=now,
                total_ms=frame_ms,
                latency_ms=self._latest_latency_ms,
                stages=dict(self._frame_stages),
            )
            try:
                self.frame_sink(record)
            except Exception:  # noqa: BLE001
                # A broken profiler must not take the game down. Drop the sink
                # rather than logging once per frame for the rest of the run.
                logger.exception("frame sink failed; disabling per-frame profiling")
                self.frame_sink = None

        return frame_ms

    @property
    def last_frame_at(self) -> float | None:
        """``perf_counter`` at the last completed frame, or ``None``.

        The watchdog's input. Deliberately distinct from ``GameState.timestamp``,
        which is when the frame was *captured*: a loop wedged inside the render
        stage has a recent capture timestamp and a stale value here, and it is
        the stale one that identifies the wedge.
        """
        return self._last_frame_at

    def seconds_since_frame(self, now: float | None = None) -> float | None:
        """Seconds since the last completed frame, or ``None`` before the first."""
        if self._last_frame_at is None:
            return None
        now = time.perf_counter() if now is None else now
        return max(0.0, now - self._last_frame_at)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named pipeline stage (``capture``, ``detect``, ``render``...)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            bucket = self._stage_times.get(name)
            if bucket is None:
                bucket = self._stage_times[name] = deque(maxlen=self.window)
            bucket.append(elapsed_ms)
            # Accumulate rather than assign: a stage entered twice in one frame
            # should report its total cost for the frame, not its last visit.
            if name not in self._frame_stages:
                seen = self._stage_frames.get(name)
                if seen is None:
                    seen = self._stage_frames[name] = deque(maxlen=self.window)
                seen.append(self.total_frames)
            self._frame_stages[name] = self._frame_stages.get(name, 0.0) + elapsed_ms

    # -- reporting ----------------------------------------------------------

    @property
    def fps(self) -> float:
        """Achieved FPS over the window, derived from mean frame time.

        This is processing throughput, not wall-clock frame rate: if the loop
        sleeps to hold 30 FPS, this reports the rate it *could* sustain.
        """
        if not self._frame_times:
            return 0.0
        mean_ms = sum(self._frame_times) / len(self._frame_times)
        return 1000.0 / mean_ms if mean_ms > 0 else 0.0

    def percentile(self, pct: float) -> float:
        """Frame time in ms at the given percentile (0-100) over the window."""
        if not self._frame_times:
            return 0.0
        ordered = sorted(self._frame_times)
        # Nearest-rank; the window is small enough that interpolating adds
        # nothing but a chance to get the edge cases wrong.
        idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
        return ordered[idx]

    def stage_coverage(self, name: str) -> float:
        """Fraction of the recent window's frames that ``name`` ran on."""
        seen = self._stage_frames.get(name)
        if not seen or self.total_frames == 0:
            return 0.0
        span = min(self.window, self.total_frames)
        oldest = self.total_frames - span
        recent = sum(1 for frame in seen if frame >= oldest)
        return min(1.0, recent / span)

    def snapshot(self) -> PerfSnapshot:
        """Current metrics, for the API and the periodic log line."""
        times = list(self._frame_times)
        coverage = {
            name: round(self.stage_coverage(name), 4)
            for name, vals in self._stage_times.items()
            if vals
        }
        means = {
            name: sum(vals) / len(vals)
            for name, vals in self._stage_times.items()
            if vals
        }
        return PerfSnapshot(
            fps=round(self.fps, 1),
            frame_ms_avg=round(sum(times) / len(times), 2) if times else 0.0,
            frame_ms_p95=round(self.percentile(95), 2),
            frame_ms_max=round(max(times), 2) if times else 0.0,
            latency_ms=round(self._latest_latency_ms, 2),
            dropped_frames=self.dropped_frames,
            total_frames=self.total_frames,
            stage_ms={name: round(value, 2) for name, value in means.items()},
            stage_coverage=coverage,
            stage_amortised_ms={
                name: round(value * coverage.get(name, 0.0), 3)
                for name, value in means.items()
            },
        )

    def log_summary(self, latency_warn_ms: float = 100.0) -> None:
        """Emit one summary line, at WARNING if we are missing our targets.

        Escalating the level is the point: on a headless Pi the log is the only
        signal, and a stall needs to stand out from the INFO stream.
        """
        snap = self.snapshot()
        # Stages that do not run every frame are annotated with what they
        # actually cost per frame. Without it the line invites a direct
        # comparison between a per-frame stage and an interval one, which is
        # how table detection came to look like the most expensive thing in the
        # pipeline while contributing a sixth of a millisecond.
        parts = []
        for name, mean in sorted(snap.stage_ms.items()):
            share = snap.stage_coverage.get(name, 1.0)
            if share >= 0.95:
                parts.append(f"{name}={mean:.1f}")
            elif share <= 0.0:
                # The mean is real but it is from *outside* the window, so
                # pairing it with "0.00 ms/frame" would read as a stage that
                # costs nothing rather than one that has not run lately.
                parts.append(f"{name}={mean:.1f}[not in the last {self.window} frames]")
            else:
                parts.append(
                    f"{name}={mean:.1f}[{share * 100:.0f}% of frames, "
                    f"{snap.stage_amortised_ms.get(name, 0.0):.2f}ms/frame]"
                )
        stages = " ".join(parts)
        message = (
            "perf fps=%.1f frame_avg=%.1fms p95=%.1fms latency=%.1fms "
            "dropped=%d/%d %s"
        )
        args = (
            snap.fps,
            snap.frame_ms_avg,
            snap.frame_ms_p95,
            snap.latency_ms,
            snap.dropped_frames,
            snap.total_frames,
            stages,
        )
        missing_target = (
            snap.fps < self.target_fps * 0.9 or snap.latency_ms > latency_warn_ms
        )
        logger.log(logging.WARNING if missing_target else logging.INFO, message, *args)

    def reset(self) -> None:
        """Clear all counters. Used when switching modes so stats are per-mode.

        Leaves :attr:`frame_sink` attached. A profile run spans mode switches,
        and silently stopping the trace halfway would be worse than useless --
        the file would look complete.
        """
        self._frame_times.clear()
        self._stage_times.clear()
        self._stage_frames.clear()
        self._frame_stages = {}
        self.total_frames = 0
        self.dropped_frames = 0
        self._latest_latency_ms = 0.0


class RateLimiter:
    """Holds a loop to a target rate without drifting.

    A naive ``sleep(1/30)`` per iteration yields *less* than 30 FPS, because the
    work time adds to the sleep. This sleeps only the remainder of the frame
    budget and reports overruns instead of silently falling behind.
    """

    def __init__(self, target_fps: int = 30) -> None:
        self.interval = 1.0 / target_fps
        self._next_tick = time.perf_counter()

    def sleep(self) -> float:
        """Block until the next tick. Returns the overrun in seconds (0 if on time)."""
        self._next_tick += self.interval
        remaining = self._next_tick - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
            return 0.0
        # Behind schedule. Resync rather than trying to claw back the deficit,
        # which would spin the loop with no sleep and starve the web server.
        overrun = -remaining
        self._next_tick = time.perf_counter()
        return overrun

    async def async_sleep(self) -> float:
        """Async form of :meth:`sleep`, for the asyncio main loop."""
        import asyncio

        self._next_tick += self.interval
        remaining = self._next_tick - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
            return 0.0
        overrun = -remaining
        self._next_tick = time.perf_counter()
        # Yield anyway, or a persistently overrunning loop never lets the
        # FastAPI handlers run and the control panel appears hung.
        await asyncio.sleep(0)
        return overrun


class FrameProfiler:
    """Writes one CSV row per frame: total, latency, and every stage.

    The aggregate numbers in :class:`PerfSnapshot` answer "are we holding 30
    FPS". They cannot answer "what happened at 14:02", and on a system that has
    to run for hours that is the question that matters -- a stall every few
    minutes averages away completely, and it is exactly the symptom a player
    notices.

    Opt-in (``--profile FILE``) because it costs a row of I/O per frame. The
    write is buffered by the file object and flushed on an interval rather than
    per row: an unflushed trace is worthless after a hard kill, and flushing
    108,000 times an hour is not free either.

    Columns are fixed at construction from ``stage_names``, because a CSV whose
    column set changes halfway down the file is not a CSV. A stage that appears
    later -- a mode that only renders once someone starts a game -- lands in the
    ``extra_ms`` column rather than being silently dropped.
    """

    #: Stages the pipeline can report, in pipeline order. Ordered rather than
    #: sorted so the trace reads left to right the way the frame runs.
    DEFAULT_STAGES = ("capture", "table", "detect", "physics", "mode", "render", "project")

    def __init__(
        self,
        path: Path | str,
        stage_names: tuple[str, ...] | None = None,
        flush_every: int = 60,
    ) -> None:
        self.path = Path(path)
        self.stages = tuple(stage_names or self.DEFAULT_STAGES)
        self.flush_every = max(1, flush_every)
        self._file = None
        self._writer: csv.writer | None = None  # type: ignore[valid-type]
        self._rows = 0
        #: perf_counter at the first row, so ``t_s`` is seconds into the run
        #: rather than an arbitrary epoch-less float nobody can interpret.
        self._origin: float | None = None

    def __enter__(self) -> FrameProfiler:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def open(self) -> FrameProfiler:
        """Create the file and write the header row."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" per the csv module's contract: without it, the writer's
        # own \r\n is translated again on Windows and every other line is blank.
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["frame", "t_s", "total_ms", "latency_ms", *[f"{s}_ms" for s in self.stages], "extra_ms"]
        )
        logger.info("per-frame profile -> %s", self.path)
        return self

    def record(self, frame: FrameRecord) -> None:
        """Write one frame. Suitable as a :attr:`PerformanceTracker.frame_sink`."""
        if self._writer is None:
            return
        if self._origin is None:
            self._origin = frame.at
        known = 0.0
        cells = []
        for name in self.stages:
            value = frame.stages.get(name)
            if value is None:
                cells.append("")
            else:
                known += value
                cells.append(f"{value:.3f}")
        extra = sum(v for k, v in frame.stages.items() if k not in self.stages)
        self._writer.writerow(
            [
                frame.index,
                f"{frame.at - self._origin:.4f}",
                f"{frame.total_ms:.3f}",
                f"{frame.latency_ms:.3f}",
                *cells,
                f"{extra:.3f}" if extra else "",
            ]
        )
        self._rows += 1
        if self._rows % self.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        """Flush and close. Safe to call more than once."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            logger.info("wrote %d profiled frames to %s", self._rows, self.path)
        self._file = None
        self._writer = None

    def attach(self, tracker: PerformanceTracker) -> FrameProfiler:
        """Open the file and wire this profiler into ``tracker``."""
        self.open()
        tracker.frame_sink = self.record
        return self


class LoopWatchdog:
    """Notices when the vision loop stops producing frames.

    The failure this exists for is not a crash -- a crashed loop is obvious from
    the logs and from ``running: false``. It is the loop that is *alive and
    stuck*: blocked in a camera read that never returns, or wedged in a driver
    call. From outside, that looks exactly like a healthy idle system, and over
    a session measured in hours it is the failure mode you actually hit.

    So this runs on its own thread, reads
    :attr:`PerformanceTracker.last_frame_at`, and fires when it goes stale. It
    does not attempt a recovery: there is no safe way to interrupt a thread
    blocked in native code, and killing the process on a false positive would be
    a worse bug than the stall. It logs, sets a flag the panel and ``/health``
    surface, and calls an optional callback.
    """

    def __init__(
        self,
        tracker: PerformanceTracker,
        stall_seconds: float = 5.0,
        poll_seconds: float = 1.0,
        on_stall: Callable[[float], None] | None = None,
        on_recover: Callable[[], None] | None = None,
    ) -> None:
        self.tracker = tracker
        self.stall_seconds = stall_seconds
        self.poll_seconds = poll_seconds
        self.on_stall = on_stall
        self.on_recover = on_recover
        #: Whether the loop is currently considered stalled.
        self.stalled = False
        #: How many distinct stalls have been seen this run. A single long stall
        #: counts once -- the panel needs "this has happened four times", not a
        #: number that climbs by one per poll while nothing new is wrong.
        self.stall_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def check(self, now: float | None = None) -> bool:
        """One poll. Returns whether the loop is stalled. Exposed for tests."""
        now = time.perf_counter() if now is None else now
        idle = self.tracker.seconds_since_frame(now)
        # Before the first frame there is nothing to be stale about. A loop that
        # never starts is reported by the startup path, not by this.
        stalled = idle is not None and idle > self.stall_seconds

        if stalled and not self.stalled:
            self.stall_count += 1
            self.stalled = True
            logger.error(
                "vision loop stalled: no frame for %.1fs (stall #%d)", idle, self.stall_count
            )
            if self.on_stall is not None:
                self.on_stall(idle)
        elif not stalled and self.stalled:
            self.stalled = False
            logger.warning("vision loop recovered after stall #%d", self.stall_count)
            if self.on_recover is not None:
                self.on_recover()
        return stalled

    def start(self) -> LoopWatchdog:
        """Begin polling on a daemon thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        # Event.wait rather than sleep, so stop() is immediate rather than
        # taking up to a full poll interval on shutdown.
        while not self._stop.wait(self.poll_seconds):
            try:
                self.check()
            except Exception:  # noqa: BLE001
                logger.exception("watchdog check failed")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop polling and join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


def get_system_metrics() -> dict[str, float | None]:
    """CPU, memory and SoC temperature, for the web panel's system section.

    ``psutil`` is optional -- a missing dependency should degrade the panel, not
    break the API -- so every value can come back ``None``.
    """
    metrics: dict[str, float | None] = {
        "cpu_pct": None,
        "mem_pct": None,
        "temp_c": None,
    }
    try:
        import psutil
    except ImportError:
        return metrics

    # interval=None returns the value since the last call, which is
    # non-blocking. A blocking sample here would stall the API handler.
    metrics["cpu_pct"] = psutil.cpu_percent(interval=None)
    metrics["mem_pct"] = psutil.virtual_memory().percent

    # Pi 5 exposes the SoC temperature; the sensor name differs by kernel
    # version, and on non-Linux there is no sensors_temperatures at all.
    sensors = getattr(psutil, "sensors_temperatures", None)
    if sensors is not None:
        try:
            for readings in (sensors() or {}).values():
                if readings:
                    metrics["temp_c"] = readings[0].current
                    break
        except (OSError, NotImplementedError):
            pass
    return metrics
