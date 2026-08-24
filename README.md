# AR Pool Table

Projection-mapped AR pool table for Raspberry Pi 5. Camera watches the table, physics
predicts the shot, projector draws the prediction onto the felt.

**Complete and playable.** The system captures frames, finds the table on any
cloth colour, detects and tracks balls, reads cue aim, drives the shot state
machine, simulates the shot, renders the overlays, projects them, and plays five
game modes. A first-time user is walked through projector alignment by a
seven-screen wizard. Measured end to end on an x86 dev box: **29.9 FPS** with a
16 ms average frame.

The only thing left unbuilt is optional Hailo NPU inference, which the OpenCV
path already covers.

Frame budget on an x86 dev box, against 33 ms for 30 FPS:

| Stage | Cost |
|---|---|
| Capture | ~1.5 ms |
| Table detection (felt) | ~2 ms, every 150 frames (~0.01 ms amortised) |
| Table detection (pockets) | ~22 ms, every 150 frames (~0.15 ms amortised) |
| Ball / cue / pocket detection | ~7 ms |
| Physics (cached) | ~0.1 ms |
| Mode logic + overlay | ~2 ms |
| Projector output | ~6 ms |
| **Total** | **~16 ms** |

Measured in the live loop with pocket detection on: **29.9 FPS**, 7.3 ms average
frame, with the 22 ms detection spike arriving once every five seconds.

Measured accuracy against synthetic ground truth, across 11 scenarios spanning
keystone, rotation to 20°, shadows, sensor noise, vignetting and projected
overlay light:

| Metric | Target | Worst measured |
|---|---|---|
| Table corner error (felt) | < 20 px | **11.3 px** |
| Table corner error (pockets) | < 20 px | **4.5 px** |
| Table size, measured from a ball | — | **±8%** (±2% with a low camera) |
| Ball position error | < 3 in | **0.35 in** |
| Cue aim error | — | **0.8°** |
| Balls found / colours | 8/8 | **8/8 in every scenario** |
| Per-frame detection | < 33 ms | **~9 ms** (x86 dev box) |

Physics is verified against closed-form results rather than golden values:

| Check | Result |
|---|---|
| 90° rule (cut angles 15–75°) | separation exact to **0.01°** |
| Newton's cradle (full-ball hit) | cue ball stops dead, all speed transferred |
| Free-roll distance | matches `s²/(2a)` to **0.05 in** |
| Full 15-ball rack simulation | **1.5 ms** |
| Single-ball aiming line | **0.4 ms** |

## Quick start

Requires **Python 3.10+** (Raspberry Pi OS Bookworm ships 3.11). See the
deviations section.

```bash
python3 -m venv --system-site-packages .venv   # --system-site-packages so apt's picamera2 stays importable
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m tools.camera_preview     # check the camera FIRST, before anything else
python -m calibration_ui.calibration_app   # align the projector (do this second)
python launcher.py --mock          # no hardware needed
python launcher.py                 # real camera + projector
python -m pytest tests/ -q
```

`launcher.py` is the way to start the system. It runs a preflight check, then
hands over to `app.main`:

```
  Preflight
  ----------------------------------------------------
  + python        3.11.9 at /home/pi/ar_pool_table/.venv/bin/python
  + dependencies  fastapi, uvicorn, pydantic, numpy, cv2, yaml
  + config        config.yaml: 7ft table, 30 FPS target, classic theme
  + data dirs     /home/pi/ar_pool_table/data
  + camera        picamera2, v4l2 (/dev/video0)
  ! calibration   none saved -- overlays will not line up with the felt
                    -> python -m calibration_ui.calibration_app
  + display       1920x1080 on :0
  x web port      0.0.0.0:8000 is already in use (Address already in use)
                    -> Something is already listening -- most likely another copy
                       of this app. Stop it, or pick another port: --port 8001

  Not starting: 1 check(s) failed (web port).
```

Every failure it catches is one that otherwise surfaces as a traceback several
imports deep, or — worse — as a system that starts, looks healthy, and projects
something wrong. Each check reports what to **do**, not just what is wrong.
It also prints the LAN address, because the panel is meant to be used from a
phone at the table and `localhost:8000` is useless there.

It runs from any working directory. `python -m app.main` does not: the project
is imported as top-level packages (`app`, `vision`, `physics`…), so it needs the
cwd set to `ar_pool_table/`, and getting that wrong gives you
`ModuleNotFoundError: No module named 'app'`. The launcher puts its own
directory on `sys.path`, which is what a systemd unit or an SSH one-liner needs.

```bash
python launcher.py --check             # preflight only, then exit (exit 1 on failure)
python launcher.py --force             # start even if a check failed
python launcher.py --skip-checks       # straight through, no preflight

python launcher.py --headless          # vision loop only, no web server
python launcher.py --no-loop           # web panel only, no camera
python launcher.py --frames 500        # bounded run, for smoke tests
python launcher.py --profile run.csv   # one CSV row per frame, per stage
```

Every `app.main` flag works on the launcher, because it imports that parser
rather than declaring a second copy. `python -m app.main` remains a supported
way in — it is what the tests use — it just skips the checks.

Then open <http://localhost:8000/>.

### Then calibrate the projector

```bash
python -m calibration_ui.calibration_app
```

Seven screens, and a first-timer should be through them in a few minutes. Two
windows open: a **console** showing the camera's view of the table with the
wizard's annotations, and the **projector** output on the felt. The whole task
is making those two agree.

The step that matters is corner mapping, and there are three ways through it
because each fails somewhere different:

| Mode | What you do | When to use it |
|---|---|---|
| **Auto-Adjust** | Press one button | Default. Blanks the projector, photographs the felt, projects four marks, photographs again, and takes the four centroids out of the difference |
| **Manual Adjust** | Tap each mark on the console | Auto picked up a reflection, or the room is bright enough that the difference image is marginal |
| **Arrow keys + Record** | Walk a mark onto the cushion nose, press Record | The camera cannot see the projected light at all. Always works |

Everything is driveable by keyboard, mouse or touch — every shortcut is also an
on-screen button, because the projector window shares HighGUI's key queue and
can occasionally swallow a keystroke.

Two things about it worth knowing before you trust the numbers:

**Line the marks up with the cushion nose**, not the pocket jaw or the rail
edge. Table coordinates are defined at the inside of the cushion, so any other
reference point adds a constant offset that fine-tuning will chase forever and
never remove.

**The corner RMSE is not a measure of accuracy.** Four correspondences give an
exact fit, so the reported reprojection error is near zero whether or not the
transform is right anywhere else on the table. Screen 6 is the one that can
actually fail: it projects a ring at each detected ball, photographs the felt,
and measures how far each ring landed from its ball. The completion screen gates
on that, on grid squareness and on coverage — not on the RMSE.

On finish it writes `data/calibration/projector_calibration.json`, which is what
the application loads, plus a `camera_calibration.yaml` /
`projector_calibration.yaml` / `calibration_timestamp.txt` report which nothing
loads and which each say so in a header.

### Check the camera first

On new hardware, run this before anything else:

```bash
python -m tools.camera_preview --report --seconds 10   # can the camera hold 30 FPS?
python -m tools.camera_preview                         # live window, focus check
python -m tools.camera_preview --mask                  # tune the felt thresholds
python -m tools.camera_preview --headless              # over SSH: writes JPEGs
```

`--mask` tunes the felt thresholds. Table detection no longer depends on them
by default -- the pocket-based path ignores cloth colour entirely -- but **ball**
detection still inverts the felt mask to find balls, so `felt_hue_range`,
`felt_sat_min` and `felt_val_min` remain worth getting right. On non-green cloth
the adaptive fallback covers it; tuning here is faster than the fallback. The tool shows the live mask with a coverage
percentage (aim for 55–80% on a well-framed overhead shot) and a box around the
largest region found — which is what detection will find too. `[` / `]` and
`-` / `=` adjust the thresholds live; `p` prints them as YAML to paste into
`config.yaml`. Tuning here takes minutes; guessing and then debugging detection
takes hours.

It refuses to give an FPS verdict on synthetic frames, and warns loudly when it
falls back to the mock camera — thresholds tuned against a fake table are worse
than no thresholds at all.

| Flag | Effect |
|---|---|
| `--mock` | Synthetic camera, projector output discarded |
| `--no-loop` | Serve the panel only |
| `--frames N` | Stop the loop after N frames (smoke tests) |
| `--config PATH` | Use a different `config.yaml` |

Tuning lives in [`config.yaml`](config.yaml); fields worth revisiting on a new
table are marked `TUNE`. Values are validated once at startup, so a bad one
fails with the field name rather than causing a mysterious detection bug later.

## What works, what doesn't

Implemented and tested (487 tests, no hardware required):

- **Config** — Pydantic-validated YAML, table presets, physical constants
- **Domain models** — the objects every layer passes around
- **Camera** — picamera2 / OpenCV / synthetic backends with automatic fallback
- **Display** — full-screen OpenCV / mock backends, RGBA→BGR flattening
- **Coordinate transforms** — camera↔table and table↔projector homographies,
  solving, persistence, degeneracy rejection
- **Shot state machine** — `idle → aiming → shot_in_progress → settling`
- **Instrumentation** — FPS, percentile frame times, per-stage timing
- **REST API + control panel** — every endpoint returns real values; status,
  mode, settings, calibration, training, projector control
- **Camera preview endpoint** — JPEG of the live frame, optionally with the
  projection warped into camera space and blended over it
- **Projected test patterns from the panel** — align a projector from a
  phone, which a CLI cannot do while your hands are on the projector
- **Training score curve**
- **Camera preview / threshold tuning tool** (`tools/camera_preview.py`)
- **Table detection, two ways** — pocket-based (any cloth colour, measures the
  table in feet) with felt segmentation as the fallback; rail fitting and corner
  reconstruction shared by both
- **Ball detection** — position, colour, stripe/solid, confidence
- **Cue ball and cue stick** — aim direction in table space
- **Pocket location** — geometry-derived, image-refined
- **Ball tracking** — stable ids and velocities across frames
- **Shot simulation** — event-driven, exact contact points, no tunnelling
- **Cushion rebound** — restitution plus rail grip, and side-spin throw
- **Pocket capture, ghost-ball and pot-aim helpers**
- **Cross-frame prediction cache** — 77% hit rate while aiming
- **Overlay rendering** — trajectory, training, calibration and game-UI
  overlays; dashed animated aiming line, impact markers with cut angles,
  ghost balls, pocket highlights
- **Animation system** — ball trails, collision bursts, pocket vortex, score
  popups, combo badge, countdown; eased, table-anchored, clock-driven
- **Themes** — `classic`, `neon`, `dark_mode`, `pro`, switchable over the API
- **Test patterns + render profiler** (`tools/projection_test.py`)
- **Calibration wizard** (`calibration_ui/`) — seven screens, three ways to place
  a corner, live error in inches with a physical instruction, and an end-to-end
  check measured against photographed light
- **Five game modes** (`modes/`) — freeplay, classic 8-ball, king of the hill,
  trick shots and training, over a shared state machine, scoring layer and
  overlay composer
- **Synthetic test harness** (`tests/synthetic.py`) — images with exact ground truth

Stubbed, each raising `NotImplementedError` with a note on the intended approach:

| Stage | Where | Phase |
|---|---|---|
| Hailo inference (optional) | `vision/inference.py` | 2.4 |

`GET /api/status` reports the pending list, and the panel displays it — so a
blank projection explains itself instead of looking like a crash. Endpoints that
depend on a pending stage return **503 with the stage name**, distinguishing "not
built yet" from "broken", and the panel greys out the modes that cannot be
loaded rather than letting you tap one and quietly get freeplay.

### Control panel

One self-contained HTML file at `/`, no CDN — the Pi's LAN may have no internet.
Polls `/api/status` at 2 Hz; the camera preview is off by default and refreshes
at 1 Hz, because each request warps, resizes and JPEG-encodes on the same cores
the vision loop needs.

Two things about it are structural rather than cosmetic:

**The web layer never draws.** A full-screen OpenCV window belongs to the thread
that created it, which is the vision loop. So "blank the projector" and "project
the grid pattern" are *recorded* on `AppState` and picked up by the loop on its
next pass — 33 ms later, imperceptible to a thumb on a phone. Calling
`display.send_frame` from a request handler ranges from an ignored repaint to a
segfault depending on the platform's GUI backend.

**The preview warps before it blends.** The overlay is in projector pixels and
the frame is in camera pixels; the two are the same *size* by default, so
blending them directly yields an image that looks plausible and is wrong. The
preview composes projector → table → camera first, which is what makes it the
view the calibration wizard is built around.

The loop is closed: `mode_manager.update()` returns an RGBA overlay every frame
and the display layer puts it on the felt. `tools/projection_test.py` still
drives the renderer directly, which is the faster way to iterate on a single
overlay without a table in front of you.

### Render cost

Measured at 1080p on an x86 dev box (a Pi 5 will be slower, but the ratios
hold), against a 33 ms frame budget:

| Stage | ms | % budget |
|---|---|---|
| Canvas clear | 0.25 | 0.8% |
| Trajectory overlay | 3.7 – 6.4 | 11 – 19% |
| Game UI + effects | 4.9 – 7.6 | 15 – 23% |
| Ball trails | 1.7 – 2.1 | 5 – 6% |
| Test pattern (grid) | 2.5 – 4.0 | 8 – 12% |

Ranges, not points: the spread is run-to-run variance on a shared dev box, and
quoting the low end alone would be optimistic about a Pi.

Those are per-stage averages. For a specific stall — the one frame in nine
thousand that took 400 ms — `--profile run.csv` writes a row per frame with
every stage broken out, which is the only view that survives averaging. A stage
that did not run that frame is blank rather than zero: "skipped" and "took no
time" are different facts, and conflating them is how a stage silently stops
running.

No frame runs every stage — a typical one is a single overlay plus the UI.
The `neon` theme roughly doubles the cost of any line-heavy overlay, because
`glow` adds two wider passes per polyline. Re-measure with
`python -m tools.projection_test --profile`.

## Game modes

Five, over one state machine. `modes/mode_manager.py` answers the only question
that matters — *has a shot been taken?* — from noisy per-frame detections, and
every mode's scoring hangs off that answer. Modes never touch the state machine;
they get told a shot finished and what went down.

| Mode | What it is | Turn ends on |
|---|---|---|
| `freeplay` | Aiming line, no rules. The default | — |
| `classic` | Eight-ball: open table, group assignment, the 8 | miss, scratch, wrong ball |
| `king_of_the_hill` | 2–4 players, 90 s turns, combo multipliers, first to 100 | miss, scratch, clock |
| `trick_shots` | 10 preset layouts, 1–3 stars | one shot per attempt |
| `training` | Layout-aware drills with a coach line | — |

Switch with `POST /api/mode`; `POST /api/mode/difficulty` and
`POST /api/mode/challenge` configure the two modes that have options. Everything
the panel offers comes from `implemented_modes()`, so a mode that is not built
is greyed out rather than silently substituted.

### Overlays stack, and that took a fix

Each mode composes through `modes/rendering.py`, which layers the frame back to
front: the state layer (aiming line, or nothing while the balls roll), the mode
layer (a highlighted target, a challenge's ball placements), then the UI layer
(scores, timer, combo badge, effects). Four modes each assembling their own
overlay would drift apart within a week.

The first version projected a scoreboard and no trajectory. Every `render_*`
function called `ensure_canvas`, which **zeroes** — correctly, since a stale
overlay accumulating frame on frame fills the felt with light within a second —
so the UI layer wiped the aiming line underneath it. Layering is now opt-in via
`clear=False`, the caller clears exactly once per frame, and the default is
still the safe one. The unsafe direction fails visibly (lower layers vanish)
rather than slowly.

### Rules that are observable, and rules that are not

Classic implements an open table, group assignment from the first legal pot,
potting your own group keeping you at the table, and the 8 winning or losing.
It does **not** implement called pockets, ball-in-hand, two-shot carry or
push-outs. Every one of those needs input the system does not have — a player
declaring a pocket, or picking the cue ball up. The mode is silent about the
rules it cannot see rather than guessing and being wrong in front of somebody
who knows the real ones.

Two consequences of only seeing the table before and after a shot:

**Rail counts come from the prediction that was live at the strike.** Trick
shots and bank drills need to know how many cushions were involved, and the
flight happens between frames. That is an inference, and it is the same one the
aiming line already makes — if it is wrong the player can see it is wrong,
because the line was drawn on the cloth in front of them.

**A ball that vanishes is not necessarily potted.** A hand over the table is the
most common false pot, which is what the settle timer is for: by the time it
fires, hands are usually clear. There is a test that drives exactly this.

### Trick shots have to ask for the balls

Every other mode reads the table and reacts. Trick shots needs the balls in
*specific places* and the system is a camera and a projector, not a hand. So it
projects a ring on the cloth for each ball the challenge needs, waits until they
are on the rings, and only then arms the shot. Layouts live in
[`data/challenges.json`](data/challenges.json) in **table inches**, not pixels —
pixels are a property of the camera mount, so a pixel file would describe a
different challenge on every installation and stop being valid the moment
somebody nudged the camera. Layouts scale proportionally, so one file plays the
same on a 6 ft table and a 9 ft one.

### Difficulty, honestly

The spec asks King of the Hill's difficulty to change "number of balls on table
or pocket complexity". The system cannot take balls off a real table, so it
changes which ball it *asks* for: easy nominates the straightest available pot,
hard the hardest, and the points follow. The shot genuinely gets harder and the
furniture does not have to cooperate. Pots whose path is blocked are dropped
first — asking for a shot that cannot be made is the one thing that would make
the mode feel broken rather than hard.

## Finding the table without knowing what colour it is

Felt segmentation finds the cloth by looking for green, which is fast and
accurate and stops working the moment somebody re-covers their table in
burgundy. `vision/pockets.py` finds the same table by locating its **six pocket
mouths** and reconstructing the rails from them. A hole is a hole.

`vision.table_detection_method` picks: `pockets`, `felt`, or `auto` (the
default — pockets first, felt as the fallback, since pocket detection needs six
visible mouths and felt detection does not). Either way the return type is the
same `TableBoundary`, so nothing downstream changed.

Measured on the synthetic harness, one table rendered in five cloth colours:

| Cloth | Pocket detection | Felt detection |
|---|---|---|
| green | found, 1.7 px corner error | found |
| red | found, 2.0 px | **not found** |
| blue | found, 1.8 px | **not found** |
| burgundy | found, 2.0 px | **not found** |
| black | found, 1.6 px | **not found** |

Two passes, because the parameters that find a pocket depend on how big the
pockets are: a loose size-agnostic pass measures the table, and
`get_dynamic_hough_params` scales `minDist`, `minRadius` and `maxRadius` from
the *measured* pocket spacing for a second pass. No table size is hardcoded
anywhere in the path.

### Measuring the table, and the thing that cannot work

The retrofit brief specifies deriving real-world size from a fixed reference:

```
scale_factor = measured_width_px / 2000
actual_ft    = 7.0 * scale_factor
```

This cannot recover table size. A camera sees `f·L/h` pixels for a table of size
`L` at height `h`, so doubling the table and doubling the height give an
identical image — pixel width constrains only the ratio `L/h`, and one number
cannot be split into two unknowns. The 2:1 aspect does not help, because every
pool table from 6 ft to 10 ft is 2:1.

One unchanged 6.33 ft table, three camera heights:

| Camera | Table px | Reference formula | Ball ratio |
|---|---:|---:|---:|
| low | 1843 | 6.45 ft | **6.33 ft** |
| typical | 1459 | 5.11 ft | **6.33 ft** |
| high | 845 | 2.96 ft | **6.33 ft** |

The fix is a second known length in the same image, and there already is one: a
**ball** is 2.25 in whatever table it is sitting on, so `table_px / ball_px =
table_in / 2.25` and the height cancels. That is what `vision.scale_source`
prefers (`ball`, then `config`, then the brief's `reference` with a warning).
It is what makes measuring a 7.5 ft table as 7.5 ft possible at all.

Two things to know:

**Accuracy is ±2% at best and ±8% with a high camera.** The dominant error is
that a ball's antialiased edge reads as part of the ball, so the blob is about
half a pixel fat all round — 2% on a 23 px radius and worse as the ball shrinks
in frame. Adjacent standard tables are only 7% apart, which is why
`adopt_measured_table_size` is **off by default**: the measurement cannot
reliably tell a 7 ft table from a 7.5 ft one, and a wrong automatic resize
silently rescales every physics prediction. The disagreement is always logged
with the nearest standard size, so setting `table_preset` stays a one-line fix.

**It assumes regulation balls.** A mini table with undersized balls measures
large — the same ambiguity as camera height, one step down. Set
`vision.scale_source: config` and the right preset for those.

### The failure this introduced, and the fallback for it

Pocket detection finding a red table is worthless if nothing on it can be seen.
Ball detection works by *inverting the felt mask*, so a table the green
thresholds do not match reads as one enormous non-felt blob and yields zero
balls — a failure that looks like everything working right up until nothing is
ever detected. `vision.adaptive_cloth_min_coverage` triggers an adaptive cloth
mask (the median colour inside the table, whatever it happens to be) when the
configured thresholds cover too little of the cloth. Measured end to end, on a
red table: 4 of 5 balls found, worst position error 0.06 in.

Black cloth is the one case colour cannot solve — the 8 ball is the same colour
as the table — and geometry carries it instead, which is why ball detection
leans on circularity rather than hue.

### Cost

| Stage | Felt | Pockets |
|---|---:|---:|
| Table detection | ~2 ms | ~22 ms |

Table detection runs every 150 frames, so this arrives as a periodic spike, not
in the 33 ms budget. Measured in the live loop: **29.9 FPS** with a 7.3 ms
average frame and a 23 ms spike every five seconds. The scale is resolved once
per detection rather than once per pass, and the adaptive cloth mask is cropped
to the table's bounding box; without those two it was 41 ms.

## Architecture

```
camera ─→ detection ─→ physics ─→ mode ─→ renderer ─→ display
             ↑                              ↑
        calibration                    projection
      (camera↔table)                 (table↔projector)
```

Two concurrent parts: the **vision loop** on a dedicated thread, and the **web
server** on uvicorn's event loop. A thread rather than an asyncio task on
purpose — the loop is CPU-bound OpenCV work that holds the GIL in
millisecond chunks, and as a coroutine it would starve the event loop and make
the panel feel frozen exactly when the system is busiest. OpenCV releases the
GIL inside its native calls, so a real thread genuinely overlaps.
`app/state.py` holds everything both sides touch; the loop is the only writer.

### Running for hours

The target is not "runs" but "runs stably for hours", and that changes what the
loop owes you. Two hours at 30 FPS is 216,000 frames, so anything with a
one-in-ten-thousand failure rate happens twenty times a session. Nothing is
allowed to end the run:

| Failure | What happens |
|---|---|
| A stage throws | Contained to that frame and counted. Logged on the first occurrence and every 300th after — a stage failing every frame would otherwise write 1800 tracebacks a minute. |
| A stage keeps throwing | After 30 *consecutive* failures it is switched off and the loop carries on without it. The mode stage holds its last overlay rather than going dark. |
| The camera drops off USB | Reopened with backoff for up to 30 s, ball tracks reset, and the reconnect counted. |
| The camera is unplugged | Recovery is bounded, then the loop stops. A loop retrying forever burns a core and hides the failure behind a process that still looks alive. |
| Frames blow the budget | Optional work is shed (table re-detection rate, prediction rate) and picked back up when there is room. |
| The loop wedges in a driver call | A watchdog thread reads the frame heartbeat and flags it. From outside, a wedged loop and an idle one look identical. |
| SIGINT / SIGTERM | The projector is cleared and the camera released before exit. `--headless` installs its own handlers; under uvicorn the lifespan does it. |

*Consecutive* failures is what disables a stage, not total. A stage that fails
on one frame in a thousand is noisy, not broken, and losing the overlay for the
rest of the session over a flake would be worse than the flake.

Everything above is a cumulative counter on `/api/status` under `health`, and on
the panel's Health card — deliberately cumulative, because a camera that dropped
out twice an hour ago and is fine now is exactly what a long session needs to
surface, and every instantaneous view of it reads "healthy". `/health` answers
**503** when the loop is stalled or stopped, so it is safe to point a monitor at.

Measured on a 6,000-frame mock soak: no stage errors, no stalls, no reconnects,
and every bounded structure stayed bounded (rolling window 90 frames, prediction
cache 64 entries, no growth in effect or tracker containers). RSS rises ~17 MB
over the first few thousand frames and flattens — allocator arenas, not a leak;
a container census across the run shows no accumulation.

### Coordinate systems

Three spaces, and keeping them straight is the largest single source of bugs in a
projection-mapped system. Every function crossing a boundary names both spaces.

| Space | Units | Origin |
|---|---|---|
| `camera px` | pixels | top-left of the camera frame |
| `table` | **inches** | inside of the top-left cushion, +x along the long axis |
| `projector px` | pixels | top-left of the HDMI output |

Physics runs in table inches — the only space where distances are physically
meaningful and independent of where the camera happens to be mounted.
Conversions live in `vision/calibration.py` and `projection/mapper.py`; nothing
else should do coordinate math.

## Things worth knowing before implementing a stage

**The projector cannot subtract light.** `(0,0,0)` leaves the felt untouched, and
there is no way to darken a region. Overlays must be bright marks on black — a
"dimmed panel" background is not physically possible. Saturated green also
fights green felt, which is why the default palette leans cyan, mint, white and
magenta.

**Detection is looking at a surface the system is painting on.** Absolute colour
is unreliable when an overlay may be crossing the felt behind a ball. Prefer
geometry — circularity, size, position continuity — and when colour is
unavoidable, sample the ball centre, which is convex and least affected. The
eventual fix is blanking the overlay for one frame periodically and detecting
against that; the seam for it belongs in `extract_game_state`.

**Detection runs downscaled, and that is load-bearing.** Full-resolution
detection measured 45 ms against a 33 ms budget — over the limit before capture
or rendering get a share. `vision.detection_width` (960 default) is the knob;
dropping to 640 or 480 roughly halves the cost with almost no accuracy loss, and
`config.yaml` carries the measured table. Raising it to 1920 does not help
accuracy meaningfully and will not hold 30 FPS.

**Balls are found as non-felt blobs, not per-colour Hough circles.** The spec
suggests a circular Hough pass per colour band — eight passes, when a single one
already exceeds the frame budget at this resolution. Inverting the felt mask
inside the table outline finds every ball in one pass regardless of colour, and
colour is classified afterwards per blob. It is both faster and more robust: it
still finds a ball whose colour is being altered by projected light.

**A green ball on green felt is the hardest case, and hue cannot solve it.** The
6 ball's hue sits inside any felt hue range wide enough for real cloth, so
hue-only segmentation masks it as part of the table and it vanishes. Felt is
matte wool and a ball is glossy resin, so `vision.felt_sat_max` — a saturation
*ceiling* on what counts as felt — is what separates them. Tune it with
`--mask` and confirm the 6 ball stays visible.

**A pocket is a textbook false 8 ball.** Dark, round, ball-sized. The six pocket
mouths are punched out of the ball search mask; insetting the whole boundary far
enough instead would reject balls frozen on the cushion, which are in play.

**The pockets cut the table's corners off.** A corner pocket is a hole centred on
the corner, so the cloth never reaches the point the homography needs. Corners
have to be *reconstructed* by fitting the rails and intersecting them — reading
polygon vertices directly lands 30–50 px inside the true corner.

**Fixed-step physics integration cannot work here.** A NumPy-vectorised step
loop costs ~31 µs per step in Python, so the "balanced" profile (4 ms steps over
8 s) would be 62 ms per shot and "accurate" would be 374 ms — against a 33 ms
budget. The simulator is event-driven: it solves analytically for the next
collision and jumps there, so cost scales with the number of collisions (2–20)
rather than with simulated time. It also cannot tunnel a fast ball through a
slow one, and since paths are straight between events the returned polyline is
exact with no sampling.

**The textbook rolling-friction figure is wrong for a struck ball.** The
standard 0.01 coefficient gives 3.86 in/s², which is correct for a ball already
rolling — but a struck ball spends its fastest phase *sliding*, at roughly 20×
that friction. Using the pure-rolling value gave 9–10 s settle times, about
double reality. `rolling_friction` is an effective blend, calibrated to settle
time, and is the first thing to measure on a real table.

**Cut angle is the physics that matters.** Players know exactly where an object
ball should go, so an error in the line of centres at contact is immediately
visible. A slightly wrong friction coefficient is not.

**`time.perf_counter()`, not `time.monotonic()`.** Both are monotonic, but
`monotonic` has ~15.6 ms granularity on Windows, which quantises a 33 ms frame
budget to 0/15.6/31.2 ms and makes timing worthless for development off-Pi.

**Render at 1080p even though the projector accepts 4K.** A 3840×2160 RGBA
overlay is ~33 MB of pixel writes per frame and will not hold 30 FPS on a Pi 5.
The overlay is line art, so letting the projector upscale costs nothing visible.

**Manual focus in production.** Continuous autofocus hunts when a hand or cue
enters frame, and a focus shift changes the apparent ball radius mid-shot, which
breaks detection.

## Deviations from the build spec

- **Target is 64-bit.** The spec describes the Pi 5 as "32-bit ARM". It is
  aarch64, and Bookworm 64-bit is the normal target. This changes which wheels
  are available (picamera2, Hailo runtime, numpy, opencv all ship aarch64 builds)
  and removes the ~3 GB per-process ceiling the spec was implicitly designing
  around.
- **`app/models.py` uses dataclasses, not Pydantic.** `GameState` and its ball
  list are rebuilt 30 times a second; per-field validation on the hot path would
  cost CPU that detection needs. Pydantic is used where it earns its cost —
  `app/config.py` (validating user-edited YAML once) and `web/schemas.py`
  (validating untrusted HTTP input).
- **Cushion restitution defaults to 0.80, not 0.90.** Real cloth-covered rails
  measure 0.75–0.85.
- **`lifespan` instead of `@app.on_event("startup")`.** The spec's pseudocode
  used the deprecated form.
- **Python 3.10+, not the 3.9+ the spec states.** Two hard blockers:
  `dataclass(slots=True)`, used throughout `app/models.py` to hold down
  per-frame allocation cost, and PEP 604 `X | None` annotations in the Pydantic
  schemas, which Pydantic evaluates at runtime. Bookworm ships 3.11, so the
  intended target is comfortably met. `app/__init__.py` carries a runtime guard
  and `pyproject.toml` declares `requires-python = ">=3.10"`, so an older
  interpreter fails with a clear message instead of a cryptic `SyntaxError`.
- **`scipy` is listed in the spec but not installed.** Nothing imports it — the
  physics engine is deliberately pure NumPy, which the spec's own implementation
  notes ask for. It is ~90 MB on the SD card for no current benefit. Add it if a
  profiled bottleneck justifies it.
- **`httpx` is pinned `<0.28`.** Starlette's `TestClient` uses httpx's `app=`
  shortcut, which 0.28 removed. Test-only dependency.
- **Two extra files.** `app/state.py` (shared state, which the two threads
  require) and `physics/models.py` split into simulation-vs-observation objects —
  a `Ball` is what vision saw, a `SimBall` is what the simulator is moving, and
  conflating them invites writing predictions over measurements.
- **Knockout is not built.** The games spec describes five modes; four are here
  plus freeplay. Knockout is King of the Hill's turn machinery with a bracket on
  top, and four modes that work is better than five where one is a sketch.
- **Pocket effects are spawned by the mode, not inferred from motion.** The
  effect system can detect a pot by watching a ball vanish near a pocket, and
  does when nothing better is available. Once a mode is running it knows exactly
  what went down and where, so it spawns the celebration itself — the inference
  is a fallback, not the mechanism.
- **Table size is measured against a ball, not against a pixel reference.** The
  Session 2 retrofit brief specifies `scale_factor = measured_px / 2000`, which
  cannot work — see the section above for the measurements showing one table
  reported at three sizes from three camera heights. The reference formula is
  retained as `vision.scale_source: reference` for a bare table with no ball to
  measure against, and warns every time it is used.
- **The measurement dict names the long axis `length`, not `width`.** The brief
  asks for `table_width_ft` with `table_length_ft = width × 2`, but a "7 ft
  table" is 7 ft on its long axis — 76 × 38 in of playing surface. Following it
  literally would report a 7 ft table as 7 ft wide and 14 ft long and transpose
  every table coordinate downstream. `TableMeasurement.as_dict` uses this
  codebase's convention, matching `settings.table`.
- **Pockets are found by contour, not by `cv2.HoughCircles`.** Same reasoning as
  ball detection: a Hough pass costs more than the whole frame budget, and a
  corner pocket is a rounded wedge that Hough scores poorly and a circularity
  gate accepts. `get_dynamic_hough_params` still produces the brief's three
  parameters and a complete OpenCV parameter set — they gate the contour pass
  instead.
- **The calibration YAML is a report, not the source of truth.** The spec asks
  the wizard to "save calibration to YAML files". It writes the three the spec
  names, and the application loads none of them — it loads
  `projector_calibration.json`, as it did before the wizard existed. Two files
  claiming to be the calibration is worse than one, so each YAML opens with a
  header saying it changes nothing. They earn their place by carrying what the
  JSON does not: the table boundary the transform was solved against, the grid
  metrics and the end-to-end error, which are what tell you six weeks later
  whether the calibration was ever any good.
- **Fine-tuning keeps the keystone correction.** The stub planned for the nudge
  controls to drop to an affine transform and warn the user that keystone was
  being discarded. Instead the nudge is applied to the recorded corner
  correspondences and the homography is re-solved, so it survives — and there is
  nothing to warn about. `ProjectionMapper.nudge` still has the affine
  behaviour for the web panel's nudge endpoint.
- **Three ways to place a corner, not one.** The spec describes arrowing a
  projected target onto each cushion nose. That works everywhere and is kept as
  the fallback, but it is around forty keypresses for four corners against a
  ten-minute budget — so the default detects all four marks automatically by
  differencing a lit frame against a blanked one, and tapping the mark on the
  console is the middle option.
- **The wizard opens a second window.** The projector shows marks on the felt;
  a separate console window shows the camera's view of them. One surface cannot
  do both, because the user's entire task is comparing the two.

## Layout

```
launcher.py   preflight + start; the entry point
app/          main.py, config.py, models.py, state.py
vision/       camera.py, calibration.py, pockets.py, detection.py, colors.py,
              inference.py
physics/      simulator.py, models.py
projection/   mapper.py, renderer.py, display.py, draw.py, effects.py,
              themes.py, patterns.py
modes/        mode_manager.py, rendering.py, scoring.py, freeplay.py,
              classic.py, king_of_the_hill.py, trick_shots.py, training.py
calibration_ui/  calibration_app.py, overlay_renderer.py, console.py,
              metrics.py, report.py
web/          api.py, schemas.py, static/index.html
utils/        logging.py, performance.py
tools/        camera_preview.py, projection_test.py   # diagnostics; not
              imported by the app
tests/        test_scaffold.py, test_vision.py, test_pockets.py,
              test_physics.py, test_rendering.py, test_modes.py,
              test_web.py, test_calibration_ui.py, test_integration.py,
              test_launcher.py, synthetic.py
```

Full specs: [`ar_pool_table_prompt.md`](../ar_pool_table_prompt.md),
[`ar_pool_calibration_ui_prompt.md`](../ar_pool_calibration_ui_prompt.md),
[`ar_pool_games_and_animations.md`](../ar_pool_games_and_animations.md).
