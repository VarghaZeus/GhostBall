# GhostBall

**Your pool table, with the answers written on it.**

A camera watches the felt. Physics works out where the balls are going. A projector
draws it back onto the cloth in front of you — aiming lines, cut angles, ghost balls,
cushion rebounds — while you're still lining up the shot.

No screen. No app to look at. No sensors in the balls. Just the table, lit up.

```
camera ─→ detection ─→ physics ─→ mode ─→ renderer ─→ projector
             ↑                              ↑
        calibration                    projection
      (camera↔table)                 (table↔projector)
```

<!-- TODO: drop a GIF here. This project is 100% visual and the README currently
     asks people to imagine it. A 5-second clip of the aiming line tracking a cue
     is worth every word below. -->

---

## Does it work?

Yes. It plays five game modes, finds the table on any cloth colour, and walks a
first-timer through projector alignment from their phone.

Honest numbers, because there are two and only one of them is the real one:

| | Raspberry Pi 5 | x86 dev box |
|---|---|---|
| Frame rate | **22 FPS** | 29.9 FPS |
| Frame time | 44 ms | 16 ms |
| Latency | 41 ms | — |

The Pi number is the one that matters and the one I'd quote. Detection is ~18 ms,
projection ~20 ms, capture ~2.5 ms. It holds that for hours without drifting — a
6,000-frame soak showed no stalls, no leaks, no reconnects.

Accuracy, measured against synthetic ground truth across 11 scenarios (keystone,
rotation to 20°, shadows, sensor noise, vignetting, projected light):

| | Target | Worst measured |
|---|---|---|
| Table corner error | < 20 px | **4.5 px** |
| Ball position | < 3 in | **0.35 in** |
| Cue aim | — | **0.8°** |
| Balls found | 8/8 | **8/8, every scenario** |

Physics is checked against closed-form results rather than golden values: the 90°
rule holds to 0.01° across cut angles 15–75°, Newton's cradle transfers all speed,
free-roll distance matches `s²/(2a)` to 0.05 in. A full 15-ball rack simulates in
1.5 ms.

798 tests, no hardware required. One thing left unbuilt: optional Hailo NPU
inference, which the OpenCV path already covers.

---

## The hardware

| | |
|---|---|
| Raspberry Pi 5, 16 GB | 64-bit, Trixie, Python 3.13 |
| Arducam IMX519 | 16 MP, autofocus, overhead |
| 1080p projector | overhead, pointing straight down |
| Hailo-8L AI HAT+ | 13 TOPS, plugged in, currently unemployed |

Both the camera and the projector look straight down at the middle of the table.
That's the whole rig.

---

## Quick start

```bash
python3 -m venv --system-site-packages .venv   # so apt's picamera2 stays importable
source .venv/bin/activate
pip install -r requirements.txt

python -m tools.camera_preview     # check the camera FIRST
python launcher.py --mock          # no hardware needed
python launcher.py                 # the real thing
```

Then open the panel — `launcher.py` prints the LAN address, because the panel is
meant to be used from a phone at the table and `localhost:8000` is useless there.

Calibration runs from the **Setup tab** on your phone, not from SSH. Seven screens,
a few minutes. The projector shows patterns; the phone carries every instruction and
number. That split isn't a preference — the focus step measures edge sharpness inside
projected targets, so a line of text on the cloth would get measured along with them.

`launcher.py` runs a preflight first, and every check tells you what to *do*:

```
! calibration   none saved -- overlays will not line up with the felt
                  -> python -m calibration_ui.calibration_app
x web port      0.0.0.0:8000 is already in use
                  -> Something is already listening -- most likely another copy
                     of this app. Stop it, or pick another port: --port 8001

Not starting: 1 check(s) failed (web port).
```

Every one of those is a failure that otherwise arrives as a traceback six imports
deep, or — worse — as a system that starts, looks healthy, and projects something
wrong.

---

## Things that were harder than they looked

This is the interesting part. Every entry below is a bug that shipped, passed tests,
or wasted a day, and each one changed how the code works.

### The window that was 1920×548

A warning reported the projector window as 1920×548 on a 1920×1080 desktop. Not half,
not a real video mode, not anything else on the machine. It sent me hunting a
fullscreen bug for two days.

It was wrong twice. `getWindowImageRect` returns the image *drawing* area, not the
window — and it was being called before any `imshow`, so there was no image to have
an area. Then fullscreen turns out to be **asynchronous**: `setWindowProperty` posts
a request the window manager answers milliseconds later, and the code read geometry
on the very next line. So it measured a nonexistent thing, mid-map.

The window had been fine the whole time. `python -m tools.window_probe` now watches
the area over several seconds and cross-checks against what the X server says, which
is a different question and where the answer actually lives.

### You cannot measure a table with one number

The original plan for working out real-world table size:

```python
scale_factor = measured_width_px / 2000
actual_ft    = 7.0 * scale_factor
```

This can't work, and it took a measurement to prove rather than an argument. A camera
sees `f·L/h` pixels for a table of size `L` at height `h` — so doubling the table
*and* doubling the height give an identical image. One number can't be split into two
unknowns. The 2:1 aspect ratio doesn't rescue it, because every pool table from 6 ft
to 10 ft is 2:1.

One unchanged 6.33 ft table, three camera heights:

| Camera | Table px | The formula says | Truth |
|---|---|---|---|
| low | 1843 | 6.45 ft | **6.33 ft** |
| typical | 1459 | 5.11 ft | **6.33 ft** |
| high | 845 | 2.96 ft | **6.33 ft** |

The fix was already sitting on the table: a **ball** is 2.25 inches wherever it is.
`table_px / ball_px = table_in / 2.25`, and the camera height cancels out.

### The ceiling that was a pool table

Pointed at a ceiling with no table in the room, the system confidently reported
finding one — 6/6 pockets, 41% confidence. Six light fittings are, geometrically
speaking, six pocket mouths.

The confidence gate was set at 0.45. But pocket confidence is
`0.55 × (pockets/6) + 0.45 × aspect_score`, so six dark blobs score **0.55 on their
own** with the aspect ratio contributing nothing at all. The threshold was below its
own floor — it could never have rejected anything.

It's 0.75 now, and the rule is structural rather than tuned: the gate has to sit
above what any single term can produce, so both have to agree. A real table
measures 0.97.

The false positive wasn't cosmetic either. With a boundary cached, detection runs ball
detection, cue detection and pocket refinement inside it — measured at **5× the
no-table path**. An imaginary table was tripling the frame time.

### `node --check` is not a test

A patch renamed a function. The follow-up fix that was supposed to update the caller
silently didn't match, leaving `paintStatus` called and never defined. I ran
`node --check`, it passed, and I shipped it.

`node --check` proves a file *parses*. A dangling identifier is perfectly valid
syntax — it's a runtime `ReferenceError`. The panel had been completely broken, not
degraded, and 581 tests stayed green because none of them executed the panel's
JavaScript.

Now they do, two ways: against a modelled DOM in the suite, and against real Chrome
over the DevTools protocol. The assertion that would have caught it is
`test_no_field_is_left_at_its_placeholder` — a blanket check, so a new card that
never gets painted fails without anyone remembering to add an assertion for it.

There's also a test that reintroduces the original bug and asserts the suite goes
red. Guarding the guard.

### And the same class of bug, five more times

A pattern emerged: **a number that reported something other than what it meant.**

- `paintStatus` threw during render, and the catch block reported "Lost contact with
  the Pi" — sending me to debug the network while the Pi served perfectly and the
  camera preview kept updating next to the word "disconnected".
- `table=98.8ms` sat at the top of the stage times looking like the most expensive
  thing in the pipeline. It runs once every 600 frames — an amortised 0.16 ms, about
  a four-hundredth of what capture costs *every* frame. Stage times are now annotated
  with coverage: `table=98.8[3% of frames, 1.03ms/frame]`.
- `physics=1.3[0% of frames]` paired a real mean from outside the window with a zero
  cost, reading as "free" rather than "hasn't run lately". Now it says
  `[not in the last 90 frames]`.
- `dropped=1174/1174` — dropped always equalled total, which carries no information.
- **libcamera reported success while dropping the control entirely.** Setting
  `AfMode` or `LensPosition` on the IMX519 warns on libcamera's own log stream and
  discards the request. `set_controls` returns cleanly. A `try/except` catches
  nothing. Every layer above believed focus was set while the lens sat wherever it
  powered up.

That last one is why focus now goes straight at the motor over V4L2, and why **every
set is read back**. A readback mismatch is an error, not a warning — it's exactly what
a half-seated ribbon cable looks like: the I2C device answers, the write returns 0,
and the lens doesn't move.

### The overlay that erased itself

First version of the mode renderer projected a scoreboard and no aiming line.

Every `render_*` function called `ensure_canvas`, which zeroes the canvas —
correctly, because a stale overlay accumulating frame on frame fills the felt with
light inside a second. So the UI layer was wiping the trajectory underneath it. Two
functions each doing the right thing, producing nothing together.

Layering is opt-in now via `clear=False`, the caller clears exactly once per frame,
and the default stays safe. The unsafe direction fails *visibly* — lower layers
vanish immediately — rather than slowly filling the table with light.

### The 4K tax

One command doubled the frame rate:

```bash
xrandr --output HDMI-1 --mode 1920x1080 --rate 60
```

The projector had negotiated 3840×2160, so every frame was a CPU upscale from 1080p
to 4K. Projection cost went 58 ms → 19.6 ms, frame rate 10.5 → 22 FPS.

The overlay is line art. Letting the projector do the upscaling costs nothing
visible, and a 3840×2160 RGBA buffer is ~33 MB of pixel writes per frame that a Pi 5
will not do thirty times a second.

---

## Finding the table when you don't know what colour it is

Felt segmentation looks for green. It's fast, it's accurate, and it stops working the
moment somebody recovers their table in burgundy.

So GhostBall finds the table by its **six pocket mouths** instead and reconstructs the
rails from them. A hole is a hole.

| Cloth | Pockets | Felt |
|---|---|---|
| green | found, 1.7 px error | found |
| red | found, 2.0 px | **not found** |
| blue | found, 1.8 px | **not found** |
| burgundy | found, 2.0 px | **not found** |
| black | found, 1.6 px | **not found** |

No table size is hardcoded anywhere in that path. A loose size-agnostic pass measures
the table, then the Hough parameters are scaled from the *measured* pocket spacing for
a second pass. Works on 6 ft, works on 9 ft, works on 7.5 ft.

Two things this had to survive:

**The pockets cut the corners off.** A corner pocket is a hole centred exactly where
the corner is, so the cloth never reaches the point the homography needs. Corners get
*reconstructed* by fitting the rails and intersecting them — reading polygon vertices
directly lands 30–50 px inside the true corner.

**A pocket is a textbook false 8 ball.** Dark, round, ball-sized. The six mouths are
punched out of the ball search mask. Insetting the whole boundary instead would reject
balls frozen on the cushion, which are in play.

---

## Five modes

One state machine underneath, answering the only question that matters: *has a shot
been taken?* Modes never touch it — they get told a shot finished and what went down.

| Mode | What it is | Turn ends on |
|---|---|---|
| `freeplay` | Aiming line, no rules. The default | — |
| `classic` | Eight-ball: open table, group assignment, the 8 | miss, scratch, wrong ball |
| `king_of_the_hill` | 2–4 players, 90 s turns, combo multipliers, first to 100 | miss, scratch, clock |
| `trick_shots` | 10 preset layouts, 1–3 stars | one shot per attempt |
| `training` | Layout-aware drills with a coach line | — |

**Classic only implements the rules a camera can see.** Open table, group assignment
from the first legal pot, the 8 winning or losing. Not called pockets, ball-in-hand,
or push-outs — every one of those needs a player *declaring* something, and being
confidently wrong about 8-ball rules in front of someone who plays seriously is worse
than staying quiet.

**Difficulty can't remove balls from a real table**, so King of the Hill changes which
ball it *asks* for: easy nominates the straightest available pot, hard the hardest.
Blocked pots are never nominated — asking for a shot that can't be made is the one
thing that would make it feel broken rather than hard.

**Trick shots have to ask for the balls.** Every other mode reads the table and
reacts; trick shots needs balls in specific places, and the system is a camera and a
projector, not a hand. So it projects a ring for each ball the challenge needs, waits
until they're on the rings, and only then arms the shot. Layouts live in table
**inches**, not pixels — pixels are a property of the camera mount, so a pixel file
would describe a different challenge on every installation.

---

## Things worth knowing before you touch the rendering

**The projector cannot subtract light.** `(0,0,0)` leaves the felt untouched and
there's no way to darken a region. Overlays are bright marks on black — a "dimmed
panel" background is not physically possible. Saturated green also fights green felt,
which is why the palette leans cyan, mint, white and magenta.

**Detection is looking at a surface the system is painting on.** Absolute colour is
unreliable when an overlay might be crossing the felt behind a ball. Geometry first —
circularity, size, position continuity — and when colour is unavoidable, sample the
ball centre, which is convex and least affected.

**A green ball on green felt is the hardest case, and hue cannot solve it.** The 6
ball's hue sits inside any felt range wide enough for real cloth. Felt is matte wool
and a ball is glossy resin, so a *saturation ceiling* is what separates them.

**Fixed-step physics integration can't work here.** A vectorised step loop costs
~31 µs per step, so a 4 ms step over 8 s is 62 ms per shot against a 33 ms budget. The
simulator is event-driven: it solves analytically for the next collision and jumps
there, so cost scales with collisions (2–20) rather than simulated time. It also can't
tunnel a fast ball through a slow one, and paths are straight between events, so the
returned polyline is exact with no sampling.

**The textbook rolling-friction figure is wrong for a struck ball.** The standard 0.01
coefficient is right for a ball already rolling, but a struck ball spends its fastest
phase *sliding*, at roughly 20× that friction. Using the pure-rolling value gave 9–10
second settle times, about double reality.

**Cut angle is the physics that matters.** Players know exactly where an object ball
should go, so an error in the line of centres at contact is immediately visible. A
slightly wrong friction coefficient is not.

---

## Running for hours

The target isn't "runs", it's "runs for a whole evening". Two hours at 30 FPS is
216,000 frames, so anything with a one-in-ten-thousand failure rate happens twenty
times a session. Nothing is allowed to end the run.

| Failure | What happens |
|---|---|
| A stage throws | Contained to that frame and counted |
| A stage keeps throwing | Disabled after 30 **consecutive** failures; the loop carries on. The mode stage holds its last overlay rather than going dark |
| Camera drops off | Reopened with backoff for 30 s, ball tracks reset, reconnect counted |
| Camera unplugged | Recovery is bounded, then stop. A loop retrying forever burns a core and hides the failure behind a process that still looks alive |
| Frames blow the budget | Optional work is shed, picked back up when there's room |
| Loop wedges in a driver call | A watchdog thread reads the frame heartbeat. From outside, a wedged loop and an idle one look identical |

*Consecutive*, not total. A stage failing one frame in a thousand is noisy, not
broken, and losing the overlay for a whole session over a flake is worse than the
flake.

Per-frame code never logs unconditionally, either. At 30 FPS one ungated
`logger.warning` is 1,800 lines a minute — that doesn't just wear out an SD card, it
buries everything else. Conditions report once, again on change, and again on
recovery, because "detection started failing" and "detection recovered" are both
events.

---

## Three coordinate systems, and why that's the whole ballgame

| Space | Units | Origin |
|---|---|---|
| `camera px` | pixels | top-left of the camera frame |
| `table` | **inches** | inside of the top-left cushion, +x along the long axis |
| `projector px` | pixels | top-left of the HDMI output |

Physics runs in table inches — the only space where distances are physically
meaningful and independent of where the camera happens to be mounted. Every function
that crosses a boundary names both spaces, because mixing them up is the single
largest source of bugs in a projection-mapped system.

The web panel's preview is a good example of why. The overlay is in projector pixels
and the frame is in camera pixels. They're the same *size* by default, so blending
them directly produces an image that looks completely plausible and is wrong. The
preview composes projector → table → camera first.

---

## Layout

```
launcher.py      preflight + start; the entry point
app/             main.py, config.py, models.py, state.py, readiness.py,
                 wizard.py, exclusive.py, calibration_status.py
vision/          camera.py, calibration.py, pockets.py, detection.py, colors.py,
                 focus.py, focus_calibration.py, inference.py
physics/         simulator.py, models.py
projection/      mapper.py, renderer.py, display.py, draw.py, effects.py,
                 themes.py, patterns.py, onboarding.py
modes/           mode_manager.py, rendering.py, scoring.py, freeplay.py,
                 classic.py, king_of_the_hill.py, trick_shots.py, training.py
calibration_ui/  calibration_app.py, overlay_renderer.py, console.py,
                 metrics.py, report.py
web/             api.py, schemas.py, static/index.html
utils/           logging.py, performance.py
tools/           camera_preview.py, projection_test.py, focus_sweep.py,
                 focus_calibrate.py, window_probe.py
tests/           798 of them
```

Two concurrent parts: the vision loop on a dedicated thread, and the web server on
uvicorn's event loop. A thread rather than an asyncio task on purpose — the loop is
CPU-bound OpenCV work, and as a coroutine it would starve the event loop and make the
panel feel frozen exactly when the system is busiest. OpenCV releases the GIL inside
its native calls, so a real thread genuinely overlaps.

**The web layer never draws.** A full-screen OpenCV window belongs to the thread that
created it. So "blank the projector" is *recorded* on shared state and picked up by
the loop 33 ms later — imperceptible to a thumb on a phone, and the difference between
working and a segfault depending on the platform's GUI backend.

---

## Docs

- [Calibration](docs/calibration.md) — the wizard, focus over V4L2, and why the
  projector finds its own focus targets
- [Detection](docs/detection.md) — pocket-based table finding, ball detection, the
  measurement problem
- [Performance](docs/performance.md) — full frame budgets, profiling, what to measure
  on a new rig
- [Deviations from spec](docs/deviations.md) — where this differs from the original
  brief and why

---

## License

MIT. Go build one.