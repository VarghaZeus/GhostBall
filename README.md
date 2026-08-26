# GhostBall

**Your pool table, with the answers written on it.**

A camera watches the felt. A projector draws on it. Line up a shot and the table
lights up in front of you — where the cue ball is going, where it'll hit, the angle
the object ball leaves at, which cushions it'll come off, where everything ends up.

No screen to glance at. No app in your hand. No sensors in the balls. You just play,
and the table shows you what's about to happen.

<!-- TODO: GIF goes here. Five seconds of the aiming line tracking your cue sells
     this better than the entire README. -->

---

## What you see on the cloth

**A live aiming line** that follows your cue as you move it. Dashed, animated, and it
updates while you're still settling into the shot.

**The ghost ball** — the exact spot the cue ball has to occupy at contact. Every
player already visualises this. GhostBall just draws it.

**Impact markers with cut angles**, so you can see the geometry instead of guessing
at it.

**Cushion rebounds**, traced all the way around the table. Bank shots stop being a
mystery.

**Where the balls stop.** Position play, drawn out before you commit.

And when you sink one, the pocket erupts — vortex, sparks, score popping up off the
cloth. It's a pool table that celebrates with you.

Four themes: `classic` green, `neon`, `dark_mode`, and a stripped-back `pro` look for
when the effects get in the way.

---

## Five ways to play

### Freeplay
Just the aiming line. No rules, no score. Rack up and shoot — the table helps and
otherwise stays out of your way. This is the one you'll leave it on.

### Classic
Real eight-ball. Open table, groups assigned on your first legal pot, the 8 to finish.
Pot yours and you stay at the table.

### King of the Hill
2–4 players, 90 seconds a turn. Every pot buys you 5 more seconds and pushes your
combo multiplier up. Miss and the clock passes to the next player. First to 100.

It gets loud.

### Trick Shots
Ten preset layouts. The table projects a ring where each ball needs to go, waits for
you to place them, then sets you the challenge. One shot per attempt, 1–3 stars.

### Training
Drills that read the table you've actually got in front of you and give you a coach
line. Potting, position, banks.

---

## The rig

| | |
|---|---|
| Raspberry Pi 5, 16 GB | does all the work |
| Arducam 16 MP autofocus | overhead, looking straight down |
| 1080p projector | overhead, same |

That's it. Both bits of hardware point down at the middle of the table. Nothing is
attached to the table, nothing goes in the balls, and it comes down as easily as it
went up.

Runs at **22 FPS on the Pi** with 41 ms of latency — fast enough that the line moves
with your cue rather than after it. Ball positions land within 0.35 in, cue aim within
0.8°, and it finds the table on green, red, blue, burgundy or black cloth because it
looks for the pockets rather than the colour.

---

## Getting it going

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

python launcher.py --mock     # try it with no hardware at all
python launcher.py            # the real thing
```

It prints a LAN address on startup. Open that on your phone — that's the control
panel, and it's where you switch modes, tweak the overlays, and run setup.

---

## Setting it up

Mount the camera and projector above the middle of the table. Then open the Setup tab
on your phone and hit **Full setup**.

Seven screens, a few minutes. It finds the table, walks you through aiming the
projector, focuses the camera itself, and then checks its own work by projecting a
ring at each ball and measuring how close it landed. If something's off it tells you
which thing to move and which direction.

Bump the rig later? It figures out what actually drifted and offers to redo only that
part, instead of making you start over.

---

## Where it's at

Complete and playable. Five modes, full physics, calibration wizard, phone control
panel. 798 tests.

Ideas I haven't built yet:

- **Spin.** Reading english off the cue tip, and drawing the curve it puts on the ball
- **Shot power** from cue speed, so the prediction knows how hard you're hitting
- **Knockout brackets** for tournament nights
- **Reading ball numbers** so it can talk about the 7 instead of "the maroon one"
- **Hand tracking**, so you tap projected buttons on the cloth instead of your phone

The Hailo NPU is plugged in and doing nothing so far. That's headroom for whatever
comes next.

---

## Why

Because a pool table is a physics engine you can touch, and nobody had gotten around
to drawing on it yet.

MIT licensed. Go build one.