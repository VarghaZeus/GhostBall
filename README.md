# GhostBall

**Your pool table, with the answers written on it.**

A camera watches the felt. A projector draws on it. Line up a shot and the table
lights up in front of you â€” where the cue ball is going, where it'll hit, the angle
the object ball leaves at, which cushions it'll come off, where everything ends up.

No screen to glance at. No app in your hand. No sensors in the balls. You just play,
and the table shows you what's about to happen.

<!-- TODO: GIF goes here. Five seconds of the aiming line tracking your cue sells
     this better than the entire README. -->

---

## What you see on the cloth

**A live aiming line** that follows your cue as you move it. Dashed, animated, and it
updates while you're still settling into the shot.

**The ghost ball** â€” the exact spot the cue ball has to occupy at contact. Every
player already visualises this. GhostBall just draws it.

**Impact markers with cut angles**, so you can see the geometry instead of guessing
at it.

**Cushion rebounds**, traced all the way around the table. Bank shots stop being a
mystery.

**Where the cue ball ends up** — the whole of position play, and the line most aiming
aids leave out. Drawn lighter than the aiming line, because it is a consequence of the
shot rather than something you point at.

**Five power ticks** along that line, labelled *very soft* through *very hard*. How far
the cue ball travels depends on how hard you hit it, and nothing here measures that — so
instead of guessing a number and planting a ghost ball at it, every answer is drawn and
you pick the one that leaves the position you want. On a full hit all five collapse into
a couple of inches, and the table says so: *power barely matters*. That is worth knowing
before you choose a stroke.

**And a recommendation, where the rules say what you're aiming for.** In classic, King of
the Hill, and the position drill, each of the five leaves is scored — can you get at a
ball you're allowed to hit, is the angle sensible or desperate, are you frozen on a rail,
did you scratch — and the best one is highlighted. The other four stay on the cloth,
dimmed. A black-box *hit MEDIUM* teaches nothing and can't be argued with; the ticks are
what let you see what the advice gave up and overrule it.

When no pace leaves anything playable it says so — *no good leave from here, play safe* —
rather than quietly picking the least bad. And freeplay stays silent: there's no next
ball, so there's no goal to score against, and advice against a guessed goal is worse
than none.

**A tip contact target** in training and trick shots — the cue ball's face projected on
the cloth with rings, a centre crosshair, and a mark showing where to strike it. The
drill defines the english, so this needs no cue tracking: it is an instruction, and the
predicted path is computed from the same offset it draws.

**Where the balls stop.** Position play, drawn out before you commit.

And when you sink one, the pocket erupts â€” vortex, sparks, score popping up off the
cloth. It's a pool table that celebrates with you.

Four themes: `classic` green, `neon`, `dark_mode`, and a stripped-back `pro` look for
when the effects get in the way.

---

## Five ways to play

### Freeplay
Just the aiming line. No rules, no score. Rack up and shoot â€” the table helps and
otherwise stays out of your way. This is the one you'll leave it on.

### Classic
Real eight-ball. Open table, groups assigned on your first legal pot, the 8 to finish.
Pot yours and you stay at the table.

### King of the Hill
2â€“4 players, 90 seconds a turn. Every pot buys you 5 more seconds and pushes your
combo multiplier up. Miss and the clock passes to the next player. First to 100.

It gets loud.

### Trick Shots
Ten preset layouts. The table projects a ring where each ball needs to go, waits for
you to place them, then sets you the challenge. One shot per attempt, 1â€“3 stars.

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

Runs at **22 FPS on the Pi** with 41 ms of latency â€” fast enough that the line moves
with your cue rather than after it. Ball positions land within 0.35 in, cue aim within
0.8Â°, and it finds the table on green, red, blue, burgundy or black cloth because it
looks for the pockets rather than the colour.

---

## Try it with no hardware

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

python launcher.py --mock
```

Synthetic camera, projector output discarded, panel live on
`http://localhost:8000/`. Good for poking around before you own a projector.

---

## Getting it running on the Pi

Four things will bite you. All four are quick, and each one costs an evening if you
find it the hard way.

### 1. The venv has to see system packages

`picamera2` comes from apt, not pip. A plain `venv` can't import it and the camera
silently falls back to a synthetic one.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Force the projector to 1080p

It'll happily negotiate 3840Ã—2160, and then every frame is a CPU upscale from 1080p
to 4K. That alone halves your frame rate â€” 22 FPS down to 10.

```bash
xrandr --output HDMI-1 --mode 1920x1080 --rate 60
```

Make it stick across reboots â€” `~/.config/autostart/ghostball-resolution.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Set projector resolution
Exec=xrandr --output HDMI-1 --mode 1920x1080 --rate 60
```

The overlay is line art, so letting the projector do the upscaling costs nothing you
can see.

### 3. `export DISPLAY=:0` over SSH

Otherwise the projector window can't open at all and you get a Qt platform plugin
error several imports deep. The preflight check catches this and tells you, but it's
faster to just set it.

### 4. Focus the camera

The lens powers up at position 0 and stays there. Autofocus doesn't work through
libcamera on the IMX519, so GhostBall drives the motor directly â€” but it needs to
know where to put it.

**Focus the projector first, with its own remote.** The camera can't resolve detail
the projector never drew.

```bash
python -m tools.focus_calibrate
```

It projects five checkerboard targets onto the felt, sweeps the lens, and finds the
sharpest position. Saves it, applies it on every boot from then on. If something's
wrong it tells you which thing â€” projector off, room too bright, camera too far,
mount tilted.

### Then go

```bash
python launcher.py
```

It prints a LAN address on startup. Open that on your phone â€” that's the control
panel, and it's where you switch modes, tweak the overlays, and run setup.

---

## Aiming the projector

Mount the camera and projector above the middle of the table. Then open the Setup tab
on your phone and hit **Full setup**.

Seven screens, a few minutes. It finds the table, walks you through aiming the
projector, focuses the camera itself, and then checks its own work by projecting a
ring at each ball and measuring how close it landed. If something's off it tells you
which thing to move and which direction.

Bump the rig later? It figures out what actually drifted and offers to redo only that
part, instead of making you start over.

---

## Rebooting from the panel

**Diagnostics -> Restart -> Reboot the Pi.** For when something has locked up hard
enough that nothing else on the panel will fix it: a camera that won't reopen, a
projector window swallowed by a wedged compositor. Pulling the power is the
alternative, and that's how an SD card gets corrupted.

It asks before firing. While the Pi is away the panel says it's waiting for it, rather
than claiming it lost the connection - and it tells you when it's back.

The service account has to be allowed to run `reboot` without a password, because
there's no terminal for `sudo` to prompt at:

```bash
echo "$USER ALL=(root) NOPASSWD: /sbin/reboot" | sudo tee /etc/sudoers.d/010-ghostball-reboot
```

Without that entry the button reports what `sudo` actually said, instead of claiming a
reboot that never happened. It also declines on a host running mock hardware, or one
that isn't Linux, so it can't reboot the machine you're developing on.

---

## Where it's at

Complete and playable. Five modes, full physics, calibration wizard, phone control
panel. 890 tests.

Ideas I haven't built yet:

- **Reading english off the cue tip.** Training mode *prescribes* spin and honours it in
  the prediction, but the system cannot yet see what you actually did — nor draw the
  curve swerve puts on the ball
- **Shot power** from cue speed. The five power ticks make this optional rather than
  urgent: measuring it would collapse the fan to one answer, which is nicer, but the
  fan is honest without it
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