"""Print exactly what the focus-path selector sees, and what it decides.

Why this exists: the choice between the two focus controls is made from one
fact -- whether libcamera advertises ``AfMode`` for this sensor -- and getting it
wrong does not degrade focus, it makes the lens undrivable. That fact was being
read inside a camera backend during startup, so when the decision came out wrong
there was no way to see *what it saw*, only what it concluded, several minutes
later, in a message about a ribbon cable.

So this reproduces the selection outside the application, prints the evidence at
each step, and states the answer. It opens the real camera and starts streaming,
because that is the only state in which the question can be asked: on an
AF-bound sensor there is nothing to interrogate until the stream is up.

    python -m tools.focus_probe
    python -m tools.focus_probe --controls     # dump every control, not just AF
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CameraSettings, load_settings  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("focus_probe")

#: The controls whose presence decides the path, and the ones worth seeing next
#: to them. ``AfMode`` is the only one the decision turns on; the rest are shown
#: because "AfMode absent" is much more convincing alongside its neighbours.
AF_CONTROLS = (
    "AfMode",
    "LensPosition",
    "AfRange",
    "AfSpeed",
    "AfTrigger",
    "AfMetering",
    "AfPause",
    "AfWindows",
    "AfState",
)


def say(line: str = "") -> None:
    print(line, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    parser.add_argument(
        "--controls", action="store_true", help="list every advertised control"
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    setup_logging(args.log_level, log_to_file=False)

    from vision.camera import Camera, CameraError
    from vision.focus import FOCUS_DIOPTRES, LibcameraFocus, V4L2Focus, find_lens_subdev
    from vision.focus_calibration import (
        coarse_step,
        dioptres_to_metres,
        focus_positions,
        sweep_band,
        sweep_bounds,
    )

    say()
    say("  Config")
    say(f"    focus_path          {settings.camera.focus_path}")
    say(f"    lens_driver         {settings.camera.lens_driver!r}")
    say(f"    focus_enabled       {settings.camera.focus_enabled}")
    say(f"    focus_absolute      {settings.camera.focus_absolute}   (counts)")
    say(f"    focus_dioptres      {settings.camera.focus_dioptres}   (dioptres)")
    say()

    # The V4L2 half can be answered without a camera, so it is answered first --
    # if the subdev is missing that is worth knowing even on a rig that ends up
    # using libcamera.
    say("  V4L2 subdev")
    lens = find_lens_subdev(settings.camera.lens_driver)
    if lens is None:
        say(f"    none matching {settings.camera.lens_driver!r}")
    else:
        say(f"    {lens.path}  ({lens.name})")
        try:
            v4l2_range = V4L2Focus(lens).range()
            say(
                f"    focus_absolute: min={v4l2_range.minimum} max={v4l2_range.maximum} "
                f"step={v4l2_range.step} default={v4l2_range.default}"
            )
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            say(f"    could not query it: {type(exc).__name__}: {exc}")
    say()

    # Real hardware only. A mock camera advertises no controls, so it would
    # report "AfMode absent" and look exactly like a genuine IMX519 -- a
    # confident wrong answer, which is the thing this tool exists to prevent.
    camera_settings = CameraSettings(**{**settings.camera.model_dump(), "use_mock": False})
    try:
        camera = Camera(camera_settings).open()
    except CameraError as exc:
        say(f"  Cannot open the camera: {exc}")
        return 2
    if camera.is_mock:
        say("  The camera fell back to the mock backend, so there is nothing to probe.")
        say("  This needs the real camera on the Pi.")
        camera.close()
        return 2

    try:
        backend = camera._backend
        say(f"  Backend               {backend.name}")

        names, problem = (
            backend._control_names()
            if hasattr(backend, "_control_names")
            else ((), f"{backend.name} backend exposes no control list")
        )
        say(f"  Controls advertised   {len(names)}")
        if problem:
            say(f"  PROBLEM               {problem}")
        say()

        say("  Autofocus controls")
        for control in AF_CONTROLS:
            say(f"    {control:<16} {'present' if control in names else '-'}")
        say()

        if args.controls and names:
            say("  Every control")
            for control in sorted(names):
                say(f"    {control}")
            say()

        say("  Decision")
        controller = camera.focus_controller()
        say(f"    {camera.focus_path()}")
        if controller is None:
            say("    -> no lens can be driven on this rig")
            return 1

        focus_range = controller.range()
        say(
            f"    range               {focus_range.format(focus_range.minimum)} to "
            f"{focus_range.format(focus_range.maximum)}"
        )
        say(f"    readback tolerance  {focus_range.tolerance:g}")

        band = sweep_band(settings)
        low, high = sweep_bounds(focus_range, band)
        step = coarse_step(focus_range, band)
        stops = focus_positions(focus_range, step, low, high)
        say(
            f"    sweep               {len(stops)} stops, "
            f"{focus_range.format(low)} to {focus_range.format(high)}, step {step:g}"
        )
        if focus_range.kind == FOCUS_DIOPTRES:
            say(
                f"    i.e. distances      {dioptres_to_metres(high)} to "
                f"{dioptres_to_metres(low)}"
            )
        say()

        say("  Live position")
        try:
            if isinstance(controller, LibcameraFocus):
                controller.prepare()
            say(f"    reads {focus_range.format(controller.read())}")
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            say(f"    could not read it: {type(exc).__name__}: {exc}")
        say()
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    raise SystemExit(main())
