"""Persistence for the digital crop.

A JSON file beside the focus and projector calibrations, **not** a section
written back into ``config.yaml``. That is a deliberate repeat of the reasoning
in :class:`vision.focus.FocusCalibration`, and it is the established rule in this
codebase: ``config.yaml`` is written by a person and is full of their comments,
so machine-written values in it get clobbered in one direction or the other.
``web.api.update_settings`` says the same thing about sliders.

The user asked for the crop to "save to config". This is that -- persisted
configuration that survives a restart -- stored where software is allowed to
write. ``camera.crop`` in ``config.yaml`` is still read, and is the value a rig
with no saved crop starts from, so the setting remains hand-editable; the saved
file simply wins when it exists. Deleting it returns you to the YAML.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from vision.crop import CropRect

logger = logging.getLogger(__name__)

__all__ = ["crop_path", "load", "save"]


def crop_path() -> Path:
    """Where the saved crop lives. Beside the other calibrations."""
    from app.config import CALIBRATION_DIR

    return CALIBRATION_DIR / "crop.json"


def save(
    rect: CropRect | None, sensor_size: tuple[int, int], path: Path | None = None
) -> Path:
    """Write the crop, or record that there is deliberately no crop.

    ``rect=None`` writes ``enabled: false`` rather than deleting the file. The
    distinction matters on the next boot: an absent file means "fall back to
    ``config.yaml``", while a disabled one means "somebody chose the full frame",
    and silently reverting a deliberate choice to a stale YAML value is the kind
    of surprise this file exists to avoid.

    The sensor size is stored alongside. A rectangle is only meaningful against
    the frame it was chosen in, and a capture-resolution change makes a saved
    crop mean somewhere else -- so :func:`load` can say so instead of applying
    it.

    Written via a temporary file and replaced, so an interrupted save leaves the
    previous crop rather than a truncated file.
    """
    target = path or crop_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "enabled": rect is not None,
        "sensor_width": int(sensor_size[0]),
        "sensor_height": int(sensor_size[1]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if rect is not None:
        payload.update(rect.as_dict())

    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(target)
    logger.info("saved camera crop to %s: %s", target, rect.as_dict() if rect else "full frame")
    return target


def load(
    sensor_size: tuple[int, int] | None = None, path: Path | None = None
) -> tuple[CropRect | None, bool]:
    """Read the saved crop.

    Returns:
        ``(rect, present)``. ``present`` is whether a saved file was found at
        all, which is what tells a caller whether to fall back to
        ``config.yaml``; ``rect`` is ``None`` both when no file exists and when
        the file says the full frame was chosen, and ``present`` separates those.

    A corrupt file is logged and treated as absent, matching
    :func:`vision.focus.load_focus_calibration`: failing to boot over a typo in
    a calibration file is far worse than booting uncropped, and uncropped is a
    state the panel already reports.

    A crop saved against a different capture resolution is **rejected**, not
    rescaled. Rescaling would silently hand back a rectangle nobody chose --
    plausible, wrong, and framing the table slightly off with no symptom that
    points here. Re-fitting takes one tap.
    """
    source = path or crop_path()
    if not source.is_file():
        return None, False

    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"expected an object, got {type(data).__name__}")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("camera crop %s is unreadable (%s); ignoring it", source, exc)
        return None, False

    if not data.get("enabled", False):
        return None, True

    try:
        rect = CropRect(
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("camera crop %s is missing or malformed (%s); ignoring it", source, exc)
        return None, False

    if sensor_size is not None:
        saved = (int(data.get("sensor_width", 0)), int(data.get("sensor_height", 0)))
        if saved != (0, 0) and saved != (int(sensor_size[0]), int(sensor_size[1])):
            logger.warning(
                "the saved crop %s was chosen against a %dx%d frame but this rig now "
                "captures %dx%d, so it no longer means the same region. Ignoring it -- "
                "re-fit the crop from the Setup tab.",
                rect.as_dict(), saved[0], saved[1], sensor_size[0], sensor_size[1],
            )
            return None, True

    return rect, True
