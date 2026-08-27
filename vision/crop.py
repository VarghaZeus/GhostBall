"""Digital crop on the captured frame.

The IMX708's 120-degree field of view is a great deal wider than a pool table
needs. Everything outside the cushions is cost without benefit: pixels to
downscale, felt-coloured clutter to reject, and a smaller table within the frame
than the detector would like. Cropping is the cheapest possible fix -- a NumPy
slice -- and it makes every stage after it faster rather than slower.

Two coordinate systems, and keeping them apart is the whole job here
---------------------------------------------------------------------

**Sensor space.** The full captured frame *after rotation*. This is the space a
:class:`CropRect` lives in, and the space the crop is stored and configured in.
It does not change when the crop changes, which is what makes it usable as the
frame of reference for everything the panel does.

**Frame space.** What the rest of the program sees: the cropped sub-rectangle,
with its own origin at its own top-left. Detected balls, pockets, table corners
and the camera->table homography are all in frame space, because that is the
only image any of them is ever handed.

The conversion is a translation by the crop origin::

    sensor = frame + (crop.x, crop.y)
    frame  = sensor - (crop.x, crop.y)

That is trivial, and it is exactly the kind of trivial that gets skipped. A
detected pocket is in frame space; a proposed crop is in sensor space; comparing
them directly "works" whenever the current crop happens to start at the origin,
which is the default, which is the state every test starts in. So the conversion
is a named function with its own tests rather than two additions inline.

Why the crop happens where it does
----------------------------------

In :meth:`vision.camera.Camera.capture_frame`, after rotation, before anything
else in the system sees a pixel. Two consequences worth stating:

* **Nothing downstream knows.** There is one frame coordinate system at a time,
  and "full frame" simply comes to mean "the cropped frame". No consumer needs
  an offset threaded through it, so no consumer can forget to apply one -- which
  would be wrong by a constant and would present as a calibration fault.
* **After rotation, not before.** The panel's pan controls have to mean the same
  thing at any ``rotation_deg``. Cropping first would make "pan left" depend on
  how the camera is mounted.

The crop is a *crop*, not a digital zoom: the output frame is smaller and is
never scaled back up. That choice is what keeps absolute-pixel quantities
literally true -- a ball is the same number of pixels across whether or not the
frame around it was trimmed -- and it keeps sharpness measurements comparable,
since nothing is resampled. See :data:`vision.crop.__doc__` callers in
``app/config.py`` for the one quantity this does *not* hold for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_CROP_PX",
    "CropRect",
    "fit_to_table",
    "frame_to_sensor",
    "pockets_outside",
    "sensor_to_frame",
]

#: Smallest crop edge, in sensor px. Below this the table cannot plausibly fit
#: and the preview becomes unusable for setting the very control that produced
#: it -- so the zoom stops here rather than letting someone zoom into a corner
#: and lose the picture they were aiming with.
MIN_CROP_PX = 160


@dataclass(frozen=True, slots=True)
class CropRect:
    """A crop rectangle in sensor space (the rotated full frame).

    Integer px, because this indexes a NumPy array. Half-pixel crops are not a
    thing, and rounding at the point of use rather than here would let two
    callers disagree about the same rectangle.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        """Exclusive right edge, as a slice bound."""
        return self.x + self.width

    @property
    def y1(self) -> int:
        """Exclusive bottom edge, as a slice bound."""
        return self.y + self.height

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def full(cls, frame_width: int, frame_height: int) -> CropRect:
        """The whole frame -- the identity crop."""
        return cls(x=0, y=0, width=int(frame_width), height=int(frame_height))

    def is_full(self, frame_width: int, frame_height: int) -> bool:
        return (
            self.x == 0
            and self.y == 0
            and self.width == int(frame_width)
            and self.height == int(frame_height)
        )

    def clamped(self, frame_width: int, frame_height: int) -> CropRect:
        """This rectangle, forced inside a frame of the given size.

        Clamps the size first and then the origin, so a rectangle that is too
        big becomes a valid one at the frame edge rather than a valid size at an
        impossible origin. Never returns a zero or negative extent: a crop that
        slices an empty array would surface far away from here as "the camera
        stopped producing frames".
        """
        frame_width, frame_height = int(frame_width), int(frame_height)
        width = max(1, min(int(self.width), frame_width))
        height = max(1, min(int(self.height), frame_height))
        x = max(0, min(int(self.x), frame_width - width))
        y = max(0, min(int(self.y), frame_height - height))
        return CropRect(x=x, y=y, width=width, height=height)

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Whether a sensor-space point sits inside, with ``margin`` to spare."""
        return (
            self.x + margin <= x <= self.x1 - margin
            and self.y + margin <= y <= self.y1 - margin
        )

    def zoomed(self, factor: float, frame_width: int, frame_height: int) -> CropRect:
        """Scaled about its own centre, then clamped.

        About the centre rather than the origin, because zooming is what the
        operator thinks they are doing to the *picture*: a zoom that walked the
        subject towards a corner would need a pan after every step to undo it.
        """
        centre_x = self.x + self.width / 2.0
        centre_y = self.y + self.height / 2.0
        width = max(MIN_CROP_PX, round(self.width * factor))
        height = max(MIN_CROP_PX, round(self.height * factor))
        return CropRect(
            x=round(centre_x - width / 2.0),
            y=round(centre_y - height / 2.0),
            width=width,
            height=height,
        ).clamped(frame_width, frame_height)

    def panned(self, dx: int, dy: int, frame_width: int, frame_height: int) -> CropRect:
        """Translated, then clamped. The size is preserved."""
        return CropRect(
            x=self.x + int(dx), y=self.y + int(dy), width=self.width, height=self.height
        ).clamped(frame_width, frame_height)


def frame_to_sensor(x: float, y: float, crop: CropRect) -> tuple[float, float]:
    """Frame-space point -> sensor-space point.

    Named, rather than written inline at each of its four call sites, because
    the bug it prevents is invisible in the default configuration: with no crop
    the origin is ``(0, 0)`` and omitting this is indistinguishable from
    applying it.
    """
    return x + crop.x, y + crop.y


def sensor_to_frame(x: float, y: float, crop: CropRect) -> tuple[float, float]:
    """Sensor-space point -> frame-space point. Inverse of :func:`frame_to_sensor`."""
    return x - crop.x, y - crop.y


def pockets_outside(
    proposed: CropRect, pockets, current: CropRect, margin_px: float = 0.0
) -> list[str]:
    """Names of detected pockets a proposed crop would cut off.

    Detection needs all six pockets: they are what
    :func:`vision.pockets.dynamic_detect_table_size` fits a table to, and what
    every pot is judged against. A crop that loses one does not degrade
    gracefully -- it removes the thing the table's identity is established from.
    So this is a refusal, not a warning.

    ``pockets`` are in frame space (the current crop); ``proposed`` and
    ``current`` are in sensor space. The whole pocket has to survive, not just
    its centre, so each is tested at its own radius plus ``margin_px``.

    Returns:
        Pocket names, in detection order. Empty means the crop is safe. An
        empty ``pockets`` list also returns empty -- nothing is known to be at
        risk, which is a different statement from "this crop is fine" and is the
        caller's to interpret.
    """
    lost: list[str] = []
    for pocket in pockets or []:
        centre = getattr(pocket, "center_px", None)
        if centre is None:
            continue
        radius = float(getattr(pocket, "radius_px", 0.0) or 0.0)
        sensor_x, sensor_y = frame_to_sensor(float(centre.x), float(centre.y), current)
        if not proposed.contains_point(sensor_x, sensor_y, margin=radius + margin_px):
            identifier = getattr(pocket, "id", None)
            lost.append(getattr(identifier, "value", None) or str(identifier or "pocket"))
    return lost


def fit_to_table(
    boundary,
    pockets,
    current: CropRect,
    frame_width: int,
    frame_height: int,
    margin_frac: float = 0.06,
) -> CropRect:
    """Crop to the detected table, plus a margin.

    The union of the table corners *and* every detected pocket, not just the
    cloth. Pockets sit outside the playing surface -- in the rails, at the
    corners and the middle of the long cushions -- so a rectangle fitted to the
    boundary alone clips them at any margin small enough to be worth applying.
    Including them means the result satisfies :func:`pockets_outside` by
    construction rather than by a margin that happens to be generous enough.

    The margin is a fraction of the union's longer edge, so it scales with how
    much of the frame the table occupies instead of being a pixel count that is
    generous on a wide view and negligible on a tight one.

    Args:
        boundary: Detected :class:`app.models.TableBoundary`, in frame space.
        pockets: Detected pockets, in frame space. May be empty.
        current: The crop those detections were made under.
        margin_frac: Padding as a fraction of the union's longer edge.

    Returns:
        A clamped :class:`CropRect` in sensor space.
    """
    xs: list[float] = []
    ys: list[float] = []

    for corner in boundary.corners():
        sensor_x, sensor_y = frame_to_sensor(float(corner.x), float(corner.y), current)
        xs.append(sensor_x)
        ys.append(sensor_y)

    for pocket in pockets or []:
        centre = getattr(pocket, "center_px", None)
        if centre is None:
            continue
        radius = float(getattr(pocket, "radius_px", 0.0) or 0.0)
        sensor_x, sensor_y = frame_to_sensor(float(centre.x), float(centre.y), current)
        xs.extend((sensor_x - radius, sensor_x + radius))
        ys.extend((sensor_y - radius, sensor_y + radius))

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    margin = max(x1 - x0, y1 - y0) * float(margin_frac)

    return CropRect(
        x=round(x0 - margin),
        y=round(y0 - margin),
        width=max(MIN_CROP_PX, round((x1 - x0) + 2 * margin)),
        height=max(MIN_CROP_PX, round((y1 - y0) + 2 * margin)),
    ).clamped(frame_width, frame_height)
