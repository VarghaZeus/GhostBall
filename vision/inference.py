"""Optional Hailo AI HAT+ accelerated detection.

Phase 2.4, and genuinely optional: the OpenCV path in :mod:`vision.detection` is
the primary implementation and must stay complete on its own. The availability
check here is implemented so callers can branch on it today; the inference calls
are stubs.

Why bother, given classical CV already finds circles on green felt? Two cases it
handles badly and a trained detector handles well: balls under projected
overlay light, and balls partially occluded by a hand or the cue. If those turn
out not to matter in practice, this module can stay unimplemented forever
without blocking anything -- which is the point of keeping the seam narrow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import Settings
from app.models import Ball

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    """One raw detector output, before it becomes a :class:`~app.models.Ball`."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def radius(self) -> float:
        """Half the mean box side. Balls are circular, so the box is square-ish
        and averaging the two sides is more stable than trusting either."""
        return (abs(self.x2 - self.x1) + abs(self.y2 - self.y1)) / 4.0


@dataclass(slots=True)
class InferenceResult:
    """Output of one accelerated inference pass."""

    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    used_hailo: bool = False


@lru_cache(maxsize=1)
def is_hailo_available() -> bool:
    """Whether the Hailo runtime is importable and a device is present.

    Cached: probing the device costs tens of ms, and the answer cannot change
    without a reboot. Deliberately broad in what it catches -- any failure here
    means "use OpenCV", and a HAT that is present but misbehaving should degrade
    exactly like one that is absent rather than take the game down.
    """
    try:
        from hailo_platform import VDevice  # type: ignore[import-not-found]
    except ImportError:
        logger.info("hailort not installed; using the OpenCV detection path")
        return False

    try:
        with VDevice():
            logger.info("Hailo device present")
            return True
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("Hailo runtime present but no usable device (%s)", exc)
        return False


def init_hailo_runtime(settings: Settings | None = None) -> object | None:
    """Open the Hailo device and load the compiled HEF model.

    Returns ``None`` rather than raising when Hailo is unavailable or the HEF is
    missing, so the caller's fallback is a plain ``if runtime is None`` and not
    an exception handler.

    The model itself is out of scope for this codebase -- it is a YOLOv8n trained
    on pool balls and compiled to HEF with the Hailo Dataflow Compiler. Point
    ``vision.hailo_model_path`` at the result.
    """
    if not is_hailo_available():
        return None

    settings = settings or __import__("app.config", fromlist=["get_settings"]).get_settings()
    model_path = Path(settings.vision.hailo_model_path)
    if not model_path.is_absolute():
        from app.config import PACKAGE_ROOT

        model_path = PACKAGE_ROOT / model_path

    if not model_path.is_file():
        logger.warning(
            "Hailo is available but no HEF model at %s; using the OpenCV path",
            model_path,
        )
        return None

    raise NotImplementedError("Phase 2.4: implement HEF load and network-group setup")


def run_hailo_inference(runtime: object, frame: np.ndarray) -> InferenceResult:
    """Run one accelerated detection pass.

    Must return an empty :class:`InferenceResult` rather than raise on a
    transient device error, so a single failed inference degrades that frame
    only.
    """
    raise NotImplementedError("Phase 2.4: implement Hailo inference")


def parse_hailo_detections(
    result: InferenceResult, settings: Settings | None = None
) -> list[Ball]:
    """Convert raw detections into :class:`~app.models.Ball` objects.

    The class-id to ball-number mapping is a property of the trained model, so
    it belongs next to the model definition rather than hardcoded here. Same
    contract as :func:`vision.detection.detect_balls`: ``table_pos`` stays
    ``None`` for the caller to fill in.
    """
    raise NotImplementedError("Phase 2.4: implement detection -> Ball mapping")
