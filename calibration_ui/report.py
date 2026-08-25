"""Human-readable calibration export.

Phase 6.5. The spec asks the wizard to save ``camera_calibration.yaml``,
``projector_calibration.yaml`` and ``calibration_timestamp.txt``, and this
writes all three.

These files are a **report, not the source of truth.** The application loads
``projector_calibration.json`` through :func:`projection.mapper.load_calibration`
and never reads the YAML -- so a header saying exactly that goes at the top of
every file. The alternative is somebody editing the YAML, restarting, seeing no
change, and concluding the calibration is broken.

They are still worth writing. The JSON is a flat dump of one dataclass, which
means it holds the transform and nothing else: not the table boundary the
transform was solved against, not the grid metrics, not the end-to-end
trajectory error. Those are what tell you *whether the calibration was any
good* six weeks later when the projection has drifted and nobody remembers
whether it was ever right. Keeping them next to the transform, in a format a
person can read over SSH, costs a few kilobytes.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from app.config import CALIBRATION_DIR, Settings, get_settings
from app.models import ProjectorCalibration, TableBoundary
from calibration_ui.metrics import CORNER_NAMES, GridMetrics

logger = logging.getLogger(__name__)

__all__ = ["write_calibration_report", "CAMERA_YAML", "PROJECTOR_YAML", "TIMESTAMP_FILE"]

CAMERA_YAML = "camera_calibration.yaml"
PROJECTOR_YAML = "projector_calibration.yaml"
TIMESTAMP_FILE = "calibration_timestamp.txt"

_HEADER = (
    "# Written by the GhostBall calibration wizard. READ-ONLY REPORT.\n"
    "#\n"
    "# The application does NOT load this file. The transform it actually uses\n"
    "# lives in projector_calibration.json alongside it. Editing anything here\n"
    "# changes nothing; re-run `python -m calibration_ui.calibration_app`.\n"
    "#\n"
)


def write_calibration_report(
    calibration: ProjectorCalibration,
    boundary: TableBoundary | None,
    camera_to_table: np.ndarray | None,
    grid: GridMetrics | None,
    trajectory_error_in: float | None,
    settings: Settings | None = None,
    directory: Path | None = None,
) -> dict[str, Path]:
    """Write the three report files and return where they went.

    Every argument except ``calibration`` is optional and may be ``None``,
    because the wizard can legitimately finish without some of them -- a user
    who skips the test-projection screen has no trajectory error, and that
    should produce a report saying so rather than a crash or a fabricated zero.

    Args:
        calibration: The solved transform, as saved to JSON.
        boundary: Table corners in camera px, if the table was found.
        camera_to_table: The camera->table homography matching ``boundary``.
        grid: Fine-tune screen metrics, if that screen ran.
        trajectory_error_in: End-to-end error in inches, if the test ran.
        settings: Config. Defaults to the global settings.
        directory: Where to write. Defaults to ``data/calibration``.

    Returns:
        ``{"camera": path, "projector": path, "timestamp": path}``.
    """
    settings = settings or get_settings()
    target = directory or CALIBRATION_DIR
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone()

    camera_path = target / CAMERA_YAML
    projector_path = target / PROJECTOR_YAML
    timestamp_path = target / TIMESTAMP_FILE

    _write_yaml(camera_path, _camera_document(boundary, camera_to_table, settings))
    _write_yaml(
        projector_path,
        _projector_document(calibration, grid, trajectory_error_in, settings),
    )
    timestamp_path.write_text(
        f"{stamp.isoformat(timespec='seconds')}\n"
        f"{_verdict_line(calibration, trajectory_error_in)}\n",
        encoding="utf-8",
    )

    logger.info("wrote calibration report to %s", target)
    return {"camera": camera_path, "projector": projector_path, "timestamp": timestamp_path}


def _write_yaml(path: Path, document: dict[str, object]) -> None:
    """Write one YAML file with the read-only header on top.

    Temp file then rename, matching :func:`projection.mapper.save_calibration`:
    a report truncated by a crash mid-write is a report that reads as a
    calibration failure when the calibration was fine.
    """
    body = yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(_HEADER + body, encoding="utf-8")
    temp.replace(path)


def _camera_document(
    boundary: TableBoundary | None,
    camera_to_table: np.ndarray | None,
    settings: Settings,
) -> dict[str, object]:
    """What the camera saw: the table boundary and the homography off it."""
    document: dict[str, object] = {
        "camera": {
            "width": settings.camera.width,
            "height": settings.camera.height,
            "rotation_deg": settings.camera.rotation_deg,
            "focus_absolute": settings.camera.focus_absolute,
        },
        "table": {
            "preset": settings.table_preset,
            "length_in": settings.table.length_in,
            "width_in": settings.table.width_in,
        },
    }

    if boundary is None:
        document["table_boundary"] = None
        document["notes"] = "No table boundary was recorded; detection did not succeed."
    else:
        from vision.calibration import pixels_per_inch

        document["table_boundary"] = {
            "corners_camera_px": {
                name: [round(corner.x, 2), round(corner.y, 2)]
                for name, corner in zip(CORNER_NAMES, boundary.corners(), strict=True)
            },
            "center_camera_px": [round(boundary.center.x, 2), round(boundary.center.y, 2)],
            "width_px": round(boundary.width_px, 2),
            "height_px": round(boundary.height_px, 2),
            "confidence": round(boundary.confidence, 3),
            "pixels_per_inch": round(pixels_per_inch(boundary, settings), 3),
        }

    document["perspective_matrix_camera_to_table"] = (
        None if camera_to_table is None else _matrix_rows(camera_to_table)
    )
    return document


def _projector_document(
    calibration: ProjectorCalibration,
    grid: GridMetrics | None,
    trajectory_error_in: float | None,
    settings: Settings,
) -> dict[str, object]:
    """The transform, plus the measurements that say how much to trust it."""
    data = asdict(calibration)
    document: dict[str, object] = {
        "projector": {
            "width": data["projector_width"],
            "height": data["projector_height"],
            "display_index": settings.projector.display_index,
        },
        "transform": {
            "homography_table_in_to_projector_px": data["homography"],
            "offset_x": round(data["offset_x"], 3),
            "offset_y": round(data["offset_y"], 3),
            "scale_x": round(data["scale_x"], 5),
            "scale_y": round(data["scale_y"], 5),
            "rotation_deg": round(data["rotation_deg"], 4),
        },
        "quality": {
            "corner_rmse_px": round(data["rmse_px"], 3),
            # Spelled out, because a four-point fit reports its own training
            # error and someone will otherwise read a 0.0 here as proof the
            # projection is perfect everywhere on the table.
            "corner_rmse_note": (
                "Reprojection error of the solved transform against the four corners "
                "it was fitted to. With exactly four correspondences this is near zero "
                "by construction and says nothing about accuracy mid-table -- that is "
                "what trajectory_error_in measures."
            ),
            "trajectory_error_in": (
                None if trajectory_error_in is None else round(trajectory_error_in, 3)
            ),
        },
        "created_at": data["created_at"],
        "is_calibrated": data["is_calibrated"],
    }

    quality = document["quality"]
    assert isinstance(quality, dict)
    if grid is None:
        quality["grid"] = None
    else:
        quality["grid"] = {
            "perpendicularity_deg": round(grid.perpendicularity_deg, 2),
            "rotation_deg": round(grid.rotation_deg, 2),
            "coverage_x_pct": round(grid.coverage_x_pct, 1),
            "coverage_y_pct": round(grid.coverage_y_pct, 1),
            "is_acceptable": grid.is_acceptable,
        }
    return document


def _matrix_rows(matrix: np.ndarray) -> list[list[float]]:
    """A 3x3 as plain nested floats, so PyYAML emits a readable block."""
    return [[round(float(value), 8) for value in row] for row in np.asarray(matrix)]


def _verdict_line(calibration: ProjectorCalibration, trajectory_error_in: float | None) -> str:
    """One line a person can read without opening the YAML."""
    parts = [f"corner RMSE {calibration.rmse_px:.1f} px"]
    if trajectory_error_in is not None:
        parts.append(f"trajectory error {trajectory_error_in:.2f} in")
    parts.append("calibrated" if calibration.is_calibrated else "NOT calibrated")
    return "  |  ".join(parts)
