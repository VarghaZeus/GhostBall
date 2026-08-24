"""Table inches -> projector pixels.

Implemented rather than stubbed: it is pure geometry, it is what the whole
projection illusion rests on, and a 20 px alignment target means it needs to be
under test rather than eyeballed.

Two transforms are supported. The **homography** path is the accurate one and is
used whenever the calibration wizard has solved one from four corner
correspondences -- it handles the keystoning you always get from a projector
that is not perfectly perpendicular to the table. The **affine** path
(offset/scale/rotation) is the fallback and what the fine-tune nudge controls
manipulate; it cannot represent keystone, so it is only good enough for a
squarely-mounted projector.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.config import CALIBRATION_FILE, Settings, get_settings
from app.models import ProjectorCalibration, TableBoundary, Vec2

logger = logging.getLogger(__name__)


class ProjectionMapper:
    """Applies a stored calibration to convert table coords to projector pixels.

    Holds the matrix as NumPy state so batch conversion of a whole trajectory is
    one matmul, and caches the inverse, which the calibration UI needs for
    click-to-table hit testing.
    """

    def __init__(self, calibration: ProjectorCalibration) -> None:
        self.calibration = calibration
        self._matrix: np.ndarray | None = None
        self._inverse: np.ndarray | None = None
        self._rebuild()

    def _rebuild(self) -> None:
        """Materialise the forward and inverse matrices from the calibration."""
        cal = self.calibration
        if cal.homography is not None:
            matrix = np.array(cal.homography, dtype=np.float64)
            if matrix.shape != (3, 3):
                raise ValueError(f"homography must be 3x3, got {matrix.shape}")
        else:
            matrix = self._affine_matrix(cal)

        self._matrix = matrix
        try:
            self._inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            # A singular forward transform means a broken calibration file.
            # Forward projection would produce nonsense too, so fail here rather
            # than let it through and debug it as a mysterious alignment bug.
            raise ValueError(
                "calibration transform is singular and cannot be inverted"
            ) from exc

    @staticmethod
    def _affine_matrix(cal: ProjectorCalibration) -> np.ndarray:
        """Build a 3x3 from the human-legible offset/scale/rotation fields.

        Order is scale, then rotate, then translate. Rotating before scaling
        would make a non-uniform scale shear the image, which is not what the
        fine-tune controls are meant to do.
        """
        theta = math.radians(cal.rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return np.array(
            [
                [cal.scale_x * cos_t, -cal.scale_y * sin_t, cal.offset_x],
                [cal.scale_x * sin_t, cal.scale_y * cos_t, cal.offset_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    # -- forward / inverse --------------------------------------------------

    def table_to_projector(self, point: Vec2) -> Vec2:
        """Table inches -> projector px. Result may fall outside the frame."""
        assert self._matrix is not None
        m = self._matrix
        x, y = point
        denom = m[2, 0] * x + m[2, 1] * y + m[2, 2]
        if abs(denom) < 1e-12:
            # On the transform's horizon. Clamp to the frame centre rather than
            # raise: one bad trajectory point must not kill the render.
            logger.debug("table point %s maps to infinity; clamping", point)
            return Vec2(self.calibration.projector_width / 2.0, self.calibration.projector_height / 2.0)
        return Vec2(
            (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / denom,
            (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / denom,
        )

    def table_to_projector_batch(self, points: list[Vec2]) -> np.ndarray:
        """Convert many points at once.

        Returns an ``Nx2`` float64 array, which is the shape OpenCV's polyline
        drawing wants after an astype -- avoiding a per-point Python round trip
        on every rendered trajectory.
        """
        assert self._matrix is not None
        if not points:
            return np.empty((0, 2), dtype=np.float64)
        array = np.array([[p.x, p.y, 1.0] for p in points], dtype=np.float64)
        projected = array @ self._matrix.T
        w = projected[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, np.nan, w)
        return projected[:, :2] / w

    def projector_to_table(self, point: Vec2) -> Vec2:
        """Projector px -> table inches. Used for click input in the calibration UI."""
        assert self._inverse is not None
        m = self._inverse
        x, y = point
        denom = m[2, 0] * x + m[2, 1] * y + m[2, 2]
        if abs(denom) < 1e-12:
            raise ValueError(f"projector point {point} has no table-plane preimage")
        return Vec2(
            (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / denom,
            (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / denom,
        )

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 table -> projector transform.

        Exposed read-only for callers that need to *compose* transforms rather
        than apply them -- the web preview warps the projector-space overlay into
        camera space, which is this matrix inverted and then multiplied by the
        table -> camera homography. Doing that through repeated
        :meth:`table_to_projector` calls would mean a per-pixel Python loop.
        """
        assert self._matrix is not None
        return self._matrix

    @property
    def inverse_matrix(self) -> np.ndarray:
        """The 3x3 projector -> table transform. See :attr:`matrix`."""
        assert self._inverse is not None
        return self._inverse

    def is_on_screen(self, point: Vec2) -> bool:
        """Whether a projector-space point is inside the output frame."""
        return (
            0 <= point.x < self.calibration.projector_width
            and 0 <= point.y < self.calibration.projector_height
        )

    def pixels_per_inch(self, at: Vec2 | None = None) -> float:
        """Local scale of the transform in projector px per table inch.

        The renderer needs this constantly -- a ball radius, a marker size and a
        text height are all specified in inches so they stay physically
        consistent, and every one of them has to become pixels before it can be
        drawn.

        Measured as a local derivative rather than read off the calibration's
        ``scale_x``, because under a homography the scale genuinely varies across
        the table: the far end of a keystoned projection is smaller, and a
        single global number would size the near and far ghost balls
        identically when they should differ by 10-20%. Sampling one inch in each
        axis and averaging is the finite-difference form of the Jacobian
        determinant's square root, which is the scalar "how many pixels is an
        inch here" that the caller actually wants.

        Args:
            at: Table point to measure at. Defaults to whatever table point
                lands in the middle of the projector frame, which is the right
                default because that is where most of the drawing happens.
        """
        if at is None:
            center = Vec2(
                self.calibration.projector_width / 2.0,
                self.calibration.projector_height / 2.0,
            )
            try:
                at = self.projector_to_table(center)
            except ValueError:
                # Frame centre has no table preimage -- a badly degenerate
                # calibration. The origin is at least on the table.
                at = Vec2(0.0, 0.0)

        origin = self.table_to_projector(at)
        along_x = self.table_to_projector(Vec2(at.x + 1.0, at.y))
        along_y = self.table_to_projector(Vec2(at.x, at.y + 1.0))
        scale = (origin.distance_to(along_x) + origin.distance_to(along_y)) / 2.0
        # A zero here would make every marker vanish and every division by it
        # blow up. It means the transform has collapsed, so fall back to a
        # plausible scale (a 76 in table across a 1920 px frame is ~25 px/in)
        # and let the visibly wrong overlay be the signal.
        if not math.isfinite(scale) or scale < 1e-6:
            logger.warning("degenerate projector scale at table point %s", at)
            return self.calibration.projector_width / 76.0
        return scale

    # -- mutation -----------------------------------------------------------

    def nudge(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dscale: float = 0.0,
        drotation: float = 0.0,
    ) -> None:
        """Adjust the affine parameters, for the fine-tune screen's arrow keys.

        Discards any homography, because mixing a nudge into a solved keystone
        transform gives an incoherent result -- if the user is nudging, they are
        working in the affine model. The wizard should warn before the first
        nudge on a keystone-calibrated setup.
        """
        cal = self.calibration
        if cal.homography is not None:
            logger.warning("nudging discards the solved homography; keystone will be lost")
            cal.homography = None
        cal.offset_x += dx
        cal.offset_y += dy
        cal.scale_x += dscale
        cal.scale_y += dscale
        cal.rotation_deg += drotation
        self._rebuild()


# ---------------------------------------------------------------------------
# Construction and persistence
# ---------------------------------------------------------------------------


def solve_projector_homography(
    table_points: list[Vec2],
    projector_points: list[Vec2],
    projector_width: int,
    projector_height: int,
) -> ProjectorCalibration:
    """Solve the table -> projector transform from corner correspondences.

    Four points give an exact solution via ``getPerspectiveTransform``; more
    than four are least-squared via ``findHomography``, which is better -- extra
    correspondences from the fine-tune step let the RMSE actually mean
    something, since a 4-point exact fit always reports zero error whether or
    not it is right.

    Args:
        table_points: Known table positions, inches. At least 4.
        projector_points: Where the user aligned each one, projector px.
        projector_width: Output width, px.
        projector_height: Output height, px.

    Returns:
        A calibration with ``homography`` and ``rmse_px`` populated.

    Raises:
        ValueError: On mismatched or insufficient input, or if the solve fails.
    """
    import cv2

    if len(table_points) != len(projector_points):
        raise ValueError(
            f"got {len(table_points)} table points but {len(projector_points)} projector points"
        )
    if len(table_points) < 4:
        raise ValueError(f"need at least 4 correspondences, got {len(table_points)}")

    src = np.array([[p.x, p.y] for p in table_points], dtype=np.float32)
    dst = np.array([[p.x, p.y] for p in projector_points], dtype=np.float32)

    if len(table_points) == 4:
        matrix = cv2.getPerspectiveTransform(src, dst)
    else:
        matrix, _mask = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError("homography solve failed; check for duplicate points")

    matrix = matrix.astype(np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("homography solve produced non-finite values")

    rmse = _reprojection_rmse(src, dst, matrix)
    logger.info(
        "solved projector calibration from %d points, RMSE %.2f px",
        len(table_points),
        rmse,
    )

    return ProjectorCalibration(
        projector_width=projector_width,
        projector_height=projector_height,
        homography=matrix.tolist(),
        rmse_px=rmse,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        is_calibrated=True,
    )


def _reprojection_rmse(src: np.ndarray, dst: np.ndarray, matrix: np.ndarray) -> float:
    """RMSE in projector px of the fitted transform against its own inputs.

    Reported to the user as alignment quality, and checked against the 20 px
    target. Note this is training error -- with exactly four points it is
    structurally zero and says nothing about accuracy elsewhere on the table.
    """
    homogeneous = np.hstack([src.astype(np.float64), np.ones((len(src), 1))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    predicted = projected[:, :2] / w
    errors = np.linalg.norm(predicted - dst.astype(np.float64), axis=1)
    return float(np.sqrt(np.nanmean(errors**2)))


def identity_calibration(settings: Settings | None = None) -> ProjectorCalibration:
    """A calibration that stretches the table to fill the projector frame.

    Not a real calibration -- there is no keystone correction and the alignment
    will be visibly off -- but it lets the pipeline render something before the
    wizard has been run, which makes the system debuggable on first boot.
    ``is_calibrated`` stays ``False`` so the UI can say so.
    """
    settings = settings or get_settings()
    return ProjectorCalibration(
        projector_width=settings.projector.width,
        projector_height=settings.projector.height,
        scale_x=settings.projector.width / settings.table.length_in,
        scale_y=settings.projector.height / settings.table.width_in,
        is_calibrated=False,
    )


def save_calibration(
    calibration: ProjectorCalibration, path: Path | None = None
) -> Path:
    """Persist a calibration as JSON.

    Written to a temp file and then moved into place, so a crash mid-write
    cannot leave a truncated file that fails to parse on next boot -- which
    would mean re-running the whole wizard.
    """
    from dataclasses import asdict

    target = path or CALIBRATION_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")
    temp.replace(target)
    logger.info("saved calibration to %s (RMSE %.2f px)", target, calibration.rmse_px)
    return target


def load_calibration(path: Path | None = None) -> ProjectorCalibration | None:
    """Load a saved calibration.

    Returns ``None`` when there is nothing to load or the file is unreadable --
    the caller falls back to :func:`identity_calibration` and prompts the user to
    calibrate. A corrupt file is logged loudly but does not raise, because
    failing to boot is worse than booting uncalibrated.
    """
    source = path or CALIBRATION_FILE
    if not source.is_file():
        logger.info("no calibration file at %s", source)
        return None

    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        calibration = ProjectorCalibration(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("calibration file %s is unreadable (%s); recalibration needed", source, exc)
        return None

    logger.info(
        "loaded calibration from %s (RMSE %.2f px, calibrated=%s)",
        source,
        calibration.rmse_px,
        calibration.is_calibrated,
    )
    return calibration


def init_projector_calibration(settings: Settings | None = None) -> ProjectionMapper:
    """Build a mapper from the saved calibration, or an identity fallback."""
    calibration = load_calibration() or identity_calibration(settings)
    return ProjectionMapper(calibration)


def table_boundary_to_projector_quad(
    boundary: TableBoundary, mapper: ProjectionMapper, settings: Settings | None = None
) -> list[Vec2]:
    """The table outline in projector px, for drawing a border during setup.

    Takes the corners from config rather than from ``boundary``, since the
    boundary's corners are in camera space and the mapper's input space is table
    inches. ``boundary`` is accepted only to make the call site read naturally
    and to assert the table has in fact been found.
    """
    settings = settings or get_settings()
    length, width = settings.table.length_in, settings.table.width_in
    corners = [Vec2(0, 0), Vec2(length, 0), Vec2(length, width), Vec2(0, width)]
    return [mapper.table_to_projector(c) for c in corners]
