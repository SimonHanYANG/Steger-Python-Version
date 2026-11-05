"""Geometric default with explicit legacy measurement entry points."""

from dataclasses import asdict, dataclass

import numpy as np

from .types import Contour
from .robust_measurement import RobustBendingMeasurement, measure_robust_bending


@dataclass(frozen=True)
class EndDirectionMeasurement:
    angle_deg: float
    direction_start: tuple[float, float]
    direction_end: tuple[float, float]
    start_window: tuple[float, float]
    end_window: tuple[float, float]
    start_fit_rms_px: float
    end_fit_rms_px: float
    arc_length_px: float
    cumulative_turn_deg: float
    method: str = "arclength-resampled orthogonal line fits; straight = 0 degrees"

    def to_dict(self):
        return asdict(self)


def _fit(points):
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    direction = vt[0]
    if np.dot(direction, points[-1] - points[0]) < 0:
        direction = -direction
    normal = np.array([direction[1], -direction[0]])
    rms = float(np.sqrt(np.mean(((points - center) @ normal) ** 2)))
    return direction, rms


def measure_end_direction_bending(
    contour: Contour,
    *,
    window_fraction=0.2,
    trim_fraction=0.05,
    min_window_length=3.0,
    sample_spacing=0.25,
):
    """Fit end segments; no hidden smoothing or change to the source contour.

    Reported angle depends on the explicit windows, not the mathematical
    endpoint tangents. For S-curves inspect cumulative_turn_deg as well.
    """
    if contour.closed:
        raise ValueError("Neck must be an open contour")
    if not 0 < window_fraction < 0.5 or not 0 <= trim_fraction < 0.5 - window_fraction:
        raise ValueError("Require nonoverlapping trimmed end windows")
    if sample_spacing <= 0 or min_window_length <= 0:
        raise ValueError("Lengths must be positive")
    p = contour.points
    completed = contour.attributes.get("junction_completed", np.zeros(len(p), dtype=bool))
    if completed.any():
        raise ValueError(
            "Contour contains inferred junction points; select measured neck segment first"
        )
    if len(p) < 6:
        raise ValueError("Too few centerline points for reliable two-segment measurement")
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-9]
    p, s = p[keep], s[keep]
    length = float(s[-1])
    if window_fraction * length < min_window_length:
        raise ValueError(
            "Neck is too short for requested fit windows; do not silently report a precise angle"
        )
    windows = (
        (trim_fraction * length, (trim_fraction + window_fraction) * length),
        ((1 - trim_fraction - window_fraction) * length, (1 - trim_fraction) * length),
    )
    directions = []
    rms = []
    for lo, hi in windows:
        q = np.linspace(lo, hi, max(5, int(np.ceil((hi - lo) / sample_spacing)) + 1))
        pts = np.column_stack([np.interp(q, s, p[:, k]) for k in range(2)])
        v, r = _fit(pts)
        directions.append(v)
        rms.append(r)
    v1, v2 = directions
    angle = float(np.rad2deg(np.arctan2(abs(v1[0] * v2[1] - v1[1] * v2[0]), np.dot(v1, v2))))
    # Cumulative turn at a deliberately coarser 1 px scale; never sum
    # direction differences of adjacent subpixel samples (noise explodes).
    q = np.linspace(windows[0][0], windows[1][1], max(4, int(length)))
    pts = np.column_stack([np.interp(q, s, p[:, k]) for k in range(2)])
    tangents = np.gradient(pts, axis=0)
    tangent_angles = np.unwrap(np.arctan2(tangents[:, 0], tangents[:, 1]))
    cumulative = float(np.rad2deg(np.abs(np.diff(tangent_angles)).sum()))
    return EndDirectionMeasurement(
        angle, tuple(v1), tuple(v2), windows[0], windows[1], rms[0], rms[1], length, cumulative
    )


@dataclass(frozen=True)
class BendingMeasurement:
    angle_deg: float
    point_count: int
    window_count: int
    min_window: tuple[int, int]
    max_window: tuple[int, int]
    min_angle_rad: float
    max_angle_rad: float
    smoothed_angles_rad: tuple[float, ...]
    arc_length_px: float
    filter_size: int = 10
    method: str = "cpp_legacy_10point_float32_direction_range"

    def to_dict(self):
        return asdict(self)


def measure_direction_angles(angles, *, filter_size=10, arc_length_px=0.0):
    """Match OneSperm.cpp:887-903 for a supplied Halcon-style angle sequence.

    Preserve the C++ float storage/addition/division, N-filterSize windows
    (not N-filterSize+1), raw periodic angles without unwrap, and float result.
    Detection coordinates and normals themselves remain float64.
    """
    if not isinstance(filter_size, int) or filter_size < 1:
        raise ValueError("filter_size must be a positive integer")
    raw = np.asarray(angles, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError("Expected finite one-dimensional normal angles")
    count = len(raw) - filter_size
    if count <= 0:
        raise ValueError(f"C++ rule requires more than {filter_size} direction samples")
    values = raw.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Direction samples must be representable as C++ float")
    total = np.zeros(count, dtype=np.float32)
    # Vectorize across windows, retaining the ten sequential float additions
    # within each individual window, exactly as the original C++ loop.
    for j in range(filter_size):
        np.add(total, values[j:j + count], out=total)
    smooth = np.divide(total, np.float32(filter_size), dtype=np.float32)
    lo, hi = int(np.argmin(smooth)), int(np.argmax(smooth))
    difference = np.float32(smooth[hi] - smooth[lo])
    angle = float(np.float32(float(difference) * 180.0 / np.pi))
    return BendingMeasurement(
        angle_deg=angle, point_count=len(raw), window_count=count,
        min_window=(lo, lo + filter_size), max_window=(hi, hi + filter_size),
        min_angle_rad=float(smooth[lo]), max_angle_rad=float(smooth[hi]),
        smoothed_angles_rad=tuple(map(float, smooth)), arc_length_px=float(arc_length_px),
        filter_size=filter_size, method=f"cpp_legacy_{filter_size}point_float32_direction_range",
    )


def measure_cpp_bending(contour: Contour):
    """Legacy neck angle: 10-point averaged normal-angle maximum minus minimum.

    No arclength resampling, endpoint fits or angle unwrapping. At least 11
    ordered, measured points are required by the original C++ loop bounds.
    """
    if contour.closed:
        raise ValueError("Neck must be an open contour")
    if np.any(contour.attributes.get("junction_completed", [])):
        raise ValueError("Contour contains inferred junction points")
    angles = contour.get_attribute("angle")
    return measure_direction_angles(angles, arc_length_px=contour.length)


def measure_bending(contour: Contour, **kwargs):
    """Default: continuous local geometric direction range with tip validation."""
    return measure_robust_bending(contour, **kwargs)


class NeckBendingHistory:
    """One instance per tracked sperm: C++ m_neckAngleAll temporal median.

    Independent ROIs must never share this history. Invalid frames do not add a
    sample; the previous median is retained, as in the original application.
    """

    def __init__(self):
        self._angles = []

    def update(self, angle_deg):
        if angle_deg is not None:
            value = np.float32(angle_deg)
            if not np.isfinite(value):
                raise ValueError("angle_deg must be finite or None")
            self._angles.append(value)
        return None if not self._angles else float(np.median(np.asarray(self._angles, dtype=np.float32)))

    def reset(self):
        self._angles.clear()
