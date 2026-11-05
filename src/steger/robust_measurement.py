"""Arc-length local geometry directions with auditable terminal quality checks.

The detector's original points/normals are never overwritten. Normal axes help
identify an inconsistent terminal trace, but raw Hessian angles are not averaged
to calculate bending. Defaults are geometry rules, independent of sample IDs.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .types import Contour


@dataclass(frozen=True)
class RobustBendingMeasurement:
    angle_deg: float
    point_count: int
    original_point_count: int
    valid_point_start: int
    valid_point_stop: int
    excluded_indices: tuple[int, ...]
    arc_length_px: float
    original_arc_length_px: float
    min_window_arc_px: tuple[float, float]
    max_window_arc_px: tuple[float, float]
    direction_min: tuple[float, float]
    direction_max: tuple[float, float]
    window_starts_arc_px: tuple[float, ...]
    direction_angles_rad: tuple[float, ...]
    max_fit_rms_px: float
    quality_flags: tuple[str, ...]
    window_fraction: float = .2
    trim_fraction: float = .05
    method: str = "geometry_arclength_direction_range_v2"

    def to_dict(self):
        return asdict(self)


def _axis_error(normal, tangent):
    a,b=normal
    x,y=tangent
    if a*a+b*b <= 1e-24:
        return 90.0
    # atan2 is scale invariant and avoids constructing/normalizing small arrays.
    return math.degrees(math.atan2(abs(a*x+b*y),abs(a*y-b*x)))


def _tls_directions(centered, travel):
    """Closed-form 2D orthogonal TLS, equivalent to the first right SVD vector.

    Center first to avoid subtracting large coordinate moments. Use float64
    atan2 for the principal covariance axis; reject an undetermined axis.
    """
    rr = np.sum(centered[..., 0]**2, axis=1)
    cc = np.sum(centered[..., 1]**2, axis=1)
    rc = np.sum(centered[..., 0] * centered[..., 1], axis=1)
    separation = np.hypot(cc - rr, 2 * rc)
    if np.any(separation <= 1e-10 * (rr + cc)):
        raise ValueError("Local fit has no determined principal direction")
    theta = .5 * np.arctan2(2 * rc, cc - rr)
    directions = np.column_stack((np.sin(theta), np.cos(theta)))
    alignment = np.einsum('ij,ij->i', directions, travel)
    if np.any(np.abs(alignment) <= 1e-10):
        raise ValueError("Local fit has no determined traversal direction")
    directions[alignment < 0] *= -1
    return directions


def _terminal_segment(points, normals, cumulative):
    """Exclude only demonstrably inconsistent tips; bounded by physical length.

    A consistent smooth hairpin is retained. A reversal alone is insufficient:
    the neighboring normal must also disagree with its inward segment. At an
    endpoint a >45deg normal/edge mismatch is itself evidence of inconsistency.
    Never remove an interior point, join a gap or change measured coordinates.
    """
    start, stop = 0, len(points)
    original_length = cumulative[-1]
    while stop - start >= 7:
        changed = False
        for side in ("start", "end"):
            if stop - start <= 6:
                break
            p, n = points[start:stop], normals[start:stop]
            if side == "start":
                outer,inner=p[1]-p[0],p[2]-p[1]
                normal_error = _axis_error(n[0], outer)
                reversal = np.dot(outer,inner) < 0
                neighbor_error = _axis_error(n[1],inner) if reversal else 0.
                proposed_start, proposed_stop = start + 1, stop
            else:
                outer,inner=p[-1]-p[-2],p[-2]-p[-3]
                normal_error = _axis_error(n[-1], outer)
                reversal = np.dot(outer,inner) < 0
                neighbor_error = _axis_error(n[-2],inner) if reversal else 0.
                proposed_start, proposed_stop = start, stop - 1
            if not (normal_error > 45. or (reversal and neighbor_error > 45.)):
                continue
            retained = cumulative[proposed_stop - 1] - cumulative[proposed_start]
            left_loss = cumulative[proposed_start]
            right_loss = original_length - cumulative[proposed_stop - 1]
            if (retained < .75 * original_length or left_loss > .2 * original_length
                    or right_loss > .2 * original_length):
                raise ValueError("Unreliable terminal region exceeds the usable-segment budget")
            start, stop = proposed_start, proposed_stop
            changed = True
        if not changed:
            break
    return start, stop


def _fit_direction_windows(points, arc, fraction, trim=.05, spacing=.25, step=1.):
    length=float(arc[-1]);width=fraction*length
    first=trim*length;last=(1-trim-fraction)*length
    count=max(2,int(np.ceil((last-first)/step))+1)
    starts=np.linspace(first,last,count)
    offsets=np.linspace(0.,width,max(5,int(np.ceil(width/spacing))+1))
    q=starts[:,None]+offsets[None,:]
    samples=np.stack([np.interp(q,arc,points[:,axis]) for axis in range(2)],axis=-1)
    centered=samples-samples.mean(axis=1,keepdims=True)
    directions=_tls_directions(centered,samples[:,-1]-samples[:,0])
    angles=np.unwrap(np.arctan2(directions[:,0],directions[:,1]))
    return starts,centered,directions,angles


def window_direction_ranges(contour, measurement, fractions=(.15,.25)):
    """Reuse the validated segment for sensitivity; identical fit computation."""
    points=contour.points[measurement.valid_point_start:measurement.valid_point_stop]
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-9]
    points=points[keep]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    values=[dict(window_fraction=measurement.window_fraction,angle_deg=measurement.angle_deg)]
    for fraction in fractions:
        if fraction*arc[-1]<3.:
            continue
        try:
            *_,angles=_fit_direction_windows(points,arc,fraction,measurement.trim_fraction)
        except ValueError:
            continue
        values.append(dict(window_fraction=fraction,angle_deg=float(np.degrees(np.ptp(angles)))))
    return sorted(values,key=lambda item:item['window_fraction'])


def measure_robust_bending(contour: Contour, *, window_fraction=.2,
                           trim_fraction=.05, min_window_length=3.,
                           sample_spacing=.25, direction_spacing=1.):
    """Range of continuous, forward-facing TLS directions along the whole neck.

    Fit equal-arc-length windows throughout 5–95% of the valid segment. Include
    interior windows so opposite bends do not cancel as endpoint differences do.
    Use float64, true arc-length positions, orthogonal TLS and atan2/unwrap, keeping
    normal sign flips and 0/360 degree encoding out of the angle computation.
    """
    if contour.closed:
        raise ValueError("Neck must be an open contour")
    if np.any(contour.attributes.get("junction_completed", [])):
        raise ValueError("Contour contains inferred junction points")
    if not 0 < window_fraction < .5 or not 0 <= trim_fraction < .5 - window_fraction:
        raise ValueError("Require nonoverlapping trimmed end windows")
    if min(sample_spacing, direction_spacing, min_window_length) <= 0:
        raise ValueError("Lengths must be positive")
    original = np.asarray(contour.points, dtype=np.float64)
    normals = np.asarray(contour.normals, dtype=np.float64)
    if (original.ndim != 2 or original.shape[1] != 2 or normals.shape != original.shape
            or not np.isfinite(original).all() or not np.isfinite(normals).all()):
        raise ValueError("Expected finite Nx2 coordinates and normals")
    if len(original) < 6:
        raise ValueError("Too few centerline points for geometric measurement")
    keep = np.r_[True, np.linalg.norm(np.diff(original, axis=0), axis=1) > 1e-9]
    ids = np.flatnonzero(keep)
    points, normals = original[keep], normals[keep]
    if len(points) < 6:
        raise ValueError("Too few distinct centerline points")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    original_arc = np.r_[0., np.cumsum(lengths)]
    start, stop = _terminal_segment(points, normals, original_arc)
    valid_ids = ids[start:stop]
    points, normals = points[start:stop], normals[start:stop]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    length = float(arc[-1])
    width = window_fraction * length
    if width < min_window_length:
        raise ValueError("Neck is too short for a 3-pixel geometric fit window")
    starts,centered,directions,angles=_fit_direction_windows(
        points,arc,window_fraction,trim_fraction,sample_spacing,direction_spacing)
    low, high = int(np.argmin(angles)), int(np.argmax(angles))
    right = np.column_stack((directions[:, 1], -directions[:, 0]))
    residuals = np.einsum('wki,wi->wk', centered, right)
    rms = np.sqrt(np.mean(residuals**2, axis=1))
    flags = []
    if start or stop != len(ids):
        flags.append("terminal_samples_excluded")
    if np.any(~keep):
        flags.append("duplicate_samples_ignored")
    if max(_axis_error(normals[0],points[1]-points[0]),
           _axis_error(normals[-1],points[-1]-points[-2])) > 30.:
        flags.append("terminal_direction_requires_review")
    if float(rms.max()) > .25:
        flags.append("nonlinear_fit_window")
    if np.any(np.abs(np.diff(angles)) > np.pi / 4):
        flags.append("abrupt_local_direction_change")
    origin_offset = float(original_arc[start])
    original_length = float(original_arc[-1])
    valid_set = set(valid_ids.tolist())
    excluded = tuple(i for i in range(len(original)) if i not in valid_set)
    return RobustBendingMeasurement(
        angle_deg=float(np.degrees(angles[high] - angles[low])), point_count=len(points),
        original_point_count=len(original), valid_point_start=int(valid_ids[0]),
        valid_point_stop=int(valid_ids[-1]) + 1, excluded_indices=excluded,
        arc_length_px=length, original_arc_length_px=original_length,
        min_window_arc_px=(float(starts[low] + origin_offset), float(starts[low] + width + origin_offset)),
        max_window_arc_px=(float(starts[high] + origin_offset), float(starts[high] + width + origin_offset)),
        direction_min=tuple(map(float, directions[low])), direction_max=tuple(map(float, directions[high])),
        window_starts_arc_px=tuple(map(float, starts + origin_offset)),
        direction_angles_rad=tuple(map(float, angles)), max_fit_rms_px=float(rms.max()),
        quality_flags=tuple(flags), window_fraction=window_fraction, trim_fraction=trim_fraction,
    )
