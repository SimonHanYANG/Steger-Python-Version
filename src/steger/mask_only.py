"""Strict mask-only experiment: compact normalized filtering and cubic fits.

This is intentionally a changed detector, not an exact-output speedup. It reads
no original ROI intensities outside the selected mask. Labeling/distance/index
maps still inspect the binary mask to identify the domain. Main APIs are intact.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from . import worker as shared
from .derivatives import gaussian_masks, hessian_normals
from .linking import link_points
from .measurement import measure_bending
from .neck import (
    LOW, HIGH, SCALES, RESCUE_SCALES, _Candidate, _neck_result,
    choose_candidate, contour_metrics, mask_geometry,
)


def _backend():
    try:
        from . import _mask_native
    except ImportError as exc:
        raise RuntimeError('Build the experiment: uv run --extra native python tools/build_native.py') from exc
    return _mask_native


def prepare(roi, neck_mask):
    """Scan mask metadata; convert/validate only selected original intensities."""
    raw = np.asarray(roi)
    supplied = np.asarray(neck_mask)
    mask = np.any(supplied[..., :3] != 0, axis=-1) if supplied.ndim == 3 else supplied != 0
    if raw.shape[:2] != mask.shape or raw.ndim not in (2, 3):
        raise ValueError('Expected aligned grayscale/RGB ROI and mask')
    if raw.ndim == 3 and raw.shape[-1] < 3:
        raise ValueError('RGB ROI requires at least three channels')
    if raw.dtype.kind not in 'buif':
        raise ValueError('Expected real numeric ROI')
    geometry = mask_geometry(mask)
    selected = np.asarray(raw[geometry['mask']], dtype=np.float64)
    gray = selected[..., :3] @ np.array([.299, .587, .114]) if raw.ndim == 3 else selected
    if not np.isfinite(gray).all():
        raise ValueError('Mask contains nonfinite ROI intensities')
    return np.ascontiguousarray(gray), geometry


def index_map(mask):
    pixels = np.ascontiguousarray(np.argwhere(mask), dtype=np.intp)
    index = np.full(mask.shape, -1, dtype=np.intp)
    index[mask] = np.arange(len(pixels))
    return pixels, index


def contains(index, points):
    cells = np.floor(points + .5).astype(np.intp)
    valid = np.all((cells >= 0) & (cells < np.array(index.shape)), axis=1)
    ids = np.flatnonzero(valid)
    valid[ids] &= index[cells[ids, 0], cells[ids, 1]] >= 0
    return valid


def spline_system(pixels, index):
    """Masked cubic interpolation: K coefficients = row_sum(K) * values.

    K only couples existing mask nodes. Its symmetric positive-definite cubic
    basis is factored once per scale and reused for both polarities/five fields.
    No full-image recursive spline_filter or outside coefficients are used.
    """
    rows, cols, weights = [], [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            targets = pixels + [dr, dc]
            valid = np.all((targets >= 0) & (targets < index.shape), axis=1)
            source_ids = np.flatnonzero(valid)
            neighbors = index[targets[valid, 0], targets[valid, 1]]
            keep = neighbors >= 0
            rows.extend(source_ids[keep]); cols.extend(neighbors[keep])
            value = (2/3 if dr == 0 else 1/6) * (2/3 if dc == 0 else 1/6)
            weights.extend([value] * int(keep.sum()))
    matrix = coo_matrix((weights, (rows, cols)), shape=(len(pixels), len(pixels))).tocsc()
    return splu(matrix), np.asarray(matrix.sum(axis=1)).ravel()


def mask_contours(gray, mask, sigma):
    native = _backend()
    pixels, index = index_map(mask)
    fields = native.gaussian_fields(gray, pixels, *gaussian_masks(sigma))
    derivatives = dict(zip(('r', 'c', 'rr', 'rc', 'cc'), fields, strict=True))
    normals, dark = hessian_normals(derivatives, 'dark')
    nr, nc = normals.T
    denominator = fields[2]*nr*nr + 2*fields[3]*nr*nc + fields[4]*nc*nc
    step = np.divide(-(fields[0]*nr + fields[1]*nc), denominator,
                     out=np.zeros(len(gray)), where=np.abs(denominator)>1e-14)
    offsets = step[:, None]*normals
    coefficients = None
    results = []
    for polarity, response in (('dark', dark), ('light', -dark)):
        valid = ((response >= LOW) & (response > 0) & (np.abs(denominator)>1e-14)
                 & np.all(np.abs(offsets) <= 1.5+1e-12, axis=1))
        origins = pixels[valid]
        points = origins.astype(np.float64) + offsets[valid]
        ns, strength = normals[valid].copy(), response[valid].copy()
        keep = contains(index, points)
        points, origins, ns, strength = [np.ascontiguousarray(a[keep]) for a in (points, origins, ns, strength)]
        if len(points):
            if coefficients is None:
                solver, mass = spline_system(pixels, index)
                coefficients = np.ascontiguousarray(solver.solve(fields.T * mass[:, None]))
            points = native.refine_points(coefficients, index, points, ns, origins)
            keep = np.all(np.abs(points-origins) <= .5+1e-6, axis=1) & contains(index, points)
            points, ns, strength = points[keep], ns[keep], strength[keep]
        contours, _ = link_points(points, ns, strength, HIGH)
        results.append((polarity, contours))
    return results


def candidate_runs_mask_only(gray, geometry, scales=SCALES):
    records = []
    for sigma in scales:
        for polarity, contours in mask_contours(gray, geometry['mask'], sigma):
            for index, contour in enumerate(contours):
                if contour.closed or len(contour.points)<6:
                    continue
                try:
                    local = measure_bending(contour)
                except ValueError:
                    local = None
                records.append(_Candidate(contour, local, dict(
                    sigma=sigma, polarity=polarity, contour_index=index,
                    **contour_metrics(contour, geometry),
                    local_angle_deg=None if local is None else local.angle_deg,
                    points=contour.points,
                )))
    return records


def analyze_mask_only(roi, neck_mask):
    gray, geometry = prepare(roi, neck_mask)
    candidates = candidate_runs_mask_only(gray, geometry)
    if not any(c['span_coverage']>=.75 and c['length_px']>=15 for c in candidates):
        candidates.extend(candidate_runs_mask_only(gray, geometry, RESCUE_SCALES))
    return _neck_result(geometry, candidates), choose_candidate(candidates), candidates, geometry


def measure_neck_mask_only(roi, neck_mask):
    return analyze_mask_only(roi, neck_mask)[0]


def _run_mask_scale(request):
    shape, axis, span, sigma, count = request
    memory, capacity = shared._shared_input
    image, distance, mask = shared._workspace_views(memory, capacity, shape)
    gray = image.ravel()[:count]
    for a in (gray, distance, mask):
        a.flags.writeable = False
    geometry = dict(mask=mask, axis=axis, span=span, distance=distance)
    return candidate_runs_mask_only(gray, geometry, (sigma,))


class MaskOnlyNeckAngleWorker(shared.NeckAngleWorker):
    """Same five-process/shared-memory setup, containing only M gray samples."""
    def _candidates(self, unused, geometry, scales):
        mask = geometry['mask']
        requests = [(mask.shape, geometry['axis'], geometry['span'], sigma, int(mask.sum()))
                    for sigma in scales]
        return [c for group in self._executor.map(_run_mask_scale, requests) for c in group]

    def measure(self, roi, neck_mask):
        with self._lock:
            self._check_open()
            gray, geometry = prepare(roi, neck_mask)
            shape = geometry['mask'].shape
            if any(a>b for a,b in zip(shape, self.max_shape, strict=True)):
                raise ValueError('ROI exceeds workspace capacity')
            image, distance, mask = shared._workspace_views(self._memory, self._capacity, shape)
            np.copyto(image.ravel()[:len(gray)], gray)
            np.copyto(distance, geometry['distance'])
            np.copyto(mask, geometry['mask'])
            try:
                candidates = self._candidates(None, geometry, SCALES)
                if not any(c['span_coverage']>=.75 and c['length_px']>=15 for c in candidates):
                    candidates.extend(self._candidates(None, geometry, RESCUE_SCALES))
                return _neck_result(geometry, candidates)
            except BaseException:
                self._shutdown()
                raise
