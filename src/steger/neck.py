"""v1.0 neck policy with reusable fields; no I/O or visualization dependencies.

Measures bending of the visible grayscale neck centerline using aligned ROI and
mask arrays. Detection scales are unchanged; bending uses continuous local
geometric directions and bounded terminal quality checks.
"""

import numpy as np
from scipy import ndimage

from .detector import _detect_neck_polarities, lines_gauss
from .measurement import measure_bending
from .robust_measurement import window_direction_ranges

SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
RESCUE_SCALES = (3.5, 4.0, 4.5)
LOW, HIGH = 0.05, 0.15


class _Candidate(dict):
    """Keep public record keys unchanged; retain the actual contour and fit."""

    def __init__(self, contour, local_measurement, record):
        super().__init__(record)
        self.contour = contour
        self.local_measurement = local_measurement


def mask_geometry(mask):
    """Use the largest 8-connected component; preserve the original for display."""
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if not count:
        raise ValueError("Empty neck mask")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = labels == np.argmax(sizes)
    pixels = np.argwhere(selected)
    if len(pixels) < 6:
        raise ValueError("Neck mask contains fewer than six pixels")
    _, _, vt = np.linalg.svd(pixels - pixels.mean(axis=0), full_matrices=False)
    axis = vt[0]
    span = float(np.ptp(pixels @ axis))
    if span < 1:
        raise ValueError("Degenerate mask extent")
    distance = ndimage.distance_transform_edt(np.pad(selected, 1))[1:-1, 1:-1]
    return {
        "mask": selected,
        "axis": axis,
        "span": span,
        "distance": distance,
        "component_count": int(count),
        "ignored_mask_pixels": int(mask.sum() - selected.sum()),
        "touches_roi_border": bool(
            selected[0].any() or selected[-1].any() or selected[:, 0].any() or selected[:, -1].any()
        ),
    }


def contour_metrics(contour, geometry):
    coverage = float(np.ptp(contour.points @ geometry["axis"]) / geometry["span"])
    depth = ndimage.map_coordinates(geometry["distance"], contour.points.T, order=1)
    centrality = float(np.median(depth) / geometry["distance"].max())
    return {"span_coverage": coverage, "centrality": centrality, "length_px": contour.length}


def candidate_runs(image, geometry, scales=SCALES):
    records = []
    for sigma in scales:
        results = _detect_neck_polarities(image, geometry["mask"], sigma, LOW, HIGH)
        for polarity, result in zip(("dark", "light"), results, strict=True):
            for index, contour in enumerate(result.contours):
                if contour.closed or len(contour.points) < 6:
                    continue
                metrics = contour_metrics(contour, geometry)
                try:
                    local = measure_bending(contour)
                except ValueError:
                    local = None
                records.append(
                    _Candidate(
                        contour,
                        local,
                        {
                            "sigma": sigma,
                            "polarity": polarity,
                            "contour_index": index,
                            **metrics,
                            "local_angle_deg": None if local is None else local.angle_deg,
                            "points": contour.points,
                        },
                    )
                )
    return records


def detect(image, mask, sigma, polarity, *, trace=False):
    # Width/asymmetry/profile models have not been validated on these phase-like
    # real micrographs. Keep the measured grayscale ridge and expose every stage.
    return lines_gauss(
        image,
        sigma,
        LOW,
        HIGH,
        polarity,
        False,
        "none",
        False,
        mask=mask,
        profile_refinement=False,
        return_trace=trace,
    )


def choose_candidate(candidates):
    """Favor extent, then a declared reference scale; never select by angle."""
    if not candidates:
        return None
    best_coverage = max(c["span_coverage"] for c in candidates)
    eligible = [c for c in candidates if c["span_coverage"] >= best_coverage - 0.05]
    adequate = [c for c in eligible if c["span_coverage"] >= 0.75 and c["length_px"] >= 15]
    if adequate:
        eligible = adequate
    measurable = [c for c in eligible if c["local_angle_deg"] is not None]
    if measurable:
        eligible = measurable
    return max(
        eligible,
        key=lambda c: (-abs(c["sigma"] - 2.0), c["centrality"], c["sigma"]),
    )


def assess(contour, selected, candidates, geometry=None):
    local = (
        selected.local_measurement
        if isinstance(selected, _Candidate) and selected.contour is contour
        else None
    )
    if local is None:
        try:
            local = measure_bending(contour)
        except ValueError:
            pass
    flags = []
    if selected["span_coverage"] < 0.75:
        flags.append("insufficient_span_coverage")
    measurement = local
    if measurement is None:
        flags.append("insufficient_direction_samples")
    else:
        flags.extend(measurement.quality_flags)
    trusted_coverage = selected["span_coverage"]
    if measurement is not None and geometry is not None:
        trusted = contour.points[measurement.valid_point_start:measurement.valid_point_stop]
        trusted_coverage = float(np.ptp(trusted @ geometry["axis"]) / geometry["span"])
        if trusted_coverage < .75:
            flags.append("insufficient_trusted_span_coverage")
    window_sensitivity = []
    if measurement is not None:
        window_sensitivity=window_direction_ranges(contour,measurement)
    window_angles = [item["angle_deg"] for item in window_sensitivity]
    window_range = [min(window_angles), max(window_angles)] if window_angles else None
    if window_range and window_range[1] - window_range[0] > 5:
        flags.append("window_sensitive_over_5deg")
    # Extent measures projection on the mask's major axis, NOT anatomical truth.
    nearby = []
    for sigma in sorted({c["sigma"] for c in candidates}):
        options = [
            c for c in candidates if c["sigma"] == sigma and c["polarity"] == selected["polarity"]
        ]
        if not options:
            continue
        item = max(options, key=lambda c: (c["span_coverage"], c["centrality"]))
        if (item["span_coverage"] >= selected["span_coverage"] - 0.08
                and item["length_px"] >= 15 and item["local_angle_deg"] is not None):
            nearby.append(item["local_angle_deg"])
    scale_range = [float(min(nearby)), float(max(nearby))] if nearby else None
    if len(nearby) < 2:
        flags.append("insufficient_comparable_scales")
    elif scale_range[1] - scale_range[0] > 5:
        flags.append("scale_sensitive_over_5deg")
    if selected["centrality"] < 0.4:
        flags.append("ridge_near_mask_boundary")
    if selected["sigma"] > max(SCALES):
        flags.append("large_scale_rescue_requires_review")
    insufficient = ("insufficient_span_coverage" in flags or measurement is None
                    or "insufficient_trusted_span_coverage" in flags)
    status = "insufficient_segment" if insufficient else "review_required" if flags else "estimated"
    return {
        "status": status,
        "angle_deg": None if insufficient else measurement.angle_deg,
        "local_segment_angle_deg": None if local is None else local.angle_deg,
        "measurement": {} if measurement is None else measurement.to_dict(),
        "quality_flags": flags,
        "scale_angle_range_deg": scale_range,
        "comparable_scale_count": len(nearby),
        "angle_method": "geometry_arclength_direction_range_v2",
        "trusted_span_coverage": trusted_coverage,
        "window_angle_range_deg": window_range,
        "window_sensitivity": window_sensitivity,
    }


def _prepare_neck_inputs(roi, neck_mask):
    """The same validation/conversion for serial and parallel scale workers."""
    image = np.asarray(roi)
    if image.ndim == 3 and image.shape[-1] >= 3:
        image = np.asarray(image[..., :3], dtype=np.float64) @ np.array([0.299, 0.587, 0.114])
    else:
        image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(neck_mask)
    if mask.ndim == 3:
        mask = np.any(mask[..., :3] != 0, axis=-1)
    else:
        mask = mask != 0
    if image.ndim != 2 or image.shape != mask.shape:
        raise ValueError("Expected aligned 2-D or RGB ROI and neck mask")
    if not np.isfinite(image).all():
        raise ValueError("image contains NaN or infinity")
    return image, mask


def _neck_result(geometry, candidates):
    """Select and assess after all required scales, always in policy order."""
    selected = choose_candidate(candidates)
    if selected is None:
        return {"angle_deg": None, "status": "no_line"}
    return {
        **{k: v for k, v in selected.items() if k not in ("points", "local_angle_deg")},
        **assess(selected.contour, selected, candidates, geometry),
        **{
            k: geometry[k] for k in ("component_count", "ignored_mask_pixels", "touches_roi_border")
        },
    }


def measure_neck(roi, neck_mask):
    """Return angle and quality diagnostics from resident ROI/mask arrays.

    Accepts grayscale or RGB(A) ROI and aligned 2-D or RGB(A) nonzero mask.
    Preserve original intensities (normally 0..255); do not normalize to 0..1.
    A fresh complete adaptive search runs for every input. No file reads, plots,
    saved per-image parameters, or cross-image mutable caches are involved.
    angle_deg is None when no sufficiently complete measurable segment exists.
    """
    image, mask = _prepare_neck_inputs(roi, neck_mask)
    geometry = mask_geometry(mask)
    candidates = candidate_runs(image, geometry)
    if not any(c["span_coverage"] >= 0.75 and c["length_px"] >= 15 for c in candidates):
        candidates.extend(candidate_runs(image, geometry, RESCUE_SCALES))
    return _neck_result(geometry, candidates)
