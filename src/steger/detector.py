"""Public HALCON-shaped interface, implemented independently in Python/SciPy."""

import os

import numpy as np
from scipy.ndimage import convolve1d, map_coordinates, spline_filter

from .derivatives import gaussian_masks, hessian_normals, image_derivatives
from .linking import complete_junctions as _complete
from .linking import link_points
from .refinement import refine_contour
from .types import Contour, DetectionResult, DetectionTrace
from .width import extract_widths

try:
    if os.environ.get("STEGER_DISABLE_NATIVE") == "1":
        raise ImportError("Native sampling disabled explicitly")
    from ._spline_native import refine_points as _native_refine_points
    from ._spline_native import sample_fields as _native_sample_fields
except ImportError:
    _native_sample_fields = None
    _native_refine_points = None


def _boolean(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in ("true", "false"):
        return value == "true"
    raise ValueError(f"{name} must be bool or 'true'/'false'")


def _snapshot(contour):
    return Contour(
        contour.points.copy(),
        contour.normals.copy(),
        contour.response.copy(),
        {key: value.copy() for key, value in contour.attributes.items()},
        contour.closed,
    )


def lines_gauss(
    image,
    sigma=1.5,
    low=3.0,
    high=8.0,
    light_dark="light",
    extract_width=True,
    line_model="bar-shaped",
    complete_junctions=True,
    *,
    mask=None,
    min_length=0.0,
    border="reflect",
    refine=True,
    profile_refinement=True,
    return_trace=False,
):
    """Extract ordered, subpixel centerlines and per-point attributes.

    The first eight arguments mirror HALCON lines_gauss. This is not a HALCON
    binding and does not claim bitwise or experimentally established parity.
    Input intensities retain their original scale. Mask limits point selection,
    never convolution: do not zero out the image outside a segmentation mask.

    refine=True performs additional cubic-interpolated Newton updates of ridge
    positions. Set False for the literal single Taylor step of the 1998 paper.
    return_trace=True retains actual intermediate arrays and contour snapshots
    for process visualization, without modifying the numerical calculation.
    """
    raw = np.asarray(image)
    if raw.ndim != 2 or min(raw.shape, default=0) < 3:
        raise ValueError("image must be a 2-D grayscale image at least 3x3")
    if raw.dtype.kind not in "buif":
        raise ValueError("image must contain real numeric intensities")
    img = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(img).all():
        raise ValueError("image contains NaN or infinity")
    for name, v in (("sigma", sigma), ("low", low), ("high", high), ("min_length", min_length)):
        if not np.isscalar(v) or not np.isfinite(v):
            raise ValueError(f"{name} must be finite")
    sigma, low, high = map(float, (sigma, low, high))
    if sigma < 0.4:
        raise ValueError("sigma < 0.4 is numerically undersampled; choose sigma >= 0.4")
    if low < 0 or high < low:
        raise ValueError("require 0 <= low <= high")
    if min_length < 0:
        raise ValueError("min_length must be nonnegative")
    if light_dark not in ("light", "dark"):
        raise ValueError("light_dark must be 'light' or 'dark'")
    if line_model not in ("none", "bar-shaped", "parabolic", "gaussian"):
        raise ValueError("invalid line_model")
    if border not in ("reflect", "mirror", "nearest"):
        raise ValueError("unsupported border mode")
    extract_width = _boolean(extract_width, "extract_width")
    complete_junctions = _boolean(complete_junctions, "complete_junctions")
    refine = _boolean(refine, "refine")
    profile_refinement = _boolean(profile_refinement, "profile_refinement")
    return_trace = _boolean(return_trace, "return_trace")
    selected = np.ones(img.shape, dtype=bool) if mask is None else np.asarray(mask) != 0
    if selected.shape != img.shape:
        raise ValueError("mask and image shapes must match; supply an aligned mask")
    fields = _RidgeFields(img, sigma, border, selected, return_trace)
    return _extract(
        fields,
        sigma=sigma,
        low=low,
        high=high,
        light_dark=light_dark,
        extract_width=extract_width,
        line_model=line_model,
        complete_junctions=complete_junctions,
        selected=selected,
        mask_used=mask is not None,
        min_length=min_length,
        border=border,
        refine=refine,
        profile_refinement=profile_refinement,
        return_trace=return_trace,
    )


class _RidgeFields:
    """Per-image/per-scale fields, shared by the two independent polarities.

    No global image cache. Convolution and spline filtering still use the entire
    original ROI. Masked eigensystems are independent 2x2 problems; trace mode
    computes the full field so process visualizations retain all original values.
    """

    def __init__(self, image, sigma, border, selected, trace):
        self.image = image
        self.derivatives = d = image_derivatives(image, sigma, border=border)
        self.eigenvalues = None
        self.pixels = None
        samples = d
        if trace:
            self.normals, self.dark_response, self.eigenvalues = hessian_normals(
                d, "dark", with_eigenvalues=True
            )
        elif selected.all():
            self.normals, self.dark_response = hessian_normals(d, "dark")
        else:
            self.pixels = np.argwhere(selected)
            samples = {k: v[selected] for k, v in d.items()}
            self.normals, self.dark_response = hessian_normals(samples, "dark")
        nr, nc = self.normals[..., 0], self.normals[..., 1]
        self.denominator = denominator = (
            samples["rr"] * nr * nr + 2 * samples["rc"] * nr * nc + samples["cc"] * nc * nc
        )
        self.t = np.zeros(self.dark_response.shape)
        np.divide(
            -(samples["r"] * nr + samples["c"] * nc),
            denominator,
            out=self.t,
            where=np.abs(denominator) > 1e-14,
        )
        self.off = self.t[..., None] * self.normals
        self._coefficients = None
        self._native_coefficients = None

    @property
    def coefficients(self):
        if self._coefficients is None:
            self._coefficients = {
                k: spline_filter(v, order=3, mode="reflect") for k, v in self.derivatives.items()
            }
        return self._coefficients

    @property
    def native_coefficients(self):
        if self._native_coefficients is None:
            co = self.coefficients
            self._native_coefficients = np.stack([co[k] for k in ("r", "c", "rr", "rc", "cc")])
        return self._native_coefficients


def _detect_neck_polarities(image, mask, sigma, low, high):
    """Internal fixed-policy path; keep separate Newton stopping per polarity."""
    img = np.asarray(image, dtype=np.float64)
    selected = np.asarray(mask) != 0
    if img.ndim != 2 or min(img.shape, default=0) < 3:
        raise ValueError("image must be a 2-D grayscale image at least 3x3")
    if selected.shape != img.shape:
        raise ValueError("mask and image shapes must match; supply an aligned mask")
    if not np.isfinite(img).all():
        raise ValueError("image contains NaN or infinity")
    fields = _RidgeFields(img, sigma, "reflect", selected, False)
    return tuple(
        _extract(
            fields,
            sigma=sigma,
            low=low,
            high=high,
            light_dark=polarity,
            extract_width=False,
            line_model="none",
            complete_junctions=False,
            selected=selected,
            mask_used=True,
            min_length=0.0,
            border="reflect",
            refine=True,
            profile_refinement=False,
            return_trace=False,
        )
        for polarity in ("dark", "light")
    )


def _extract(
    fields,
    *,
    sigma,
    low,
    high,
    light_dark,
    extract_width,
    line_model,
    complete_junctions,
    selected,
    mask_used,
    min_length,
    border,
    refine,
    profile_refinement,
    return_trace,
):
    img, derivatives = fields.image, fields.derivatives
    normals, eigenvalues = fields.normals, fields.eigenvalues
    response = -fields.dark_response if light_dark == "light" else fields.dark_response
    denominator, t, off = fields.denominator, fields.t, fields.off
    # A single Taylor step can slightly overshoot 0.5 px for a ridge exactly
    # between two pixel centers, dropping BOTH pixels. Admit a wider initial
    # neighborhood only when Newton refinement will verify the final location.
    initial_limit = 1.5 if refine else 0.5
    valid = (
        (selected if fields.pixels is None else True)
        & (response > 0)
        & (response >= low)
        & (np.abs(denominator) > 1e-14)
        & np.all(np.abs(off) <= initial_limit + 1e-12, axis=-1)
    )
    rc = np.argwhere(valid) if fields.pixels is None else fields.pixels[valid]
    points = rc.astype(np.float64) + off[valid]
    ns = normals[valid].copy()
    strength = response[valid].copy()
    trace = None
    if return_trace:
        g0 = gaussian_masks(sigma)[0]
        smoothed = convolve1d(convolve1d(img, g0, axis=0, mode=border), g0, axis=1, mode=border)
        trace = DetectionTrace(
            image=img.copy(),
            mask=selected.copy(),
            smoothed=smoothed,
            derivatives={key: value.copy() for key, value in derivatives.items()},
            eigenvalues=eigenvalues.copy(),
            hessian_normals=normals.copy(),
            response=response.copy(),
            taylor_t=t.copy(),
            taylor_offset=off.copy(),
            initial_candidate_pixels=rc.copy(),
            initial_candidate_points=points.copy(),
            candidate_pixels=rc.copy(),
            candidate_points=points.copy(),
            candidate_normals=ns.copy(),
            candidate_response=strength.copy(),
        )
    if refine and len(points):
        co = fields.coefficients
        native_co = fields.native_coefficients if _native_sample_fields is not None else None
        if _native_refine_points is not None:
            points, positions = _native_refine_points(native_co, points, ns, rc, trace is not None)
            if trace is not None:
                trace.newton_positions.extend(positions)
        for _ in range(0 if _native_refine_points is not None else 6):
            if native_co is None:
                d = {
                    k: map_coordinates(v, points.T, order=3, mode="reflect", prefilter=False)
                    for k, v in co.items()
                }
                dr, dc, drr, drc, dcc = (d[k] for k in ("r", "c", "rr", "rc", "cc"))
            else:
                dr, dc, drr, drc, dcc = _native_sample_fields(native_co, points)
            a, b = ns[:, 0], ns[:, 1]
            denom = drr * a * a + 2 * drc * a * b + dcc * b * b
            step = np.divide(
                -(dr * a + dc * b),
                denom,
                out=np.zeros(len(points)),
                where=np.abs(denom) > 1e-14,
            )
            step = np.clip(step, -0.25, 0.25)
            proposed = points + step[:, None] * ns
            # Keep refinement in the originating pixel's neighborhood.
            acceptable = np.all(np.abs(proposed - rc) <= 1.5, axis=1)
            points[acceptable] = proposed[acceptable]
            if trace is not None:
                trace.newton_positions.append(points.copy())
            if np.max(np.abs(step), initial=0) < 1e-8:
                break
        accepted = np.all(np.abs(points - rc) <= 0.5 + 1e-6, axis=1)
        points, ns, strength = points[accepted], ns[accepted], strength[accepted]
        if trace is not None:
            trace.candidate_pixels = rc[accepted].copy()
    if trace is not None:
        trace.candidate_points = points.copy()
        trace.candidate_normals = ns.copy()
        trace.candidate_response = strength.copy()
    contours, junctions = link_points(points, ns, strength, high)
    if trace is not None:
        trace.linked_contours = [_snapshot(c) for c in contours]
    if complete_junctions:
        contours, junctions = _complete(contours, sigma)
    if min_length:
        # Reindex junction memberships after length filtering.
        keep = [i for i, c in enumerate(contours) if c.length >= min_length]
        index = {old: new for new, old in enumerate(keep)}
        contours = [contours[i] for i in keep]
        for j in junctions:
            j.contour_indices = tuple(index[i] for i in j.contour_indices if i in index)
        junctions = [j for j in junctions if len(j.contour_indices) > 1]
    if extract_width:
        for c in contours:
            extract_widths(c, derivatives, sigma, line_model, light_dark)
            if trace is not None:
                trace.bias_corrected_contours.append(_snapshot(c))
            if profile_refinement and line_model != "none":
                c.attributes["steger_row"] = c.rows.copy()
                c.attributes["steger_col"] = c.cols.copy()
                c.attributes["steger_width_left"] = c.attributes["width_left"].copy()
                c.attributes["steger_width_right"] = c.attributes["width_right"].copy()
                refine_contour(c, img, sigma, line_model, light_dark)
    elif trace is not None:
        trace.bias_corrected_contours = [_snapshot(c) for c in contours]
    parameters = dict(
        sigma=sigma,
        low=low,
        high=high,
        light_dark=light_dark,
        extract_width=extract_width,
        line_model=line_model,
        complete_junctions=complete_junctions,
        border=border,
        refine=refine,
        profile_refinement=profile_refinement,
        min_length=float(min_length),
    )
    total = sum(len(c.points) for c in contours)
    corrected = sum(np.count_nonzero(c.attributes.get("correction_valid", [])) for c in contours)
    return DetectionResult(
        contours,
        junctions,
        parameters,
        {
            "candidate_points": len(points),
            "output_points": total,
            "corrected_points": int(corrected),
            "halcon_parity": "not_verified",
            "dtype": "float64",
            "mask_used": mask_used,
            "intensity_rescaled": False,
            "profile_refined_points": int(
                sum(np.count_nonzero(c.attributes.get("profile_fit_valid", [])) for c in contours)
            ),
        },
        trace=trace,
    )


def calculate_lines_gauss_parameters(max_line_width, contrast, low_ratio=0.33):
    """Paper-based parameter suggestion, not a clone of the HDevelop procedure.

    max_line_width is FULL diameter; contrast uses the image's intensity units.
    """
    if not np.isfinite(max_line_width) or max_line_width <= 0:
        raise ValueError("max_line_width must be positive")
    if not np.isfinite(contrast) or contrast <= 0:
        raise ValueError("contrast must be positive")
    if not 0 < low_ratio < 1:
        raise ValueError("low_ratio must lie in (0,1)")
    w = float(max_line_width) / 2
    sigma = max(0.7, w / np.sqrt(3) + 0.05)
    high = (
        2
        * float(contrast)
        * w
        * np.exp(-w * w / (2 * sigma * sigma))
        / (np.sqrt(2 * np.pi) * sigma**3)
    )
    return sigma, low_ratio * high, high
