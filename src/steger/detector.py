"""Public HALCON-shaped interface, implemented independently in Python/SciPy."""

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

from .derivatives import hessian_normals, image_derivatives
from .linking import complete_junctions as _complete
from .linking import link_points
from .refinement import refine_contour
from .types import DetectionResult
from .width import extract_widths


def _boolean(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in ("true", "false"):
        return value == "true"
    raise ValueError(f"{name} must be bool or 'true'/'false'")


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
):
    """Extract ordered, subpixel centerlines and per-point attributes.

    The first eight arguments mirror HALCON lines_gauss. This is not a HALCON
    binding and does not claim bitwise or experimentally established parity.
    Input intensities retain their original scale. Mask limits point selection,
    never convolution: do not zero out the image outside a segmentation mask.

    refine=True performs additional cubic-interpolated Newton updates of ridge
    positions. Set False for the literal single Taylor step of the 1998 paper.
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
    selected = np.ones(img.shape, dtype=bool) if mask is None else np.asarray(mask) != 0
    if selected.shape != img.shape:
        raise ValueError("mask and image shapes must match; supply an aligned mask")
    derivatives = image_derivatives(img, sigma, border=border)
    normals, response = hessian_normals(derivatives, light_dark)
    nr, nc = normals[..., 0], normals[..., 1]
    denominator = (
        derivatives["rr"] * nr * nr + 2 * derivatives["rc"] * nr * nc + derivatives["cc"] * nc * nc
    )
    t = np.zeros(img.shape)
    np.divide(
        -(derivatives["r"] * nr + derivatives["c"] * nc),
        denominator,
        out=t,
        where=np.abs(denominator) > 1e-14,
    )
    off = t[..., None] * normals
    # A single Taylor step can slightly overshoot 0.5 px for a ridge exactly
    # between two pixel centers, dropping BOTH pixels. Admit a wider initial
    # neighborhood only when Newton refinement will verify the final location.
    initial_limit = 1.5 if refine else 0.5
    valid = (
        selected
        & (response > 0)
        & (response >= low)
        & (np.abs(denominator) > 1e-14)
        & np.all(np.abs(off) <= initial_limit + 1e-12, axis=-1)
    )
    rc = np.argwhere(valid)
    points = rc.astype(np.float64) + off[valid]
    ns = normals[valid].copy()
    strength = response[valid].copy()
    if refine and len(points):
        co = {k: spline_filter(v, order=3, mode="reflect") for k, v in derivatives.items()}
        for _ in range(6):
            d = {
                k: map_coordinates(v, points.T, order=3, mode="reflect", prefilter=False)
                for k, v in co.items()
            }
            a, b = ns[:, 0], ns[:, 1]
            denom = d["rr"] * a * a + 2 * d["rc"] * a * b + d["cc"] * b * b
            step = np.divide(
                -(d["r"] * a + d["c"] * b),
                denom,
                out=np.zeros(len(points)),
                where=np.abs(denom) > 1e-14,
            )
            step = np.clip(step, -0.25, 0.25)
            proposed = points + step[:, None] * ns
            # Keep refinement in the originating pixel's neighborhood.
            acceptable = np.all(np.abs(proposed - rc) <= 1.5, axis=1)
            points[acceptable] = proposed[acceptable]
            if np.max(np.abs(step), initial=0) < 1e-8:
                break
        accepted = np.all(np.abs(points - rc) <= 0.5 + 1e-6, axis=1)
        points, ns, strength = points[accepted], ns[accepted], strength[accepted]
    contours, junctions = link_points(points, ns, strength, high)
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
            if profile_refinement and line_model != "none":
                c.attributes["steger_row"] = c.rows.copy()
                c.attributes["steger_col"] = c.cols.copy()
                c.attributes["steger_width_left"] = c.attributes["width_left"].copy()
                c.attributes["steger_width_right"] = c.attributes["width_right"].copy()
                refine_contour(c, img, sigma, line_model, light_dark)
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
            "mask_used": mask is not None,
            "intensity_rescaled": False,
            "profile_refined_points": int(
                sum(np.count_nonzero(c.attributes.get("profile_fit_valid", [])) for c in contours)
            ),
        },
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
