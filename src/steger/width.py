"""Subpixel edge localization, width estimation and asymmetry bias removal."""

from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import minimize_scalar

from .profiles import invert_bias


@lru_cache(maxsize=16384)
def _inverse_cached(width, ratio, model):
    return invert_bias(width, ratio, model)


def extract_widths(contour, derivatives, sigma, line_model, light_dark):
    p, n = contour.points, contour.normals
    count = len(p)
    attrs = contour.attributes
    left = np.full(count, np.nan)
    right = left.copy()
    gl = left.copy()
    gr = left.copy()
    # Interpolate signed gradient components BEFORE taking their norm. Taking
    # abs/norm on the pixel grid introduces a cusp at the ridge; cubic spline
    # interpolation of that cusp can shift edge maxima by ~0.1 pixel.
    coeff = [spline_filter(derivatives[k], order=3, mode="reflect") for k in ("r", "c")]
    max_distance = 2.5 * sigma
    grid = np.linspace(-max_distance, max_distance, max(31, int(np.ceil(20 * sigma)) + 1))
    locations = p[:, None, :] + grid[None, :, None] * n[:, None, :]
    components = [
        map_coordinates(
            v, locations.reshape(-1, 2).T, order=3, prefilter=False, mode="reflect"
        ).reshape(count, -1)
        for v in coeff
    ]
    values = np.hypot(*components)

    def value(point, normal, t):
        return float(
            np.hypot(
                *[
                    map_coordinates(
                        v, (point + t * normal)[:, None], order=3, prefilter=False, mode="reflect"
                    )[0]
                    for v in coeff
                ]
            )
        )

    for i in range(count):
        for side in (-1, 1):
            peaks = (
                np.flatnonzero(
                    (values[i, 1:-1] > values[i, :-2]) & (values[i, 1:-1] >= values[i, 2:])
                )
                + 1
            )
            peaks = peaks[grid[peaks] * side > 0.25]
            if not len(peaks):
                continue
            k = peaks[np.argmin(np.abs(grid[peaks]))]
            fit = minimize_scalar(
                lambda t, point=p[i], normal=n[i]: -value(point, normal, t),
                bounds=(grid[k - 1], grid[k + 1]),
                method="bounded",
                options={"xatol": 1e-10},
            )
            if not fit.success or -fit.fun < 1e-10:
                continue
            if side < 0:
                left[i], gl[i] = -fit.x, -fit.fun
            else:
                right[i], gr[i] = fit.x, -fit.fun
    measured = np.isfinite(left) & np.isfinite(right)
    attrs["width_measured"] = measured.copy()
    attrs["width_interpolated"] = np.zeros(count, dtype=bool)
    # Interpolate each side by arclength, and extrapolate using the nearest
    # measured width, as described in Steger 1998 section 4.2.
    arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    for array in (left, right, gl, gr):
        valid = np.isfinite(array)
        if valid.any() and not valid.all():
            attrs["width_interpolated"] |= ~valid
            array[~valid] = np.interp(arclength[~valid], arclength[valid], array[valid])
    attrs["width_left"] = np.nan_to_num(left)
    attrs["width_right"] = np.nan_to_num(right)
    if line_model == "none":
        return
    attrs["asymmetry"] = np.zeros(count)
    attrs["contrast"] = np.zeros(count)
    attrs["correction_valid"] = np.zeros(count, dtype=bool)
    attrs["correction_residual"] = np.full(count, np.nan)
    attrs["position_correction"] = np.zeros(count)
    for i in range(count):
        if not np.isfinite(left[i] + right[i] + gl[i] + gr[i]):
            continue
        # Keep the measured side labels for 'none'. Once corrected, both widths
        # are measured from the true center and are w in these profile models.
        weak_right = gr[i] <= gl[i]
        strong = max(gl[i], gr[i])
        weak = min(gl[i], gr[i])
        ratio = min(1.0, max(1e-12, weak / strong))
        try:
            w, a, geometry, residual = _inverse_cached(
                round(float((left[i] + right[i]) / sigma), 9), round(float(ratio), 9), line_model
            )
        except ValueError:
            continue
        shift = geometry.ridge * sigma * (1 if weak_right else -1)
        # An inversion far outside the measured cross section is unreliable.
        if abs(shift) > max_distance:
            continue
        p[i] -= shift * n[i]
        attrs["width_left"][i] = attrs["width_right"][i] = w * sigma
        attrs["asymmetry"][i] = a * (1 if weak_right else -1)
        attrs["contrast"][i] = (
            strong / abs(geometry.gradient_left) * sigma * (1 if light_dark == "light" else -1)
        )
        attrs["correction_valid"][i] = True
        attrs["correction_residual"][i] = residual
        attrs["position_correction"][i] = shift
