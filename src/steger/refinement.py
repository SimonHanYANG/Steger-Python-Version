"""Optional accuracy extension: fit a pixel-integrated profile to original data.

This is explicitly NOT a documented HALCON internal step. It compensates pixel
aperture/discretization error left by continuous bias inversion. It must be
validated on the intended imaging modality; diagnostics expose every rejection.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr


def _primitive(x, w, model, order):
    """Antiderivative of symmetric profile and H(x)*(1-profile)."""
    x = np.asarray(x, dtype=float)
    if model == "gaussian":
        first = w * np.sqrt(2 * np.pi) * (ndtr(x / w) - 0.5)
        if order == 1:
            sym = first
            right = np.where(x > 0, x - first, 0.0)
        else:
            sym = x * first + w * w * np.exp(-0.5 * (x / w) ** 2)
            right = np.where(x > 0, x * x / 2 - (sym - w * w), 0.0)
    elif model == "bar-shaped":
        if order == 1:
            sym = np.maximum(x + w, 0) - np.maximum(x - w, 0)
            right = np.maximum(x - w, 0)
        else:
            sym = (np.maximum(x + w, 0) ** 2 - np.maximum(x - w, 0) ** 2) / 2
            right = np.maximum(x - w, 0) ** 2 / 2
    elif model == "parabolic":
        hi = np.clip(x, -w, w)
        if order == 1:
            sym = hi - hi**3 / (3 * w * w) + 2 * w / 3
            pos = np.clip(x, 0, w)
            right = np.where(x > 0, x - (pos - pos**3 / (3 * w * w)), 0.0)
        else:

            def integral(u):
                return x * u - u * u / 2 - x * u**3 / (3 * w * w) + u**4 / (4 * w * w)

            sym = integral(hi) - integral(-w)
            pos = np.clip(x, 0, w)
            right = np.where(x > 0, x * x / 2 - integral(pos), 0.0)
    else:
        raise ValueError(model)
    return sym, right


def integrated_profile(distance, w, skew, normal, model):
    """Exact area average for a straight line over a unit square pixel.

    Signed skew gives backgrounds +/-skew and peak 1. This parameterization
    is smooth through the symmetric case, unlike swapping weak/strong sides.
    """
    d = np.asarray(distance)
    normal = np.asarray(normal)
    a = np.maximum(np.abs(normal[..., 0]), np.abs(normal[..., 1]))
    b = np.minimum(np.abs(normal[..., 0]), np.abs(normal[..., 1]))

    def antiderivative(x, order):
        sym, right = _primitive(x, w, model, order)
        unit = x if order == 1 else x * x / 2
        return sym + skew * (2 * right - unit + sym)

    a, b = np.broadcast_arrays(a, b, d)[0:2]
    result = np.empty_like(d, dtype=float)
    axis = b < 1e-4
    if np.any(axis):
        result[axis] = (
            antiderivative(d[axis] + a[axis] / 2, 1) - antiderivative(d[axis] - a[axis] / 2, 1)
        ) / a[axis]
    if np.any(~axis):
        z = d[~axis]
        aa = a[~axis]
        bb = b[~axis]
        result[~axis] = (
            antiderivative(z + (aa + bb) / 2, 2)
            - antiderivative(z + (aa - bb) / 2, 2)
            - antiderivative(z + (-aa + bb) / 2, 2)
            + antiderivative(z - (aa + bb) / 2, 2)
        ) / (aa * bb)
    return result


def refine_contour(contour, image, sigma, model, light_dark, *, max_nfev=60):
    """Local constant-curvature cross-section fitting with explicit fit status."""
    p, n = contour.points, contour.normals
    attrs = contour.attributes
    count = len(p)
    attrs["profile_fit_valid"] = np.zeros(count, dtype=bool)
    attrs["profile_fit_rms"] = np.full(count, np.nan)
    attrs["profile_fit_shift"] = np.zeros(count)
    attrs["profile_fit_condition"] = np.full(count, np.nan)
    source = image if light_dark == "light" else -image
    original = p.copy()
    # Every fit uses original intensities, never intensities sampled from a
    # smoothed/thresholded image or synthetic junction extension points.
    inferred = attrs.get("junction_completed", np.zeros(count, dtype=bool))
    for i in range(count):
        if inferred[i]:
            continue
        w0 = max(0.35, float(attrs.get("width_left", np.full(count, sigma))[i]))
        if w0 <= 0:
            w0 = sigma
        a0 = float(attrs.get("asymmetry", np.zeros(count))[i])
        skew0 = np.clip(a0 / (2 - abs(a0)), -0.85, 0.85)
        normal = n[i].copy()
        tangent = np.array([-normal[1], normal[0]])
        radius = int(np.ceil(max(3 * sigma, 3 * w0) + 4))
        r0, c0 = np.floor(p[i]).astype(int)
        rr, cc = np.mgrid[
            max(0, r0 - radius) : min(image.shape[0], r0 + radius + 1),
            max(0, c0 - radius) : min(image.shape[1], c0 + radius + 1),
        ]
        q = np.column_stack([rr.ravel(), cc.ravel()]) - p[i]
        u = q @ tangent
        v = q @ normal
        choose = (np.abs(u) <= 3.5) & (np.abs(v) <= max(3.5 * sigma, 3 * w0))
        q = q[choose]
        y = source[rr.ravel()[choose], cc.ravel()[choose]]
        if len(y) < 25 or np.ptp(y) < 1e-10:
            continue
        scale = max(float(np.ptp(y)), 1e-8)
        kmax = min(0.2, 1 / (3 * w0 + 1))

        def predicted(x, details=False, normal=normal, tangent=tangent, q=q, y=y, scale=scale):
            shift, w, skew, theta, k = x
            nn = normal * np.cos(theta) + tangent * np.sin(theta)
            tt = np.array([-nn[1], nn[0]])
            v = q @ nn - shift
            u = q @ tt
            rad = np.sqrt((1 + k * v) ** 2 + (k * u) ** 2)
            distance = (2 * v + k * (v * v + u * u)) / (rad + 1)
            local_normal = ((1 + k * v)[:, None] * nn + (k * u)[:, None] * tt) / np.maximum(
                rad[:, None], 1e-12
            )
            # Leading pixel-curvature aperture term. Exact straight-line pixel
            # integration accounts for both image-axis pixel dimensions.
            distance = distance + k / (24 * np.maximum(rad, 0.25))
            template = integrated_profile(distance, w, skew, local_normal, model)
            z = template - template.mean()
            den = float(z @ z)
            amplitude = float(z @ (y - y.mean()) / max(den, 1e-15))
            bg = float(y.mean() - amplitude * template.mean())
            prediction = bg + amplitude * template
            if details:
                return prediction, amplitude, bg, nn
            return (prediction - y) / scale

        lower = [-max(sigma, 0.75), 0.2, -0.95, -0.35, -kmax]
        upper = [max(sigma, 0.75), max(4 * sigma, 2 * w0), 0.95, 0.35, kmax]
        try:
            fit = least_squares(
                predicted,
                [0, w0, skew0, 0, 0],
                bounds=(lower, upper),
                x_scale=[0.2, w0, 0.2, 0.05, 0.025],
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=max_nfev,
                jac="3-point",
            )
            prediction, amplitude, _bg, nn = predicted(fit.x, True)
            singular = np.linalg.svd(fit.jac, compute_uv=False)
            condition = float(singular[0] / max(singular[-1], 1e-15))
            rms = float(np.sqrt(np.mean((prediction - y) ** 2)))
            attrs["profile_fit_rms"][i] = rms
            attrs["profile_fit_condition"][i] = condition
            # Reject low-contrast, ill-conditioned or model-mismatched fits.
            if (
                not fit.success
                or amplitude <= 0
                or condition > 1e7
                or rms / scale > 0.06
                or np.any(fit.active_mask)
            ):
                continue
            shift, w, skew, _theta, _k = fit.x
            p[i] = original[i] + shift * nn
            n[i] = nn
            attrs["profile_fit_valid"][i] = True
            attrs["profile_fit_shift"][i] = shift
            attrs["width_left"][i] = attrs["width_right"][i] = w
            attrs["asymmetry"][i] = 2 * skew / (1 + abs(skew))
            attrs["contrast"][i] = (
                amplitude * (1 + abs(skew)) * (1 if light_dark == "light" else -1)
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
