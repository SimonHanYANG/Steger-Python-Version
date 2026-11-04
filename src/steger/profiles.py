"""Continuous cross-section models and numerical inversion of Steger's bias map.

Model coordinate x increases from the strong edge toward the weak edge.
The peak is 1, the strong-side background is 0 and weak-side background is a.
For bar/parabolic profiles w is the half-support; for Gaussian it is its standard
deviation, NOT its FWHM. All distances can be normalized to sigma=1.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr


def phi(x):
    return np.exp(-0.5 * np.asarray(x)**2) / np.sqrt(2 * np.pi)


def _moments(a, b):
    fa, fb = phi(a), phi(b)
    m = [ndtr(b) - ndtr(a), fa - fb]
    for k in range(2, 5):
        m.append(a**(k-1) * fa - b**(k-1) * fb + (k-1) * m[k-2])
    return m


def profile_response(x, w: float, asymmetry: float, model: str, sigma: float = 1.0):
    """Return (g*f, g'*f, g''*f) analytically, without sampling a synthetic image."""
    x = np.asarray(x, dtype=np.float64)
    a = asymmetry
    if model == "bar-shaped":
        u, v = (x + w) / sigma, (x - w) / sigma
        return (ndtr(u) - (1-a)*ndtr(v),
                (phi(u) - (1-a)*phi(v)) / sigma,
                (-u*phi(u) + (1-a)*v*phi(v)) / sigma**2)
    if model == "gaussian":
        s2 = w*w + sigma*sigma
        beta = w / (sigma * np.sqrt(s2))
        base = w / np.sqrt(s2) * np.exp(-x*x / (2*s2))
        b = 1 - a*ndtr(beta*x)
        bp = -a*beta*phi(beta*x)
        bpp = a*beta**3*x*phi(beta*x)
        return (base*b + a*ndtr(x/sigma),
                base*(-x/s2*b + bp) + a*phi(x/sigma)/sigma,
                base*((x*x/s2**2-1/s2)*b - 2*x/s2*bp + bpp)
                - a*x*phi(x/sigma)/sigma**3)
    if model != "parabolic":
        raise ValueError(f"Unknown profile model: {model}")
    result = [a*ndtr(x/sigma), a*phi(x/sigma)/sigma, -a*x*phi(x/sigma)/sigma**3]
    aa, bb, cc = 1-x*x/w**2, -2*x*sigma/w**2, -sigma*sigma/w**2
    for left, right, factor in [(-w, 0, 1.0), (0, w, 1-a)]:
        m = _moments((left-x)/sigma, (right-x)/sigma)
        result[0] = result[0] + factor*(aa*m[0]+bb*m[1]+cc*m[2])
        result[1] = result[1] + factor*(aa*m[1]+bb*m[2]+cc*m[3])/sigma
        result[2] = result[2] + factor*(aa*(m[2]-m[0])+bb*(m[3]-m[1])+cc*(m[4]-m[2]))/sigma**2
    return tuple(result)


@dataclass(frozen=True)
class ProfileGeometry:
    ridge: float
    edge_left: float
    edge_right: float
    gradient_left: float
    gradient_right: float

    @property
    def total_width(self):
        return self.edge_right-self.edge_left

    @property
    def gradient_ratio(self):
        return abs(self.gradient_right/self.gradient_left)


@lru_cache(maxsize=65536)
def profile_geometry(w: float, a: float, model: str) -> ProfileGeometry:
    if model == "bar-shaped":
        ridge = -np.log1p(-a)/(2*w)
    else:
        far = max(12.0, 9*w)
        ridge = brentq(lambda x: float(profile_response(x, w, a, model)[1]),
                       -max(1.0, w), far, xtol=2e-12)
    bound = max(12.0, abs(ridge)+8*w+8)
    def second(x):
        return float(profile_response(x, w, a, model)[2])
    # The first zero on either side is the edge associated with this ridge.
    def edge(sign):
        prev = ridge
        for step in np.geomspace(0.15, bound, 24):
            q = ridge + sign*step
            if second(q) >= 0:
                return brentq(second, min(prev,q), max(prev,q), xtol=2e-12)
            prev = q
        raise ValueError("No associated edge for this profile")
    left, right = edge(-1), edge(1)
    return ProfileGeometry(ridge, left, right,
                           float(profile_response(left,w,a,model)[1]),
                           float(profile_response(right,w,a,model)[1]))


def invert_bias(total_width: float, gradient_ratio: float, model: str,
                initial: tuple[float,float] | None = None) -> tuple[float,float,ProfileGeometry,float]:
    """Invert normalized (observed total width, weak/strong gradient ratio).

    Direct bounded numerical inversion replaces the coarse 1998 lookup table.
    An unattainable/ill-conditioned observation raises ValueError instead of
    inventing a valid-looking width. Returned residual is dimensionless.
    """
    if total_width <= 2.00001 or not 0 < gradient_ratio <= 1:
        raise ValueError("Unidentifiable profile: observed width <= 2 sigma or invalid gradient ratio")
    if initial is None:
        initial = (max(0.12, np.sqrt(max((total_width/2)**2-1, 0.01))),
                   min(0.9, max(0,1-gradient_ratio)))
    upper = max(6.0, total_width*2)
    def residual(x):
        try:
            g = profile_geometry(float(x[0]),float(x[1]),model)
            return np.array([(g.total_width-total_width)/total_width,
                             np.log(max(g.gradient_ratio,1e-12)/gradient_ratio)])
        except (ValueError, FloatingPointError):
            return np.array([10.0,10.0])
    fit = least_squares(residual, np.clip(initial,[0.03,1e-10],[upper-1e-6,0.995]),
                        bounds=([0.025,0],[upper,0.999]), xtol=1e-10, ftol=1e-10,
                        gtol=1e-10, max_nfev=50, x_scale="jac")
    err = float(np.linalg.norm(residual(fit.x)))
    if not fit.success or err > 0.025:
        raise ValueError(f"Profile inversion failed or model mismatch (residual {err:.4g})")
    w,a = map(float,fit.x)
    return w,a,profile_geometry(w,a,model),err
