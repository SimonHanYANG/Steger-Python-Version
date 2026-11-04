"""Integrated Gaussian masks, Steger 1998 section 2.3, equations (16)-(18).

No Sobel finite differences, integer derivative buffers, image rescaling, or
scale-normalized response. All convolution and eigendecomposition use float64.
"""

from functools import lru_cache

import numpy as np
from scipy.ndimage import convolve1d, map_coordinates
from scipy.special import ndtr

from .types import FloatArray


@lru_cache(maxsize=64)
def gaussian_masks(sigma: float, truncate: float = 6.0) -> tuple[FloatArray, ...]:
    radius = max(2, int(np.ceil(truncate * sigma + 0.5)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    lo, hi = x - 0.5, x + 0.5
    pdf_lo = np.exp(-0.5 * (lo / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    pdf_hi = np.exp(-0.5 * (hi / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    g0 = ndtr(hi / sigma) - ndtr(lo / sigma)
    g1 = pdf_hi - pdf_lo
    g2 = (-hi * pdf_hi + lo * pdf_lo) / sigma**2
    # Fold the tiny truncated tails into the boundary bins, rather than
    # renormalizing derivative amplitudes (which changes HALCON thresholds).
    g0[[0, -1]] += [ndtr(lo[0] / sigma), ndtr(-hi[-1] / sigma)]
    g1[0], g1[-1] = pdf_hi[0], -pdf_lo[-1]
    g2[0], g2[-1] = -hi[0] * pdf_hi[0] / sigma**2, lo[-1] * pdf_lo[-1] / sigma**2
    for g in (g0, g1, g2):
        g.flags.writeable = False
    return g0, g1, g2


def image_derivatives(
    image: FloatArray, sigma: float, *, border: str = "reflect"
) -> dict[str, FloatArray]:
    g0, g1, g2 = gaussian_masks(sigma)
    smooth_r = convolve1d(image, g0, axis=0, mode=border)
    first_r = convolve1d(image, g1, axis=0, mode=border)
    second_r = convolve1d(image, g2, axis=0, mode=border)
    return {
        "r": convolve1d(first_r, g0, axis=1, mode=border),
        "c": convolve1d(smooth_r, g1, axis=1, mode=border),
        "rr": convolve1d(second_r, g0, axis=1, mode=border),
        "rc": convolve1d(first_r, g1, axis=1, mode=border),
        "cc": convolve1d(smooth_r, g2, axis=1, mode=border),
    }


def sample(array: FloatArray, points: FloatArray, *, order: int = 3) -> FloatArray:
    return map_coordinates(
        array, np.asarray(points).T, order=order, mode="reflect", prefilter=order > 1
    )


def hessian_normals(d: dict[str, FloatArray], polarity: str) -> tuple[FloatArray, FloatArray]:
    shape = d["r"].shape
    h = np.empty((*shape, 2, 2), dtype=np.float64)
    h[..., 0, 0], h[..., 0, 1] = d["rr"], d["rc"]
    h[..., 1, 0], h[..., 1, 1] = d["rc"], d["cc"]
    values, vectors = np.linalg.eigh(h)
    choose = np.argmax(np.abs(values), axis=-1)
    lam = np.take_along_axis(values, choose[..., None], axis=-1)[..., 0]
    normal = np.take_along_axis(vectors, choose[..., None, None], axis=-1)[..., 0]
    response = -lam if polarity == "light" else lam
    return normal, response
