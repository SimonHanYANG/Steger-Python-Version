# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Float64 cubic-reflect sampling of five fields, sharing coordinate weights.

Specializes scipy.ndimage.map_coordinates(order=3, mode='reflect',
prefilter=False). The spline coefficients still come from SciPy. The weight
formulas and row/column multiplication and accumulation order match ndimage;
compilation must disable fast-math and floating-point contraction.

Adapted from SciPy ndimage ni_splines.c / ni_interpolation.c (Peter J. Verveer,
2003-2005; SciPy developers), BSD-3-Clause. Full notice: THIRD_PARTY_NOTICES.md.
"""

import numpy as np
from libc.math cimport floor, fmod, isfinite, fabs, isnan


cdef inline double reflected(double x, Py_ssize_t n) noexcept nogil:
    cdef double period = 2.0 * n
    if n <= 1:
        return 0.0
    if x < 0.0:
        if x < -period:
            x = fmod(x, period)
        if x < -n:
            x += period
        else:
            x = (1e-15 if x > -1e-15 else -x) - 1.0
    elif x > n - 1:
        x = fmod(x, period)
        if x >= n:
            x = period - x - 1.0
    return x


cdef inline void weights(double x, double* w) noexcept nogil:
    cdef double y = x - floor(x)
    cdef double z = 1.0 - y
    w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0
    w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0
    w[0] = z * z * z / 6.0
    w[3] = 1.0
    w[3] -= w[0]
    w[3] -= w[1]
    w[3] -= w[2]


def sample_fields(const double[:, :, ::1] fields, const double[:, ::1] points):
    """Return [field, point] samples; never modifies input arrays."""
    if fields.shape[0] != 5 or fields.shape[1] < 3 or fields.shape[2] < 3:
        raise ValueError('Expected five float64 fields with shape at least 3x3')
    if points.shape[1] != 2:
        raise ValueError('Expected Nx2 float64 coordinates')
    cdef Py_ssize_t count = points.shape[0]
    cdef Py_ssize_t h = fields.shape[1], w = fields.shape[2]
    result = np.empty((5, count), dtype=np.float64)
    cdef double[:, ::1] out = result
    cdef Py_ssize_t i, f, a, b, sr, sc
    cdef Py_ssize_t ri[4]
    cdef Py_ssize_t ci[4]
    cdef double r, c, v, total
    cdef double rw[4]
    cdef double cw[4]
    for i in range(count):
        if not isfinite(points[i, 0]) or not isfinite(points[i, 1]):
            raise ValueError('coordinates must be finite')
    with nogil:
        for i in range(count):
            r = reflected(points[i, 0], h)
            c = reflected(points[i, 1], w)
            sr = <Py_ssize_t>floor(r) - 1
            sc = <Py_ssize_t>floor(c) - 1
            weights(r, rw)
            weights(c, cw)
            for a in range(4):
                ri[a] = <Py_ssize_t>reflected(<double>(sr + a), h)
                ci[a] = <Py_ssize_t>reflected(<double>(sc + a), w)
            for f in range(5):
                total = 0.0
                for a in range(4):
                    for b in range(4):
                        v = fields[f, ri[a], ci[b]]
                        v *= rw[a]
                        v *= cw[b]
                        total += v
                out[f, i] = total
    return result


def refine_points(const double[:, :, ::1] fields, double[:, ::1] points,
                  const double[:, ::1] normals, const Py_ssize_t[:, :] origins,
                  bint trace=False):
    """Same six per-polarity Newton iterations, without NumPy temporary arrays.

    Every point still participates in each iteration, even after convergence;
    the batch stopping rule, step cap and originating-pixel gate are unchanged.
    """
    cdef Py_ssize_t count = points.shape[0]
    if fields.shape[0] != 5 or fields.shape[1] < 3 or fields.shape[2] < 3:
        raise ValueError('Invalid coefficient fields')
    if points.shape[1] != 2 or normals.shape[0] != count or normals.shape[1] != 2:
        raise ValueError('Invalid point/normal shapes')
    if origins.shape[0] != count or origins.shape[1] != 2:
        raise ValueError('Invalid origin shape')
    cdef Py_ssize_t h = fields.shape[1], w = fields.shape[2]
    cdef Py_ssize_t iteration, i, f, a, b, sr, sc
    cdef Py_ssize_t ri[4]
    cdef Py_ssize_t ci[4]
    cdef double r, c, v, total, nr, nc, denom, step
    cdef double rw[4]
    cdef double cw[4]
    cdef double d[5]
    cdef double proposed_r, proposed_c, max_step
    cdef bint nan_step
    positions = []
    for i in range(count):
        if not isfinite(points[i, 0]) or not isfinite(points[i, 1]):
            raise ValueError('coordinates must be finite')
    for iteration in range(6):
        max_step = 0.0
        nan_step = False
        with nogil:
            for i in range(count):
                r = reflected(points[i, 0], h)
                c = reflected(points[i, 1], w)
                sr = <Py_ssize_t>floor(r) - 1
                sc = <Py_ssize_t>floor(c) - 1
                weights(r, rw)
                weights(c, cw)
                for a in range(4):
                    ri[a] = <Py_ssize_t>reflected(<double>(sr + a), h)
                    ci[a] = <Py_ssize_t>reflected(<double>(sc + a), w)
                for f in range(5):
                    total = 0.0
                    for a in range(4):
                        for b in range(4):
                            v = fields[f, ri[a], ci[b]]
                            v *= rw[a]
                            v *= cw[b]
                            total += v
                    d[f] = total
                nr, nc = normals[i, 0], normals[i, 1]
                denom = d[2] * nr * nr + 2 * d[3] * nr * nc + d[4] * nc * nc
                step = -(d[0] * nr + d[1] * nc) / denom if fabs(denom) > 1e-14 else 0.0
                if step < -0.25:
                    step = -0.25
                elif step > 0.25:
                    step = 0.25
                proposed_r = points[i, 0] + step * nr
                proposed_c = points[i, 1] + step * nc
                if fabs(proposed_r - origins[i, 0]) <= 1.5 and fabs(proposed_c - origins[i, 1]) <= 1.5:
                    points[i, 0] = proposed_r
                    points[i, 1] = proposed_c
                if isnan(step):
                    nan_step = True
                elif fabs(step) > max_step:
                    max_step = fabs(step)
        if trace:
            positions.append(np.asarray(points).copy())
        if not nan_step and max_step < 1e-8:
            break
    return np.asarray(points), positions
