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
from libc.math cimport floor, fmod, isfinite, fabs, isnan, sqrt, acos


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


def deduplicate_indices(neighbors, const Py_ssize_t[::1] order):
    """Identical stable-priority suppression; neighbors still come from cKDTree."""
    keep = np.ones(len(neighbors), dtype=np.uint8)
    cdef unsigned char[::1] flags = keep
    cdef Py_ssize_t i, j, k
    for k in range(order.shape[0]):
        i = order[k]
        if flags[i]:
            for item in neighbors[i]:
                j = item
                if j != i:
                    flags[j] = 0
    return keep.astype(np.bool_)


cdef void _trace_path(list graph, set used, list paths, dict membership,
                      Py_ssize_t start, Py_ssize_t second):
    cdef list path = [start]
    cdef Py_ssize_t prev = start, cur = second, nxt, k, index
    while True:
        used.add((prev, cur) if prev < cur else (cur, prev))
        path.append(cur)
        if len(graph[cur]) != 2 or cur == start:
            break
        for item in graph[cur]:
            nxt = item
            if nxt != prev:
                break
        if ((cur, nxt) if cur < nxt else (nxt, cur)) in used:
            break
        prev, cur = cur, nxt
    if len(path) >= 2:
        index = len(paths)
        paths.append(path)
        for k in (path[0], path[len(path)-1]):
            membership.setdefault(k, set()).add(index)


def link_paths(list graph, const double[::1] responses, double high):
    """Same graph traversal, edge order, components and junction membership."""
    cdef Py_ssize_t count = len(graph), i, j, k
    if responses.shape[0] != count:
        raise ValueError('Graph/response sizes must match')
    active = np.zeros(count, dtype=np.uint8)
    seen = np.zeros(count, dtype=np.uint8)
    cdef unsigned char[::1] active_view = active, seen_view = seen
    cdef list stack, component, starts, paths = []
    cdef dict membership = {}
    cdef set used = set()
    cdef bint seeded
    for i in range(count):
        if seen_view[i]:
            continue
        component, stack = [], [i]
        seen_view[i] = 1
        seeded = False
        while stack:
            k = stack.pop()
            component.append(k)
            if responses[k] >= high:
                seeded = True
            for item in graph[k]:
                j = item
                if not seen_view[j]:
                    seen_view[j] = 1
                    stack.append(j)
        if seeded:
            for item in component:
                active_view[<Py_ssize_t>item] = 1
    starts = [i for i in range(count) if active_view[i] and len(graph[i]) != 2]
    starts += [i for i in range(count) if active_view[i] and len(graph[i]) == 2]
    for item in starts:
        i = item
        for other in sorted(graph[i]):
            j = other
            if ((i, j) if i < j else (j, i)) not in used:
                _trace_path(graph, used, paths, membership, i, j)
    return paths, membership


def orient_path_normals(const double[:, :] points, const double[:, :] normals):
    """Original first/centered differences and two-product sign check."""
    cdef Py_ssize_t count = points.shape[0], i
    if count < 2 or points.shape[1] != 2 or normals.shape[0] != count or normals.shape[1] != 2:
        raise ValueError('Expected matching Nx2 points and normals, N >= 2')
    cdef double tr, tc, dot
    result = np.array(normals, copy=True)
    cdef double[:, ::1] out = result
    with nogil:
        for i in range(count):
            if i == 0:
                tr = points[1, 0] - points[0, 0]
                tc = points[1, 1] - points[0, 1]
            elif i == count - 1:
                tr = points[i, 0] - points[i-1, 0]
                tc = points[i, 1] - points[i-1, 1]
            else:
                tr = (points[i+1, 0] - points[i-1, 0]) / 2.0
                tc = (points[i+1, 1] - points[i-1, 1]) / 2.0
            dot = normals[i, 0] * tc + normals[i, 1] * -tr
            if dot < 0:
                out[i, 0] *= -1
                out[i, 1] *= -1
    return result


def neighbor_graph(const double[:, :] points, const double[:, :] normals, neighbors):
    """Same edge score/order; defer numerically ambiguous decisions to NumPy."""
    cdef Py_ssize_t count=points.shape[0], i, j, best_j, sign, current_sign
    cdef double dr, dc, distance, projection, advance, cosine, angle, score, best
    cdef double limit = 1.0471975511965976
    if points.shape[1] != 2 or normals.shape[0] != count or normals.shape[1] != 2 or len(neighbors) != count:
        raise ValueError('Invalid point/normal/neighbor shapes')
    graph = [set() for _ in range(count)]
    for i in range(count):
        for sign in range(2):
            best_j = -1
            best = 1e300
            for item in neighbors[i]:
                j = item
                if j == i:
                    continue
                dr = points[j,0]-points[i,0]
                dc = points[j,1]-points[i,1]
                distance = sqrt(dr*dr+dc*dc)
                projection = dr * -normals[i,1] + dc * normals[i,0]
                advance = fabs(projection)
                if fabs(advance-.22)<1e-12 or (distance>0 and fabs(advance/distance-.5)<1e-12):
                    return None
                if not (advance >= .22 and distance > 0 and advance/distance >= .5):
                    continue
                current_sign = 1 if projection >= 0 else 0
                if current_sign != sign:
                    continue
                cosine = fabs(normals[i,0]*normals[j,0]+normals[i,1]*normals[j,1])
                if fabs(cosine-.5)<1e-12:
                    return None
                angle = acos(min(cosine,1.0))
                if not angle <= limit:
                    continue
                score = distance+angle
                if best_j >= 0 and fabs(score-best)<1e-12*(1+fabs(score)):
                    return None
                if score < best:
                    best, best_j = score, j
            if best_j >= 0:
                graph[i].add(best_j)
                graph[best_j].add(i)
    return graph


def window_samples(const double[:, :] points, const double[::1] arc,
                    const double[::1] starts, const double[::1] offsets):
    """Linear arc interpolation and centering with the original arithmetic."""
    cdef Py_ssize_t count=points.shape[0], windows=starts.shape[0], samples=offsets.shape[0]
    cdef Py_ssize_t i,j,k,axis
    cdef double q,v,total_r,total_c,mean_r,mean_c
    if count < 2 or arc.shape[0] != count or points.shape[1] != 2 or samples < 1:
        raise ValueError('Invalid window sampling shapes')
    centered = np.empty((windows,samples,2), dtype=np.float64)
    travel = np.empty((windows,2), dtype=np.float64)
    slopes = np.empty((count-1,2), dtype=np.float64)
    cdef double[:,:,::1] out=centered
    cdef double[:,::1] movement=travel, gradient=slopes
    with nogil:
        for k in range(count-1):
            for axis in range(2):
                gradient[k,axis]=(points[k+1,axis]-points[k,axis])/(arc[k+1]-arc[k])
        for i in range(windows):
            k=0
            total_r=0.
            total_c=0.
            for j in range(samples):
                q=starts[i]+offsets[j]
                while k < count-1 and arc[k+1] <= q:
                    k+=1
                for axis in range(2):
                    if q <= arc[0]:
                        v=points[0,axis]
                    elif k == count-1:
                        v=points[count-1,axis]
                    elif q == arc[k]:
                        v=points[k,axis]
                    else:
                        v=gradient[k,axis]*(q-arc[k])+points[k,axis]
                    out[i,j,axis]=v
                total_r+=out[i,j,0]
                total_c+=out[i,j,1]
            movement[i,0]=out[i,samples-1,0]-out[i,0,0]
            movement[i,1]=out[i,samples-1,1]-out[i,0,1]
            mean_r=total_r/samples
            mean_c=total_c/samples
            for j in range(samples):
                out[i,j,0]-=mean_r
                out[i,j,1]-=mean_c
    return centered,travel


def polyline_arcs(const double[:, :] points):
    """Original two-coordinate Euclidean lengths and sequential cumulative sum."""
    cdef Py_ssize_t count=points.shape[0], i
    cdef double dr,dc,total=0.
    if count < 1 or points.shape[1] != 2:
        raise ValueError('Expected nonempty Nx2 points')
    result=np.empty(count,dtype=np.float64)
    cdef double[::1] out=result
    out[0]=0.
    with nogil:
        for i in range(1,count):
            dr=points[i,0]-points[i-1,0]
            dc=points[i,1]-points[i-1,1]
            total+=sqrt(dr*dr+dc*dc)
            out[i]=total
    return result


def distinct_samples(const double[:, :] points):
    cdef Py_ssize_t count=points.shape[0],i
    cdef double dr,dc
    if count < 1 or points.shape[1] != 2:
        raise ValueError('Expected nonempty Nx2 points')
    result=np.empty(count,dtype=np.uint8)
    cdef unsigned char[::1] flags=result
    flags[0]=1
    with nogil:
        for i in range(1,count):
            dr=points[i,0]-points[i-1,0]
            dc=points[i,1]-points[i-1,1]
            flags[i]=1 if sqrt(dr*dr+dc*dc)>1e-9 else 0
    return result.view(np.bool_)


def radius_neighbors(const double[:, :] points, double radius):
    """Exact small-cloud radius graph; ambiguous boundary uses the tree path."""
    cdef Py_ssize_t count=points.shape[0],i,j
    cdef double dr,dc,squared,limit=radius*radius
    if points.shape[1] != 2 or not isfinite(radius) or radius <= 0:
        raise ValueError('Expected Nx2 points and a finite positive radius')
    neighbors=[[] for _ in range(count)]
    for i in range(count):
        neighbors[i].append(i)
        for j in range(i+1,count):
            dr=points[j,0]-points[i,0]
            dc=points[j,1]-points[i,1]
            squared=dr*dr+dc*dc
            if fabs(squared-limit) < 1e-12 * max(1.0,limit):
                return None
            if squared <= limit:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors
