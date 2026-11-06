"""Direction-aware hysteresis linking, topology and explicit junction completion."""

import os
import numpy as np
from scipy.spatial import cKDTree

from .types import Contour, Junction

try:
    if os.environ.get('STEGER_DISABLE_NATIVE') == '1':
        raise ImportError('Native linking disabled explicitly')
    from ._spline_native import deduplicate_indices as _native_deduplicate
    from ._spline_native import link_paths as _native_link_paths
    from ._spline_native import orient_path_normals as _native_orient
    from ._spline_native import neighbor_graph as _native_neighbor_graph
    from ._spline_native import radius_neighbors as _native_radius_neighbors
except ImportError:
    _native_deduplicate = _native_link_paths = _native_orient = _native_neighbor_graph = None
    _native_radius_neighbors = None


def orient_normals(points, normals):
    if len(points) < 2:
        return normals.copy()
    if (_native_orient is not None and isinstance(points, np.ndarray)
            and points.dtype == np.float64 and normals.dtype == np.float64):
        return _native_orient(points, normals)
    # Same unit-spacing first-order boundary / centered interior differences as
    # np.gradient, avoiding its general N-D argument processing for short paths.
    points = np.asarray(points)
    if points.dtype.kind not in "fc":
        points = points.astype(np.float64)
    tangent = np.empty_like(points)
    tangent[1:-1] = (points[2:] - points[:-2]) / 2.0
    tangent[0] = points[1] - points[0]
    tangent[-1] = points[-1] - points[-2]
    right = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    out = normals.copy()
    out[np.einsum("ij,ij->i", out, right) < 0] *= -1
    return out


def _neighbor_graph(points, normals, neighbors):
    """Batch the original two-coordinate norm/dot/arccos edge scoring.

    Keep i/sign order and the original (score, j) tie break when inserting
    undirected edges, so Python set traversal and contour ordering stay intact.
    """
    if (_native_neighbor_graph is not None and points.dtype == np.float64
            and normals.dtype == np.float64):
        native_graph = _native_neighbor_graph(points, normals, neighbors)
        if native_graph is not None:
            return native_graph
    graph = [set() for _ in points]
    ii = np.repeat(np.arange(len(points)), [len(x) for x in neighbors])
    jj = np.concatenate(neighbors).astype(np.intp, copy=False)
    other = ii != jj
    ii, jj = ii[other], jj[other]
    if not len(ii):
        return graph
    delta = points[jj] - points[ii]
    distance = np.linalg.norm(delta, axis=1)
    tangent = np.column_stack((-normals[ii, 1], normals[ii, 0]))
    projection = np.einsum("ij,ij->i", delta, tangent)
    advance = np.abs(projection)
    valid = (advance >= 0.22) & (advance / distance >= 0.5)
    ii, jj = ii[valid], jj[valid]
    distance, projection = distance[valid], projection[valid]
    cosine = np.abs(np.einsum("ij,ij->i", normals[ii], normals[jj]))
    angle = np.arccos(np.minimum(cosine, 1.0))
    valid = angle <= np.pi / 3
    ii, jj, projection = ii[valid], jj[valid], projection[valid]
    if not len(ii):
        return graph
    scores = distance[valid] + angle[valid]
    sign = (projection >= 0).astype(np.int8)
    order = np.lexsort((jj, scores, sign, ii))
    ii, jj, sign = ii[order], jj[order], sign[order]
    first = np.r_[True, (ii[1:] != ii[:-1]) | (sign[1:] != sign[:-1])]
    for i, j in zip(ii[first].tolist(), jj[first].tolist(), strict=True):
        graph[i].add(j)
        graph[j].add(i)
    return graph


def link_points(points, normals, responses, high):
    """Link only components containing a high-threshold seed. No binary thinning."""
    if not len(points):
        return [], []
    # Several pixels may describe essentially the same subpixel line point.
    keep = np.ones(len(points), dtype=bool)
    # Query dispatch is expensive for short necks. Batch small clouds; retain
    # the sequential path for large/dense inputs to bound temporary storage.
    duplicates = None
    if _native_radius_neighbors is not None and points.dtype == np.float64 and len(points) <= 192:
        duplicates = _native_radius_neighbors(points, .22)
    if duplicates is None:
        tree = cKDTree(points)
        duplicates = tree.query_ball_point(points, .22, return_sorted=False) if len(points) <= 512 else None
    order = np.argsort(-responses, kind="stable")
    if _native_deduplicate is not None and duplicates is not None:
        keep = _native_deduplicate(duplicates, order)
    else:
        for i in order:
            if keep[i]:
                for j in (
                    duplicates[i] if duplicates is not None else tree.query_ball_point(points[i], 0.22)
                ):
                    if j != i:
                        keep[j] = False
    points, normals, responses = points[keep], normals[keep], responses[keep]
    neighbors = None
    if _native_radius_neighbors is not None and points.dtype == np.float64 and len(points) <= 192:
        neighbors = _native_radius_neighbors(points, 1.85)
    if neighbors is None:
        tree = cKDTree(points)
        neighbors = tree.query_ball_point(points, 1.85)
    graph = _neighbor_graph(points, normals, neighbors)
    if _native_link_paths is not None and responses.dtype == np.float64:
        paths, membership = _native_link_paths(graph, np.ascontiguousarray(responses), high)
        contours = []
        for path in paths:
            ids = np.asarray(path)
            p, n = points[ids], normals[ids]
            contours.append(Contour(p, orient_normals(p, n), responses[ids], closed=path[0] == path[-1]))
        junctions = [Junction(points[k].copy(), tuple(sorted(v)))
                     for k, v in membership.items() if len(v) > 1]
        return contours, junctions
    active = np.zeros(len(points), dtype=bool)
    seen = set()
    for i in range(len(points)):
        if i in seen:
            continue
        component, stack = [], [i]
        seen.add(i)
        while stack:
            k = stack.pop()
            component.append(k)
            for j in graph[k]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        if np.any(responses[component] >= high):
            active[component] = True
    used = set()
    contours = []
    membership = {}

    def trace(start, second):
        path = [start]
        prev, cur = start, second
        while True:
            used.add((prev, cur) if prev < cur else (cur, prev))
            path.append(cur)
            if len(graph[cur]) != 2 or cur == start:
                break
            nxt = next(j for j in graph[cur] if j != prev)
            if ((cur, nxt) if cur < nxt else (nxt, cur)) in used:
                break
            prev, cur = cur, nxt
        if len(path) >= 2:
            ids = np.asarray(path)
            idx = len(contours)
            p, n = points[ids], normals[ids]
            contours.append(
                Contour(p, orient_normals(p, n), responses[ids], closed=path[0] == path[-1])
            )
            for k in (path[0], path[-1]):
                membership.setdefault(k, set()).add(idx)

    # Branches/endpoints first; remaining degree-two edges form closed loops.
    starts = [i for i in range(len(points)) if active[i] and len(graph[i]) != 2]
    starts += [i for i in range(len(points)) if active[i] and len(graph[i]) == 2]
    for i in starts:
        for j in sorted(graph[i]):
            if ((i, j) if i < j else (j, i)) not in used:
                trace(i, j)
    junctions = [
        Junction(points[k].copy(), tuple(sorted(v))) for k, v in membership.items() if len(v) > 1
    ]
    return contours, junctions


def complete_junctions(contours, sigma):
    """Conservative endpoint-to-contour continuation across a missing junction.

    This is an independent documented implementation, not a claim to reproduce
    HALCON's undocumented completion decisions. Added samples are explicitly
    flagged and must not be used as measured data in angle estimation.
    """
    completed_points = []
    for _ in range(max(1, len(contours) * 2)):
        best = None
        for i, c in enumerate(contours):
            if c.closed or len(c.points) < 4:
                continue
            for end in (0, -1):
                p = c.points[end]
                if any(
                    np.min(np.linalg.norm(other.points - p, axis=1)) < 0.25
                    for j, other in enumerate(contours)
                    if j != i
                ):
                    continue
                k = min(4, len(c.points) - 1)
                t = p - c.points[k if end == 0 else -1 - k]
                t /= max(np.linalg.norm(t), 1e-12)
                for j, other in enumerate(contours):
                    if i == j:
                        continue
                    delta = other.points - p
                    dist = np.linalg.norm(delta, axis=1)
                    advance = delta @ t
                    lateral = np.abs(delta[:, 0] * t[1] - delta[:, 1] * t[0])
                    valid = (
                        (dist > 0.35)
                        & (dist <= 2.5 * sigma)
                        & (advance > 0.3)
                        & (advance / np.maximum(dist, 1e-12) > 0.92)
                        & (lateral < 0.65)
                    )
                    # Endpoint already attached to any contour: never extend it again.
                    ids = np.flatnonzero(valid)
                    if len(ids):
                        k2 = ids[np.argmin(dist[ids])]
                        item = (float(dist[k2]), i, end, j, int(k2))
                        if best is None or item < best:
                            best = item
        if best is None:
            break
        _, i, end, j, k = best
        c, other = contours[i], contours[j]
        target = other.points[k].copy()
        # If an endpoint is already a topological junction, leave it alone.
        if any(np.linalg.norm(c.points[end] - p) < 0.25 for p in completed_points):
            break
        steps = max(2, int(np.ceil(np.linalg.norm(target - c.points[end]) * 2)))
        added = np.linspace(c.points[end], target, steps + 1)[1:]
        normals = np.repeat(c.normals[end][None, :], steps, axis=0)
        response = np.repeat(c.response[end], steps)
        old_flag = c.attributes.get("junction_completed", np.zeros(len(c.points), dtype=bool))
        if end == 0:
            c.points = np.vstack([added[::-1], c.points])
            c.normals = np.vstack([normals[::-1], c.normals])
            c.response = np.r_[response, c.response]
            c.attributes["junction_completed"] = np.r_[np.ones(steps, dtype=bool), old_flag]
        else:
            c.points = np.vstack([c.points, added])
            c.normals = np.vstack([c.normals, normals])
            c.response = np.r_[c.response, response]
            c.attributes["junction_completed"] = np.r_[old_flag, np.ones(steps, dtype=bool)]
        completed_points.append(target)
        if 0 < k < len(other.points) - 1:

            def part(sl, other=other):
                return Contour(
                    other.points[sl].copy(),
                    other.normals[sl].copy(),
                    other.response[sl].copy(),
                    {key: v[sl].copy() for key, v in other.attributes.items()},
                )

            contours[j] = part(slice(0, k + 1))
            contours.append(part(slice(k, None)))
    # Reconstruct memberships after splitting so no stale indices are returned.
    endpoints = [(i, c.points[e]) for i, c in enumerate(contours) for e in (0, -1)]
    junctions = []
    seen = []
    for _i, p in endpoints:
        if any(np.linalg.norm(p - q) < 0.25 for q in seen):
            continue
        members = tuple(sorted({j for j, q in endpoints if np.linalg.norm(p - q) < 0.25}))
        if len(members) > 1:
            seen.append(p)
            junctions.append(
                Junction(
                    p.copy(), members, any(np.linalg.norm(p - q) < 0.25 for q in completed_points)
                )
            )
    return contours, junctions
