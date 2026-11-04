"""Direction-aware hysteresis linking, topology and explicit junction completion."""

import numpy as np
from scipy.spatial import cKDTree

from .types import Contour, Junction


def orient_normals(points, normals):
    if len(points) < 2:
        return normals.copy()
    tangent = np.gradient(points, axis=0)
    right = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    out = normals.copy()
    out[np.einsum("ij,ij->i", out, right) < 0] *= -1
    return out


def link_points(points, normals, responses, high):
    """Link only components containing a high-threshold seed. No binary thinning."""
    if not len(points):
        return [], []
    # Several pixels may describe essentially the same subpixel line point.
    tree = cKDTree(points)
    keep = np.ones(len(points), dtype=bool)
    for i in np.argsort(-responses, kind="stable"):
        if keep[i]:
            for j in tree.query_ball_point(points[i], 0.22):
                if j != i:
                    keep[j] = False
    points, normals, responses = points[keep], normals[keep], responses[keep]
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, 1.85)
    graph = [set() for _ in points]
    for i, adjacent in enumerate(neighbors):
        tangent = np.array([-normals[i, 1], normals[i, 0]])
        for sign in (-1, 1):
            choices = []
            for j in adjacent:
                if i == j:
                    continue
                delta = points[j] - points[i]
                distance = np.linalg.norm(delta)
                advance = np.dot(delta, tangent) * sign
                if advance < 0.22 or advance / distance < 0.5:
                    continue
                cosine = np.clip(abs(np.dot(normals[i], normals[j])), 0, 1)
                angle = np.arccos(cosine)
                if angle > np.pi / 3:
                    continue
                choices.append((distance + angle, j))
            if choices:
                j = min(choices)[1]
                graph[i].add(j)
                graph[j].add(i)
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
            used.add(tuple(sorted((prev, cur))))
            path.append(cur)
            if len(graph[cur]) != 2 or cur == start:
                break
            nxt = next(j for j in graph[cur] if j != prev)
            if tuple(sorted((cur, nxt))) in used:
                break
            prev, cur = cur, nxt
        if len(path) >= 2:
            ids = np.asarray(path)
            idx = len(contours)
            p, n = points[ids].copy(), normals[ids].copy()
            contours.append(
                Contour(p, orient_normals(p, n), responses[ids].copy(), closed=path[0] == path[-1])
            )
            for k in (path[0], path[-1]):
                membership.setdefault(k, set()).add(idx)

    # Branches/endpoints first; remaining degree-two edges form closed loops.
    starts = [i for i in range(len(points)) if active[i] and len(graph[i]) != 2]
    starts += [i for i in range(len(points)) if active[i] and len(graph[i]) == 2]
    for i in starts:
        for j in sorted(graph[i]):
            if tuple(sorted((i, j))) not in used:
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
