"""Bidirectional centerline comparison; never treat point spacing as error."""

import numpy as np


def _curves(data):
    return [np.column_stack([c["row"], c["col"]]).astype(float) for c in data["contours"]]


def _densify(curves, spacing=0.25):
    out = []
    for p in curves:
        if len(p) < 2:
            continue
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        keep = np.r_[True, np.diff(s) > 1e-12]
        s, p = s[keep], p[keep]
        if s[-1] <= 0:
            continue
        q = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / spacing)) + 1))
        out.append(np.column_stack([np.interp(q, s, p[:, k]) for k in range(2)]))
    return np.vstack(out) if out else np.empty((0, 2))


def _distance(points, curves):
    segments = [(p[:-1], p[1:]) for p in curves if len(p) > 1]
    if not len(points):
        return np.empty(0)
    if not segments:
        return np.full(len(points), np.inf)
    a = np.vstack([s[0] for s in segments])
    b = np.vstack([s[1] for s in segments])
    v = b - a
    d2 = np.einsum("ij,ij->i", v, v)
    # Chunk to avoid unbounded N*M allocation for large images.
    answer = []
    for start in range(0, len(points), 128):
        q = points[start : start + 128, None, :] - a[None, :, :]
        t = np.clip(np.einsum("nsi,si->ns", q, v) / np.maximum(d2, 1e-20), 0, 1)
        d = np.linalg.norm(q - t[:, :, None] * v[None, :, :], axis=2)
        answer.append(d.min(axis=1))
    return np.concatenate(answer)


def compare_results(candidate, reference, *, tolerance_px=0.1, p95_limit_px=0.05):
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("tolerance_px must be positive")
    a, b = _curves(candidate), _curves(reference)
    aa, bb = _densify(a), _densify(b)

    def stats(d):
        if not len(d):
            return {"count": 0, "mean_px": None, "p95_px": None, "max_px": None, "coverage": 0.0}
        return {
            "count": len(d),
            "mean_px": float(np.mean(d)),
            "p95_px": float(np.percentile(d, 95)),
            "max_px": float(np.max(d)),
            "coverage": float(np.mean(d <= tolerance_px)),
        }

    da, db = stats(_distance(aa, b)), stats(_distance(bb, a))
    passed = bool(
        len(aa)
        and len(bb)
        and da["p95_px"] <= p95_limit_px
        and db["p95_px"] <= p95_limit_px
        and da["coverage"] >= 0.99
        and db["coverage"] >= 0.99
    )
    return {
        "scope": "centerline geometry only; not attribute/topology parity",
        "candidate_to_reference": da,
        "reference_to_candidate": db,
        "candidate_contours": len(a),
        "reference_contours": len(b),
        "tolerance_px": tolerance_px,
        "p95_limit_px": p95_limit_px,
        "passes_geometry_gate": passed,
        "reference_engine": reference.get("engine", "unspecified"),
    }
