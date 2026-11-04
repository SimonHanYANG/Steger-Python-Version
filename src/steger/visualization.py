"""Recorded Steger process visualizations; plotting never reruns the detector.

Requires the optional matplotlib dependency (uv sync --extra visualization).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.ndimage import map_coordinates, spline_filter

from .io import write_json


def _bounds(trace):
    selected = np.argwhere(trace.mask)
    if not len(selected) or np.all(trace.mask):
        return (0, trace.image.shape[0] - 1, 0, trace.image.shape[1] - 1)
    pad = 9
    lo = np.maximum(selected.min(axis=0) - pad, 0)
    hi = np.minimum(selected.max(axis=0) + pad, np.asarray(trace.image.shape) - 1)
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _frame(axis, bounds, title):
    r0, r1, c0, c1 = bounds
    axis.set(xlim=(c0 - 0.5, c1 + 0.5), ylim=(r1 + 0.5, r0 - 0.5), xticks=[], yticks=[])
    axis.set_title(title, fontsize=10, pad=9)


def _gray(axis, trace):
    lo, hi = np.percentile(trace.image, [0.5, 99.5])
    axis.imshow(trace.image, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-9), interpolation="nearest")


def _heat(fig, axis, array, bounds, *, signed=False, cmap="magma"):
    r0, r1, c0, c1 = bounds
    subset = array[r0 : r1 + 1, c0 : c1 + 1]
    scale = max(float(np.percentile(np.abs(subset), 99)), 1e-12)
    art = axis.imshow(
        array,
        cmap="coolwarm" if signed else cmap,
        vmin=-scale if signed else 0,
        vmax=scale,
        interpolation="nearest",
    )
    fig.colorbar(art, ax=axis, shrink=0.72, fraction=0.035, pad=0.02).ax.tick_params(labelsize=7)


def _axes_at(axis, points, normals, *, tangent=False, color="#24d8ff", length=2.5, max_axes=24):
    if not len(points):
        return
    stride = max(1, int(np.ceil(len(points) / max_axes)))
    points, vectors = points[::stride], normals[::stride]
    if tangent:
        vectors = np.column_stack([-vectors[:, 1], vectors[:, 0]])
    segments = np.stack([points - length * vectors, points + length * vectors], axis=1)
    axis.add_collection(LineCollection(segments[..., ::-1], colors=color, linewidths=1.2))


def _curves(axis, curves, *, normals=False, tangents=False):
    for c in curves:
        if normals:
            _axes_at(axis, c.points, c.normals)
        if tangents:
            _axes_at(
                axis, c.points, c.normals, tangent=True, color="#ffbd38", length=1.4, max_axes=10
            )
        axis.plot(c.cols, c.rows, color="#38ff8b", lw=1.1, zorder=4)


def _eigen_maps(trace):
    choice = np.argmax(np.abs(trace.eigenvalues), axis=-1)
    dominant = np.take_along_axis(trace.eigenvalues, choice[..., None], axis=-1)[..., 0]
    other = trace.eigenvalues.sum(axis=-1) - dominant
    gap = (np.abs(dominant) - np.abs(other)) / np.maximum(np.abs(dominant), 1e-15)
    return dominant, other, gap


def _save_overview(result, target, label):
    trace = result.trace
    bounds = _bounds(trace)
    dominant, _, _ = _eigen_maps(trace)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.subplots_adjust(left=0.025, right=0.98, top=0.88, bottom=0.075, wspace=0.13, hspace=0.26)
    titles = [
        "1  Original grayscale + mask boundary",
        "2  Gaussian-smoothed image",
        "3  First derivative magnitude |grad L|",
        "4  Dominant Hessian eigenvalue",
        "5  Positive line response",
        "6  Initial pixel candidates",
        "7  Subpixel candidates + Hessian normal axes",
        "8  Linked line + normal / tangent axes",
        "9  Final corrected centerline",
    ]
    for ax, title in zip(axes.flat, titles, strict=True):
        _frame(ax, bounds, title)
    _gray(axes[0, 0], trace)
    if not np.all(trace.mask) and np.any(trace.mask):
        axes[0, 0].contour(trace.mask, levels=[0.5], colors=["#24d8ff"], linewidths=0.7)
    lo, hi = np.percentile(trace.image, [0.5, 99.5])
    axes[0, 1].imshow(trace.smoothed, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-9))
    _heat(fig, axes[0, 2], np.hypot(trace.derivatives["r"], trace.derivatives["c"]), bounds)
    _heat(fig, axes[1, 0], dominant, bounds, signed=True)
    _heat(fig, axes[1, 1], np.maximum(trace.response, 0), bounds)
    for ax in [axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]:
        _gray(ax, trace)
    pixels = trace.initial_candidate_pixels
    if len(pixels):
        strong = trace.response[pixels[:, 0], pixels[:, 1]] >= result.parameters["high"]
        for select, color, name in [(~strong, "#ffbd38", "Weak"), (strong, "#ff5366", "Strong")]:
            shown = pixels[select][:: max(1, int(np.ceil(select.sum() / 3000)))]
            axes[1, 2].scatter(shown[:, 1], shown[:, 0], s=5, color=color, label=name)
        axes[1, 2].legend(fontsize=7, loc="upper left")
    candidates = trace.candidate_points
    if len(candidates):
        shown = candidates[:: max(1, int(np.ceil(len(candidates) / 3000)))]
        axes[2, 0].scatter(shown[:, 1], shown[:, 0], s=3, color="#ff70e1")
        _axes_at(axes[2, 0], candidates, trace.candidate_normals)
    _curves(axes[2, 1], trace.linked_contours, normals=True, tangents=True)
    _curves(axes[2, 2], result.contours)
    fig.suptitle(
        "Steger algorithm: recorded intermediate stages", fontsize=18, y=0.97, fontweight="bold"
    )
    fig.text(0.5, 0.925, label, ha="center", fontsize=10, color="#526071")
    fig.text(
        0.5,
        0.025,
        "Cyan: Hessian normal axis   |   Amber: perpendicular tangent axis   |   Green: ordered centerline\n"
        "Raw Hessian normals have a +/- sign ambiguity. Linking orients normals consistently. Display axes have fixed length.",
        ha="center",
        fontsize=9,
        linespacing=1.7,
    )
    fig.savefig(target, dpi=150, facecolor="white")
    plt.close(fig)


def _save_derivatives(result, target, label):
    trace = result.trace
    dominant, other, gap = _eigen_maps(trace)
    arrays = [trace.derivatives[k] for k in ("r", "c", "rr", "rc", "cc")]
    arrays += [dominant, other, gap]
    titles = [
        "L_r: first row derivative",
        "L_c: first column derivative",
        "L_rr: second row derivative",
        "L_rc: mixed derivative",
        "L_cc: second column derivative",
        "lambda_n: largest absolute eigenvalue",
        "lambda_t: other eigenvalue",
        "Eigenvalue separation (0 = ambiguous)",
    ]
    bounds = _bounds(trace)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.subplots_adjust(top=0.85, bottom=0.07, left=0.025, right=0.98, wspace=0.20, hspace=0.30)
    for i, (ax, array, title) in enumerate(zip(axes.flat, arrays, titles, strict=True)):
        _frame(ax, bounds, title)
        _heat(fig, ax, array, bounds, signed=i < 7, cmap="viridis")
    fig.suptitle(
        "Gaussian derivatives and Hessian eigenvalues", fontsize=18, y=0.97, fontweight="bold"
    )
    fig.text(
        0.5,
        0.91,
        label + "\nColor scales are for display; raw float64 arrays are included in process.npz.",
        ha="center",
        fontsize=10,
        color="#526071",
        linespacing=1.6,
    )
    fig.savefig(target, dpi=150, facecolor="white")
    plt.close(fig)


def _save_normals(result, target, label):
    trace = result.trace
    bounds = _bounds(trace)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    fig.subplots_adjust(top=0.79, bottom=0.15, left=0.025, right=0.98, wspace=0.09)
    for ax in axes:
        _gray(ax, trace)
    _frame(axes[0], bounds, "Hessian normals before width / bias correction")
    _frame(axes[1], bounds, "Final normals (may include extra profile fitting)")
    _curves(axes[0], trace.linked_contours, normals=True, tangents=True)
    _curves(axes[1], result.contours, normals=True, tangents=True)
    fig.suptitle(
        "Normal and tangent directions along the line", fontsize=18, y=0.97, fontweight="bold"
    )
    fig.text(0.5, 0.885, label, ha="center", fontsize=10, color="#526071")
    fig.text(
        0.5,
        0.045,
        "Cyan segments: normal axes   |   Amber segments: tangent axes   |   Green: centerline\n"
        "Axis lengths are illustrative, not measured widths. Normals are unreliable when Hessian eigenvalues are similar.",
        ha="center",
        fontsize=10,
        linespacing=1.7,
    )
    fig.savefig(target, dpi=165, facecolor="white")
    plt.close(fig)


def _save_point(result, target, requested=None):
    trace = result.trace
    if not len(trace.candidate_points):
        return None
    if requested is None:
        curve = (
            max(trace.linked_contours, key=lambda c: c.length) if trace.linked_contours else None
        )
        requested = (
            curve.points[len(curve.points) // 2]
            if curve is not None
            else trace.candidate_points[np.argmax(trace.candidate_response)]
        )
    else:
        requested = np.asarray(requested, dtype=float)
        if requested.shape != (2,) or not np.isfinite(requested).all():
            raise ValueError("inspect_point must be a finite (row, column) pair")
    index = int(np.argmin(np.linalg.norm(trace.candidate_points - requested, axis=1)))
    pixel = trace.candidate_pixels[index]
    row, col = map(int, pixel)
    n = trace.hessian_normals[row, col]
    tangent = np.array([-n[1], n[0]])
    h = np.array(
        [
            [trace.derivatives["rr"][row, col], trace.derivatives["rc"][row, col]],
            [trace.derivatives["rc"][row, col], trace.derivatives["cc"][row, col]],
        ]
    )
    gradient = np.array([trace.derivatives["r"][row, col], trace.derivatives["c"][row, col]])
    lam = float(n @ h @ n)
    taylor = float(trace.taylor_t[row, col])
    refined = trace.candidate_points[index]
    t_refined = float((refined - pixel) @ n)
    distance = np.linspace(
        -max(4, 3 * result.parameters["sigma"]), max(4, 3 * result.parameters["sigma"]), 401
    )
    locations = pixel + distance[:, None] * n

    def sample(array):
        return map_coordinates(
            spline_filter(array, order=3, mode="reflect"),
            locations.T,
            order=3,
            prefilter=False,
            mode="reflect",
        )

    original, smooth = sample(trace.image), sample(trace.smoothed)
    derivative = sample(trace.derivatives["r"]) * n[0] + sample(trace.derivatives["c"]) * n[1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7))
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.075, right=0.97, wspace=0.30, hspace=0.33)
    fig.suptitle(
        f"Inspect an actual candidate: pixel (row={row}, col={col})",
        fontsize=17,
        y=0.97,
        fontweight="bold",
    )
    _gray(axes[0, 0], trace)
    axes[0, 0].set(
        xlim=(col - 4.5, col + 4.5),
        ylim=(row + 4.5, row - 4.5),
        xlabel="Column (px)",
        ylabel="Row (px)",
        title="Pixel center, axes, and subpixel point",
    )
    axes[0, 0].set_xticks(np.arange(col - 4.5, col + 5, 1), minor=True)
    axes[0, 0].set_yticks(np.arange(row - 4.5, row + 5, 1), minor=True)
    axes[0, 0].grid(which="minor", alpha=0.15, color="white")
    _axes_at(axes[0, 0], pixel[None], n[None], length=3.5)
    _axes_at(axes[0, 0], pixel[None], tangent[None], length=2.8, color="#ffbd38")
    axes[0, 0].scatter([col], [row], color="#ff5366", s=50, label="Pixel center", zorder=5)
    axes[0, 0].scatter(
        [refined[1]],
        [refined[0]],
        color="#38ff8b",
        marker="x",
        s=80,
        label="Located subpixel point",
        zorder=6,
    )
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[0, 1].plot(distance, original, color="#94a3b8", label="Original grayscale (cubic sample)")
    axes[0, 1].plot(distance, smooth, color="#2563eb", label="Gaussian-smoothed grayscale")
    axes[0, 1].set(
        xlabel="Signed distance along n (px)",
        ylabel="Intensity",
        title="Cross section through the same pixel",
    )
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(distance, derivative, color="#2563eb", label="Sampled grad L dot n")
    # The Taylor approximation is local; do not extend it across the entire profile.
    local = np.abs(distance) <= 1
    axes[1, 0].plot(
        distance[local],
        gradient @ n + distance[local] * lam,
        color="#d97706",
        ls="--",
        label="Local Taylor: grad L dot n + s lambda_n",
    )
    axes[1, 0].axhline(0, color="#64748b", lw=0.8)
    axes[1, 0].axvline(taylor, color="#ff5366", lw=1, ls=":", label=f"Taylor t = {taylor:.4f} px")
    location_stage = "After Newton" if trace.newton_positions else "Located (Taylor)"
    axes[1, 0].axvline(
        t_refined, color="#059669", lw=1, label=f"{location_stage} = {t_refined:.4f} px"
    )
    axes[1, 0].set(
        xlabel="Signed distance along n (px)",
        ylabel="First directional derivative",
        title="Find the zero crossing along the normal",
    )
    axes[1, 0].legend(fontsize=7.5)
    axes[1, 1].axis("off")
    info = (
        "Values recorded by the detector\n\n"
        f"H = [[{h[0, 0]: .6f}, {h[0, 1]: .6f}],\n"
        f"     [{h[1, 0]: .6f}, {h[1, 1]: .6f}]]\n\n"
        f"grad L = [{gradient[0]: .6f}, {gradient[1]: .6f}]\n"
        f"n = [{n[0]: .6f}, {n[1]: .6f}]\n"
        f"lambda_n = n.T H n = {lam: .6f}\n\n"
        f"t = -(grad L dot n) / lambda_n\n"
        f"  = {taylor: .6f} pixels\n\n"
        f"Subpixel row = {refined[0]:.6f}\n"
        f"Subpixel col = {refined[1]:.6f}\n\n"
        "This point is BEFORE asymmetry bias correction\n"
        "and before the optional original-profile fit."
    )
    axes[1, 1].text(0.02, 0.98, info, va="top", family="monospace", fontsize=10)
    for ax in [axes[0, 1], axes[1, 0]]:
        ax.grid(alpha=0.15)
    fig.savefig(target, dpi=150, facecolor="white")
    plt.close(fig)
    return {
        "requested_point": np.asarray(requested).tolist(),
        "candidate_index": index,
        "pixel_row": row,
        "pixel_col": col,
        "hessian": h.tolist(),
        "gradient": gradient.tolist(),
        "normal": n.tolist(),
        "lambda_normal": lam,
        "taylor_t": taylor,
        "subpixel_point_before_bias_correction": refined.tolist(),
    }


def _arrays(result):
    trace = result.trace
    arrays = {
        "image": trace.image,
        "mask": trace.mask,
        "smoothed": trace.smoothed,
        "eigenvalues_algebraically_sorted": trace.eigenvalues,
        "hessian_normals": trace.hessian_normals,
        "response": trace.response,
        "taylor_t": trace.taylor_t,
        "taylor_offset": trace.taylor_offset,
        "initial_candidate_pixels": trace.initial_candidate_pixels,
        "initial_candidate_points": trace.initial_candidate_points,
        "candidate_pixels": trace.candidate_pixels,
        "candidate_points": trace.candidate_points,
        "candidate_normals": trace.candidate_normals,
        "candidate_response": trace.candidate_response,
        "newton_positions": np.stack(trace.newton_positions)
        if trace.newton_positions
        else np.empty((0, len(trace.initial_candidate_points), 2)),
    }
    arrays.update({"derivative_" + key: value for key, value in trace.derivatives.items()})
    for name, contours in [
        ("linked", trace.linked_contours),
        ("bias_corrected", trace.bias_corrected_contours),
        ("final", result.contours),
    ]:
        arrays[name + "_offsets"] = np.r_[0, np.cumsum([len(c.points) for c in contours])].astype(
            np.int64
        )
        for field in ("points", "normals", "response"):
            arrays[name + "_" + field] = (
                np.concatenate([getattr(c, field) for c in contours])
                if contours
                else np.empty((0, 2) if field != "response" else (0,))
            )
        for i, contour in enumerate(contours):
            arrays.update(
                {f"{name}_{i}_attribute_{key}": value for key, value in contour.attributes.items()}
            )
    return arrays


def save_process_report(result, directory, *, inspect_point=None, label="Input ROI"):
    """Export recorded arrays and process figures for any ROI/mask detection.

    Call lines_gauss(..., return_trace=True) first. No additional detection is
    performed. The mask controls display bounds but never changes saved arrays.
    """
    if result.trace is None:
        raise ValueError("No recorded process: call lines_gauss(..., return_trace=True)")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    _save_overview(result, output / "process-overview.png", label)
    _save_derivatives(result, output / "derivatives.png", label)
    _save_normals(result, output / "normals.png", label)
    point = _save_point(result, output / "point-inspection.png", inspect_point)
    if point is None and (output / "point-inspection.png").exists():
        (output / "point-inspection.png").unlink()
    np.savez_compressed(output / "process.npz", **_arrays(result))
    metadata = {
        "schema_version": 1,
        "label": label,
        "parameters": result.parameters,
        "diagnostics": result.diagnostics,
        "normal_definition": "unit eigenvector of maximum-absolute Hessian eigenvalue; row,col components",
        "normal_sign": "raw eigenvectors have +/- ambiguity; linked contour normals point to the right of traversal",
        "stages": {
            "initial_candidates": len(result.trace.initial_candidate_points),
            "located_candidates": len(result.trace.candidate_points),
            "linked_contours_before_completion": len(result.trace.linked_contours),
            "final_contours": len(result.contours),
        },
        "point_inspection": point,
        "raw_arrays": "process.npz",
    }
    write_json(output / "process.json", metadata)
    links = [
        "# Recorded Steger process",
        "",
        label,
        "",
        "Figures and arrays below come from the same detector invocation. Fixed-length cyan axes show normals; amber axes show tangents.",
        "",
        "![Recorded stages](process-overview.png)",
        "",
        "![Derivatives](derivatives.png)",
        "",
        "![Normal and tangent axes](normals.png)",
    ]
    if point is not None:
        links += ["", "![Inspect one actual candidate](point-inspection.png)"]
    links += [
        "",
        "[Raw float64 arrays](process.npz) | [Parameters and point values](process.json)",
        "",
        "Normals become ambiguous when the two Hessian eigenvalues have similar absolute magnitudes. A 90-degree rotation of a displayed normal gives its perpendicular tangent axis; it is not a separate curve fit.",
        "",
    ]
    (output / "README.md").write_text("\n".join(links), encoding="utf-8")
    return metadata
