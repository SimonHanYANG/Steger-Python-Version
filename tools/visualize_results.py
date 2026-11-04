"""Render measured results from synthetic data, without changing the detector.

Run: uv run --extra dev python tools/visualize_results.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

from steger import lines_gauss
from steger.io import read_image, write_json
from steger.synthetic import straight_line


def render_bending(root, output):
    image = read_image(root / "reports/examples/circular-neck.npy")
    mask = read_image(root / "reports/examples/circular-neck-mask.png", mask=True)
    stored = json.loads((root / "reports/examples/circular-neck-result.json").read_text())
    curve = max(stored["contours"], key=lambda c: len(c["row"]))
    points = np.column_stack([curve["row"], curve["col"]])
    measurement = stored["neck_measurement"]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    centers = []
    for key in ("start_window", "end_window"):
        lo, hi = measurement[key]
        q = np.linspace(lo, hi, max(5, int(np.ceil((hi - lo) / 0.25)) + 1))
        centers.append(
            np.column_stack([np.interp(q, s, points[:, k]) for k in range(2)]).mean(axis=0)
        )
    directions = [np.asarray(measurement[k]) for k in ("direction_start", "direction_end")]
    matrix = np.column_stack([directions[0], -directions[1]])
    coefficients = np.linalg.solve(matrix, centers[1] - centers[0])
    intersection = centers[0] + coefficients[0] * directions[0]
    bounds = np.vstack([points, intersection])
    lower = np.maximum(0, bounds.min(axis=0) - 9)
    upper = np.minimum(np.asarray(image.shape) - 1, bounds.max(axis=0) + 9)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.2))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.77, bottom=0.16, wspace=0.08)
    fig.suptitle("Neck bending measurement", fontsize=20, y=0.97, fontweight="bold")
    fig.text(
        0.5,
        0.865,
        "Synthetic circular arc with known geometry - not a real sperm image",
        ha="center",
        color="#526071",
        fontsize=11,
    )
    for axis, title in zip(
        axes,
        [
            "1  Original grayscale ROI",
            "2  Segmentation mask",
            "3  Centerline and fitted directions",
        ],
        strict=True,
    ):
        axis.set_title(title, fontsize=12, pad=12)
        axis.set_xlim(lower[1], upper[1])
        axis.set_ylim(upper[0], lower[0])
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].imshow(image, cmap="gray", vmin=30, vmax=180, interpolation="nearest")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2].imshow(image, cmap="gray", vmin=30, vmax=180, interpolation="nearest")
    axes[2].plot(points[:, 1], points[:, 0], color="#38ff8b", lw=1.8, label="Detected centerline")
    colors = ["#29cfff", "#ffbf36"]
    for center, direction, color, label in zip(
        centers, directions, colors, ["Start fit", "End fit"], strict=True
    ):
        segment = center + np.array([-7.0, 7.0])[:, None] * direction
        axes[2].plot(segment[:, 1], segment[:, 0], color=color, lw=2, label=label)
        axes[2].plot(
            [center[1], intersection[1]],
            [center[0], intersection[0]],
            color=color,
            lw=1,
            ls="--",
            alpha=0.75,
        )
    angles = np.rad2deg([np.arctan2(v[0], v[1]) for v in directions])
    angles = np.rad2deg(np.unwrap(np.deg2rad(angles)))
    a, b = sorted(angles)
    axes[2].add_patch(Arc(intersection[::-1], 12, 12, theta1=a, theta2=b, color="white", lw=1.3))
    axes[2].legend(loc="upper left", fontsize=8, framealpha=0.9)
    value = measurement["angle_deg"]
    axes[2].text(
        0.97,
        0.04,
        f"{value:.4f} deg",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=15,
        color="white",
        fontweight="bold",
        bbox={"facecolor": "#142334", "edgecolor": "none", "alpha": 0.9, "pad": 5},
    )
    fig.text(
        0.5,
        0.055,
        f"Known window direction difference: 63.0000 deg   |   Measured: {value:.4f} deg\n"
        "Cyan and amber show the two measured directions; the green line is extracted from the image.",
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.7,
    )
    fig.savefig(output / "neck-bending-demo.png", dpi=170, facecolor="white")
    plt.close(fig)
    return {
        "measured_angle_deg": value,
        "expected_angle_deg": 63.0,
        "absolute_angle_error_deg": abs(value - 63.0),
        "source": "stored CLI result",
    }


def render_noisy(root, output):
    report = json.loads((root / "reports/accuracy.json").read_text())
    case = next(
        c
        for c in report["cases"]
        if c["case"]["model"] == "gaussian"
        and c["case"].get("noise") == 3
        and c["case"].get("polarity") == "dark"
    )
    image, truth = straight_line(shape=(72, 72), **case["case"])
    result = lines_gauss(image, case["sigma"], 0.4, 1.2, "dark", True, "gaussian", False)
    contour = max(result.contours, key=lambda c: c.length)
    points = contour.points
    valid = np.all((points > 14) & (points < 57), axis=1)
    error = (points[valid] - truth["center"]) @ truth["normal"]
    baseline = np.column_stack([contour.attributes["steger_row"], contour.attributes["steger_col"]])
    base_error = (baseline[valid] - truth["center"]) @ truth["normal"]
    tangent = np.array([-truth["normal"][1], truth["normal"][0]])
    distance = (points[valid] - truth["center"]) @ tangent
    order = np.argsort(distance)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.7))
    fig.subplots_adjust(left=0.035, right=0.98, top=0.77, bottom=0.18, wspace=0.30)
    fig.suptitle(
        "Centerline extraction with noise and asymmetric backgrounds",
        fontsize=17,
        y=0.98,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.875,
        "Synthetic dark Gaussian line | contrast 150 | noise SD 3 | 8-bit quantization",
        ha="center",
        fontsize=10.5,
        color="#526071",
    )
    for axis in axes[:2]:
        axis.imshow(image, cmap="gray", vmin=20, vmax=190, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].set_title("1  Original noisy image", fontsize=12)
    axes[1].set_title("2  Extracted centerline", fontsize=12)
    axes[1].plot(points[:, 1], points[:, 0], color="#38ff8b", lw=1.4)
    axes[2].set_title("3  Signed distance to true centerline", fontsize=11)
    axes[2].axhline(0, color="#334155", lw=1, label="Known true centerline")
    axes[2].plot(
        distance[order],
        base_error[order],
        color="#9ca3af",
        lw=1.3,
        label="Before extra profile fit",
    )
    axes[2].plot(distance[order], error[order], color="#059669", lw=1.6, label="Final result")
    axes[2].set(xlabel="Distance along line (px)", ylabel="Signed error (px)")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=7.5, loc="upper right")
    p95 = float(np.percentile(abs(error), 95))
    fig.text(
        0.5,
        0.055,
        f"Final P95 absolute position error: {p95:.4f} px. "
        "Error is measured against independent geometric ground truth, excluding image boundaries.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output / "noisy-line-demo.png", dpi=170, facecolor="white")
    plt.close(fig)
    return {
        "case": case["case"],
        "final_position_error_p95_px": p95,
        "before_extra_fit_p95_px": float(np.percentile(abs(base_error), 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/visualizations")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "metrics.json",
        {
            "data_type": "synthetic; no real sperm samples",
            "bending": render_bending(root, output),
            "noisy_line": render_noisy(root, output),
        },
    )
    print(output / "neck-bending-demo.png")
    print(output / "noisy-line-demo.png")


if __name__ == "__main__":
    main()
