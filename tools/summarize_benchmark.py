"""Generate readable evidence and a static scientific chart from the raw report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", default="reports/accuracy.json")
    args = parser.parse_args()
    path = Path(args.report)
    report = json.loads(path.read_text())
    lines = [
        "# Synthetic accuracy validation",
        "",
        f"Passed: **{report['passed_cases']}/{report['total_cases']}** straight-line cases; "
        f"circular neck: **{report['circle']['passed']}**.",
        "",
        "These are independent geometric ground-truth tests on specified synthetic profiles. "
        "They are not tests on real sperm images, and they are not HALCON measurements.",
        "",
        "| Noise standard deviation | Cases | Worst case P95 centerline error (px) | "
        "Worst case P95 half-width error (px) |",
        "|---:|---:|---:|---:|",
    ]
    for noise in [0, 1, 3]:
        group = [c for c in report["cases"] if c["case"].get("noise", 0) == noise]
        position = max(c["position_px"]["p95"] for c in group)
        width = max(c["half_width_error_px"]["p95"] for c in group)
        lines.append(f"| {noise} | {len(group)} | {position:.8g} | {width:.8g} |")
    circle = report["circle"]
    lines += [
        "",
        "Contrast is 150 gray levels. Noisy cases include 8-bit rounding. "
        "Noiseless cases intentionally match the assumed profile model; "
        "near-zero errors demonstrate numerical consistency, not real-image metrology accuracy.",
        "",
        f"Circular neck: expected window direction difference {circle['expected_angle_deg']:.6f} degrees; "
        f"measured {circle['bending_angle_deg']:.6f} degrees; "
        f"absolute difference **{circle['angle_error_deg']:.6f} degrees**.",
        "",
        f"Source hash: `{report.get('implementation_source_sha256', 'not recorded')}`.",
        "",
        "The raw report retains every case, including pre-refinement errors, width errors, "
        "fit acceptance coverage, parameters, and execution time.",
        "",
        "![Noise validation](noise-validation.png)",
        "",
        "![Circular-neck centerline overlay](examples/circular-neck.png)",
    ]
    (path.parent / "accuracy-summary.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = ["#2563eb", "#059669", "#d97706"]
    models = ["bar-shaped", "gaussian", "parabolic"]
    for axis, metric, label in zip(
        axes,
        ["position_px", "half_width_error_px"],
        ["Centerline error", "Half-width error"],
        strict=True,
    ):
        for k, model in enumerate(models):
            values = []
            for noise in [1, 3]:
                values.append(
                    max(
                        c[metric]["p95"]
                        for c in report["cases"]
                        if c["case"]["model"] == model and c["case"].get("noise", 0) == noise
                    )
                )
            axis.bar(np.arange(2) + (k - 1) * 0.24, values, 0.22, color=colors[k], label=model)
        axis.set(
            xticks=[0, 1],
            xticklabels=["1 gray level", "3 gray levels"],
            xlabel="Gaussian noise standard deviation",
            ylabel="Worst case P95 error (pixels)",
            title=label,
            ylim=(0, 0.05),
        )
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Synthetic pixel-integrated profiles: contrast 150, 8-bit quantization")
    fig.savefig(path.parent / "noise-validation.png", dpi=180)
    plt.close(fig)
    print(path.parent / "accuracy-summary.md")


if __name__ == "__main__":
    main()
