"""Publish aggregate performance evidence without real images or per-image IDs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from steger.io import write_json


def aggregate(report):
    keys = (
        "measured_utc",
        "reference_commit",
        "implementation_source_sha256",
        "dataset_sha256",
        "environment",
        "scope",
        "rounds",
        "parity",
        "bitwise_exact",
        "images_returning_angle",
        "stages",
        "samples_ms",
        "speedup",
        "first_worker_call_after_parity_ms",
        "profiles",
    )
    public = {key: report[key] for key in keys if key in report}
    public["first_call_fresh_process"] = {
        name: {"processing_ms": item["processing_ms"]}
        for name, item in report["first_call_fresh_process"].items()
    }
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--repeat-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/performance"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    # Intentionally omit per-image identities, measurements and input paths.
    public = aggregate(report)
    repeat = None
    if args.repeat_report:
        repeat = json.loads(args.repeat_report.read_text())
        for key in (
            "reference_commit",
            "implementation_source_sha256",
            "dataset_sha256",
            "environment",
            "images_returning_angle",
        ):
            if report[key] != repeat[key]:
                raise ValueError(f"Repeat report differs in {key}")
        if not repeat["bitwise_exact"]:
            raise ValueError("Repeat must pass exact comparisons")
        public["repeat_run"] = aggregate(repeat)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "optimization.json", public)
    rows = []
    for name, values in report["stages"].items():
        rows.append(
            f"| {name} | {values['calls']} | {values['mean_ms']:.3f} | "
            f"{values['median_ms']:.3f} | {values['p95_ms']:.3f} | {values['max_ms']:.3f} |"
        )
    parity = report["parity"]
    lines = [
        "# ROI/mask worker optimization",
        "",
        f"Measured UTC: {report['measured_utc']}. Platform: {report['environment']['platform']}.",
        "",
        report["scope"] + ".",
        "",
        f"Both versions receive the same **{parity['images']} real ROI/mask pairs**; "
        f"**{report['images_returning_angle']}** return a full angle. Real images, "
        "per-image measurements and identities stay local.",
        "",
        "| Version | Calls | Mean (ms) | Median (ms) | P95 (ms) | Maximum (ms) |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"Mean speedup: **{report['speedup']:.2f}×**. These measurements describe this "
        "machine and dataset, not a latency guarantee for other images or concurrent workloads. "
        "Profiling was performed separately, outside the reported timing samples.",
        "",
        "Occasional long calls remain in both versions. The optimized maximum exceeds "
        "the reference maximum in this run; the cause of those individual stalls has "
        "not been isolated. Every sample is retained, without trimming or capping. "
        "This worker does not provide a hard real-time latency bound.",
        "",
        "![Runtime comparison](runtime-comparison.png)",
        "",
        "## Numerical equivalence",
        "",
        f"Reference commit: `{report['reference_commit']}` (accepted v1.0). "
        "The tool loads its source from immutable Git objects under a separate package name.",
        "",
        f"Bitwise comparisons passed for {parity['images']} real inputs, "
        f"{parity['candidate_contours']} actual search candidate contours and "
        f"{parity['full_traces']} complete intermediate traces. They include coordinates, "
        "normals, responses, selected scale/polarity, fit diagnostics, quality flags and statuses. "
        "No tolerance was used to hide changed outputs.",
        "",
        f"Synthetic full-detector comparisons: {parity.get('straight_cases', 'not run')} "
        f"straight cases; {parity.get('circles', 'not run')} circle. Ground-truth acceptance "
        "is recorded separately in [the accuracy report](../accuracy-summary.md). "
        "Equivalence to v1.0 on tested inputs is distinct from real-image ground-truth accuracy.",
        "",
        f"Source SHA-256: `{report['implementation_source_sha256']}`.",
        "",
        f"Input-array dataset SHA-256: `{report['dataset_sha256']}`.",
        "",
        "## What changed",
        "",
        "- Share full-ROI Gaussian derivatives, Hessian results, Taylor offsets and lazy "
        "cubic spline coefficients between dark and light detection at each scale.",
        "- Solve independent 2×2 Hessian eigenproblems only inside the candidate mask "
        "when trace capture is off. Convolution and interpolation still use the entire "
        "original ROI; trace mode retains the full original fields.",
        "- Retain the search contour instead of running the winning detector again. "
        "Reuse identical angle fits while retaining length validation and sensitivity checks.",
        "",
        "Float64, all five search scales, all three rescue scales, both polarities, "
        "Newton iteration rules, linking, selection rules, end windows and quality gates "
        "are preserved. No new runtime dependency or compiled extension was introduced.",
        "",
        "## First-call check",
        "",
        "Each value below is one first computation in a separate fresh process on the "
        "first dataset input. Imports, file decoding and thread startup are outside the "
        "timer. These single samples are not steady-state means.",
        "",
        *[
            f"- {name}: {item['processing_ms']:.3f} ms."
            for name, item in report["first_call_fresh_process"].items()
        ],
        "",
        "The raw [aggregate JSON](optimization.json) retains environment details and "
        "separate profiler summaries. See [reproduction and API notes](../../docs/performance.md).",
    ]
    if repeat is not None:
        lines += [
            "",
            "## Independent repeat",
            "",
            "A fresh process repeated the same real-input parity checks and timing after "
            "the long outliers above were observed. Source, dataset, environment and returned-angle "
            "count match the primary run. Synthetic comparisons are recorded in the primary run. "
            "The repeat does not replace or remove primary-run samples.",
            "",
            "| Version | Calls | Mean (ms) | Median (ms) | P95 (ms) | Maximum (ms) |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {name} | {v['calls']} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | "
                f"{v['p95_ms']:.3f} | {v['max_ms']:.3f} |"
                for name, v in repeat["stages"].items()
            ],
            "",
            f"Repeat mean speedup: **{repeat['speedup']:.2f}×**.",
        ]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    keys = ("mean_ms", "median_ms", "p95_ms")
    for offset, (name, values), color in zip(
        (-0.18, 0.18), report["stages"].items(), ("#64748b", "#059669"), strict=True
    ):
        bars = ax.bar(
            [x + offset for x in range(3)],
            [values[k] for k in keys],
            width=0.34,
            label=name,
            color=color,
        )
        ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set(
        xticks=range(3),
        xticklabels=("Mean", "Median", "P95"),
        ylabel="Milliseconds / ROI",
        title=f"Full adaptive ROI/mask worker · {parity['images']} real inputs",
    )
    ax.set_ylim(0, max(v[k] for v in report["stages"].values() for k in keys) * 1.2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / "runtime-comparison.png", dpi=180)
    plt.close(fig)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
