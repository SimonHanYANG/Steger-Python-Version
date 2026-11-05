"""Benchmark resident-data neck computation; all I/O stays outside timers.

Run from the repository using:
uv run --extra visualization python tools/benchmark_neck_runtime.py
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import analyze_necks as batch
import numpy as np
import scipy

from steger.io import read_image, write_json
from steger.measurement import measure_bending
from steger.types import Contour


def load_cases(directory):
    records = json.loads((directory / "results.json").read_text())["results"]
    cases = []
    for record in records:
        saved = json.loads((directory / record["id"] / "result.json").read_text())
        raw = saved["detector"]["contours"][record["contour_index"]]
        contour = Contour(
            np.column_stack([raw["row"], raw["col"]]),
            np.column_stack([raw["normal_row"], raw["normal_col"]]),
            np.asarray(raw["response"], dtype=np.float64),
            {key: np.asarray(value) for key, value in raw["attributes"].items()},
            raw["closed"],
        )
        image = read_image(record["image_path"])
        mask = read_image(record["mask_path"], mask=True)
        # Input identity is checked before timing, along with all decoding/conversion.
        for key in ("image", "mask"):
            digest = hashlib.sha256(Path(record[key + "_path"]).read_bytes()).hexdigest()
            if digest != record[key + "_sha256"]:
                raise ValueError(f"Input changed since analysis: {record['id']} {key}")
        cases.append({"record": record, "contour": contour, "image": image, "mask": mask})
    return cases


def angle_only(case):
    # The actual public function, including fits, angle, RMS, cumulative turn,
    # checks and result construction. This is not an isolated arctan benchmark.
    return measure_bending(case["contour"])


def fixed_parameters(case):
    record = case["record"]
    geometry = batch.mask_geometry(case["mask"])
    result = batch.detect(case["image"], geometry["mask"], record["sigma"], record["polarity"])
    # Previously established parameters and selected contour index are reused;
    # this stage does not search parameters or repeat sensitivity analysis.
    return measure_bending(result.contours[record["contour_index"]])


def automatic_search(case):
    geometry = batch.mask_geometry(case["mask"])
    candidates = batch.candidate_runs(case["image"], geometry)
    if not any(c["span_coverage"] >= 0.75 and c["length_px"] >= 15 for c in candidates):
        candidates.extend(batch.candidate_runs(case["image"], geometry, batch.RESCUE_SCALES))
    selected = batch.choose_candidate(candidates)
    if selected is None:
        return {"status": "no_line", "angle_deg": None}
    return batch.assess(selected.contour, selected, candidates)


def statistics(milliseconds):
    a = np.asarray(milliseconds, dtype=np.float64)
    return {
        "calls": int(a.size),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def time_stage(name, cases, function, *, rounds, inner, warmup):
    for case in cases:
        for _ in range(warmup):
            function(case)
    rng = np.random.default_rng(20261104)
    timings = {case["record"]["id"]: [] for case in cases}
    # Shuffle cases between rounds; equal work per case keeps the pooled mean
    # equally weighted across images. GC remains in its normal runtime state.
    for _ in range(rounds):
        for index in rng.permutation(len(cases)):
            case = cases[index]
            samples = timings[case["record"]["id"]]
            for _ in range(inner):
                start = time.perf_counter_ns()
                result = function(case)
                stop = time.perf_counter_ns()
                samples.append((stop - start) / 1e6)
            del result
    combined = [sample for samples in timings.values() for sample in samples]
    report = {
        **statistics(combined),
        "image_count": len(cases),
        "rounds": rounds,
        "calls_per_case_per_round": inner,
        "warmup_calls_per_case": warmup,
        "per_image": {key: statistics(value) for key, value in timings.items()},
    }
    print(
        f"{name}: mean={report['mean_ms']:.6f} ms; median={report['median_ms']:.6f} ms; "
        f"P95={report['p95_ms']:.6f} ms; calls={report['calls']}",
        flush=True,
    )
    return report, timings


def verify(cases):
    successful = [c for c in cases if c["record"]["angle_deg"] is not None]
    for case in successful:
        expected = case["record"]["angle_deg"]
        for function in (angle_only, fixed_parameters):
            if not np.isclose(function(case).angle_deg, expected, rtol=0, atol=1e-10):
                raise ValueError(f"Angle mismatch: {case['record']['id']} {function.__name__}")
    for case in cases:
        expected = case["record"]
        actual = automatic_search(case)
        if actual["status"] != expected["status"]:
            raise ValueError(f"Status mismatch: {expected['id']}")
        if expected["angle_deg"] is None:
            if actual["angle_deg"] is not None:
                raise ValueError(f"Unexpected full-neck angle: {expected['id']}")
        elif not np.isclose(actual["angle_deg"], expected["angle_deg"], rtol=0, atol=1e-10):
            raise ValueError(f"Automatic angle mismatch: {expected['id']}")
    print(
        f"Verified saved results: {len(successful)} angles and all {len(cases)} statuses.",
        flush=True,
    )
    return successful


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("reports/neck-analysis/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/neck-analysis/runtime"))
    parser.add_argument("--angle-rounds", type=int, default=10)
    parser.add_argument("--angle-inner", type=int, default=100)
    parser.add_argument("--fixed-rounds", type=int, default=30)
    parser.add_argument("--auto-rounds", type=int, default=5)
    args = parser.parse_args()
    if min(args.angle_rounds, args.angle_inner, args.fixed_rounds, args.auto_rounds) < 1:
        parser.error("Repeat counts must be positive")
    cases = load_cases(args.results_dir)
    successful = verify(cases)
    if not successful:
        raise ValueError("No completed full-neck angles to benchmark")
    stages, raw = {}, {}
    for name, subset, function, rounds, inner, warmup in (
        ("angle_only", successful, angle_only, args.angle_rounds, args.angle_inner, 20),
        ("fixed_parameter_roi_to_angle", successful, fixed_parameters, args.fixed_rounds, 1, 2),
        ("automatic_search_and_quality", cases, automatic_search, args.auto_rounds, 1, 1),
    ):
        stages[name], raw[name] = time_stage(
            name, subset, function, rounds=rounds, inner=inner, warmup=warmup
        )
    auto_success = [
        sample
        for case in successful
        for sample in raw["automatic_search_and_quality"][case["record"]["id"]]
    ]
    stages["automatic_search_and_quality"]["successful_images_only"] = statistics(auto_success)
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or platform.machine()
    report = {
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cpu": cpu,
            "logical_cpu_count": os.cpu_count(),
            "architecture": platform.machine(),
            "os": platform.system() + " " + platform.release(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "gc_enabled": gc.isenabled(),
            "thread_settings": {
                name: os.environ.get(name, "not set")
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "timing": {
            "clock": "time.perf_counter_ns; wall-clock duration per invocation",
            "execution": "sequential; warmed; no worker parallelism; default thread settings unchanged",
            "excludes": "file reading/writing, PNG decoding, RGB-to-gray conversion, imports, trace capture, plots, reports",
            "angle_only": "existing measure_bending on resident selected contours: arclength resampling, two orthogonal fits, angle, fit RMS, cumulative turn and result construction",
            "fixed_parameter_roi_to_angle": "resident grayscale ROI/mask -> component selection -> one Steger run with recorded parameters/index -> measure_bending; no parameter search or sensitivity assessment",
            "automatic_search_and_quality": "resident grayscale ROI/mask -> actual multi-scale/multi-polarity search and optional rescue -> selected Steger run -> angle and quality/sensitivity checks; no trace capture",
            "failed_cases": "3 diagnostic-only images excluded from first two stages; automatic stage includes their rejection work, and reports a successful-images-only subset",
            "percentiles": "percentiles of individually timed calls, including OS scheduling effects; timer overhead not subtracted",
        },
        "verified": {
            "input_pairs": len(cases),
            "completed_angles": len(successful),
            "saved_outputs_match": True,
        },
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stages": stages,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "runtime.json", report)
    with (args.output_dir / "runtime-per-image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["stage", "id", "calls", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"]
        )
        for name, stage in stages.items():
            for identifier, values in stage["per_image"].items():
                writer.writerow(
                    [
                        name,
                        identifier,
                        *[
                            values[k]
                            for k in ("calls", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms")
                        ],
                    ]
                )
    np.savez_compressed(
        args.output_dir / "runtime-samples.npz",
        **{
            f"{stage}__{identifier}": np.asarray(samples)
            for stage, items in raw.items()
            for identifier, samples in items.items()
        },
    )
    lines = [
        "# Neck computation runtime",
        "",
        f"Machine: {cpu}; Python {platform.python_version()}; NumPy {np.__version__}; SciPy {scipy.__version__}.",
        "",
        "All inputs and contours were loaded before timing. Results were checked against the saved analysis before benchmarking. No file I/O, RGB conversion, plots, trace exports, or report generation are timed. Sequential warmed calls; normal garbage collection and thread settings remain enabled.",
        "",
        "| Stage | Images | Calls | Mean ms | Median ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stage in stages.items():
        lines.append(
            f"| {name} | {stage['image_count']} | {stage['calls']} | {stage['mean_ms']:.6f} | {stage['median_ms']:.6f} | {stage['p95_ms']:.6f} |"
        )
    lines += [
        "",
        "The first two stages use the 25 completed full-neck measurements. The automatic stage uses all 28 images, including three diagnostic-only outcomes. Its successful-images-only statistics are also in runtime.json.",
        "",
        "Angle-only timing includes the complete existing measure_bending function, not just the final arctan. Fixed-parameter timing reuses the already selected parameters and contour index. Automatic timing repeats the actual parameter search and quality/sensitivity work.",
        "",
        "These are warmed wall-clock measurements on this machine and these 64x64 images, not a cross-platform speed guarantee. Per-image summaries are in runtime-per-image.csv; individual samples are in runtime-samples.npz.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
