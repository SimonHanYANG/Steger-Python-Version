"""Time a single angle worker whose only per-item inputs are ROI and mask.

Disk decoding is outside the timers; RGB conversion, mask processing, ridge
extraction, contour selection and angle calculation are inside the worker.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import analyze_necks as batch
import numpy as np
import scipy
from PIL import Image

from steger.io import write_json
from steger.measurement import measure_bending


def compute_from_arrays(roi, neck_mask, *, mode="adaptive"):
    """No saved contour/index/per-image parameter is accepted by this function."""
    image = np.asarray(roi)
    if image.ndim == 3 and image.shape[-1] >= 3:
        image = np.asarray(image[..., :3], dtype=np.float64) @ np.array([0.299, 0.587, 0.114])
    else:
        image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(neck_mask)
    if mask.ndim == 3:
        mask = np.any(mask[..., :3] != 0, axis=-1)
    else:
        mask = mask != 0
    if image.ndim != 2 or image.shape != mask.shape:
        raise ValueError("Expected aligned 2-D or RGB ROI and neck mask")
    geometry = batch.mask_geometry(mask)
    scales = batch.SCALES if mode == "adaptive" else (2.0,)
    candidates = batch.candidate_runs(image, geometry, scales)
    if mode == "adaptive" and not any(
        c["span_coverage"] >= 0.75 and c["length_px"] >= 15 for c in candidates
    ):
        candidates.extend(batch.candidate_runs(image, geometry, batch.RESCUE_SCALES))
    selected = batch.choose_candidate(candidates)
    if selected is None:
        return {"angle_deg": None, "status": "no_line"}
    # This final run reproduces the existing analysis path, with trace capture off.
    result = batch.detect(image, geometry["mask"], selected["sigma"], selected["polarity"])
    contour = result.contours[selected["contour_index"]]
    if mode == "adaptive":
        assessed = batch.assess(contour, selected, candidates)
        return {"angle_deg": assessed["angle_deg"], "status": assessed["status"]}
    if selected["span_coverage"] < 0.75:
        return {"angle_deg": None, "status": "insufficient_segment"}
    try:
        measurement = measure_bending(contour)
    except ValueError:
        return {"angle_deg": None, "status": "insufficient_segment"}
    return {"angle_deg": measurement.angle_deg, "status": "single_scale_estimate"}


def timed_worker(roi, mask, mode):
    start = time.perf_counter_ns()
    result = compute_from_arrays(roi, mask, mode=mode)
    stop = time.perf_counter_ns()
    return result, (stop - start) / 1e6


def statistics(samples):
    a = np.asarray(samples)
    return {
        "calls": int(a.size),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "max_ms": float(a.max()),
        "min_ms": float(a.min()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("reports/neck-analysis/results"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/neck-analysis/worker-runtime")
    )
    parser.add_argument("--adaptive-rounds", type=int, default=20)
    parser.add_argument("--fixed-rounds", type=int, default=30)
    args = parser.parse_args()
    if min(args.adaptive_rounds, args.fixed_rounds) < 1:
        parser.error("Repeat counts must be positive")
    reference = json.loads((args.results_dir / "results.json").read_text())["results"]
    cases = []
    for row in reference:
        for kind in ("image", "mask"):
            path = Path(row[kind + "_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[kind + "_sha256"]:
                raise ValueError(f"Changed input: {path}")
        # Materialize the raw uint8 RGB and raw 0/255 mask before starting timers.
        roi = np.array(Image.open(row["image_path"]))
        mask = np.array(Image.open(row["mask_path"]))
        cases.append((row, roi, mask))
    stages, per_image_rows = {}, []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="neck-angle") as worker:
        for mode, rounds in (
            ("adaptive", args.adaptive_rounds),
            ("fixed_sigma_2", args.fixed_rounds),
        ):
            outputs = {}
            # Warm the single persistent worker, and verify adaptive outputs off-timer.
            for row, roi, mask in cases:
                output, _ = worker.submit(timed_worker, roi, mask, mode).result()
                outputs[row["id"]] = output
                if mode == "adaptive":
                    assert output["status"] == row["status"], row["id"]
                    if row["angle_deg"] is None:
                        assert output["angle_deg"] is None, row["id"]
                    else:
                        np.testing.assert_allclose(
                            output["angle_deg"], row["angle_deg"], rtol=0, atol=1e-10
                        )
            samples = {row["id"]: [] for row, _, _ in cases}
            roundtrip = []
            rng = np.random.default_rng(20261104)
            for _ in range(rounds):
                for index in rng.permutation(len(cases)):
                    row, roi, mask = cases[index]
                    start = time.perf_counter_ns()
                    output, elapsed = worker.submit(timed_worker, roi, mask, mode).result()
                    stop = time.perf_counter_ns()
                    samples[row["id"]].append(elapsed)
                    roundtrip.append((stop - start) / 1e6)
                    assert output == outputs[row["id"]], row["id"]
            flat = [value for group in samples.values() for value in group]
            successful_samples = [
                value
                for identifier, values in samples.items()
                if outputs[identifier]["angle_deg"] is not None
                for value in values
            ]
            differences = [
                abs(outputs[row["id"]]["angle_deg"] - row["angle_deg"])
                for row, _, _ in cases
                if outputs[row["id"]]["angle_deg"] is not None and row["angle_deg"] is not None
            ]
            stage = {
                **statistics(flat),
                "rounds": rounds,
                "image_count": len(cases),
                "images_returning_angle": sum(o["angle_deg"] is not None for o in outputs.values()),
                "successful_outputs_only": statistics(successful_samples)
                if successful_samples
                else None,
                "submit_to_result_including_handoff": statistics(roundtrip),
                "maximum_angle_difference_from_adaptive_deg": max(differences, default=None),
                "images_differing_over_5deg_from_adaptive": sum(v > 5 for v in differences),
                "per_image": {
                    key: {**statistics(values), **outputs[key]} for key, values in samples.items()
                },
            }
            stages[mode] = stage
            for identifier, values in stage["per_image"].items():
                per_image_rows.append({"mode": mode, "id": identifier, **values})
            print(
                f"{mode}: {json.dumps({k: v for k, v in stage.items() if k != 'per_image'})}",
                flush=True,
            )
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or platform.machine()
    report = {
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "cpu": cpu,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "scope": {
            "inputs": "resident raw RGB uint8 ROI and raw uint8 mask; 28 supplied 64x64 pairs",
            "worker": "one persistent ThreadPoolExecutor worker; no concurrent segmentation workload",
            "includes": "RGB conversion, mask normalization/component analysis, Steger derivatives/Hessian/subpixel extraction/linking, contour selection, end-window fits, angle, result construction; adaptive mode also includes full search/rescue and quality/sensitivity checks",
            "excludes": "segmentation, disk reads/decoding, visualization, trace capture, queue wait for processing time; submit-to-result latency separately reported",
            "adaptive": "same decisions and angles as previous batch; all per-item choices derived from only ROI/mask",
            "fixed_sigma_2": "global sigma=2, detects both polarities, selects contour from inputs; no cached per-image parameter or contour index; does not promise adaptive measurement behavior",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stages": stages,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "worker-runtime.json", report)
    with (args.output_dir / "worker-runtime-per-image.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
