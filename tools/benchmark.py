"""Deterministic independent synthetic accuracy suite. Never calls HALCON.

Run: uv run python tools/benchmark.py --output reports/accuracy.json
The clean-case gates are predeclared below. Stress cases are also reported;
their looser gates never modify or hide clean-case acceptance thresholds.
"""

import argparse
import hashlib
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from steger import __version__, lines_gauss, measure_bending
from steger.io import overlay, write_json
from steger.synthetic import circular_line, straight_line


def statistics(errors):
    errors = np.abs(np.asarray(errors))
    return dict(
        mean=float(np.mean(errors)),
        p95=float(np.percentile(errors, 95)),
        maximum=float(np.max(errors)),
    )


def evaluate(case, index, output_dir):
    image, gt = straight_line(shape=(72, 72), **case)
    width = case.get("half_width", 1.5)
    sigma = max(1.1, width / np.sqrt(3) + 0.4)
    start = time.perf_counter()
    result = lines_gauss(
        image, sigma, 0.4, 1.2, case.get("polarity", "light"), True, case["model"], False
    )
    item = dict(
        case=case,
        sigma=sigma,
        seconds=time.perf_counter() - start,
        contours=len(result.contours),
        success=False,
    )
    if not result.contours:
        item["failure"] = "no contours"
        return item
    c = max(result.contours, key=lambda c: c.length)
    valid = np.all((c.points > 14) & (c.points < 57), axis=1)
    if np.count_nonzero(valid) < 10:
        item["failure"] = "insufficient interior points"
        return item
    position = (c.points[valid] - gt["center"]) @ gt["normal"]
    baseline = np.column_stack([c.attributes["steger_row"], c.attributes["steger_col"]])[valid]
    baseline_error = (baseline - gt["center"]) @ gt["normal"]
    fit_valid = c.attributes["profile_fit_valid"][valid]
    noise = case.get("noise", 0)
    # Explicit targets for this specified synthetic domain, not universal claims.
    pos_limit = 0.01 if noise == 0 else 0.08 if noise <= 1 else 0.20
    width_limit = 0.025 if noise == 0 else 0.12 if noise <= 1 else 0.30
    item.update(
        success=True,
        position_px=statistics(position),
        baseline_position_px=statistics(baseline_error),
        half_width_error_px=statistics(c.attributes["width_left"][valid] - width),
        refinement_coverage=float(np.mean(fit_valid)),
        samples=int(valid.sum()),
        acceptance={
            "position_p95_px": pos_limit,
            "half_width_p95_px": width_limit,
            "min_refinement_coverage": 0.95,
        },
    )
    item["passed"] = bool(
        item["position_px"]["p95"] <= pos_limit
        and item["half_width_error_px"]["p95"] <= width_limit
        and item["refinement_coverage"] >= 0.95
    )
    if index in (0, 9, 18) or not item["passed"]:
        np.save(output_dir / f"case-{index:03d}.npy", image)
        overlay(image, result, output_dir / f"case-{index:03d}.png")
    return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/accuracy.json")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    artifacts = target.parent / "examples"
    artifacts.mkdir(exist_ok=True)
    cases = []
    for model in ["bar-shaped", "gaussian", "parabolic"]:
        for width in [1.5] if a.quick else [1.0, 1.5, 2.5]:
            for angle in [0, 23, 61, 90]:
                for asymmetry in [0, 0.5]:
                    cases.append(
                        dict(
                            model=model,
                            half_width=width,
                            angle_deg=angle,
                            offset=0.237,
                            asymmetry=asymmetry,
                            quadrature=32,
                        )
                    )
        for noise in [1, 3]:
            for polarity in ["light", "dark"]:
                cases.append(
                    dict(
                        model=model,
                        half_width=1.5,
                        angle_deg=37,
                        offset=-0.319,
                        asymmetry=0.35,
                        noise=noise,
                        polarity=polarity,
                        quantize=True,
                        seed=20261104,
                        quadrature=32,
                    )
                )
    start = time.perf_counter()
    source_digest = hashlib.sha256()
    for source in sorted((Path(__file__).resolve().parents[1] / "src" / "steger").glob("*.py")):
        source_digest.update(source.name.encode())
        source_digest.update(source.read_bytes())
    results = []
    for index, case in enumerate(cases):
        item = evaluate(case, index, artifacts)
        results.append(item)
        print(
            f"{index + 1}/{len(cases)} {case['model']} w={case['half_width']} angle={case['angle_deg']} "
            f"noise={case.get('noise', 0)} p95={item.get('position_px', {}).get('p95')} "
            f"passed={item.get('passed', False)}",
            flush=True,
        )
    image, gt = circular_line(shape=(112, 112), radius=30)
    rr, cc = np.indices(image.shape)
    center = gt["center"]
    theta = np.arctan2(rr - center[0], cc - center[1])
    mask = (theta > 0.2) & (theta < 1.8) & (abs(np.hypot(rr - center[0], cc - center[1]) - 30) < 3)
    detected = lines_gauss(image, 1.5, 0.4, 1.2, "light", True, "gaussian", False, mask=mask)
    c = max(detected.contours, key=lambda c: c.length)
    m = measure_bending(c)
    angles = np.unwrap(np.arctan2(c.rows - center[0], c.cols - center[1]))
    expected = abs(np.rad2deg(angles[-1] - angles[0])) * 0.7
    circle = {
        "radius_error_px": statistics(np.linalg.norm(c.points - center, axis=1) - 30),
        "bending_angle_deg": m.angle_deg,
        "expected_angle_deg": expected,
        "angle_error_deg": abs(m.angle_deg - expected),
        "measurement": m.to_dict(),
    }
    circle["passed"] = bool(
        circle["radius_error_px"]["p95"] < 0.01 and circle["angle_error_deg"] < 0.1
    )
    overlay(image, detected, artifacts / "circular-neck.png")
    np.save(artifacts / "circular-neck.npy", image)
    Image.fromarray((mask * 255).astype(np.uint8)).save(artifacts / "circular-neck-mask.png")
    report = {
        "implementation_version": __version__,
        "implementation_source_sha256": source_digest.hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": 20261104,
        "halcon_comparison": "NOT RUN: native licensed reference unavailable",
        "truth": "continuous geometric centerline; independent pixel-area image integration",
        "scope": "specified synthetic profile models only; no real sperm ROI/mask provided",
        "cases": results,
        "circle": circle,
        "elapsed_seconds": time.perf_counter() - start,
        "passed_cases": sum(bool(x.get("passed")) for x in results),
        "total_cases": len(results),
    }
    report["all_passed"] = report["passed_cases"] == len(results) and circle["passed"]
    write_json(target, report)
    print(
        f"Report: {target}; passed {report['passed_cases']}/{len(results)}; circle={circle['passed']}",
        flush=True,
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
