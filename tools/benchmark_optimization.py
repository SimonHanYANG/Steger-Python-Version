"""Compare the current ROI/mask worker against immutable v1.0 in one environment.

Real images stay local. The reference package is loaded under an isolated name
from Git objects, never from the working tree. Timers run inside one persistent
worker and include all computation from raw arrays, excluding I/O and queue wait.
Run with the dev extra, which supplies reference-report imports and plotting.
"""

import argparse
import cProfile
import dataclasses
import hashlib
import importlib.util
import json
import platform
import pstats
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import analyze_necks as current_batch
import numpy as np
import scipy
from benchmark_roi_worker import compute_from_arrays
from PIL import Image

from steger import lines_gauss, measure_bending
from steger.io import read_image, write_json
from steger.synthetic import circular_line, straight_line

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "93a481e62f25b853d6dd8c40b62e910100facec6"


def source_sha256():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/steger").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def reference_modules():
    """A separate package prevents either version's relative imports leaking."""
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{REFERENCE_COMMIT}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "src/steger"], cwd=ROOT, text=True
    ).splitlines()
    alias = "_steger_v1_reference"
    with tempfile.TemporaryDirectory(prefix="steger-v1-reference-") as temp:
        root = Path(temp)
        for name in names + ["tools/analyze_necks.py", "tools/benchmark_roi_worker.py"]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            if name.startswith("tools/"):
                source = source.replace(b"from steger", f"from {alias}".encode())
                source = source.replace(
                    b"import analyze_necks as batch", b"from . import analyze_necks as batch"
                )
            target.write_bytes(source)

        def load(name, path, package=False):
            spec = importlib.util.spec_from_file_location(
                name, path, submodule_search_locations=[str(path.parent)] if package else None
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        package = load(alias, root / "src/steger/__init__.py", True)
        batch = load(alias + ".analyze_necks", root / "tools/analyze_necks.py")
        worker = load(alias + ".worker", root / "tools/benchmark_roi_worker.py")
        try:
            yield package, batch, worker, commit
        finally:
            for name in list(sys.modules):
                if name == alias or name.startswith(alias + "."):
                    del sys.modules[name]


def assert_exact(a, b, path="output"):
    """Reject numerical differences, including signed-zero and array bit changes."""
    if dataclasses.is_dataclass(a):
        assert [f.name for f in dataclasses.fields(a)] == [f.name for f in dataclasses.fields(b)], (
            path
        )
        for f in dataclasses.fields(a):
            assert_exact(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype, path
        assert a.tobytes() == b.tobytes(), f"{path}: array bits changed"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys changed"
        for key in a:
            assert_exact(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_exact(x, y, f"{path}[{i}]")
    elif isinstance(a, (float, np.floating)):
        assert struct.pack("d", a) == struct.pack("d", b), f"{path}: {a!r} != {b!r}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def search(batch, image, mask):
    geometry = batch.mask_geometry(mask)
    candidates = batch.candidate_runs(image, geometry)
    if not any(c["span_coverage"] >= 0.75 and c["length_px"] >= 15 for c in candidates):
        candidates.extend(batch.candidate_runs(image, geometry, batch.RESCUE_SCALES))
    return geometry, candidates, batch.choose_candidate(candidates)


def real_parity(batch, reference_worker, cases):
    candidate_count = trace_count = 0
    for identifier, roi, mask, image, binary in cases:
        expected = reference_worker.compute_from_arrays(roi, mask)
        assert_exact(expected, compute_from_arrays(roi, mask), identifier + ".worker")
        rg, rc, rs = search(batch, image, binary)
        cg, cc, cs = search(current_batch, image, binary)
        assert_exact(rg, cg, identifier + ".geometry")
        assert_exact(rc, cc, identifier + ".candidates")
        assert_exact(rs, cs, identifier + ".selected")
        # Inspect actual candidates used by the fast path, not just fresh reruns.
        detections = {}
        for old, new in zip(rc, cc, strict=True):
            key = old["sigma"], old["polarity"]
            if key not in detections:
                detections[key] = batch.detect(image, rg["mask"], *key)
            original = detections[key].contours[old["contour_index"]]
            actual = getattr(new, "contour", None)
            if actual is None:
                actual = current_batch.detect(image, cg["mask"], *key).contours[
                    new["contour_index"]
                ]
            assert_exact(original, actual, identifier + ".contour")
            candidate_count += 1
        if rs is not None:
            key = rs["sigma"], rs["polarity"]
            old = batch.detect(image, rg["mask"], *key, trace=True)
            new = current_batch.detect(image, cg["mask"], *key, trace=True)
            assert_exact(old, new, identifier + ".trace")
            actual = getattr(cs, "contour", new.contours[cs["contour_index"]])
            assert_exact(
                batch.assess(old.contours[rs["contour_index"]], rs, rc),
                current_batch.assess(actual, cs, cc),
                identifier + ".assessment",
            )
            trace_count += 1
    return dict(images=len(cases), candidate_contours=candidate_count, full_traces=trace_count)


def synthetic_parity(reference):
    count = 0
    for model in ("bar-shaped", "gaussian", "parabolic"):
        cases = [
            dict(model=model, half_width=w, angle_deg=a, asymmetry=s, offset=0.237, quadrature=32)
            for w in (1.0, 1.5, 2.5)
            for a in (0, 23, 61, 90)
            for s in (0, 0.5)
        ]
        cases += [
            dict(
                model=model,
                half_width=1.5,
                angle_deg=37,
                offset=-0.319,
                asymmetry=0.35,
                noise=n,
                polarity=p,
                quantize=True,
                seed=20261104,
                quadrature=32,
            )
            for n in (1, 3)
            for p in ("light", "dark")
        ]
        for case in cases:
            image, _ = straight_line(shape=(72, 72), **case)
            sigma = max(1.1, case["half_width"] / np.sqrt(3) + 0.4)
            args = (image, sigma, 0.4, 1.2, case.get("polarity", "light"), True, model, False)
            assert_exact(reference.lines_gauss(*args), lines_gauss(*args), f"synthetic[{count}]")
            count += 1
            if count % 12 == 0:
                print(f"Synthetic bitwise comparisons passed: {count}/84", flush=True)
    image, gt = circular_line(shape=(112, 112), radius=30)
    rr, cc = np.indices(image.shape)
    center = gt["center"]
    theta = np.arctan2(rr - center[0], cc - center[1])
    mask = (theta > 0.2) & (theta < 1.8) & (abs(np.hypot(rr - center[0], cc - center[1]) - 30) < 3)
    args = (image, 1.5, 0.4, 1.2, "light", True, "gaussian", False)
    old, new = reference.lines_gauss(*args, mask=mask), lines_gauss(*args, mask=mask)
    assert_exact(old, new, "circle")
    assert_exact(
        reference.measure_bending(max(old.contours, key=lambda c: c.length)),
        measure_bending(max(new.contours, key=lambda c: c.length)),
        "circle.angle",
    )
    return dict(straight_cases=count, circles=1)


def stats(values):
    a = np.asarray(values)
    return dict(
        calls=len(a),
        mean_ms=float(a.mean()),
        median_ms=float(np.median(a)),
        p95_ms=float(np.percentile(a, 95)),
        max_ms=float(a.max()),
    )


def timed(function, roi, mask):
    start = time.perf_counter_ns()
    output = function(roi, mask)
    return output, (time.perf_counter_ns() - start) / 1e6


def profile(function, cases):
    profiler = cProfile.Profile()
    profiler.enable()
    for _, roi, mask, *_ in cases:
        function(roi, mask)
    profiler.disable()
    rows = []
    for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(
        profiler
    ).stats.items():
        rows.append(
            dict(
                function=f"{Path(filename).name}:{line}:{name}",
                calls=calls,
                own_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    return sorted(rows, key=lambda r: -r["cumulative_seconds"])[:25]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("neck_mask"))
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("ROI", "MASK"))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cold-only", choices=("v1.0", "optimized"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/neck-analysis/performance/optimization.json")
    )
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run without -O: exact parity assertions are required")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    source_at_start = source_sha256()
    paths = current_batch.image_pairs(args.input_dir) if args.input_dir else []
    paths += [(Path(a), Path(b)) for a, b in args.pair]
    cases = []
    digest = hashlib.sha256()
    for a, b in paths:
        roi, mask = np.array(Image.open(a)), np.array(Image.open(b))
        for array in (roi, mask):
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        cases.append((a.stem, roi, mask, read_image(a), read_image(b, mask=True)))
    with reference_modules() as (reference, batch, worker, commit):
        if args.cold_only:
            function = (
                worker.compute_from_arrays if args.cold_only == "v1.0" else compute_from_arrays
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                output, elapsed = executor.submit(
                    timed, function, cases[0][1], cases[0][2]
                ).result()
            print(json.dumps({"processing_ms": elapsed, "output": output}))
            return
        print("Checking real outputs, every search candidate, and full traces...", flush=True)
        parity = real_parity(batch, worker, cases)
        if args.synthetic:
            print("Checking the 84 synthetic cases and circle against v1.0...", flush=True)
            parity.update(synthetic_parity(reference))
        print(f"Exact parity passed: {parity}", flush=True)
        functions = {"v1.0": worker.compute_from_arrays, "optimized": compute_from_arrays}
        samples = {name: [] for name in functions}
        per_image = {name: {case[0]: [] for case in cases} for name in functions}
        outputs = {}
        rng = np.random.default_rng(20261104)
        with ThreadPoolExecutor(max_workers=1) as executor:
            cold = {}
            for name, function in functions.items():
                output, elapsed = executor.submit(
                    timed, function, cases[0][1], cases[0][2]
                ).result()
                cold[name] = elapsed  # first worker call after parity; NOT process cold start
                for identifier, roi, mask, *_ in cases:
                    output, _ = executor.submit(timed, function, roi, mask).result()
                    outputs[name, identifier] = output
            for _ in range(args.rounds):
                for index in rng.permutation(len(cases)):
                    identifier, roi, mask, *_ = cases[index]
                    for name in rng.permutation(list(functions)):
                        output, elapsed = executor.submit(
                            timed, functions[name], roi, mask
                        ).result()
                        assert_exact(outputs[name, identifier], output, identifier + ".repeat")
                        samples[name].append(elapsed)
                        per_image[name][identifier].append(elapsed)
        if source_sha256() != source_at_start:
            raise RuntimeError("Source changed during validation; rerun on an unchanged tree")
        stages = {name: stats(values) for name, values in samples.items()}
        cold_process = {}
        for name in functions:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--input-dir",
                str(args.input_dir),
                "--cold-only",
                name,
            ]
            cold_process[name] = json.loads(subprocess.check_output(command, cwd=ROOT, text=True))
        report = dict(
            measured_utc=datetime.now(timezone.utc).isoformat(),
            reference_commit=commit,
            implementation_source_sha256=source_at_start,
            dataset_sha256=digest.hexdigest(),
            environment=dict(
                platform=platform.platform(),
                python=platform.python_version(),
                numpy=np.__version__,
                scipy=scipy.__version__,
            ),
            scope="Raw resident ROI/mask to angle+status; full adaptive search/rescue and quality checks; one persistent worker; interleaved randomized calls; no I/O, plots, queue wait or concurrent segmentation; warmed timings, not process startup",
            rounds=args.rounds,
            parity=parity,
            bitwise_exact=True,
            images_returning_angle=sum(
                outputs["optimized", c[0]]["angle_deg"] is not None for c in cases
            ),
            stages=stages,
            samples_ms=samples,
            speedup=stages["v1.0"]["mean_ms"] / stages["optimized"]["mean_ms"],
            first_worker_call_after_parity_ms=cold,
            first_call_fresh_process=cold_process,
            per_image={
                name: {key: {**stats(v), **outputs[name, key]} for key, v in group.items()}
                for name, group in per_image.items()
            },
            profiles={name: profile(f, cases) for name, f in functions.items()},
        )
        write_json(args.output, report)
        print(json.dumps({"stages": stages, "speedup": report["speedup"]}, indent=2), flush=True)
        print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
