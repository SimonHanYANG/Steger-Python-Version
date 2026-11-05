"""Paired raw ROI/mask latency and bitwise parity against a frozen Git baseline.

No file decoding, visualization or startup runs inside request timers. The
parallel mode DOES include shared-memory copies, IPC, synchronization and full
result construction. Each case/mode has equal weight. Samples are not trimmed.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy
from PIL import Image

import analyze_necks as batch
from benchmark_optimization import assert_exact, real_parity, reference_modules, synthetic_parity
from steger import detector, measure_neck
from steger.io import read_image, write_json
from steger.neck import candidate_runs
from steger.worker import NeckAngleWorker

ROOT = Path(__file__).resolve().parents[1]


def statistics(values):
    a = np.asarray(values)
    return dict(calls=len(a), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)),
                min_ms=float(a.min()), max_ms=float(a.max()),
                fraction_at_or_below_5ms=float(np.mean(a <= 5)))


def fingerprint():
    paths = sorted((ROOT / 'src/steger').glob('*.py'))
    paths += sorted((ROOT / 'src/steger').glob('*.pyx'))
    paths += sorted((ROOT / 'src/steger').glob('*.pyd'))
    paths += sorted((ROOT / 'src/steger').glob('*.so'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def timed(function, roi, mask):
    start = time.perf_counter_ns()
    output = function(roi, mask)
    return output, (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('neck_mask'))
    parser.add_argument('--reference', default='aeb2feb2cbe6e0b8601b3e0480a2a64784a63140')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not __debug__ or args.rounds < 1:
        parser.error('Assertions must be enabled and rounds positive')
    cases, hashes = [], []
    for a, b in batch.image_pairs(args.input_dir):
        with Image.open(a) as im:
            roi = np.array(im)
        with Image.open(b) as im:
            mask = np.array(im)
        roi.flags.writeable = mask.flags.writeable = False
        cases.append((a.stem, roi, mask, read_image(a), read_image(b, mask=True)))
        hashes.append(dict(id=a.stem, roi_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
                           mask_sha256=hashlib.sha256(b.read_bytes()).hexdigest()))
    sources = fingerprint()
    records = []
    with reference_modules(args.reference) as (reference, reference_batch, ref_worker, commit):
        print('Checking frozen baseline: real candidates, complete traces and diagnostics...', flush=True)
        parity = real_parity(reference_batch, ref_worker, cases)
        if args.synthetic:
            parity.update(synthetic_parity(reference))
        start = time.perf_counter()
        with NeckAngleWorker(workers=args.workers) as worker:
            pool_startup_ms = (time.perf_counter() - start) * 1000
            parallel_candidates = 0
            run_candidates = worker._candidates

            def checked_candidates(image, geometry, scales):
                nonlocal parallel_candidates
                actual = run_candidates(image, geometry, scales)
                expected = candidate_runs(image, geometry, scales)
                assert_exact(expected, actual, 'parallel.candidates')
                for a, b in zip(expected, actual, strict=True):
                    assert_exact(a.contour, b.contour, 'parallel.actual_contour')
                    assert_exact(a.local_measurement, b.local_measurement, 'parallel.fit')
                parallel_candidates += len(actual)
                return actual

            worker._candidates = checked_candidates
            outputs = {}
            for identifier, roi, mask, *_ in cases:
                expected = reference.measure_neck(roi, mask)
                assert_exact(expected, measure_neck(roi, mask), identifier + '.serial_full_result')
                assert_exact(expected, worker.measure(roi, mask), identifier + '.parallel_full_result')
                outputs[identifier] = expected
            worker._candidates = run_candidates
            parity['parallel_candidate_contours'] = parallel_candidates
            print(f'Bitwise parity passed: {parity}', flush=True)
            functions = dict(baseline=reference.measure_neck, serial_native=measure_neck,
                             parallel_scales=worker.measure)
            rng = np.random.default_rng(20261105)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='neck-angle') as angle_thread:
                angle_thread.submit(lambda: None).result()
                for function in functions.values():
                    for _, roi, mask, *_ in cases:
                        for _ in range(2):
                            angle_thread.submit(function, roi, mask).result()
                for round_index in range(args.rounds):
                    for index in rng.permutation(len(cases)):
                        identifier, roi, mask, *_ = cases[index]
                        for name in rng.permutation(list(functions)):
                            before = time.perf_counter_ns()
                            result, elapsed = angle_thread.submit(timed, functions[name], roi, mask).result()
                            outer = (time.perf_counter_ns() - before) / 1e6
                            assert_exact(outputs[identifier], result, identifier + '.repeat')
                            records.append(dict(round=round_index, id=identifier, mode=name,
                                                processing_ms=elapsed, submit_to_result_ms=outer))
                    if (round_index + 1) % 10 == 0:
                        print(f'Paired rounds: {round_index + 1}/{args.rounds}', flush=True)
            # Windows-only resource snapshot, outside all computation timers.
            resources = None
            if os.name == 'nt':
                ids = ','.join(str(p) for p in worker._executor._processes)
                command = f'Get-Process -Id {ids} | Select-Object Id,WorkingSet64,CPU | ConvertTo-Json'
                resources = json.loads(subprocess.check_output(
                    ['powershell', '-NoProfile', '-Command', command], text=True))
    assert fingerprint() == sources, 'Source/binary changed during benchmark'
    stages = {}
    for mode in ('baseline', 'serial_native', 'parallel_scales'):
        group = [r for r in records if r['mode'] == mode]
        stages[mode] = dict(
            **statistics([r['processing_ms'] for r in group]),
            submit_to_result=statistics([r['submit_to_result_ms'] for r in group]),
            per_image={identifier: {
                **statistics([r['processing_ms'] for r in group if r['id'] == identifier]),
                **outputs[identifier]} for identifier, *_ in cases})
    report = dict(
        measured_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        reference_commit=commit, branch=subprocess.check_output(
            ['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        rounds=args.rounds, image_count=len(cases), input_hashes=hashes,
        source_and_binary_hashes=sources,
        native_enabled=detector._native_refine_points is not None,
        environment=dict(cpu='Intel Core i5-12500 (6 cores / 12 logical processors)',
                         platform=platform.platform(), python=platform.python_version(),
                         numpy=np.__version__, scipy=scipy.__version__,
                         thread_environment={k: os.environ.get(k) for k in
                                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}),
        scope='Resident raw RGB uint8 ROI and uint8 mask to FULL result dictionary; one persistent parent angle thread; five persistent child processes in parallel mode; full float64 policy preserved. Includes conversion, mask geometry, shared-memory copies, IPC, scale/rescue search, fits and diagnostics. Excludes import/pool startup, disk I/O/decoding, segmentation, plots, trace capture, extra queue backlog and concurrent segmentation.',
        pool_startup_ms=pool_startup_ms, child_resources_after_run=resources,
        parity=parity, bitwise_exact=True,
        images_returning_angle=sum(o['angle_deg'] is not None for o in outputs.values()),
        stages=stages,
        speedup_parallel=stages['baseline']['mean_ms']/stages['parallel_scales']['mean_ms'],
        speedup_serial=stages['baseline']['mean_ms']/stages['serial_native']['mean_ms'],
        samples=records)
    write_json(args.output, report)
    with args.output.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({mode: {k: v for k,v in s.items() if k not in ('per_image','submit_to_result')}
                      for mode,s in stages.items()}, indent=2), flush=True)
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
