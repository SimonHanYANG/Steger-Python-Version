"""Paired release/full-search latency and precision checks on resident ROI/mask."""
import argparse
import json
import hashlib
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/neck-analysis/latency-1ms-20261106'
REVISION = 'f2e3081a545df7e77d9d09a42bf92b4c339567b3'


def prepare_reference():
    """Read the immutable release Git object; preserve any unexpected files."""
    folder = OUT / 'reference/steger_release'
    manifest = folder.parent / 'manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        assert data['revision'] == REVISION
        for name,digest in data['sha256'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
        return
    files = subprocess.check_output(['git','ls-tree','-r','--name-only',REVISION,'src/steger'],cwd=ROOT).decode().splitlines()
    hashes = {}
    for name in files:
        source = subprocess.check_output(['git','show',REVISION+':'+name],cwd=ROOT)
        relative = str(Path(name).relative_to('src/steger'))
        path = folder / relative
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            assert path.read_bytes() == source, 'Unexpected reference contents: '+str(path)
        else:
            path.write_bytes(source)
        hashes[relative] = hashlib.sha256(source).hexdigest()
    manifest.write_text(json.dumps(dict(revision=REVISION,sha256=hashes),indent=2),encoding='utf-8')


prepare_reference()
sys.path.insert(0, str(OUT / 'reference'))
# CPython's single-phase extension cache cannot isolate two binaries with the
# same PyInit symbol under package aliases. Share ONLY the unchanged released
# Newton/sampling entry points with the release Python source. The release has
# no imports for any of the added native routines.
from steger import _spline_native as native
sys.modules['steger_release._spline_native'] = native
from steger_release import neck as released
from steger_release.worker import NeckAngleWorker as ReleasedWorker
from steger import neck
from steger.io import write_json
from steger.worker import FastNeckAngleWorker, fast_acceleration_available


def stats(values):
    a = np.asarray(values)
    return dict(calls=len(a), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()))


def compare(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((compare(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max((compare(x, y) for x, y in zip(a, b, strict=True)), default=0.)
    if isinstance(a, (float, np.floating)):
        assert np.isfinite(a) and np.isfinite(b)
        error = abs(a - b)
        assert error < 1e-8, (a, b)
        return error
    assert a == b, (a, b)
    return 0.


def run(worker, roi, mask):
    start = time.perf_counter_ns()
    result = worker.measure(roi, mask)
    return result, (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--name', default='paired')
    args = parser.parse_args()
    assert args.rounds > 0
    assert fast_acceleration_available(), 'Rebuild optional native module'
    from steger import linking, detector
    from steger import robust_measurement
    assert linking._native_neighbor_graph is native.neighbor_graph
    assert robust_measurement._native_window_samples is native.window_samples
    assert released._detect_neck_polarities.__globals__['_native_refine_points'] is native.refine_points
    old_native = (OUT/'reference/steger_release/_spline_native.pyx').read_text(encoding='utf-8')
    new_native = (ROOT/'src/steger/_spline_native.pyx').read_text(encoding='utf-8')
    marker = 'cdef inline double reflected'
    assert new_native[new_native.index(marker):].startswith(old_native[old_native.index(marker):].rstrip())
    cases = []
    records = []
    for path in sorted((ROOT / 'neck_mask').glob('*_mask.png')):
        identifier = path.stem.removesuffix('_mask')
        image_path = path.with_name(identifier + '.png')
        roi = np.array(Image.open(image_path))
        mask = np.array(Image.open(path))
        expected = released.measure_neck(roi, mask)
        actual = neck.measure_neck(roi, mask)
        difference = compare(expected, actual)
        gray, selected = released._prepare_neck_inputs(roi, mask)
        geometry = released.mask_geometry(selected)
        contour_count = 0
        for sigma in (*released.SCALES, *released.RESCUE_SCALES):
            old_pair = released._detect_neck_polarities(gray, geometry['mask'], sigma, released.LOW, released.HIGH)
            new_pair = neck._detect_neck_polarities(gray, geometry['mask'], sigma, neck.LOW, neck.HIGH)
            for old, new in zip(old_pair, new_pair, strict=True):
                assert old.to_dict() == new.to_dict(), (identifier, sigma)
                assert len(old.contours) == len(new.contours)
                for a,b in zip(old.contours,new.contours,strict=True):
                    for key in ('points','normals','response'):
                        assert getattr(a,key).tobytes() == getattr(b,key).tobytes(), (identifier,sigma,key)
                contour_count += len(old.contours)
        records.append(dict(id=identifier, image_path=str(image_path), mask_path=str(path),
                            baseline=expected, optimized=actual, max_output_difference=difference,
                            all_scale_contours_bitwise_equal=True, all_scale_contour_count=contour_count))
        cases.append((identifier, roi, mask, expected, actual))
    samples = []
    with ReleasedWorker() as baseline, FastNeckAngleWorker() as fast, ThreadPoolExecutor(max_workers=1) as caller:
        workers = {'baseline': baseline, 'optimized': fast}
        for _, roi, mask, expected, actual in cases:
            assert caller.submit(baseline.measure, roi, mask).result() == expected
            compare(actual, caller.submit(fast.measure, roi, mask).result())
        rng = np.random.default_rng(620261106)
        for iteration in range(args.rounds):
            for index in rng.permutation(len(cases)):
                identifier, roi, mask, expected, actual = cases[index]
                for name in rng.permutation(['baseline', 'optimized']):
                    result, elapsed = caller.submit(run, workers[name], roi, mask).result()
                    compare(expected if name == 'baseline' else actual, result)
                    assert result == (expected if name == 'baseline' else actual)
                    samples.append(dict(id=identifier, round=iteration, variant=name, processing_ms=elapsed))
    report = dict(release='v1.0.2', release_revision=REVISION, rounds=args.rounds, image_count=len(cases),
                  validation=dict(unchanged_native_newton_source=True, all_timed_results_exactly_equal=True),
                  environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,
                                   fast_native_enabled=fast_acceleration_available()),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*(ROOT/'src/steger').glob('*.py'), ROOT/'src/steger/_spline_native.pyx']},
                  scope='Paired randomized full calls in warmed five-process pools, resident 64x64 ROI/mask; all original scales, polarities and rescue; excludes I/O, segmentation, plots, startup and application backlog.',
                  statistics={name: stats([r['processing_ms'] for r in samples if r['variant'] == name])
                              for name in ('baseline', 'optimized')},
                  results=records, samples=samples)
    write_json(OUT / (args.name + '.json'), report)
    print(json.dumps(report['statistics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
