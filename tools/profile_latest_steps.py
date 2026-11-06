"""Profile the latest complete worker by additive critical-scale stages.

Instrumentation is separate from the uninstrumented production latency runs.
For each parallel wave use the slowest scale's exclusive stage times; account
for the remaining wall time as communication/scheduling. Never sum parallel CPU
work and present that as request latency. Production measure() remains intact.
"""
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from steger import detector, neck, measure_neck
from steger.io import write_json
import steger.worker as module

_times = None
_stack = []


def exclusive(function, category):
    def wrapped(*args, **kwargs):
        frame = [time.perf_counter_ns(), 0]
        _stack.append(frame)
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter_ns() - frame[0]
            _stack.pop()
            _times[category] += (elapsed - frame[1]) / 1e6
            if _stack:
                _stack[-1][1] += elapsed
    return wrapped


def initialize(name, capacity):
    module._attach_workspace(name, capacity)
    detector.image_derivatives = exclusive(detector.image_derivatives, 'gaussian')
    detector._extract = exclusive(detector._extract, 'subpixel')
    detector.link_points = exclusive(detector.link_points, 'linking')
    neck._detect_neck_polarities = exclusive(neck._detect_neck_polarities, 'hessian_taylor')


def profiled_scale(request):
    global _times
    _times = dict.fromkeys(('gaussian', 'hessian_taylor', 'subpixel', 'linking', 'candidate_fits'), 0.)
    shape, axis, span, sigma = request
    memory, capacity = module._shared_input
    image, distance, mask = module._workspace_views(memory, capacity, shape)
    for a in (image, distance, mask):
        a.flags.writeable = False
    geometry = dict(mask=mask, axis=axis, span=span, distance=distance)
    start = time.perf_counter_ns()
    result = exclusive(neck.candidate_runs, 'candidate_fits')(image, geometry, (sigma,))
    duration = (time.perf_counter_ns() - start) / 1e6
    _times['candidate_fits'] += duration - sum(_times.values())
    return result, duration, _times


class ProfiledWorker(module.NeckAngleWorker):
    def __init__(self, fast=False):
        super().__init__(workers=5)
        self._executor.shutdown()
        if fast:
            from steger._direct_pool import DirectPool
            self._executor = DirectPool(5, initialize, (self._memory.name, self._capacity))
        else:
            self._executor = ProcessPoolExecutor(
                max_workers=5, mp_context=mp.get_context('spawn'), initializer=initialize,
                initargs=(self._memory.name, self._capacity))

    def _candidates(self, image, geometry, scales):
        requests = [(image.shape, geometry['axis'], geometry['span'], s) for s in scales]
        start = time.perf_counter_ns()
        groups = list(self._executor.map(profiled_scale, requests))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        critical = max(groups, key=lambda g: g[1])
        for key, value in critical[2].items():
            self.steps[key] += value
        self.steps['handoff'] += elapsed - critical[1]
        return [c for group, _, _ in groups for c in group]

    def profiled_measure(self, roi, mask):
        self.steps = dict.fromkeys(('input_mask', 'gaussian', 'hessian_taylor', 'subpixel',
                                   'linking', 'candidate_fits', 'final_angle', 'handoff'), 0.)
        start = time.perf_counter_ns()
        result = self.measure(roi, mask)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        self.steps['handoff'] += elapsed - sum(self.steps.values())
        return result, elapsed, dict(self.steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    cases = []
    for path in sorted(Path('neck_mask').glob('*_mask.png')):
        with Image.open(path.with_name(path.name.replace('_mask', ''))) as im:
            roi = np.array(im)
        with Image.open(path) as im:
            mask = np.array(im)
        cases.append((path.stem, roi, mask, measure_neck(roi, mask)))
    records = []
    with ProfiledWorker(fast=args.fast) as worker:
        originals = {}
        for name, category in (('_prepare_neck_inputs', 'input_mask'), ('mask_geometry', 'input_mask'),
                               ('_neck_result', 'final_angle')):
            function = getattr(module, name)
            originals[name] = function

            def wrap(*args, _function=function, _category=category, **kwargs):
                start = time.perf_counter_ns()
                result = _function(*args, **kwargs)
                worker.steps[_category] += (time.perf_counter_ns() - start) / 1e6
                return result

            setattr(module, name, wrap)
        try:
            with ThreadPoolExecutor(max_workers=1) as angle_thread:
                for identifier, roi, mask, expected in cases:
                    actual, *_ = angle_thread.submit(worker.profiled_measure, roi, mask).result()
                    assert actual == expected
                rng = np.random.default_rng(20261105)
                for iteration in range(args.rounds):
                    for index in rng.permutation(len(cases)):
                        identifier, roi, mask, expected = cases[index]
                        actual, elapsed, steps = angle_thread.submit(worker.profiled_measure, roi, mask).result()
                        assert actual == expected
                        assert abs(sum(steps.values()) - elapsed) < 1e-8
                        assert min(steps.values()) >= 0
                        records.append(dict(id=identifier, round=iteration, total_ms=elapsed, **steps))
        finally:
            for name, function in originals.items():
                setattr(module, name, function)
    report = dict(
        scope='Exclusive instrumentation; each parallel wave uses its slowest scale; other elapsed time goes to shared-memory/IPC/scheduling. Additive wall-time decomposition, not summed CPU time. Instrumentation overhead included; compare separately to uninstrumented production latency.',
        calls=len(records), image_count=len(cases),
        mean_total_ms=float(np.mean([r['total_ms'] for r in records])),
        stage_mean_ms={key: float(np.mean([r[key] for r in records])) for key in records[0]
                       if key not in ('id', 'round', 'total_ms')},
        outputs_unchanged=True, samples=records)
    write_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
