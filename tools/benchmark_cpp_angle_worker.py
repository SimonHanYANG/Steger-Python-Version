"""Time the current C++-compatible measurement from resident raw ROI/mask."""
import argparse
import csv
import ctypes
import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from steger import measure_neck
from steger.io import read_image, write_json
from steger.neck import detect, mask_geometry
from steger.worker import NeckAngleWorker, native_acceleration_available

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/neck-analysis/cpp-angle-20261105'


def statistics(values):
    a=np.asarray(values)
    return dict(calls=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a,95)),max_ms=float(a.max()))


def timed(function,roi,mask):
    start=time.perf_counter_ns()
    result=function(roi,mask)
    return result,(time.perf_counter_ns()-start)/1e6


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=60)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cases=[]
    rows=json.loads((REPORT/'analysis/results.json').read_text(encoding='utf-8'))['results']
    old_root=ROOT/'reports/neck-analysis/local-20261105/analysis'
    library=ctypes.CDLL(str(REPORT/'oracle/cpp_angle.dll'))
    library.cpp_bend.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int]
    library.cpp_bend.restype=ctypes.c_float
    unchanged=0
    oracle_cases=0
    for row in rows:
        with Image.open(row['image_path']) as im:
            roi=np.array(im)
        with Image.open(row['mask_path']) as im:
            mask=np.array(im)
        expected=measure_neck(roi,mask)
        assert expected['status']==row['status'] and expected['angle_deg']==row['angle_deg']
        # Re-extract the old selected detection, independently of the new angle.
        old=json.loads((old_root/row['id']/'result.json').read_text(encoding='utf-8'))
        gray=read_image(row['image_path'])
        geometry=mask_geometry(mask!=0)
        result=detect(gray,geometry['mask'],old['sigma'],old['polarity'],trace=True)
        assert result.to_dict()==old['detector'], row['id']+' detector changed'
        unchanged+=1
        saved=json.loads((REPORT/'analysis'/row['id']/'result.json').read_text(encoding='utf-8'))
        c=saved['detector']['contours'][row['contour_index']]
        angles=np.ascontiguousarray(np.mod(np.arctan2(c['normal_col'],c['normal_row']),2*np.pi))
        if len(angles)>10:
            angle=library.cpp_bend(angles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(angles))
            local=expected['local_segment_angle_deg']
            assert np.float32(angle).tobytes()==np.float32(local).tobytes(),row['id']
            oracle_cases+=1
        cases.append((row,roi,mask,expected))
    samples=[]
    with NeckAngleWorker(workers=5) as worker, ThreadPoolExecutor(max_workers=1) as angle_thread:
        for row,roi,mask,expected in cases:
            for _ in range(2):
                assert angle_thread.submit(worker.measure,roi,mask).result()==expected
        rng=np.random.default_rng(20261105)
        for iteration in range(args.rounds):
            for index in rng.permutation(len(cases)):
                row,roi,mask,expected=cases[index]
                start=time.perf_counter_ns()
                result,elapsed=angle_thread.submit(timed,worker.measure,roi,mask).result()
                outer=(time.perf_counter_ns()-start)/1e6
                assert result==expected
                samples.append(dict(id=row['id'],round=iteration,processing_ms=elapsed,
                                    submit_to_result_ms=outer))
    per_image={row['id']:{**statistics([s['processing_ms'] for s in samples if s['id']==row['id']]),
                          'angle_deg':expected['angle_deg'],'status':expected['status']}
               for row,_,_,expected in cases}
    report=dict(method='cpp_legacy_10point_float32_direction_range',rounds=args.rounds,
                input_scope='resident raw 64x64 RGB uint8 ROI / uint8 mask; full compute, IPC and quality checks; one parent angle thread + 5 persistent processes; no I/O, segmentation, plots, startup or backlog',
                native_enabled=native_acceleration_available(),python=platform.python_version(),
                validation=dict(detectors_bitwise_unchanged=unchanged,
                                real_cpp_oracle_angles_bitwise_equal=oracle_cases,
                                serial_and_parallel_full_outputs_equal=True),
                statistics=statistics([s['processing_ms'] for s in samples]),
                submit_to_result=statistics([s['submit_to_result_ms'] for s in samples]),
                per_image=per_image,samples=samples,
                source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/'src/steger').glob('*.py')})
    write_json(args.output,report)
    with args.output.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]))
        writer.writeheader();writer.writerows(samples)
    print(json.dumps({k:report[k] for k in ('validation','statistics','submit_to_result')},indent=2))


if __name__=='__main__':
    main()
