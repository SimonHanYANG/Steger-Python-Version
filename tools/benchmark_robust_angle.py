"""Validate all local inputs and time the complete resident-array worker."""
import argparse
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

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/neck-analysis'


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
    args.output.mkdir(parents=True,exist_ok=True)
    cases=[]; records=[]; unchanged=0
    for path in sorted((ROOT/'neck_mask').glob('*_mask.png')):
        identifier=path.stem[:-5]; image_path=path.with_name(identifier+'.png')
        with Image.open(image_path) as im: roi=np.array(im)
        with Image.open(path) as im: mask=np.array(im)
        expected=measure_neck(roi,mask)
        old=json.loads((BASE/'local-20261105/analysis'/identifier/'result.json').read_text(encoding='utf-8'))
        cpp=json.loads((BASE/'cpp-angle-20261105/analysis'/identifier/'result.json').read_text(encoding='utf-8'))
        gray=read_image(image_path);geometry=mask_geometry(mask!=0)
        # Exact raw extraction parity at the historical selection, including all
        # contours, normals, responses, diagnostics and trace metadata.
        extracted=detect(gray,geometry['mask'],old['sigma'],old['polarity'],trace=True)
        assert extracted.to_dict()==old['detector'], identifier+' extraction regression'
        unchanged+=1
        changed=any(expected.get(k)!=cpp.get(k) for k in ('sigma','polarity','contour_index'))
        if changed:
            extracted=detect(gray,geometry['mask'],expected['sigma'],expected['polarity'],trace=True)
        contour=extracted.contours[expected['contour_index']]
        r=dict(id=identifier,image_path=str(image_path),mask_path=str(path),**expected,
               old_endpoint_angle_deg=old.get('angle_deg'),cpp_angle_deg=cpp.get('angle_deg'),
               selection_changed=changed,points=contour.points.tolist(),normals=contour.normals.tolist())
        records.append(r);cases.append((identifier,roi,mask,expected))
        print(identifier,expected['angle_deg'],expected['status'],expected.get('quality_flags'),flush=True)
    samples=[]
    with NeckAngleWorker(workers=5) as worker, ThreadPoolExecutor(max_workers=1) as angle_thread:
        for _,roi,mask,expected in cases:
            for _ in range(2):
                assert angle_thread.submit(worker.measure,roi,mask).result()==expected
        rng=np.random.default_rng(20261105)
        for iteration in range(args.rounds):
            for index in rng.permutation(len(cases)):
                identifier,roi,mask,expected=cases[index]
                start=time.perf_counter_ns()
                result,elapsed=angle_thread.submit(timed,worker.measure,roi,mask).result()
                outer=(time.perf_counter_ns()-start)/1e6
                assert result==expected
                samples.append(dict(id=identifier,round=iteration,processing_ms=elapsed,
                                    submit_to_result_ms=outer))
    report=dict(method='geometry_arclength_direction_range_v2',rounds=args.rounds,
                input_scope='resident raw 64x64 RGB uint8 ROI and uint8 mask; full detection, measurement, quality checks and IPC; one angle thread + 5 persistent processes; excludes file I/O, segmentation, plotting, startup, backlog and external concurrent segmentation load',
                native_enabled=native_acceleration_available(),python=platform.python_version(),
                validation=dict(detectors_bitwise_unchanged=unchanged,serial_parallel_outputs_equal=True),
                statistics=statistics([s['processing_ms'] for s in samples]),
                submit_to_result=statistics([s['submit_to_result_ms'] for s in samples]),
                per_image={i:statistics([s['processing_ms'] for s in samples if s['id']==i]) for i,_,_,_ in cases},
                samples=samples,results=records,
                source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/'src/steger').glob('*.py')})
    write_json(args.output/'benchmark.json',report)
    print(json.dumps({k:report[k] for k in ('validation','statistics','submit_to_result')},indent=2))


if __name__=='__main__':
    main()
