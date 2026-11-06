"""Paired resident-array main versus strict-mask experiment on every local ROI."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from steger import measure_neck
from steger.io import write_json
from steger.mask_only import MaskOnlyNeckAngleWorker, analyze_mask_only, contains, index_map
from steger.neck import _prepare_neck_inputs, candidate_runs, mask_geometry
from steger.worker import NeckAngleWorker, native_acceleration_available

ROOT=Path(__file__).resolve().parents[1]


def stats(values):
    a=np.asarray(values)
    return dict(calls=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a,95)),max_ms=float(a.max()))


def timed(function,roi,mask):
    start=time.perf_counter_ns();result=function(roi,mask)
    return result,(time.perf_counter_ns()-start)/1e6


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=60)
    parser.add_argument('--output',type=Path,default=Path('reports/neck-analysis/mask-only-20261106'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert native_acceleration_available()
    accepted=json.loads((ROOT/'reports/neck-analysis/robust-angle-20261105/benchmark.json').read_text(encoding='utf-8'))
    previous={r['id']:r for r in accepted['results']}
    cases=[];records=[]
    for path in sorted((ROOT/'neck_mask').glob('*_mask.png')):
        identifier=path.stem[:-5]
        roi=np.array(Image.open(path.with_name(identifier+'.png')))
        mask=np.array(Image.open(path))
        baseline=measure_neck(roi,mask)
        old=previous[identifier]
        expected={k:v for k,v in old.items() if k not in ('id','image_path','mask_path','old_endpoint_angle_deg','cpp_angle_deg','selection_changed','points','normals')}
        assert json.loads(json.dumps(baseline))==expected,identifier+' baseline changed'
        result,selected,candidates,g=analyze_mask_only(roi,mask)
        _,index=index_map(g['mask'])
        assert all(contains(index,c.contour.points).all() for c in candidates)
        # Independence proof includes outside NaN, infinity and unrelated texture.
        changed=roi.astype(float);changed[~g['mask']]=np.nan
        assert analyze_mask_only(changed,mask)[0]==result
        changed[~g['mask']]=np.inf
        assert analyze_mask_only(changed,mask)[0]==result
        mask_pixels=int(g['mask'].sum())
        gray,original_mask=_prepare_neck_inputs(roi,mask)
        initial_main=candidate_runs(gray,mask_geometry(original_mask))
        main_rescue=not any(c['span_coverage']>=.75 and c['length_px']>=15 for c in initial_main)
        initial_mask=[c for c in candidates if c['sigma']<=3.]
        mask_rescue=not any(c['span_coverage']>=.75 and c['length_px']>=15 for c in initial_mask)
        records.append(dict(id=identifier,image_path=old['image_path'],mask_path=old['mask_path'],
                            mask_pixels=mask_pixels,roi_pixels=int(g['mask'].size),
                            main=dict(**baseline,points=old['points'],normals=old['normals']),
                            mask_only=dict(**result,points=[] if selected is None else selected.contour.points.tolist(),
                                           normals=[] if selected is None else selected.contour.normals.tolist()),
                            candidate_count=len(candidates),main_rescue_used=main_rescue,
                            mask_rescue_used=mask_rescue))
        cases.append((identifier,roi,mask,baseline,result))
        print(identifier,'main',baseline['angle_deg'],'mask',result['angle_deg'],flush=True)
    samples=[]
    with NeckAngleWorker(workers=5) as full,MaskOnlyNeckAngleWorker(workers=5) as strict,ThreadPoolExecutor(max_workers=1) as thread:
        workers={'main':full,'mask_only':strict}
        for _,roi,mask,baseline,result in cases:
            for mode,expected in (('main',baseline),('mask_only',result)):
                for _ in range(2):assert thread.submit(workers[mode].measure,roi,mask).result()==expected
        rng=np.random.default_rng(20261106)
        for iteration in range(args.rounds):
            for i in rng.permutation(len(cases)):
                identifier,roi,mask,baseline,result=cases[i]
                for mode in rng.permutation(['main','mask_only']):
                    actual,elapsed=thread.submit(timed,workers[mode].measure,roi,mask).result()
                    assert actual==(baseline if mode=='main' else result)
                    samples.append(dict(id=identifier,round=iteration,mode=str(mode),processing_ms=elapsed))
    report=dict(rounds=args.rounds,paired_random_order=True,
                scope='Resident original 64x64 RGB ROI and mask to complete result; 5 persistent processes each mode; native kernels enabled; includes preparation/search/rescue/measurement/IPC; excludes I/O, segmentation, plots, startup and queue backlog. All 29 images including no-angle outcomes are timed.',
                validation=dict(main_matches_accepted_cases=29,outside_nan_infinity_invariance_cases=29,
                                all_candidate_points_inside_mask=True,serial_parallel_outputs_equal=True),
                statistics={m:stats([s['processing_ms'] for s in samples if s['mode']==m]) for m in ('main','mask_only')},
                per_image={r['id']:{m:stats([s['processing_ms'] for s in samples if s['id']==r['id'] and s['mode']==m])
                                    for m in ('main','mask_only')} for r in records},
                results=records,samples=samples,
                source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/'src/steger').glob('*') if p.suffix in ('.py','.pyx')})
    write_json(args.output/'benchmark.json',report)
    print(json.dumps({k:report[k] for k in ('statistics','validation')},indent=2))


if __name__=='__main__':main()
