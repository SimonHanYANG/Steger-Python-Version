"""Sweep support radii from the released full-ROI detector; paired timing."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from steger.derivatives import gaussian_masks,image_derivatives
from steger.io import write_json
from steger.neck import (
    SCALES,RESCUE_SCALES,_prepare_neck_inputs,mask_geometry,candidate_runs,
    choose_candidate,_neck_result,
)
from steger.worker import NeckAngleWorker,native_acceleration_available

ROOT=Path(__file__).resolve().parents[1]
SUPPORTS=(6.,5.,4.,3.5,3.,2.5,2.)


def statistics(values):
    a=np.asarray(values)
    return dict(calls=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a,95)),max_ms=float(a.max()))


def analyze(roi,mask,truncate):
    gray,mask=_prepare_neck_inputs(roi,mask);g=mask_geometry(mask)
    candidates=candidate_runs(gray,g,gaussian_truncate=truncate)
    rescue=not any(c['span_coverage']>=.75 and c['length_px']>=15 for c in candidates)
    if rescue:candidates.extend(candidate_runs(gray,g,RESCUE_SCALES,gaussian_truncate=truncate))
    selected=choose_candidate(candidates)
    result=_neck_result(g,candidates)
    detail=dict(**result,rescue_used=rescue,candidate_count=len(candidates),
                points=[] if selected is None else selected.contour.points.tolist(),
                normals=[] if selected is None else selected.contour.normals.tolist(),
                response=[] if selected is None else selected.contour.response.tolist())
    return result,detail


def _resample(points):
    if len(points)<2:return np.asarray(points)
    p=np.asarray(points);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    q=np.linspace(0,s[-1],max(2,int(np.ceil(s[-1]/.25))+1))
    return np.column_stack([np.interp(q,s,p[:,axis]) for axis in (0,1)])


def _distances(p,line):
    line=np.asarray(line);a=line[:-1];d=np.diff(line,axis=0)
    denom=np.sum(d*d,axis=1)
    t=np.divide(np.einsum('nki,ki->nk',p[:,None,:]-a,d),denom,
                out=np.zeros((len(p),len(d))),where=denom>1e-20)
    closest=a+np.clip(t,0,1)[...,None]*d
    return np.min(np.linalg.norm(p[:,None,:]-closest,axis=-1),axis=1)


def compare(current,baseline):
    a,b=current['points'],baseline['points']
    equal=all(np.asarray(current[k],dtype=np.float64).tobytes()==np.asarray(baseline[k],dtype=np.float64).tobytes()
              for k in ('points','normals','response'))
    distances=None
    if len(a)>1 and len(b)>1:
        ds=np.r_[_distances(_resample(a),b),_distances(_resample(b),a)]
        distances=dict(mean_px=float(ds.mean()),p95_px=float(np.percentile(ds,95)),max_px=float(ds.max()))
    ca,ba=current.get('angle_deg'),baseline.get('angle_deg')
    point_max=None
    if len(a)==len(b) and len(a):point_max=float(np.linalg.norm(np.asarray(a)-np.asarray(b),axis=1).max())
    return dict(raw_contour_bitwise_equal=equal,matched_index_max_displacement_px=point_max,
                selected_geometry=distances,angle_delta_deg=None if ca is None or ba is None else ca-ba,
                selection_changed=any(current.get(k)!=baseline.get(k) for k in ('sigma','polarity','contour_index')),
                status_changed=current['status']!=baseline['status'])


def timed(worker,roi,mask,truncate):
    start=time.perf_counter_ns()
    r=worker.measure(roi,mask,gaussian_truncate=truncate)
    return r,(time.perf_counter_ns()-start)/1e6


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=60)
    parser.add_argument('--output',type=Path,default=Path('reports/neck-analysis/gaussian-support-20261106'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert native_acceleration_available()
    accepted=json.loads((ROOT/'reports/neck-analysis/robust-angle-20261105/benchmark.json').read_text(encoding='utf-8'))
    old={r['id']:r for r in accepted['results']}
    cases=[];records=[]
    for mask_path in sorted((ROOT/'neck_mask').glob('*_mask.png')):
        identifier=mask_path.stem[:-5];image_path=mask_path.with_name(identifier+'.png')
        roi=np.array(Image.open(image_path));mask=np.array(Image.open(mask_path))
        variants={};expected={}
        for t in SUPPORTS:
            result,detail=analyze(roi,mask,t);variants[str(t)]=detail;expected[t]=result
            if t==6.:
                prior=old[identifier]
                ref={k:v for k,v in prior.items() if k not in ('id','image_path','mask_path','old_endpoint_angle_deg','cpp_angle_deg','selection_changed','points','normals')}
                assert json.loads(json.dumps(result))==ref,identifier+' released output changed'
                for key in ('points','normals'):
                    assert np.array(variants['6.0'][key]).tobytes()==np.array(prior[key]).tobytes(),identifier
            detail['comparison']=compare(detail,variants['6.0'])
        records.append(dict(id=identifier,image_path=str(image_path),mask_path=str(mask_path),variants=variants))
        cases.append((identifier,roi,mask,expected))
        print(identifier,'validated',flush=True)
    samples=[]
    with NeckAngleWorker(workers=5) as worker,ThreadPoolExecutor(max_workers=1) as thread:
        for _,roi,mask,expected in cases:
            for t in SUPPORTS:
                assert thread.submit(worker.measure,roi,mask,gaussian_truncate=t).result()==expected[t]
        rng=np.random.default_rng(20261106)
        for iteration in range(args.rounds):
            for i in rng.permutation(len(cases)):
                identifier,roi,mask,expected=cases[i]
                for t in rng.permutation(SUPPORTS):
                    r,elapsed=thread.submit(timed,worker,roi,mask,float(t)).result()
                    assert r==expected[t]
                    samples.append(dict(id=identifier,round=iteration,truncate=float(t),processing_ms=elapsed))
    # Independent, short timings of just the actual five derivative fields.
    derivative_samples=[];rng=np.random.default_rng(621)
    for iteration in range(40):
        for identifier,roi,mask,_ in cases:
            gray,_=_prepare_neck_inputs(roi,mask)
            for t in rng.permutation(SUPPORTS):
                start=time.perf_counter_ns();image_derivatives(gray,2.,gaussian_truncate=float(t))
                derivative_samples.append(dict(id=identifier,truncate=float(t),ms=(time.perf_counter_ns()-start)/1e6))
    report=dict(release='v1.0.2',supports=SUPPORTS,rounds=args.rounds,
                scope='One warmed five-process worker; original resident 64x64 ROI/mask; random interleaving of supports within each image; all 29 cases including no-angle outcomes; full preparation/search/rescue/measurement/IPC; excludes I/O, segmentation, plots, startup and queue backlog.',
                kernel_widths={str(t):{str(s):len(gaussian_masks(s,t)[0]) for s in (*SCALES,*RESCUE_SCALES)} for t in SUPPORTS},
                validation=dict(released_full_outputs_equal=29,released_selected_points_and_normals_bitwise_equal=29,
                                serial_and_worker_outputs_equal=True),
                statistics={str(t):statistics([s['processing_ms'] for s in samples if s['truncate']==t]) for t in SUPPORTS},
                derivative_statistics_sigma2={str(t):statistics([s['ms'] for s in derivative_samples if s['truncate']==t]) for t in SUPPORTS},
                per_image={identifier:{str(t):statistics([s['processing_ms'] for s in samples if s['id']==identifier and s['truncate']==t]) for t in SUPPORTS}
                           for identifier,_,_,_ in cases},results=records,samples=samples,derivative_samples=derivative_samples,
                source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/steger').glob('*.py')})
    write_json(args.output/'benchmark.json',report)
    print(json.dumps(report['statistics'],indent=2))


if __name__=='__main__':main()
