"""Known-truth checks: original 84 profiles + circle, plus raw neck-style lines."""
import json
import hashlib
import inspect
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np

from steger import lines_gauss,measure_bending
from steger.io import write_json
from steger.synthetic import straight_line,circular_line
from benchmark_gaussian_support import SUPPORTS

OUT=Path('reports/neck-analysis/gaussian-support-20261106')


def cases():
    result=[]
    for model in ('bar-shaped','gaussian','parabolic'):
        for width in (1.,1.5,2.5):
            for angle in (0,23,61,90):
                for asymmetry in (0,.5):
                    result.append(dict(model=model,half_width=width,angle_deg=angle,
                                       offset=.237,asymmetry=asymmetry,quadrature=32))
        for noise in (1,3):
            for polarity in ('light','dark'):
                result.append(dict(model=model,half_width=1.5,angle_deg=37,offset=-.319,
                                   asymmetry=.35,noise=noise,polarity=polarity,quantize=True,
                                   seed=20261104,quadrature=32))
    return result


def corrected(image,gt,case,t):
    width=case['half_width'];sigma=max(1.1,width/np.sqrt(3)+.4)
    r=lines_gauss(image,sigma,.4,1.2,case.get('polarity','light'),True,case['model'],False,
                  gaussian_truncate=t)
    if not r.contours:return dict(passed=False,failure='no contour')
    c=max(r.contours,key=lambda c:c.length)
    valid=np.all((c.points>14)&(c.points<57),axis=1)
    if valid.sum()<10:return dict(passed=False,failure='too few interior points')
    position=np.abs((c.points[valid]-gt['center'])@gt['normal'])
    widths=np.abs(c.attributes['width_left'][valid]-width)
    coverage=float(np.mean(c.attributes['profile_fit_valid'][valid]))
    noise=case.get('noise',0)
    pl=.01 if noise==0 else .08 if noise<=1 else .20
    wl=.025 if noise==0 else .12 if noise<=1 else .30
    pp=float(np.percentile(position,95));wp=float(np.percentile(widths,95))
    return dict(position_p95_px=pp,width_p95_px=wp,fit_coverage=coverage,
                passed=pp<=pl and wp<=wl and coverage>=.95,
                gates=dict(position_p95_px=pl,width_p95_px=wl,fit_coverage=.95))


def circle(t):
    image,gt=circular_line(shape=(112,112),radius=30)
    rr,cc=np.indices(image.shape);center=gt['center']
    theta=np.arctan2(rr-center[0],cc-center[1])
    mask=(theta>.2)&(theta<1.8)&(abs(np.hypot(rr-center[0],cc-center[1])-30)<3)
    r=lines_gauss(image,1.5,.4,1.2,'light',True,'gaussian',False,mask=mask,gaussian_truncate=t)
    if not r.contours:return dict(passed=False,failure='no contour')
    c=max(r.contours,key=lambda c:c.length)
    measured=measure_bending(c).angle_deg
    a=np.unwrap(np.arctan2(c.rows-center[0],c.cols-center[1]))
    expected=float(abs(np.degrees(a[-1]-a[0]))*.7)
    radial=float(np.percentile(abs(np.linalg.norm(c.points-center,axis=1)-30),95))
    return dict(angle_deg=measured,expected_angle_deg=expected,angle_error_deg=abs(measured-expected),
                radius_p95_px=radial,passed=radial<.01 and abs(measured-expected)<.1)


def evaluate_profile(item):
    i,case=item
    image,gt=straight_line(shape=(72,72),**case)
    return dict(id=i,case=case,variants={str(t):corrected(image,gt,case,t) for t in SUPPORTS})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    progress=OUT/'synthetic-progress.json'
    original_cases=cases()
    digest=hashlib.sha256(repr(SUPPORTS).encode())
    for function in (cases,corrected,evaluate_profile):
        digest.update(inspect.getsource(function).encode())
    for source in sorted((Path(__file__).resolve().parents[1]/'src/steger').glob('*.py')):
        digest.update(source.name.encode());digest.update(source.read_bytes())
    signature=digest.hexdigest()
    saved=json.loads(progress.read_text(encoding='utf-8')) if progress.exists() else {}
    profiles=saved.get('profiles',[]) if saved.get('signature')==signature else []
    for r in profiles:assert r['case']==original_cases[r['id']]
    done={r['id'] for r in profiles}
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as executor:
        futures=[executor.submit(evaluate_profile,(i,case)) for i,case in enumerate(original_cases) if i not in done]
        for future in as_completed(futures):
            profiles.append(future.result())
            write_json(progress,dict(signature=signature,profiles=profiles))
            print('Corrected truth profiles',len(profiles),'/84',flush=True)
    profiles.sort(key=lambda r:r['id'])
    raw=[]
    for angle in (0,23,39,61,90):
        for offset in (-.31,.237):
            for noise in (0,1,3):
                for polarity in ('light','dark'):
                    image,gt=straight_line(shape=(64,64),model='gaussian',angle_deg=angle,
                        offset=offset,noise=noise,polarity=polarity,asymmetry=.35,quantize=True,seed=20261106)
                    delta=np.moveaxis(np.indices(image.shape),0,-1)-gt['center']
                    tangent=np.array([-gt['normal'][1],gt['normal'][0]])
                    mask=(abs(delta@gt['normal'])<4)&(abs(delta@tangent)<20)
                    variants={}
                    for t in SUPPORTS:
                        r=lines_gauss(image,1.5,.05,.15,polarity,False,'none',False,mask=mask,
                                      profile_refinement=False,gaussian_truncate=t)
                        if not r.contours:
                            variants[str(t)]=dict(angle_deg=None,failure='no contour');continue
                        c=max(r.contours,key=lambda c:c.length)
                        try:a=measure_bending(c).angle_deg
                        except ValueError:a=None
                        variants[str(t)]=dict(angle_deg=a,points=c.points.tolist(),
                            position_p95_px=float(np.percentile(abs((c.points-gt['center'])@gt['normal']),95)))
                    raw.append(dict(case=dict(angle=angle,offset=offset,noise=noise,polarity=polarity),variants=variants))
    circles={str(t):circle(t) for t in SUPPORTS}
    summary={}
    for t in SUPPORTS:
        key=str(t)
        angle_deltas=[abs(r['variants'][key]['angle_deg']-r['variants']['6.0']['angle_deg']) for r in raw
                      if all(r['variants'][k]['angle_deg'] is not None for k in (key,'6.0'))]
        pos_deltas=[abs(r['variants'][key]['position_p95_px']-r['variants']['6.0']['position_p95_px']) for r in raw
                    if all('position_p95_px' in r['variants'][k] for k in (key,'6.0'))]
        summary[key]=dict(corrected_profile_passed=sum(r['variants'][key]['passed'] for r in profiles),
                          corrected_profile_total=len(profiles),circle=circles[key],raw_cases=len(raw),
                          raw_angle_count=sum(r['variants'][key]['angle_deg'] is not None for r in raw),
                          raw_max_angle_change_deg=max(angle_deltas),
                          raw_max_position_p95_change_px=max(pos_deltas))
    write_json(OUT/'synthetic.json',dict(summary=summary,corrected_profiles=profiles,
        circle=circles,raw_neck_style=raw,
        scope='Original 84 corrected-profile truth gates and circle unchanged. Additional 60 raw production-style Gaussian lines disable width/profile refinement, matching neck detection; compare known-line positions and zero-angle truth. Real images still lack manual ground truth.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
