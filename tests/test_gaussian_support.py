"""Only Gaussian support changes; default remains the frozen released behavior."""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import convolve1d

from steger import lines_gauss, measure_neck
from steger.derivatives import gaussian_masks, image_derivatives
from steger.neck import _detect_neck_polarities
from steger.synthetic import straight_line
from steger.worker import NeckAngleWorker


def example():
    image, gt=straight_line(shape=(40,40),angle_deg=31,model='gaussian',polarity='dark',quantize=True)
    grid=np.moveaxis(np.indices(image.shape),0,-1)-gt['center']
    tangent=np.array([-gt['normal'][1],gt['normal'][0]])
    mask=(abs(grid@gt['normal'])<4)&(abs(grid@tangent)<15)
    return image,mask


@pytest.mark.parametrize('sigma',[1.,1.5,2.,2.5,3.,4.5])
@pytest.mark.parametrize('border',['reflect','mirror','nearest'])
def test_default_derivatives_are_bitwise_release(sigma,border):
    try:
        source=subprocess.check_output(['git','show','v1.0.2:src/steger/derivatives.py'],
                                      cwd=Path(__file__).resolve().parents[1],stderr=subprocess.DEVNULL)
    except (OSError,subprocess.CalledProcessError):
        pytest.skip('Release Git object required for frozen comparison')
    scope={'__name__':'steger._released_derivatives','__package__':'steger'}
    exec(compile(source,'v1.0.2/derivatives.py','exec'),scope)
    image=np.random.default_rng(71).normal(size=(21,24))
    expected=scope['image_derivatives'](image,sigma,border=border)
    actual=image_derivatives(image,sigma,border=border)
    for key in expected:assert expected[key].tobytes()==actual[key].tobytes()


@pytest.mark.parametrize('truncate,width',[(6,27),(5,23),(4,19),(3.5,17),(3,15),(2.5,13),(2,11)])
def test_kernel_support_changes_without_changing_interior_weights(truncate,width):
    ref=gaussian_masks(2.,6.);actual=gaussian_masks(2.,truncate)
    assert len(actual[0])==width
    offset=(len(ref[0])-width)//2
    for x,y in zip(actual,ref,strict=True):
        assert x.dtype==np.float64 and not x.flags.writeable
        assert x[1:-1].tobytes()==y[offset+1:offset+width-1].tobytes()


@pytest.mark.parametrize('value',[0,-1,np.nan,np.inf,[3],None])
def test_invalid_support_rejected_at_public_entry_points(value):
    image,mask=example()
    for function in (lambda:measure_neck(image,mask,gaussian_truncate=value),
                     lambda:lines_gauss(image,mask=mask,gaussian_truncate=value),
                     lambda:NeckAngleWorker(gaussian_truncate=value)):
        with pytest.raises(ValueError,match='gaussian_truncate'):function()


def test_trace_and_shared_polarities_use_the_requested_support():
    image,mask=example();sigma=1.5;truncate=3.
    pair=_detect_neck_polarities(image,mask,sigma,.05,.15,gaussian_truncate=truncate)
    for actual,polarity in zip(pair,('dark','light'),strict=True):
        traced=lines_gauss(image,sigma,.05,.15,polarity,False,'none',False,mask=mask,
                           profile_refinement=False,return_trace=True,gaussian_truncate=truncate)
        assert actual.to_dict()==traced.to_dict()
        g0=gaussian_masks(sigma,truncate)[0]
        smooth=convolve1d(convolve1d(image.astype(float),g0,axis=0,mode='reflect'),g0,axis=1,mode='reflect')
        assert traced.trace.smoothed.tobytes()==smooth.tobytes()
        assert traced.parameters['gaussian_truncate']==truncate


def test_worker_per_call_support_is_isolated_and_constructor_default_restored():
    image,mask=example()
    with NeckAngleWorker(workers=2,gaussian_truncate=4.) as worker:
        for truncate in (6.,4.,3.,5.,6.):
            expected=measure_neck(image,mask,gaussian_truncate=truncate)
            assert worker.measure(image,mask,gaussian_truncate=truncate)==expected
        with pytest.raises(ValueError,match='gaussian_truncate'):
            worker.measure(image,mask,gaussian_truncate=0)
        with ThreadPoolExecutor(max_workers=2) as callers:
            futures=[callers.submit(worker.measure,image,mask,gaussian_truncate=t) for t in (3.,6.,4.)]
            assert [f.result() for f in futures]==[measure_neck(image,mask,gaussian_truncate=t) for t in (3.,6.,4.)]
        assert worker.measure(image,mask)==measure_neck(image,mask,gaussian_truncate=4.)


def test_mask_outside_information_is_still_used():
    image,mask=example();changed=image.copy();changed[~mask]=0
    a=image_derivatives(image,2.,gaussian_truncate=3.)
    b=image_derivatives(changed,2.,gaussian_truncate=3.)
    assert any(not np.array_equal(a[k][mask],b[k][mask]) for k in a)
