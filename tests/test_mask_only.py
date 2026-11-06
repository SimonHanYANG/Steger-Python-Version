"""Mask isolation, independent numerical reference and worker parity."""
import numpy as np
import pytest

native = pytest.importorskip('steger._mask_native')
from steger.derivatives import gaussian_masks
from steger.mask_only import (
    MaskOnlyNeckAngleWorker, contains, index_map, measure_neck_mask_only,
    prepare, spline_system,
)
from steger.synthetic import straight_line


def example():
    image, truth = straight_line(shape=(40, 40), angle_deg=31, model='gaussian', polarity='dark')
    grid = np.moveaxis(np.indices(image.shape), 0, -1) - truth['center']
    tangent = np.array([-truth['normal'][1], truth['normal'][0]])
    mask = (abs(grid @ truth['normal'])<4) & (abs(grid @ tangent)<15)
    return image, mask


def test_constant_mask_has_exactly_zero_derivatives_including_its_boundary():
    mask=np.zeros((20,25),bool);mask[4:15,4:8]=True;mask[9:12,8:20]=True
    p,_=index_map(mask)
    result=native.gaussian_fields(np.full(len(p),185.),p,*gaussian_masks(2.))
    assert np.array_equal(result,np.zeros_like(result))


def test_compact_gaussian_matches_independent_numpy_quotient_rule():
    rng=np.random.default_rng(91);mask=rng.random((13,15))<.35
    p,_=index_map(mask);gray=rng.integers(0,256,len(p)).astype(float)
    g0,g1,g2=gaussian_masks(1.5);radius=len(g0)//2
    delta=p[:,None,:]-p[None,:,:]
    valid=np.all(abs(delta)<=radius,axis=-1)
    dr=np.clip(delta[...,0]+radius,0,2*radius)
    dc=np.clip(delta[...,1]+radius,0,2*radius)
    arrays=[g0[dr]*g0[dc],g1[dr]*g0[dc],g0[dr]*g1[dc],
            g2[dr]*g0[dc],g1[dr]*g1[dc],g0[dr]*g2[dc]]
    diff=gray[None,:]-gray[:,None]
    b=np.array([(w*valid).sum(axis=1) for w in arrays])
    a=np.array([(w*valid*diff).sum(axis=1) for w in arrays])
    s=a[0]/b[0];sr=(a[1]-s*b[1])/b[0];sc=(a[2]-s*b[2])/b[0]
    expected=np.array([sr,sc,(a[3]-2*sr*b[1]-s*b[3])/b[0],
                       (a[4]-sr*b[2]-sc*b[1]-s*b[4])/b[0],
                       (a[5]-2*sc*b[2]-s*b[5])/b[0]])
    actual=native.gaussian_fields(gray,p,g0,g1,g2)
    assert np.allclose(actual,expected,rtol=2e-13,atol=2e-13)
    assert np.array_equal(actual,native.gaussian_fields(gray+100,p,g0,g1,g2))


def test_mask_cubic_interpolates_nodes_and_preserves_constant_fields():
    rng=np.random.default_rng(33);mask=rng.random((16,20))<.65
    pixels,index=index_map(mask);solver,mass=spline_system(pixels,index)
    values=rng.normal(size=(len(pixels),5))
    coefficients=np.ascontiguousarray(solver.solve(values*mass[:,None]))
    actual=native.sample_fields(coefficients,index,pixels.astype(float)).T
    assert np.allclose(actual,values,rtol=0,atol=2e-14)
    const=np.ascontiguousarray(solver.solve(np.ones_like(values)*mass[:,None]))
    queries=pixels.astype(float)+[.12,-.19]
    assert contains(index,queries).all()
    assert np.allclose(native.sample_fields(const,index,queries),1.,rtol=0,atol=2e-15)


def test_outside_roi_values_including_nan_have_no_effect_and_full_filters_are_unused(monkeypatch):
    from steger import detector
    image,mask=example()
    expected=measure_neck_mask_only(image,mask)
    assert expected['angle_deg'] is not None
    original=image.copy()
    def forbidden(*args,**kwargs):raise AssertionError('Full-image processing was used')
    monkeypatch.setattr(detector,'image_derivatives',forbidden)
    monkeypatch.setattr(detector,'spline_filter',forbidden)
    for outside in (np.nan,np.inf,-1e100,1e100):
        changed=image.copy();changed[~mask]=outside
        assert measure_neck_mask_only(changed,mask)==expected
    assert np.array_equal(image,original)
    invalid=image.copy();invalid[mask]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):prepare(invalid,mask)


def test_compact_worker_equals_serial_and_does_not_read_old_workspace_values():
    image,mask=example()
    expected=measure_neck_mask_only(image,mask)
    with MaskOnlyNeckAngleWorker(workers=2,max_shape=(64,64)) as worker:
        assert worker.measure(image,mask)==expected
        changed=image.copy();changed[~mask]=np.nan
        assert worker.measure(changed,mask)==expected
        assert worker.measure(np.full((24,24),170.),np.ones((24,24),bool))=={'angle_deg':None,'status':'no_line'}
        assert worker.measure(changed,mask)==expected


def test_returned_contour_points_stay_inside_the_mask():
    from steger.mask_only import analyze_mask_only
    roi,mask=example();_,_,records,g=analyze_mask_only(roi,mask)
    _,index=index_map(g['mask'])
    assert records
    assert all(contains(index,r.contour.points).all() for r in records)
