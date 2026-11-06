"""Exact interpolation and process-workspace lifetime/concurrency regression checks."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.ndimage import map_coordinates, spline_filter

from steger import measure_neck
from steger.synthetic import straight_line
from steger.worker import NeckAngleWorker


def case(angle=31, polarity="dark", shape=(40, 40)):
    image, gt = straight_line(shape=shape, model="gaussian", angle_deg=angle,
                              polarity=polarity, offset=.237, asymmetry=.35, quantize=True)
    delta = np.moveaxis(np.indices(image.shape), 0, -1) - gt["center"]
    tangent = np.array([-gt["normal"][1], gt["normal"][0]])
    mask = (abs(delta @ gt["normal"]) < 4) & (abs(delta @ tangent) < 15)
    return image, mask


@pytest.mark.parametrize("shape", [(3, 3), (17, 21), (64, 64)])
def test_native_cubic_reflect_is_bitwise_scipy_at_random_and_border_points(shape):
    native = pytest.importorskip("steger._spline_native")
    rng = np.random.default_rng(114)
    fields = np.stack([spline_filter(rng.normal(size=shape), order=3, mode="reflect")
                       for _ in range(5)])
    points = rng.uniform([-2 * shape[0], -2 * shape[1]],
                         [3 * shape[0], 3 * shape[1]], size=(2000, 2))
    points = np.vstack([points, [[0, 0], [-1e-16, -1e-16], [-1., 0], [0, -1.],
                               [-.5, -.5], [shape[0] - 1, shape[1] - 1]]])
    fields.flags.writeable = points.flags.writeable = False
    expected = np.array([map_coordinates(f, points.T, order=3, mode="reflect", prefilter=False)
                         for f in fields])
    actual = native.sample_fields(fields, points)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("trace", [False, True])
def test_native_newton_preserves_every_iteration_and_global_stopping(trace):
    native = pytest.importorskip("steger._spline_native")
    rng = np.random.default_rng(143)
    fields = np.stack([spline_filter(rng.normal(size=(17, 21)), order=3, mode="reflect")
                       for _ in range(5)])
    origins = np.column_stack([rng.integers(0, 17, 100), rng.integers(0, 21, 100)])
    origins = np.asfortranarray(origins)  # trace-mode np.argwhere layout
    points = origins + rng.uniform(-1.5, 1.5, origins.shape)
    normals = rng.normal(size=origins.shape)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    expected = points.copy()
    positions = []
    for _ in range(6):
        dr, dc, drr, drc, dcc = [map_coordinates(f, expected.T, order=3, mode="reflect",
                                               prefilter=False) for f in fields]
        a, b = normals[:, 0], normals[:, 1]
        denom = drr * a * a + 2 * drc * a * b + dcc * b * b
        step = np.divide(-(dr * a + dc * b), denom, out=np.zeros(len(points)),
                         where=np.abs(denom) > 1e-14)
        step = np.clip(step, -.25, .25)
        proposed = expected + step[:, None] * normals
        acceptable = np.all(np.abs(proposed - origins) <= 1.5, axis=1)
        expected[acceptable] = proposed[acceptable]
        if trace:
            positions.append(expected.copy())
        if np.max(np.abs(step), initial=0) < 1e-8:
            break
    actual, actual_positions = native.refine_points(fields, points.copy(), normals, origins, trace)
    assert actual.tobytes() == expected.tobytes()
    assert len(actual_positions) == len(positions)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(actual_positions, positions, strict=True))


def test_native_zero_newton_step_stops_after_one_iteration_and_rejects_nonfinite():
    native = pytest.importorskip("steger._spline_native")
    fields = np.zeros((5, 8, 8))
    origins = np.array([[0, 0], [7, 7]], dtype=np.intp)
    points = origins.astype(float)
    normals = np.ones_like(points)
    actual, positions = native.refine_points(fields, points.copy(), normals, origins, True)
    assert actual.tobytes() == points.tobytes()
    assert len(positions) == 1
    points[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        native.refine_points(fields, points, normals, origins)
    with pytest.raises(ValueError, match="finite"):
        native.sample_fields(fields, points)


def test_scale_worker_serializes_callers_and_preserves_inputs_across_shapes():
    first = case()
    second = case(61, "light", (36, 44))
    inputs = [(np.repeat(first[0][..., None], 3, axis=2), first[1]), second]
    inputs += [(inputs[0][0][::-1], inputs[0][1][::-1]), first]
    expected = [measure_neck(*c) for c in inputs]
    saved = [[a.tobytes() for a in c] for c in inputs]
    for c in inputs:
        for a in c:
            a.flags.writeable = False
    with NeckAngleWorker(workers=2, max_shape=(64, 64)) as worker:
        with ThreadPoolExecutor(max_workers=2) as callers:
            actual = list(callers.map(lambda c: worker.measure(*c), inputs))
        assert actual == expected
        assert [[a.tobytes() for a in c] for c in inputs] == saved
        assert worker.measure(np.zeros((20, 20)), np.ones((20, 20))) == {
            "angle_deg": None, "status": "no_line"}
        with pytest.raises(ValueError):
            worker.measure(np.zeros((65, 65)), np.ones((65, 65)))
        with pytest.raises(ValueError):
            worker.measure(first[0], np.zeros(first[0].shape))
        assert worker.measure(*first) == measure_neck(*first)
    worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        worker.measure(*first)


def test_failed_child_request_closes_workspace_before_reuse(monkeypatch):
    roi, mask = case()
    worker = NeckAngleWorker(workers=1)
    worker.start()

    def failed(*args, **kwargs):
        raise RuntimeError("test worker failure")

    monkeypatch.setattr(worker, "_candidates", failed)
    with pytest.raises(RuntimeError, match="test worker failure"):
        worker.measure(roi, mask)
    assert worker._closed
    worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        worker.measure(roi, mask)


@pytest.mark.parametrize("options", [{"workers": 0}, {"max_shape": (2, 64)},
                                     {"max_shape": (3,)}, {"workers": 1.5}])
def test_scale_worker_rejects_invalid_configuration_before_allocation(options):
    with pytest.raises(ValueError):
        NeckAngleWorker(**options)
