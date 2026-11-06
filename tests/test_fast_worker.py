"""Direct-process lifetime, contention, geometry and native precision checks."""
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from steger import measure_neck
from steger.neck import mask_geometry
from steger.worker import FastNeckAngleWorker
from test_native_worker import case
from test_linking_equivalence import assert_linked_bits_equal


@pytest.mark.parametrize('workers', [1, 2, 5])
def test_direct_worker_order_contention_shapes_and_readonly_inputs(workers):
    inputs = [case(), case(61, 'light', (36, 44))]
    inputs += [(inputs[0][0][::-1], inputs[0][1][::-1])]
    expected = [measure_neck(*pair) for pair in inputs]
    saved = [[a.tobytes() for a in pair] for pair in inputs]
    for pair in inputs:
        for array in pair:
            array.flags.writeable = False
    with FastNeckAngleWorker(workers=workers, max_shape=(64, 64)) as worker:
        name = worker._memory.name
        with ThreadPoolExecutor(max_workers=3) as callers:
            actual = list(callers.map(lambda pair:worker.measure(*pair), inputs * 2))
        assert actual == expected * 2
        assert saved == [[a.tobytes() for a in pair] for pair in inputs]
        with pytest.raises(ValueError):
            worker.measure(inputs[0][0], np.zeros(inputs[0][1].shape))
        assert worker.measure(*inputs[0]) == expected[0]
        assert worker.measure(np.zeros((20,20)), np.ones((20,20))) == {'angle_deg':None, 'status':'no_line'}
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)
    worker.close()
    with pytest.raises(RuntimeError, match='closed'):
        worker.measure(*inputs[0])


def test_failed_direct_child_closes_all_workspace_handles(monkeypatch):
    worker = FastNeckAngleWorker(workers=2).start()
    name = worker._memory.name
    def fail_in_child(*args):
        return worker._executor.map(int, ['invalid integer'])
    monkeypatch.setattr(worker, '_candidates', fail_in_child)
    with pytest.raises(RuntimeError, match='Scale process failed'):
        worker.measure(*case())
    assert worker._closed
    assert all(p._closed for p in worker._executor.processes)
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)
    worker.close()


def test_unexpected_direct_process_exit_fails_closed():
    worker = FastNeckAngleWorker(workers=2).start()
    name = worker._memory.name
    worker._executor.processes[0].terminate()
    worker._executor.processes[0].join()
    with pytest.raises((OSError, EOFError, RuntimeError)):
        worker.measure(*case())
    assert worker._closed
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def test_failed_direct_initializer_releases_parent_workspace():
    worker = FastNeckAngleWorker(workers=1)
    name = worker._memory.name
    worker._executor.initializer = int
    worker._executor.initargs = ('invalid integer',)
    with pytest.raises(RuntimeError, match='invalid literal'):
        worker.start()
    assert worker._closed
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


@pytest.mark.parametrize('kind', ['full', 'border', 'middle', 'irregular'])
def test_cropped_binary_distance_is_bitwise_full_roi(kind):
    mask = np.zeros((41, 53), dtype=bool)
    if kind == 'full':mask[:] = True
    elif kind == 'border':mask[:15, 3:11] = True
    elif kind == 'middle':mask[11:30, 21:29] = True
    else:
        mask[3:35, 5:45] = True
        mask[5:23, 13:31] = np.random.default_rng(83).random((18,18)) > .4
    geometry = mask_geometry(mask)
    expected = ndimage.distance_transform_edt(np.pad(geometry['mask'],1))[1:-1,1:-1]
    assert geometry['distance'].tobytes() == expected.tobytes()


def test_native_arc_sampling_matches_numpy_on_irregular_and_strided_paths():
    native = pytest.importorskip('steger._spline_native')
    if not hasattr(native, 'window_samples'):pytest.skip('Rebuild optional extension')
    rng = np.random.default_rng(497)
    for count in (6,31,191):
        points = np.cumsum(rng.normal(size=(count*2,2)),axis=0)[::2]
        expected_arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        assert native.polyline_arcs(points).tobytes() == expected_arc.tobytes()
        expected_keep = np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-9]
        assert native.distinct_samples(points).tobytes() == expected_keep.tobytes()
        for trim in (0.,.05):
            starts = np.linspace(trim*expected_arc[-1],(.8-trim)*expected_arc[-1],27)
            offsets = np.linspace(0.,.2*expected_arc[-1],43)
            q = starts[:,None]+offsets
            samples = np.stack([np.interp(q,expected_arc,points[:,i]) for i in (0,1)],axis=-1)
            centered,travel = native.window_samples(points,expected_arc,starts,offsets)
            assert centered.tobytes() == (samples-samples.mean(axis=1,keepdims=True)).tobytes()
            assert travel.tobytes() == (samples[:,-1]-samples[:,0]).tobytes()


def test_native_linker_matches_frozen_release_on_many_small_clouds():
    try:
        source = subprocess.check_output(['git','show','v1.0.2:src/steger/linking.py'],
            cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('Release Git object required')
    namespace = {'__name__':'steger._released_linker', '__package__':'steger'}
    exec(compile(source,'v1.0.2/linking.py','exec'),namespace)
    from steger.linking import link_points
    for seed in range(100):
        rng = np.random.default_rng(seed)
        count = int(rng.integers(6,193))
        points = rng.normal(0,3,(count,2))
        normals = rng.normal(size=(count,2))
        normals /= np.linalg.norm(normals,axis=1)[:,None]
        response = rng.choice([.14,.15,.5],size=count)
        assert_linked_bits_equal(namespace['link_points'](points,normals,response,.15),
                                 link_points(points,normals,response,.15))
