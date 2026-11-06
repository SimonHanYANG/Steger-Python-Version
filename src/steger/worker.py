"""Persistent CPU scale workers with a single shared input workspace.

Each request still evaluates every policy scale/polarity and rescue gate. This
reduces per-ROI latency using multiple CPU cores, not approximate computation.
Construct once behind the application's __main__ guard on Windows. No GUI, I/O,
cached image results or changes to float64 fitting/detection are involved.
"""

import atexit
import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from .neck import (
    RESCUE_SCALES,
    SCALES,
    _neck_result,
    _prepare_neck_inputs,
    candidate_runs,
    mask_geometry,
)

_shared_input = None


def native_acceleration_available():
    """Whether this interpreter loaded the optional compiled Newton sampler."""
    from .detector import _native_refine_points

    return _native_refine_points is not None


def fast_acceleration_available():
    """True only when all routines required for this experiment are loaded."""
    from . import linking, robust_measurement
    return native_acceleration_available() and all(function is not None for function in (
        linking._native_deduplicate, linking._native_link_paths,
        linking._native_orient, linking._native_neighbor_graph,
        linking._native_radius_neighbors, robust_measurement._native_window_samples,
        robust_measurement._native_polyline_arcs, robust_measurement._native_distinct_samples,
    ))


def _attach_workspace(name, capacity):
    global _shared_input
    _shared_input = (SharedMemory(name=name), capacity)
    atexit.register(_shared_input[0].close)


def _workspace_views(memory, capacity, shape):
    # Flatten the capacity first so each live HxW image is C contiguous.
    size = int(np.prod(shape))
    image = np.ndarray((size,), dtype=np.float64, buffer=memory.buf, offset=0).reshape(shape)
    distance = np.ndarray(
        (size,), dtype=np.float64, buffer=memory.buf, offset=capacity * 8
    ).reshape(shape)
    mask = np.ndarray(
        (size,), dtype=np.bool_, buffer=memory.buf, offset=capacity * 16
    ).reshape(shape)
    return image, distance, mask


def _run_scale(request):
    shape, axis, span, sigma = request
    memory, capacity = _shared_input
    image, distance, mask = _workspace_views(memory, capacity, shape)
    # No child writes the parent's workspace. Candidate/contour outputs are
    # independent arrays, serialized back before the next input can replace it.
    for array in (image, distance, mask):
        array.flags.writeable = False
    geometry = dict(mask=mask, axis=axis, span=span, distance=distance)
    return candidate_runs(image, geometry, (sigma,))


def _ready(_):
    return True


class NeckAngleWorker:
    """Compute a complete neck result using a reusable process pool.

    measure() accepts the same raw arrays and returns the same dictionary as
    measure_neck(). Concurrent callers serialize around one shared workspace;
    callers should bound their own submission queue. workers=5 distributes the
    five initial scales across five CPU processes. Startup is not cheap: create
    and warm once, outside the per-frame path. There is no hard latency bound.
    """

    def __init__(self, workers=5, max_shape=(256, 256)):
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if len(max_shape) != 2 or any(not isinstance(n, int) or n < 3 for n in max_shape):
            raise ValueError("max_shape must contain two integer dimensions >= 3")
        self.workers = workers
        self.max_shape = tuple(max_shape)
        self._capacity = int(np.prod(max_shape))
        self._memory = SharedMemory(create=True, size=self._capacity * 17)
        self._lock = threading.Lock()
        self._closed = False
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_attach_workspace,
            initargs=(self._memory.name, self._capacity),
        )

    def start(self):
        """Start the pool before accepting frames; warm actual inputs as needed."""
        with self._lock:
            self._check_open()
            list(self._executor.map(_ready, range(self.workers)))
        return self

    def _check_open(self):
        if self._closed:
            raise RuntimeError("NeckAngleWorker is closed")

    def _candidates(self, image, geometry, scales):
        requests = [
            (image.shape, geometry["axis"], geometry["span"], sigma) for sigma in scales
        ]
        # map retains policy scale order irrespective of completion order.
        return [c for group in self._executor.map(_run_scale, requests) for c in group]

    def measure(self, roi, neck_mask):
        with self._lock:
            self._check_open()
            image, mask = _prepare_neck_inputs(roi, neck_mask)
            if any(a > b for a, b in zip(image.shape, self.max_shape, strict=True)):
                raise ValueError(f"ROI shape {image.shape} exceeds max_shape={self.max_shape}")
            geometry = mask_geometry(mask)
            shared_image, shared_distance, shared_mask = _workspace_views(
                self._memory, self._capacity, image.shape
            )
            np.copyto(shared_image, image)
            np.copyto(shared_distance, geometry["distance"])
            np.copyto(shared_mask, geometry["mask"])
            try:
                candidates = self._candidates(image, geometry, SCALES)
                if not any(c["span_coverage"] >= 0.75 and c["length_px"] >= 15 for c in candidates):
                    candidates.extend(self._candidates(image, geometry, RESCUE_SCALES))
                return _neck_result(geometry, candidates)
            except BaseException:
                # A cancelled/failed request must not leave children reading a
                # workspace which a later request could overwrite. Fail closed.
                self._shutdown()
                raise

    def _shutdown(self):
        self._closed = True
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self._memory.close()
            self._memory.unlink()

    def close(self):
        with self._lock:
            if not self._closed:
                self._shutdown()

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()


class FastNeckAngleWorker(NeckAngleWorker):
    """Same full-scale calculation with direct persistent process connections.

    The input contract, result, locking, rescue and selection are unchanged.
    Construct once under __main__ and reuse, just like NeckAngleWorker.
    """

    def __init__(self, workers=5, max_shape=(256, 256)):
        from ._direct_pool import DirectPool

        super().__init__(workers=workers, max_shape=max_shape)
        self._executor.shutdown()
        self._executor = DirectPool(workers, _attach_workspace,
                                    (self._memory.name, self._capacity))
        atexit.register(self.close)

    def start(self):
        try:
            return super().start()
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            super().close()
        finally:
            atexit.unregister(self.close)
