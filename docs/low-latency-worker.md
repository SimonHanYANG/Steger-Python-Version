# Exact-output low-latency neck worker

Current measurement note: the default angle rule has subsequently changed to
the [robust geometric direction range](robust-neck-angle.md). The frozen
endpoint-fit parity and timings described below document the earlier optimization
stage; use the newly generated robust-angle report for current values.

This extends `perf/angle-worker-v1.0.1` without changing its measurement policy.
The frozen comparison baseline is commit
`aeb2feb2cbe6e0b8601b3e0480a2a64784a63140`.

## What is accelerated

1. Batch neighbor-edge geometry, preserving thresholds, score/index tie breaks,
   edge insertion order, topology and returned contour order.
2. Optional Cython extension: share cubic interpolation weights across the five
   derivative fields and execute the original Newton updates in compiled loops.
   SciPy still supplies the original float64 convolution and spline coefficients.
   The six-iteration cap, global stopping rule, step cap, border reflection,
   originating-pixel gate, arithmetic order and trace positions remain.
3. Optional `NeckAngleWorker`: assign independent search scales to five persistent
   CPU processes. Pass the input once through a shared-memory workspace instead
   of serializing the complete image for each scale. Collect candidates in the
   original policy order, then apply the same rescue, selection and quality rules.

There is no downsampling, float32, skipped polarity, reduced scale search,
changed threshold, approximate interpolation, mask skeleton substitution or
reuse of a previous image's parameters/answer. The original serial
`measure_neck()` API remains available. Float64 numerical equivalence must still
be revalidated when changing compiler, dependencies or deployment hardware.

## Local build and integration

```powershell
uv sync --extra dev --extra native
uv run --extra dev --extra native python tools/build_native.py
```

On Windows this requires MSVC C++ Build Tools. Linux/macOS require a C compiler.
The build uses strict floating-point settings: `/fp:strict` on MSVC, no fast-math
and no FP contraction on GCC/Clang. The compiled extension is specific to the
Python ABI and platform. Build in the application's Python environment, then
restart the Python process so it loads the new extension.

The ordinary project installer does **not** automatically compile or distribute
this optional extension. When packaging an application, include the compatible
extension binary and THIRD_PARTY_NOTICES.md, or run the local build as a deployment
step. Without the extension the public API falls back to Python/SciPy; do not
assume the measured low-latency configuration is active merely because import
succeeds. `STEGER_DISABLE_NATIVE=1` explicitly selects the reference sampler.

```python
from multiprocessing import freeze_support
from steger.worker import NeckAngleWorker, native_acceleration_available

def main():
    if not native_acceleration_available():
        raise RuntimeError("Build the optional extension for the low-latency configuration")

    # Create once, before processing frames; warm representative actual inputs.
    with NeckAngleWorker(workers=5, max_shape=(256, 256)) as angle_worker:
        for roi_rgb, neck_mask in incoming_rois():
            result = angle_worker.measure(roi_rgb, neck_mask)
            # angle_deg can be None; review_required must remain visible.
            consume_result(result)

if __name__ == "__main__":
    freeze_support()
    main()
```

`incoming_rois()` and `consume_result()` above are application integration points,
not package functions. Inputs are aligned RGB(A) or grayscale ROI and nonzero
neck mask, preserving the original 0..255 scale. OpenCV BGR needs explicit
conversion upstream. Do not overwrite an input buffer while measure() is
reading/copying it. Queue length and ownership belong to the application.

On Windows, the `__main__` guard is mandatory for process spawning. An existing
angle thread can call `measure()`; create one worker object and reuse it. Calls
on the same object serialize around one shared workspace. This is **five child
processes for one ROI's latency**, not five simultaneous independent ROI workers.
The parent also performs mask geometry and final assessment. Budget CPU cores
and several hundred MB of process memory; concurrent segmentation can change
latency. Initialization/import/spawn and warmup are separate from steady-state
latency. Close explicitly or use the context manager.

## Validation and measurement

```powershell
$env:PYTHONUTF8='1'
uv run --extra dev --extra native pytest -q
uv run --extra dev --extra native python tools/benchmark_scale_worker.py `
  --rounds 60 --synthetic `
  --output reports/neck-analysis/opt-5ms-20261105/paired.json
```

The benchmark loads the immutable baseline from Git objects, checks every real
candidate and selected full trace, checks actual process-returned contours and
fits, and compares every complete returned result dictionary bit for bit. The
synthetic option additionally checks 84 full-detector cases and a circle against
the baseline. Native tests include random and boundary spline coordinates and
every Newton iteration. Workspace tests exercise different shapes, concurrent
callers, invalid input, read-only inputs and failure cleanup.

Timers include the complete resident raw ROI/mask to result computation,
shared-memory copies, IPC, synchronization, selection, fits and diagnostics.
They exclude disk I/O, PNG decoding, segmentation, plots and startup. One warmed
parent angle thread runs all modes in randomized paired order; no segmentation
load or queue backlog is simulated. Retain the raw samples and report the median,
P95, maximum and fraction below 5 ms, not just the mean. Matching the existing
algorithm on tested inputs does not establish a new real-image accuracy bound.
