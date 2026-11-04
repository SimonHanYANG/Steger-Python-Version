# Precision-preserving worker optimization

[English README](../README.md) · [中文 README](../README.zh-CN.md)

Release `v1.0.1` (package `1.0.1`) publishes the accepted optimization from
`perf/angle-worker-v1.0`. It is tagged directly from that branch, with `main`
unchanged. The v1.0 tag remains the numerical reference. [Runtime and numerical comparisons](../reports/performance/README.md)
describe the tested machine and input set.

## Array input API

```python
from steger import measure_neck

# Aligned NumPy arrays supplied by the segmentation worker.
result = measure_neck(roi, mask)
angle_deg = result["angle_deg"]  # None for missing/incomplete/short segments
status = result["status"]
if angle_deg is not None:
    print(angle_deg, result["quality_flags"])
```

Accepts grayscale or RGB(A) ROIs and aligned single-channel or RGB(A) nonzero
masks. Preserve original intensities, normally 0–255 for this fixed policy. The
function does not resize or register inputs, read files, draw images, or retain
image-dependent state between calls. Standard runtime dependencies suffice;
matplotlib is not imported. A persistent application thread can call it once per
ROI/mask pair. Scheduling and segmentation remain the application's responsibility.

Detected segments include selected scale/polarity, coverage, length, centrality,
fit RMS, cumulative turn, scale/window sensitivity, component metadata and quality
flags. Missing lines return only `angle_deg=None` and `status="no_line"`;
incomplete/short detected segments retain diagnostics with `angle_deg=None`.
Check status and quality flags alongside an angle.

This is the existing neck-specific policy, not a replacement for the general
`lines_gauss` detector or a claim that its fixed thresholds suit all imaging
modalities. Width correction and profile fitting remain disabled in this policy,
as in the accepted v1.0 real-image analysis.

## Changes and invariants

1. Prepare full-ROI Gaussian derivatives once per scale for both polarities.
   Reuse Hessian eigenvectors, Taylor offsets and lazy cubic spline coefficients.
   Keep separate dark/light Newton loops and their original convergence tests.
2. Without trace capture, solve independent 2×2 Hessian eigenproblems only at
   mask-selected pixels. Convolutions and spline interpolation still retain the
   entire original ROI. No cropping, background zeroing, reduced precision or
   smaller convolution support is introduced.
3. Retain actual candidate contours and local fits. Use the winner directly
   instead of rerunning detection, and reuse identical fits after validating
   the original minimum-window requirement.

All five scales, both polarities, three rescue scales when triggered, thresholds,
float64 arithmetic, Newton updates, linking, tie-breaking, selection, end windows
and quality gates are retained. Linking and measurement formulas are unchanged.
The ordinary detector retains its existing input validation and API.

For intermediate visualizations, use the existing batch script or
`lines_gauss(..., return_trace=True)` with `save_process_report`. Trace mode
retains full original image fields and the actual Newton history. Full traces
and fast-path contours are compared to v1.0 separately. Visualization is outside
angle-worker timing.

## Reproduce

Run from a checkout containing v1.0 history and local inputs named `sample.png`
and `sample_mask.png`:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev python tools/benchmark_optimization.py \
  --input-dir neck_mask --rounds 30 --synthetic \
  --output reports/neck-analysis/performance/optimization.json
uv run --frozen --extra dev python tools/summarize_optimization.py \
  reports/neck-analysis/performance/optimization.json
uv run --frozen python tools/benchmark.py --output reports/accuracy.json
uv run --frozen --extra dev python tools/summarize_benchmark.py reports/accuracy.json
```

Supply additional explicit pairs with `--pair ROI_PATH MASK_PATH`. The reference
is commit `93a481e62f25b853d6dd8c40b62e910100facec6`, loaded from Git objects into
a temporary directory under an isolated module name. Missing history fails the
run. Assertions must be enabled; `python -O` is rejected. Both versions use the
same interpreter/libraries, comparing raw array bytes and scalar bits without
angle tolerances.

The published report also retains an independent 30-round timing repeat in a
fresh process. To reproduce both runs and publish their aggregates together:

```bash
uv run --frozen --extra dev python tools/benchmark_optimization.py \
  --input-dir neck_mask --rounds 30 \
  --output reports/neck-analysis/performance/repeat.json
uv run --frozen --extra dev python tools/summarize_optimization.py \
  reports/neck-analysis/performance/optimization.json \
  --repeat-report reports/neck-analysis/performance/repeat.json
```

The local measurements included one additional explicit pair in both runs. A
matching input-array dataset hash is required before publishing a repeat beside
the primary run. All timing samples, including long outliers, are retained.

Checks include raw-array worker outputs, every eligible search candidate's
coordinates/normals/responses, selection, all assessment fields and selected full
intermediate traces. `--synthetic` compares complete detector outputs for the
existing 84 profiles and circular neck. Pytest also checks masked/full-field
agreement across border modes, empty/disconnected masks, both polarities,
single-Taylor/Newton paths, concurrency, input immutability and all rescue scales.

Timing warms one persistent worker and interleaves versions in randomized order.
It starts inside the worker before array conversion, ending after angle/status
construction. File decoding, plots, queue wait, segmentation and concurrent
segmentation load are excluded. P95 describes individual calls, not a confidence
bound. First-call samples use separate fresh processes, excluding imports and I/O.

The local report includes per-image measurements. The summary publishes only
aggregates, hashes and profiler summaries; real images and per-image results
stay local. Rerun equivalence and timing on other platforms/library versions.
Tested bitwise agreement is evidence for these inputs/environment, not a proof
for every image or a real-image ground-truth accuracy bound.

The branch target was mean ≤30 ms and P95 ≤50 ms on the available Apple M1 data.
Integration under a real segmentation workload remains a separate measurement;
that workload is not supplied here.

The 109-test regression suite passed locally and on both Linux and macOS in
[GitHub Actions](https://github.com/SimonHanYANG/Steger-Python-Version/actions/runs/41798326605).
The real-data bitwise comparisons and runtime measurements use the local Apple M1;
the private real inputs are not uploaded to CI.

The timing/equivalence reports retain the exact development-source hash used
for measurement. Release preparation changes only `__version__` in the Python
package sources; a [release validation manifest](releases/v1.0.1-validation.json)
records the correspondence to that accepted implementation. Synthetic accuracy
is rerun with the release version and records its current source hash.
