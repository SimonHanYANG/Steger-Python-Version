# Further optimization against v1.0.1

[English README](../README.md) · [中文 README](../README.zh-CN.md)

Branch `perf/angle-worker-v1.0.1` starts at the accepted `v1.0.1` release and
uses development package version `1.0.2.dev0`. It is separate from main and is
not a new stable release. The public `measure_neck(roi, mask)` interface remains
the same. [Paired runtime results and exact comparisons](../reports/performance-followup/README.md)
retain both full timing runs, including individual samples and maximum latency.

## What changed

- Batch duplicate-neighbor queries for point clouds of at most 512 points.
  Larger clouds retain sequential queries to bound temporary storage. Stable
  response priority and duplicate suppression decisions are retained.
- Compute each neighbor pair's distance and tangent projection once. A finite
  projection can pass the positive-advance gate in only one traversal direction.
  Its absolute value is exactly the previous projection multiplied by that
  direction's sign. The original norm, dot product, arccos, angle threshold and
  lexicographic `(distance + angle, point_index)` tie-breaking remain.
- Remove the redundant lower clipping operation after absolute cosine, retaining
  the upper bound of 1 and the original arccos. Normalize undirected edge keys
  with integer comparisons and avoid redundant copies of advanced-indexed arrays.
- Compute normal-orientation tangents with the same unit-spacing forward/backward
  endpoint and centered interior differences as `np.gradient`. The original
  right-facing normal orientation and sign decision are retained.
- Keep Taylor localization and candidate selection in compact mask-pixel arrays
  when trace capture is off. This avoids full-image zero fields and operations
  on excluded pixels. Gaussian filtering and cubic interpolation still use the
  entire original ROI. Trace mode retains full derivative/Hessian/Taylor maps.

All five search scales, both polarities, the three rescue scales when triggered,
float64, convolution support/borders, Newton iteration/stopping, thresholds,
selection rules, fit windows and quality checks are retained. There is no new
runtime dependency, compiler requirement, private SciPy API, downsampling or
approximate detector. Measurement fitting formulas are unchanged.

## Evidence

The reference is immutable commit `914b3aed74b11b00b073491d6e3b91f471190853`
(`v1.0.1`), loaded under a separate module name from Git objects. Checks compare
raw array bytes and scalar bits without angle tolerances:

- 29 real ROI/mask pairs, all 210 eligible search candidate contours, all selected
  full traces, angles, selection fields and quality/sensitivity diagnostics.
- Complete detector outputs for all 84 existing synthetic profiles and the circle.
- Synthetic ground-truth errors remain exactly the same as v1.0.1. See
  [the current accuracy report](../reports/accuracy-summary.md).
- The 137-test suite includes 28 additional frozen-reference checks covering
  duplicate/neighbor radii, angle thresholds, score ties, dense clouds, response
  priority, larger-cloud fallback, dtype/layout and normal-orientation behavior.

The new Git-backed tests require v1.0.1 history; they skip from a source archive
without Git objects. CI checks out full history so these checks run on Linux
and macOS. Real images remain local and are not required by CI.

Bitwise agreement is evidence for the tested inputs and environments. It is
distinct from a quantitative real-image ground-truth accuracy bound.

## Reproduce

From a full-history checkout with local `sample.png` / `sample_mask.png` pairs:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev python tools/benchmark_optimization.py \
  --reference v1.0.1 --input-dir neck_mask --rounds 30 --synthetic \
  --output reports/neck-analysis/performance-followup/optimization.json
uv run --frozen --extra dev python tools/benchmark_optimization.py \
  --reference v1.0.1 --input-dir neck_mask --rounds 30 \
  --output reports/neck-analysis/performance-followup/repeat.json
uv run --frozen --extra dev python tools/summarize_optimization.py \
  reports/neck-analysis/performance-followup/optimization.json \
  --repeat-report reports/neck-analysis/performance-followup/repeat.json \
  --output-dir reports/performance-followup
```

The measured dataset also includes one additional explicit pair, supplied with
`--pair ROI_PATH MASK_PATH` in both runs. The source and dataset hashes must
match before the summary accepts a repeat report.

Each run warms one persistent worker, interleaves versions in randomized order,
and times every call from resident raw ROI/mask arrays to angle/status. It includes
conversion, mask geometry, full search/rescue, linking, fitting and quality checks.
It excludes disk decoding, plotting, queue wait, segmentation and concurrent
segmentation load. Profiling runs separately. Every sample is retained.

Use the paired v1.0.1 control in these runs for the speedup calculation. Historical
absolute timings from other sessions reflect different machine conditions and
are not an additional speedup. First-call samples use fresh processes, excluding
imports and file reads. Application integration and concurrent segmentation can
change latency; these measurements are not hard real-time bounds.

For intermediate visualizations, the existing batch reports and
`lines_gauss(..., return_trace=True)` remain available. Full recorded contours
are checked against the actual compact fast-path candidates.
