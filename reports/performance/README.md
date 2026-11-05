# ROI/mask worker optimization

Measured UTC: 2026-11-04T08:23:50.502983+00:00. Platform: macOS-13.0.1-arm64-arm-64bit.

Raw resident ROI/mask to angle+status; full adaptive search/rescue and quality checks; one persistent worker; interleaved randomized calls; no I/O, plots, queue wait or concurrent segmentation; warmed timings, not process startup.

Both versions receive the same **29 real ROI/mask pairs**; **26** return a full angle. Real images, per-image measurements and identities stay local.

| Version | Calls | Mean (ms) | Median (ms) | P95 (ms) | Maximum (ms) |
|---|---:|---:|---:|---:|---:|
| v1.0 | 870 | 58.758 | 50.220 | 92.959 | 482.479 |
| optimized | 870 | 26.519 | 22.020 | 49.791 | 578.809 |

Mean speedup: **2.22×**. These measurements describe this machine and dataset, not a latency guarantee for other images or concurrent workloads. Profiling was performed separately, outside the reported timing samples.

Occasional long calls remain in both versions. The optimized maximum exceeds the reference maximum in this run; the cause of those individual stalls has not been isolated. Every sample is retained, without trimming or capping. This worker does not provide a hard real-time latency bound.

![Runtime comparison](runtime-comparison.png)

## Numerical equivalence

Reference commit: `93a481e62f25b853d6dd8c40b62e910100facec6` (accepted v1.0). The tool loads its source from immutable Git objects under a separate package name.

Bitwise comparisons passed for 29 real inputs, 210 actual search candidate contours and 29 complete intermediate traces. They include coordinates, normals, responses, selected scale/polarity, fit diagnostics, quality flags and statuses. No tolerance was used to hide changed outputs.

Synthetic full-detector comparisons: 84 straight cases; 1 circle. Ground-truth acceptance is recorded separately in [the accuracy report](../accuracy-summary.md). Equivalence to v1.0 on tested inputs is distinct from real-image ground-truth accuracy.

Source SHA-256: `6b4f4b7b031c1a3dee2c259ad588cede38ccbd1b528dda0b5fdb1c380e536bda`.

Input-array dataset SHA-256: `183ef8f8070a4d455ef5c8b617c7d3db5fc7960073f1b80130f6e0f380ee7e9d`.

## What changed

- Share full-ROI Gaussian derivatives, Hessian results, Taylor offsets and lazy cubic spline coefficients between dark and light detection at each scale.
- Solve independent 2×2 Hessian eigenproblems only inside the candidate mask when trace capture is off. Convolution and interpolation still use the entire original ROI; trace mode retains the full original fields.
- Retain the search contour instead of running the winning detector again. Reuse identical angle fits while retaining length validation and sensitivity checks.

Float64, all five search scales, all three rescue scales, both polarities, Newton iteration rules, linking, selection rules, end windows and quality gates are preserved. No new runtime dependency or compiled extension was introduced.

## First-call check

Each value below is one first computation in a separate fresh process on the first dataset input. Imports, file decoding and thread startup are outside the timer. These single samples are not steady-state means.

- v1.0: 74.697 ms.
- optimized: 33.685 ms.

The raw [aggregate JSON](optimization.json) retains environment details and separate profiler summaries. See [reproduction and API notes](../../docs/performance.md).

## Independent repeat

A fresh process repeated the same real-input parity checks and timing after the long outliers above were observed. Source, dataset, environment and returned-angle count match the primary run. Synthetic comparisons are recorded in the primary run. The repeat does not replace or remove primary-run samples.

| Version | Calls | Mean (ms) | Median (ms) | P95 (ms) | Maximum (ms) |
|---|---:|---:|---:|---:|---:|
| v1.0 | 870 | 53.563 | 48.089 | 80.992 | 127.300 |
| optimized | 870 | 23.641 | 20.835 | 38.436 | 97.610 |

Repeat mean speedup: **2.27×**.
