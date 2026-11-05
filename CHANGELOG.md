# Changelog

## v1.0.2 — Python package 1.0.2

- Optimize duplicate queries and repeated neighbor geometry in the linker, preserving
  thresholds, score ties, topology and traversal order.
- Use the original unit-spacing gradient formulas for normal orientation.
- Keep masked Taylor fields compact without cropping convolution/interpolation.
- Add 28 frozen v1.0.1 boundary, topology, response-priority and dtype/layout checks.
- Record paired v1.0.1 timing runs and bitwise real/synthetic output comparisons.
- Use continuous local geometric direction ranges for neck bending, with bounded
  terminal consistency checks, arc-length windows and retained raw detections.
- Preserve the earlier endpoint and literal C++ rules as explicit comparison APIs.
- Add optional compiled float64 interpolation/Newton loops and a five-process,
  shared-input `NeckAngleWorker`; retain the Python/SciPy fallback.
- Add all-input diagnostic, timing and concise Chinese visualization report tools.
- Publish the accepted implementation on `main`, preserving the old main tip at
  `archive/main-before-v1.0.2-20261105` (94235b04db2ebb0714503069dff9eef0416cc3f4).
- Measured i5-12500 full ROI/mask-to-result latency: mean 5.59 ms, median 4.96 ms,
  P95 9.60 ms, 29 real input pairs and 1,740 calls; native module + 5 processes.

See [v1.0.2 release notes](docs/releases/v1.0.2.md) for installation and validation.

## v1.0.1 — Python package 1.0.1

- Add `measure_neck(roi, mask)` for resident-array adaptive angle measurement,
  including the accepted v1.0 quality and sensitivity diagnostics.
- Share per-scale ridge fields between polarities, restrict independent Hessian
  eigenproblems to the mask when tracing is off, and reuse contours/angle fits.
- Preserve float64, full scales/rescue, Newton rules, linking, selection and windows.
- Add isolated v1.0 bitwise regression checks and interleaved worker benchmarks.
- The project owner accepted the accelerated real-image results. Publish directly
  from `perf/angle-worker-v1.0`; `main` and the v1.0 tag remain unchanged.
- Apple M1 mean runtime: 58.76 → 26.52 ms; independent repeat 53.56 → 23.64 ms.
  Full adaptive ROI/mask computation, with all timing samples retained.

See [v1.0.1 release notes](docs/releases/v1.0.1.md) for validation and installation.

## v1.0 — Python package 1.0.0

First stable release. The project owner reviewed the current real-image results,
accepted the algorithm's measurement behavior, and approved it as the v1.0 baseline.
The numerical detection and measurement implementation is preserved in this release.

- Float64 Gaussian derivatives, Hessian normals, subpixel line localization,
  directional linking, width estimation, and model-based asymmetry correction.
- Optional original-profile refinement, with explicit fit validity diagnostics.
- Neck self-bending measured from directed end-window fits, with sensitivity
  checks and separate diagnostic values for incomplete or short segments.
- Recorded derivatives, Hessian eigenvalues, normals, candidate points, subpixel
  steps, and contour snapshots, with process figures and lossless array exports.
- Local batch reports, CSV export, per-image overlays, and visualization archives.
- Resident-data and persistent-worker runtime benchmarks.
- English README with a Simplified Chinese language switch.
- GitHub release packaging with wheel, source distribution, and SHA-256 checksums.

See [the release notes](docs/releases/v1.0.md) for validation scope, installation,
and the reference runtime measurement. Owner acceptance is distinct from a
quantitative real-image ground-truth or native HALCON equivalence measurement.
