# Changelog

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
