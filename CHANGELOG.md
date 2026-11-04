# Changelog

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
