# Steger Python Version

**English** | [简体中文](README.zh-CN.md)

**Stable release: [v1.0](https://github.com/SimonHanYANG/Steger-Python-Version/releases/tag/v1.0)** · Python package `1.0.0` · [Release notes](docs/releases/v1.0.md)

A precision-oriented Python implementation of Steger's curvilinear structure detector. It processes grayscale regions of interest (ROIs) and uses segmentation masks to select target structures. No HALCON installation or graphical interface is required.

The implementation follows the Gaussian derivatives, Hessian-based subpixel localization, directional linking, width estimation, and asymmetry bias correction described by Steger (1998). Its parameters and output attributes reference the official HALCON `lines_gauss` documentation.

**Accuracy scope:** This is an independent implementation, not a wrapper around HALCON's native libraries. An additional fit to the original grayscale profile is enabled by default to reduce pixel sampling error. The project owner reviewed the current real-image results and accepted this algorithm as the v1.0 baseline. Synthetic ground-truth validation, quantitative real-image accuracy validation, and agreement with HALCON outputs remain separate evaluations; the latter two have not been established.

## Visual examples

The following public examples use synthetic data with known geometry. The reviewed real-image results remain local.

![Grayscale ROI, segmentation mask, extracted centerline, and bending measurement](reports/visualizations/neck-bending-demo.png)

The green curve is the extracted centerline. Cyan and amber indicate the fitted end directions. The measured angle is **62.9995 degrees**, compared with a known window direction difference of **63.0000 degrees**.

![Noisy dark-line extraction and signed centerline error](reports/visualizations/noisy-line-demo.png)

For this asymmetric dark Gaussian line with contrast 150, noise standard deviation 3, and 8-bit quantization, the final P95 position error is **0.0342 px**. The error plot also shows the result before the additional grayscale profile fit; these are outputs of this implementation, not HALCON measurements.

Regenerate these figures with:

```bash
uv run --extra dev python tools/visualize_results.py
```

## Inspect the intermediate process

Steger estimates a normal axis from the Hessian eigenvector with the largest absolute eigenvalue. It then finds a subpixel line point by locating a zero of the first directional derivative along that normal. This calculation precedes contour linking and bias correction.

![Recorded Hessian normals and final normals](reports/process/circular-neck/normals.png)

Cyan segments show normal axes, amber segments show perpendicular tangent axes, and green shows the centerline. Their display lengths are fixed, not measured widths. These are synthetic examples. The [complete recorded report](reports/process/circular-neck/README.md) also includes derivative maps, eigenvalues, candidate selection, and a single-point inspection with the actual Hessian, gradient, and subpixel offset.

Generate the same report for an ROI and mask:

```bash
uv run --extra visualization steger detect roi.tif --mask neck_mask.png \
  --sigma 1.5 --low 3 --high 8 --polarity dark --model gaussian \
  --no-complete-junctions --output results/result.json --process-dir results/process
```

Add `--inspect-point ROW COL` to inspect the nearest located candidate. The report saves lossless arrays in `process.npz` as well as figures. In Python, use `lines_gauss(..., return_trace=True)` followed by `save_process_report` from `steger.visualization`. Recording is optional and tested to preserve detector outputs exactly. See [the algorithm and process guide](docs/steger-process.md).

## Batch analysis of paired neck images

For original 8-bit PNG ROIs named `sample.png` with aligned `sample_mask.png` files:

```bash
uv run --extra visualization python tools/analyze_necks.py neck_mask \
  --output reports/neck-analysis/results
```

The local report includes a filterable `index.html`, `angles.csv`, individual measurement overlays, Hessian/normal/subpixel process reports, parameter sensitivity plots, and `visualizations.zip`. The script searches bright and dark ridges at five declared scales, selects by mask extent and a reference scale rather than angle, and records every candidate. If that search yields insufficient extent or length, it checks three larger scales; any selected larger scale requires review for smoothing effects. It measures the visible neck's self-bending using directed fits over the 5–25% and 75–95% arclength windows; straight is 0 degrees.

Incomplete or short centerlines receive no full-neck angle: their local diagnostic angle is stored separately. The primary angle measures end-direction change; reverse bends in an S-shaped neck can cancel. A separate cumulative-turn diagnostic is exported, but it is sensitive to local localization fluctuations. Extent/length checks and sensitivity flags are screening rules, not validated accuracy or confidence bounds. Width bias correction and the additional original-profile fit are disabled in this batch policy because those models have not been validated for the supplied imaging modality. Review masks, centerline placement, crop boundaries, and sensitivity before interpreting estimates. Input images and reports in the default real-data folders are excluded from Git.

To measure computation time on an existing local batch, with all input decoding and disk I/O outside the timers:

```bash
uv run --extra visualization python tools/benchmark_neck_runtime.py
```

The benchmark verifies saved angles and statuses, warms the functions, and separately measures angle calculation from resident contours, one fixed-parameter ROI-to-angle run, and the automatic search with quality checks. It exports mean/median/P95 timings, per-image summaries, raw timing samples, and machine/library details to `reports/neck-analysis/runtime/`. Diagnostic-only images are excluded from the angle-only statistics. These timings depend on the machine and input data.

For the deployment boundary where a persistent angle worker receives only raw in-memory ROI and mask arrays, run:

```bash
uv run --extra visualization python tools/benchmark_roi_worker.py
```

This includes RGB conversion, mask processing, Steger extraction, contour selection, and angle calculation inside one worker thread. The adaptive mode reproduces the batch quality/search behavior without cached per-image parameters or contour indices. A separate fixed-global-scale mode reports speed, returned-angle counts, and differences from the adaptive results. Processing time and submit-to-result latency are reported separately in `reports/neck-analysis/worker-runtime/`; segmentation and simultaneous segmentation load are excluded.

## Installation

The project uses uv and Python 3.12:

```bash
uv sync --extra dev
uv run steger --help
```

Runtime dependencies are NumPy, SciPy, and Pillow. Development checks additionally use pytest. Dependencies are pinned in `uv.lock`. Localization, convolution, and fitting use `float64`; input intensities are not automatically normalized, and pixel dimensions are preserved.

## Python API

```python
from steger import lines_gauss, measure_bending
from steger.io import read_image, write_json

image = read_image("roi.tif")
mask = read_image("neck_mask.png", mask=True)

result = lines_gauss(
    image,
    sigma=1.5,
    low=3.0,
    high=8.0,
    light_dark="dark",      # Use dark for a neck darker than its background; otherwise light.
    extract_width=True,
    line_model="gaussian",  # Choose from the observed profile; not all microscopy images are Gaussian.
    complete_junctions=False,
    mask=mask,
)

write_json("result.json", result.to_dict())

# Measure a single confirmed neck contour. Inspect the overlay before selecting
# among multiple contours; do not join unrelated contours automatically.
if len(result.contours) == 1:
    neck = result.contours[0]
    measurement = measure_bending(neck)
    print(measurement.angle_deg)
```

These example parameters are not universal settings for sperm images. For intensities in `[0,1]`, adjust the thresholds accordingly; do not reuse `Low/High` values intended for `[0,255]` without rescaling them.

## Command line

```bash
uv run steger detect roi.tif --mask neck_mask.png \
  --sigma 1.5 --low 3 --high 8 --polarity dark --model gaussian \
  --no-complete-junctions --output result.json --overlay overlay.png
```

Detect and measure a single neck:

```bash
uv run steger detect roi.tif --mask neck_mask.png \
  --sigma 1.5 --low 3 --high 8 --polarity dark --model gaussian \
  --no-complete-junctions --measure-neck --output neck.json --overlay neck.png
```

Inputs include Pillow-readable images such as PNG/TIFF and `.npy` arrays. RGB images are explicitly converted to luminance by the I/O layer; the core API accepts single-channel images only. The original intensity range is preserved for 16-bit and floating-point grayscale inputs. Masks must match the ROI's dimensions and coordinates; no automatic resizing or registration is performed.

**The mask limits candidate centerline locations without setting the original image outside the mask to black.** Convolution and profile fitting retain the surrounding grayscale data to avoid artificial boundaries affecting subpixel measurements. Contrast adjustment in overlays is for display only.

## Implemented features

| Feature | Implementation |
|---|---|
| Bright and dark lines | Polarity selection using the Hessian eigenvalue with the largest absolute magnitude |
| Gaussian derivatives | Separable, integrated Gaussian kernels from the paper, preserving derivative amplitudes |
| Subpixel localization | Taylor localization with optional Newton refinement, enabled by default to handle ridges between pixels |
| Hysteresis linking | Direction constraints, strong seeds, and weak continuation; closed loops and junction topology are retained |
| Line width | Subpixel searches for gradient maxima on both sides of the normal; missing widths are interpolated along arclength |
| Asymmetry correction | Continuous `bar-shaped`, `parabolic`, and `gaussian` models with numerical bias inversion |
| Accuracy refinement | Optional pixel-area-integrated profile fitting to the original grayscale data |
| Junction completion | Conservative endpoint extension and target contour splitting; inferred points are explicitly flagged |
| Neck bending | Orthogonal line fits within fixed arclength windows; a straight neck measures 0 degrees |
| Accuracy validation | Independent continuous geometric image generation, error reports, and regression tests |

Contours contain `points` (row, column), right-facing `normals` relative to traversal, `response`, `width_left`, and `width_right`. When correction is enabled, they also contain `asymmetry` and `contrast`. Following the HALCON documentation, the `angle` attribute is measured counterclockwise from the positive row axis, in radians. **It describes the normal direction, not the bending angle.**

`width_left/right` are half-widths. For the Gaussian model, the width parameter is its standard deviation, not FWHM. For the parabolic and bar-shaped models, it is the half-support width. Convert to a consistent width definition before comparing different models.

## Quality flags

Inspect quality indicators alongside the coordinates and angles:

- `width_measured` / `width_interpolated`: whether widths came from observations or interpolation.
- `correction_valid` / `correction_residual`: whether continuous bias inversion succeeded.
- `profile_fit_valid` / `profile_fit_rms` / `profile_fit_condition`: whether fitting to the original grayscale data passed its checks.
- `junction_completed`: whether a point was inferred by junction completion; inferred points must not be treated as observations when measuring angles.
- `steger_row/col` and `steger_width_left/right`: corrected results before the additional refinement, retained for inspection.

If correction or fitting fails, the preceding available result is retained and the failure is flagged. Undefined numeric values are written as `null` in JSON.

## Accuracy tests and reports

```bash
uv run --extra dev pytest -q
uv run python tools/benchmark.py --output reports/accuracy.json
uv run --extra dev python tools/summarize_benchmark.py reports/accuracy.json
```

The full benchmark contains 84 straight-line cases and one circular arc. It covers three profile models, multiple widths and orientations, non-integer shifts, symmetric and asymmetric backgrounds, bright and dark lines, 8-bit quantization, and seeded random noise. Images are generated by integrating independent continuous models over pixel areas, rather than treating thresholded drawings as subpixel ground truth. Pixel areas are computed exactly for bar-shaped profiles; other profiles use Gauss-Legendre quadrature.

Predeclared acceptance criteria:

| Sample conditions | Centerline error P95 | Half-width error P95 | Original grayscale fit acceptance |
|---|---:|---:|---:|
| Noiseless, matching model | <= 0.01 px | <= 0.025 px | >= 95% |
| Contrast 150, noise standard deviation 1, 8-bit quantization | <= 0.08 px | <= 0.12 px | >= 95% |
| Contrast 150, noise standard deviation 3, 8-bit quantization | <= 0.20 px | <= 0.30 px | >= 95% |
| Circular arc of a specified radius | Radial error P95 <= 0.01 px | Direction-difference error between the two measurement windows <= 0.1 degrees | Reported separately |

Every case is retained, including detection failures, errors before and after refinement, fit acceptance, and execution time. See the [readable summary](reports/accuracy-summary.md) and [complete numeric report](reports/accuracy.json). The report records the implementation source files' SHA-256 hash so the tested code can be identified. These criteria apply to the specified synthetic conditions, not to arbitrary microscopy images.

## Running closer to the original paper

```python
result = lines_gauss(
    image, sigma=1.5, low=3, high=8,
    profile_refinement=False,  # Disable the additional fit to original grayscale data.
    refine=False,              # Use the paper's single Taylor localization step.
)
```

This mode is not a bitwise clone of HALCON. Numerical inversion, edge interpolation, junction completion, and boundary handling have independent implementation details. The default mode prioritizes reducing measured localization errors rather than reproducing undocumented HALCON internals.

## Optional HALCON comparison

HALCON is not required to run this project. If reference outputs become available, the standalone `tools/export_halcon.py` script can be run on a licensed Windows/Linux installation, followed by:

```bash
uv run steger compare candidate.json reference.json --output comparison.json
```

Comparison uses bidirectional point-to-segment distances so differences in contour sampling spacing are not mistaken for localization errors. Coverage is also reported so a short, accurately detected fragment cannot stand in for the complete target. Passing the geometry comparison does not establish agreement in every attribute or in topology.

## Neck bending angle definition

By default, the first and last 5% of arclength are excluded. A segment covering 20% of total arclength is fitted near each end. After aligning the traversal directions, their deflection angle is measured in `[0,180]` degrees, with a straight neck measuring 0 degrees. These parameters are configurable and should be kept consistent across samples. Outputs include the window locations, fitting residuals, and cumulative turning angle.

The measurement is the difference between two fitted segment directions, not the instantaneous tangent angle at the mathematical endpoints. For S-shaped contours, also inspect cumulative turning, which is sensitive to noise and sampling scale. Direct angle measurement is rejected for necks that are too short, closed contours, or contours containing inferred junction points.

## References and scope

- [HALCON 24.11 `lines_gauss` documentation](https://www.mvtec.com/doc/halcon/2411/en/lines_gauss.html)
- [HALCON `get_contour_attrib_xld` attribute definitions](https://www.mvtec.com/doc/halcon/2411/en/get_contour_attrib_xld.html)
- [HALCON `derivate_gauss` documentation](https://www.mvtec.com/doc/halcon/2411/en/derivate_gauss.html)
- C. Steger, *An Unbiased Detector of Curvilinear Structures*, TPAMI 20(2):113-125, 1998. [DOI](https://doi.org/10.1109/34.659930)
- C. Steger, *Unbiased extraction of lines with parabolic and Gaussian profiles*, CVIU 117(2):97-112, 2013. [DOI](https://doi.org/10.1016/j.cviu.2012.08.007)

See the [compatibility and accuracy notes (Chinese)](docs/accuracy-and-compatibility.md) for detailed mappings and known limitations.
