# How Steger estimates normals and subpixel line positions

This implementation can record the actual detector intermediates, rather than reconstructing a plausible explanation from the final contour.

## 1. Why the normal can be found before a contour exists

Treat a grayscale image as a surface. For a locally isolated line, intensity varies strongly across the line and more slowly along it. A bright line resembles a ridge and a dark line a valley. The gradient alone is unsuitable at the center because its component across the line is approximately zero there.

At the selected scale, Gaussian derivative filters estimate the first and second partial derivatives. In row/column coordinates, the Hessian is

$$
H=\begin{pmatrix}L_{rr}&L_{rc}\\L_{rc}&L_{cc}\end{pmatrix}.
$$

For any unit direction $v$, the second directional derivative is $v^T H v$. Because $H$ is symmetric, its orthonormal eigenvectors identify the principal directions of this second-order variation. Steger selects the eigenvector $n$ whose eigenvalue has the largest absolute magnitude as the local normal axis. A perpendicular vector gives the local tangent axis.

This is an estimate under the local line model. At crossings, blobs, heavy noise, or locations with similar absolute eigenvalues, the normal can be ambiguous. The process report includes the eigenvalues and their relative separation; a plotted axis alone does not establish reliability.

## 2. From a normal to a subpixel position

At an integer pixel center $p$, the local quadratic approximation along $n$ is

$$
L(p+s n)\approx L(p)+s\nabla L^T n+\tfrac12 s^2 n^T H n.
$$

The first directional derivative of that approximation vanishes at

$$
t=-\frac{\nabla L^T n}{n^T H n},\qquad p_{\rm sub}=p+t n.
$$

The implementation checks polarity, line response, and whether the located point belongs to the relevant pixel neighborhood. The optional Newton stage refines the location using interpolated derivative fields. The point-inspection figure distinguishes the initial Taylor estimate from this later location.

This position precedes line linking, width estimation, asymmetry correction, and optional fitting to the original grayscale profile. These stages can change coordinates; they are recorded separately.

## 3. Sign, angle, and the final normals

Both $n$ and $-n$ are valid eigenvectors for the same axis. Changing the sign of $n$ also changes the sign of $t$, so the physical offset $t n$ is unchanged. Linking orients normals consistently toward the right side of contour traversal. An arbitrary raw eigenvector sign must not be interpreted as a biological direction.

The final stored normal may be updated by the optional original-profile fit. The report explicitly separates the linked Hessian normals from the final normals. Cyan axis segments have fixed display length and are not width measurements. Amber segments are perpendicular tangent axes, not independently fitted anatomical axes.

## 4. Generate a report for an ROI and mask

```bash
uv run --extra visualization steger detect roi.tif --mask neck_mask.png \
  --sigma 1.5 --low 3 --high 8 --polarity dark --model gaussian \
  --no-complete-junctions --output results/result.json \
  --process-dir results/process
```

Use `--inspect-point ROW COL` to inspect the located candidate nearest a chosen location. With no location specified, an interior point on the longest linked contour is selected. If there are no located candidates, no point-inspection figure is generated.

The example parameters require adjustment for the actual image. The mask only restricts point selection and display bounds; it does not replace the surrounding grayscale values.

In Python:

```python
from steger import lines_gauss
from steger.visualization import save_process_report

result = lines_gauss(image, mask=neck_mask, return_trace=True)
save_process_report(result, "results/process", inspect_point=(40, 55))
```

Available outputs:

| File | Recorded content |
|---|---|
| `process-overview.png` | Input, smoothed image, gradient magnitude, dominant eigenvalue, response, initial candidates, subpixel candidates, linked contour with axes, and final contour |
| `derivatives.png` | Five derivative fields, the dominant and other Hessian eigenvalues, and their relative separation |
| `normals.png` | Linked Hessian normal/tangent axes and final normal/tangent axes |
| `point-inspection.png` | Local pixel grid, cross section, directional derivative zero crossing, and actual $H$, $\nabla L$, $n$, eigenvalue, and Taylor offset values |
| `process.npz` | Losslessly compressed arrays, candidate positions, Newton iterations, and linked/bias-corrected/final snapshots; load with `allow_pickle=False` |
| `process.json` | Parameters, stage counts, conventions, and inspected-point values |
| `README.md` | A navigable report with the figures |

Trace capture is optional because it retains large intermediate arrays. It is tested to leave output coordinates, normals, widths, and quality flags numerically identical to a run without recording. Plotting reads the recorded invocation and never reruns detection. Display sampling of axes does not downsample the saved arrays.

## Official references

- [HALCON lines_gauss: Gaussian derivatives, subpixel contours, polarity, and width correction](https://www.mvtec.com/doc/halcon/2411/en/lines_gauss.html)
- [HALCON contour attributes: normal orientation and angle convention](https://www.mvtec.com/doc/halcon/2411/en/get_contour_attrib_xld.html)
- [Steger 1998, An Unbiased Detector of Curvilinear Structures](https://doi.org/10.1109/34.659930)
