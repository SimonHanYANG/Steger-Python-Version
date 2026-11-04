# Recorded Steger process

Synthetic dark Gaussian line - noise SD 3, asymmetric backgrounds, 8-bit

Figures and arrays below come from the same detector invocation. Fixed-length cyan axes show normals; amber axes show tangents.

![Recorded stages](process-overview.png)

![Derivatives](derivatives.png)

![Normal and tangent axes](normals.png)

![Inspect one actual candidate](point-inspection.png)

[Raw float64 arrays](process.npz) | [Parameters and point values](process.json)

Normals become ambiguous when the two Hessian eigenvalues have similar absolute magnitudes. A 90-degree rotation of a displayed normal gives its perpendicular tangent axis; it is not a separate curve fit.
