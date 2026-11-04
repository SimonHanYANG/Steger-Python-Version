import numpy as np
import pytest
from scipy.integrate import quad

from steger.derivatives import gaussian_masks, image_derivatives
from steger.profiles import invert_bias, profile_geometry, profile_response
from steger.refinement import integrated_profile
from steger.synthetic import straight_line


@pytest.mark.parametrize("sigma", [0.7, 1.5, 3.2])
def test_integrated_derivatives_preserve_constants_and_linear_ramps(sigma):
    kernels = gaussian_masks(sigma)
    assert abs(kernels[0].sum() - 1) < 1e-14
    assert abs(kernels[1].sum()) < 1e-14
    assert abs(kernels[2].sum()) < 1e-14
    rr, cc = np.indices((100, 100), dtype=float)
    derivatives = image_derivatives(2 * rr + 3 * cc + 10, sigma)
    region = np.s_[25:75, 25:75]
    # Piecewise-constant reconstruction has a known small aliasing term. An
    # independent Poisson-summation formula predicts it; do not renormalize
    # the paper's masks just to make a continuous-ramp assumption pass.
    alias = 1 + 2 * sum((-1) ** k * np.exp(-2 * np.pi**2 * sigma**2 * k * k) for k in range(1, 8))
    assert np.max(abs(derivatives["r"][region] - 2 * alias)) < 2e-7
    assert np.max(abs(derivatives["c"][region] - 3 * alias)) < 2e-7
    assert np.max(abs(derivatives["rr"][region])) < 1e-10
    assert np.max(abs(derivatives["rc"][region])) < 1e-10


@pytest.mark.parametrize("model", ["bar-shaped", "parabolic", "gaussian"])
@pytest.mark.parametrize("asymmetry", [0, 0.4, 0.75])
def test_analytic_convolution_against_independent_quadrature(model, asymmetry):
    w = 1.3
    x = 0.37

    def f(u):
        if model == "bar-shaped":
            return 0 if u < -w else 1 if u <= w else asymmetry
        shape = np.exp(-u * u / (2 * w * w)) if model == "gaussian" else max(0, 1 - u * u / (w * w))
        return shape if u <= 0 else asymmetry + (1 - asymmetry) * shape

    for order, value in enumerate(profile_response(x, w, asymmetry, model)):

        def integrand(u, order=order):
            z = x - u
            g = np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
            derivative = g if order == 0 else -z * g if order == 1 else (z * z - 1) * g
            return f(u) * derivative

        expected = quad(integrand, -16, 16, points=[-w, 0, w], epsabs=1e-11, epsrel=1e-11)[0]
        assert abs(value - expected) < 2e-10


@pytest.mark.parametrize("model", ["bar-shaped", "parabolic", "gaussian"])
def test_bias_inversion_recovers_known_parameters(model):
    geometry = profile_geometry(1.35, 0.45, model)
    w, a, _, residual = invert_bias(geometry.total_width, geometry.gradient_ratio, model)
    assert abs(w - 1.35) < 1e-7
    assert abs(a - 0.45) < 1e-7
    assert residual < 1e-8


def test_pixel_aperture_model_against_exact_rotated_bar_area():
    for angle in [0, 17, 45, 89.99]:
        image, gt = straight_line(
            shape=(20, 20),
            angle_deg=angle,
            offset=0.213,
            half_width=1.43,
            asymmetry=0.4,
            contrast=1,
            background=0,
        )
        rr, cc = np.indices(image.shape)
        d = (rr - gt["center"][0]) * gt["normal"][0] + (cc - gt["center"][1]) * gt["normal"][1]
        skew = 0.4 / (2 - 0.4)
        actual = 0.2 + 0.8 * integrated_profile(d, 1.43, skew, gt["normal"], "bar-shaped")
        assert np.max(abs(actual - image)) < 2e-8
