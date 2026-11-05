import numpy as np
import pytest

from steger import lines_gauss, measure_bending
from steger.synthetic import circular_line, straight_line


def main_contour(result):
    assert result.contours, "Expected a detectable line"
    return max(result.contours, key=lambda c: c.length)


def interior(c, size=64):
    return np.all((c.points > 12) & (c.points < size - 13), axis=1)


@pytest.mark.parametrize("model", ["bar-shaped", "gaussian", "parabolic"])
@pytest.mark.parametrize("angle", [0, 23, 61, 90])
def test_clean_asymmetric_subpixel_localization_and_width(model, angle):
    image, gt = straight_line(
        shape=(64, 64), model=model, angle_deg=angle, offset=0.237, asymmetry=0.4, quadrature=32
    )
    result = lines_gauss(image, 1.5, 0.5, 2, "light", True, model, False)
    c = main_contour(result)
    valid = interior(c)
    errors = np.abs((c.points[valid] - gt["center"]) @ gt["normal"])
    assert len(errors) > 20
    assert np.percentile(errors, 95) < 0.01
    assert np.percentile(abs(c.attributes["width_left"][valid] - 1.5), 95) < 0.02
    assert np.mean(c.attributes["profile_fit_valid"][valid]) > 0.95


@pytest.mark.parametrize("model", ["bar-shaped", "gaussian", "parabolic"])
def test_dark_lines_preserve_precision_and_contrast_sign(model):
    image, gt = straight_line(
        shape=(64, 64), model=model, angle_deg=32, offset=-0.31, polarity="dark", asymmetry=0.5
    )
    c = main_contour(lines_gauss(image, 1.5, 0.5, 2, "dark", True, model, False))
    valid = interior(c)
    assert np.percentile(np.abs((c.points[valid] - gt["center"]) @ gt["normal"]), 95) < 0.015
    assert np.all(c.attributes["contrast"][valid] < 0)


def test_noise_and_uint8_quantization_have_explicit_tolerances():
    for noise in [0, 1, 3]:
        image, gt = straight_line(
            shape=(64, 64),
            model="gaussian",
            angle_deg=37,
            offset=0.19,
            asymmetry=0.3,
            noise=noise,
            quantize=True,
            seed=123,
        )
        c = main_contour(lines_gauss(image, 1.5, 0.5, 2, "light", True, "gaussian", False))
        valid = interior(c)
        errors = abs((c.points[valid] - gt["center"]) @ gt["normal"])
        assert np.percentile(errors, 95) < (0.015 if noise == 0 else 0.08 if noise == 1 else 0.2)


def test_mask_selects_candidates_without_changing_convolution():
    image, gt = straight_line(shape=(64, 64), model="gaussian", angle_deg=25)
    rr, cc = np.indices(image.shape)
    mask = (rr > 15) & (rr < 49) & (cc > 15) & (cc < 49)
    result = lines_gauss(
        image, 1.5, 0.5, 2, extract_width=False, complete_junctions=False, mask=mask
    )
    c = main_contour(result)
    assert np.percentile(abs((c.points - gt["center"]) @ gt["normal"]), 95) < 0.01
    assert len(c.points) < 64
    assert set(c.attributes) == set()
    assert result.diagnostics["intensity_rescaled"] is False


def test_circle_radius_and_bending_angle_with_known_truth():
    image, gt = circular_line(shape=(112, 112), radius=30)
    rr, cc = np.indices(image.shape)
    center = gt["center"]
    theta = np.arctan2(rr - center[0], cc - center[1])
    mask = (theta > 0.2) & (theta < 1.8) & (abs(np.hypot(rr - center[0], cc - center[1]) - 30) < 3)
    c = main_contour(lines_gauss(image, 1.5, 0.5, 2, "light", True, "gaussian", False, mask=mask))
    assert np.percentile(abs(np.linalg.norm(c.points - center, axis=1) - 30), 95) < 0.01
    from steger import measure_end_direction_bending
    measurement = measure_end_direction_bending(c)
    angles = np.unwrap(np.arctan2(c.rows - center[0], c.cols - center[1]))
    # The explicit fit windows span 70% of the arc between their midpoints.
    expected = abs(np.rad2deg(angles[-1] - angles[0])) * 0.7
    assert abs(measurement.angle_deg - expected) < 0.1
    assert abs(measure_bending(c).angle_deg - expected) < 0.1


def test_symmetric_straight_neck_is_zero_not_180_degrees():
    from steger import measure_end_direction_bending, measure_direction_angles
    image, _ = straight_line(shape=(64, 64), model="gaussian", angle_deg=39)
    c = main_contour(lines_gauss(image, 1.5, 0.5, 2, "light", True, "gaussian", False))
    assert measure_end_direction_bending(c).angle_deg < 0.01
    assert measure_bending(c).angle_deg < .01


def test_hysteresis_rejects_no_high_seed():
    image, _ = straight_line(shape=(64, 64), model="gaussian")
    result = lines_gauss(image, 1.5, 1, 10000, extract_width=False)
    assert result.contours == []


def test_flat_images_are_empty():
    for value in [0, 1, 127.5, 65535]:
        assert not lines_gauss(np.full((24, 24), value), low=0.001, high=0.01).contours


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0},
        {"low": -1},
        {"high": 1, "low": 2},
        {"line_model": "bogus"},
        {"mask": np.zeros((3, 4))},
        {"extract_width": "yes"},
    ],
)
def test_invalid_parameters_are_not_silently_reinterpreted(kwargs):
    with pytest.raises(ValueError):
        lines_gauss(np.zeros((20, 20)), **kwargs)


def test_pixel_size_intensity_and_input_are_not_mutated():
    image, _ = straight_line(shape=(64, 64), model="gaussian")
    backup = image.copy()
    a = main_contour(lines_gauss(image, 1.5, 0.5, 2, extract_width=False, complete_junctions=False))
    b = main_contour(
        lines_gauss(image * 100, 1.5, 50, 200, extract_width=False, complete_junctions=False)
    )
    assert np.array_equal(image, backup)
    assert np.allclose(a.points, b.points, atol=1e-10)
    assert a.points.dtype == np.float64
