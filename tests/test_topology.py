import numpy as np
from scipy.special import ndtr

from steger import Contour, lines_gauss, measure_bending
from steger.linking import complete_junctions
from steger.synthetic import circular_line, straight_line


def test_full_circle_is_a_single_closed_contour():
    image, _ = circular_line(shape=(88, 88), radius=24)
    result = lines_gauss(image, 1.5, 1, 3, extract_width=False, complete_junctions=False)
    assert len(result.contours) == 1
    assert result.contours[0].closed
    assert not result.junctions


def test_t_junction_is_split_into_three_branches():
    rr, cc = np.indices((80, 80))
    center = 39.3
    image = 20 + 150 * np.maximum(
        np.exp(-0.5 * ((rr - center) / 1.6) ** 2),
        np.exp(-0.5 * ((cc - center) / 1.6) ** 2) * ndtr((center - rr) / 0.3),
    )
    result = lines_gauss(image, 1.5, 1, 3, extract_width=False, complete_junctions=True)
    assert len(result.contours) == 3
    assert any(len(j.contour_indices) == 3 for j in result.junctions)


def test_missing_junction_completion_is_flagged_not_treated_as_measurement():
    def contour(points):
        points = np.array(points, dtype=float)
        tangent = np.gradient(points, axis=0)
        normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        return Contour(points, normals, np.full(len(points), 10.0))

    horizontal = contour([[10, c] for c in range(2, 19)])
    vertical = contour([[r, 10] for r in range(2, 9)])
    contours, junctions = complete_junctions([horizontal, vertical], 1.5)
    assert len(contours) == 3
    assert any(j.completed and len(j.contour_indices) == 3 for j in junctions)
    assert any(c.attributes.get("junction_completed", np.array([])).any() for c in contours)
    for c in contours:
        if c.attributes.get("junction_completed", np.array([])).any():
            import pytest

            with pytest.raises(ValueError, match="inferred junction"):
                measure_bending(c)


def test_subpixel_phase_sweep_does_not_drop_half_pixel_ridges():
    for phase in np.linspace(-0.5, 0.5, 11):
        image, _ = straight_line(shape=(40, 40), model="gaussian", offset=phase)
        result = lines_gauss(image, 1.5, 0.4, 1.2, extract_width=False, complete_junctions=False)
        assert len(result.contours) == 1
        assert len(result.contours[0].points) == 40


def test_narrow_asymmetric_parabola_survives_newton_pixel_reassignment():
    image, gt = straight_line(
        shape=(64, 64), model="parabolic", half_width=1, asymmetry=0.5, offset=0.237
    )
    result = lines_gauss(image, 1.1, 0.4, 1.2, line_model="parabolic", complete_junctions=False)
    c = max(result.contours, key=lambda c: c.length)
    assert np.percentile(abs((c.points[12:-12] - gt["center"]) @ gt["normal"]), 95) < 0.01
