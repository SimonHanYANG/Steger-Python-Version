"""General geometry and invariance tests, independent of named real samples."""
import numpy as np
import pytest

from steger import Contour, measure_bending, measure_end_direction_bending
from steger.robust_measurement import _tls_directions, window_direction_ranges


def make(points, normals=None):
    points = np.asarray(points, dtype=float)
    if normals is None:
        tangent = np.gradient(points, axis=0)
        normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
    return Contour(points, np.asarray(normals), np.ones(len(points)))


def circle(theta=1.2, radius=30, count=301, exponent=1):
    t = np.linspace(0., 1., count)**exponent * theta
    return make(radius * np.column_stack([np.sin(t), np.cos(t)]),
                np.column_stack([np.sin(t), np.cos(t)]))


def test_closed_form_tls_agrees_with_independent_svd_for_rotated_noisy_windows():
    rng = np.random.default_rng(813)
    angles = rng.uniform(-np.pi, np.pi, 500)
    axes = np.column_stack([np.sin(angles), np.cos(angles)])
    windows = np.linspace(-3, 3, 25)[None, :, None] * axes[:, None, :]
    windows += rng.normal(0, .15, windows.shape)
    centered = windows - windows.mean(axis=1, keepdims=True)
    travel = windows[:, -1] - windows[:, 0]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    expected = vt[:, 0, :].copy()
    expected[np.einsum('ij,ij->i', expected, travel) < 0] *= -1
    assert np.allclose(_tls_directions(centered, travel), expected, rtol=0, atol=2e-14)


def test_straight_geometry_ignores_periodic_hessian_encoding_and_sign_flips():
    p = np.column_stack([np.zeros(100), np.linspace(0, 30, 100)])
    angles = np.r_[np.full(50, -.01), np.full(50, .01)]
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    normals[::2] *= -1
    assert measure_bending(make(p, normals)).angle_deg < 1e-10


@pytest.mark.parametrize('theta,radius', [(1.2, 30), (np.pi, 6), (4., 10)])
def test_known_circular_arc_keeps_real_bending_and_smooth_hairpins(theta, radius):
    contour = circle(theta, radius)
    m = measure_bending(contour)
    assert m.angle_deg == pytest.approx(np.degrees(theta) * .7, abs=.02)
    assert m.excluded_indices == ()


@pytest.mark.parametrize('angle', [0., .7, 2.7, 4., 6.1])
def test_rotation_reversal_normal_sign_and_translation_preserve_measurement(angle):
    c = circle()
    a = measure_bending(c)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = make(c.points[::-1] @ rotation + [64, -100], -c.normals[::-1] @ rotation)
    b = measure_bending(transformed)
    assert b.angle_deg == pytest.approx(a.angle_deg, abs=1e-10)
    assert b.excluded_indices == ()


def test_arc_resampling_removes_nonuniform_density_bias():
    assert measure_bending(circle(exponent=2)).angle_deg == pytest.approx(
        measure_bending(circle()).angle_deg, abs=.01)


def test_reused_sensitivity_matches_full_independent_measurements():
    c=circle()
    for row in window_direction_ranges(c,measure_bending(c)):
        assert row['angle_deg'] == measure_bending(c,window_fraction=row['window_fraction']).angle_deg


def test_interior_s_bend_is_not_cancelled_by_matching_end_directions():
    t = np.linspace(0, 1, 601)
    contour = make(np.column_stack([4 * np.sin(2*np.pi*t), 40*t]))
    assert measure_end_direction_bending(contour).angle_deg < .01
    assert measure_bending(contour).angle_deg > 45


def test_inconsistent_reversing_tip_is_excluded_and_raw_detection_preserved():
    p = np.column_stack([np.zeros(31), np.arange(31.)])
    p = np.vstack([p, [[2., 30.2], [1.8, 30.0]]])
    n = np.tile([1., 0.], (len(p), 1))
    n[-1] = [.8, -.6]
    contour = make(p, n)
    before = [a.tobytes() for a in (contour.points, contour.normals)]
    m = measure_bending(contour)
    assert m.angle_deg < 1e-9
    assert m.excluded_indices == (31, 32)
    assert 'terminal_samples_excluded' in m.quality_flags
    assert before == [a.tobytes() for a in (contour.points, contour.normals)]
    reversed_m = measure_bending(make(p[::-1], n[::-1]))
    assert reversed_m.excluded_indices == (0, 1)
    assert reversed_m.angle_deg == pytest.approx(m.angle_deg, abs=1e-9)


def test_large_inconsistent_terminal_region_is_refused_not_trimmed_to_a_straight_line():
    p = np.column_stack([np.zeros(31), np.arange(31.)])
    n = np.tile([1., 0.], (31, 1))
    n[-15:] = [0., 1.]
    with pytest.raises(ValueError, match='budget'):
        measure_bending(make(p, n))


def test_duplicate_points_are_audited_without_changing_result():
    c = circle()
    m = measure_bending(make(np.repeat(c.points, 2, axis=0), np.repeat(c.normals, 2, axis=0)))
    assert m.angle_deg == pytest.approx(measure_bending(c).angle_deg, abs=1e-10)
    assert 'duplicate_samples_ignored' in m.quality_flags


@pytest.mark.parametrize('kind', ['short', 'closed', 'inferred', 'nonfinite'])
def test_unmeasurable_geometry_is_refused(kind):
    c = circle()
    if kind == 'short':
        c.points *= .1
    elif kind == 'closed':
        c.closed = True
    elif kind == 'inferred':
        c.attributes['junction_completed'] = np.ones(len(c.points), dtype=bool)
    else:
        c.points[0, 0] = np.nan
    with pytest.raises(ValueError):
        measure_bending(c)
