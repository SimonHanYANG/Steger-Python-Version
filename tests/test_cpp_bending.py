"""Independent literal C++ loop translation, boundaries and per-track history."""
import numpy as np
import pytest

from steger import Contour, NeckBendingHistory, measure_cpp_bending, measure_direction_angles


def literal_cpp(angles):
    values = np.asarray(angles, dtype=np.float32)
    smooth = []
    for k in range(len(values) - 10):
        total = np.float32(0)
        for j in range(10):
            total = np.float32(total + values[k + j])
        smooth.append(np.float32(total / np.float32(10)))
    smooth = np.asarray(smooth, dtype=np.float32)
    if not len(smooth):
        raise ValueError("No C++ windows")
    result = np.float32(float(np.float32(max(smooth) - min(smooth))) * 180.0 / np.pi)
    return float(result), smooth


@pytest.mark.parametrize("count", [11, 12, 20, 31, 101, 999])
def test_cpp_window_average_and_angle_are_bitwise_literal_loop(count):
    rng = np.random.default_rng(count)
    angles = rng.uniform(0, 2 * np.pi, count)
    expected, smooth = literal_cpp(angles)
    actual = measure_direction_angles(angles)
    assert actual.angle_deg == expected
    assert np.asarray(actual.smoothed_angles_rad, dtype=np.float32).tobytes() == smooth.tobytes()
    assert actual.window_count == count - 10
    assert actual.min_window == (int(np.argmin(smooth)), int(np.argmin(smooth)) + 10)


@pytest.mark.parametrize("count", [0, 6, 9, 10])
def test_ten_or_fewer_samples_cannot_produce_cpp_angle(count):
    with pytest.raises(ValueError, match="more than 10"):
        measure_direction_angles(np.zeros(count))


def test_last_input_sample_and_final_mathematically_possible_window_are_omitted():
    angles = np.zeros(12)
    angles[-1] = 6
    assert measure_direction_angles(angles).angle_deg == 0
    angles[-2] = 1
    assert measure_direction_angles(angles).angle_deg == literal_cpp(angles)[0]
    assert measure_direction_angles(np.arange(11.)).angle_deg == 0


def test_periodic_normal_angles_are_not_unwrapped_or_folded():
    angles = np.r_[np.full(20, .01), np.full(20, 2 * np.pi - .01)]
    actual = measure_direction_angles(angles)
    assert actual.angle_deg == literal_cpp(angles)[0]
    assert actual.angle_deg > 350  # preserve old behavior at the period boundary


def test_normal_direction_sequence_drives_angle_not_endpoint_coordinates():
    angles = np.linspace(.3, .9, 30)
    points = np.column_stack([np.arange(30.), np.zeros(30)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    contour = Contour(points, normals, np.ones(30))
    actual = measure_cpp_bending(contour)
    expected = measure_direction_angles(contour.get_attribute('angle'))
    assert actual.angle_deg == expected.angle_deg > 0
    assert actual.method == 'cpp_legacy_10point_float32_direction_range'


def test_constant_direction_has_zero_bending():
    assert measure_direction_angles(np.full(100, .7)).angle_deg == 0


def test_s_curve_turns_are_not_cancelled_by_matching_endpoint_directions():
    angles = np.r_[np.linspace(.3, 1.3, 30), np.linspace(1.3, .3, 30)]
    assert measure_direction_angles(angles).angle_deg > 40


def test_history_is_per_track_ignores_invalid_frames_and_keeps_zero():
    a, b = NeckBendingHistory(), NeckBendingHistory()
    assert a.update(None) is None
    assert a.update(0) == 0
    assert a.update(20) == 10
    assert a.update(40) == 20
    assert a.update(None) == 20
    assert b.update(100) == 100
    assert a.update(None) == 20
    a.reset()
    assert a.update(None) is None


@pytest.mark.parametrize('angles', [[np.nan] * 12, [np.inf] * 12, np.zeros((12, 2))])
def test_invalid_angles_are_rejected(angles):
    with pytest.raises(ValueError, match='finite one-dimensional'):
        measure_direction_angles(angles)
