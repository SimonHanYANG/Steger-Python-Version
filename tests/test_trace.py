import json

import numpy as np
import pytest

from steger import lines_gauss
from steger.synthetic import straight_line


def test_recording_does_not_change_localization_width_or_normals():
    image, _ = straight_line(
        shape=(48, 48), model="gaussian", angle_deg=31, asymmetry=0.4, offset=0.237
    )
    options = dict(sigma=1.5, low=0.4, high=1.2, line_model="gaussian", complete_junctions=False)
    plain = lines_gauss(image, **options)
    recorded = lines_gauss(image, **options, return_trace=True)
    assert plain.trace is None
    assert len(plain.contours) == len(recorded.contours)
    assert plain.diagnostics == recorded.diagnostics
    for a, b in zip(plain.contours, recorded.contours, strict=True):
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.normals, b.normals)
        np.testing.assert_array_equal(a.response, b.response)
        for name in a.attributes:
            np.testing.assert_array_equal(a.attributes[name], b.attributes[name])


def test_recorded_normals_are_actual_hessian_eigenvectors():
    image, truth = straight_line(shape=(48, 48), angle_deg=32, model="gaussian")
    result = lines_gauss(
        image, 1.5, 0.4, 1.2, extract_width=False, complete_junctions=False, return_trace=True
    )
    trace = result.trace
    pixels = trace.candidate_pixels
    select = np.all((pixels > 10) & (pixels < 37), axis=1)
    pixels = pixels[select]
    rr, cc = pixels.T
    n = trace.hessian_normals[rr, cc]
    h = np.empty((len(pixels), 2, 2))
    h[:, 0, 0] = trace.derivatives["rr"][rr, cc]
    h[:, 1, 1] = trace.derivatives["cc"][rr, cc]
    h[:, 0, 1] = h[:, 1, 0] = trace.derivatives["rc"][rr, cc]
    eigenvalues = trace.eigenvalues[rr, cc]
    lam = np.take_along_axis(eigenvalues, np.argmax(abs(eigenvalues), axis=1)[:, None], axis=1)[
        :, 0
    ]
    assert np.max(abs(np.einsum("ijk,ik->ij", h, n) - lam[:, None] * n)) < 1e-10
    assert np.min(abs(n @ truth["normal"])) > 0.999
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-14)


def test_taylor_offset_satisfies_directional_zero_derivative_equation():
    image, _ = straight_line(shape=(48, 48), angle_deg=47, model="gaussian", offset=0.19)
    trace = lines_gauss(image, 1.5, 0.4, 1.2, extract_width=False, return_trace=True).trace
    r, c = trace.initial_candidate_pixels.T
    n = trace.hessian_normals[r, c]
    g = np.column_stack([trace.derivatives["r"][r, c], trace.derivatives["c"][r, c]])
    curvature = (
        trace.derivatives["rr"][r, c] * n[:, 0] ** 2
        + 2 * trace.derivatives["rc"][r, c] * n[:, 0] * n[:, 1]
        + trace.derivatives["cc"][r, c] * n[:, 1] ** 2
    )
    assert np.max(abs(np.einsum("ij,ij->i", g, n) + trace.taylor_t[r, c] * curvature)) < 1e-10
    assert len(trace.newton_positions) > 0


def test_hessian_normal_exists_where_first_gradient_is_zero():
    image, _ = straight_line(shape=(45, 45), model="gaussian", angle_deg=0, offset=0)
    trace = lines_gauss(image, 1.5, 0.4, 1.2, extract_width=False, return_trace=True).trace
    assert np.hypot(trace.derivatives["r"][22, 22], trace.derivatives["c"][22, 22]) < 1e-10
    assert abs(trace.hessian_normals[22, 22, 0]) > 0.999999
    assert trace.response[22, 22] > 1


def test_trace_arrays_and_contours_are_independent_snapshots():
    image, _ = straight_line(shape=(40, 40), model="gaussian", asymmetry=0.3)
    original = image.copy()
    result = lines_gauss(
        image, 1.5, 0.4, 1.2, line_model="gaussian", complete_junctions=False, return_trace=True
    )
    final_points = result.contours[0].points.copy()
    trace = result.trace
    assert not np.shares_memory(trace.linked_contours[0].points, result.contours[0].points)
    assert not np.shares_memory(trace.bias_corrected_contours[0].points, result.contours[0].points)
    assert np.max(abs(trace.linked_contours[0].points - final_points)) > 0.1
    trace.image[:] = 0
    trace.candidate_points[:] = 0
    trace.linked_contours[0].points[:] = 0
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(result.contours[0].points, final_points)


def test_process_export_preserves_float64_and_actual_point_values(tmp_path):
    from steger.visualization import save_process_report

    image, _ = straight_line(shape=(40, 40), model="gaussian", angle_deg=25)
    result = lines_gauss(
        image, 1.5, 0.4, 1.2, extract_width=False, complete_junctions=False, return_trace=True
    )
    metadata = save_process_report(
        result, tmp_path, inspect_point=(20, 20), label="Synthetic unit-test ROI"
    )
    for filename in [
        "process-overview.png",
        "derivatives.png",
        "normals.png",
        "point-inspection.png",
        "process.json",
        "README.md",
    ]:
        assert (tmp_path / filename).exists()
    with np.load(tmp_path / "process.npz", allow_pickle=False) as data:
        assert data["derivative_rr"].dtype == np.float64
        np.testing.assert_array_equal(data["candidate_points"], result.trace.candidate_points)
        np.testing.assert_array_equal(data["final_points"], result.contours[0].points)
    saved = json.loads((tmp_path / "process.json").read_text())
    assert saved["point_inspection"] == metadata["point_inspection"]
    point = saved["point_inspection"]
    assert point["lambda_normal"] == pytest.approx(
        np.asarray(point["normal"]) @ np.asarray(point["hessian"]) @ np.asarray(point["normal"])
    )


def test_process_export_requires_recording(tmp_path):
    from steger.visualization import save_process_report

    result = lines_gauss(np.zeros((12, 12)))
    with pytest.raises(ValueError, match="return_trace=True"):
        save_process_report(result, tmp_path)


def test_trace_without_newton_keeps_initial_candidate_mapping():
    image, _ = straight_line(shape=(36, 36), model="gaussian", offset=0.37)
    trace = lines_gauss(
        image, 1.5, 0.4, 1.2, extract_width=False, refine=False, return_trace=True
    ).trace
    assert not trace.newton_positions
    np.testing.assert_array_equal(trace.initial_candidate_pixels, trace.candidate_pixels)
    np.testing.assert_array_equal(trace.initial_candidate_points, trace.candidate_points)


def test_empty_detection_report_does_not_invent_an_inspection_point(tmp_path):
    from steger.visualization import save_process_report

    result = lines_gauss(np.zeros((12, 12)), return_trace=True)
    (tmp_path / "point-inspection.png").write_bytes(b"previous generated artifact")
    metadata = save_process_report(result, tmp_path)
    assert metadata["point_inspection"] is None
    assert not (tmp_path / "point-inspection.png").exists()
    with np.load(tmp_path / "process.npz", allow_pickle=False) as data:
        assert data["candidate_points"].shape == (0, 2)
        assert data["final_offsets"].tolist() == [0]


def test_cli_process_option_exports_trace(tmp_path):
    from steger.cli import main

    image, _ = straight_line(shape=(36, 36), model="gaussian")
    source = tmp_path / "input.npy"
    np.save(source, image)
    output = tmp_path / "result.json"
    folder = tmp_path / "process"
    status = main(
        [
            "detect",
            str(source),
            "--low",
            ".4",
            "--high",
            "1.2",
            "--no-width",
            "--process-dir",
            str(folder),
            "--inspect-point",
            "18",
            "18",
            "--output",
            str(output),
        ]
    )
    assert status == 0
    assert (folder / "process.npz").exists()
    assert json.loads(output.read_text())["process_directory"] == str(folder)
