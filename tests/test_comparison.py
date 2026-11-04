from steger.comparison import compare_results


def result(rows, cols):
    return {"contours": [{"row": rows, "col": cols}]}


def test_point_spacing_is_not_reported_as_position_error():
    report = compare_results(result([1, 1, 1], [0, 2, 4]), result([1, 1, 1, 1, 1], [0, 1, 2, 3, 4]))
    assert report["passes_geometry_gate"]
    assert report["candidate_to_reference"]["max_px"] < 1e-12


def test_bidirectional_coverage_detects_missing_line_part():
    report = compare_results(result([1, 1], [0, 2]), result([1, 1], [0, 20]))
    assert report["candidate_to_reference"]["coverage"] == 1
    assert report["reference_to_candidate"]["coverage"] < 0.2
    assert not report["passes_geometry_gate"]


def test_empty_output_cannot_pass_accuracy_gate():
    assert not compare_results({"contours": []}, {"contours": []})["passes_geometry_gate"]
