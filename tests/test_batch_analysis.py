"""Batch-policy tests: pairing, selection, and refusal of incomplete angles."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from steger.types import Contour

spec = importlib.util.spec_from_file_location(
    "analyze_necks", Path(__file__).resolve().parents[1] / "tools/analyze_necks.py"
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def curve(length=30.0, radius=None, count=101):
    s = np.linspace(0, length, count)
    if radius is None:
        p = np.column_stack([s, np.zeros_like(s)])
    else:
        theta = s / radius
        p = radius * np.column_stack([np.sin(theta), np.cos(theta)])
    normals = (np.tile([0.0, 1.0], (len(s), 1)) if radius is None
               else np.column_stack([-np.sin(theta), -np.cos(theta)]))
    return Contour(p, normals, np.ones(len(s)))


def candidate(sigma, coverage=0.9, length=30.0, angle=0.0):
    return dict(
        sigma=sigma,
        span_coverage=coverage,
        centrality=0.8,
        length_px=length,
        local_angle_deg=angle,
        polarity="dark",
    )


def test_pairing_rejects_missing_and_orphan_masks(tmp_path):
    (tmp_path / "a.png").touch()
    (tmp_path / "b_mask.png").touch()
    with pytest.raises(ValueError, match="Unpaired"):
        batch.image_pairs(tmp_path)
    (tmp_path / "b_mask.png").unlink()
    (tmp_path / "a_mask.png").touch()
    assert batch.image_pairs(tmp_path) == [(tmp_path / "a.png", tmp_path / "a_mask.png")]


def test_mask_geometry_keeps_diagonal_connections_and_reports_removed_island():
    mask = np.zeros((20, 30), dtype=bool)
    mask[3:7, 3:20] = True
    mask[7, 20] = True
    mask[15, 25] = True
    g = batch.mask_geometry(mask)
    assert g["component_count"] == 2
    assert g["ignored_mask_pixels"] == 1
    assert g["mask"][7, 20]
    assert not g["mask"][15, 25]
    assert mask[15, 25]  # the supplied mask is not modified


def test_candidate_choice_uses_extent_and_reference_scale_not_angle():
    a = candidate(1.0, 0.93, angle=1)
    b = candidate(2.0, 0.91, angle=70)
    assert batch.choose_candidate([a, b]) is b
    # Do not select an incomplete reference-scale line when a nearby adequate one exists.
    a = candidate(1.5, 0.77)
    b = candidate(2.0, 0.73)
    assert batch.choose_candidate([a, b]) is a
    assert batch.choose_candidate([]) is None


def test_measurable_similarly_complete_candidate_wins_without_angle_magnitude_ranking():
    bad = candidate(2, .92, angle=None)
    good = candidate(1.5, .91, angle=65)
    assert batch.choose_candidate([bad, good]) is good
    # An incomplete measurable line must not displace a much fuller line.
    good['span_coverage'] = .5
    assert batch.choose_candidate([bad, good]) is bad


def test_trimming_cannot_keep_original_coverage_as_evidence_for_complete_angle():
    p = np.column_stack([np.zeros(31), np.arange(31.)])
    p = np.vstack([p, [[2., 30.2], [1.8, 30.0]]])
    n = np.tile([1., 0.], (len(p), 1))
    n[-1] = [.8, -.6]
    c = Contour(p, n, np.ones(len(p)))
    selected = candidate(2, .8, c.length)
    result = batch.assess(c, selected, [selected], {'axis': np.array([0., 1.]), 'span': 41.})
    assert result['measurement']['excluded_indices'] == (31, 32)
    assert result['trusted_span_coverage'] < .75
    assert result['angle_deg'] is None
    assert 'insufficient_trusted_span_coverage' in result['quality_flags']


@pytest.mark.parametrize("length,coverage,count", [(5, 0.95, 10), (30, 0.5, 101)])
def test_incomplete_or_short_line_has_no_full_neck_angle(length, coverage, count):
    selected = candidate(2, coverage, length)
    result = batch.assess(curve(length, count=count), selected, [selected])
    assert result["status"] == "insufficient_segment"
    assert result["angle_deg"] is None
    assert result["local_segment_angle_deg"] == (None if count <= 10 else pytest.approx(0))


def test_circular_bending_uses_geometric_direction_range():
    from steger import measure_bending
    contour = curve(30, radius=30)
    angle = measure_bending(contour).angle_deg
    candidates = [candidate(s, angle=angle) for s in batch.SCALES]
    result = batch.assess(contour, candidates[2], candidates)
    assert result["angle_deg"] == pytest.approx(angle, abs=0.01)
    assert angle == pytest.approx(np.degrees(1.) * .7, abs=.02)
    assert result["quality_flags"] == ['window_sensitive_over_5deg']
    assert len(result["window_sensitivity"]) == 3


def test_short_dense_contour_does_not_gain_false_precision_from_point_count():
    selected = candidate(2, .95, 5)
    result = batch.assess(curve(5, count=30), selected, [selected])
    assert result['angle_deg'] is None
    assert result['status'] == 'insufficient_segment'


def test_scale_variation_is_reported_as_review_not_hidden():
    candidates = [candidate(1.5, angle=1), candidate(2, angle=20)]
    result = batch.assess(curve(), candidates[1], candidates)
    assert result["status"] == "review_required"
    assert result["scale_angle_range_deg"] == [1, 20]
    assert "scale_sensitive_over_5deg" in result["quality_flags"]


def test_large_scale_rescue_is_always_marked_for_review():
    candidates = [candidate(3.5), candidate(4)]
    result = batch.assess(curve(), candidates[0], candidates)
    assert result["status"] == "review_required"
    assert "large_scale_rescue_requires_review" in result["quality_flags"]


def test_error_only_report_does_not_crash_or_link_missing_images(tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "result.json").write_text("{}")
    counts = batch.write_summary([{"id": "bad", "status": "input_error"}], [], tmp_path)
    assert counts == {"input_error": 1}
    assert 'src="bad/measurement.png"' not in (tmp_path / "index.html").read_text(encoding='utf-8')
    assert (tmp_path / "visualizations.zip").exists()
