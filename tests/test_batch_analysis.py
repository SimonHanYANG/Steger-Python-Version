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


def curve(length=30.0, radius=None):
    s = np.linspace(0, length, 101)
    if radius is None:
        p = np.column_stack([s, np.zeros_like(s)])
    else:
        theta = s / radius
        p = radius * np.column_stack([np.sin(theta), np.cos(theta)])
    return Contour(p, np.tile([0.0, 1.0], (len(s), 1)), np.ones(len(s)))


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


@pytest.mark.parametrize("length,coverage", [(5, 0.95), (30, 0.5)])
def test_incomplete_or_short_line_has_no_full_neck_angle(length, coverage):
    selected = candidate(2, coverage, length)
    result = batch.assess(curve(length), selected, [selected])
    assert result["status"] == "insufficient_segment"
    assert result["angle_deg"] is None
    assert result["local_segment_angle_deg"] == pytest.approx(0)


def test_circular_bending_uses_directed_trimmed_windows():
    contour = curve(30, radius=30)
    angle = np.degrees(0.7)  # centers of 5-25% and 75-95% windows differ by 70% of arc
    candidates = [candidate(s, angle=angle) for s in batch.SCALES]
    result = batch.assess(contour, candidates[2], candidates)
    assert result["angle_deg"] == pytest.approx(angle, abs=0.01)
    # Even an exact circle changes angle when the end-window centers move.
    assert result["quality_flags"] == ["window_sensitive_over_5deg"]


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
    assert 'src="bad/measurement.png"' not in (tmp_path / "index.html").read_text()
    assert (tmp_path / "visualizations.zip").exists()
