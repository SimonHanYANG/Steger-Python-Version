"""Exercise the shared-field path against independent full-field detections."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from steger import detector, lines_gauss, measure_neck, neck
from steger.synthetic import straight_line


def example(angle=31, polarity="dark"):
    image, gt = straight_line(
        shape=(40, 40),
        model="gaussian",
        angle_deg=angle,
        polarity=polarity,
        offset=0.237,
        asymmetry=0.35,
        quantize=True,
    )
    coordinates = np.moveaxis(np.indices(image.shape), 0, -1) - gt["center"]
    tangent = np.array([-gt["normal"][1], gt["normal"][0]])
    mask = (abs(coordinates @ gt["normal"]) < 4) & (abs(coordinates @ tangent) < 15)
    return image, mask


def assert_detection_equal(a, b):
    assert a.parameters == b.parameters
    assert a.diagnostics == b.diagnostics
    assert len(a.contours) == len(b.contours)
    assert len(a.junctions) == len(b.junctions)
    for x, y in zip(a.contours, b.contours, strict=True):
        assert x.closed == y.closed
        for field in ("points", "normals", "response"):
            assert getattr(x, field).tobytes() == getattr(y, field).tobytes()
        assert x.attributes.keys() == y.attributes.keys()
        for key in x.attributes:
            assert x.attributes[key].tobytes() == y.attributes[key].tobytes()
    for x, y in zip(a.junctions, b.junctions, strict=True):
        assert x.point.tobytes() == y.point.tobytes()
        assert x.contour_indices == y.contour_indices
        assert x.completed == y.completed


@pytest.mark.parametrize("border", ["reflect", "mirror", "nearest"])
@pytest.mark.parametrize("refine", [True, False])
@pytest.mark.parametrize("kind", ["empty", "disconnected", "border", "all"])
def test_masked_eigensystems_preserve_full_field_outputs(border, refine, kind):
    rng = np.random.default_rng(173)
    image = rng.normal(100, 25, (17, 21))
    mask = np.zeros(image.shape, dtype=bool)
    if kind == "disconnected":
        mask = rng.random(image.shape) < 0.35
    elif kind == "border":
        mask[:4] = True
        mask[:, -4:] = True
    elif kind == "all":
        mask[:] = True
    for polarity in ("dark", "light"):
        args = dict(
            sigma=1.5,
            low=0.05,
            high=0.15,
            light_dark=polarity,
            mask=mask,
            border=border,
            refine=refine,
            extract_width=False,
            complete_junctions=False,
        )
        fast = lines_gauss(image, **args)
        full = lines_gauss(image, **args, return_trace=True)
        assert_detection_equal(fast, full)


@pytest.mark.parametrize("angle", [0, 31, 90])
def test_shared_polarities_match_independent_full_field_runs(angle):
    image, mask = example(angle)
    # This contrasts shared fields AND masked eigenproblems with full ROI fields.
    results = detector._detect_neck_polarities(image, mask, 2.0, neck.LOW, neck.HIGH)
    for polarity, actual in zip(("dark", "light"), results, strict=True):
        expected = neck.detect(image, mask, 2.0, polarity, trace=True)
        assert_detection_equal(expected, actual)


def test_worker_shares_preparation_and_still_searches_every_scale(monkeypatch):
    image, mask = example()
    calls = []
    original = detector.image_derivatives

    def recorded(image, sigma, **kwargs):
        calls.append(sigma)
        return original(image, sigma, **kwargs)

    monkeypatch.setattr(detector, "image_derivatives", recorded)
    output = measure_neck(image, mask)
    assert output["angle_deg"] is not None
    assert calls == list(neck.SCALES)  # one preparation per scale, no final rerun
    json.dumps(output, allow_nan=False)  # no private contour or fit leaks into results


def test_no_line_retains_all_rescue_scales(monkeypatch):
    calls = []
    original = detector.image_derivatives

    def recorded(image, sigma, **kwargs):
        calls.append(sigma)
        return original(image, sigma, **kwargs)

    monkeypatch.setattr(detector, "image_derivatives", recorded)
    output = measure_neck(np.zeros((20, 20)), np.ones((20, 20), dtype=np.uint8))
    assert output == {"angle_deg": None, "status": "no_line"}
    assert calls == list(neck.SCALES + neck.RESCUE_SCALES)


def test_rgb_input_and_concurrent_calls_do_not_mutate_or_share_image_state():
    image, mask = example()
    rgb = np.repeat(image[..., None], 3, axis=2)
    rgb_mask = np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)
    gray = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    expected = measure_neck(gray, mask)
    rgb.flags.writeable = rgb_mask.flags.writeable = False
    saved_roi, saved_mask = rgb.copy(), rgb_mask.copy()
    other_image, other_mask = example(61, "light")
    other_expected = measure_neck(other_image, other_mask)
    cases = [(rgb, rgb_mask), (other_image, other_mask)] * 3
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda pair: measure_neck(*pair), cases))
    assert results == [expected, other_expected] * 3
    np.testing.assert_array_equal(rgb, saved_roi)
    np.testing.assert_array_equal(rgb_mask, saved_mask)


@pytest.mark.parametrize("problem", ["empty_mask", "tiny_mask", "mismatch", "nonfinite"])
def test_worker_rejects_unmeasurable_or_invalid_inputs(problem):
    image, mask = example()
    if problem == "empty_mask":
        mask[:] = False
    elif problem == "tiny_mask":
        mask[:] = False
        mask[10, 10] = True
    elif problem == "mismatch":
        mask = mask[:-1]
    else:
        image = image.astype(float)
        image[0, 0] = np.nan
    with pytest.raises(ValueError):
        measure_neck(image, mask)


def test_worker_import_does_not_load_visualization():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from steger import measure_neck; import sys; assert 'matplotlib' not in sys.modules",
        ],
        check=True,
    )
