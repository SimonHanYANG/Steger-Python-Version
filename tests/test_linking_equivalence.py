"""Adversarial topology checks against the immutable v1.0.1 linker.

Git history is required for this additional frozen-reference suite. The ordinary
package tests remain runnable from a source archive without Git objects.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from steger.linking import link_points, orient_normals

REFERENCE = "914b3aed74b11b00b073491d6e3b91f471190853"


@pytest.fixture(scope="module")
def frozen_linker():
    try:
        source = subprocess.check_output(
            ["git", "show", f"{REFERENCE}:src/steger/linking.py"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("v1.0.1 Git object unavailable; full-history CI runs this suite")
    namespace = {"__name__": "steger._frozen_linking", "__package__": "steger"}
    exec(compile(source, "v1.0.1/linking.py", "exec"), namespace)
    return namespace["link_points"]


def assert_linked_bits_equal(expected, actual):
    old_contours, old_junctions = expected
    contours, junctions = actual
    assert len(old_contours) == len(contours)
    assert len(old_junctions) == len(junctions)
    for old, new in zip(old_contours, contours, strict=True):
        assert old.closed == new.closed
        for key in ("points", "normals", "response"):
            assert getattr(old, key).tobytes() == getattr(new, key).tobytes(), key
    for old, new in zip(old_junctions, junctions, strict=True):
        assert old.point.tobytes() == new.point.tobytes()
        assert old.contour_indices == new.contour_indices
        assert old.completed == new.completed


@pytest.mark.parametrize("count", [0, 1, 2, 40, 100, 513, 700])
def test_random_clouds_preserve_topology_order_and_input_arrays(frozen_linker, count):
    rng = np.random.default_rng(778 + count)
    points = rng.normal(0, 3, (count, 2))
    normals = rng.normal(size=(count, 2))
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    responses = rng.choice([0.15, np.nextafter(0.15, 0), 0.4], size=count)
    saved = [a.tobytes() for a in (points, normals, responses)]
    expected = frozen_linker(points, normals, responses, 0.15)
    actual = link_points(points, normals, responses, 0.15)
    assert_linked_bits_equal(expected, actual)
    assert saved == [a.tobytes() for a in (points, normals, responses)]


@pytest.mark.parametrize(
    "delta",
    [
        np.nextafter(0.22, 0),
        0.22,
        np.nextafter(0.22, 1),
        np.nextafter(1.85, 0),
        1.85,
        np.nextafter(1.85, 2),
    ],
)
def test_dedup_and_neighbor_radius_boundaries(frozen_linker, delta):
    points = np.array([[0, 0], [0, delta], [0, 2], [0, 3], [0, 4]], dtype=np.float64)
    normals = np.tile([1.0, 0.0], (len(points), 1))
    responses = np.array([1.0, 1.0, 0.5, 0.5, 1.0])
    assert_linked_bits_equal(
        frozen_linker(points, normals, responses, 0.8),
        link_points(points, normals, responses, 0.8),
    )


@pytest.mark.parametrize(
    "normal_cosine", [np.nextafter(0.5, 0), 0.5, np.nextafter(0.5, 1), np.nextafter(1.0, 2)]
)
def test_angle_threshold_clipping_and_lexicographic_score_ties(frozen_linker, normal_cosine):
    points = np.array([[0, 0], [0, 1], [0.1, 1], [-0.1, 1], [0, 2], [0, 3]], dtype=float)
    normals = np.tile([1.0, 0.0], (len(points), 1))
    normals[1] = [normal_cosine, np.sqrt(max(0.0, 1 - normal_cosine**2))]
    responses = np.ones(len(points))
    assert_linked_bits_equal(
        frozen_linker(points, normals, responses, 0.8),
        link_points(points, normals, responses, 0.8),
    )


@pytest.mark.parametrize("count", [20, 520])
def test_dense_coincident_points_and_response_priority(frozen_linker, count):
    points = np.vstack([np.zeros((count, 2)), np.column_stack([np.zeros(8), np.arange(1, 9)])])
    normals = np.tile([1.0, 0.0], (len(points), 1))
    responses = np.linspace(0.1, 1.0, len(points))
    assert_linked_bits_equal(
        frozen_linker(points, normals, responses, 0.8),
        link_points(points, normals, responses, 0.8),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("layout", ["dense", "strided", "list"])
def test_normal_orientation_matches_original_gradient(frozen_linker, dtype, layout):
    points = np.arange(40).reshape(20, 2).astype(dtype)
    if dtype == np.int64:
        points = 2**55 + points * 1024
    if layout == "strided":
        points = points[::3]
    elif layout == "list":
        points = points.tolist()
    normals = np.tile([0.6, -0.8], (len(points), 1))
    original = frozen_linker.__globals__["orient_normals"](points, normals)
    actual = orient_normals(points, normals)
    assert original.tobytes() == actual.tobytes()
