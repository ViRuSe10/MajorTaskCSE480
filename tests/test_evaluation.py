"""
test_evaluation.py
Unit tests for src/evaluation.py
"""

import os
import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import evaluation as ev


# ---------------------------------------------------------------------------
# compute_intrinsic_quality
# ---------------------------------------------------------------------------

def _fake_result(grid, shape, unplaced, start_cell):
    return {"grid": grid, "shape": shape, "unplaced_pieces": unplaced,
            "complete": len(unplaced) == 0, "start_cell": start_cell}


def test_intrinsic_quality_perfect_complete_rectangle():
    grid = {
        (0, 0): {"piece": 0, "rotation": 0, "score": 0.0},
        (0, 1): {"piece": 1, "rotation": 0, "score": 0.02},
        (1, 0): {"piece": 2, "rotation": 0, "score": 0.01},
        (1, 1): {"piece": 3, "rotation": 0, "score": 0.015},
    }
    result = _fake_result(grid, (2, 2), [], start_cell=(0, 0))
    q = ev.compute_intrinsic_quality(result)
    assert q["completion_ratio"] == 1.0
    assert q["fill_ratio"] == 1.0
    assert q["placed_pieces"] == 4
    assert q["total_pieces"] == 4
    # mean should exclude the start cell's placeholder 0.0
    assert np.isclose(q["mean_edge_score"], (0.02 + 0.01 + 0.015) / 3)
    assert q["quality_score"] == q["mean_edge_score"]


def test_intrinsic_quality_incomplete_and_sparse():
    grid = {
        (0, 0): {"piece": 0, "rotation": 0, "score": 0.0},
        (0, 5): {"piece": 1, "rotation": 0, "score": 0.3},  # sprawled -> low fill_ratio
    }
    result = _fake_result(grid, (1, 6), unplaced=[2, 3], start_cell=(0, 0))
    q = ev.compute_intrinsic_quality(result)
    assert q["total_pieces"] == 4
    assert q["placed_pieces"] == 2
    assert q["completion_ratio"] == 0.5
    assert np.isclose(q["fill_ratio"], 2 / 6)
    assert q["mean_edge_score"] == 0.3


def test_intrinsic_quality_empty_grid_no_crash():
    result = _fake_result({}, (0, 0), unplaced=[], start_cell=None)
    q = ev.compute_intrinsic_quality(result)
    assert q["total_pieces"] == 0
    assert q["completion_ratio"] == 1.0
    assert q["mean_edge_score"] == 0.0


# ---------------------------------------------------------------------------
# compare_to_ground_truth
# ---------------------------------------------------------------------------

def test_compare_to_ground_truth_perfect_match_no_rotation_needed():
    grid = {
        (0, 0): {"piece": 0, "rotation": 0, "score": 0.0},
        (0, 1): {"piece": 1, "rotation": 2, "score": 0.01},
    }
    result = _fake_result(grid, (1, 2), [], start_cell=(0, 0))
    ground_truth = {0: (0, 0, 0), 1: (0, 1, 2)}
    out = ev.compare_to_ground_truth(result, ground_truth)
    assert out["position_accuracy"] == 1.0
    assert out["orientation_accuracy"] == 1.0
    assert out["best_global_rotation"] == 0


def test_compare_to_ground_truth_needs_global_rotation():
    # assembled grid is the ground truth rotated 90 degrees; the search
    # over k in {0,1,2,3} should find k=... giving perfect position accuracy
    grid = {
        (0, 0): {"piece": 0, "rotation": 0, "score": 0.0},
        (1, 0): {"piece": 1, "rotation": 0, "score": 0.01},
    }
    result = _fake_result(grid, (2, 1), [], start_cell=(0, 0))
    # ground truth has these as a horizontal pair instead of vertical
    ground_truth = {0: (0, 0, 0), 1: (0, 1, 0)}
    out = ev.compare_to_ground_truth(result, ground_truth)
    assert out["position_accuracy"] == 1.0


def test_compare_to_ground_truth_no_overlap_returns_zero():
    grid = {(0, 0): {"piece": 0, "rotation": 0, "score": 0.0}}
    result = _fake_result(grid, (1, 1), [], start_cell=(0, 0))
    out = ev.compare_to_ground_truth(result, ground_truth={})
    assert out["position_accuracy"] == 0.0
    assert out["compared"] == 0


# ---------------------------------------------------------------------------
# reconstruct_image
# ---------------------------------------------------------------------------

def _fake_piece(size=20, color=(200, 100, 50)):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[2:size - 2, 2:size - 2] = 255
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[mask > 0] = color
    return {"image": image, "mask": mask, "orientation_deg": 0.0}


def test_reconstruct_image_output_shape_matches_grid():
    pieces = [_fake_piece(), _fake_piece(), _fake_piece(), _fake_piece()]
    grid = {
        (0, 0): {"piece": 0, "rotation": 0, "score": 0.0},
        (0, 1): {"piece": 1, "rotation": 0, "score": 0.01},
        (1, 0): {"piece": 2, "rotation": 0, "score": 0.01},
        (1, 1): {"piece": 3, "rotation": 0, "score": 0.01},
    }
    result = _fake_result(grid, (2, 2), [], start_cell=(0, 0))
    img = ev.reconstruct_image(pieces, result)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] >= 2 * 16  # roughly 2 cells tall (minus rotation cropping variance)
    assert img.shape[1] >= 2 * 16


def test_reconstruct_image_empty_grid_no_crash():
    result = _fake_result({}, (0, 0), [], start_cell=None)
    img = ev.reconstruct_image([], result)
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_reconstruct_image_applies_rotation():
    # asymmetric piece (distinct quadrant colours) so a 90-degree rotation
    # actually changes the pixel content, unlike a symmetric solid square
    size = 30
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[2:size - 2, 2:size - 2] = 255
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[2:size // 2, 2:size // 2] = [255, 0, 0]         # top-left red
    image[2:size // 2, size // 2:size - 2] = [0, 255, 0]  # top-right green
    image[size // 2:size - 2, :] = [0, 0, 255]            # bottom blue
    piece = {"image": image, "mask": mask, "orientation_deg": 0.0}

    grid0 = {(0, 0): {"piece": 0, "rotation": 0, "score": 0.0}}
    grid1 = {(0, 0): {"piece": 0, "rotation": 1, "score": 0.0}}
    r0 = ev.reconstruct_image([piece], _fake_result(grid0, (1, 1), [], (0, 0)))
    r1 = ev.reconstruct_image([piece], _fake_result(grid1, (1, 1), [], (0, 0)))
    assert r0.shape == r1.shape  # same canvas sizing logic
    assert not np.array_equal(r0, r1)  # but visibly different content due to rotation


def _stripe_row_fraction(img, background=30, brightness_threshold=150):
    """Row index (as a fraction of image height) of a bright horizontal stripe marker."""
    gray = img.mean(axis=2)
    row_means = gray.mean(axis=1)
    bright_rows = np.where(row_means > brightness_threshold)[0]
    if len(bright_rows) == 0:
        return None
    return float(bright_rows.mean()) / img.shape[0]


def test_reconstruct_image_rotation_direction_is_correct():
    # Regression test for a real sign bug: a piece "photographed" pre-
    # rotated by -90 degrees (its white top-stripe marker ends up on the
    # side instead of the top), assigned rotation=1 by assembly, must be
    # rendered back with the stripe on TOP again -- matching where the
    # stripe lands for an equivalent piece that never needed correction
    # (rotation=0). The previous formula's sign error would apply an
    # extra -90 instead of correcting it, leaving the stripe on the
    # BOTTOM instead (a 180-degree-off result) for this exact case.
    size = 30
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[3:27, 3:27] = 255
    image = np.full((size, size, 3), 100, dtype=np.uint8)
    image[mask == 0] = 0
    image[3:7, 3:27] = (255, 255, 255)  # white stripe on what should be "top"

    natural_piece = {"image": image, "mask": mask, "orientation_deg": 0.0}

    pil = Image.fromarray(np.dstack([image, mask]))
    pre_rotated = pil.rotate(-90, expand=True, resample=Image.NEAREST)
    arr = np.array(pre_rotated)
    rotated_piece = {"image": arr[..., :3], "mask": arr[..., 3], "orientation_deg": 0.0}

    grid_natural = {(0, 0): {"piece": 0, "rotation": 0, "score": 0.0}}
    grid_corrected = {(0, 0): {"piece": 0, "rotation": 1, "score": 0.0}}

    r_natural = ev.reconstruct_image([natural_piece], _fake_result(grid_natural, (1, 1), [], (0, 0)))
    r_corrected = ev.reconstruct_image([rotated_piece], _fake_result(grid_corrected, (1, 1), [], (0, 0)))

    frac_natural = _stripe_row_fraction(r_natural)
    frac_corrected = _stripe_row_fraction(r_corrected)
    assert frac_natural is not None and frac_corrected is not None
    assert frac_natural < 0.4   # stripe genuinely near the top in the reference case
    assert abs(frac_natural - frac_corrected) < 0.15  # corrected piece lands in the same place


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
