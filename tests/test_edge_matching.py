"""
test_edge_matching.py
Unit tests for src/edge_matching.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import edge_matching as em


# ---------------------------------------------------------------------------
# side_shape_profile
# ---------------------------------------------------------------------------

def test_side_shape_profile_flat_line_is_zero():
    side = [(0, 0), (0, 2), (0, 4)]
    profile, length = em.side_shape_profile(side, piece_centroid=(5, 2))
    assert np.allclose(profile, 0, atol=1e-6)
    assert length > 0


def test_side_shape_profile_bulge_outward_positive():
    # centroid far below -> "outward" means away from (5,2), i.e. upward (negative row)
    side = [(0, 0), (-3, 2), (0, 4)]
    profile, length = em.side_shape_profile(side, piece_centroid=(5, 2))
    assert profile.max() > 0


# ---------------------------------------------------------------------------
# shape_compatibility
# ---------------------------------------------------------------------------

def test_shape_compatibility_perfect_complementary_fit_near_zero():
    profile_a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    profile_b = -profile_a[::-1]  # exact complementary shape when reversed
    score = em.shape_compatibility(profile_a, profile_b, ref_length=10.0)
    assert score < 1e-9


def test_shape_compatibility_mismatched_shapes_scores_higher():
    profile_a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    profile_b_good = -profile_a[::-1]
    profile_b_bad = np.array([0.0, -0.2, -0.1, -0.3, 0.0])
    score_good = em.shape_compatibility(profile_a, profile_b_good, ref_length=10.0)
    score_bad = em.shape_compatibility(profile_a, profile_b_bad, ref_length=10.0)
    assert score_bad > score_good


# ---------------------------------------------------------------------------
# color_ssd
# ---------------------------------------------------------------------------

def test_color_ssd_identical_reversed_strips_is_zero():
    strip = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    strip_b = strip[::-1].copy()  # reversed copy -> after internal reversal, matches strip exactly
    score = em.color_ssd(strip, strip_b)
    assert score < 1e-9


def test_color_ssd_very_different_strips_scores_higher():
    strip_a = np.zeros((10, 3), dtype=np.uint8)
    strip_b_close = np.full((10, 3), 5, dtype=np.uint8)
    strip_b_far = np.full((10, 3), 255, dtype=np.uint8)
    score_close = em.color_ssd(strip_a, strip_b_close)
    score_far = em.color_ssd(strip_a, strip_b_far)
    assert score_far > score_close


def test_color_ssd_bounded_near_one_at_max_difference():
    strip_a = np.zeros((5, 3), dtype=np.uint8)
    strip_b = np.full((5, 3), 255, dtype=np.uint8)
    score = em.color_ssd(strip_a, strip_b)
    assert np.isclose(score, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# match_score
# ---------------------------------------------------------------------------

def _make_side(side_type, points, color_strip):
    return {"type": side_type, "points": points, "color_strip": np.array(color_strip, dtype=np.uint8)}


def test_match_score_rejects_non_complementary_types():
    tab = _make_side("tab", [(0, 0), (-2, 2), (0, 4)], np.zeros((32, 3)))
    tab2 = _make_side("tab", [(0, 0), (-2, 2), (0, 4)], np.zeros((32, 3)))
    score = em.match_score(tab, tab2, (5, 2), (5, 2), num_samples=8)
    assert score == float("inf")


def test_match_score_rejects_flat_pairs():
    flat = _make_side("flat", [(0, 0), (0, 2), (0, 4)], np.zeros((32, 3)))
    blank = _make_side("blank", [(0, 0), (2, 2), (0, 4)], np.zeros((32, 3)))
    score = em.match_score(flat, blank, (5, 2), (-5, 2), num_samples=8)
    assert score == float("inf")


def test_match_score_good_fit_scores_lower_than_bad_fit():
    # tab bulges "up" (away from its own centroid below at row 5)
    tab_points = [(0, 0), (-3, 2), (0, 4)]
    tab_centroid = (5, 2)
    # good blank: same raw points, but its centroid sits on the OPPOSITE
    # side (row -5) -- this flips the outward-normal direction, so the
    # identical point (-3, 2) now reads as an *inward* dip of matching
    # magnitude (3) relative to the blank piece's own centroid, i.e. a
    # geometrically complementary fit for the tab above.
    good_blank_points = [(0, 0), (-3, 2), (0, 4)]
    good_blank_centroid = (-5, 2)
    # bad blank: much shallower dip -> poor geometric fit (wrong depth)
    bad_blank_points = [(0, 0), (-0.3, 2), (0, 4)]
    bad_blank_centroid = (-5, 2)

    color = [[100, 100, 100]] * 32
    tab = _make_side("tab", tab_points, color)
    good_blank = _make_side("blank", good_blank_points, color[::-1])
    bad_blank = _make_side("blank", bad_blank_points, color[::-1])

    score_good = em.match_score(tab, good_blank, tab_centroid, good_blank_centroid, num_samples=16)
    score_bad = em.match_score(tab, bad_blank, tab_centroid, bad_blank_centroid, num_samples=16)
    assert score_good < score_bad


def test_match_score_weights_affect_result():
    tab_points = [(0, 0), (-3, 2), (0, 4)]
    # same coordinates, opposite-side centroid -> geometrically complementary
    blank_points = [(0, 0), (-3, 2), (0, 4)]
    tab = _make_side("tab", tab_points, [[0, 0, 0]] * 16)
    blank = _make_side("blank", blank_points, [[255, 255, 255]] * 16)

    score_shape_only = em.match_score(tab, blank, (5, 2), (-5, 2), num_samples=16, alpha=1.0, beta=0.0)
    score_color_only = em.match_score(tab, blank, (5, 2), (-5, 2), num_samples=16, alpha=0.0, beta=1.0)
    # shape-only should be near-zero (good geometric fit), colour-only should be high (black vs white)
    assert score_shape_only < 0.05
    assert score_color_only > 0.5


# ---------------------------------------------------------------------------
# compute_all_scores / best_matches_for_side
# ---------------------------------------------------------------------------

def _fake_piece(mask_box, sides):
    mask = np.zeros((20, 20), dtype=np.uint8)
    y0, x0, y1, x1 = mask_box
    mask[y0:y1, x0:x1] = 255
    return {"mask": mask, "sides": sides}


def test_compute_all_scores_skips_same_piece_and_flat():
    color = [[50, 50, 50]] * 8
    side_tab = _make_side("tab", [(0, 0), (-2, 1), (0, 2)], color)
    side_blank = _make_side("blank", [(0, 0), (2, 1), (0, 2)], color[::-1])
    side_flat = _make_side("flat", [(0, 0), (0, 1), (0, 2)], color)

    piece0 = _fake_piece((2, 2, 6, 6), [side_tab, side_flat, side_flat, side_flat])
    piece1 = _fake_piece((10, 10, 14, 14), [side_blank, side_flat, side_flat, side_flat])

    scores = em.compute_all_scores([piece0, piece1], num_samples=8)
    # only the (0,0)-(1,0) tab/blank pair is valid; no self-pairs, no flat pairs
    assert set(scores.keys()) == {(0, 0, 1, 0)}


def test_best_matches_for_side_returns_sorted_candidates():
    scores = {
        (0, 0, 1, 0): 0.5,
        (0, 0, 2, 0): 0.1,
        (0, 0, 3, 0): 0.9,
    }
    best = em.best_matches_for_side(pieces=None, piece_idx=0, side_idx=0, scores=scores, top_k=2)
    assert best[0][1] == 0.1
    assert best[1][1] == 0.5
    assert len(best) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
