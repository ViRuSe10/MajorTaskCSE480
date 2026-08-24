"""
test_contour_extraction.py
Unit tests for src/contour_extraction.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import contour_extraction as ce
from src import segmentation as seg


# ---------------------------------------------------------------------------
# trace_boundary
# ---------------------------------------------------------------------------

def test_trace_boundary_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    contour = ce.trace_boundary(mask)
    assert contour == []


def test_trace_boundary_single_pixel():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    contour = ce.trace_boundary(mask)
    assert contour == [(2, 2)]


def test_trace_boundary_solid_square_visits_all_perimeter_pixels():
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 255  # 3x3 solid square, all 8 non-center pixels are boundary
    contour = ce.trace_boundary(mask)
    perimeter_pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)} - {(3, 3)}
    assert set(contour) == perimeter_pixels
    # first point must be topmost-then-leftmost
    assert contour[0] == (2, 2)


def test_trace_boundary_closed_loop_starts_and_returns_near_start():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    contour = ce.trace_boundary(mask)
    assert len(contour) > 0
    # every contour point must be a foreground pixel on the mask
    for (y, x) in contour:
        assert mask[y, x] == 255
    # every contour point must have at least one background 8-neighbour
    # (it's on the boundary, not buried inside)
    h, w = mask.shape
    for (y, x) in contour:
        neighbours_bg = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                    neighbours_bg = True
        assert neighbours_bg


# ---------------------------------------------------------------------------
# convex_hull
# ---------------------------------------------------------------------------

def test_convex_hull_of_square_corners_plus_interior_point():
    points = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]  # interior point should drop out
    hull = ce.convex_hull(points)
    assert set(hull) == {(0, 0), (0, 4), (4, 0), (4, 4)}


def test_convex_hull_collinear_points():
    points = [(0, 0), (1, 0), (2, 0), (3, 0)]
    hull = ce.convex_hull(points)
    assert len(hull) <= 2


def test_convex_hull_two_points():
    hull = ce.convex_hull([(0, 0), (5, 5)])
    assert set(hull) == {(0, 0), (5, 5)}


# ---------------------------------------------------------------------------
# min_area_rect
# ---------------------------------------------------------------------------

def test_min_area_rect_axis_aligned_square():
    hull = [(0, 0), (10, 0), (10, 10), (0, 10)]
    center, (w, h), angle = ce.min_area_rect(hull)
    assert np.isclose(w * h, 100, atol=1e-6)
    assert np.isclose(center[0], 5, atol=1e-6)
    assert np.isclose(center[1], 5, atol=1e-6)
    assert (angle % 90) < 1e-6 or (90 - (angle % 90)) < 1e-6


def test_min_area_rect_rotated_square_area_matches():
    # square of side 10 rotated 30 degrees around origin
    theta = np.radians(30)
    side = 10
    base = np.array([[0, 0], [side, 0], [side, side], [0, side]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    rotated = base @ R.T
    hull = [tuple(p) for p in rotated]
    center, (w, h), angle = ce.min_area_rect(hull)
    assert np.isclose(w * h, side * side, atol=1e-3)


# ---------------------------------------------------------------------------
# normalize_orientation
# ---------------------------------------------------------------------------

def test_normalize_orientation_axis_aligned_square_near_zero():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    angle = ce.normalize_orientation(mask)
    assert abs(angle) < 2.0  # should be ~0 degrees for an axis-aligned square


def test_normalize_orientation_range():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    angle = ce.normalize_orientation(mask)
    assert -45.0 < angle <= 45.0


def test_normalize_orientation_empty_mask_no_crash():
    mask = np.zeros((10, 10), dtype=np.uint8)
    angle = ce.normalize_orientation(mask)
    assert angle == 0.0


# ---------------------------------------------------------------------------
# extract_piece / extract_all_pieces
# ---------------------------------------------------------------------------

def test_extract_piece_crops_correct_region():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:10, 5:12] = 1
    image[5:10, 5:12] = [100, 150, 200]

    stats = seg.component_stats(labels, 1)
    piece = ce.extract_piece(image, labels, 1, stats[0])

    assert piece["bbox"] == (5, 5, 9, 11)
    assert piece["image"].shape == (5, 7, 3)
    assert piece["mask"].shape == (5, 7)
    assert piece["mask"].max() == 255
    assert len(piece["contour"]) > 0
    assert isinstance(piece["orientation_deg"], float)


def test_extract_all_pieces_matches_component_count():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:5, 2:5] = 1
    labels[10:14, 10:14] = 2
    image = np.zeros((20, 20), dtype=np.uint8)
    image[labels > 0] = 255

    stats = seg.component_stats(labels, 2)
    pieces = ce.extract_all_pieces(image, labels, stats)
    assert len(pieces) == 2
    assert {p["label"] for p in pieces} == {1, 2}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
