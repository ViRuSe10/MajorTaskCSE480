"""
test_piece_description.py
Unit tests for src/piece_description.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import piece_description as pd
from src import contour_extraction as ce


# ---------------------------------------------------------------------------
# helpers to build synthetic piece masks
# ---------------------------------------------------------------------------

def _plain_square_mask(size=60, box=(15, 15, 44, 44)):
    """A plain axis-aligned square with 4 flat sides."""
    mask = np.zeros((size, size), dtype=np.uint8)
    y0, x0, y1, x1 = box
    mask[y0:y1 + 1, x0:x1 + 1] = 255
    return mask


def _square_with_tab_and_blank(size=60, box=(15, 15, 44, 44)):
    """
    A square base with:
    - a tab (protrusion) bump added to the TOP side
    - a blank (indentation) notch cut from the BOTTOM side
    - LEFT and RIGHT sides left flat
    """
    mask = _plain_square_mask(size, box)
    y0, x0, y1, x1 = box
    mid_x = (x0 + x1) // 2

    # tab: extend upward beyond the top edge, centered horizontally
    mask[y0 - 6:y0 + 1, mid_x - 4:mid_x + 5] = 255

    # blank: carve a notch out of the bottom edge, centered horizontally
    mask[y1 - 6:y1 + 1, mid_x - 4:mid_x + 5] = 0

    return mask


# ---------------------------------------------------------------------------
# find_corners / split_into_sides
# ---------------------------------------------------------------------------

def test_find_corners_returns_four_distinct_indices():
    mask = _plain_square_mask()
    contour = ce.trace_boundary(mask)
    corners = pd.find_corners(contour)
    assert len(corners) == 4
    assert len(set(corners)) == 4


def test_find_corners_near_actual_square_corners():
    mask = _plain_square_mask(box=(15, 15, 44, 44))
    contour = ce.trace_boundary(mask)
    corners = pd.find_corners(contour)
    corner_points = [contour[i] for i in corners]
    expected = {(15, 15), (15, 44), (44, 15), (44, 44)}
    for (y, x) in corner_points:
        # each detected corner should be close to one of the true corners
        dists = [abs(y - ey) + abs(x - ex) for (ey, ex) in expected]
        assert min(dists) <= 3


def test_split_into_sides_returns_four_sides_covering_contour():
    mask = _plain_square_mask()
    contour = ce.trace_boundary(mask)
    corners = pd.find_corners(contour)
    sides = pd.split_into_sides(contour, corners)
    assert len(sides) == 4
    total_points = sum(len(s) for s in sides)
    # sides share their endpoint corners, so total >= len(contour)
    assert total_points >= len(contour)


# ---------------------------------------------------------------------------
# classify_side
# ---------------------------------------------------------------------------

def test_classify_side_flat_for_plain_square():
    mask = _plain_square_mask()
    contour = ce.trace_boundary(mask)
    corners = pd.find_corners(contour)
    sides = pd.split_into_sides(contour, corners)
    ys, xs = np.where(mask > 0)
    centroid = (ys.mean(), xs.mean())

    types = [pd.classify_side(s, centroid)[0] for s in sides]
    assert all(t == "flat" for t in types), f"expected all flat, got {types}"


def test_classify_side_detects_tab_and_blank():
    mask = _square_with_tab_and_blank()
    contour = ce.trace_boundary(mask)
    corners = pd.find_corners(contour)
    sides = pd.split_into_sides(contour, corners)
    ys, xs = np.where(mask > 0)
    centroid = (ys.mean(), xs.mean())

    types = [pd.classify_side(s, centroid)[0] for s in sides]
    # exactly one tab and one blank should be found among the 4 sides
    assert types.count("tab") == 1, f"types were {types}"
    assert types.count("blank") == 1, f"types were {types}"


def test_classify_side_straight_line_is_flat():
    side = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    result, mag = pd.classify_side(side, piece_centroid=(5, 2))
    assert result == "flat"
    assert mag < 1e-6


def test_classify_side_bulge_is_tab():
    # a side that bulges away from the centroid
    side = [(0, 0), (-3, 2), (0, 4)]
    result, mag = pd.classify_side(side, piece_centroid=(5, 2))
    assert result == "tab"
    assert mag > 0


# ---------------------------------------------------------------------------
# sample_color_strip
# ---------------------------------------------------------------------------

def test_sample_color_strip_output_shape():
    image = np.random.randint(0, 255, (60, 60, 3)).astype(np.uint8)
    side_points = [(20, 20), (20, 25), (20, 30), (20, 35)]
    strip = pd.sample_color_strip(image, side_points, piece_centroid=(30, 30), num_samples=16)
    assert strip.shape == (16, 3)
    assert strip.dtype == np.uint8


def test_sample_color_strip_grayscale_image():
    image = np.random.randint(0, 255, (60, 60)).astype(np.uint8)
    side_points = [(20, 20), (20, 25), (20, 30)]
    strip = pd.sample_color_strip(image, side_points, piece_centroid=(30, 30), num_samples=10)
    assert strip.shape == (10, 1)


def test_sample_color_strip_stays_in_bounds():
    # side right at the image edge -- offset sampling must not go out of range
    image = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    side_points = [(0, 0), (0, 5), (0, 9)]
    strip = pd.sample_color_strip(image, side_points, piece_centroid=(9, 9), offset=4, num_samples=8)
    assert strip.shape == (8, 3)
    assert np.all(strip >= 0) and np.all(strip <= 255)


# ---------------------------------------------------------------------------
# describe_piece / describe_all_pieces
# ---------------------------------------------------------------------------

def test_describe_piece_full_pipeline():
    mask = _square_with_tab_and_blank()
    image = np.random.randint(0, 255, (60, 60, 3)).astype(np.uint8)
    contour = ce.trace_boundary(mask)

    piece = {
        "label": 1,
        "bbox": (0, 0, 59, 59),
        "image": image,
        "mask": mask,
        "contour": contour,
        "orientation_deg": 0.0,
    }
    described = pd.describe_piece(piece)
    assert len(described["corners"]) == 4
    assert len(described["sides"]) == 4
    for side in described["sides"]:
        assert side["type"] in ("tab", "blank", "flat")
        assert side["color_strip"].shape == (32, 3)


def test_describe_all_pieces_count():
    mask = _plain_square_mask()
    image = np.random.randint(0, 255, (60, 60, 3)).astype(np.uint8)
    contour = ce.trace_boundary(mask)
    piece = {
        "label": 1, "bbox": (0, 0, 59, 59), "image": image,
        "mask": mask, "contour": contour, "orientation_deg": 0.0,
    }
    results = pd.describe_all_pieces([piece, piece])
    assert len(results) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
