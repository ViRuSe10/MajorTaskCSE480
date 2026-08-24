"""
test_edge_detection.py
Unit tests for src/edge_detection.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import edge_detection as ed


# ---------------------------------------------------------------------------
# Sobel / Prewitt
# ---------------------------------------------------------------------------

def _vertical_edge_image(h=20, w=20, split=10):
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, split:] = 255
    return img


def test_sobel_detects_vertical_edge():
    img = _vertical_edge_image()
    mag, orient = ed.sobel_edges(img)
    assert mag.shape == img.shape
    # strongest response should be at/near the edge column
    col_response = mag.sum(axis=0)
    edge_col = np.argmax(col_response)
    assert 8 <= edge_col <= 12


def test_sobel_flat_image_zero_magnitude():
    img = np.full((15, 15), 100, dtype=np.uint8)
    mag, orient = ed.sobel_edges(img)
    assert np.allclose(mag, 0, atol=1e-6)


def test_prewitt_detects_vertical_edge():
    img = _vertical_edge_image()
    mag, orient = ed.prewitt_edges(img)
    col_response = mag.sum(axis=0)
    edge_col = np.argmax(col_response)
    assert 8 <= edge_col <= 12


def test_sobel_rejects_color_image():
    img = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    with pytest.raises(ValueError):
        ed.sobel_edges(img)


def test_orientation_range():
    img = np.random.randint(0, 255, (20, 20)).astype(np.uint8)
    mag, orient = ed.sobel_edges(img)
    assert orient.min() >= 0
    assert orient.max() < 180


# ---------------------------------------------------------------------------
# non_max_suppression
# ---------------------------------------------------------------------------

def test_nms_thins_a_thick_edge_band():
    # a gradual ramp produces a "thick" band of nonzero gradient in Sobel;
    # NMS should narrow it down to a thin ridge.
    img = np.zeros((20, 20), dtype=np.uint8)
    for i in range(20):
        img[:, i] = min(255, i * 30)
    mag, orient = ed.sobel_edges(img)
    suppressed = ed.non_max_suppression(mag, orient)
    assert suppressed.shape == mag.shape
    assert (suppressed > 0).sum() <= (mag > 0).sum()


def test_nms_preserves_local_maximum():
    img = _vertical_edge_image()
    mag, orient = ed.sobel_edges(img)
    suppressed = ed.non_max_suppression(mag, orient)
    # the edge column should still have nonzero response after suppression
    assert suppressed[:, 9:12].max() > 0


# ---------------------------------------------------------------------------
# double_threshold / hysteresis
# ---------------------------------------------------------------------------

def test_double_threshold_classifies_correctly():
    img = np.array([[0, 60, 80], [150, 200, 255]], dtype=np.float64)
    result, weak, strong = ed.double_threshold(img, low=50, high=150)
    assert result[0, 0] == 0
    assert result[0, 1] == weak
    assert result[0, 2] == weak
    assert result[1, 0] == strong
    assert result[1, 1] == strong
    assert result[1, 2] == strong


def test_hysteresis_promotes_connected_weak_pixel():
    result = np.zeros((5, 5), dtype=np.uint8)
    result[2, 2] = ed.STRONG
    result[2, 3] = ed.WEAK  # adjacent to strong -> should be promoted
    result[0, 0] = ed.WEAK  # isolated -> should be suppressed
    out = ed.hysteresis(result)
    assert out[2, 3] == 255
    assert out[0, 0] == 0
    assert out[2, 2] == 255


def test_hysteresis_multi_hop_chain():
    # chain of weak pixels leading to one strong pixel; all should be promoted
    result = np.zeros((1, 6), dtype=np.uint8)
    result[0, 0] = ed.STRONG
    result[0, 1:5] = ed.WEAK
    out = ed.hysteresis(result)
    assert np.all(out[0, 0:5] == 255)


# ---------------------------------------------------------------------------
# full canny pipeline
# ---------------------------------------------------------------------------

def test_canny_detects_edge_on_synthetic_square():
    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:30, 10:30] = 255
    edges = ed.canny(img, gaussian_size=5, sigma=1.0)
    assert edges.shape == img.shape
    assert set(np.unique(edges)).issubset({0, 255})
    # some edge pixels should be found around the square boundary
    assert edges.sum() > 0


def test_canny_flat_image_no_edges():
    img = np.full((20, 20), 128, dtype=np.uint8)
    edges = ed.canny(img)
    assert edges.sum() == 0


def test_canny_rejects_color_image():
    img = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    with pytest.raises(ValueError):
        ed.canny(img)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
