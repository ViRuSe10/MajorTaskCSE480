"""
test_thresholding.py
Unit tests for src/thresholding.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import thresholding as th


# ---------------------------------------------------------------------------
# global_threshold
# ---------------------------------------------------------------------------

def test_global_threshold_basic_split():
    img = np.array([[10, 200], [50, 220]], dtype=np.uint8)
    mask = th.global_threshold(img, 100)
    expected = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    assert np.array_equal(mask, expected)


def test_global_threshold_output_is_binary():
    img = np.random.randint(0, 255, (20, 20)).astype(np.uint8)
    mask = th.global_threshold(img, 127)
    assert set(np.unique(mask)).issubset({0, 255})


def test_global_threshold_rejects_color_image():
    img = np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    with pytest.raises(ValueError):
        th.global_threshold(img, 127)


# ---------------------------------------------------------------------------
# otsu_threshold
# ---------------------------------------------------------------------------

def test_otsu_separates_bimodal_image():
    # two clear clusters: dark background (~30) and bright foreground (~220)
    rng = np.random.default_rng(0)
    dark = rng.integers(20, 40, (30, 30))
    bright = rng.integers(210, 230, (30, 30))
    img = np.vstack([dark, bright]).astype(np.uint8)

    mask, t = th.otsu_threshold(img)
    assert 35 <= t <= 215
    # top half (dark) should be mostly background(0), bottom half (bright) mostly foreground(255)
    assert mask[:30].mean() < 50
    assert mask[30:].mean() > 200


def test_otsu_returns_binary_mask():
    img = np.random.randint(0, 255, (25, 25)).astype(np.uint8)
    mask, t = th.otsu_threshold(img)
    assert set(np.unique(mask)).issubset({0, 255})
    assert 0 <= t <= 255


def test_otsu_flat_image_no_crash():
    img = np.full((10, 10), 100, dtype=np.uint8)
    mask, t = th.otsu_threshold(img)
    assert mask.shape == img.shape


# ---------------------------------------------------------------------------
# adaptive_threshold
# ---------------------------------------------------------------------------

def test_adaptive_threshold_mean_handles_illumination_gradient():
    # illumination ramp from 50 (left) to 200 (right), with a fixed-offset
    # foreground blob riding on top of it -- global threshold would fail,
    # adaptive should still recover the blob locally.
    h, w = 60, 60
    ramp = np.tile(np.linspace(50, 200, w), (h, 1))
    img = ramp.copy()
    img[20:40, 20:40] += 40  # local bright blob relative to local background
    img = np.clip(img, 0, 255).astype(np.uint8)

    mask = th.adaptive_threshold(img, block_size=15, C=5, method="mean")
    assert set(np.unique(mask)).issubset({0, 255})
    # blob region should be predominantly foreground
    assert mask[20:40, 20:40].mean() > 150


def test_adaptive_threshold_gaussian_output_shape():
    img = np.random.randint(0, 255, (40, 40)).astype(np.uint8)
    mask = th.adaptive_threshold(img, block_size=11, C=3, method="gaussian")
    assert mask.shape == img.shape
    assert set(np.unique(mask)).issubset({0, 255})


def test_adaptive_threshold_rejects_bad_method():
    img = np.random.randint(0, 255, (20, 20)).astype(np.uint8)
    with pytest.raises(ValueError):
        th.adaptive_threshold(img, method="bogus")


def test_adaptive_threshold_rejects_even_block_size():
    img = np.random.randint(0, 255, (20, 20)).astype(np.uint8)
    with pytest.raises(ValueError):
        th.adaptive_threshold(img, block_size=10)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
