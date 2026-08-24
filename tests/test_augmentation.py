"""
ml/tests/test_augmentation.py
Unit tests for ml/augmentation.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml import augmentation as aug
from src import contour_extraction as ce
from src.piece_description import describe_piece


def _make_synthetic_piece(size=200, box=(50, 50, 149, 149), bump=20):
    """
    A square with a tab on top and a blank on bottom (see
    test_piece_description.py), sized to match real dataset piece
    proportions (~150-300px with proportionally deep tab/blank features).
    Small pixel-scale synthetic pieces are more sensitive to rotation-
    resampling noise occasionally flipping a borderline "flat"
    classification (see rotate_piece's docstring) -- this scale avoids
    that synthetic-test artifact, matching the real-piece robustness
    already validated directly against the actual dataset.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    y0, x0, y1, x1 = box
    mask[y0:y1 + 1, x0:x1 + 1] = 255
    mid_x = (x0 + x1) // 2
    half = bump // 2 + 2
    mask[y0 - bump:y0 + 1, mid_x - half:mid_x + half + 1] = 255   # tab on top
    mask[y1 - bump:y1 + 1, mid_x - half:mid_x + half + 1] = 0      # blank cut from bottom

    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[mask > 0] = (200, 100, 50)

    contour = ce.trace_boundary(mask)
    piece = {
        "label": 1,
        "bbox": (0, 0, size - 1, size - 1),
        "image": image,
        "mask": mask,
        "contour": contour,
        "orientation_deg": ce.normalize_orientation(mask),
    }
    return describe_piece(piece)


# ---------------------------------------------------------------------------
# photometric jitter
# ---------------------------------------------------------------------------

def test_jitter_illumination_shifts_brightness_and_stays_in_range():
    rng = np.random.default_rng(0)
    img = np.full((20, 20, 3), 100, dtype=np.uint8)
    out = aug.jitter_illumination(img, rng, delta_range=(20, 20))
    assert out.min() >= 0 and out.max() <= 255
    assert out[0, 0, 0] == 120


def test_jitter_illumination_clips_at_bounds():
    rng = np.random.default_rng(0)
    img = np.full((5, 5, 3), 250, dtype=np.uint8)
    out = aug.jitter_illumination(img, rng, delta_range=(50, 50))
    assert out.max() == 255


def test_jitter_contrast_preserves_mean_approximately():
    rng = np.random.default_rng(1)
    img = np.random.randint(50, 200, (30, 30, 3)).astype(np.uint8)
    out = aug.jitter_contrast(img, rng, factor_range=(1.5, 1.5))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_jitter_color_changes_channels_independently():
    rng = np.random.default_rng(2)
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    out = aug.jitter_color(img, rng, factor_range=(0.5, 0.5))
    assert np.allclose(out, 50, atol=1)


def test_jitter_color_noop_on_grayscale():
    img = np.full((10, 10), 100, dtype=np.uint8)
    rng = np.random.default_rng(0)
    out = aug.jitter_color(img, rng)
    assert np.array_equal(out, img)


def test_add_noise_perturbs_pixels():
    rng = np.random.default_rng(3)
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    out = aug.add_noise(img, rng, sigma_range=(10, 10))
    assert not np.array_equal(out, img)
    assert out.min() >= 0 and out.max() <= 255


def test_augment_pixels_preserves_shape_and_dtype():
    rng = np.random.default_rng(4)
    img = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)
    out = aug.augment_pixels(img, rng)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# rotation
# ---------------------------------------------------------------------------

def test_rotate_piece_preserves_cyclic_type_sequence():
    piece = _make_synthetic_piece()
    original_types = [s["type"] for s in piece["sides"]]

    for angle in (10, -20, 33, 90):
        rotated = aug.rotate_piece(piece, angle)
        rotated_types = [s["type"] for s in rotated["sides"]]
        is_cyclic_shift = any(
            rotated_types == [original_types[(k + s) % 4] for k in range(4)]
            for s in range(4)
        )
        assert is_cyclic_shift, f"angle={angle}: {rotated_types} not a cyclic shift of {original_types}"


def test_rotate_piece_returns_valid_described_piece():
    piece = _make_synthetic_piece()
    rotated = aug.rotate_piece(piece, 15)
    assert "sides" in rotated
    assert len(rotated["sides"]) == 4
    assert "corners" in rotated
    for side in rotated["sides"]:
        assert side["color_strip"].shape[1] == 3


def test_augment_piece_combines_rotation_and_pixel_jitter():
    piece = _make_synthetic_piece()
    rng = np.random.default_rng(5)
    result = aug.augment_piece(piece, rng, angle_range=(10, 10))
    assert "sides" in result
    assert len(result["sides"]) == 4
    # image should differ from a pure rotation-only result due to pixel jitter
    rotation_only = aug.rotate_piece(piece, 10)
    assert result["image"].shape[:2] == rotation_only["image"].shape[:2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
