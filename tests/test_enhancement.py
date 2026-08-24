"""
test_enhancement.py
Unit tests for src/enhancement.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import enhancement as enh


# ---------------------------------------------------------------------------
# gaussian_kernel
# ---------------------------------------------------------------------------

def test_gaussian_kernel_shape_and_normalisation():
    k = enh.gaussian_kernel(5, 1.0)
    assert k.shape == (5, 5)
    assert np.isclose(k.sum(), 1.0)


def test_gaussian_kernel_peak_at_center():
    k = enh.gaussian_kernel(5, 1.0)
    center = k[2, 2]
    assert center == k.max()


def test_gaussian_kernel_rejects_even_size():
    with pytest.raises(ValueError):
        enh.gaussian_kernel(4, 1.0)


# ---------------------------------------------------------------------------
# convolve2d / gaussian_blur
# ---------------------------------------------------------------------------

def test_convolve2d_identity_kernel_preserves_image():
    img = np.random.randint(0, 255, (20, 20)).astype(np.float64)
    identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
    out = enh.convolve2d(img, identity)
    assert np.allclose(out, img)


def test_gaussian_blur_reduces_variance_on_noisy_image():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (50, 50)).astype(np.uint8)
    blurred = enh.gaussian_blur(noisy, size=5, sigma=1.5)
    assert blurred.shape == noisy.shape
    assert blurred.std() < noisy.std()


def test_gaussian_blur_output_dtype_and_range():
    img = np.full((10, 10), 128, dtype=np.uint8)
    out = enh.gaussian_blur(img, size=3, sigma=1.0)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    # a flat image should stay (approximately) flat after blurring
    assert np.allclose(out, 128, atol=1)


def test_gaussian_blur_color_image_shape_preserved():
    img = np.random.randint(0, 255, (16, 16, 3)).astype(np.uint8)
    out = enh.gaussian_blur(img, size=3, sigma=1.0)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# median_filter
# ---------------------------------------------------------------------------

def test_median_filter_removes_salt_and_pepper_noise():
    img = np.full((30, 30), 100, dtype=np.uint8)
    rng = np.random.default_rng(1)
    noisy = img.copy()
    coords = rng.choice(30 * 30, size=40, replace=False)
    noisy_flat = noisy.ravel()
    noisy_flat[coords[:20]] = 0
    noisy_flat[coords[20:]] = 255
    noisy = noisy_flat.reshape(30, 30)

    denoised = enh.median_filter(noisy, size=3)
    # most of the impulse noise should be gone -> much closer to the flat 100 image
    assert np.abs(denoised.astype(int) - 100).mean() < np.abs(noisy.astype(int) - 100).mean()


def test_median_filter_shape_preserved():
    img = np.random.randint(0, 255, (12, 12)).astype(np.uint8)
    out = enh.median_filter(img, size=3)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# histogram + contrast
# ---------------------------------------------------------------------------

def test_compute_histogram_total_count_matches_pixels():
    img = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    hist = enh.compute_histogram(img)
    assert hist.sum() == img.size
    assert hist.shape == (256,)


def test_compute_histogram_single_value_image():
    img = np.full((5, 5), 42, dtype=np.uint8)
    hist = enh.compute_histogram(img)
    assert hist[42] == 25
    assert hist.sum() == 25


def test_histogram_equalization_increases_or_preserves_spread():
    # low-contrast synthetic image confined to [100, 120]
    img = np.random.randint(100, 120, (40, 40)).astype(np.uint8)
    eq = enh.histogram_equalization(img)
    assert eq.shape == img.shape
    assert (eq.max() - eq.min()) >= (img.max() - img.min())


def test_contrast_stretching_expands_range_to_0_255():
    img = np.random.randint(100, 120, (40, 40)).astype(np.uint8)
    stretched = enh.contrast_stretching(img, low_pct=0, high_pct=100)
    assert stretched.min() <= 5
    assert stretched.max() >= 250


def test_contrast_stretching_flat_image_no_crash():
    img = np.full((10, 10), 77, dtype=np.uint8)
    out = enh.contrast_stretching(img)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# sharpening
# ---------------------------------------------------------------------------

def test_laplacian_sharpen_shape_and_dtype():
    img = np.random.randint(0, 255, (20, 20)).astype(np.uint8)
    out = enh.laplacian_sharpen(img, amount=1.0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_unsharp_mask_increases_high_frequency_energy():
    # a step edge image -- sharpening should increase local contrast at the edge
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 200
    sharpened = enh.unsharp_mask(img, size=5, sigma=1.0, amount=1.5)
    # gradient magnitude near the edge column should increase after sharpening
    orig_grad = np.abs(np.diff(img[:, 8:12].astype(int), axis=1)).sum()
    sharp_grad = np.abs(np.diff(sharpened[:, 8:12].astype(int), axis=1)).sum()
    assert sharp_grad >= orig_grad


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
