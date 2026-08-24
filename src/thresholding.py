"""
thresholding.py
================
Binarization methods built from scratch, used to separate puzzle pieces
from the background after enhancement.

Functions
---------
- global_threshold(image, thresh)
- otsu_threshold(image)              -> (binary_mask, chosen_threshold)
- adaptive_threshold(image, block_size, C, method="mean")

All functions expect a single-channel (grayscale) uint8 image and return a
binary mask with values in {0, 255}.
"""

import numpy as np

from .enhancement import compute_histogram, gaussian_kernel, convolve2d, _to_float


# ---------------------------------------------------------------------------
# Global thresholding
# ---------------------------------------------------------------------------

def global_threshold(image, thresh):
    """
    Simple fixed-value binarization: pixels >= thresh -> 255, else 0.
    """
    if image.ndim != 2:
        raise ValueError("global_threshold expects a single-channel image.")
    mask = np.where(image >= thresh, 255, 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Otsu's method
# ---------------------------------------------------------------------------

def otsu_threshold(image):
    """
    Otsu's method: choose the threshold t in [0, 255] that maximises the
    between-class variance of the foreground/background split implied by
    the image histogram (computed via enhancement.compute_histogram).

    sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2

    Returns
    -------
    (binary_mask, best_t)
    """
    if image.ndim != 2:
        raise ValueError("otsu_threshold expects a single-channel image.")

    hist = compute_histogram(image, bins=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return np.zeros_like(image, dtype=np.uint8), 0

    prob = hist / total
    intensities = np.arange(256)

    # cumulative sums for class 0 (below threshold) at every possible t
    w0 = np.cumsum(prob)                         # P(intensity <= t)
    mu_cumsum = np.cumsum(prob * intensities)     # cumulative first moment
    mu_total = mu_cumsum[-1]

    w1 = 1.0 - w0
    # avoid divide-by-zero for t where a class is empty
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, mu_cumsum / w0, 0.0)
        mu1 = np.where(w1 > 0, (mu_total - mu_cumsum) / w1, 0.0)

    between_class_var = w0 * w1 * (mu0 - mu1) ** 2
    best_t = int(np.argmax(between_class_var))

    mask = global_threshold(image, best_t)
    return mask, best_t


# ---------------------------------------------------------------------------
# Adaptive thresholding
# ---------------------------------------------------------------------------

def _local_mean_map(image, block_size):
    """Per-pixel local mean via a uniform (box) convolution."""
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd.")
    kernel = np.ones((block_size, block_size), dtype=np.float64) / (block_size ** 2)
    return convolve2d(_to_float(image), kernel)


def _local_gaussian_map(image, block_size, sigma=None):
    """Per-pixel Gaussian-weighted local mean."""
    if sigma is None:
        sigma = block_size / 6.0 if block_size > 1 else 1.0
    kernel = gaussian_kernel(block_size, sigma)
    return convolve2d(_to_float(image), kernel)


def adaptive_threshold(image, block_size=15, C=5, method="mean"):
    """
    Adaptive (local) thresholding: each pixel is compared against the mean
    (or Gaussian-weighted mean) of its own neighbourhood, minus a constant C.
    Useful when illumination varies across the scene (e.g. shadows across
    a tray of scattered pieces), where a single global/Otsu threshold fails.

    method : "mean" or "gaussian"
    """
    if image.ndim != 2:
        raise ValueError("adaptive_threshold expects a single-channel image.")
    if method == "mean":
        local_map = _local_mean_map(image, block_size)
    elif method == "gaussian":
        local_map = _local_gaussian_map(image, block_size)
    else:
        raise ValueError("method must be 'mean' or 'gaussian'.")

    mask = np.where(_to_float(image) >= (local_map - C), 255, 0).astype(np.uint8)
    return mask
