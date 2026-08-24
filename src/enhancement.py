"""
enhancement.py
================
Image enhancement primitives built from scratch (no cv2.GaussianBlur / cv2.equalizeHist
/ scipy filters — only numpy is used for array storage and vectorised arithmetic).

Functions
---------
- gaussian_kernel(size, sigma)
- convolve2d(image, kernel, mode="reflect")
- gaussian_blur(image, size, sigma)
- median_filter(image, size)
- compute_histogram(image, bins=256)
- histogram_equalization(image)
- contrast_stretching(image, low_pct=2, high_pct=98)
- laplacian_sharpen(image, amount=1.0)
- unsharp_mask(image, size=5, sigma=1.0, amount=1.5)

All functions accept either a 2D grayscale array (H, W) or a 3D colour array
(H, W, C) with dtype uint8 or float. Colour images are processed per-channel
via `_apply_per_channel`.
"""

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _apply_per_channel(image, func, *args, **kwargs):
    """Apply a single-channel function to every channel of a colour image."""
    if image.ndim == 2:
        return func(image, *args, **kwargs)
    channels = [func(image[:, :, c], *args, **kwargs) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def _to_float(image):
    return image.astype(np.float64)


def _to_uint8(image):
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def _pad_image(image, pad_h, pad_w, mode="reflect"):
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)


# ---------------------------------------------------------------------------
# Gaussian kernel + generic convolution
# ---------------------------------------------------------------------------

def gaussian_kernel(size, sigma):
    """
    Build a normalised 2D Gaussian kernel.

    size  : odd integer, kernel width/height.
    sigma : standard deviation of the Gaussian.

    G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    """
    if size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd.")
    half = size // 2
    ax = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def convolve2d(image, kernel, mode="reflect"):
    """
    2D correlation/convolution of a single-channel image with a kernel.
    Implemented from scratch using a sliding window (im2col via stride
    tricks) rather than a per-pixel Python loop, for tractable runtime.
    Padding keeps output the same size as the input.
    """
    image = _to_float(image)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = _pad_image(image, pad_h, pad_w, mode=mode)

    # sliding_window_view gives every kh x kw patch without a manual double loop
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    # windows shape: (H, W, kh, kw)
    result = np.einsum("ijkl,kl->ij", windows, kernel)
    return result


def gaussian_blur(image, size=5, sigma=1.0):
    """Apply Gaussian smoothing. Kernel is derived from `size` and `sigma`."""
    kernel = gaussian_kernel(size, sigma)
    blurred = _apply_per_channel(image, convolve2d, kernel)
    return _to_uint8(blurred)


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------

def median_filter(image, size=3):
    """
    Median filtering for salt-and-pepper style noise.

    Why a loop is required here (and not a convolution):
    the median is a nonlinear, order-statistic operation -- it cannot be
    expressed as a weighted sum of neighbourhood pixels the way Gaussian/
    Laplacian filtering can, so it is not expressible as `convolve2d` with
    a fixed kernel. We still avoid a raw pixel-by-pixel Python loop (which
    would be extremely slow) by building all neighbourhood windows at once
    with `sliding_window_view` and reducing each window with `np.median` in
    a single vectorised call. The one unavoidable "loop" is this internal
    reduction across the window axis, which numpy performs in C.
    """
    def _median_single(chan, size):
        pad = size // 2
        padded = _pad_image(_to_float(chan), pad, pad, mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        # windows: (H, W, size, size) -> flatten last two dims and take median
        med = np.median(windows.reshape(*windows.shape[:2], -1), axis=-1)
        return med

    filtered = _apply_per_channel(image, _median_single, size)
    return _to_uint8(filtered)


# ---------------------------------------------------------------------------
# Histogram + contrast adjustment
# ---------------------------------------------------------------------------

def compute_histogram(image, bins=256):
    """
    Compute the intensity histogram of a single-channel image from scratch
    (no np.histogram). Returns an array of length `bins`.
    """
    if image.ndim != 2:
        raise ValueError("compute_histogram expects a single-channel image.")
    hist = np.zeros(bins, dtype=np.int64)
    flat = image.astype(np.int64).ravel()
    # bincount is a vectorised counting op, not a hidden histogram function
    counts = np.bincount(flat, minlength=bins)
    hist[: len(counts)] = counts[:bins]
    return hist


def histogram_equalization(image):
    """
    Classic global histogram equalization using the CDF of the image's
    own histogram (computed via compute_histogram).
    """
    def _eq_single(chan):
        chan_u8 = _to_uint8(chan)
        hist = compute_histogram(chan_u8, bins=256)
        cdf = np.cumsum(hist).astype(np.float64)
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
        total = chan_u8.size
        # standard equalization mapping
        lut = np.round((cdf - cdf_min) / max(total - cdf_min, 1) * 255)
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return lut[chan_u8]

    result = _apply_per_channel(image, _eq_single)
    return result.astype(np.uint8)


def contrast_stretching(image, low_pct=2, high_pct=98):
    """
    Linear contrast stretch: map the [low_pct, high_pct] intensity
    percentiles to [0, 255]. Using percentiles (rather than raw min/max)
    makes the stretch robust to a few outlier pixels.
    """
    def _stretch_single(chan):
        chan_f = _to_float(chan)
        lo = np.percentile(chan_f, low_pct)
        hi = np.percentile(chan_f, high_pct)
        if hi - lo < 1e-6:
            return chan_f
        stretched = (chan_f - lo) * (255.0 / (hi - lo))
        return np.clip(stretched, 0, 255)

    result = _apply_per_channel(image, _stretch_single)
    return _to_uint8(result)


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------

_LAPLACIAN_KERNEL = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=np.float64)


def laplacian_sharpen(image, amount=1.0):
    """
    Sharpen by subtracting a scaled Laplacian (built on convolve2d) from
    the original image: sharpened = original - amount * laplacian.
    """
    def _sharp_single(chan):
        chan_f = _to_float(chan)
        lap = convolve2d(chan_f, _LAPLACIAN_KERNEL)
        return chan_f - amount * lap

    result = _apply_per_channel(image, _sharp_single)
    return _to_uint8(result)


def unsharp_mask(image, size=5, sigma=1.0, amount=1.5):
    """
    Unsharp masking: sharpened = original + amount * (original - blurred).
    Built directly on gaussian_blur / convolve2d.
    """
    def _unsharp_single(chan):
        chan_f = _to_float(chan)
        kernel = gaussian_kernel(size, sigma)
        blurred = convolve2d(chan_f, kernel)
        return chan_f + amount * (chan_f - blurred)

    result = _apply_per_channel(image, _unsharp_single)
    return _to_uint8(result)
