"""
edge_detection.py
==================
Gradient-based edge detectors built from scratch on top of
enhancement.convolve2d / enhancement.gaussian_kernel.

Functions
---------
- sobel_edges(image)               -> (magnitude, orientation)
- prewitt_edges(image)             -> (magnitude, orientation)
- non_max_suppression(magnitude, orientation)
- double_threshold(image, low, high) -> (result, weak_val, strong_val)
- hysteresis(result, weak_val, strong_val)
- canny(image, gaussian_size=5, sigma=1.0, low_ratio=0.05, high_ratio=0.15)

Orientation is returned in degrees, in the range [0, 180) (gradient
direction mod 180, since an edge and its 180-degree-opposite are the
same edge).
"""

import numpy as np

from .enhancement import convolve2d, gaussian_kernel, _to_float, _to_uint8


# ---------------------------------------------------------------------------
# Sobel / Prewitt
# ---------------------------------------------------------------------------

_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]], dtype=np.float64)

_PREWITT_X = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], dtype=np.float64)
_PREWITT_Y = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]], dtype=np.float64)


def _gradients(image, kx, ky):
    if image.ndim != 2:
        raise ValueError("Gradient operators expect a single-channel image.")
    gx = convolve2d(image, kx)
    gy = convolve2d(image, ky)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.degrees(np.arctan2(gy, gx)) % 180
    return magnitude, orientation


def sobel_edges(image):
    """Sobel gradient operator. Returns (magnitude, orientation_degrees)."""
    return _gradients(_to_float(image), _SOBEL_X, _SOBEL_Y)


def prewitt_edges(image):
    """Prewitt gradient operator. Returns (magnitude, orientation_degrees)."""
    return _gradients(_to_float(image), _PREWITT_X, _PREWITT_Y)


# ---------------------------------------------------------------------------
# Canny: non-maximum suppression
# ---------------------------------------------------------------------------

def non_max_suppression(magnitude, orientation):
    """
    Thin edges by keeping only local maxima of `magnitude` along the
    gradient direction `orientation` (degrees, mod 180).

    The direction is quantised to 4 discrete bins (0, 45, 90, 135 degrees)
    so that each pixel can be compared to its two nearest neighbours along
    that direction using simple array shifts.
    """
    h, w = magnitude.shape
    padded_mag = np.pad(magnitude, 1, mode="constant", constant_values=0)
    out = np.zeros_like(magnitude)

    # quantise orientation into 4 bins
    angle = orientation.copy()
    bin_idx = np.zeros_like(angle, dtype=np.int8)
    bin_idx[(angle >= 22.5) & (angle < 67.5)] = 1     # ~45 deg
    bin_idx[(angle >= 67.5) & (angle < 112.5)] = 2    # ~90 deg
    bin_idx[(angle >= 112.5) & (angle < 157.5)] = 3   # ~135 deg
    # else stays 0 (~0 deg / horizontal gradient)

    # neighbour offsets (dy, dx) for each of the 4 gradient directions
    offsets = {
        0: ((0, -1), (0, 1)),    # gradient horizontal -> compare left/right
        1: ((-1, 1), (1, -1)),   # 45 deg diagonal
        2: ((-1, 0), (1, 0)),    # gradient vertical -> compare up/down
        3: ((-1, -1), (1, 1)),   # 135 deg diagonal
    }

    center = padded_mag[1:h + 1, 1:w + 1]
    keep = np.ones_like(magnitude, dtype=bool)
    for direction, ((dy1, dx1), (dy2, dx2)) in offsets.items():
        mask = bin_idx == direction
        n1 = padded_mag[1 + dy1:1 + dy1 + h, 1 + dx1:1 + dx1 + w]
        n2 = padded_mag[1 + dy2:1 + dy2 + h, 1 + dx2:1 + dx2 + w]
        local_not_max = (center < n1) | (center < n2)
        keep &= ~(mask & local_not_max)

    out = np.where(keep, magnitude, 0)
    return out


# ---------------------------------------------------------------------------
# Canny: double thresholding + hysteresis
# ---------------------------------------------------------------------------

WEAK = 75
STRONG = 255


def double_threshold(image, low, high):
    """
    Classify suppressed-gradient pixels into strong / weak / non-edges.

    Returns an array with values in {0, WEAK, STRONG}.
    """
    result = np.zeros_like(image, dtype=np.uint8)
    result[image >= high] = STRONG
    result[(image >= low) & (image < high)] = WEAK
    return result, WEAK, STRONG


def hysteresis(result, weak_val=WEAK, strong_val=STRONG):
    """
    Connect weak edges to strong edges via 8-connectivity. A weak pixel is
    promoted to a strong (final) edge if it is adjacent to a strong pixel,
    propagated iteratively until no more changes occur (flood-fill style).
    All remaining weak pixels are suppressed.
    """
    edges = (result == strong_val)
    weak = (result == weak_val)

    # iterative dilation of the strong-edge set, restricted to weak pixels,
    # until it stabilises -- equivalent to connected-component flood fill
    # restricted to weak+strong pixels.
    changed = True
    while changed:
        padded = np.pad(edges, 1, mode="constant", constant_values=False)
        neighbour_strong = np.zeros_like(edges)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbour_strong |= padded[1 + dy:1 + dy + edges.shape[0],
                                            1 + dx:1 + dx + edges.shape[1]]
        newly_promoted = weak & neighbour_strong & ~edges
        changed = bool(newly_promoted.any())
        edges |= newly_promoted

    return (edges * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Full Canny pipeline
# ---------------------------------------------------------------------------

def canny(image, gaussian_size=5, sigma=1.0, low_ratio=0.05, high_ratio=0.15):
    """
    Complete Canny edge detector:
      1. Gaussian smoothing (gaussian_size, sigma)
      2. Sobel gradient computation (magnitude + orientation)
      3. Non-maximum suppression
      4. Double thresholding (low/high derived from low_ratio/high_ratio
         of the max post-NMS gradient magnitude)
      5. Hysteresis edge linking

    Default parameters (size=5, sigma=1.0, low_ratio=0.05, high_ratio=0.15)
    follow common practice: a 5x5/sigma=1 Gaussian gives light smoothing
    that removes sensor noise without eroding piece-boundary detail, and a
    ~3:1 high:low ratio is the classic Canny recommendation for connecting
    broken edge segments without admitting excessive noise as "weak" edges.
    """
    if image.ndim != 2:
        raise ValueError("canny expects a single-channel image.")

    img_f = _to_float(image)
    kernel = gaussian_kernel(gaussian_size, sigma)
    smoothed = convolve2d(img_f, kernel)

    magnitude, orientation = _gradients(smoothed, _SOBEL_X, _SOBEL_Y)
    suppressed = non_max_suppression(magnitude, orientation)

    max_mag = suppressed.max()
    if max_mag <= 0:
        # no gradient anywhere (perfectly flat image) -> no edges
        return np.zeros_like(image, dtype=np.uint8)
    high = high_ratio * max_mag
    low = low_ratio * max_mag
    thresholded, weak_val, strong_val = double_threshold(suppressed, low, high)
    edges = hysteresis(thresholded, weak_val, strong_val)
    return edges
