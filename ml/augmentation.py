"""
ml/augmentation.py
====================
Data augmentation for Milestone 2 training data, as required by the
assignment spec: "Data augmentation may be applied to the training set
and may include rotation, noise, illumination variation, contrast
variation, and colour variation."

Two categories, applied at different points in the pipeline:

1. Photometric jitter (illumination, contrast, colour, noise) -- purely
   pixel-value transforms, geometry-preserving, so they're safe to apply
   directly to a piece's image crop with no relabeling needed at all.

2. Rotation (`rotate_piece`) -- rotates an already-extracted piece by an
   arbitrary angle and fully re-runs contour tracing, orientation
   normalization, and side description on the rotated result. This is
   safe with respect to the ground-truth labeling pipeline in dataset.py:
   tab/blank/flat classification (piece_description.classify_side) is
   computed relative to the piece's own centroid, so it is rotation-
   invariant -- rotating a piece doesn't change its side-type sequence,
   only how it looks in the image frame -- so dataset.py's existing
   canonical-fingerprint alignment (_align_shift) still correctly
   recovers the true compass mapping for a rotated/augmented piece
   without any changes to that logic.
"""

import numpy as np
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Photometric jitter (illumination / contrast / colour / noise)
# ---------------------------------------------------------------------------

def jitter_illumination(image, rng, delta_range=(-30, 30)):
    """Add a random constant brightness offset."""
    delta = rng.uniform(*delta_range)
    return np.clip(image.astype(np.float64) + delta, 0, 255).astype(np.uint8)


def jitter_contrast(image, rng, factor_range=(0.7, 1.3)):
    """Scale pixel values around the image's own mean by a random factor."""
    factor = rng.uniform(*factor_range)
    mean = image.astype(np.float64).mean()
    out = (image.astype(np.float64) - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def jitter_color(image, rng, factor_range=(0.8, 1.2)):
    """Scale each RGB channel independently by a random factor (colour cast)."""
    if image.ndim != 3:
        return image
    factors = np.array([rng.uniform(*factor_range) for _ in range(image.shape[2])])
    out = image.astype(np.float64) * factors
    return np.clip(out, 0, 255).astype(np.uint8)


def add_noise(image, rng, sigma_range=(2, 10)):
    """Add zero-mean Gaussian sensor-noise."""
    sigma = rng.uniform(*sigma_range)
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def augment_pixels(image, rng, p_each=0.7, p_noise=0.5):
    """
    Apply a random combination of illumination/contrast/colour jitter and
    noise to a color image crop. `rng` must be a numpy Generator (e.g.
    np.random.default_rng(seed)) so illumination/contrast/color/noise
    draws are reproducible together. Geometry-preserving -- safe to use
    on an already-extracted piece crop with no relabeling.
    """
    out = image.copy()
    if rng.random() < p_each:
        out = jitter_illumination(out, rng)
    if rng.random() < p_each:
        out = jitter_contrast(out, rng)
    if rng.random() < p_each and out.ndim == 3:
        out = jitter_color(out, rng)
    if rng.random() < p_noise:
        out = add_noise(out, rng)
    return out


# ---------------------------------------------------------------------------
# Rotation (geometric -- re-runs contour/side extraction on the result)
# ---------------------------------------------------------------------------

def rotate_piece(piece, angle_degrees, num_samples=32):
    """
    Rotate an already-extracted, already-described piece dict (as
    produced by piece_description.describe_piece) by an arbitrary angle,
    and fully recompute its contour, orientation, and side description
    from the rotated mask/image. See module docstring for why this is
    safe with respect to dataset.py's ground-truth labeling.

    Note: this relies on flat/tab/blank classification being unaffected
    by the rotation. That holds robustly at real piece scale (validated
    directly against the dataset), but very small pieces with shallow
    tab/blank features can occasionally have a borderline "flat"
    classification flip due to rotation-resampling noise at the pixel
    level -- not a concern for the actual dataset's piece sizes, but
    worth knowing if reusing this on small synthetic/test imagery.
    """
    from src.contour_extraction import trace_boundary, normalize_orientation
    from src.piece_description import describe_piece

    img = piece["image"]
    mask = piece["mask"]

    pil_img = PILImage.fromarray(img)
    pil_mask = PILImage.fromarray(mask)
    rotated_img = np.array(pil_img.rotate(angle_degrees, expand=True, resample=PILImage.BICUBIC))
    rotated_mask = np.array(pil_mask.rotate(angle_degrees, expand=True, resample=PILImage.NEAREST))

    new_piece = {
        "label": piece.get("label"),
        "bbox": piece.get("bbox"),
        "image": rotated_img,
        "mask": rotated_mask,
        "contour": trace_boundary(rotated_mask),
        "orientation_deg": normalize_orientation(rotated_mask),
    }
    return describe_piece(new_piece, num_samples=num_samples)


def augment_piece(piece, rng, angle_range=(-25, 25), num_samples=32):
    """
    Combined augmentation for one piece: random rotation followed by
    photometric jitter on the rotated crop.
    """
    angle = rng.uniform(*angle_range)
    rotated = rotate_piece(piece, angle, num_samples=num_samples)
    rotated["image"] = augment_pixels(rotated["image"], rng)
    return rotated
