"""
visualize.py
=============
Debug/inspection visualizations for the pipeline: the foreground mask,
colored connected components, and per-piece contour/side overlays. Not
part of the required from-scratch algorithms (enhancement/thresholding/
edge detection/etc.) -- this module exists purely to make intermediate
pipeline results easy to look at while developing or debugging.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_TYPE_COLORS = {"tab": (255, 60, 60), "blank": (60, 120, 255), "flat": (60, 220, 60)}


# ---------------------------------------------------------------------------
# Mask / connected-components visualizations
# ---------------------------------------------------------------------------

def colorize_labels(labels, seed=42):
    """Random distinct color per connected-component label, black background."""
    h, w = labels.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        color = rng.integers(60, 255, 3)
        vis[labels == lbl] = color
    return vis


def save_mask_and_components(mask, labels, output_dir, stem):
    """Save the binary foreground mask and a colorized connected-components image."""
    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, f"{stem}_foreground_mask.png")
    comp_path = os.path.join(output_dir, f"{stem}_components.png")
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(colorize_labels(labels)).save(comp_path)
    return mask_path, comp_path


# ---------------------------------------------------------------------------
# Per-piece contour / side visualizations
# ---------------------------------------------------------------------------

def _get_font(size=16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_piece_debug(piece):
    """
    Render one piece's crop with its contour drawn on top. If the piece
    has been through piece_description (has "sides"/"corners"), each side
    is colour-coded by type (tab=red, blank=blue, flat=green) and corners
    are marked, matching the convention used throughout this project.
    Otherwise (contour_extraction stage only) the raw contour is drawn in
    plain red.
    """
    img = Image.fromarray(piece["image"].copy())
    draw = ImageDraw.Draw(img)

    if "sides" in piece:
        for side in piece["sides"]:
            color = _TYPE_COLORS.get(side["type"], (255, 255, 255))
            for (y, x) in side["points"]:
                draw.point((x, y), fill=color)
        for (y, x) in piece.get("corners", []):
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 255, 0), width=2)
    else:
        for (y, x) in piece.get("contour", []):
            draw.point((x, y), fill=(255, 0, 0))

    return img


def save_piece_crops(pieces, output_dir, stem, upscale_to=None):
    """
    Save each piece's own crop (with contour/side overlay) and its raw
    binary mask as separate numbered files: <stem>_piece_00.png, etc.
    Returns the list of saved piece-crop paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, piece in enumerate(pieces):
        crop_img = draw_piece_debug(piece)
        if upscale_to and max(crop_img.size) < upscale_to:
            scale = upscale_to / max(crop_img.size)
            crop_img = crop_img.resize(
                (int(crop_img.width * scale), int(crop_img.height * scale)), Image.NEAREST
            )
        crop_path = os.path.join(output_dir, f"{stem}_piece_{i:02d}.png")
        crop_img.save(crop_path)

        mask_path = os.path.join(output_dir, f"{stem}_piece_{i:02d}_mask.png")
        Image.fromarray(piece["mask"]).save(mask_path)
        paths.append(crop_path)
    return paths


def save_all_contours_overlay(color_img, pieces, output_dir, stem):
    """
    Save one image of the full original scene with every piece's contour
    (colour-coded by side type where available) drawn on top, plus its
    piece index as a text label -- useful for a single at-a-glance check
    of the whole segmentation+description result.
    """
    os.makedirs(output_dir, exist_ok=True)
    vis = Image.fromarray(color_img.copy())
    draw = ImageDraw.Draw(vis)
    font = _get_font(18)

    for i, piece in enumerate(pieces):
        y0, x0, y1, x1 = piece["bbox"]
        if "sides" in piece:
            for side in piece["sides"]:
                color = _TYPE_COLORS.get(side["type"], (255, 255, 255))
                for (cy, cx) in side["points"]:
                    draw.point((x0 + cx, y0 + cy), fill=color)
        else:
            for (cy, cx) in piece.get("contour", []):
                draw.point((x0 + cx, y0 + cy), fill=(255, 0, 0))
        draw.text((x0, max(y0 - 20, 0)), str(i), fill=(255, 255, 0), font=font)

    out_path = os.path.join(output_dir, f"{stem}_all_contours.png")
    vis.save(out_path)
    return out_path
