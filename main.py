"""
main.py
=======
End-to-end routine: accepts a scrambled puzzle photo and returns the
reconstructed image together with a numerical measure of reconstruction
quality, by chaining every stage of the classical Milestone 1 pipeline.
"""

import sys
from PIL import Image
import numpy as np

from src import enhancement as enh
from src import segmentation as seg
from src import contour_extraction as ce
from src import piece_description as pdesc
from src import assembly as asm
from src import evaluation as ev


def reconstruct_puzzle(image_path, gaussian_size=5, gaussian_sigma=1.0,
                       threshold_method="otsu", morph_open_size=3, morph_close_size=7,
                       min_area=3000, max_aspect_ratio=2.2, max_edge_density=0.13,
                       num_samples=32, alpha=0.6, beta=0.4, return_intermediate=False):
    """
    Full Milestone 1 pipeline for one scrambled-puzzle photo:
      enhancement -> thresholding/segmentation -> contour extraction ->
      piece description -> (edge scoring happens inside assembly) ->
      assembly -> reconstructed image + quality score.

    Returns a dict with "pieces", "assembly", "reconstructed_image",
    and "quality". If return_intermediate=True, also includes "color_img",
    "mask", "labels", "filtered_labels" -- the intermediate arrays needed
    to visualize the mask/segmentation stages (see src/visualize.py and
    main.py's --debug flag).
    """
    color_img = np.array(Image.open(image_path).convert("RGB"))
    gray = np.array(Image.open(image_path).convert("L"))

    denoised = enh.gaussian_blur(
        gray, size=gaussian_size, sigma=gaussian_sigma)
    mask, _bg_color = seg.foreground_mask_color(color_img)
    opened = seg.morphological_open(mask, size=morph_open_size)
    labels, num_labels = seg.connected_components(opened, connectivity=8)
    stats = seg.component_stats(labels, num_labels)
    labels, stats = seg.split_touching_components(labels, stats)
    filtered_labels, kept_stats = seg.filter_components(
        labels, stats, min_area=min_area, max_aspect_ratio=max_aspect_ratio
    )
    filtered_labels, kept_stats = seg.filter_by_texture_density(
        filtered_labels, kept_stats, gray, max_edge_density=max_edge_density
    )

    pieces = ce.extract_all_pieces(color_img, filtered_labels, kept_stats)

    # per-piece cleanup: close small edge-touching notches and fill fully
    # enclosed holes caused by dark printed text/graphics near a piece's
    # boundary (applied per already-isolated piece crop, never on the
    # whole scene, so it can't accidentally bridge together two separate
    # nearby pieces -- see segmentation.py's morphological_close/fill_holes
    # docstrings). The contour and orientation are then recomputed from
    # the cleaned mask.
    for piece in pieces:
        cleaned = seg.fill_holes(seg.morphological_close(
            piece["mask"], size=morph_close_size))
        piece["mask"] = cleaned
        piece["contour"] = ce.trace_boundary(cleaned)
        piece["orientation_deg"] = ce.normalize_orientation(cleaned)

    described = pdesc.describe_all_pieces(pieces, num_samples=num_samples)

    result = asm.assemble(described, alpha=alpha,
                          beta=beta, num_samples=num_samples)
    reconstructed = ev.reconstruct_image(described, result)
    quality = ev.compute_intrinsic_quality(result)

    out = {
        "pieces": described,
        "assembly": result,
        "reconstructed_image": reconstructed,
        "quality": quality,
    }
    if return_intermediate:
        out.update({
            "color_img": color_img,
            "mask": mask,
            "labels": labels,
            "filtered_labels": filtered_labels,
        })
    return out


def main(image_path=None, debug=False):
    # Fallback to your default image if none is provided
    if image_path is None:
        image_path = "detection/images/train/0718-1_Color_png.rf.f6b7f8ba974357f79903f0d9fcf4264e.jpg"

    # Run the reconstruction pipeline
    out = reconstruct_puzzle(image_path, return_intermediate=debug)

    q = out["quality"]
    print(f"Input: {image_path}")
    print(f"Pieces found: {q['total_pieces']}  (placed: {q['placed_pieces']}, "
          f"completion: {q['completion_ratio']:.2%})")
    print(
        f"Grid shape: {q['grid_shape']}  (fill ratio: {q['fill_ratio']:.2%})")
    print(
        f"Quality score (mean edge mismatch, lower is better): {q['quality_score']:.4f}")

    out_path = "results/reconstruction.png"
    Image.fromarray(out["reconstructed_image"]).save(out_path)
    print(f"Saved reconstruction to {out_path}")

    if debug:
        import os
        from src import visualize as viz
        stem = os.path.splitext(os.path.basename(image_path))[0]

        mask_path, comp_path = viz.save_mask_and_components(out["mask"], out["labels"],
                                                            "results/masks", stem)
        print(f"Saved foreground mask to {mask_path}")
        print(f"Saved colored connected components to {comp_path}")

        piece_paths = viz.save_piece_crops(
            out["pieces"], "results/contours", stem, upscale_to=200)
        print(f"Saved {len(piece_paths)} individual piece crops to results/contours/"
              f"{stem}_piece_XX.png (+ _mask.png for each)")

        overlay_path = viz.save_all_contours_overlay(out["color_img"], out["pieces"],
                                                     "results/contours", stem)
        print(f"Saved full-scene contour overlay to {overlay_path}")


if __name__ == "__main__":
    main(debug=True)
