"""
ml/make_comparison_images.py
==============================
Produces one reconstructed puzzle image per method (classical, Siamese,
GNN) on the same scrambled photo, for a side-by-side demo/report figure.

Usage
-----
    python -m ml.make_comparison_images \
        --image detection/input/scattered_pieces_01.jpg \
        --siamese_ckpt ml/checkpoints/siamese_best.pt \
        --gnn_ckpt ml/checkpoints/gnn_best.pt

Extracts pieces from the photo ONCE (same segmentation/description used
throughout this project) and reuses that same piece set for all three
methods, so the comparison isolates the scoring method (classical
formula vs. Siamese CNN vs. GNN) exactly like ml/evaluate_ml.py does for
its metrics -- this script just additionally saves the actual
reconstructed image for each, since a TA/report figure needs to *see*
the result, not just read a quality-score number.

If a checkpoint isn't found, that method is skipped with a clear message
rather than crashing -- so this still produces the classical image (and
whichever checkpoints ARE ready) even if training isn't finished yet.
"""

import os
import argparse
import numpy as np
from PIL import Image

from src import enhancement as enh
from src import segmentation as seg
from src import contour_extraction as ce
from src import piece_description as pdesc
from src import assembly as asm
from src import evaluation as ev


def extract_pieces(image_path, min_area=3000, max_aspect_ratio=2.2,
                   max_edge_density=0.13, morph_open_size=3, morph_close_size=7,
                   num_samples=32):
    """Same extraction pipeline as main.py's reconstruct_puzzle, factored out
    so all three methods below reconstruct from the identical piece set."""
    color_img = np.array(Image.open(image_path).convert("RGB"))
    gray = np.array(Image.open(image_path).convert("L"))

    mask, _bg = seg.foreground_mask_color(color_img)
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
    for p in pieces:
        cleaned = seg.fill_holes(seg.morphological_close(
            p["mask"], size=morph_close_size))
        p["mask"] = cleaned
        p["contour"] = ce.trace_boundary(cleaned)
        p["orientation_deg"] = ce.normalize_orientation(cleaned)

    return pdesc.describe_all_pieces(pieces, num_samples=num_samples)


def reconstruct_and_save(pieces, score_fn, out_path, method_name,
                         alpha=0.6, beta=0.4, num_samples=32):
    result = asm.assemble(pieces, alpha=alpha, beta=beta,
                          num_samples=num_samples, score_fn=score_fn)
    img = ev.reconstruct_image(pieces, result)
    Image.fromarray(img).save(out_path)
    q = ev.compute_intrinsic_quality(result)
    print(f"[{method_name}] saved {out_path}")
    print(f"  completion={q['completion_ratio']:.0%}  fill={q['fill_ratio']:.0%}  "
          f"quality_score={q['quality_score']:.4f}  grid={q['grid_shape']}")
    return result, q


def main():
    parser = argparse.ArgumentParser(
        description="Save one reconstructed image per method (classical/Siamese/GNN).")
    parser.add_argument("--image", required=True,
                        help="Scrambled puzzle photo to reconstruct")
    parser.add_argument(
        "--siamese_ckpt", default="ml/checkpoints/siamese_best.pt")
    parser.add_argument("--gnn_ckpt", default="ml/checkpoints/gnn_best.pt")
    parser.add_argument("--output_dir", default="results/reconstructed_images")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]

    print(f"Extracting pieces from {args.image} ...")
    pieces = extract_pieces(args.image, num_samples=args.num_samples)
    print(f"Extracted {len(pieces)} pieces.\n")

    # 1) classical (Milestone 1 default -- score_fn=None uses edge_matching.match_score)
    reconstruct_and_save(
        pieces, None, os.path.join(args.output_dir, f"{stem}_classical.png"),
        "classical", alpha=args.alpha, beta=args.beta, num_samples=args.num_samples,
    )

    # 2) Siamese CNN
    if os.path.exists(args.siamese_ckpt):
        import torch
        from ml.siamese_model import SiameseEdgeNet
        from ml.evaluate_ml import make_siamese_score_fn, load_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ckpt = load_checkpoint(args.siamese_ckpt, SiameseEdgeNet)
        model.to(device)
        score_fn = make_siamese_score_fn(
            model, pieces, args.num_samples, device)
        reconstruct_and_save(
            pieces, score_fn, os.path.join(
                args.output_dir, f"{stem}_siamese.png"),
            "siamese", alpha=args.alpha, beta=args.beta, num_samples=args.num_samples,
        )
    else:
        print(
            f"[siamese] skipped -- checkpoint not found at {args.siamese_ckpt}")

    # 3) GNN
    if os.path.exists(args.gnn_ckpt):
        import torch
        from ml.gnn_model import GNNEdgeNet
        from ml.evaluate_ml import make_gnn_score_fn, load_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ckpt = load_checkpoint(args.gnn_ckpt, GNNEdgeNet)
        model.to(device)
        score_fn = make_gnn_score_fn(model, pieces, args.num_samples, device)
        reconstruct_and_save(
            pieces, score_fn, os.path.join(args.output_dir, f"{stem}_gnn.png"),
            "gnn", alpha=args.alpha, beta=args.beta, num_samples=args.num_samples,
        )
    else:
        print(f"[gnn] skipped -- checkpoint not found at {args.gnn_ckpt}")


if __name__ == "__main__":
    main()
