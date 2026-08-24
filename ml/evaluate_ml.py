"""
ml/evaluate_ml.py
===================
Loads a trained Siamese and/or GNN checkpoint, wraps each as a score_fn
compatible with src/assembly.py's assemble(score_fn=...), and reconstructs
a set of test-split puzzle photos three ways: the classical formula
(edge_matching.match_score, Milestone 1), the Siamese CNN, and the GNN --
all three using the *exact same* placement/rotation/tie-break/dead-end
algorithm, so the comparison isolates the scoring method itself.

For each method, reports (via src/evaluation.py):
  - completion_ratio, fill_ratio, quality_score (mean edge mismatch)
  - position_accuracy / orientation_accuracy against the manual-solve
    ground truth, where available for the test image's pieces
  - wall-clock time and (for the ML models) parameter count / checkpoint size

Usage
-----
    python -m ml.evaluate_ml --dataset_root <path> --manual_solve <path> \
        --reference_image <path> --reference_label <path> \
        --siamese_ckpt ml/checkpoints/siamese_best.pt \
        --gnn_ckpt ml/checkpoints/gnn_best.pt \
        --num_test_images 10
"""

import os
import time
import glob
import argparse

import numpy as np
import torch

from src import assembly as asm
from src import evaluation as ev
from src.edge_matching import side_shape_profile

from ml.dataset import (extract_and_identify_pieces, compute_canonical_fingerprints,
                         compute_true_adjacency, load_manual_solve, side_feature_vector)
from ml.siamese_model import SiameseEdgeNet
from ml.gnn_model import GNNEdgeNet


# ---------------------------------------------------------------------------
# score_fn adapters: model -> the (i, si, j, sj) -> float interface assembly.py expects
# ---------------------------------------------------------------------------

def make_siamese_score_fn(model, pieces, num_samples, device):
    model.eval()
    feats = [[side_feature_vector(p, s, num_samples) for s in p["sides"]] for p in pieces]

    def score_fn(i, si, j, sj):
        type_i, type_j = pieces[i]["sides"][si]["type"], pieces[j]["sides"][sj]["type"]
        if type_i == "flat" or type_j == "flat" or type_i == type_j:
            return float("inf")
        fa = torch.tensor(feats[i][si], dtype=torch.float32, device=device).unsqueeze(0)
        fb = torch.tensor(feats[j][sj], dtype=torch.float32, device=device).unsqueeze(0)
        return float(model.compatibility_score(fa, fb).item())

    return score_fn


def make_gnn_score_fn(model, pieces, num_samples, device):
    model.eval()
    all_sides = [np.stack([side_feature_vector(p, s, num_samples) for s in p["sides"]])
                 for p in pieces]

    def score_fn(i, si, j, sj):
        type_i, type_j = pieces[i]["sides"][si]["type"], pieces[j]["sides"][sj]["type"]
        if type_i == "flat" or type_j == "flat" or type_i == type_j:
            return float("inf")
        sa = torch.tensor(all_sides[i], dtype=torch.float32, device=device).unsqueeze(0)
        sb = torch.tensor(all_sides[j], dtype=torch.float32, device=device).unsqueeze(0)
        si_t = torch.tensor([si], dtype=torch.long, device=device)
        sj_t = torch.tensor([sj], dtype=torch.long, device=device)
        return float(model.compatibility_score(sa, si_t, sb, sj_t).item())

    return score_fn


# ---------------------------------------------------------------------------
# Per-image, per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_one_image(image_path, label_path, ground_truth, method_name, score_fn=None,
                        alpha=0.6, beta=0.4, num_samples=32):
    matches = extract_and_identify_pieces(image_path, label_path)
    pieces = [m[0] for m in matches]
    true_ids = [m[1] for m in matches]
    if not pieces:
        return None

    t0 = time.time()
    result = asm.assemble(pieces, alpha=alpha, beta=beta, num_samples=num_samples, score_fn=score_fn)
    elapsed = time.time() - t0

    quality = ev.compute_intrinsic_quality(result)

    gt_lookup = {}
    for idx, tid in enumerate(true_ids):
        if tid in ground_truth:
            r, c, rot = ground_truth[tid]
            gt_lookup[idx] = (r, c, rot)
    acc = ev.compare_to_ground_truth(result, gt_lookup) if gt_lookup else None

    return {
        "method": method_name,
        "image": os.path.basename(image_path),
        "elapsed_sec": elapsed,
        **quality,
        **({f"gt_{k}": v for k, v in acc.items()} if acc else {}),
    }


def load_checkpoint(path, model_cls, feature_len_hint=None):
    ckpt = torch.load(path, map_location="cpu")
    feature_len = ckpt.get("feature_len", feature_len_hint)
    model = model_cls(feature_len=feature_len)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--manual_solve", required=True)
    parser.add_argument("--reference_image", required=True)
    parser.add_argument("--reference_label", required=True)
    parser.add_argument("--siamese_ckpt", default=None)
    parser.add_argument("--gnn_ckpt", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_test_images", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--output_csv", default="results/evaluation_results/milestone2_comparison.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manual_solve = load_manual_solve(args.manual_solve)

    img_dir = os.path.join(args.dataset_root, "Images", args.split)
    label_dir = os.path.join(args.dataset_root, "Label", args.split)
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                          glob.glob(os.path.join(img_dir, "*.png")))[: args.num_test_images]

    siamese_model, gnn_model = None, None
    if args.siamese_ckpt:
        siamese_model, _ = load_checkpoint(args.siamese_ckpt, SiameseEdgeNet)
        siamese_model.to(device)
    if args.gnn_ckpt:
        gnn_model, _ = load_checkpoint(args.gnn_ckpt, GNNEdgeNet)
        gnn_model.to(device)

    rows = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        if not os.path.exists(label_path):
            continue

        matches = extract_and_identify_pieces(img_path, label_path)
        pieces = [m[0] for m in matches]
        if not pieces:
            continue

        # classical (Milestone 1 default, score_fn=None)
        row = evaluate_one_image(img_path, label_path, manual_solve, "classical", score_fn=None,
                                  num_samples=args.num_samples)
        if row:
            rows.append(row)

        if siamese_model is not None:
            sfn = make_siamese_score_fn(siamese_model, pieces, args.num_samples, device)
            row = evaluate_one_image(img_path, label_path, manual_solve, "siamese", score_fn=sfn,
                                      num_samples=args.num_samples)
            if row:
                rows.append(row)

        if gnn_model is not None:
            sfn = make_gnn_score_fn(gnn_model, pieces, args.num_samples, device)
            row = evaluate_one_image(img_path, label_path, manual_solve, "gnn", score_fn=sfn,
                                      num_samples=args.num_samples)
            if row:
                rows.append(row)

        print(f"done: {stem}")

    _write_csv(rows, args.output_csv)
    _print_summary(rows)


def _write_csv(rows, path):
    if not rows:
        print("no rows to write.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"saved {path}")


def _print_summary(rows):
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    print("\n=== Summary (mean across evaluated images) ===")
    for method, rs in by_method.items():
        q = np.mean([r["quality_score"] for r in rs])
        fill = np.mean([r["fill_ratio"] for r in rs])
        comp = np.mean([r["completion_ratio"] for r in rs])
        t = np.mean([r["elapsed_sec"] for r in rs])
        gt_pos = [r.get("gt_position_accuracy") for r in rs if "gt_position_accuracy" in r]
        gt_str = f", pos_acc={np.mean(gt_pos):.2%}" if gt_pos else ""
        print(f"  {method:10s}: quality_score={q:.4f}  fill_ratio={fill:.2%}  "
              f"completion={comp:.2%}  avg_time={t:.2f}s{gt_str}")


if __name__ == "__main__":
    main()
