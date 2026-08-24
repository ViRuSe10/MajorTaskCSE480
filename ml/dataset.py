"""
ml/dataset.py
==============
Turns the full Roboflow/Kaggle jigsaw dataset (train/valid/test images +
YOLO detection labels) into labeled side-pair samples for training the
Siamese CNN and GNN in Milestone 2.

Key idea
--------
The physical puzzle is IDENTICAL across every photo in the dataset (same
35 pieces, same true solved layout) -- only how the pieces are scattered
and photographed changes. So a SINGLE manually-solved reference layout
(piece ID -> (row, col, rotation), gathered once -- see
detection/ground_truth/manual_solve.json) gives us ground truth we can
propagate automatically to every one of the 4000+ training images:

  1. For each training image, run the Milestone 1 classical pipeline
     (src.enhancement / segmentation / contour_extraction /
     piece_description) to extract that image's pieces and their sides.
  2. Match each extracted piece to its true piece ID via the image's own
     YOLO label file (Hungarian assignment on bounding-box centroids --
     same technique used to build the labeled contact sheet).
  3. Each piece's *photographed* rotation is arbitrary and differs from
     the reference photo used for the manual solve. We recover, for this
     image, which of the piece's 4 (locally-ordered) sides is the "true"
     North/East/South/West side by matching this photo's tab/blank/flat
     side-type sequence (cyclically) against the same piece's canonical
     type sequence captured once from the reference image.
  4. True adjacency (which compass side of which piece ID touches which
     compass side of which other piece ID) is computed once from the
     manual solve's grid positions.
  5. Every cross-piece side pair in every image is then labeled: positive
     if its (id, compass-direction) pair matches a true-adjacency pair,
     negative otherwise. Negatives vastly outnumber positives, so they
     are randomly subsampled to a fixed ratio.

Each side's feature vector is [shape_profile (num_samples) ++
color_strip (num_samples*3)], reusing piece_description/edge_matching
directly -- this is also exactly what the Siamese CNN (1D-conv) and GNN
(node/edge features) consume, so Milestone 1 and Milestone 2 share the
same underlying geometric+photometric representation.
"""

import os
import json
import glob
import random

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from src import enhancement as enh
from src import segmentation as seg
from src import contour_extraction as ce
from src import piece_description as pdesc
from src.edge_matching import side_shape_profile


_COMPASS = ("N", "E", "S", "W")
_DIR_OFFSET = {"N": 0, "E": 1, "S": 2, "W": 3}
_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_manual_solve(path):
    """
    Load the manually-solved reference layout.

    Expected JSON format: {"<piece_id>": {"row": int, "col": int, "rotation": int}, ...}
    rotation follows the same convention as src/assembly.py: the cyclic
    shift r such that, in the piece's *reference-image* local side order
    [s0, s1, s2, s3], North = s[(0+r)%4], East = s[(1+r)%4], etc.
    """
    with open(path) as f:
        raw = json.load(f)
    return {int(k): (int(v["row"]), int(v["col"]), int(v["rotation"])) for k, v in raw.items()}


def compute_true_adjacency(manual_solve):
    """
    From grid positions alone, compute the set of true-neighbour
    (id, compass_dir) <-> (id, compass_dir) pairs.
    """
    pos_to_id = {(r, c): pid for pid, (r, c, rot) in manual_solve.items()}
    pairs = set()
    for pid, (r, c, rot) in manual_solve.items():
        for d, (dr, dc) in _DELTA.items():
            npos = (r + dr, c + dc)
            if npos in pos_to_id:
                pairs.add(
                    frozenset({(pid, d), (pos_to_id[npos], _OPPOSITE[d])}))
    return pairs


# ---------------------------------------------------------------------------
# Per-image piece extraction + true-ID matching (reused from the contact-
# sheet workflow, generalised to any image/label pair)
# ---------------------------------------------------------------------------

def extract_and_identify_pieces(image_path, label_path, min_area=1500, max_aspect_ratio=2.5):
    """
    Run the Milestone 1 classical pipeline on one image and match each
    extracted piece to its true YOLO class-derived piece ID.

    Returns list of (piece_dict, true_id) tuples. Pieces that can't be
    confidently matched (more detections than labels, or vice versa) are
    matched via optimal (Hungarian) assignment regardless -- callers
    should treat large-distance matches with suspicion (see
    build_dataset's `max_match_dist` filter).
    """
    color_img = np.array(Image.open(image_path).convert("RGB"))
    gray = np.array(Image.open(image_path).convert("L"))
    img_h, img_w = gray.shape

    denoised = enh.gaussian_blur(gray, size=5, sigma=1.0)
    mask = seg.foreground_mask(denoised, method="otsu")
    opened = seg.morphological_open(mask, size=3)
    labels, num = seg.connected_components(opened, connectivity=8)
    stats = seg.component_stats(labels, num)
    filtered_labels, kept_stats = seg.filter_components(
        labels, stats, min_area=min_area, max_aspect_ratio=max_aspect_ratio
    )
    pieces = ce.extract_all_pieces(color_img, filtered_labels, kept_stats)
    described = pdesc.describe_all_pieces(pieces)

    with open(label_path) as f:
        yolo_lines = [l.strip().split() for l in f if l.strip()]
    yolo_boxes = []
    for parts in yolo_lines:
        cls_idx = int(parts[0])
        xc, yc = float(parts[1]) * img_w, float(parts[2]) * img_h
        yolo_boxes.append({"class_idx": cls_idx, "cx": xc, "cy": yc})

    if not described or not yolo_boxes:
        return []

    def centroid(p):
        y0, x0, y1, x1 = p["bbox"]
        return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)

    n_p, n_b = len(described), len(yolo_boxes)
    cost = np.zeros((n_p, n_b))
    for i, p in enumerate(described):
        py, px = centroid(p)
        for j, b in enumerate(yolo_boxes):
            cost[i, j] = ((b["cx"] - px) ** 2 + (b["cy"] - py) ** 2) ** 0.5

    row_ind, col_ind = linear_sum_assignment(cost)
    class_names = _CLASS_NAMES  # alphabetically-sorted printed-name lookup, see below
    results = []
    for i, j in zip(row_ind, col_ind):
        true_id = int(class_names[yolo_boxes[j]["class_idx"]])
        dist = cost[i, j]
        results.append((described[i], true_id, dist))
    return results


# data.yaml's class list: alphabetically-sorted strings '1'..'35' -> index 0..34
_CLASS_NAMES = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32',
                '33', '34', '35', '4', '5', '6', '7', '8', '9']


# ---------------------------------------------------------------------------
# Canonical fingerprint + rotation alignment across photos
# ---------------------------------------------------------------------------

def compute_canonical_fingerprints(reference_image_path, reference_label_path, max_match_dist=30):
    """
    Build the reference tab/blank/flat type sequence for every piece ID,
    from the same reference image used for the manual solve / contact
    sheet. Returns {true_id: (type0, type1, type2, type3)} in that
    image's local side order (the same order manual_solve's "rotation"
    values are defined relative to).
    """
    matches = extract_and_identify_pieces(
        reference_image_path, reference_label_path)
    fingerprints = {}
    for piece, true_id, dist in matches:
        if dist > max_match_dist:
            # unreliable match (e.g. the touching-pieces / clutter cases)
            continue
        fingerprints[true_id] = tuple(s["type"] for s in piece["sides"])
    return fingerprints


def _piece_from_cut_image(path):
    """
    Build a fully described piece dict directly from a Cut/N.png -- an
    individual, isolated photo of one piece with a real alpha channel
    (true transparency, not just a white background), as produced by
    e.g. a background-removal tool. The alpha channel IS the piece's
    silhouette, so this bypasses our own classical segmentation
    (foreground_mask_color/connected_components/etc.) entirely for these
    images -- there's no thresholding ambiguity to resolve when the mask
    is already exact, and skipping it avoids introducing any of that
    pipeline's own noise into the canonical reference fingerprints.
    """
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = (alpha > 127).astype(np.uint8) * 255
    rgb = arr[:, :, :3].copy()
    white_bg = np.full_like(rgb, 255)
    composited = np.where((mask > 0)[:, :, None], rgb, white_bg)

    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop_mask = mask[y0:y1 + 1, x0:x1 + 1]
    crop_img = composited[y0:y1 + 1, x0:x1 + 1]

    contour = ce.trace_boundary(crop_mask)
    piece = {
        "label": 1,
        "bbox": (0, 0, crop_mask.shape[0] - 1, crop_mask.shape[1] - 1),
        "image": crop_img,
        "mask": crop_mask,
        "contour": contour,
        "orientation_deg": ce.normalize_orientation(crop_mask),
    }
    return pdesc.describe_piece(piece)


def compute_canonical_fingerprints_from_cut_folder(cut_dir, num_pieces=35, num_samples=32):
    """
    Alternative to compute_canonical_fingerprints: builds the reference
    type-sequence fingerprint for every piece ID directly from a folder
    of individually-isolated, alpha-masked piece images named "<id>.png"
    (e.g. Cut/1.png .. Cut/35.png). More reliable than extracting
    fingerprints from one shared multi-piece scattered photo, since each
    image here has an exact ground-truth mask and only ever contains one
    piece (no touching-pieces/clutter/Hungarian-matching-uncertainty
    concerns at all).

    Returns {picture_number: (type0, type1, type2, type3)}, in that
    image's own local side order -- manual_solve's "rotation" values for
    these pieces must be defined relative to THIS order.
    """
    fingerprints = {}
    for n in range(1, num_pieces + 1):
        path = os.path.join(cut_dir, f"{n}.png")
        if not os.path.exists(path):
            continue
        piece = _piece_from_cut_image(path)
        fingerprints[n] = tuple(s["type"] for s in piece["sides"])
    return fingerprints


def _align_shift(local_types, canonical_types):
    """
    Find shift s in {0,1,2,3} such that local_types[(k+s)%4] == canonical_types[k]
    for all k (i.e. local side (k+s)%4 plays the same role as canonical side k).
    Falls back to the shift with the most matching positions if no exact
    match exists (real data noise), or None if every shift scores 0/4.
    """
    best_s, best_score = None, -1
    for s in range(4):
        score = sum(1 for k in range(
            4) if local_types[(k + s) % 4] == canonical_types[k])
        if score > best_score:
            best_score, best_s = score, s
    return best_s if best_score > 0 else None


# ---------------------------------------------------------------------------
# Feature vector per side
# ---------------------------------------------------------------------------

def side_feature_vector(piece, side, num_samples=32):
    """[shape_profile (num_samples) ++ color_strip (num_samples*3)] as one flat float32 vector."""
    ys, xs = np.where(piece["mask"] > 0)
    centroid = (float(ys.mean()), float(xs.mean())) if ys.size else (0.0, 0.0)
    profile, _len = side_shape_profile(side["points"], centroid, num_samples)
    strip = side["color_strip"].astype(np.float32).reshape(-1)
    if len(strip) != num_samples * 3:
        idx = np.linspace(
            0, len(side["color_strip"]) - 1, num_samples).astype(int)
        strip = side["color_strip"][idx].astype(np.float32).reshape(-1)
    return np.concatenate([profile.astype(np.float32), strip / 255.0])


# ---------------------------------------------------------------------------
# Per-image sample generation
# ---------------------------------------------------------------------------

def _build_piece_compass(matches, canonical_fingerprints, max_match_dist, num_samples):
    """
    Given (piece, true_id, dist) matches, filter to confidently-matched
    pieces, recover each one's local-side-index -> compass-direction
    mapping via canonical fingerprint alignment, and compute its 4 side
    feature vectors. Shared by the original pass and every augmented pass
    in generate_pairs_for_image, so both use identical, already-validated
    labeling logic.
    """
    piece_compass = []
    for piece, true_id, dist in matches:
        if dist > max_match_dist or true_id not in canonical_fingerprints:
            continue
        local_types = tuple(s["type"] for s in piece["sides"])
        shift = _align_shift(local_types, canonical_fingerprints[true_id])
        if shift is None:
            continue
        mapping = {local_idx: _COMPASS[(local_idx - shift) % 4]
                   for local_idx in range(4)}
        all_feats = np.stack([side_feature_vector(
            piece, s, num_samples) for s in piece["sides"]])
        piece_compass.append((piece, true_id, mapping, all_feats))
    return piece_compass


def _pairs_from_piece_compass(piece_compass, true_adjacency):
    """Generate (feats_a, si, feats_b, sj, label, rel_rot) for every valid cross-piece side pair in one piece_compass list."""
    positives, negatives = [], []
    n = len(piece_compass)
    for i in range(n):
        piece_a, id_a, map_a, feats_a = piece_compass[i]
        for si in range(4):
            side_a = piece_a["sides"][si]
            if side_a["type"] == "flat":
                continue
            dir_a = map_a[si]
            for j in range(i + 1, n):
                piece_b, id_b, map_b, feats_b = piece_compass[j]
                for sj in range(4):
                    side_b = piece_b["sides"][sj]
                    if side_b["type"] == "flat" or side_a["type"] == side_b["type"]:
                        continue
                    dir_b = map_b[sj]
                    is_true_pair = frozenset(
                        {(id_a, dir_a), (id_b, dir_b)}) in true_adjacency
                    if is_true_pair:
                        rel_rot = (_DIR_OFFSET[dir_a] -
                                   _DIR_OFFSET[_OPPOSITE[dir_b]]) % 4
                        positives.append(
                            (feats_a, si, feats_b, sj, 1, rel_rot))
                    else:
                        negatives.append((feats_a, si, feats_b, sj, 0, 0))
    return positives, negatives


def generate_pairs_for_image(image_path, label_path, canonical_fingerprints, true_adjacency,
                             num_samples=32, negative_ratio=8, max_match_dist=30, rng=None,
                             augment=False, augmentations_per_image=2, rotation_range=(-25, 25)):
    """
    Returns a list of (piece_a_sides, si, piece_b_sides, sj, label, rel_rotation)
    samples for one image, where:
      - piece_a_sides, piece_b_sides : (4, feature_len) arrays -- ALL 4 sides
        of each piece (not just the queried one), so models that want
        intra-piece graph context (ml/gnn_model.py) have it available.
        Models that only need the pair (ml/siamese_model.py) simply index
        piece_a_sides[si] / piece_b_sides[sj].
      - si, sj      : which of the 4 sides is the query side for a and b.
      - label       : 1 (true neighbours) or 0 (not neighbours).
      - rel_rotation: true relative-orientation class for positives (0 for
        negatives, unused in the loss for those).

    augment / augmentations_per_image : if augment=True, this image
    additionally contributes `augmentations_per_image` extra independent
    passes, each with every matched piece randomly rotated (rotation_range,
    degrees) and photometrically jittered (ml/augmentation.py) before its
    side features are recomputed. Safe with respect to the ground-truth
    labels -- see ml/augmentation.py's module docstring for why rotation
    doesn't change a piece's recovered compass mapping.
    """
    rng = rng or random.Random()
    matches = extract_and_identify_pieces(image_path, label_path)
    if not matches:
        return []

    piece_compass = _build_piece_compass(
        matches, canonical_fingerprints, max_match_dist, num_samples)
    positives, negatives = _pairs_from_piece_compass(
        piece_compass, true_adjacency)

    if augment and augmentations_per_image > 0:
        from ml.augmentation import augment_piece
        np_rng = np.random.default_rng(rng.randint(0, 2 ** 31 - 1))
        for _ in range(augmentations_per_image):
            aug_matches = []
            for piece, true_id, dist in matches:
                if dist > max_match_dist or true_id not in canonical_fingerprints:
                    continue
                try:
                    aug_piece = augment_piece(
                        piece, np_rng, angle_range=rotation_range, num_samples=num_samples)
                except Exception:
                    continue
                aug_matches.append((aug_piece, true_id, dist))
            aug_compass = _build_piece_compass(
                aug_matches, canonical_fingerprints, max_match_dist, num_samples)
            aug_pos, aug_neg = _pairs_from_piece_compass(
                aug_compass, true_adjacency)
            positives.extend(aug_pos)
            negatives.extend(aug_neg)

    rng.shuffle(negatives)
    keep_neg = negatives[: max(len(positives) * negative_ratio, 20)]
    samples = positives + keep_neg
    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Full-dataset build
# ---------------------------------------------------------------------------

_IMAGE_DIR_CANDIDATES = ("Images", "images")
_LABEL_DIR_CANDIDATES = ("Label", "labels")


def _resolve_split_dirs(dataset_root, split):
    """
    Find the images/ and labels/ directories for one split, tolerating
    different dataset export conventions (this project's own
    zip used "Label", the Kaggle/Roboflow export used "labels" or
    "obj_det"). Raises a clear error if neither can be found rather than
    silently returning an empty split.
    """
    img_dir = None
    for cand in _IMAGE_DIR_CANDIDATES:
        candidate_path = os.path.join(dataset_root, cand, split)
        if os.path.isdir(candidate_path):
            img_dir = candidate_path
            break

    label_dir = None
    for cand in _LABEL_DIR_CANDIDATES:
        candidate_path = os.path.join(dataset_root, cand, split)
        if os.path.isdir(candidate_path):
            label_dir = candidate_path
            break

    if img_dir is None:
        raise FileNotFoundError(
            f"Could not find an images directory for split '{split}' under {dataset_root} "
            f"(tried: {[os.path.join(c, split) for c in _IMAGE_DIR_CANDIDATES]})"
        )
    if label_dir is None:
        raise FileNotFoundError(
            f"Could not find a labels directory for split '{split}' under {dataset_root} "
            f"(tried: {[os.path.join(c, split) for c in _LABEL_DIR_CANDIDATES]})"
        )
    return img_dir, label_dir


def build_split(dataset_root, split, canonical_fingerprints, true_adjacency,
                num_samples=32, negative_ratio=8, max_images=None, seed=0,
                augment=False, augmentations_per_image=0):
    """
    Walk the images/labels directories for one split (tolerating a few
    different folder-naming conventions -- see _resolve_split_dirs),
    generating pair samples for every image. Returns stacked numpy arrays:
      sides_a : (N, 4, feature_len)  -- all 4 sides of piece A per sample
      si      : (N,)                 -- query side index within piece A
      sides_b : (N, 4, feature_len)
      sj      : (N,)
      labels  : (N,)
      rel_rot : (N,)

    augment / augmentations_per_image : see augmentation.py -- if
    augment=True, each image additionally contributes
    augmentations_per_image extra augmented passes (rotation, noise,
    illumination/contrast/colour jitter applied to the source image
    before piece extraction), each generating its own independent set of
    pair samples.
    """
    rng = random.Random(seed)
    img_dir, label_dir = _resolve_split_dirs(dataset_root, split)
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                         glob.glob(os.path.join(img_dir, "*.png")))
    if max_images:
        image_paths = image_paths[:max_images]

    all_sa, all_si, all_sb, all_sj, all_labels, all_rot = [], [], [], [], [], []
    for n_done, img_path in enumerate(image_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, stem + ".txt")
        label_path = label_path.replace("\\", "/")
        if not os.path.exists(label_path):
            print("Pass does not exist")
            continue
        try:
            samples = generate_pairs_for_image(
                img_path, label_path, canonical_fingerprints, true_adjacency,
                num_samples=num_samples, negative_ratio=negative_ratio, rng=rng,
                augment=augment, augmentations_per_image=augmentations_per_image,
            )
        except Exception as exc:  # a handful of malformed/edge-case images shouldn't kill a 4000-image run
            print(f"  [skip] {img_path}: {exc}")
            continue

        for feats_a, si, feats_b, sj, label, rel_rot in samples:
            all_sa.append(feats_a)
            all_si.append(si)
            all_sb.append(feats_b)
            all_sj.append(sj)
            all_labels.append(label)
            all_rot.append(rel_rot)

        if (n_done + 1) % 50 == 0:
            print(f"  processed {n_done + 1}/{len(image_paths)} images, "
                  f"{len(all_labels)} samples so far")

    if not all_labels:
        L = num_samples * 4
        return (np.zeros((0, 4, L), dtype=np.float32), np.zeros((0,), dtype=np.int64),
                np.zeros((0, 4, L), dtype=np.float32), np.zeros(
                    (0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64))

    return (np.stack(all_sa), np.array(all_si, dtype=np.int64),
            np.stack(all_sb), np.array(all_sj, dtype=np.int64),
            np.array(all_labels, dtype=np.int64), np.array(all_rot, dtype=np.int64))


def build_dataset(dataset_root, manual_solve_path, output_dir, splits=("train", "valid", "test"),
                  reference_image_path=None, reference_label_path=None, cut_dir=None,
                  num_samples=32, negative_ratio=8, max_images=None,
                  augment_train=True, augmentations_per_image=2):
    """
    Full pipeline: load manual solve -> derive canonical fingerprints and
    true adjacency -> walk every split -> save one .npz per split to
    `output_dir`.

    Canonical fingerprints come from EITHER a single reference scattered
    photo (reference_image_path + reference_label_path) OR a folder of
    individually-isolated, alpha-masked per-piece images named "<id>.png"
    (cut_dir) -- the latter is preferred when available (see
    compute_canonical_fingerprints_from_cut_folder's docstring for why).
    Provide exactly one of the two.

    augment_train : if True, only the "train" split gets the extra
    augmented passes (rotation + photometric jitter, see augmentation.py)
    -- validation and test data should stay unaugmented/real, since they
    exist to measure performance on genuine data, not augmented copies.
    """
    os.makedirs(output_dir, exist_ok=True)
    manual_solve = load_manual_solve(manual_solve_path)

    if cut_dir:
        canonical_fingerprints = compute_canonical_fingerprints_from_cut_folder(
            cut_dir)
    elif reference_image_path and reference_label_path:
        canonical_fingerprints = compute_canonical_fingerprints(
            reference_image_path, reference_label_path)
    else:
        raise ValueError(
            "Provide either cut_dir, or both reference_image_path and reference_label_path.")

    true_adjacency = compute_true_adjacency(manual_solve)
    print(f"Loaded manual solve for {len(manual_solve)} pieces, "
          f"{len(canonical_fingerprints)} canonical fingerprints, "
          f"{len(true_adjacency)} true-adjacency pairs.")

    for split in splits:
        do_augment = augment_train and split == "train"
        print(f"Building split: {split} (augmented: {do_augment})")
        sides_a, si, sides_b, sj, labels, rel_rot = build_split(
            dataset_root, split, canonical_fingerprints, true_adjacency,
            num_samples=num_samples, negative_ratio=negative_ratio, max_images=max_images,
            augment=do_augment, augmentations_per_image=augmentations_per_image,
        )
        out_path = os.path.join(output_dir, f"{split}_pairs.npz")
        np.savez_compressed(out_path, sides_a=sides_a, si=si, sides_b=sides_b, sj=sj,
                            labels=labels, rel_rot=rel_rot)
        pos = int(labels.sum())
        print(
            f"  saved {out_path}: {len(labels)} samples ({pos} positive, {len(labels)-pos} negative)")


if __name__ == "__main__":

    build_dataset(
        dataset_root="detection/", manual_solve_path="detection/ground truth/manual_solve.json", output_dir="ml/pairs_dataset",
        cut_dir="ml/Images Here/Cut", max_images=1000)
