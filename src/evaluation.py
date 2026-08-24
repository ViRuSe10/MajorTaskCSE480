"""
evaluation.py
==============
Turns an assembly.assemble() result into (a) a numerical measure of
reconstruction quality, and (b) a rendered preview image of the
reconstructed puzzle.

Quality metric
--------------
Ground truth (true grid position/orientation per piece) is not always
available -- see the dataset notes in README.md. compute_intrinsic_quality
therefore reports three complementary, ground-truth-free numbers derived
purely from the assembly process itself:

  - completion_ratio : placed_pieces / total_pieces. 1.0 iff every piece
    found a spot (assembly.py's "complete" flag).
  - fill_ratio        : placed_pieces / (grid_rows * grid_cols). A
    perfectly reconstructed rectangular puzzle has fill_ratio == 1.0;
    values well below 1.0 indicate the grid grew sparser/larger than a
    tight rectangle, e.g. because some genuinely flat sides were
    misclassified as shallow tabs/blanks (see piece_description.py) and
    the algorithm therefore didn't recognise a piece as being on the
    border where it should have stopped growing.
  - mean_edge_score / quality_score : the average match_score (see
    edge_matching.py -- lower is better, 0 is a perfect fit) across every
    placed piece's connection(s) to its already-placed neighbours,
    excluding the arbitrary starting piece (which has no real "match" to
    score). This is reported as `quality_score`, the primary numerical
    reconstruction-quality measure requested by the assignment: the
    average geometric+photometric mismatch of the seams in the final
    assembly. Lower is better; 0 is a perfect reconstruction.

compare_to_ground_truth() is provided for when a true piece-ID -> grid
position/rotation mapping becomes available (see README's dataset notes);
it searches over the 4 possible global rotations of the assembled grid
(since the greedy algorithm's absolute grid orientation is arbitrary
relative to any externally-defined convention) and reports position and
orientation accuracy under the best-fitting alignment.
"""

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Intrinsic (ground-truth-free) quality metrics
# ---------------------------------------------------------------------------

def compute_intrinsic_quality(result):
    """
    result : output of assembly.assemble().
    Returns a dict of metrics; see module docstring for definitions.
    """
    grid = result["grid"]
    total_placed = len(grid)
    total_pieces = total_placed + len(result["unplaced_pieces"])
    rows, cols = result["shape"]

    completion_ratio = total_placed / total_pieces if total_pieces > 0 else 1.0
    fill_ratio = total_placed / (rows * cols) if (rows * cols) > 0 else 1.0

    start_cell = result.get("start_cell")
    scores = [info["score"] for cell, info in grid.items() if cell != start_cell]
    mean_edge_score = float(np.mean(scores)) if scores else 0.0
    max_edge_score = float(np.max(scores)) if scores else 0.0

    return {
        "total_pieces": total_pieces,
        "placed_pieces": total_placed,
        "completion_ratio": completion_ratio,
        "grid_shape": (rows, cols),
        "fill_ratio": fill_ratio,
        "mean_edge_score": mean_edge_score,
        "max_edge_score": max_edge_score,
        "quality_score": mean_edge_score,
    }


# ---------------------------------------------------------------------------
# Ground-truth comparison (used once a piece-ID -> grid mapping is available)
# ---------------------------------------------------------------------------

def _rotate_cell(cell, rows, cols, k):
    """Rotate a (row, col) grid coordinate by k*90 degrees within a rows x cols grid."""
    r, c = cell
    for _ in range(k % 4):
        r, c, rows, cols = c, rows - 1 - r, cols, rows
    return (r, c)


def compare_to_ground_truth(result, ground_truth):
    """
    ground_truth : dict piece_idx -> (true_row, true_col, true_rotation).

    Searches the 4 possible global rotations of the assembled grid for the
    one that best matches the ground truth (since the algorithm's absolute
    grid orientation/origin is arbitrary), then reports:
      - position_accuracy    : fraction of (placed & ground-truthed) pieces
                                at the correct grid cell under that alignment.
      - orientation_accuracy : fraction with the correct rotation (mod 4,
                                after adding the global rotation offset).
    """
    grid = result["grid"]
    rows, cols = result["shape"]

    best = None
    for k in range(4):
        pos_correct, rot_correct, n_compared = 0, 0, 0
        for cell, info in grid.items():
            piece_idx = info["piece"]
            if piece_idx not in ground_truth:
                continue
            true_r, true_c, true_rot = ground_truth[piece_idx]
            rc, cc = _rotate_cell(cell, rows, cols, k)
            n_compared += 1
            if (rc, cc) == (true_r, true_c):
                pos_correct += 1
                if (info["rotation"] + k) % 4 == true_rot % 4:
                    rot_correct += 1
        if n_compared == 0:
            continue
        pos_acc = pos_correct / n_compared
        rot_acc = rot_correct / n_compared
        candidate = (pos_acc, rot_acc, n_compared, k)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return {"position_accuracy": 0.0, "orientation_accuracy": 0.0, "compared": 0, "best_global_rotation": 0}

    pos_acc, rot_acc, n_compared, k = best
    return {
        "position_accuracy": pos_acc,
        "orientation_accuracy": rot_acc,
        "compared": n_compared,
        "best_global_rotation": k,
    }


# ---------------------------------------------------------------------------
# Reconstructed image compositing
# ---------------------------------------------------------------------------

def reconstruct_image(pieces, result, background=(30, 30, 30)):
    """
    Composite the assembled grid into a single preview image. Each piece is
    rotated by -(orientation_deg + 90*rotation) -- undoing its photographed
    tilt (contour_extraction.normalize_orientation), then applying its
    resolved grid rotation -- and alpha-composited (via its own binary
    mask, so background pixels stay transparent) into its grid cell.

    Note: this uses PIL purely for the rotate/paste compositing of the
    final preview image; it is not one of the from-scratch algorithms
    required by the assignment spec (those are enhancement/thresholding/
    edge detection/segmentation/contour tracing, all implemented in their
    respective modules).
    """
    grid = result["grid"]
    if not grid:
        return np.full((10, 10, 3), background, dtype=np.uint8)

    rows, cols = result["shape"]
    heights = [pieces[info["piece"]]["image"].shape[0] for info in grid.values()]
    widths = [pieces[info["piece"]]["image"].shape[1] for info in grid.values()]
    cell_h = max(int(np.median(heights)), 1)
    cell_w = max(int(np.median(widths)), 1)

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), background)

    for (r, c), info in grid.items():
        piece = pieces[info["piece"]]
        rotation = info["rotation"]
        orientation = piece.get("orientation_deg", 0.0)
        # Undo the piece's own photographed tilt (-orientation) and apply
        # its resolved grid rotation as a POSITIVE quarter-turn per unit
        # of `rotation` -- assembly.py's rotation convention is a cyclic
        # shift of the piece's side list in the direction that requires a
        # +90-degree-per-step correction, not -90 (verified against a
        # controlled synthetic test with a known pre-rotated piece; the
        # previous "-(orientation + 90*rotation)" formula had this term's
        # sign backwards, which happened to look correct only for the
        # degenerate 180-degree case where +180 and -180 coincide).
        angle = 90 * rotation - orientation

        img = piece["image"]
        rgb = img if img.ndim == 3 else np.stack([img] * 3, axis=-1)
        rgba = np.dstack([rgb[..., :3], piece["mask"]]).astype(np.uint8)

        piece_img = Image.fromarray(rgba, mode="RGBA").rotate(
            angle, expand=True, resample=Image.BICUBIC
        )

        cell_x0, cell_y0 = c * cell_w, r * cell_h
        px = cell_x0 + (cell_w - piece_img.width) // 2
        py = cell_y0 + (cell_h - piece_img.height) // 2
        canvas.paste(piece_img, (px, py), piece_img)

    return np.array(canvas)
