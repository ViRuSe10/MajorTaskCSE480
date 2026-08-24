"""
assembly.py
============
Greedy best-first reconstruction of the puzzle grid from the described,
scored pieces (piece_description.describe_all_pieces +
edge_matching.match_score).

Algorithm
---------
1. Start piece: prefer a corner piece (two adjacent flat sides), else a
   border piece (one flat side), else any piece. It is placed at grid
   cell (0, 0) with whichever rotation orients its flat side(s) to North
   and/or West.
2. Frontier: the set of empty grid cells adjacent to at least one placed
   piece, added only in directions where the placed piece's facing side
   is NOT flat (a flat side means "this is the outer border of the
   puzzle here", so no piece is expected beyond it).
3. At each iteration, across *every* frontier cell and *every* unused
   piece and rotation (0/90/180/270, expressed as a cyclic shift of the
   piece's 4-side list -- this is how each piece's rotation is resolved
   during placement), compute the mean edge-dissimilarity score against
   all already-placed neighbours of that cell (edge_matching.match_score,
   averaged so cells with different neighbour counts are comparable).
   The globally best (lowest-mean-score) (cell, piece, rotation) triple
   is placed.
4. Tie-breaking rule (stated explicitly, as required by the assignment):
   ties on mean score (within `tie_epsilon`) are broken first by
   preferring the *more constrained* cell (the one with more already-
   placed neighbours, since agreement across more neighbours is stronger
   evidence), then by the lower piece index (for reproducibility).
5. Dead ends / unplaceable pieces: if no frontier cell has any candidate
   with a finite score (i.e. no remaining piece's rotation is
   geometrically/photometrically compatible with what's already placed
   around any open cell), the algorithm stops growing and returns
   immediately. The grid built so far -- the best arrangement obtained --
   is always returned, along with the list of any pieces that could not
   be placed, rather than raising an error or discarding partial work.
"""

import numpy as np

from .edge_matching import side_shape_profile, shape_compatibility, color_ssd


_DIRS = ("N", "E", "S", "W")
_DIR_OFFSET = {"N": 0, "E": 1, "S": 2, "W": 3}
_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


# ---------------------------------------------------------------------------
# Piece / side helpers
# ---------------------------------------------------------------------------

def _piece_centroid(piece):
    ys, xs = np.where(piece["mask"] > 0)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))


def _side_index(rotation, direction):
    """Index into a piece's 4-entry 'sides' list facing `direction` under `rotation`."""
    return (_DIR_OFFSET[direction] + rotation) % 4


def get_side(piece, rotation, direction):
    return piece["sides"][_side_index(rotation, direction)]


def count_flat_sides(piece):
    return sum(1 for s in piece["sides"] if s["type"] == "flat")


def _adjacent_flat_rotation(piece):
    """
    If two ADJACENT sides (consecutive in the piece's natural cyclic
    order) are both flat, return the rotation that places them at North
    and West (the natural top-left-corner orientation). Else None.
    """
    types = [s["type"] for s in piece["sides"]]
    for k in range(4):
        if types[k] == "flat" and types[(k - 1) % 4] == "flat":
            return k  # rotation r=k -> N = sides[k], W = sides[(k-1)%4]
    return None


def _single_flat_rotation(piece):
    """Rotation that places a piece's (first found) flat side at North."""
    types = [s["type"] for s in piece["sides"]]
    if "flat" in types:
        return types.index("flat")
    return None


def choose_start_piece(pieces):
    """
    Choose the starting (piece_index, rotation): prefer a corner piece
    (flats at N & W), else a border piece (flat at N), else piece 0 with
    rotation 0 as a last resort.
    """
    for i, p in enumerate(pieces):
        r = _adjacent_flat_rotation(p)
        if r is not None:
            return i, r
    for i, p in enumerate(pieces):
        r = _single_flat_rotation(p)
        if r is not None:
            return i, r
    return 0, 0


# ---------------------------------------------------------------------------
# Precomputed per-side scoring data (avoids recomputing resampled shape
# profiles on every candidate evaluation)
# ---------------------------------------------------------------------------

def _precompute_side_data(pieces, num_samples):
    data = {}
    for i, p in enumerate(pieces):
        centroid = _piece_centroid(p)
        for si, s in enumerate(p["sides"]):
            if s["type"] == "flat":
                data[(i, si)] = None
                continue
            profile, length = side_shape_profile(s["points"], centroid, num_samples)
            data[(i, si)] = {
                "type": s["type"],
                "profile": profile,
                "length": length,
                "color_strip": s["color_strip"],
            }
    return data


def _fast_score(side_data, i, si, j, sj, alpha, beta):
    a, b = side_data[(i, si)], side_data[(j, sj)]
    if a is None or b is None or a["type"] == b["type"]:
        return float("inf")
    ref_len = (a["length"] + b["length"]) / 2.0
    shape_score = shape_compatibility(a["profile"], b["profile"], ref_len)
    color_score = color_ssd(a["color_strip"], b["color_strip"])
    return alpha * shape_score + beta * color_score


# ---------------------------------------------------------------------------
# Main assembly routine
# ---------------------------------------------------------------------------

def assemble(pieces, alpha=0.6, beta=0.4, num_samples=32, tie_epsilon=1e-6, score_fn=None):
    """
    Greedily reconstruct the puzzle grid from a list of described pieces
    (piece_description.describe_all_pieces output: each piece must have
    a "mask" and a "sides" list of 4 dicts with "type"/"points"/"color_strip").

    score_fn : optional callable (piece_i, side_i, piece_j, side_j) -> float,
               where piece_i/piece_j are indices into `pieces` and side_i/
               side_j are indices into that piece's "sides" list. Lower
               must mean "better match"; return float("inf") for an
               impossible pairing. If omitted, the built-in classical
               shape+colour formula (edge_matching.match_score) is used.
               This lets Milestone 2's trained models (siamese/GNN) supply
               learned compatibility scores while reusing this exact same
               placement/rotation/tie-break/dead-end algorithm unchanged.

    Returns
    -------
    dict with:
      "grid"            : {(row, col): {"piece": idx, "rotation": r, "score": s}}
                            normalised so the top-left occupied cell is (0, 0).
      "shape"           : (num_rows, num_cols) of the occupied grid.
      "unplaced_pieces" : list of piece indices never placed (empty if complete).
      "complete"        : True iff every piece was placed.
      "start_cell"      : the (row, col) of the arbitrarily-chosen start piece.
    """
    n = len(pieces)
    if n == 0:
        return {"grid": {}, "shape": (0, 0), "unplaced_pieces": [], "complete": True, "start_cell": None}

    if score_fn is None:
        side_data = _precompute_side_data(pieces, num_samples)

        def score_fn(i, si, j, sj):
            return _fast_score(side_data, i, si, j, sj, alpha, beta)

    start_idx, start_rot = choose_start_piece(pieces)
    grid = {(0, 0): {"piece": start_idx, "rotation": start_rot, "score": 0.0}}
    used = {start_idx}

    frontier = set()

    def add_frontier_from(cell, piece_idx, rotation):
        r, c = cell
        for d, (dr, dc) in _DELTA.items():
            if get_side(pieces[piece_idx], rotation, d)["type"] == "flat":
                continue
            ncell = (r + dr, c + dc)
            if ncell not in grid:
                frontier.add(ncell)

    add_frontier_from((0, 0), start_idx, start_rot)

    while frontier and len(used) < n:
        best = None  # (mean_score, -num_constraints, piece_idx, rotation, cell)

        for cell in frontier:
            r, c = cell
            requirements = []
            for d, (dr, dc) in _DELTA.items():
                ncell = (r + dr, c + dc)
                if ncell in grid:
                    requirements.append((d, grid[ncell]["piece"], grid[ncell]["rotation"]))
            if not requirements:
                continue

            for piece_idx in range(n):
                if piece_idx in used:
                    continue
                for rotation in range(4):
                    total, valid = 0.0, True
                    for (d, neighbor_piece, neighbor_rot) in requirements:
                        si = _side_index(rotation, d)
                        sj = _side_index(neighbor_rot, _OPPOSITE[d])
                        sc = score_fn(piece_idx, si, neighbor_piece, sj)
                        if not np.isfinite(sc):
                            valid = False
                            break
                        total += sc
                    if not valid:
                        continue

                    mean_score = total / len(requirements)
                    candidate = (mean_score, -len(requirements), piece_idx, rotation, cell)
                    if best is None or _is_better(candidate, best, tie_epsilon):
                        best = candidate

        if best is None:
            break  # dead end: no valid placement anywhere on the current frontier

        mean_score, _neg_constraints, piece_idx, rotation, cell = best
        grid[cell] = {"piece": piece_idx, "rotation": rotation, "score": mean_score}
        used.add(piece_idx)
        frontier.discard(cell)
        add_frontier_from(cell, piece_idx, rotation)

    grid, shape, grid_start_cell = _normalize_grid(grid)
    unplaced = [i for i in range(n) if i not in used]
    return {
        "grid": grid,
        "shape": shape,
        "unplaced_pieces": unplaced,
        "complete": len(unplaced) == 0,
        "start_cell": grid_start_cell,
    }


def _is_better(candidate, current_best, eps):
    """
    Tuple comparison implementing the stated tie-breaking rule:
    1) lower mean_score wins outright if the difference exceeds eps;
    2) on a tie, the candidate with MORE placed-neighbour constraints
       wins (stored as -count, so a smaller value here means more
       constraints);
    3) final tie-break: lower piece_idx, for deterministic output.
    """
    c_score, c_negcount, c_piece = candidate[0], candidate[1], candidate[2]
    b_score, b_negcount, b_piece = current_best[0], current_best[1], current_best[2]

    if c_score < b_score - eps:
        return True
    if c_score > b_score + eps:
        return False
    if c_negcount < b_negcount:
        return True
    if c_negcount > b_negcount:
        return False
    return c_piece < b_piece


def _normalize_grid(grid):
    """Shift grid coordinates so the top-left occupied cell is (0, 0)."""
    rows = [r for (r, c) in grid.keys()]
    cols = [c for (r, c) in grid.keys()]
    min_r, min_c = min(rows), min(cols)
    new_grid = {(r - min_r, c - min_c): v for (r, c), v in grid.items()}
    num_rows = max(rows) - min_r + 1
    num_cols = max(cols) - min_c + 1
    return new_grid, (num_rows, num_cols), (0 - min_r, 0 - min_c)
