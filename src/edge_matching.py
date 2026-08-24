"""
edge_matching.py
=================
Scores how well a candidate pair of piece sides fit together, combining:

1. Shape compatibility -- does a tab's protrusion geometrically fill a
   blank's indentation?
2. Colour compatibility -- does the printed image continue smoothly
   across the join (sum of squared differences between the two sides'
   photometric colour-strip signatures)?

Exact formula
-------------
For a candidate pair of sides (a, b), each resampled to N points along
their length (piece_description.sample_color_strip / side_shape_profile
below both use N = num_samples):

    shape_score(a, b) = ( 1/N * sum_t [ profile_a(t) + profile_b(N-1-t) ]^2 ) / L^2

where profile_x(t) is the signed perpendicular deviation of side x from
the straight line joining its two corners (positive = bulges outward/tab,
negative = indents inward/blank), and L is the average of the two sides'
straight-line lengths (this normalises the score to be roughly scale
invariant across different piece/tab sizes). The two profiles are
compared with one of them *reversed*, because when two pieces are placed
edge to edge, their shared boundary is traced in opposite directions by
each piece's own (consistently clockwise) contour walk. A perfect
tab/blank fit has profile_a(t) == -profile_b(N-1-t) everywhere, i.e.
shape_score == 0.

    color_score(a, b) = ( 1/N * sum_t || strip_a(t) - strip_b(N-1-t) ||^2 ) / 255^2

strip_x(t) is the RGB colour sampled just inside side x at position t
(piece_description.sample_color_strip); again one strip is reversed for
the same opposite-traversal reason. Dividing by 255^2 keeps this term
roughly comparable in scale to shape_score.

    match_score(a, b) = alpha * shape_score(a, b) + beta * color_score(a, b)

Default weights alpha=0.6, beta=0.4: shape is weighted somewhat more
heavily because a geometrically wrong tab/blank fit is a hard
impossibility, whereas colour mismatch can be softer evidence (lighting,
JPEG compression, print wear). Both weights are exposed as parameters so
this can be tuned per the report's requirement to state and justify them.

A pair of sides can only be neighbours if their types are complementary
(one "tab", one "blank"); "flat" sides never match another side (they
represent the outer border of the puzzle). Such invalid pairs return
float("inf").
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shape profile
# ---------------------------------------------------------------------------

def _resample_polyline(points, num_samples):
    """Resample a polyline to `num_samples` points evenly spaced by arc length."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    if n == 1:
        return np.tile(pts[0], (num_samples, 1))

    deltas = np.diff(pts, axis=0)
    seg_lens = np.sqrt((deltas ** 2).sum(axis=1))
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cumlen[-1]
    if total_len < 1e-6:
        return np.tile(pts[0], (num_samples, 1))

    targets = np.linspace(0, total_len, num_samples)
    resampled = np.zeros((num_samples, 2), dtype=np.float64)
    for i, t in enumerate(targets):
        idx = np.searchsorted(cumlen, t)
        idx = min(max(idx, 1), n - 1)
        seg_frac = (t - cumlen[idx - 1]) / max(cumlen[idx] - cumlen[idx - 1], 1e-9)
        resampled[i] = pts[idx - 1] + seg_frac * (pts[idx] - pts[idx - 1])
    return resampled


def side_shape_profile(side_points, piece_centroid, num_samples=32):
    """
    Resample a side to `num_samples` points and compute its signed
    perpendicular deviation from the straight line joining its endpoints
    (positive = outward/tab-like bulge, negative = inward/blank-like dip).
    """
    pts = _resample_polyline(side_points, num_samples)
    p1, p2 = pts[0], pts[-1]
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return np.zeros(num_samples), 0.0
    line_unit = line_vec / line_len
    normal = np.array([-line_unit[1], line_unit[0]])

    mid = (p1 + p2) / 2.0
    to_centroid = np.array(piece_centroid, dtype=np.float64) - mid
    if np.dot(normal, to_centroid) > 0:
        normal = -normal

    profile = (pts - p1) @ normal
    return profile, float(line_len)


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

def shape_compatibility(profile_a, profile_b, ref_length):
    """
    Normalised mean-squared mismatch between profile_a and the reversed,
    negated profile_b. 0.0 = perfect complementary fit.
    """
    b_rev = profile_b[::-1]
    diff = profile_a + b_rev
    mse = float(np.mean(diff ** 2))
    return mse / max(ref_length ** 2, 1e-6)


def color_ssd(strip_a, strip_b):
    """
    Normalised mean-squared colour difference between strip_a and the
    reversed strip_b, scaled into roughly [0, 1] by dividing by 255^2.
    """
    a = strip_a.astype(np.float64)
    b = strip_b[::-1].astype(np.float64)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    mse = float(np.mean(diff ** 2))
    return mse / (255.0 ** 2)


# ---------------------------------------------------------------------------
# Combined match score
# ---------------------------------------------------------------------------

def match_score(side_a, side_b, centroid_a, centroid_b, num_samples=32, alpha=0.6, beta=0.4):
    """
    Combined compatibility score for a candidate pair of sides. Lower is
    a better match. Returns float("inf") if the two sides cannot
    possibly be neighbours (not a complementary tab/blank pair).

    match_score = alpha * shape_score + beta * color_score   (see module
    docstring for the exact formulas).
    """
    types = {side_a["type"], side_b["type"]}
    if types != {"tab", "blank"}:
        return float("inf")

    profile_a, len_a = side_shape_profile(side_a["points"], centroid_a, num_samples)
    profile_b, len_b = side_shape_profile(side_b["points"], centroid_b, num_samples)
    ref_len = (len_a + len_b) / 2.0
    shape_score = shape_compatibility(profile_a, profile_b, ref_len)

    strip_a = side_a["color_strip"]
    strip_b = side_b["color_strip"]
    if len(strip_a) != num_samples or len(strip_b) != num_samples:
        idx_a = np.linspace(0, len(strip_a) - 1, num_samples).astype(int)
        idx_b = np.linspace(0, len(strip_b) - 1, num_samples).astype(int)
        strip_a = strip_a[idx_a]
        strip_b = strip_b[idx_b]
    color_score = color_ssd(strip_a, strip_b)

    return alpha * shape_score + beta * color_score


# ---------------------------------------------------------------------------
# Bulk scoring across all pieces
# ---------------------------------------------------------------------------

def _piece_centroid(piece):
    ys, xs = np.where(piece["mask"] > 0)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(ys.mean()), float(xs.mean()))


def compute_all_scores(pieces, alpha=0.6, beta=0.4, num_samples=32):
    """
    Compute match_score for every candidate (piece, side) x (piece, side)
    pair across a list of described pieces (as produced by
    piece_description.describe_all_pieces), skipping same-piece pairs,
    flat sides, and same-type (tab-tab / blank-blank) pairs.

    Returns a dict keyed by (piece_i, side_i, piece_j, side_j) -> score,
    with i < j, containing only finite (valid) candidate pairs.
    """
    centroids = [_piece_centroid(p) for p in pieces]
    scores = {}
    n = len(pieces)
    for i in range(n):
        for si, side_a in enumerate(pieces[i]["sides"]):
            if side_a["type"] == "flat":
                continue
            for j in range(i + 1, n):
                for sj, side_b in enumerate(pieces[j]["sides"]):
                    if side_b["type"] == "flat":
                        continue
                    score = match_score(side_a, side_b, centroids[i], centroids[j],
                                         num_samples=num_samples, alpha=alpha, beta=beta)
                    if np.isfinite(score):
                        scores[(i, si, j, sj)] = score
    return scores


def best_matches_for_side(pieces, piece_idx, side_idx, scores, top_k=3):
    """
    Return the top_k best (lowest-score) candidate matches for one
    specific (piece_idx, side_idx), searching both key orderings since
    compute_all_scores only stores i < j.
    """
    candidates = []
    for (i, si, j, sj), score in scores.items():
        if i == piece_idx and si == side_idx:
            candidates.append(((j, sj), score))
        elif j == piece_idx and sj == side_idx:
            candidates.append(((i, si), score))
    candidates.sort(key=lambda x: x[1])
    return candidates[:top_k]
