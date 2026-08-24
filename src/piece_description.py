"""
piece_description.py
=====================
Describes each of a piece's four sides so that candidate neighbours can
later be compared (edge_matching.py):

- find_corners(contour)                  -- locate 4 corner indices on the
                                             boundary contour.
- split_into_sides(contour, corners)     -- divide the contour into 4 side
                                             point-lists between consecutive
                                             corners.
- classify_side(side, centroid, ...)     -- "tab" / "blank" / "flat".
- sample_color_strip(image, side, ...)   -- fixed-length photometric
                                             signature sampled just inside
                                             the piece along the side.
- describe_piece(piece)                  -- ties the above together for one
                                             piece dict produced by
                                             contour_extraction.extract_piece.

Corner-finding design choice
-----------------------------
Puzzle pieces are, at their core, a roughly square/rectangular body with
small tab/blank protrusions and indentations layered on top. Rather than
running a separate curvature-based corner detector (which is easily
thrown off by the tab bumps themselves), we reuse the piece's minimum-area
bounding rectangle (contour_extraction.min_area_rect) as an estimate of
the 4 true corners, then snap each rectangle corner to its nearest actual
point on the traced contour. This is robust because the bounding
rectangle is dominated by the piece's stable square body, not by the
comparatively small tabs/blanks.
"""

import numpy as np

from .contour_extraction import convex_hull, min_area_rect


# ---------------------------------------------------------------------------
# Corner detection + side splitting
# ---------------------------------------------------------------------------

def _ensure_four_indices(idxs, n):
    """Guarantee exactly 4 distinct, sorted contour indices, with a simple
    even-spacing fallback if the primary method collapses (very small or
    degenerate contours)."""
    idxs = sorted(set(int(i) for i in idxs))
    if len(idxs) == 4:
        return idxs
    if n == 0:
        return [0, 0, 0, 0]
    step = max(n // 4, 1)
    fallback = sorted(set((i * step) % n for i in range(4)))
    i = 0
    while len(fallback) < 4 and i < n:
        if i not in fallback:
            fallback.append(i)
        i += 1
    return sorted(fallback)[:4]


def find_corners(contour):
    """
    Locate 4 corner indices (into `contour`) approximating the piece's
    true square corners, via nearest-contour-point snapping of the
    minimum-area bounding rectangle's 4 corners.
    """
    n = len(contour)
    if n < 8:
        return _ensure_four_indices(np.linspace(0, max(n - 1, 0), min(n, 4), dtype=int), n)

    xy_points = [(x, y) for (y, x) in contour]  # contour is (row, col) = (y, x)
    hull = convex_hull(xy_points)
    if len(hull) < 3:
        return _ensure_four_indices(np.linspace(0, n - 1, 4, dtype=int), n)

    center, (w, h), angle = min_area_rect(hull)
    cx, cy = center
    angle_rad = np.radians(angle)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    hw, hh = w / 2.0, h / 2.0
    local_corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

    rect_corners_xy = []
    for (lx, ly) in local_corners:
        rx = cx + lx * c - ly * s
        ry = cy + lx * s + ly * c
        rect_corners_xy.append((rx, ry))

    contour_arr = np.array(contour, dtype=np.float64)  # (row, col)
    idxs = []
    for (rx, ry) in rect_corners_xy:
        target = np.array([ry, rx])  # (row, col)
        d2 = np.sum((contour_arr - target) ** 2, axis=1)
        idxs.append(int(np.argmin(d2)))

    return _ensure_four_indices(idxs, n)


def split_into_sides(contour, corner_indices):
    """
    Split the boundary contour into 4 ordered point-lists, one per side,
    walking the contour between consecutive corner indices.
    """
    idxs = sorted(set(corner_indices))
    n = len(contour)
    sides = []
    for i in range(4):
        start, end = idxs[i], idxs[(i + 1) % 4]
        if start <= end:
            side = contour[start:end + 1]
        else:
            side = contour[start:] + contour[:end + 1]
        if len(side) < 2:
            side = [contour[start], contour[end]]
        sides.append(side)
    return sides


# ---------------------------------------------------------------------------
# tab / blank / flat classification
# ---------------------------------------------------------------------------

def classify_side(side_points, piece_centroid, flat_thresh_ratio=0.1):
    """
    Classify a side as "tab" (protrusion), "blank" (indentation), or
    "flat" (straight border edge), from its geometry alone.

    Method: fit the straight line between the side's two endpoints, then
    measure each point's signed perpendicular distance from that line
    (oriented so positive = bulging away from the piece centroid, negative
    = indenting toward it). A side is "flat" if the largest deviation in
    either direction is small relative to the side's own length; otherwise
    it is "tab" or "blank" according to whichever direction dominates.

    Returns
    -------
    (side_type, magnitude) where magnitude is the largest deviation (px).
    """
    pts = np.array(side_points, dtype=np.float64)
    if len(pts) < 3:
        return "flat", 0.0

    p1, p2 = pts[0], pts[-1]
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return "flat", 0.0
    line_unit = line_vec / line_len
    normal = np.array([-line_unit[1], line_unit[0]])

    mid = (p1 + p2) / 2.0
    to_centroid = np.array(piece_centroid, dtype=np.float64) - mid
    if np.dot(normal, to_centroid) > 0:
        normal = -normal  # make `normal` point outward (away from centroid)

    signed_dist = (pts - p1) @ normal
    max_out = float(signed_dist.max())
    max_in = float(-signed_dist.min()) if signed_dist.min() < 0 else 0.0

    flat_thresh = flat_thresh_ratio * line_len
    if max_out < flat_thresh and max_in < flat_thresh:
        return "flat", max(max_out, max_in)
    if max_out >= max_in:
        return "tab", max_out
    return "blank", max_in


# ---------------------------------------------------------------------------
# Photometric (color-strip) signature
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


def sample_color_strip(image, side_points, piece_centroid, offset=4, num_samples=32):
    """
    Sample a fixed-length strip of colour just inside the piece boundary,
    along `side_points`, to serve as a photometric signature for matching.

    image           : the piece's cropped (H, W, 3) or (H, W) image.
    side_points     : ordered (row, col) points along this side's contour.
    piece_centroid  : (row, col) of the piece's centroid, used to sample
                       *inward* (offset toward the centroid) rather than
                       off the piece into the background.
    offset          : how many pixels inward (toward centroid) to sample.
    num_samples     : fixed output length, so strips from differently
                       sized sides are directly comparable in edge_matching.
    """
    resampled = _resample_polyline(side_points, num_samples)
    h, w = image.shape[0], image.shape[1]
    is_color = image.ndim == 3
    n_channels = image.shape[2] if is_color else 1

    colors = np.zeros((num_samples, n_channels), dtype=np.float64)
    centroid = np.array(piece_centroid, dtype=np.float64)

    for i, (ry, rx) in enumerate(resampled):
        direction = centroid - np.array([ry, rx])
        norm = np.linalg.norm(direction)
        unit = direction / norm if norm > 1e-6 else np.zeros(2)
        sy, sx = np.array([ry, rx]) + unit * offset
        sy = int(round(np.clip(sy, 0, h - 1)))
        sx = int(round(np.clip(sx, 0, w - 1)))
        colors[i] = image[sy, sx] if is_color else image[sy, sx]

    return colors.astype(np.uint8)


# ---------------------------------------------------------------------------
# Full per-piece description
# ---------------------------------------------------------------------------

def describe_piece(piece, offset=4, num_samples=32):
    """
    Given a piece dict from contour_extraction.extract_piece, compute its
    4 corners, split the boundary into sides, and classify + photometrically
    sample each side.

    Returns the piece dict with an added "sides" key: a list of 4 dicts,
    each with "points", "type" (tab/blank/flat), "magnitude", and
    "color_strip".
    """
    contour = piece["contour"]
    mask = piece["mask"]
    image = piece["image"]

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        centroid = (0.0, 0.0)
    else:
        centroid = (float(ys.mean()), float(xs.mean()))

    corner_idxs = find_corners(contour)
    sides_points = split_into_sides(contour, corner_idxs)

    sides = []
    for side_points in sides_points:
        side_type, magnitude = classify_side(side_points, centroid)
        strip = sample_color_strip(image, side_points, centroid, offset=offset, num_samples=num_samples)
        sides.append({
            "points": side_points,
            "type": side_type,
            "magnitude": magnitude,
            "color_strip": strip,
        })

    piece = dict(piece)
    piece["corners"] = [contour[i] for i in corner_idxs]
    piece["sides"] = sides
    return piece


def describe_all_pieces(pieces, offset=4, num_samples=32):
    return [describe_piece(p, offset=offset, num_samples=num_samples) for p in pieces]
