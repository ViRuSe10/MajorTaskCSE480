"""
contour_extraction.py
======================
Turns a labeled connected-component mask into per-piece data: an ordered
boundary contour, a cropped bounding box (image + mask), and a normalized
rotation angle.

Functions
---------
- trace_boundary(mask)          -- Moore-neighbor boundary tracing (from
                                    scratch; no cv2.findContours).
- convex_hull(points)           -- Andrew's monotone-chain convex hull.
- min_area_rect(hull_points)    -- rotating-calipers minimum-area bounding
                                    rectangle: (center, (w, h), angle_deg).
- normalize_orientation(mask)   -- canonical rotation angle in (-45, 45]
                                    degrees, derived from min_area_rect.
- extract_piece(image, labels, label_id, stat)
- extract_all_pieces(image, labels, stats)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Moore-neighbor boundary tracing
# ---------------------------------------------------------------------------

# 8 neighbour offsets in clockwise order starting from North.
_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def trace_boundary(mask):
    """
    Trace the outer boundary of a single connected foreground blob using
    Moore-neighbor tracing. Returns an ordered list of (row, col) pixel
    coordinates walking clockwise around the blob's boundary back to the
    start pixel. Returns [] for an empty mask, or a single point for an
    isolated 1-pixel blob.
    """
    binary = mask > 0
    h, w = binary.shape
    ys, xs = np.where(binary)
    if ys.size == 0:
        return []

    start_y = int(ys.min())
    start_x = int(xs[ys == start_y].min())
    start = (start_y, start_x)

    def in_bounds(p):
        return 0 <= p[0] < h and 0 <= p[1] < w

    def is_fg(p):
        return in_bounds(p) and binary[p[0], p[1]]

    # initial backtrack neighbour: the pixel to the west of start, which is
    # guaranteed background since `start` was the leftmost foreground pixel
    # on the topmost occupied row.
    backtrack_dir = 6  # index of West in _DIRS

    current = start
    boundary = [current]
    max_steps = binary.size * 4  # safety bound against pathological inputs

    steps = 0
    while steps < max_steps:
        steps += 1
        found = None
        found_idx = None
        for i in range(8):
            dir_idx = (backtrack_dir + 1 + i) % 8
            dy, dx = _DIRS[dir_idx]
            cand = (current[0] + dy, current[1] + dx)
            if is_fg(cand):
                found = cand
                found_idx = dir_idx
                break

        if found is None:
            # isolated pixel: no foreground neighbour at all
            break

        # new backtrack direction = direction from `found` back to the
        # background neighbour that was checked immediately before it
        prev_dir_idx = (found_idx - 1) % 8
        pdy, pdx = _DIRS[prev_dir_idx]
        prev_bg = (current[0] + pdy, current[1] + pdx)
        rel = (prev_bg[0] - found[0], prev_bg[1] - found[1])
        backtrack_dir = _DIRS.index(rel)

        current = found
        if current == start and len(boundary) > 1:
            break
        boundary.append(current)

    return boundary


# ---------------------------------------------------------------------------
# Convex hull (Andrew's monotone chain)
# ---------------------------------------------------------------------------

def convex_hull(points):
    """
    Convex hull of a set of (x, y) points, returned counter-clockwise,
    without duplicating the start point at the end.
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


# ---------------------------------------------------------------------------
# Minimum-area bounding rectangle (rotating calipers over hull edges)
# ---------------------------------------------------------------------------

def min_area_rect(hull_points):
    """
    Minimum-area bounding rectangle of a convex polygon, found by testing
    the orientation of every hull edge (the optimal rectangle always has
    one side flush with a hull edge).

    hull_points : list of (x, y) tuples, convex hull, any order.

    Returns
    -------
    (center_xy, (width, height), angle_degrees)
    angle_degrees is the rectangle's rotation, in [-90, 90).
    """
    pts = np.array(hull_points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return (0.0, 0.0), (0.0, 0.0), 0.0
    if n == 1:
        return (float(pts[0, 0]), float(pts[0, 1])), (0.0, 0.0), 0.0
    if n == 2:
        edge = pts[1] - pts[0]
        angle = np.degrees(np.arctan2(edge[1], edge[0]))
        center = tuple(pts.mean(axis=0))
        length = float(np.linalg.norm(edge))
        return center, (length, 0.0), angle

    min_area = np.inf
    best = None
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        edge = p2 - p1
        angle = np.arctan2(edge[1], edge[0])
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s], [s, c]])
        rotated = pts @ rot.T

        min_xy = rotated.min(axis=0)
        max_xy = rotated.max(axis=0)
        wdt, hgt = (max_xy - min_xy)
        area = wdt * hgt

        if area < min_area:
            min_area = area
            center_rot = (min_xy + max_xy) / 2.0
            inv_rot = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            center = inv_rot @ center_rot
            best = (tuple(center), (float(wdt), float(hgt)), float(np.degrees(angle)))

    return best


def normalize_orientation(mask):
    """
    Canonical rotation angle of a piece, in degrees, range (-45, 45].
    Derived from the minimum-area bounding rectangle of the piece's
    boundary contour. Since puzzle pieces are roughly square blobs, the
    rectangle's angle is only meaningful modulo 90 degrees; folding it
    into (-45, 45] gives a consistent "how far off axis-aligned" measure
    regardless of which of the 4 nearly-equal sides the calipers picked.
    """
    contour = trace_boundary(mask)
    if len(contour) < 3:
        return 0.0
    # convex_hull expects (x, y); contour points are (row, col) = (y, x)
    xy_points = [(x, y) for (y, x) in contour]
    hull = convex_hull(xy_points)
    _, _, angle = min_area_rect(hull)

    angle = angle % 90
    if angle > 45:
        angle -= 90
    return angle


# ---------------------------------------------------------------------------
# Piece extraction
# ---------------------------------------------------------------------------

def extract_piece(image, labels, label_id, stat):
    """
    Crop a single labeled piece to its bounding box and package its image
    crop, binary mask, boundary contour (in crop-local coordinates), and
    normalized orientation angle.

    image  : the original (H, W) or (H, W, C) scene image.
    labels : connected-component label array, same (H, W) as image.
    stat   : one entry from segmentation.component_stats() for this piece.
    """
    y0, x0, y1, x1 = stat["bbox"]
    crop_img = image[y0:y1 + 1, x0:x1 + 1].copy()
    crop_labels = labels[y0:y1 + 1, x0:x1 + 1]
    crop_mask = np.where(crop_labels == label_id, 255, 0).astype(np.uint8)

    contour = trace_boundary(crop_mask)
    angle = normalize_orientation(crop_mask)

    return {
        "label": label_id,
        "bbox": stat["bbox"],
        "image": crop_img,
        "mask": crop_mask,
        "contour": contour,
        "orientation_deg": angle,
    }


def extract_all_pieces(image, labels, stats):
    """Run extract_piece for every component in `stats`."""
    return [extract_piece(image, labels, s["label"], s) for s in stats]
