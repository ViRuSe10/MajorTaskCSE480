"""
segmentation.py
================
Isolates individual puzzle pieces from a scene photo:
  1. foreground_mask()        -- binarize pieces vs. background, reusing
                                  the thresholding module.
  2. connected_components()   -- label each connected foreground blob with
                                  a distinct integer ID, implemented from
                                  scratch (no cv2.connectedComponents /
                                  scipy.ndimage.label).
  3. component_stats()        -- per-blob area / bounding box / centroid.
  4. filter_components()      -- drop blobs that are too small/large or
                                  too elongated to plausibly be a puzzle
                                  piece (e.g. tape, screws, foam scraps).

Connected-component implementation notes
-----------------------------------------
A naive per-pixel two-pass labeling in pure Python is O(H*W) with a large
constant factor and is far too slow for a ~2-megapixel photo. Instead we
use the classic run-length-based two-pass algorithm: each row is first
decomposed into contiguous foreground "runs" (a vectorised numpy diff, not
a per-pixel loop), runs are given provisional labels, and provisional
labels are merged via union-find whenever a run in row y overlaps a run in
row y-1 (8-connectivity checks a 1-pixel-wider overlap window to also
catch diagonal touches). This keeps the required loop bounded by the
number of *runs* (typically a few hundred per image) rather than the
number of *pixels* (millions), while remaining a from-scratch labeling
algorithm rather than a call to an existing connected-components routine.
"""

import numpy as np

from .thresholding import otsu_threshold, adaptive_threshold, global_threshold


# ---------------------------------------------------------------------------
# Foreground mask
# ---------------------------------------------------------------------------

def foreground_mask(image_gray, method="otsu", **kwargs):
    """
    Produce a binary foreground mask (pieces = 255, background = 0) using
    one of the thresholding routines from thresholding.py.

    method : "otsu" | "adaptive" | "global"
    kwargs : forwarded to the chosen thresholding function
             (e.g. block_size=, C=, method= for adaptive; thresh= for global)
    """
    if image_gray.ndim != 2:
        raise ValueError("foreground_mask expects a single-channel image.")

    if method == "otsu":
        mask, _ = otsu_threshold(image_gray)
    elif method == "adaptive":
        mask = adaptive_threshold(image_gray, **kwargs)
    elif method == "global":
        mask = global_threshold(image_gray, kwargs.get("thresh", 127))
    else:
        raise ValueError("method must be 'otsu', 'adaptive', or 'global'.")
    return mask


# ---------------------------------------------------------------------------
# Colour-distance foreground mask (handles hue-different-but-similarly-dark
# foreground content that plain grayscale thresholding cannot separate)
# ---------------------------------------------------------------------------

def estimate_background_color(color_img, border_width=10):
    """
    Estimate the scene's background colour from a strip around the image
    border. For a full scattered-pieces scene photo the border is
    overwhelmingly background/tray, not piece content, so its median
    colour is a robust background estimate.
    """
    top = color_img[:border_width, :, :].reshape(-1, 3)
    bottom = color_img[-border_width:, :, :].reshape(-1, 3)
    left = color_img[:, :border_width, :].reshape(-1, 3)
    right = color_img[:, -border_width:, :].reshape(-1, 3)
    border_pixels = np.concatenate([top, bottom, left, right], axis=0)
    return np.median(border_pixels, axis=0)


def color_distance_map(color_img, background_color):
    """Per-pixel Euclidean distance (in RGB space) from `background_color`."""
    diff = color_img.astype(np.float64) - np.asarray(background_color, dtype=np.float64)
    return np.sqrt((diff ** 2).sum(axis=-1))


def foreground_mask_color(color_img, background_color=None, border_width=10):
    """
    Foreground mask via colour-distance thresholding instead of plain
    grayscale intensity.

    Plain grayscale Otsu thresholding (foreground_mask) picks a single
    brightness cut point, which fails whenever a large area of printed
    foreground content is genuinely dark in luminance despite having a
    colour clearly different from the background -- e.g. saturated red
    ink converts to a fairly low grayscale value (the standard luminance
    formula under-weights red), so red printed text/graphics near a dark
    background can fall on the "background" side of a brightness-only
    threshold, cutting large chunks out of a piece's silhouette (not just
    a small edge notch -- morphological_close's small kernel can't bridge
    a defect this large).

    This function instead measures each pixel's colour distance from an
    estimated (or supplied) background colour and thresholds *that*
    distance map with Otsu -- a saturated red pixel is far from a
    near-black background in colour space even though it is dark in
    grayscale, so it is correctly kept as foreground.

    Returns (mask, background_color_used).
    """
    if color_img.ndim != 3:
        raise ValueError("foreground_mask_color expects a 3-channel color image.")
    if background_color is None:
        background_color = estimate_background_color(color_img, border_width)

    dist = color_distance_map(color_img, background_color)
    max_dist = dist.max()
    dist_norm = ((dist / max_dist) * 255).astype(np.uint8) if max_dist > 0 else dist.astype(np.uint8)
    mask, _t = otsu_threshold(dist_norm)
    return mask, background_color


# ---------------------------------------------------------------------------
# Morphological cleanup (erosion / dilation / opening)
# ---------------------------------------------------------------------------

def erode(mask, size=3):
    """
    Binary erosion: a pixel stays foreground only if every pixel in its
    size x size neighbourhood is foreground. Shrinks blobs and snaps thin
    connections (like two pieces touching at a single point/edge).
    Implemented via a vectorised sliding-window `.all()` reduction, not a
    per-pixel loop.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd.")
    binary = mask > 0
    pad = size // 2
    padded = np.pad(binary, pad, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    eroded = windows.all(axis=(-1, -2))
    return (eroded.astype(np.uint8)) * 255


def dilate(mask, size=3):
    """
    Binary dilation: a pixel becomes foreground if any pixel in its
    size x size neighbourhood is foreground. Grows blobs back after
    erosion, restoring piece size while keeping thin bridges broken.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd.")
    binary = mask > 0
    pad = size // 2
    padded = np.pad(binary, pad, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    dilated = windows.any(axis=(-1, -2))
    return (dilated.astype(np.uint8)) * 255


def morphological_open(mask, size=3):
    """
    Erosion followed by dilation with the same kernel size. Removes thin
    connections between blobs (e.g. two puzzle pieces resting against
    each other) while approximately preserving the size/shape of blobs
    that were not thinly connected to begin with.
    """
    return dilate(erode(mask, size), size)


def morphological_close(mask, size=3):
    """
    Dilation followed by erosion with the same kernel size. Bridges small
    gaps/notches in an otherwise-solid blob -- e.g. a dark printed detail
    (text, a small graphic) near a piece's edge that thresholding
    misclassified as background, biting a small notch out of the piece's
    silhouette right at its boundary. Closing fills such small bays back
    in while leaving the piece's real, much larger-scale tab/blank
    geometry essentially unchanged (for a small-enough kernel).
    """
    return erode(dilate(mask, size), size)


def fill_holes(mask):
    """
    Fill any background-labeled region that is fully enclosed within
    foreground (not reachable from the image border) -- e.g. dark printed
    text/graphics well inside a piece's body that thresholding
    misclassified as background, leaving a small "island" hole. True
    background is, by definition, connected to the image border, so any
    background component that ISN'T is treated as a hole and filled in.

    Complements morphological_close: closing fixes small edge-touching
    bays (background notches cutting in from the boundary), fill_holes
    fixes fully-enclosed interior islands. Using both together handles
    both textual-edge cases in one pass.
    """
    background = mask == 0
    bg_mask_uint8 = (background.astype(np.uint8)) * 255
    bg_labels, num_bg = connected_components(bg_mask_uint8, connectivity=4)

    h, w = mask.shape
    border_labels = set(bg_labels[0, :].tolist()) | set(bg_labels[-1, :].tolist()) \
        | set(bg_labels[:, 0].tolist()) | set(bg_labels[:, -1].tolist())
    border_labels.discard(0)

    is_hole = background & ~np.isin(bg_labels, list(border_labels))
    filled = mask.copy()
    filled[is_hole] = 255
    return filled


# ---------------------------------------------------------------------------
# Connected component labeling (run-based, from scratch)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self.parent = [0]  # index 0 unused (0 = background label)

    def make_label(self):
        label = len(self.parent)
        self.parent.append(label)
        return label

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _row_runs(row):
    """Return list of (start, end) half-open intervals of True runs in a 1D bool array."""
    if row.size == 0 or not row.any():
        return []
    diff = np.diff(row.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if row[0]:
        starts = [0] + starts
    if row[-1]:
        ends = ends + [row.size]
    return list(zip(starts, ends))


def connected_components(mask, connectivity=8):
    """
    Label connected foreground (nonzero) regions of a binary mask.

    Returns
    -------
    labels     : int32 array, same shape as mask, 0 = background,
                 1..num_labels = distinct connected components.
    num_labels : number of connected components found.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    binary = mask > 0
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    uf = _UnionFind()

    prev_runs = []  # list of (start, end, label) for the previous row
    for y in range(h):
        runs = _row_runs(binary[y])
        current_runs = []
        for (s, e) in runs:
            if connectivity == 8:
                s_chk, e_chk = max(s - 1, 0), min(e + 1, w)
            else:
                s_chk, e_chk = s, e

            overlapping = [pl for (ps, pe, pl) in prev_runs if pe > s_chk and ps < e_chk]

            if not overlapping:
                lbl = uf.make_label()
            else:
                lbl = min(overlapping)
                for other in overlapping:
                    uf.union(lbl, other)

            labels[y, s:e] = lbl
            current_runs.append((s, e, lbl))
        prev_runs = current_runs

    # resolve provisional labels to final consecutive labels via union-find roots
    used = np.unique(labels[labels > 0])
    root_of = {int(l): uf.find(int(l)) for l in used}
    roots_sorted = sorted(set(root_of.values()))
    final_of_root = {r: i + 1 for i, r in enumerate(roots_sorted)}

    lut = np.zeros(len(uf.parent), dtype=np.int32)
    for l, r in root_of.items():
        lut[l] = final_of_root[r]

    final_labels = lut[labels]
    num_labels = len(roots_sorted)
    return final_labels, num_labels


# ---------------------------------------------------------------------------
# Component statistics + filtering
# ---------------------------------------------------------------------------

def component_stats(labels, num_labels):
    """
    Compute area, bounding box (y0, x0, y1, x1 inclusive), and centroid
    (cy, cx) for every labeled component.
    """
    stats = []
    for lbl in range(1, num_labels + 1):
        ys, xs = np.where(labels == lbl)
        if ys.size == 0:
            continue
        stats.append({
            "label": lbl,
            "area": int(ys.size),
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
            "centroid": (float(ys.mean()), float(xs.mean())),
        })
    return stats


def filter_components(labels, stats, min_area=1000, max_area=None, max_aspect_ratio=3.0):
    """
    Keep only components that plausibly look like a puzzle piece: bounded
    area, and a roughly square (not extremely elongated) bounding box --
    this filters out thin clutter (tape, screws) that otherwise survives
    simple thresholding.

    Returns
    -------
    filtered_labels : labels array with rejected components zeroed out
    kept_stats       : list of stats dicts for the components that passed
    """
    kept_stats = []
    keep_ids = []
    for s in stats:
        y0, x0, y1, x1 = s["bbox"]
        bh, bw = (y1 - y0 + 1), (x1 - x0 + 1)
        aspect = max(bh, bw) / max(min(bh, bw), 1)

        if s["area"] < min_area:
            continue
        if max_area is not None and s["area"] > max_area:
            continue
        if aspect > max_aspect_ratio:
            continue

        keep_ids.append(s["label"])
        kept_stats.append(s)

    if keep_ids:
        mask = np.isin(labels, keep_ids)
    else:
        mask = np.zeros_like(labels, dtype=bool)
    filtered_labels = np.where(mask, labels, 0)
    return filtered_labels, kept_stats


# ---------------------------------------------------------------------------
# Texture-based filtering (catches clutter that passes geometric filters)
# ---------------------------------------------------------------------------

def filter_by_texture_density(labels, stats, gray_image, max_edge_density=0.13, canny_kwargs=None):
    """
    Remove components whose internal Canny edge-pixel density exceeds
    max_edge_density. Some clutter objects (e.g. a perforated/mesh object
    that happens to be piece-sized and roughly square) pass the geometric
    filter_components check but are visually far busier/higher-frequency
    than any real puzzle piece -- real pieces are mostly smooth printed
    regions with comparatively low internal edge density, so a density
    threshold catches this kind of object that pure area/aspect-ratio
    filtering cannot.

    gray_image : the full-scene grayscale image (same one components were
                 detected in), used to crop and run Canny per component.
    """
    from .edge_detection import canny
    canny_kwargs = canny_kwargs or {}

    kept_stats = []
    keep_ids = []
    for s in stats:
        y0, x0, y1, x1 = s["bbox"]
        crop_gray = gray_image[y0:y1 + 1, x0:x1 + 1]
        crop_mask = labels[y0:y1 + 1, x0:x1 + 1] == s["label"]
        area = int(crop_mask.sum())
        if area == 0:
            continue
        edges = canny(crop_gray, **canny_kwargs)
        density = float(((edges > 0) & crop_mask).sum()) / area

        if density <= max_edge_density:
            keep_ids.append(s["label"])
            kept_stats.append(s)

    if keep_ids:
        mask = np.isin(labels, keep_ids)
    else:
        mask = np.zeros_like(labels, dtype=bool)
    filtered_labels = np.where(mask, labels, 0)
    return filtered_labels, kept_stats


# ---------------------------------------------------------------------------
# Watershed split for touching pieces (substantial shared edge, not just a
# thin single-pixel bridge -- morphological_open alone cannot separate these)
# ---------------------------------------------------------------------------

def distance_transform(mask):
    """
    Chamfer distance transform: approximate Euclidean distance from every
    foreground pixel to the nearest background pixel, computed from
    scratch via the classic two-pass chamfer 3-4 algorithm (orthogonal
    step cost 3, diagonal step cost 4, result scaled by 1/3 to
    approximate true Euclidean distance) -- no scipy.ndimage.distance_transform.

    A piece's centre is, by definition, the point farthest from its own
    boundary, so this transform's local maxima are good candidate piece
    centres -- the basis for splitting two touching pieces below.
    """
    binary = mask > 0
    h, w = binary.shape
    INF = float(h + w) * 4.0
    dist = np.where(binary, INF, 0.0)

    for y in range(h):
        for x in range(w):
            if not binary[y, x]:
                continue
            d = dist[y, x]
            if y > 0:
                d = min(d, dist[y - 1, x] + 3)
                if x > 0:
                    d = min(d, dist[y - 1, x - 1] + 4)
                if x < w - 1:
                    d = min(d, dist[y - 1, x + 1] + 4)
            if x > 0:
                d = min(d, dist[y, x - 1] + 3)
            dist[y, x] = d

    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if not binary[y, x]:
                continue
            d = dist[y, x]
            if y < h - 1:
                d = min(d, dist[y + 1, x] + 3)
                if x < w - 1:
                    d = min(d, dist[y + 1, x + 1] + 4)
                if x > 0:
                    d = min(d, dist[y + 1, x - 1] + 4)
            if x < w - 1:
                d = min(d, dist[y, x + 1] + 3)
            dist[y, x] = d

    dist = dist / 3.0
    dist[~binary] = 0.0
    return dist


def find_local_maxima(dist_map, min_distance=15, threshold_ratio=0.4):
    """
    Find well-separated local maxima of a distance map: candidate piece
    centres. A pixel qualifies if it is the maximum within its own
    (2*min_distance+1) window AND its value is at least threshold_ratio
    times the map's global maximum (rejecting shallow, noise-level bumps).
    Candidates are then greedily de-duplicated so no two selected maxima
    are closer than min_distance apart (keeping the higher one).
    """
    h, w = dist_map.shape
    max_val = dist_map.max()
    if max_val <= 0:
        return []
    threshold = threshold_ratio * max_val

    candidates = []
    for y in range(h):
        for x in range(w):
            v = dist_map[y, x]
            if v < threshold:
                continue
            y0, y1 = max(0, y - min_distance), min(h, y + min_distance + 1)
            x0, x1 = max(0, x - min_distance), min(w, x + min_distance + 1)
            if v >= dist_map[y0:y1, x0:x1].max():
                candidates.append((y, x, v))

    candidates.sort(key=lambda c: -c[2])
    selected = []
    for (y, x, v) in candidates:
        if all((y - sy) ** 2 + (x - sx) ** 2 >= min_distance ** 2 for (sy, sx, sv) in selected):
            selected.append((y, x, v))
    return selected


def watershed_split(mask, markers):
    """
    Marker-controlled watershed via priority-flood (implemented from
    scratch with a min-heap; no skimage/cv2 watershed). Floods the
    "elevation" surface (max(distance) - distance -- so distance-transform
    peaks become basins) outward from each labeled marker, assigning each
    foreground pixel to whichever marker's flood reaches it first (lowest
    elevation path). The boundary that forms between two markers' regions
    is the watershed line, splitting the blob along the natural neck
    between two touching pieces.

    markers : int array, same shape as mask, 0 except at seed pixels
              (positive integer label per seed).
    """
    import heapq

    binary = mask > 0
    dist = distance_transform(mask)
    elevation = dist.max() - dist

    labels = markers.copy()
    h, w = mask.shape
    heap = []
    counter = 0  # tie-break by insertion order, not (y, x) -- ties on elevation
    # are common across flat/interior regions, and breaking ties by raw
    # coordinate systematically favours whichever marker has smaller (y, x),
    # letting it flood the whole region unfairly. Insertion order gives a
    # proper simultaneous (round-robin) multi-source flood instead.
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        heapq.heappush(heap, (elevation[ny, nx], counter, ny, nx, labels[y, x]))
                        counter += 1

    while heap:
        elev, _order, y, x, lbl = heapq.heappop(heap)
        if labels[y, x] != 0:
            continue
        labels[y, x] = lbl
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                heapq.heappush(heap, (elevation[ny, nx], counter, ny, nx, lbl))
                counter += 1

    return labels


def _farthest_point_select(peaks, n):
    """
    Select up to n peaks from `peaks` (list of (y, x, value)) via farthest-
    point sampling: start with the highest-value peak, then repeatedly add
    whichever remaining peak has the largest minimum distance to the
    already-selected set. Favours spatially distinct points over merely
    high-value ones.
    """
    if len(peaks) <= n:
        return list(peaks)
    remaining = list(peaks)
    remaining.sort(key=lambda p: -p[2])
    selected = [remaining.pop(0)]
    while len(selected) < n and remaining:
        best_idx, best_dist = 0, -1.0
        for i, (y, x, v) in enumerate(remaining):
            d = min((y - sy) ** 2 + (x - sx) ** 2 for (sy, sx, sv) in selected)
            if d > best_dist:
                best_dist, best_idx = d, i
        selected.append(remaining.pop(best_idx))
    return selected


def split_touching_components(labels, stats, area_ratio_threshold=1.6, min_distance=15,
                               min_split_area=1500, min_reference_area=3000):
    """
    Detect components whose area is anomalously large relative to the
    typical (median) piece area -- a strong signal that two touching
    pieces were merged into one connected component during segmentation
    -- and split each such component via watershed on its own distance
    transform. Components at or below the threshold are passed through
    unchanged.

    The reference median is computed only from components with
    area >= min_reference_area, so it isn't dragged down by small noise
    specks (dust, tape fragments) that always vastly outnumber real
    pieces in the raw, pre-filtered component list -- using the full,
    unfiltered list here would make the median far too small and cause
    ordinary single pieces to be misidentified as merged pairs.

    Returns an updated (labels, stats) pair with merged components
    replaced by their split sub-components (assigned fresh label IDs).
    """
    if not stats:
        return labels, stats

    reference_areas = [s["area"] for s in stats if s["area"] >= min_reference_area]
    if not reference_areas:
        return labels, stats
    median_area = float(np.median(reference_areas))

    next_label = max(s["label"] for s in stats) + 1
    new_labels = labels.copy()
    new_stats = []

    for s in stats:
        if s["area"] < min_reference_area or s["area"] <= area_ratio_threshold * median_area:
            new_stats.append(s)
            continue

        y0, x0, y1, x1 = s["bbox"]
        crop_mask = (labels[y0:y1 + 1, x0:x1 + 1] == s["label"]).astype(np.uint8) * 255
        dist = distance_transform(crop_mask)
        peaks = find_local_maxima(dist, min_distance=min_distance, threshold_ratio=0.4)

        # Real pieces are non-convex (tabs create their own local distance-
        # transform bumps), so find_local_maxima typically returns more
        # candidates than the number of merged pieces -- and, because two
        # touching pieces share connectivity right at their join, the
        # raw peak VALUE is not reliable for picking which candidates
        # belong to which piece (one piece's geometry can inflate its
        # peak value well above the other piece's true centre). Instead,
        # select markers by farthest-point sampling: start from the
        # strongest peak, then repeatedly add whichever remaining peak is
        # farthest from all markers picked so far. This favours spatially
        # distinct piece centres over merely "high distance-transform
        # value" candidates, which are often multiple bumps on one piece.
        n_expected = max(2, round(s["area"] / median_area))
        peaks = _farthest_point_select(peaks, n_expected)

        if len(peaks) < 2:
            new_stats.append(s)  # couldn't confidently find 2 separate centres; leave as-is
            continue

        markers = np.zeros(crop_mask.shape, dtype=np.int32)
        for i, (py, px, pv) in enumerate(peaks):
            markers[py, px] = i + 1
        split_crop = watershed_split(crop_mask, markers)

        new_labels[y0:y1 + 1, x0:x1 + 1][labels[y0:y1 + 1, x0:x1 + 1] == s["label"]] = 0
        for i in range(1, len(peaks) + 1):
            sub_mask = split_crop == i
            if sub_mask.sum() < min_split_area:
                continue
            global_label = next_label
            next_label += 1
            ys, xs = np.where(sub_mask)
            new_labels[y0:y1 + 1, x0:x1 + 1][sub_mask] = global_label
            new_stats.append({
                "label": global_label,
                "area": int(sub_mask.sum()),
                "bbox": (int(ys.min()) + y0, int(xs.min()) + x0,
                         int(ys.max()) + y0, int(xs.max()) + x0),
                "centroid": (float(ys.mean()) + y0, float(xs.mean()) + x0),
            })

    return new_labels, new_stats
