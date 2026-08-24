"""
test_segmentation.py
Unit tests for src/segmentation.py
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import segmentation as seg


# ---------------------------------------------------------------------------
# foreground_mask
# ---------------------------------------------------------------------------

def test_foreground_mask_otsu_basic():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 20, (20, 20)).astype(np.uint8)  # noisy dark background
    img[5:15, 5:15] = 200
    mask = seg.foreground_mask(img, method="otsu")
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[10, 10] == 255
    assert mask[0, 0] == 0


def test_foreground_mask_rejects_bad_method():
    img = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
    with pytest.raises(ValueError):
        seg.foreground_mask(img, method="nonsense")


# ---------------------------------------------------------------------------
# foreground_mask_color / estimate_background_color
# ---------------------------------------------------------------------------

def test_estimate_background_color_from_border():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :] = [20, 20, 20]  # background everywhere
    img[15:35, 15:35] = [200, 200, 200]  # interior "piece" -- not touching border
    bg = seg.estimate_background_color(img, border_width=5)
    assert np.allclose(bg, [20, 20, 20], atol=1)


def test_color_distance_map_zero_at_background_color():
    img = np.full((10, 10, 3), 50, dtype=np.uint8)
    dist = seg.color_distance_map(img, background_color=(50, 50, 50))
    assert np.allclose(dist, 0)


def test_foreground_mask_color_recovers_dark_hued_text_on_dark_background():
    # a near-black background with a white piece body containing a
    # saturated-red "text" region that is dark in grayscale luminance
    # (real red ink genuinely converts to a low grayscale value) but
    # clearly different in colour from the near-black background.
    img = np.full((40, 40, 3), (15, 15, 15), dtype=np.uint8)  # background
    img[5:35, 5:35] = (230, 230, 230)  # white piece body
    img[10:20, 10:20] = (150, 20, 25)  # saturated red text region on the piece

    # sanity check: this red really is dark enough in grayscale to fool
    # plain intensity thresholding (luminance formula under-weights red)
    luminance = 0.299 * 150 + 0.587 * 20 + 0.114 * 25
    assert luminance < 70  # much closer to the background's ~15 than to piece white

    mask, bg = seg.foreground_mask_color(img)
    assert mask[15, 15] == 255  # red-text region correctly kept as foreground
    assert mask[0, 0] == 0      # true background still background
    assert mask[6, 6] == 255    # plain white piece body still foreground


def test_foreground_mask_color_accepts_explicit_background_color():
    rng = np.random.default_rng(0)
    img = np.full((20, 20, 3), (100, 100, 100), dtype=np.uint8)
    noise = rng.integers(-3, 4, (20, 20, 3))
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    img[5:15, 5:15] = (250, 250, 250)
    mask, bg = seg.foreground_mask_color(img, background_color=(100, 100, 100))
    assert tuple(bg) == (100, 100, 100)
    assert mask[10, 10] == 255
    assert mask[0, 0] == 0


def test_foreground_mask_color_rejects_grayscale_input():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        seg.foreground_mask_color(img)


# ---------------------------------------------------------------------------
# erode / dilate / morphological_open
# ---------------------------------------------------------------------------

def test_erode_shrinks_blob():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255  # 10x10 block, area 100
    eroded = seg.erode(mask, size=3)
    assert (eroded > 0).sum() < (mask > 0).sum()


def test_dilate_grows_blob():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255
    dilated = seg.dilate(mask, size=3)
    assert (dilated > 0).sum() > (mask > 0).sum()


def test_erode_removes_single_pixel_bridge():
    # two 4x4 blobs connected by a 1-pixel-wide bridge
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    mask[2:6, 10:14] = 255
    mask[3:4, 6:10] = 255  # thin bridge
    labels_before, num_before = seg.connected_components(mask)
    assert num_before == 1  # bridge merges them

    opened = seg.morphological_open(mask, size=3)
    labels_after, num_after = seg.connected_components(opened)
    assert num_after == 2  # opening should break the thin bridge


def test_morphological_open_rejects_even_size():
    mask = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        seg.erode(mask, size=4)
    with pytest.raises(ValueError):
        seg.dilate(mask, size=4)


# ---------------------------------------------------------------------------
# morphological_close / fill_holes
# ---------------------------------------------------------------------------

def test_morphological_close_fixes_edge_notch():
    # a solid square with a small notch bitten out of one edge (simulating
    # dark printed text cutting into the piece's boundary)
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    mask[5:9, 12:16] = 0  # small notch cut into the top edge
    closed = seg.morphological_close(mask, size=7)
    # the notch region should now read as foreground again
    assert closed[6, 13] == 255
    # bulk of the square should be unaffected
    assert closed[15, 15] == 255
    assert closed[0, 0] == 0


def test_fill_holes_fills_enclosed_hole_only():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:17, 3:17] = 255
    mask[8:12, 8:12] = 0  # fully enclosed hole (dark text well inside the piece)
    filled = seg.fill_holes(mask)
    assert filled[10, 10] == 255  # hole is filled
    assert filled[0, 0] == 0      # true background untouched
    assert filled[3, 3] == 255    # original foreground untouched


def test_fill_holes_does_not_fill_border_connected_background():
    # background that touches the image border must NOT be filled, even
    # if a piece sits right next to it
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:17, 3:17] = 255
    filled = seg.fill_holes(mask)
    assert filled[0, 0] == 0
    assert filled[19, 19] == 0
    assert np.array_equal(filled, mask)  # nothing to fill, no enclosed holes


def test_close_and_fill_together_fix_edge_touching_notch():
    # a bay-shaped notch cut into the middle of one edge (open only to the
    # exterior on one side, surrounded by foreground on the other three --
    # the realistic shape of a dark-text artifact, unlike a full corner
    # cutout which morphological closing genuinely cannot bridge).
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    mask[5:9, 13:17] = 0  # small bay cut into the middle of the top edge
    closed = seg.morphological_close(mask, size=7)
    filled = seg.fill_holes(closed)
    assert filled[6, 15] == 255  # notch area now foreground
    assert filled[0, 0] == 0     # true background still untouched


# ---------------------------------------------------------------------------
# filter_by_texture_density
# ---------------------------------------------------------------------------

def test_filter_by_texture_density_removes_high_frequency_blob():
    # a smooth solid square (low edge density, "real piece"-like) vs a
    # coarse checkerboard-textured square of the same size (high edge
    # density, "junk mesh object"-like). Block size 4 keeps the pattern
    # from being smoothed away entirely by Canny's Gaussian stage.
    h, w = 60, 60
    gray = np.zeros((h, w), dtype=np.uint8)
    labels = np.zeros((h, w), dtype=np.int32)

    gray[5:25, 5:25] = 180
    labels[5:25, 5:25] = 1

    block = np.indices((5, 5)).sum(axis=0) % 2
    cb = np.kron(block, np.ones((4, 4))) * 220 + 20
    gray[35:55, 5:25] = cb.astype(np.uint8)
    labels[35:55, 5:25] = 2

    stats = seg.component_stats(labels, 2)
    filtered, kept = seg.filter_by_texture_density(labels, stats, gray, max_edge_density=0.3)
    kept_ids = {s["label"] for s in kept}
    assert 1 in kept_ids
    assert 2 not in kept_ids


def test_filter_by_texture_density_keeps_all_when_threshold_high():
    h, w = 30, 30
    gray = np.zeros((h, w), dtype=np.uint8)
    labels = np.zeros((h, w), dtype=np.int32)
    gray[5:25, 5:25] = 180
    labels[5:25, 5:25] = 1
    stats = seg.component_stats(labels, 1)
    filtered, kept = seg.filter_by_texture_density(labels, stats, gray, max_edge_density=1.0)
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# distance_transform / find_local_maxima / watershed_split
# ---------------------------------------------------------------------------

def test_distance_transform_peak_at_center_of_square():
    mask = np.zeros((41, 41), dtype=np.uint8)
    mask[5:36, 5:36] = 255  # 31x31 square, center at (20,20)
    dist = seg.distance_transform(mask)
    py, px = np.unravel_index(np.argmax(dist), dist.shape)
    assert abs(py - 20) <= 2 and abs(px - 20) <= 2
    # distance at center of a 31-wide square should be close to 15 (half-width)
    assert 12 <= dist[20, 20] <= 17


def test_distance_transform_zero_outside_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    dist = seg.distance_transform(mask)
    assert dist[0, 0] == 0
    assert dist[19, 19] == 0


def test_find_local_maxima_two_separated_blobs():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    mask[35:55, 35:55] = 255
    dist = seg.distance_transform(mask)
    peaks = seg.find_local_maxima(dist, min_distance=15, threshold_ratio=0.5)
    assert len(peaks) == 2
    ys = sorted(p[0] for p in peaks)
    assert ys[0] < 30 < ys[1]


def test_watershed_split_separates_two_touching_squares():
    # two 20x20 squares touching along one full edge (a substantial shared
    # boundary, not a thin bridge -- exactly the case morphological_open
    # cannot fix but watershed can)
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[:, :] = 255  # entirely foreground: two "pieces" fused into one blob
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[10, 5] = 1
    markers[10, 34] = 2
    split = seg.watershed_split(mask, markers)
    # every foreground pixel should be assigned to one of the two labels
    assert set(np.unique(split)) == {1, 2}
    # split should be roughly symmetric (left half mostly 1, right half mostly 2)
    assert (split[:, :10] == 1).sum() > (split[:, :10] == 2).sum()
    assert (split[:, 30:] == 2).sum() > (split[:, 30:] == 1).sum()


def test_farthest_point_select_picks_spatially_distinct_points():
    # three candidate peaks: two very close together (same "piece"), one far away
    peaks = [(10, 10, 5.0), (11, 11, 4.9), (100, 100, 3.0)]
    selected = seg._farthest_point_select(peaks, 2)
    ys = sorted(p[0] for p in selected)
    assert ys == [10, 100] or ys == [11, 100]  # picks the far one, not both close ones


def test_split_touching_components_splits_anomalously_large_blob():
    # 5 normal ~10x10=100-area pieces + one "merged pair": two 12x12 bodies
    # joined by a narrower neck (a dumbbell shape) -- this is the real
    # geometric signature of two touching, irregularly-shaped pieces (a
    # plain undifferentiated rectangle, by contrast, has no actual seam
    # for a distance transform to find, so it wouldn't be a fair test).
    labels = np.zeros((100, 100), dtype=np.int32)
    lbl = 1
    for i in range(5):
        y0 = 5 + i * 15
        labels[y0:y0 + 10, 5:15] = lbl
        lbl += 1

    merged_label = lbl
    labels[5:17, 60:72] = merged_label   # top body, 12x12
    labels[17:20, 63:69] = merged_label  # narrow neck, 3x6
    labels[20:32, 60:72] = merged_label  # bottom body, 12x12

    num_labels = lbl
    stats = seg.component_stats(labels, num_labels)
    merged_original_area = next(s["area"] for s in stats if s["label"] == merged_label)

    split_labels, split_stats = seg.split_touching_components(
        labels, stats, area_ratio_threshold=1.5, min_distance=5,
        min_reference_area=50, min_split_area=10
    )
    # the merged blob is replaced by 2 new components -> total goes from 6 to 7
    assert len(split_stats) == 7
    new_pieces = [s for s in split_stats if s["label"] not in
                  {s2["label"] for s2 in stats if s2["label"] != merged_label}]
    assert len(new_pieces) == 2
    assert sum(s["area"] for s in new_pieces) == merged_original_area  # no pixels lost


def test_split_touching_components_leaves_normal_components_untouched():
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[5:15, 5:15] = 1  # single normal 10x10 piece
    stats = seg.component_stats(labels, 1)
    split_labels, split_stats = seg.split_touching_components(labels, stats, min_reference_area=50)
    assert len(split_stats) == 1
    assert split_stats[0]["area"] == 100


# ---------------------------------------------------------------------------
# connected_components
# ---------------------------------------------------------------------------

def test_connected_components_two_separate_blobs():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    mask[15:18, 15:18] = 255
    labels, num = seg.connected_components(mask)
    assert num == 2
    # the two blobs must have different labels
    l1 = labels[3, 3]
    l2 = labels[16, 16]
    assert l1 != l2
    assert l1 != 0 and l2 != 0


def test_connected_components_single_blob_l_shape():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2] = 255      # vertical arm
    mask[7, 2:8] = 255      # horizontal arm -> L shape, one component
    labels, num = seg.connected_components(mask)
    assert num == 1


def test_connected_components_diagonal_8_vs_4_connectivity():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 255
    mask[2, 2] = 255  # touches (1,1) only diagonally

    labels8, num8 = seg.connected_components(mask, connectivity=8)
    labels4, num4 = seg.connected_components(mask, connectivity=4)
    assert num8 == 1   # merged under 8-connectivity
    assert num4 == 2   # separate under 4-connectivity


def test_connected_components_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    labels, num = seg.connected_components(mask)
    assert num == 0
    assert labels.sum() == 0


def test_connected_components_rejects_bad_connectivity():
    mask = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        seg.connected_components(mask, connectivity=6)


# ---------------------------------------------------------------------------
# component_stats
# ---------------------------------------------------------------------------

def test_component_stats_area_and_bbox():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255  # 3 rows x 4 cols = 12 pixels, bbox (2,3,4,6)
    labels, num = seg.connected_components(mask)
    stats = seg.component_stats(labels, num)
    assert len(stats) == 1
    assert stats[0]["area"] == 12
    assert stats[0]["bbox"] == (2, 3, 4, 6)


def test_component_stats_centroid():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:7, 4:7] = 255  # centered 3x3 block, centroid should be (5,5)
    labels, num = seg.connected_components(mask)
    stats = seg.component_stats(labels, num)
    cy, cx = stats[0]["centroid"]
    assert abs(cy - 5) < 1e-6
    assert abs(cx - 5) < 1e-6


# ---------------------------------------------------------------------------
# filter_components
# ---------------------------------------------------------------------------

def test_filter_components_removes_small_noise():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 255     # 6x6 = 36 px, a "real" piece
    mask[15, 15] = 255       # 1 px noise speck
    labels, num = seg.connected_components(mask)
    stats = seg.component_stats(labels, num)
    filtered, kept = seg.filter_components(labels, stats, min_area=10)
    assert len(kept) == 1
    assert kept[0]["area"] == 36


def test_filter_components_removes_elongated_clutter():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:8, 2:8] = 255      # 6x6 square -> aspect ratio 1.0, keep
    mask[10:12, 2:28] = 255   # thin 2x26 strip -> aspect ratio 13, drop (tape-like)
    labels, num = seg.connected_components(mask)
    stats = seg.component_stats(labels, num)
    filtered, kept = seg.filter_components(labels, stats, min_area=10, max_aspect_ratio=3.0)
    assert len(kept) == 1
    y0, x0, y1, x1 = kept[0]["bbox"]
    assert (y1 - y0 + 1) == 6 and (x1 - x0 + 1) == 6


def test_filter_components_max_area():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, :] = 255  # entire image, 400 px -> too big
    labels, num = seg.connected_components(mask)
    stats = seg.component_stats(labels, num)
    filtered, kept = seg.filter_components(labels, stats, min_area=1, max_area=100)
    assert len(kept) == 0
    assert filtered.sum() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
