# MajorTask — CSE480 Machine Vision: Jigsaw Puzzle Reconstruction

Reusable image-processing library that reconstructs a jigsaw puzzle from its
shuffled, possibly rotated pieces using classical techniques (Milestone 1),
and later compares against learned models (Milestone 2).

## Dataset

Sourced from Roboflow / Kaggle "Jigsaw Puzzle Pieces Dataset" (single 35-piece
physical puzzle, photographed under many scattered/rotated/partial
configurations, with YOLO-format detection labels).

Local copy lives under `detection/`:

- `detection/input/` — full scene photos (scattered pieces, possibly with
  non-puzzle clutter).
- `detection/sample_pieces/` — isolated single-piece reference images.
- `detection/ground_truth/` — YOLO label files and the derived piece
  ID -> grid position mapping (solved once, reused across all images since
  the physical puzzle is the same throughout the dataset).

## Status

- [x] `src/enhancement.py` — Gaussian blur, median filter, histogram
      equalization, contrast stretching, unsharp mask / Laplacian sharpen.
- [x] `src/thresholding.py` — global, Otsu, adaptive (mean/Gaussian).
- [x] `src/edge_detection.py` — Sobel, Prewitt, full Canny (smoothing,
      NMS, double threshold, hysteresis).
- [x] `src/segmentation.py` — foreground mask, from-scratch connected
      components (run-based, union-find), size/aspect-ratio filtering.
- [x] `src/contour_extraction.py` — Moore-neighbor boundary tracing,
      convex hull, min-area-rect orientation normalization, piece crop.
- [x] `src/piece_description.py` — corner detection, side splitting,
      tab/blank/flat classification, color-strip photometric signature.
- [x] `src/edge_matching.py` — shape-fit + colour-SSD compatibility score
      for candidate tab/blank side pairs (weighted, formula documented
      in-module).
- [x] `src/assembly.py` — greedy best-first grid placement, rotation
      resolution, tie-breaking rule, dead-end handling.
- [x] `src/evaluation.py` — intrinsic quality metrics (completion, fill
      ratio, mean edge-mismatch score) + reconstructed-image compositing.
- [x] `main.py` — end-to-end routine (photo in -> reconstructed image +
      quality score out).

## Known limitation (real data)

On the real 35-piece scattered photo, quality_score is low (~0.018,
meaning individual matched seams fit very well), but fill_ratio is only
~44% -- the assembled grid is looser/larger than a tight rectangle.
Root cause: only 2 of the expected 4 corner pieces are detected, because
`classify_side`'s flat/tab/blank threshold misclassifies some genuinely-
flat real, worn/tilted piece edges as very shallow tabs/blanks. Candidate
fix to explore next: tune `flat_thresh_ratio` in
piece_description.classify_side, or add curvature-based corner refinement.

Four real-data issues have been fixed (see segmentation.py):

- **Saturated red printed text/graphics** were being thresholded as
  background over large regions (not just a small edge notch) because
  red converts to a low grayscale value (the standard luminance formula
  under-weights red), putting it on the "background" side of a plain
  grayscale Otsu threshold even though its actual _colour_ is clearly
  different from the (near-black) tray. Fixed with
  `foreground_mask_color`: thresholds each pixel's colour distance from
  an estimated background colour (sampled from the image border) instead
  of raw grayscale brightness, so hue differences are preserved even when
  luminance isn't. This is now the pipeline's default segmentation mask.
- **Dark printed text/graphics near a piece's edge** were being
  thresholded as background, biting small notches out of the piece
  silhouette. Fixed with `morphological_close` + `fill_holes`, applied
  per already-isolated piece crop (never on the whole scene, to avoid
  ever bridging two separate nearby pieces together).
- **A recurring non-puzzle clutter object** (visually a perforated/mesh
  item) was passing the area/aspect-ratio filter since it happens to be
  piece-sized. Fixed with `filter_by_texture_density`: real pieces have
  much lower internal Canny edge density than that object.
- **Two touching real pieces merging into one connected component**
  (a substantial shared edge, not a thin bridge, so `morphological_open`
  alone can't separate them). Fixed with a from-scratch marker-controlled
  watershed split (`distance_transform` + `find_local_maxima` +
  `watershed_split`, tied together by `split_touching_components`):
  anomalously large components (relative to the median piece area) are
  split at the natural "waist" between two piece bodies, found via
  distance-transform peaks selected by farthest-point sampling (raw peak
  _value_ is not reliable for picking markers here, since two touching
  pieces share connectivity right at their join and one piece's geometry
  can inflate its peak value well above the other's true centre).

## Running tests

```
pip install -r requirements.txt
pytest tests/ -v
```
# MajorTaskCSE480
