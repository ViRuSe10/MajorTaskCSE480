# Milestone 2 — Siamese CNN + GNN

## What's here

- `dataset.py` — turns the full labeled dataset + one manually-solved
  reference layout into labeled side-pair training data, automatically,
  for every image in train/valid/test. See its module docstring for how
  ground truth is propagated from one solve to 4000+ photos.
- `augmentation.py` — rotation + illumination/contrast/colour/noise
  jitter (per the assignment spec), applied to the train split only.
- `siamese_model.py` — two-tower 1D-CNN comparing two sides in isolation.
- `gnn_model.py` — message-passing GNN giving each side access to the
  other 3 sides of its own piece before comparison (genuinely different
  architecture/paradigm from the Siamese net, not just a hyperparameter
  change).
- `train.py` — shared training loop for both models (BCE loss, Adam,
  best-checkpoint-on-val-loss, saved loss/accuracy curve PNGs for the report).
- `evaluate_ml.py` — runs classical (Milestone 1) vs. Siamese vs. GNN
  scoring through the **same** `assembly.assemble()` algorithm
  (`score_fn=` plug-in point added to `src/assembly.py`), so the
  comparison isolates the scoring method, and reports quality/accuracy/
  timing for all three.
- `tests/` — unit tests for `augmentation.py` (the alignment/labeling
  logic in `dataset.py` itself is validated with small synthetic test
  images in-chat rather than as a committed test file, since it needs
  real dataset images to run meaningfully).

None of this has been run against the real dataset in this sandbox (no
PyTorch, no internet, and only a handful of sample images were available
here) — the alignment/labeling logic in `dataset.py` was validated with
small synthetic test images instead (see chat). **Run everything below
locally, where you have the full dataset and can `pip install torch`.**

## Given your deadline — do this, in this order

1. **Ground truth is ready.** `detection/ground_truth/manual_solve.json` is
   built from your solved-puzzle data (7x5 grid, picture numbers 1-35).

2. **Install requirements** on your laptop:
   ```
   pip install -r ml/requirements.txt
   ```

3. **Build the dataset — start small, not with all 4000+ images.**
   Use `--cut_dir` (your individually-isolated, alpha-masked piece photos)
   for canonical fingerprints — this is more reliable than a single
   shared reference photo:
   ```
   python -m ml.dataset \
     --dataset_root /path/to/dataset \
     --manual_solve detection/ground_truth/manual_solve.json \
     --cut_dir "/path/to/Images Here/Cut" \
     --output_dir ml/pairs_dataset \
     --max_images 300
   ```
   This prints how many positive/negative samples were generated per
   split — sanity-check that positives > 0 before moving on. Increase
   `--max_images` (or drop the flag entirely) later if there's time.

   The images/labels folder names are auto-detected (`Images`/`Label`,
   `images`/`labels`, or `images`/`obj_det` are all accepted), so this
   works directly against either your own zipped dataset or the
   Kaggle/Roboflow export layout.

   **Data augmentation** (rotation + illumination/contrast/colour jitter,
   per the assignment spec) is applied automatically to the **train**
   split only (valid/test stay as real, unaugmented data, since they
   exist to measure genuine performance). Each training image
   contributes `--augmentations_per_image` extra passes (default 2) on
   top of the original — see `ml/augmentation.py` for exactly what's
   applied and why it's safe with respect to the ground-truth labels.
   Disable with `--no_augment` if you want a faster/smaller run first.

4. **Train both models** — given you don't know how long training will
   take, start with few epochs; the best checkpoint (by validation loss)
   is saved every time it improves, so even an interrupted run leaves you
   with a usable model:
   ```
   python -m ml.train --model siamese --data_dir ml/pairs_dataset --epochs 8
   python -m ml.train --model gnn      --data_dir ml/pairs_dataset --epochs 8
   ```
   Each saves `ml/checkpoints/<model>_best.pt` and a training-curves PNG
   (put both in your report). If time allows afterward, re-run with more
   epochs/images — training always resumes from a full dataset rebuild,
   there's no incremental-resume, so treat each run as a fresh attempt
   using whatever `--max_images`/`--epochs` budget you have left.

5. **Compare all three methods** on a handful of test images:
   ```
   python -m ml.evaluate_ml \
     --dataset_root /path/to/dataset \
     --manual_solve detection/ground_truth/manual_solve.json \
     --reference_image detection/input/scattered_pieces_01.jpg \
     --reference_label detection/ground_truth/scattered_pieces_01_labels.txt \
     --siamese_ckpt ml/checkpoints/siamese_best.pt \
     --gnn_ckpt ml/checkpoints/gnn_best.pt \
     --num_test_images 10
   ```
   Prints a summary table and saves
   `results/evaluation_results/milestone2_comparison.csv` — this is your
   Milestone 2 report's comparison table (matching/neighbour accuracy,
   position/orientation accuracy, quality score, timing).

## If time runs out before the manual solve is done

`dataset.py` requires the manual-solve ground truth to generate correct
labels — there isn't a good way around that for *correct* supervision.
If you're truly out of time, tell me and I'll add a lower-quality
fallback (pseudo-labels from Milestone 1's own classical mutual-best-match
scores instead of true ground truth) so you at least have *something*
trained and running end-to-end to submit, clearly caveated in the report
as weak/self-supervised labeling rather than true ground truth.
