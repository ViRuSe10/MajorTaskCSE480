"""
ml/train.py
============
Trains either the Siamese CNN or the GNN on the .npz pair dataset built
by ml/dataset.py, with train/val monitoring, best-checkpoint selection on
validation loss, and a saved loss-curve plot for the report.

Usage
-----
    python -m ml.train --model siamese --data_dir ml/pairs_dataset --epochs 20
    python -m ml.train --model gnn      --data_dir ml/pairs_dataset --epochs 20

Requires PyTorch (pip install torch). Not runnable in this sandbox (no
GPU/torch here) -- run locally with the full dataset.
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml.siamese_model import SiameseEdgeNet
from ml.gnn_model import GNNEdgeNet


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Wraps one split's .npz (sides_a, si, sides_b, sj, labels, rel_rot)."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.sides_a = data["sides_a"].astype(np.float32)  # (N, 4, L)
        self.si = data["si"].astype(np.int64)
        self.sides_b = data["sides_b"].astype(np.float32)
        self.sj = data["sj"].astype(np.int64)
        self.labels = data["labels"].astype(np.float32)
        self.rel_rot = data["rel_rot"].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.sides_a[idx], self.si[idx], self.sides_b[idx], self.sj[idx],
                self.labels[idx], self.rel_rot[idx])


def _feature_len(dataset):
    return dataset.sides_a.shape[-1]


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _forward_batch(model, model_name, batch, device):
    sides_a, si, sides_b, sj, labels, rel_rot = [t.to(device) for t in batch]
    if model_name == "siamese":
        feat_a = sides_a[torch.arange(sides_a.shape[0]), si]
        feat_b = sides_b[torch.arange(sides_b.shape[0]), sj]
        logits = model(feat_a, feat_b)
    else:  # gnn
        logits = model(sides_a, si, sides_b, sj)
    return logits, labels


def run_epoch(model, model_name, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            logits, labels = _forward_batch(model, model_name, batch, device)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_n += labels.size(0)

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def train(model_name, data_dir, epochs=20, batch_size=64, lr=1e-3, output_dir="ml/checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    train_ds = PairDataset(os.path.join(data_dir, "train_pairs.npz"))
    val_ds = PairDataset(os.path.join(data_dir, "valid_pairs.npz"))
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    feature_len = _feature_len(train_ds)
    if model_name == "siamese":
        model = SiameseEdgeNet(feature_len=feature_len).to(device)
    elif model_name == "gnn":
        model = GNNEdgeNet(feature_len=feature_len).to(device)
    else:
        raise ValueError("model_name must be 'siamese' or 'gnn'")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model_name}, parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, f"{model_name}_best.pt")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, model_name, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, model_name, val_loader, device, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "feature_len": feature_len,
                        "model_name": model_name, "epoch": epoch, "val_loss": val_loss},
                       best_path)
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")

    _save_loss_plot(history, model_name, output_dir)
    print(f"Best checkpoint: {best_path} (val_loss={best_val_loss:.4f})")
    return history, best_path


def _save_loss_plot(history, model_name, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping loss plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{model_name}: loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{model_name}: accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    fig.savefig(out_path, dpi=120)
    print(f"saved training curves to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese CNN or GNN for puzzle edge matching.")
    parser.add_argument("--model", choices=["siamese", "gnn"], required=True)
    parser.add_argument("--data_dir", default="ml/pairs_dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="ml/checkpoints")
    args = parser.parse_args()

    train(args.model, args.data_dir, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, output_dir=args.output_dir)
