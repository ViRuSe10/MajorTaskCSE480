"""
ml/siamese_model.py
=====================
Siamese Convolutional Neural Network for puzzle-piece side matching.

Input representation
---------------------
Each side is represented as the same fixed-length vector used elsewhere
in this project (ml/dataset.py:side_feature_vector): the concatenation of
its resampled shape-deviation profile (num_samples floats) and its
resampled RGB colour-strip signature (num_samples*3 floats). This reuses
Milestone 1's geometric+photometric side representation directly rather
than re-deriving features from raw pixels, and is a genuine 1D sequence
(profile magnitude / colour value as a function of position along the
side) -- exactly the kind of signal 1D convolution is suited to.

Architecture
------------
Two-tower (weight-shared) 1D-CNN encoder: each side's feature vector goes
through several Conv1d + BatchNorm + ReLU blocks, global-average-pooled
into a fixed-size embedding. The two embeddings are combined (absolute
difference, elementwise product, and sum -- a richer comparison than raw
distance) and passed through a small MLP head producing a binary
neighbour logit.

This is a genuinely different architecture family from the GNN
(ml/gnn_model.py: message passing over a piece graph), satisfying the
assignment's "two fundamentally different models" requirement.
"""

import torch
import torch.nn as nn


class SiameseEncoder(nn.Module):
    """Shared-weight 1D-CNN tower mapping one side's feature vector to an embedding."""

    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        # x: (B, L) -> (B, 1, L)
        x = x.unsqueeze(1)
        x = self.net(x)          # (B, 128, 1)
        x = x.squeeze(-1)        # (B, 128)
        return self.proj(x)      # (B, embed_dim)


class SiameseEdgeNet(nn.Module):
    """
    Full Siamese network: shared SiameseEncoder + comparison head producing
    a single neighbour-probability logit for a pair of side feature vectors.
    """

    def __init__(self, feature_len, embed_dim=64):
        super().__init__()
        self.encoder = SiameseEncoder(in_channels=1, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.feature_len = feature_len

    def forward(self, feat_a, feat_b):
        emb_a = self.encoder(feat_a)
        emb_b = self.encoder(feat_b)
        diff = torch.abs(emb_a - emb_b)
        prod = emb_a * emb_b
        combined = torch.cat([diff, prod, emb_a + emb_b], dim=-1)
        logit = self.head(combined).squeeze(-1)
        return logit  # apply sigmoid outside (use BCEWithLogitsLoss during training)

    def compatibility_score(self, feat_a, feat_b):
        """
        Convenience for inference: returns a "lower is better" mismatch
        score (1 - sigmoid(logit)) in [0, 1], matching the convention used
        by edge_matching.match_score / assembly.py's score_fn (lower =
        better match), so this model can be plugged straight into
        assembly.assemble(score_fn=...).
        """
        with torch.no_grad():
            logit = self.forward(feat_a, feat_b)
            prob_neighbor = torch.sigmoid(logit)
            return 1.0 - prob_neighbor
