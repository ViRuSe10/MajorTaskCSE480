"""
ml/gnn_model.py
=================
Graph Neural Network for puzzle-piece side matching -- architecturally
distinct from the Siamese CNN (ml/siamese_model.py), per the assignment's
requirement for two fundamentally different models.

Graph structure
----------------
Where the Siamese network compares two side feature vectors in complete
isolation, the GNN gives each queried side access to *graph context*: the
other 3 sides of its own piece. A piece's 4 sides are not independent --
their tab/blank/flat pattern and geometry are jointly determined by the
same physical piece, so a side's "true" embedding should be informed by
what the rest of its piece looks like (e.g. a very unusual/large tab is
easier to disambiguate when you also know this piece has two flat sides
and is therefore a border piece). We model each piece as a small 4-node
star graph (every side connected to every other side of the same piece)
and run several rounds of message passing (implemented manually with
scatter-add via index_add_, so no torch_geometric dependency is required)
to produce a graph-refined embedding for the queried side before the
final pairwise comparison.

Input representation
---------------------
Each of the 4 sides is the same feature vector used by the Siamese model
(ml/dataset.py:side_feature_vector): resampled shape profile ++ resampled
colour strip. A piece is therefore a (4, feature_len) tensor; a batch is
(B, 4, feature_len).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    """
    One round of message passing over a fully-connected 4-node graph
    (every side <-> every other side of the same piece). Implemented with
    plain tensor ops (no torch_geometric): for 4 nodes this is small
    enough to just do a dense all-pairs aggregation directly.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, node_states):
        # node_states: (B, 4, H)
        B, N, H = node_states.shape
        # build all-pairs (i, j) messages: expand to (B, N, N, H) each way
        src = node_states.unsqueeze(2).expand(B, N, N, H)   # node i repeated across j
        dst = node_states.unsqueeze(1).expand(B, N, N, H)   # node j repeated across i
        pair_feat = torch.cat([src, dst], dim=-1)           # (B, N, N, 2H)
        messages = self.message_mlp(pair_feat)              # (B, N, N, H)

        # zero out self-loops (i == j), then aggregate incoming messages per node j
        eye = torch.eye(N, device=node_states.device, dtype=torch.bool)
        messages = messages.masked_fill(eye.view(1, N, N, 1), 0.0)
        aggregated = messages.sum(dim=1)  # sum over source i -> (B, N, H)
        aggregated = aggregated / max(N - 1, 1)

        updated = self.update_mlp(torch.cat([node_states, aggregated], dim=-1))
        return updated  # (B, N, H)


class SideGNNEncoder(nn.Module):
    """Encodes one piece's 4 sides into 4 graph-refined embeddings."""

    def __init__(self, feature_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(feature_len, hidden_dim)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, sides):
        # sides: (B, 4, feature_len)
        x = F.relu(self.input_proj(sides))  # (B, 4, H)
        for layer in self.layers:
            x = x + layer(x)  # residual connection
        return x  # (B, 4, H) graph-refined per-side embeddings


class GNNEdgeNet(nn.Module):
    """
    Full GNN model: SideGNNEncoder (shared weights, applied to both
    pieces) + pairwise comparison head, producing a binary neighbour logit
    for the queried (si, sj) side pair -- now informed by each piece's
    full local graph context, not just the isolated pair.
    """

    def __init__(self, feature_len, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = SideGNNEncoder(feature_len, hidden_dim=hidden_dim, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, sides_a, si, sides_b, sj):
        """
        sides_a, sides_b : (B, 4, feature_len) -- all 4 sides of each piece.
        si, sj            : (B,) long -- which side index is the query in
                             piece A / piece B.
        """
        emb_a_all = self.encoder(sides_a)  # (B, 4, H)
        emb_b_all = self.encoder(sides_b)  # (B, 4, H)

        batch_idx = torch.arange(sides_a.shape[0], device=sides_a.device)
        emb_a = emb_a_all[batch_idx, si]   # (B, H) -- the queried side's graph-refined embedding
        emb_b = emb_b_all[batch_idx, sj]   # (B, H)

        diff = torch.abs(emb_a - emb_b)
        prod = emb_a * emb_b
        combined = torch.cat([diff, prod, emb_a + emb_b], dim=-1)
        logit = self.head(combined).squeeze(-1)
        return logit

    def compatibility_score(self, sides_a, si, sides_b, sj):
        """Same 'lower is better' convention as SiameseEdgeNet.compatibility_score."""
        with torch.no_grad():
            logit = self.forward(sides_a, si, sides_b, sj)
            prob_neighbor = torch.sigmoid(logit)
            return 1.0 - prob_neighbor
