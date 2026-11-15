"""GNN edge classifier for the TSP heatmap.

A 2-layer GCN node encoder over the directed k-NN graph (node i has an edge to
its k nearest out-neighbors by D[i, :]), plus an edge head on the pair
(h_i, h_j) producing one logit per candidate edge: "is (i, j) in the tour?".

forward(x, edge_index) -> one logit per row of edge_index.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TSPHeatmapGNN(torch.nn.Module):
    def __init__(self, num_features: int = 20, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.edge_head = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.dropout_p = dropout  # named like src/cora/model.py (avoids shadowing issues, keeps the repo convention)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, self.dropout_p, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.dropout(h, self.dropout_p, training=self.training)
        src, dst = edge_index[0], edge_index[1]
        return self.edge_head(torch.cat([h[src], h[dst]], dim=1)).squeeze(-1)
