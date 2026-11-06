"""A 2-layer GCN for node classification (Planetoid-style)."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features: int = 1433, hidden: int = 64,
                 num_classes: int = 7, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout_p = dropout  # note: name is not `dropout` (module has a .dropout method)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # logits
