"""Load and preprocess the Cora dataset (Planetoid) into DataLoaders.

Cora ships with 1433-dim text features and the standard Kipf split
(140 train / 500 val / 1000 test nodes). First run downloads the raw files
into the root dir (default ``data/cora``); everything else (processed cache)
also lands there.
"""

import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

DEFAULT_ROOT = "data/cora"


def _to_dense_row_normalized(x):
    """Make features a dense float tensor, L2-normalize each row (Planetoid convention)."""
    # torch.sparse.issparse / is_sparse helpers moved across torch versions;
    # use whatever is available.
    if isinstance(x, torch.Tensor) and getattr(x, "is_sparse", False):
        x = x.to_dense()
    x = x.float()
    return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)


def load_cora(root: str = DEFAULT_ROOT):
    """Download (if needed) and return the Cora graph as one Data object."""
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]  # the whole graph is a single Data object
    data.x = _to_dense_row_normalized(data.x)
    return data


def get_dataloaders(root: str = DEFAULT_ROOT):
    """Preprocess Cora and return (train_loader, val_loader, test_loader, data).

    Full-batch training (Planetoid convention): each loader wraps the whole
    graph as a single batch, so any loader yields the full graph once per
    epoch; the train/val/test split is selected via ``train_mask`` /
    ``val_mask`` / ``test_mask`` at train/validate/infer time.
    """
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    data.x = _to_dense_row_normalized(data.x)
    loaders = [DataLoader([data], batch_size=1) for _ in range(3)]
    return loaders[0], loaders[1], loaders[2], data
