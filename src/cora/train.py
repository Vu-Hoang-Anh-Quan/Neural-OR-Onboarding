"""Train a GCN on Cora, select the best model on validation accuracy,
then evaluate the best (test-accuracy) model on the test split.

Run from the repo root:
    python src/cora/train.py [--epochs 200] [--lr 0.01] [--seed 42] [--device auto]

Outputs:
    results/cora/log.txt                        (also mirrored to the terminal)
    src/cora/checkpoints/cora_gcn.pt            (best-val model + metadata)
"""

import argparse
import logging
import sys
from pathlib import Path

# Repo structure: this file lives in src/cora/. The script dir (src/cora/) is
# auto-added to sys.path, but utils/ lives in src/ — add that too.
SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
import torch_geometric

import dataset as dataset_mod
import model as model_mod
from utils.logging_utils import setup_logging

LOG_DIR = Path(__file__).resolve().parents[2] / "results" / "cora"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "cora_gcn.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def evaluate(model, loader, device, mask_name):
    """Accuracy over the nodes selected by ``mask_name`` (train/val/test_mask)."""
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            mask = getattr(batch, mask_name)
            correct += (pred[mask] == batch.y[mask]).sum().item()
            total += mask.sum().item()
    return correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    setup_logging(LOG_DIR)
    log = logging.getLogger("cora")

    log.info("Cora node classification with GCN")
    log.info("python %s | torch %s | torch_geometric %s | device %s",
             sys.version.split()[0], torch.__version__,
             torch_geometric.__version__, device)
    log.info("config: epochs=%d lr=%.4f weight_decay=%.0e seed=%d",
             args.epochs, args.lr, args.weight_decay, args.seed)

    train_loader, val_loader, test_loader, data = dataset_mod.get_dataloaders()
    log.info("data: nodes=%d edges=%d features=%d classes=%d "
             "split(train/val/test)=%d/%d/%d",
             data.num_nodes, int(data.edge_index.shape[1]),
             data.x.shape[1], 7,
             int(data.train_mask.sum()), int(data.val_mask.sum()),
             int(data.test_mask.sum()))

    model = model_mod.GCN(num_features=data.x.shape[1], hidden=64,
                          num_classes=7, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        val_acc = evaluate(model, val_loader, device, "val_mask")
        log.info("Epoch %3d | train loss %.4f | val acc %.4f",
                 epoch, epoch_loss, val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc}, CHECKPOINT_PATH)

    # Inference: load the best (val-selected) checkpoint and evaluate on the
    # held-out test split exactly once.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_acc = evaluate(model, test_loader, device, "test_mask")
    log.info("best val acc %.4f @ epoch %d | checkpoint %s",
             best_val_acc, best_epoch, CHECKPOINT_PATH)
    log.info("FINAL TEST ACCURACY: %.4f", test_acc)


if __name__ == "__main__":
    main()
