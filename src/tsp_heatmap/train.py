"""Train the TSP heatmap GNN.

BCEWithLogits on the candidate edges. Best-val model selection uses the mean
per-instance val top-20 F1: unlike the probability-threshold F1, the top-20
budget prediction never degenerates to an empty set (it always predicts
exactly k edges for a non-empty pool), so its per-instance F1 is
well-defined for every instance in every epoch. The @0.5 threshold F1 is
degenerate here (sigmoid probabilities sit near the 12.5% class prior, almost
no edge crosses 0.5, per-instance P/R/F1 are pinned to the 0.0 fallback
every epoch), so selecting on it with a strict `>` comparison would freeze
the least-trained epoch — epoch 1 — as the "best" checkpoint whenever the
metric stays ~0 (exactly what happened when val F1 @0.5 stayed 0.0000 for
all epochs of a 50-epoch run, overwriting a good later checkpoint). The
threshold F1 is still logged every epoch for reporting, but no longer drives
checkpoint selection.

The checkpoint is saved to src/tsp_heatmap/checkpoints/tsp_heatmap.pt (model
state + metadata, mirroring the src/cora/train.py checkpoint convention:
epoch, the val metrics at that epoch, full training config). Final test
inference loads that checkpoint, so the test split is scored exactly once by
the orchestrator.

Called by orchestrator.py (the only setup_logging caller), so this module only
uses its named logger.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import evaluate as evaluate_mod
import model as model_mod

log = logging.getLogger("tsp_heatmap.train")

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "tsp_heatmap.pt"


def load_best_checkpoint(device):
    """Load the best-val checkpoint and the config that trained it."""
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = model_mod.TSPHeatmapGNN(
        num_features=ckpt["config"]["n"], hidden=ckpt["config"]["hidden"],
        dropout=ckpt["config"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def train(datasets, meta, cfg, device):
    """Full training loop on datasets as built by dataset.get_dataloaders.

    Returns a dict with: best_val_topk_f1 (selection metric), best_val_f1
    (@0.5 threshold F1 at the best epoch, for reporting), best_epoch,
    final_train_loss, train_time_s, infer_val_total_s, checkpoint path, and
    history (per-epoch train loss, val loss, val top-k F1 — the plot input).
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model = model_mod.TSPHeatmapGNN(
        num_features=cfg["n"], hidden=cfg["hidden"],
        dropout=cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()

    loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"],
                        shuffle=True)

    best_val_topk_f1, best_val_f1, best_epoch = -np.inf, float("nan"), 0
    last_loss = float("nan")
    infer_val_total_s = 0.0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        last_loss = total / n_batches
        val_summary, _, val_infer_s = evaluate_mod.evaluate_split(
            model, datasets, meta, "val", device, topk=cfg["topk"],
            loss_fn=criterion)
        infer_val_total_s += val_infer_s
        val_loss = val_summary["loss"]
        val_f1 = val_summary["threshold"]["f1"]["avg"]
        val_topk_f1 = val_summary["topk"]["f1"]["avg"]
        log.info("epoch %3d | train loss %.4f | val loss %.4f | val F1@0.5 %.4f | "
                 "val top-%d F1 %.4f", epoch, last_loss, val_loss, val_f1,
                 cfg["topk"], val_topk_f1)
        history.append({"epoch": epoch, "train_loss": last_loss,
                        "val_loss": val_loss, "val_topk_f1": val_topk_f1})
        if val_topk_f1 > best_val_topk_f1:
            best_val_topk_f1, best_val_f1, best_epoch = val_topk_f1, val_f1, epoch
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_f1": val_f1,
                        "val_topk_f1": val_topk_f1,
                        "config": dict(cfg)}, CHECKPOINT_PATH)
    train_time_s = time.perf_counter() - t0
    log.info("training done: %d epochs in %.1f s | best val top-%d F1 %.4f "
             "@ epoch %d (val F1@0.5 %.4f)",
             cfg["epochs"], train_time_s, cfg["topk"],
             best_val_topk_f1, best_epoch, best_val_f1)
    log.info("checkpoint: %s", CHECKPOINT_PATH)
    return {
        "best_val_topk_f1": float(best_val_topk_f1),
        "best_val_f1": float(best_val_f1),
        "best_epoch": best_epoch,
        "final_train_loss": float(last_loss),
        "train_time_s": train_time_s,
        "infer_val_total_s": float(infer_val_total_s),
        "history": history,
        "checkpoint": str(CHECKPOINT_PATH),
    }
