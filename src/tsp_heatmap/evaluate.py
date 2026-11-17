"""Evaluation for the TSP heatmap: per-instance precision/recall/F1 of the edge
heatmap over the directed k-NN candidate pool, at threshold 0.5 (primary) and at
a fixed top-20 predicted budget per instance (secondary), plus the k-NN tour-edge
coverage (the per-instance recall upper bound).

Per-instance metrics are summarized as avg ± std, Q1, Q3, median (numpy
percentiles, linear interpolation).

evaluate_split() returns (summary, per_instance, infer_time_s); the caller
logs the summary (train.py / orchestrator.py). Pass loss_fn (e.g.
torch.nn.BCEWithLogitsLoss()) to also get summary["loss"]: the mean loss over
ALL candidate edges of the split, computed from the raw logits (one reduction,
not the per-instance metric average).
"""

import logging
import time

import numpy as np
import torch

log = logging.getLogger("tsp_heatmap.evaluate")


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "avg": float(values.mean()),
        "std": float(values.std()),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "median": float(np.median(values)),
    }


def per_instance_pr(y, p, threshold=0.5):
    """(precision, recall, f1) for one instance at a probability threshold.

    y, p: 1-D numpy arrays over the instance's candidate pool.
    """
    pos = (y == 1)
    pred = p >= threshold
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    n_y = int(np.sum(pos))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / n_y if n_y else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def pr_at_topk(y, p, topk):
    """(precision, recall, f1) for the top-`topk` edges by predicted probability
    on ONE instance (ties broken toward y = 1). y, p: 1-D numpy arrays over
    the instance's candidate pool.

    Unlike the probability-threshold metrics, the top-k prediction is never
    empty (it always predicts exactly k edges, as long as k > 0), so its
    precision/recall/F1 are well-defined for every instance — this is what
    makes it a usable model-selection criterion (see train.py)."""
    k = min(int(topk), y.size)
    if k == 0:
        return 0.0, 0.0, 0.0
    order = np.lexsort((-y, -p))          # primary: p desc; tie: y desc
    take = order[:k]
    hits = int(y[take].sum())
    prec = hits / k
    n_pos = int(y.sum())
    rec = hits / n_pos if n_pos else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(prec), float(rec), float(f1)


def evaluate_split(model, datasets, meta, split, device, topk=20, loss_fn=None):
    """Evaluate ONE split in a single inference pass.

    Args: model (module taking (x, edge_index) -> logits), datasets as returned
    by dataset.get_dataloaders ({split: HeatmapDataset}), meta as returned by
    dataset.get_dataloaders (for the k-NN coverage), split ("train"/"val"/"test"),
    loss_fn (optional) — callable(logits, targets); when given, summary["loss"]
    is the mean loss over all candidate edges of the split.

    Returns (summary, per_instance, infer_time_s):
      summary = {
        "n": int,
        "loss": mean float (only when loss_fn is given),
        "coverage": summarize of the k-NN tour-edge coverage,
        "threshold": {"precision": d, "recall": d, "f1": d},
        "topk": {"k": topk, "precision": d, "recall": d, "f1": d},
      }  (each d is a summarize dict)
      per_instance = {
        "threshold": {"precision": [..], "recall": [..], "f1": [..]},
        "topk": {"precision": [..], "recall": [..], "f1": [..]},
      }
      infer_time_s: wall time of the inference pass over the split.
    """
    ds = datasets[split]
    model.eval()
    n = len(ds)
    t0 = time.perf_counter()
    with torch.no_grad():
        ys, ps, losses = [], [], []
        for i in range(n):
            g = ds[i].to(device)
            out = model(g.x, g.edge_index)
            ps.append(torch.sigmoid(out).cpu().numpy())
            ys.append(g.y.cpu().numpy())
            if loss_fn is not None:
                losses.append(loss_fn(out, g.y))
    infer_time_s = time.perf_counter() - t0

    prec_thr, rec_thr, f1_thr = [], [], []
    prec_top, rec_top, f1_top = [], [], []
    for i in range(n):
        pt, rt, ft = per_instance_pr(ys[i], ps[i], threshold=0.5)
        prec_thr.append(pt)
        rec_thr.append(rt)
        f1_thr.append(ft)
        pt_top, rt_top, ft_top = pr_at_topk(ys[i], ps[i], topk)
        prec_top.append(pt_top)
        rec_top.append(rt_top)
        f1_top.append(ft_top)

    summary = {
        "n": n,
        "coverage": summarize(meta["coverage"][split]),
        "threshold": {
            "precision": summarize(prec_thr),
            "recall": summarize(rec_thr),
            "f1": summarize(f1_thr),
        },
        "topk": {
            "k": topk,
            "precision": summarize(prec_top),
            "recall": summarize(rec_top),
            "f1": summarize(f1_top),
        },
    }
    if loss_fn is not None:
        summary["loss"] = float(torch.stack(losses).mean().item())
    per_instance = {
        "threshold": {"precision": np.array(prec_thr),
                      "recall": np.array(rec_thr),
                      "f1": np.array(f1_thr)},
        "topk": {"precision": np.array(prec_top),
                 "recall": np.array(rec_top),
                 "f1": np.array(f1_top)},
    }
    return summary, per_instance, infer_time_s


def format_summary(d, name):
    return (f"{name}: avg {d['avg']:.4f} ± {d['std']:.4f} | Q1 {d['q1']:.4f} | "
            f"median {d['median']:.4f} | Q3 {d['q3']:.4f}")
