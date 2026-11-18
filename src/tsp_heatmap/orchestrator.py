"""Orchestrator for the TSP heatmap experiment — the only script you run.

Run from the repo root:
    conda activate nor && python src/tsp_heatmap/orchestrator.py \
        [--train-n 1000 --val-n 1000 --test-n 1000] [--n 20] [--k 8] \
        [--hidden 64] [--epochs 50] [--batch-size 64] [--lr 1e-3] \
        [--weight-decay 0.0] [--seed 42] [--device auto]

Pipeline: build the 3 splits (gen + LKH-3 labeling + directed k-NN PyG graphs,
idempotent — re-runs reuse the persisted tours), train the GNN edge classifier
with best-val top-20 F1 checkpointing, then load the checkpoint and evaluate
the test split exactly once. Logs to results/tsp_heatmap/log.txt (mirrored to
the terminal), draws the training curve to results/tsp_heatmap/training_curve.png
(always overwritten on a new training run), and writes
results/tsp_heatmap/results.md.
"""

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

# Repo structure: this file lives in src/tsp_heatmap/. The script dir
# (src/tsp_heatmap/) is auto-added to sys.path by Python, but gen/ and
# utils/ live in src/ — add that too (same pattern as src/cora/train.py).
SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_geometric  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # file output only, never opens a window
import matplotlib.pyplot as plt  # noqa: E402

import dataset  # noqa: E402
import evaluate  # noqa: E402
import train as train_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "results" / "tsp_heatmap"
RESULTS_MD = LOG_DIR / "results.md"
PLOT_PATH = LOG_DIR / "training_curve.png"
TOPK = 20
THRESHOLD = 0.5

HARDWARE = ("Ryzen 7 9700X", "RTX 5060 Ti")


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def fmt_summary(d):
    return (f"avg {d['avg']:.4f} ± {d['std']:.4f} | Q1 {d['q1']:.4f} | "
            f"median {d['median']:.4f} | Q3 {d['q3']:.4f}")


def plot_training_curve(history, cfg, env_line):
    """Per-epoch train loss, val loss (left axis) and val top-k F1 (right axis)
    vs epoch, saved to PLOT_PATH (overwritten every run). File-based only
    (matplotlib Agg backend set at import); no window is opened."""
    epochs = np.array([h["epoch"] for h in history], dtype=np.int64)
    train_losses = np.array([h["train_loss"] for h in history], dtype=np.float64)
    val_losses = np.array([h["val_loss"] for h in history], dtype=np.float64)
    val_topk_f1 = np.array([h["val_topk_f1"] for h in history], dtype=np.float64)
    loss_colors = {"train": "#1f77b4", "val": "#d62728"}
    topk_color = "#2ca02c"

    fig, ax1 = plt.subplots(figsize=(9, 5))
    for name, values in (("train", train_losses), ("val", val_losses)):
        ax1.plot(epochs, values, color=loss_colors[name],
                 label=f"{name} loss", lw=1.2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_xlim(1, max(len(epochs), 1))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_topk_f1, color=topk_color,
             label=f"val top-{cfg['topk']} F1",
             lw=1.2, ls="--", alpha=0.9)
    ax2.set_ylabel(f"val top-{cfg['topk']} F1")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    title = (f"TSP heatmap ({cfg['device']}, hidden={cfg['hidden']}, "
             f"k-NN k={cfg['k']}, bs={cfg['batch_size']})\n"
             f"{env_line} | split {cfg['train_n']}/{cfg['val_n']}/{cfg['test_n']}, "
             f"n={cfg['n']}")
    ax1.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)


def write_results_md(results, cfg, env_line):
    """results/tsp_heatmap/results.md, mirroring results/*/result.md."""
    s, pi = results["test"], results["per_instance_test"]
    lkh = results["meta"]["lkh"]
    tp = pi["threshold"]["precision"]
    tr = pi["threshold"]["recall"]
    tf = pi["threshold"]["f1"]
    tt = pi["topk"]["precision"]
    tt_r = pi["topk"]["recall"]
    tt_f = pi["topk"]["f1"]
    covs = results["meta"]["coverage"]["test"]
    train_times = (results["train_time_s"] - results["infer_val_total_s"]) / cfg["epochs"]

    lines = [
        "# TSP heatmap GNN results (random ATSP, n = %d, k-NN k = %d)" % (cfg["n"], cfg["k"]),
        "",
        "Supervised edge classifier (GNN, hidden %d) for the NC20/NC10 " % cfg["hidden"]
        + "'heatmap' idea: for each random asymmetric TSP instance (LKH-3 tour as label), "
          "one PyG graph per instance whose edges are the directed k nearest out-neighbors "
          f"(k = {cfg['k']}, ~{cfg['n'] * cfg['k']} candidate edges) and whose node features "
          "are the distance-matrix rows D[i, :]/100. The model classifies each candidate "
          "directed edge (i, j) as in-tour or not (positive rate %d/%d = %s)."
          % (cfg["n"], cfg["n"] * cfg["k"], f"{cfg['n'] / (cfg['n'] * cfg['k']):.1%}"),
        "Instances: %d train / %d val / %d test, one global index i with seed = i "
          "(no shared seeds). Model selection on val top-%d F1 (fixed budget; the "
          "@%s threshold F1 is degenerate here — see Interpretation in "
          "markdown/tsp_heatmap_instructions.md); test evaluated once on the "
          "best-val checkpoint."
          % (cfg["train_n"], cfg["val_n"], cfg["test_n"], cfg["topk"], THRESHOLD),
        f"Environment: {env_line}.",
        "",
        f"## Test metrics (n = {s['n']} instances)",
        "",
        "| metric | avg ± std | Q1 | median | Q3 |",
        "|--------|-----------|----|--------|----|",
        f"| precision (p ≥ {THRESHOLD}) | {fmt_summary(s['threshold']['precision'])} |",
        f"| recall (p ≥ {THRESHOLD}) | {fmt_summary(s['threshold']['recall'])} |",
        f"| F1 (p ≥ {THRESHOLD}) | {fmt_summary(s['threshold']['f1'])} |",
        f"| precision (top-{TOPK}) | {fmt_summary(s['topk']['precision'])} |",
        f"| recall (top-{TOPK}) | {fmt_summary(s['topk']['recall'])} |",
        f"| F1 (top-{TOPK}, fixed budget) | {fmt_summary(s['topk']['f1'])} |",
        f"| k-NN tour-edge coverage | {fmt_summary(s['coverage'])} |",
        "",
        f"Per-instance test: min precision {tp.min():.4f} / min recall {tr.min():.4f} / "
        f"min F1 {tf.min():.4f} / min top-{TOPK} precision {tt.min():.4f} / "
        f"min top-{TOPK} recall {tt_r.min():.4f} / min top-{TOPK} F1 {tt_f.min():.4f} / "
        f"coverage range "
        f"[{min(covs):.4f}, {max(covs):.4f}].",
        "",
        "## Training runtimes",
        "",
        "| quantity | value |",
        "|----------|-------|",
        f"| LKH-3 labeling | {lkh['run']} run, {lkh['reused']} reused; "
        f"{lkh['wall_total_s'] / max(lkh['run'], 1):.3f} s/instance averaged over runs |",
        f"| dataset build (gen + label + k-NN) | {results['meta']['build_time_s']:.1f} s |",
        f"| training | {results['train_time_s']:.1f} s over {cfg['epochs']} epochs, "
        f"avg {train_times:.3f} s/epoch |",
        f"| per-epoch val inference (over {cfg['epochs']} epochs, incl.) | "
        f"{results['infer_val_total_s']:.1f} s |",
        "",
        "## Test inference runtime",
        "",
        "| quantity | value |",
        "|----------|-------|",
        f"| test inference | {results['infer_test_s']:.1f} s over {s['n']} instances, "
        f"avg {results['infer_test_s'] / s['n']:.4f} s/instance |",
        "",
        "## Configuration",
        "",
        "| key | value |",
        "|-----|-------|",
    ]
    for key in ("n", "k", "train_n", "val_n", "test_n", "hidden", "dropout", "epochs",
                "batch_size", "lr", "weight_decay", "seed", "device", "topk"):
        lines.append("| %s | %s |" % (key, cfg[key]))
    lines += [
        "",
        "## Hardware specification",
        f"CPU: {HARDWARE[0]}",
        f"GPU: {HARDWARE[1]}",
        "",
    ]
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-n", type=int, default=1000)
    parser.add_argument("--val-n", type=int, default=1000)
    parser.add_argument("--test-n", type=int, default=1000)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    setup_logging(LOG_DIR)
    log = logging.getLogger("tsp_heatmap.orchestrator")

    cfg = {
        "n": args.n, "k": args.k,
        "train_n": args.train_n, "val_n": args.val_n, "test_n": args.test_n,
        "hidden": args.hidden, "dropout": 0.1, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "seed": args.seed, "topk": TOPK,
    }
    device = pick_device(args.device)
    cfg["device"] = str(device)
    env_line = (f"Python {sys.version.split()[0]}, torch {torch.__version__}, "
                f"PyG {torch_geometric.__version__}, NumPy {np.__version__}")

    log.info("run started | %s | device=%s | %s",
             datetime.datetime.now().isoformat(timespec="seconds"), device, env_line)
    for key, val in cfg.items():
        log.info("cfg %-13s = %s", key, val)

    t0 = time.perf_counter()
    datasets, meta = dataset.get_dataloaders(cfg)
    log.info("dataset ready in %.1f s", time.perf_counter() - t0)

    tr = train_mod.train(datasets, meta, cfg, device)

    # Load the best-val checkpoint and evaluate the test split exactly once.
    model, ckpt = train_mod.load_best_checkpoint(device)
    log.info("loaded checkpoint: epoch %d, val top-%d F1 %.4f, val F1@%.2f %.4f "
             "(config seed=%s)",
             ckpt["epoch"], ckpt["config"]["topk"], ckpt.get("val_topk_f1", float("nan")),
             THRESHOLD, ckpt.get("val_f1", float("nan")), ckpt["config"]["seed"])
    t, pi, infer_test_s = evaluate.evaluate_split(
        model, datasets, meta, "test", device, topk=cfg["topk"],
        loss_fn=torch.nn.BCEWithLogitsLoss())
    # Keep the numpy arrays (write_results_md mins them); test is scored exactly
    # once, so these are the final test-split per-instance values.
    per_instance_test = {
        "threshold": {m: pi["threshold"][m] for m in ("precision", "recall", "f1")},
        "topk": {m: pi["topk"][m] for m in ("precision", "recall", "f1")},
    }
    results = {
        "test": t, "per_instance_test": per_instance_test, "meta": meta,
        "train_time_s": tr["train_time_s"], "infer_val_total_s": tr["infer_val_total_s"],
        "infer_test_s": infer_test_s, "best_val_topk_f1": tr["best_val_topk_f1"],
        "best_val_f1": tr["best_val_f1"], "best_epoch": tr["best_epoch"],
        "history": tr["history"],
    }
    log.info("test loss: %.4f", t["loss"])
    log.info("test: precision@%.2f: %s", THRESHOLD,
             fmt_summary(t["threshold"]["precision"]))
    log.info("test: recall@%.2f: %s", THRESHOLD, fmt_summary(t["threshold"]["recall"]))
    log.info("test: F1@%.2f: %s", THRESHOLD, fmt_summary(t["threshold"]["f1"]))
    log.info("test: precision top-%d: %s", cfg["topk"], fmt_summary(t["topk"]["precision"]))
    log.info("test: recall top-%d: %s", cfg["topk"], fmt_summary(t["topk"]["recall"]))
    log.info("test: F1 top-%d: %s", cfg["topk"], fmt_summary(t["topk"]["f1"]))
    log.info("test: k-NN coverage: %s", fmt_summary(t["coverage"]))
    log.info("test inference: %.1f s for %d instances (avg %.4f s/instance)",
             infer_test_s, t["n"], infer_test_s / t["n"])
    log.info("total wall: %.1f s", time.perf_counter() - t0)

    write_results_md(results, cfg, env_line)
    log.info("results written to %s", RESULTS_MD)

    plot_training_curve(results["history"], cfg, env_line)
    log.info("training curve written to %s", PLOT_PATH)


if __name__ == "__main__":
    main()
