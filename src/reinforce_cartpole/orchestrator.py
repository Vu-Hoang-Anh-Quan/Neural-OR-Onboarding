"""Orchestrator for the REINFORCE CartPole experiment — the only script you run.

Run from the repo root:
    conda activate nor && python src/reinforce_cartpole/orchestrator.py \
        [--epochs 100] [--seeds 0 1 2 3 4] [--lr 0.01] [--baseline 0.5] \
        [--device auto]

Policy model: 2 variants x 5 seeds = 10 distinct policies, each trained
independently for `epochs` epochs of `per_epoch` (= 10) fresh env episodes
("instances"). Pipeline: train vanilla REINFORCE (gamma = 1.0, reward-to-go,
truncated episodes kept as-is) for every policy, then write
results/reinforce_cartpole/training.csv (one row per (variant, seed, epoch):
per-epoch return stats and the baseline), returns_curve.png (x = epoch 1..E,
y = mean of the epoch's 10 instance returns, band = mean ± 1 std — one line
per variant, showing the variant's best-seed policy) and results.md
(per-variant best-seed policy: mean ± std at its best epoch, full config,
hardware, device used, versions). Logs to
results/reinforce_cartpole/log.txt (mirrored to the terminal).
"""

import argparse
import csv
import datetime
import logging
import sys
import time
from pathlib import Path

# Repo structure: this file lives in src/reinforce_cartpole/. The script dir
# (src/reinforce_cartpole/) is auto-added to sys.path by Python, but utils/
# lives in src/ — add that too (same pattern as src/tsp_heatmap/orchestrator.py).
SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import gymnasium  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402

matplotlib.use("Agg")  # file output only, never opens a window
import matplotlib.pyplot as plt  # noqa: E402

import train as train_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "results" / "reinforce_cartpole"
CSV_PATH = LOG_DIR / "training.csv"
PLOT_PATH = LOG_DIR / "returns_curve.png"
RESULTS_MD = LOG_DIR / "results.md"

HARDWARE = ("Ryzen 7 9700 X", "RTX 5060 Ti")
VARIANT_COLORS = {"no_baseline": "#1f77b4", "moving_avg": "#d62728"}
GAMMA = 1.0  # fixed by the task invariants — no discount
PER_EPOCH = 10  # env episodes ("instances") sampled per policy per epoch (fixed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def write_csv(data, path):
    """One row per (variant, seed, epoch):
    variant,seed,epoch,return_mean,return_std,baseline — per-epoch return
    stats over the epoch's per_epoch fresh env episodes ("instances");
    baseline = the policy's baseline value after the epoch's update (0 for
    the no_baseline variant)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "seed", "epoch", "return_mean",
                         "return_std", "baseline"])
        for variant in data["variants"]:
            for seed in data["seeds"]:
                res = data["runs"][variant][seed]
                for e in range(data["epochs"]):
                    writer.writerow([
                        variant, seed, e + 1,
                        f"{res['mean'][e]:.2f}", f"{res['std'][e]:.2f}",
                        f"{res['baseline'][e]:.4f}",
                    ])


def plot_returns_curve(data, cfg, env_line):
    """One line per variant, showing the variant's best-seed policy
    (highest best-epoch mean g): x = epoch 1..E, line = mean of the epoch's
    PER_EPOCH instance returns, shaded band = mean ± 1 std (population std,
    ddof=0) over those instances. Saved to PLOT_PATH (overwritten every run).
    File-based only (Agg backend set at import)."""
    epochs = cfg["epochs"]
    x = np.arange(1, epochs + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in cfg["variants"]:
        b = data["best"][variant]
        res = data["runs"][variant][b["seed"]]
        color = VARIANT_COLORS[variant]
        ax.fill_between(x, res["mean"] - res["std"], res["mean"] + res["std"],
                        color=color, alpha=0.2, label=None)
        ax.plot(x, res["mean"], color=color, lw=1.2,
                label=(f"{variant} — best seed {b['seed']}, "
                       f"mean of {cfg['per_epoch']} episodes/epoch"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Return")
    ax.set_xlim(1, max(epochs, 1))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    title = (f"REINFORCE CartPole-v1 — return = episode length (gamma = {GAMMA})\n"
             f"epochs={epochs} x {cfg['per_epoch']} episodes, "
             f"lr={cfg['lr']}, B={cfg['baseline_b']}, "
             f"seeds={cfg['seeds']}, device={cfg['device']}")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def write_results_md(data, cfg, runtimes, env_line):
    """results/reinforce_cartpole/results.md, mirroring results/*/result.md.

    Depicts only the best-seed policy of each variant: that policy's mean ±
    std at its best epoch (the epoch with the highest per-epoch mean return,
    earliest on a tie)."""
    n_seeds = len(cfg["seeds"])
    n_policies = len(cfg["variants"]) * n_seeds
    lines = [
        "# REINFORCE CartPole-v1 results",
        "",
        "Vanilla REINFORCE (on-policy policy gradient, gamma = 1.0, reward-to-go) "
        f"on CartPole-v1, two variants — `no_baseline` and `moving_avg` (moving-average "
        f"baseline updated once per epoch, b <- b*(1-B) + G_epoch*B, B = {cfg['baseline_b']}, "
        f"initial 0.0) — with {n_seeds} seed policies per variant: {n_policies} "
        "distinct "
        "policies, each trained independently for "
        f"{cfg['epochs']} epochs of {cfg['per_epoch']} fresh env episodes ("
        f"\"instances\"). A policy's epoch return = mean of its {cfg['per_epoch']} instance "
        "returns of that epoch (std = population std, ddof = 0). "
        "Model: Linear(4,64) -> ReLU -> Linear(64,2) -> Softmax; "
        f"Adam lr = {cfg['lr']}; one optimizer update per env episode over its "
        "full-episode batch (loss = -sum_t logprob_t * (G_t - baseline), "
        "G_t = sum_{t' >= t} R_{t'}). CartPole-v1 rewards +1.0 per step "
        "(the terminal step included), so each episode return = its length "
        "(<= 500); truncated (500-step) episodes are kept as-is, no bootstrap "
        "or bonus.",
        f"Environment: {env_line}.",
        "",
        f"## Best results per variant (best of the {n_seeds} seed policies)",
        "",
        "| variant | best seed | best-g (mean ± std) | at epoch | final g (mean ± std) |",
        "|---------|-----------|---------------------|----------|----------------------|",
    ]
    for variant in data["variants"]:
        b = data["best"][variant]
        fs = data["policy_stats"][variant][b["seed"]]
        lines.append(
            f"| {variant} | {b['seed']} | {b['g']:.1f} ± {b['g_std']:.1f} "
            f"| {b['epoch']} | {fs['final_g']:.1f} ± {fs['final_g_std']:.1f} |"
        )
    lines += [
        "",
        f"All per-epoch lines of `returns_curve.png` are the mean ± 1 std of the "
        f"{cfg['per_epoch']} instance returns of the best-seed policy of each variant; "
        "'best seed' = the policy with the highest best-epoch mean g; 'best epoch' "
        "= the epoch with the highest per-epoch mean g of that policy.",
        "",
        "## Runtimes",
        "",
        "| quantity | value |",
        "|----------|-------|",
    ]
    for variant in data["variants"]:
        lines.append(
            f"| training, {variant} "
            f"({len(cfg['seeds'])} seed policies x {cfg['epochs']} epochs) "
            f"| {runtimes[variant]:.1f} s |"
        )
    lines += [
        f"| total (all {n_policies} policies) | {runtimes['total']:.1f} s |",
        "",
        "## Configuration",
        "",
        "| key | value |",
        "|-----|-------|",
    ]
    for key in ("epochs", "per_epoch", "lr", "baseline_b", "seeds", "gamma",
                "variants", "device"):
        lines.append(f"| {key} | {cfg[key]} |")
    lines += [
        "",
        "## Environment",
        "",
        env_line + ".",
        "",
        "## Hardware specification",
        f"CPU: {HARDWARE[0]}",
        f"GPU: {HARDWARE[1]}",
        f"Device used (resolved --device={cfg['device_arg']}): {cfg['device']}",
        "",
    ]
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--baseline", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    setup_logging(LOG_DIR)
    log = logging.getLogger("reinforce_cartpole.orchestrator")

    device = pick_device(args.device)
    env_line = ("Python %s, torch %s, gymnasium %s, NumPy %s, matplotlib %s"
                % (sys.version.split()[0], torch.__version__,
                   gymnasium.__version__, np.__version__, matplotlib.__version__))
    cfg = {
        "epochs": args.epochs, "per_epoch": PER_EPOCH,
        "lr": args.lr, "baseline_b": args.baseline,
        "seeds": args.seeds, "gamma": GAMMA,
        "variants": list(train_mod.VARIANTS), "device": str(device),
        "device_arg": args.device,
    }

    log.info("run started | %s | device=%s | %s",
             datetime.datetime.now().isoformat(timespec="seconds"), device,
             env_line)
    for key, val in cfg.items():
        log.info("cfg %-12s = %s", key, val)
    log.info("CUDA available at runtime: %s", torch.cuda.is_available())

    t_total = time.perf_counter()
    data = train_mod.train(variants=tuple(train_mod.VARIANTS), seeds=args.seeds,
                           epochs=args.epochs, per_epoch=PER_EPOCH, lr=args.lr,
                           baseline_b=args.baseline, device=device)
    total_s = time.perf_counter() - t_total

    for variant in data["variants"]:
        b = data["best"][variant]
        log.info("summary %-12s best seed %d: best g %.1f ± %.1f at epoch %d "
                 "| final g %.1f ± %.1f",
                 variant, b["seed"], b["g"], b["g_std"], b["epoch"],
                 data["policy_stats"][variant][b["seed"]]["final_g"],
                 data["policy_stats"][variant][b["seed"]]["final_g_std"])

    runtimes = dict(data["runtimes"])
    runtimes["total"] = total_s

    write_csv(data, CSV_PATH)
    n_rows = len(data["seeds"]) * data["epochs"] * len(data["variants"])
    log.info("CSV written to %s (%d rows)", CSV_PATH, n_rows)
    plot_returns_curve(data, cfg, env_line)
    log.info("returns curve written to %s", PLOT_PATH)
    write_results_md(data, cfg, runtimes, env_line)
    log.info("results written to %s", RESULTS_MD)
    log.info("total wall: %.1f s", total_s)


if __name__ == "__main__":
    main()
