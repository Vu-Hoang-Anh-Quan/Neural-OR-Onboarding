"""Orchestrator for the REINFORCE CartPole experiment — the only script you run.

Run from the repo root:
    conda activate nor && python src/reinforce_cartpole/orchestrator.py \
        [--episodes 1000] [--seeds 0 1 2 3 4] [--lr 0.01] [--baseline 0.5] \
        [--device auto]

Pipeline: train vanilla REINFORCE (gamma = 1.0, reward-to-go, truncated
episodes kept as-is) on CartPole-v1 for two variants — `no_baseline` and
`moving_avg` — over all (variant, seed) runs, then write
results/reinforce_cartpole/training.csv (one row per episode of every run),
returns_curve.png (mean +/- 1 std over the seeds, one line per variant) and
results.md (per-variant best mean return + episode it occurs at, full config,
hardware, device used, versions). Logs to results/reinforce_cartpole/log.txt
(mirrored to the terminal).
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


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def aggregate_by_variant(data):
    """Per variant: returns matrix (episodes, seeds) and mean/std (ddof=0)
    across seeds at each episode index; plus best mean value + episode index
    (1-based, matching the plot x-axis) and final (last-episode) mean."""
    out = {}
    for variant in train_mod.VARIANTS:
        runs = [data["runs"][variant][seed]["returns"] for seed in data["seeds"]]
        matrix = np.stack(runs, axis=1)  # (E, S)
        mean = matrix.mean(axis=1)
        std = matrix.std(axis=1, ddof=0)
        best_idx = int(np.argmax(mean))
        out[variant] = {
            "returns": matrix,
            "mean": mean,
            "std": std,
            "best_mean": float(mean[best_idx]),
            "best_episode": best_idx + 1,
            "final_mean": float(mean[-1]),
        }
    return out


def write_csv(data, path):
    """One row per episode of every (variant, seed) run:
    variant,seed,episode,return,baseline,steps. episode numbered 1..E."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "seed", "episode", "return", "baseline", "steps"])
        for variant in train_mod.VARIANTS:
            for seed in data["seeds"]:
                res = data["runs"][variant][seed]
                for i, ep in enumerate(res["returns"]):
                    # Returns are exact integers here (sum of +1.0 rewards =
                    # episode length, <= 500).
                    writer.writerow([
                        variant, seed, i + 1,
                        f"{ep:.0f}", f"{res['baseline'][i]:.4f}", int(res["steps"][i]),
                    ])


def plot_returns_curve(stats, cfg, env_line):
    """One line per variant: solid mean return over all seeds, shaded
    mean +/- 1 std band, x = episode 1..E. Saved to PLOT_PATH (overwritten
    every run). File-based only (Agg backend set at import)."""
    episodes = cfg["episodes"]
    x = np.arange(1, episodes + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in train_mod.VARIANTS:
        s = stats[variant]
        color = VARIANT_COLORS[variant]
        ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"],
                        color=color, alpha=0.2, label=None)
        ax.plot(x, s["mean"], color=color, lw=1.2,
                label=f"{variant} — mean ± std ({len(cfg['seeds'])} seeds)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_xlim(1, max(episodes, 1))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    title = (f"REINFORCE CartPole-v1 — return = episode length (gamma = {GAMMA})\n"
             f"episodes={episodes}, lr={cfg['lr']}, B={cfg['baseline_b']}, "
             f"seeds={cfg['seeds']}, device={cfg['device']}")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def write_results_md(stats, data, cfg, runtimes, env_line):
    """results/reinforce_cartpole/results.md, mirroring results/*/result.md."""
    lines = [
        "# REINFORCE CartPole-v1 results",
        "",
        "Vanilla REINFORCE (on-policy policy gradient, gamma = 1.0, reward-to-go) "
        "on CartPole-v1, two variants: `no_baseline` and `moving_avg` (per-episode "
        "moving-average baseline, B = 0.5, initial baseline 0.0). "
        f"{cfg['episodes']} episodes per (variant, seed) run, seeds "
        f"{cfg['seeds']} — same episode index comparable across variants/seeds. "
        "Model: Linear(4,64) -> ReLU -> Linear(64,2) -> Softmax; "
        f"Adam lr = {cfg['lr']}; one optimizer update per episode over the "
        "full-episode batch (loss = -sum_t logprob_t * (G_t - baseline), "
        "G_t = sum_{t' >= t} R_{t'}). CartPole-v1 rewards +1.0 per step (the "
        "terminal step included), so return = episode length (<= 500); "
        "truncated (500-step) episodes are kept as-is, no bootstrap or bonus.",
        f"Environment: {env_line}.",
        "",
        "## Best results per variant",
        "",
        "| variant | best mean return | at episode | mean return at final episode |",
        "|---------|------------------|------------|------------------------------|",
    ]
    for variant in train_mod.VARIANTS:
        s = stats[variant]
        lines.append(
            f"| {variant} | {s['best_mean']:.1f} | {s['best_episode']} "
            f"| {s['final_mean']:.1f} |"
        )
    lines += [
        "",
        "Best mean return = max over episode index of the per-episode mean over "
        f"the {len(cfg['seeds'])} seeds (std ddof = 0); curve in "
        f"`returns_curve.png` (mean line, filled band = mean ± 1 std).",
        "",
        "## Runtimes",
        "",
        "| quantity | value |",
        "|----------|-------|",
    ]
    for variant in train_mod.VARIANTS:
        per_run = np.array([data["runs"][variant][seed]["run_time_s"]
                            for seed in data["seeds"]])
        lines.append(
            f"| training, {variant} ({len(cfg['seeds'])} x {cfg['episodes']} episodes) "
            f"| {per_run.sum():.1f} s total, {per_run.mean():.1f} s/run avg |"
        )
    lines += [
        f"| total (all {2 * len(cfg['seeds'])} runs) | {runtimes['total_s']:.1f} s |",
        "",
        "## Configuration",
        "",
        "| key | value |",
        "|-----|-------|",
    ]
    for key in ("episodes", "lr", "baseline_b", "seeds", "gamma", "variants",
                "device"):
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
    parser.add_argument("--episodes", type=int, default=1000)
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
        "episodes": args.episodes, "lr": args.lr, "baseline_b": args.baseline,
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
                           episodes=args.episodes, lr=args.lr,
                           baseline_b=args.baseline, device=device)
    total_s = time.perf_counter() - t_total

    stats = aggregate_by_variant(data)
    for variant in train_mod.VARIANTS:
        s = stats[variant]
        log.info("summary %-12s best mean return %.1f at episode %d | final mean %.1f",
                 variant, s["best_mean"], s["best_episode"], s["final_mean"])

    write_csv(data, CSV_PATH)
    n_rows = len(data["seeds"]) * data["episodes"] * len(train_mod.VARIANTS)
    log.info("CSV written to %s (%d rows)", CSV_PATH, n_rows)
    plot_returns_curve(stats, cfg, env_line)
    log.info("returns curve written to %s", PLOT_PATH)
    write_results_md(stats, data, cfg, {"total_s": total_s}, env_line)
    log.info("results written to %s", RESULTS_MD)
    log.info("total wall: %.1f s", total_s)


if __name__ == "__main__":
    main()
