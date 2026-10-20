"""Run MTZ on 25 randomly generated TSP instances and summarize results.

- Generate 25 instances (n in {8, 10, 12, 14, 16}, seed 0..4) into data/tsp_mtz_test.
- Solve each instance with solve_tsp_mtz at a 300s limit per instance; time with time.perf_counter.
- Determine optimality from the flag returned by solve_tsp_mtz itself (status == OPTIMAL).
- Write a per-instance log to results/tsp_mtz_log.csv, a summary table to
  results/tsp_mtz_result.md, and a plot of time vs n to results/tsp_mtz_plot.png.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt

from gen import gen_tsp
from tsp_mtz import solve_tsp_mtz

N_LIST = [8, 10, 12, 14, 16]
SEEDS = range(5)
TIME_LIMIT = 300                     # seconds / instance
DATA_DIR = "data/tsp_mtz_test"
RESULTS_DIR = "results"
LOG_PATH = os.path.join(RESULTS_DIR, "tsp_mtz_log.csv")
MD_PATH = os.path.join(RESULTS_DIR, "tsp_mtz_result.md")
PLOT_PATH = os.path.join(RESULTS_DIR, "tsp_mtz_plot.png")


def load_tsp(path):
    """Read an instance written by gen_tsp: first line is n, followed by the n x n distance matrix."""
    with open(path) as f:
        n = int(f.readline())
    D = np.loadtxt(path, skiprows=1)
    assert D.shape == (n, n), f"{path}: expected {n}x{n}, got {D.shape}"
    return D.astype(int)


def run_all():
    """Generate instances + solve + return log rows (n, seed, time, cost, optimal)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    for n in N_LIST:
        for seed in SEEDS:
            path = gen_tsp(n, seed, path=os.path.join(DATA_DIR, f"tsp-n{n}-s{seed}.txt"))
            D = load_tsp(path)
            t = time.perf_counter()
            cost, _tour, optimal = solve_tsp_mtz(D, time_limit_s=TIME_LIMIT)
            dt = time.perf_counter() - t
            rows.append((n, seed, dt, int(cost), int(bool(optimal))))
            print(f"n={n} seed={seed} time={dt:.2f}s cost={int(cost)} optimal={bool(optimal)}",
                  flush=True)
    return rows


def aggregate(rows):
    """Aggregate by n: average time and number of instances solved optimally."""
    agg = []
    for n in N_LIST:
        sub = [r for r in rows if r[0] == n]
        avg = sum(r[2] for r in sub) / len(sub)
        solved = sum(r[4] for r in sub)
        agg.append((n, avg, solved, len(sub)))
    return agg


def write_log(rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write("n,seed,time_s,cost,optimal\n")
        for n, seed, dt, cost, optimal in rows:
            f.write(f"{n},{seed},{dt:.4f},{cost},{optimal}\n")


def write_markdown(agg):
    lines = [
        "# MTZ results (TSP)",
        "",
        f"25 random instances in `{DATA_DIR}`, {TIME_LIMIT}s limit per instance.",
        "",
        "| n | average time solving (s) | number of instances solved optimally |",
        "|---|--------------------------|--------------------------------------|",
    ]
    for n, avg, solved, total in agg:
        lines.append(f"| {n} | {avg:.2f} | {solved}/{total} |")
    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_plot():
    """Plot average time vs n, read back from the CSV log."""
    n_vals, avg_vals = [], []
    with open(LOG_PATH) as f:
        next(f)
        rows = [line.split(",") for line in f if line.strip()]
    for n in N_LIST:
        sub = [float(r[2]) for r in rows if int(r[0]) == n]
        n_vals.append(n)
        avg_vals.append(sum(sub) / len(sub))
    plt.figure(figsize=(6, 4))
    plt.plot(n_vals, avg_vals, marker="o")
    plt.xlabel("n (number of cities)")
    plt.ylabel("average time solving (s)")
    plt.title("MTZ TSP: average solving time vs n")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main():
    rows = run_all()
    agg = aggregate(rows)
    write_log(rows)
    write_markdown(agg)
    write_plot()
    print(f"\nWrote {LOG_PATH}, {MD_PATH}, {PLOT_PATH}")


if __name__ == "__main__":
    main()
