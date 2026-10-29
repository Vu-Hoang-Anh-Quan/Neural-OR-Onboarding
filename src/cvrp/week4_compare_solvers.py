"""Week 4: compare CVRP solvers (PyVRP, OR-Tools, ALNS) on the 10 chosen instances.

For each instance in data/chosen (X-n101-k25 ... X-n502-k39) and each time
limit in TIME_LIMITS (seconds):

  - PyVRP    (src/run_pyvrp.py,   invoked as a CLI):  once per seed in SEEDS
  - OR-Tools (src/run_ortools.py, invoked as a CLI):  once per (instance, limit)
  - ALNS     (alns.alns_solve,    in-process):        once per seed in SEEDS

The PyVRP / OR-Tools runs are delegated to the existing CLI scripts in src/,
which each print a single CSV line per run; this script parses that line rather
than re-implementing the solver. Per-run rows are written to
results/week4/week4.csv with columns
    solver, instance, time_limit, seed, cost, feasible, wallclock, gap, hardware
where gap (%) is with respect to the best-known solution (BKS) from each
instance's `.sol` file (a reference, not a proof of optimality). A plot of the
mean gap (%) against time limit (log x-axis) per solver is written to
results/week4/week4_gap_vs_time.png.

Usage (from the repo root):  python src/cvrp/week4_compare_solvers.py
"""
import importlib.metadata
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# alns.py / cvrp_utils.py live next to this script (src/cvrp/).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from alns import alns_solve  # noqa: E402
from cvrp_utils import check, cost, gap, load  # noqa: E402

TIME_LIMITS = [1, 10, 60]  # seconds
SEEDS = [1, 2, 3]          # seeds used by PyVRP and ALNS (OR-Tools: recorded as "-")

ROOT = Path(__file__).resolve().parents[2]
INSTANCES_DIR = ROOT / "data" / "chosen"
RESULTS_DIR = ROOT / "results" / "week4"
CSV_PATH = RESULTS_DIR / "week4.csv"
PLOT_PATH = RESULTS_DIR / "week4_gap_vs_time.png"

PYVRP_CLI = ROOT / "src" / "run_pyvrp.py"
ORTOOLS_CLI = ROOT / "src" / "run_ortools.py"

# Same machine as the committed week 3 results (results/cvrp_alns/result.md);
# kept comma-free so the CSV needs no quoting.
HARDWARE = "Ryzen 7 9700 X - no GPU used"
ENV_NOTE = (f"Python {platform.python_version()}, numpy {np.__version__}, "
            f"ortools {importlib.metadata.version('ortools')}, "
            f"pyvrp {importlib.metadata.version('pyvrp')}")


def _parse_cli_line(line):
    """Parse one CSV line from run_pyvrp.py / run_ortools.py.

    Format: {solver},{path},{sec},{seed-or--},{cost},{feasible},{wall}
    Returns (cost, feasible, wallclock_s).
    """
    p = line.strip().split(",")
    return int(p[4]), p[5].strip() == "True", float(p[6])


def _run_pyvrp(path, sec, seed):
    """Run src/run_pyvrp.py as a CLI. Returns (cost, feasible, wallclock_s)."""
    out = subprocess.run(
        [sys.executable, str(PYVRP_CLI), str(path), str(sec), str(seed)],
        cwd=ROOT, capture_output=True, text=True, check=True)
    return _parse_cli_line(out.stdout)


def _run_ortools(path, sec):
    """Run src/run_ortools.py as a CLI. Returns (cost, feasible, wallclock_s)."""
    out = subprocess.run(
        [sys.executable, str(ORTOOLS_CLI), str(path), str(sec)],
        cwd=ROOT, capture_output=True, text=True, check=True)
    return _parse_cli_line(out.stdout)


def _run_alns(inst, sec, seed):
    """One ALNS run (imported from alns.py). Returns (cost, feasible, wallclock_s)."""
    t0 = time.perf_counter()
    routes = alns_solve(inst, sec, seed)
    wall = time.perf_counter() - t0
    return cost(routes, inst["D"]), check(routes, inst)[0], wall


def bks_cost(path):
    """Best-known cost from a CVRPLIB .sol file (the 'Cost <n>' line)."""
    for line in path.read_text().splitlines():
        if line.startswith("Cost"):
            return int(line.split()[1])
    raise ValueError(f"no 'Cost' line in {path}")


def main():
    # X-n<size>-k<veh>: sort by customer count (same order as data/README.md).
    inst_files = sorted(INSTANCES_DIR.glob("X-*.vrp"),
                        key=lambda p: int(p.stem.split("-")[1][1:]))
    assert len(inst_files) == 10, f"expected 10 instances, found {len(inst_files)}"
    instances = []
    for f in inst_files:
        inst = load(f)
        inst["path"] = f
        inst["bks"] = bks_cost(f.with_suffix(".sol"))
        instances.append(inst)
        print(f"loaded {f.stem}  BKS={inst['bks']}", flush=True)

    print(f"{ENV_NOTE} | {HARDWARE}\n", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Open the CSV up front (truncate) and append rows as runs finish, so a
    # killed run does not lose its earlier rows.
    with open(CSV_PATH, "w") as fp:
        fp.write("solver,instance,time_limit,seed,cost,feasible,wallclock,gap,hardware\n")
        for inst in instances:
            name = inst["path"].stem
            for tlim in TIME_LIMITS:
                # (seed, solver): PyVRP and ALNS run once per seed; OR-Tools once,
                # recorded with seed "-" (it has no seed, like the CLI prints).
                jobs = [(s, "pyvrp") for s in SEEDS] + \
                       [("-", "ortools")] + \
                       [(s, "alns") for s in SEEDS]
                for seed, solver in jobs:
                    if solver == "pyvrp":
                        c, ok, wc = _run_pyvrp(inst["path"], tlim, seed)
                    elif solver == "ortools":
                        c, ok, wc = _run_ortools(inst["path"], tlim)
                    else:
                        c, ok, wc = _run_alns(inst, tlim, seed)
                    g = gap(c, inst["bks"]) if ok else float("nan")
                    fp.write(f"{solver},{name},{tlim},{seed},{c},{ok},{wc:.1f},"
                             f"{g:.4f},{HARDWARE}\n")
                    fp.flush()
                    print(f"{name}  {solver:8s}  t={tlim:>2d}s  seed={seed}  "
                          f"cost={c}  feas={ok}  gap={g:.2f}%  wall={wc:.2f}s",
                          flush=True)

    # ---- Plot: gap (%) vs time limit (log x), one panel per solver ----------
    by = {}  # (solver, tlim) -> [gap per (instance, seed)]
    with open(CSV_PATH) as fp:
        next(fp)  # skip header
        for line in fp:
            p = line.rstrip("\n").split(",")
            if p[5] == "True":
                by.setdefault((p[0], int(p[2])), []).append(float(p[7]))

    solver_labels = {"pyvrp": "PyVRP", "ortools": "OR-Tools", "alns": "ALNS"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, solver in zip(axes, ("pyvrp", "ortools", "alns")):
        means = [np.mean(by.get((solver, t), [np.nan])) for t in TIME_LIMITS]
        stds = [np.std(by.get((solver, t), [np.nan])) for t in TIME_LIMITS]
        ax.errorbar(TIME_LIMITS, means, yerr=stds, fmt="o-", capsize=5)
        ax.set_xscale("log")
        ax.set_xticks(TIME_LIMITS)
        ax.set_xticklabels([str(t) for t in TIME_LIMITS])
        ax.set_xlabel("Time limit (s)")
        ax.set_title(solver_labels[solver])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Gap vs BKS (%)")
    fig.suptitle("CVRP: gap vs BKS (%) vs time limit, 10 chosen instances. "
                 "Marker: mean gap at that limit; bar: std over the "
                 "10 instances (x seeds) at that limit.", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight", dpi=150)
    print(f"\nWrote {CSV_PATH} and {PLOT_PATH}")


if __name__ == "__main__":
    main()
