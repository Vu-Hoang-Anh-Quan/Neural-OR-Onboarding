"""Run ALNS on the 10 chosen CVRPLIB instances and log the results.

For each instance in data/chosen (X-n101-k25 ... X-n502-k39), alns_solve is
called TIME_LIMIT seconds per run, once per seed in SEEDS. Per-run rows are
written to results/cvrp_alns/log.csv, and a summary table
{instance | BKS | avg cost ± std | gap %} to results/cvrp_alns/result.md.

Usage (from the repo root):  python src/cvrp/run_alns.py
"""
import platform
import time
from pathlib import Path

import numpy as np

from alns import alns_solve
from cvrp_utils import cost, gap, load

TIME_LIMIT = 60.0  # seconds per (instance, seed) run
SEEDS = [4, 6, 8]

ROOT = Path(__file__).resolve().parents[2]
INSTANCES_DIR = ROOT / "data" / "chosen"
RESULTS_DIR = ROOT / "results" / "cvrp_alns"
LOG_CSV = RESULTS_DIR / "log.csv"
RESULT_MD = RESULTS_DIR / "result.md"


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

    rows, summary = [], []
    for f in inst_files:
        name = f.stem
        inst = load(f)
        ref = bks_cost(f.with_suffix(".sol"))
        costs = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            routes = alns_solve(inst, TIME_LIMIT, seed)
            el = time.perf_counter() - t0
            c = cost(routes, inst["D"])
            g = gap(c, ref)
            rows.append((name, seed, c, g, el))
            costs.append(c)
            print(f"{name}  seed={seed}  cost={c}  gap={g:.2f}%  time={el:.2f}s", flush=True)
        costs = np.asarray(costs, dtype=float)
        summary.append((name, ref, costs.mean(), costs.std()))
        print(f"=== {name}: BKS={ref}  avg={costs.mean():.1f} ± {costs.std():.1f}  "
              f"gap={gap(costs.mean(), ref):.2f}%", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_CSV, "w") as fp:
        fp.write("instance,seed,cost,gap,time\n")
        for name, seed, c, g, el in rows:
            fp.write(f"{name},{seed},{c},{g:.4f},{el:.4f}\n")

    with open(RESULT_MD, "w") as fp:
        fp.write("# ALNS results (CVRP)\n\n")
        fp.write(f"ALNS (`src/cvrp/alns.py`) on the 10 chosen CVRPLIB instances "
                 f"(`data/chosen`), {TIME_LIMIT:g}s per run, "
                 f"{len(SEEDS)} random seeds ({', '.join(map(str, SEEDS))}).\n"
                 f"Reference for the gap: best-known solution (BKS) from each "
                 f"instance's `.sol` file (a reference, not a proof of optimality).\n"
                 f"Costs use the CVRPLIB `nint` rounding convention "
                 f"(`cvrp_utils.load`).\n"
                 f"Environment: Python {platform.python_version()}, "
                 f"NumPy {np.__version__}.\n\n")
        fp.write("| instance | BKS | avg cost ± std | gap % |\n")
        fp.write("|----------|-----|----------------|-------|\n")
        for name, ref, mean, std in summary:
            fp.write(f"| {name} | {ref} | {mean:.1f} ± {std:.1f} | "
                     f"{gap(mean, ref):.2f} |\n")

    print(f"\nWrote {LOG_CSV} and {RESULT_MD}")


if __name__ == "__main__":
    main()
