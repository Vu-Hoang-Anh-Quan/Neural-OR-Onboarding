"""Run TSP heuristics and compare with LKH-3 (and the MIP optimum for n = 12).

Heuristics (from heuristics.py):
  - nearest_neighbor(D, start=0)   nearest-neighbor construction
  - two_opt(tour, D)               first-improvement 2-opt
  - or_opt(tour, D)                first-improvement Or-opt (segments of 1-3 nodes)
  - alternating 2-opt + Or-opt until no improvement

Experiments: random uniform TSP in [0,1]^2, n in {12, 20, 50, 100, 200},
20 instances per n, all drawn from a single NumPy RNG seeded 42 (so only the
seed 42 is used; the instance index distinguishes the 20 draws).

Reference:
  - LKH-3 (3.0.14) for n != 12
  - MIP optimum (MTZ formulation, SCIP, from tsp_mtz.py) for n = 12

Outputs (results/tsp_heuristics/):
  - log.csv     per-instance costs, gaps and wall-clock times
  - result.md   table averaged over each n

Run from the repo root:
    python src/run_tsp_heuristics.py
"""
import csv
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

from heuristics import nearest_neighbor, or_opt, tour_length, two_opt
from tsp_mtz import solve_tsp_mtz

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LKH_BIN = os.path.join(REPO_ROOT, "data", "LKH-3.0.14", "LKH")
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "tsp_heuristics")
LOG_PATH = os.path.join(RESULTS_DIR, "log.csv")
MD_PATH = os.path.join(RESULTS_DIR, "result.md")

N_LIST = [12, 20, 50, 100, 200]
INSTANCES_PER_N = 20
SEED = 42
SCALE = 1_000_000          # integer scale used for LKH's EUC_2D instances
MIP_N = 12                 # only this n is compared against the MIP optimum
MIP_TIME_LIMIT = 60        # seconds per MIP instance
LKH_TIMEOUT = 180          # seconds per LKH run


# --------------------------------------------------------------------------- #
# Instance helpers
# --------------------------------------------------------------------------- #
def euclidean_D(xy):
    """Continuous Euclidean distance matrix of the [n, 2] point cloud `xy`."""
    diff = xy[:, None, :] - xy[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def scaled_coords(xy):
    """Integer coordinates scaled by SCALE, as expected by LKH's EUC_2D."""
    return np.rint(xy * SCALE).astype(np.int64)


def gap(cost, ref):
    """Percentage gap of `cost` with respect to the reference `ref`."""
    return 100.0 * (cost - ref) / ref


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #
def run_heuristics(D):
    """Return (nn_cost, two_opt_cost, or_opt_cost, (t_nn, t_two_opt, t_or_opt)).

    `or_opt_cost` is the tour after alternating 2-opt + Or-opt until no move
    improves the objective. All times are wall-clock seconds.
    """
    t0 = time.perf_counter()
    tour = nearest_neighbor(D, start=0)
    nn_cost = tour_length(tour, D)
    t1 = time.perf_counter()

    tour = two_opt(tour, D)
    two_opt_cost = tour_length(tour, D)
    t2 = time.perf_counter()

    # Alternate 2-opt + Or-opt until no improvement.
    while True:
        before = tour_length(tour, D)
        or_opt(tour, D)
        two_opt(tour, D)
        after = tour_length(tour, D)
        if after >= before - 1e-9:
            break
    or_opt_cost = tour_length(tour, D)
    t3 = time.perf_counter()

    return nn_cost, two_opt_cost, or_opt_cost, (t1 - t0, t2 - t1, t3 - t2)


# --------------------------------------------------------------------------- #
# LKH-3
# --------------------------------------------------------------------------- #
def write_tsp(path, coords, name):
    """Write a TSPLIB .tsp file with integer EUC_2D coordinates."""
    with open(path, "w") as f:
        f.write(f"NAME : {name}\n")
        f.write(f"COMMENT : random uniform in [0,1]^2 scaled by {SCALE}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {len(coords)}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")


def write_par(path, tsp_path, tour_path):
    """Parameter file modelled on src/tsp.par (Appendix B)."""
    with open(path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\n")
        f.write(f"TOUR_FILE = {tour_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
        f.write("RUNS = 1\n")
        f.write("MAX_TRIALS = 1000\n")
        f.write("SEED = 1\n")


def parse_lkh_cost(tour_path):
    """LKH reports its cost in the `COMMENT : Length = ...` header line."""
    with open(tour_path) as f:
        for line in f:
            m = re.search(r"Length\s*=\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def run_lkh(tsp_path, workdir, tag):
    """Run LKH-3 on one instance; return (cost_scaled, wallclock_s)."""
    tour_path = os.path.join(workdir, f"{tag}.tour")
    par_path = os.path.join(workdir, f"{tag}.par")
    write_par(par_path, tsp_path, tour_path)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [LKH_BIN, par_path],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=LKH_TIMEOUT,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"LKH failed on {par_path} (rc={proc.returncode}):\n"
            f"{proc.stdout.decode(errors='replace')}"
        )
    if not os.path.exists(tour_path):
        raise RuntimeError(f"LKH did not write {tour_path}")

    cost = parse_lkh_cost(tour_path)
    if cost is None:
        raise RuntimeError(f"Could not read LKH cost from {tour_path}")
    return cost, dt


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def solve_mip(D):
    """Return (cost, optimal, wallclock_s) from the MTZ formulation."""
    t0 = time.perf_counter()
    cost, _tour, optimal = solve_tsp_mtz(D, time_limit_s=MIP_TIME_LIMIT)
    dt = time.perf_counter() - t0
    return float(cost), bool(optimal), dt


def run_all():
    if not os.path.exists(LKH_BIN):
        raise FileNotFoundError(
            f"LKH binary not found at {LKH_BIN}; build/place it there first."
        )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    workdir = tempfile.mkdtemp(prefix="lkh_tsp_")

    rows = []
    for n in N_LIST:
        for inst in range(INSTANCES_PER_N):
            xy = rng.random((n, 2))
            D = euclidean_D(xy)
            coords = scaled_coords(xy)

            nn_c, opt_c, oro_c, (t_nn, t_opt, t_oro) = run_heuristics(D)

            tsp_path = os.path.join(workdir, f"tsp-n{n}-i{inst}.tsp")
            write_tsp(tsp_path, coords, f"tsp-n{n}-i{inst}")
            lkh_cost_scaled, t_lkh = run_lkh(tsp_path, workdir, f"tsp-n{n}-i{inst}")
            lkh_cost = lkh_cost_scaled / SCALE

            if n == MIP_N:
                mip_c, mip_opt, t_mip = solve_mip(D)
                ref_cost, ref_type = mip_c, "mip"
            else:
                mip_c, mip_opt, t_mip = None, False, 0.0
                ref_cost, ref_type = lkh_cost, "lkh"

            rows.append({
                "n": n,
                "instance": inst,
                "seed": SEED,
                "nn_cost": nn_c,
                "nn2opt_cost": opt_c,
                "nn2opt_oropt_cost": oro_c,
                "lkh_cost": lkh_cost,
                "mip_cost": mip_c,
                "ref_type": ref_type,
                "ref_cost": ref_cost,
                "gap_nn": gap(nn_c, ref_cost),
                "gap_nn2opt": gap(opt_c, ref_cost),
                "gap_nn2opt_oropt": gap(oro_c, ref_cost),
                "t_nn": t_nn,
                "t_nn2opt": t_opt,
                "t_nn2opt_oropt": t_oro,
                "t_lkh": t_lkh,
                "t_mip": t_mip,
                "mip_optimal": int(mip_opt),
            })
            print(
                f"n={n:>3} inst={inst:>2} nn={nn_c:.4f} 2opt={opt_c:.4f} "
                f"oro={oro_c:.4f} ref({ref_type})={ref_cost:.4f} "
                f"gap={gap(oro_c, ref_cost):.3f}%",
                flush=True,
            )
    return rows


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
LOG_FIELDS = [
    "n", "instance", "seed",
    "nn_cost", "nn2opt_cost", "nn2opt_oropt_cost",
    "lkh_cost", "mip_cost", "ref_type", "ref_cost",
    "gap_nn", "gap_nn2opt", "gap_nn2opt_oropt",
    "t_nn", "t_nn2opt", "t_nn2opt_oropt", "t_lkh", "t_mip", "mip_optimal",
]


def write_log(rows):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r[k]) for k in LOG_FIELDS})


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def aggregate(rows):
    """Average every metric over the INSTANCES_PER_N draws of each n."""
    agg = []
    for n in N_LIST:
        sub = [r for r in rows if r["n"] == n]
        k = len(sub)
        agg.append((
            n,
            sum(r["gap_nn"] for r in sub) / k,
            sum(r["gap_nn2opt"] for r in sub) / k,
            sum(r["gap_nn2opt_oropt"] for r in sub) / k,
            sum(r["t_nn"] + r["t_nn2opt"] + r["t_nn2opt_oropt"] for r in sub) / k,
            k,
        ))
    return agg


def ortools_version():
    try:
        import ortools
        return ortools.__version__
    except Exception:
        return "unknown"


def write_markdown(agg):
    lines = [
        "# TSP heuristics results",
        "",
        "Random uniform TSP instances in `[0,1]^2`, generated with a single NumPy "
        f"RNG seeded `{SEED}` (only this seed is used); {INSTANCES_PER_N} instances "
        "per n (instance index 0..19).",
        "",
        "- Heuristics: **NN** = nearest neighbor (start 0); **2opt** = first-improvement "
        "2-opt; **Oropt** = alternating 2-opt + Or-opt (1-3 node segments) until no improvement.",
        f"- Reference for the gap: **MIP optimum** (MTZ / SCIP, from `tsp_mtz.py`, "
        f"{MIP_TIME_LIMIT}s limit) for n = {MIP_N}; **LKH-3** (`data/LKH-3.0.14`) for the other n.",
        "- Costs are Euclidean tour lengths on `[0,1]^2`. LKH is run on integer "
        f"coordinates scaled by {SCALE} (TSPLIB `EUC_2D`, `nint` rounding); its "
        "cost is divided back by the scale.",
        "- Gap (%) = `100 * (heuristic - reference) / reference`.",
        "- Time (s) is the wall-clock time of the full NN + 2-opt + Or-opt pipeline.",
        f"- Environment: Python {sys.version.split()[0]}, NumPy {np.__version__}, "
        f"OR-Tools {ortools_version()}, LKH-3.0.14.",
        "",
        "| n | NN gap | NN+2opt gap | NN+2opt+Oropt gap | time (s) |",
        "|---|--------|-------------|-------------------|----------|",
    ]
    for n, g_nn, g_opt, g_oro, t_total, k in agg:
        lines.append(f"| {n} | {g_nn:.2f} | {g_opt:.2f} | {g_oro:.2f} | {t_total:.4f} |")
    lines += [
        "",
        f"Gaps are averaged over the {INSTANCES_PER_N} instances of each n. "
        "Per-instance values (costs, per-stage times, MIP optimality flags) are in `log.csv`.",
    ]
    with open(MD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = run_all()
    write_log(rows)
    write_markdown(aggregate(rows))
    print(f"\nWrote {LOG_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
