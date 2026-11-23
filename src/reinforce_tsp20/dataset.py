"""Evaluation dataset + LKH-3 labeling for the REINFORCE TSP-20 solver.

Owns the *fixed* evaluation instances and their reference labels:

- Instances: generated once by `gen.gen_tsp` and persisted under
  `data/reinforce_tsp20/eval/<split>/<s>.txt` (raw gen_tsp format: line 1 = n,
  then the n x n integer matrix).  Splits are seeded from a single base offset
  (``BASE``) so the eval sets never collide with the training-seed space and
  are reproducible exactly:
    - val  : ``s = 0..val_n-1``,   seed = ``BASE + s``
    - test : ``s = 0..test_n-1``,  seed = ``BASE + 1000 + s``
  (``BASE`` is recorded in `results.md`.)

- Reference labels, computed **only** on the eval instances (never used for
  training): NN length, NN+2-opt length and the LKH-3 tour length, all in one
  pass, persisted to `data/reinforce_tsp20/eval/lkh_<split>.csv` with columns
  ``instance_id, nn_len, nn2opt_len, lkh_len`` (instance_id = the local index
  ``s``).  The LKH-3 tour is wrapped into a TSPLIB explicit-matrix ATSP file
  (`TYPE : ATSP` with `EDGE_WEIGHT_TYPE : EXPLICIT` / `FULL_MATRIX` — a plain
  `TSP` mis-solves explicit asymmetric matrices; the row-block carries **no**
  row-index column and no `EDGE_WEIGHT_FIRST_LAST_COLUMN`, which this LKH build
  rejects — verified against the working `tsp_heatmap` reference).  LKH's tour
  direction is arbitrary, so the *orientation gate* checks which of the forward
  or backward tour sum equals the reported length and stores the matching one.
  Labeling is idempotent: existing CSVs are parsed and re-asserted on re-run
  and are only recomputed when `--relabel` is passed.
"""

import csv
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Repo layout: this file lives in src/reinforce_tsp20/; `gen` / `heuristics`
# live in src/.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gen import gen_tsp  # noqa: E402
from heuristics import nearest_neighbor, two_opt, tour_length  # noqa: E402

log = logging.getLogger("reinforce_tsp20.dataset")

DATA_ROOT = REPO_ROOT / "data" / "reinforce_tsp20"
EVAL_DIR = DATA_ROOT / "eval"
SCRATCH_DIR = EVAL_DIR / "scratch"
LKH_DIR = REPO_ROOT / "data" / "LKH-3.0.14"
LKH_BIN = LKH_DIR / "LKH"

# Base offset for eval seeds (recorded in results.md); chosen so the eval sets
# never collide with the training batch seeds (~ policy_seed * 100000 + ...).
BASE = 1_000_000_000
# test seeds start at BASE + 1000, so val/test never share a seed either.
_TEST_OFFSET = 1000


# ------------------------------------------------------------------ instances
def read_instance(path):
    """gen_tsp format: first line n, then a full n x n integer matrix."""
    with open(path) as f:
        n = int(f.readline().strip())
        rows = [ln.strip() for ln in f]
        rows = [ln for ln in rows if ln]
    D = np.array([list(map(int, ln.split())) for ln in rows])
    assert D.shape == (n, n), f"{path}: matrix {D.shape} != ({n}, {n})"
    return D


def write_split_files(split, n, n_split, base_seed):
    """Ensure `<split>/<s>.txt` exists for s in [0, n_split); return (seeds, Ds).

    Deterministic: gen_tsp writes the same matrix for a given seed; existing
    files are left untouched.
    """
    d = EVAL_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    seeds, Ds = [], []
    for s in range(n_split):
        seed = base_seed + s
        p = d / f"{s}.txt"
        if not p.exists():
            gen_tsp(n, seed=seed, path=str(p))
        Ds.append(read_instance(p))
        seeds.append(seed)
    return seeds, Ds


# ------------------------------------------------------------------ NN helpers
def nn_and_2opt_lengths(D):
    """NN tour length and NN+2-opt tour length on the raw matrix (start = 0).

    Works for asymmetric matrices: `two_opt` in src/heuristics.py uses the
    exact asymmetric delta.  `two_opt` still mutates in place, so it is called
    on a copy of the NN tour.
    """
    nn = nearest_neighbor(D, start=0)
    nn_len = tour_length(nn, D)
    t2 = two_opt(list(nn), D)  # mutates in place
    nn2opt_len = tour_length(t2, D)
    return float(nn_len), float(nn2opt_len)


# -------------------------------------------------------------------- LKH-3
def _write_tsp_explicit(path, D, name):
    n = D.shape[0]
    with open(path, "w") as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : ATSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for row in D:
            f.write(" ".join(map(str, row)) + "\n")
        f.write("EOF\n")


def _write_par(path, tsp_path, tour_path):
    with open(path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
        f.write("RUNS = 1\n")
        f.write("MAX_TRIALS = 1000\n")
        f.write("SEED = 1\n")


def _parse_lkh_tour(tour_path):
    """Parse LKH's tour file: (length, 0-based tour). Validates a permutation."""
    with open(tour_path) as f:
        raw = f.read()
    m = re.search(r"Length\s*=\s*(\d+)", raw)
    assert m, f"no Length in {tour_path}"
    lkh_len = int(m.group(1))
    lines = [ln.strip() for ln in raw.splitlines()]
    start = lines.index("TOUR_SECTION") + 1
    nodes = []
    for ln in lines[start:]:
        if not ln:
            continue
        v = int(ln)
        if v == -1:
            break
        nodes.append(v)
    n = len(nodes)
    assert set(nodes) == set(range(1, n + 1)), f"bad tour in {tour_path}: {nodes}"
    return lkh_len, [x - 1 for x in nodes]


def _run_lkh(D, seed):
    """Run LKH-3 on one instance, return (lkh_len, tour, dt_wall_s).

    The tour is returned in the orientation whose sum equals the reported
    length (LKH's direction is arbitrary).  Scratch files are cleaned up.
    """
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    tsp_path = SCRATCH_DIR / f"{seed}.tsp"
    par_path = SCRATCH_DIR / f"{seed}.par"
    tour_tmp = SCRATCH_DIR / f"{seed}.lkh.tour"
    _write_tsp_explicit(str(tsp_path), D, f"s{seed}")
    _write_par(str(par_path), str(tsp_path), str(tour_tmp))
    t0 = time.perf_counter()
    proc = subprocess.run([str(LKH_BIN), str(par_path)], cwd=str(LKH_DIR),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=180)
    dt = time.perf_counter() - t0
    try:
        if proc.returncode != 0:
            raise RuntimeError(
                f"LKH failed (seed {seed}, rc {proc.returncode}):\n"
                f"{proc.stdout.decode(errors='replace')[-2000:]}")
        lkh_len, tour = _parse_lkh_tour(tour_tmp)
        fwd = tour_length(tour, D)
        bwd = tour_length(tour[::-1], D)
        if int(lkh_len) == int(fwd):
            pass
        elif int(lkh_len) == int(bwd):
            tour = list(reversed(tour))
        else:
            raise RuntimeError(
                f"seed {seed}: LKH length {lkh_len} matches neither forward {fwd} "
                f"nor backward {bwd} — LKH output invalid for this instance")
        return int(lkh_len), tour, dt
    finally:
        tsp_path.unlink(missing_ok=True)
        par_path.unlink(missing_ok=True)
        tour_tmp.unlink(missing_ok=True)


# -------------------------------------------------------------------- CSV in
def _try_reuse_csv(csv_path, n_split):
    """Return {s: (nn, nn2opt, lkh)} if `csv_path` fully covers 0..n_split-1,
    else None (regeneration needed)."""
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != n_split:
        return None
    if {int(r["instance_id"]) for r in rows} != set(range(n_split)):
        return None
    out = {}
    for r in rows:
        s = int(r["instance_id"])
        out[s] = (float(r["nn_len"]), float(r["nn2opt_len"]), int(r["lkh_len"]))
    return out


def _write_csv(csv_path, rows_by_s, n_split):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "nn_len", "nn2opt_len", "lkh_len"])
        for s in range(n_split):
            nn, nn2, lkh = rows_by_s[s]
            w.writerow([s, int(nn), int(nn2), int(lkh)])


# ------------------------------------------------------------------- build
def build_eval(n, val_n, test_n, relabel=False):
    """Generate + persist the val/test instances and run the LKH-3 labeling pass.

    Returns a dict:
      {
        "base": BASE,
        "val":  {"n", "seeds",[...], "D":[D,...],
                 "nn_len":[...], "nn2opt_len":[...], "lkh_len":[...]},
        "test": {...},
        "lkh_wall_s": {"val": .., "test": .., "total": ..},
      }
    """
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    wall = {"val": 0.0, "test": 0.0, "total": 0.0}
    out = {"base": BASE}

    for split, n_split, base_seed in (
        ("val", val_n, BASE),
        ("test", test_n, BASE + _TEST_OFFSET),
    ):
        seeds, Ds = write_split_files(split, n, n_split, base_seed)

        csv_path = EVAL_DIR / f"lkh_{split}.csv"
        rows_by_s = None if relabel else _try_reuse_csv(csv_path, n_split)
        if rows_by_s is None:
            rows_by_s = {}
            for s, seed, D in zip(range(n_split), seeds, Ds):
                nn_len, nn2opt_len = nn_and_2opt_lengths(D)
                lkh_len, _tour, dt = _run_lkh(D, seed)
                wall[split] += dt
                rows_by_s[s] = (nn_len, nn2opt_len, lkh_len)
                log.info("LKH %-4s [instance %d/%d seed %d]: nn %.0f, "
                         "nn2opt %.0f, lkh %d (LKH wall %.3f s)",
                         split, s + 1, n_split, seed, nn_len, nn2opt_len,
                         lkh_len, dt)
            _write_csv(csv_path, rows_by_s, n_split)
            log.info("LKH %-4s: %d instances (seeds %d..%d), wall %.2f s "
                     "(recomputed)",
                     split, n_split, base_seed, base_seed + n_split - 1,
                     wall[split])
        else:
            log.info("LKH %-4s: %d instances reused from %s",
                     split, n_split, csv_path)

        out[split] = {
            "n": n_split,
            "seeds": seeds,
            "D": Ds,
            "nn_len": [rows_by_s[s][0] for s in range(n_split)],
            "nn2opt_len": [rows_by_s[s][1] for s in range(n_split)],
            "lkh_len": [rows_by_s[s][2] for s in range(n_split)],
        }

    wall["total"] = wall["val"] + wall["test"]
    out["lkh_wall_s"] = wall
    return out


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(message)s",
                        datefmt="%H:%M")
    data = build_eval(n=20, val_n=4, test_n=4, relabel=True)
    for split in ("val", "test"):
        d = data[split]
        log.info("%-4s: n=%d base seed=%d | lkh mean %.1f | nn mean %.1f "
                 "| nn2opt mean %.1f",
                 split, d["n"], d["seeds"][0],
                 float(np.mean(d["lkh_len"])),
                 float(np.mean(d["nn_len"])),
                 float(np.mean(d["nn2opt_len"])))
    return data


if __name__ == "__main__":
    main()
