"""Dataset construction for the TSP heatmap experiment.

- Generates random asymmetric TSP-n instances with `gen.gen_tsp`: one global
  instance index i, seed = i (so no seed is ever shared across splits). Files:
  `data/tsp_heatmap/{train,val,test}/{i}.txt` (line 1: n, then the n x n matrix).
- Labels each instance with an LKH-3 tour: the matrix is wrapped into a TSPLIB
  file (`TYPE : ATSP` with `EDGE_WEIGHT_TYPE : EXPLICIT` / `FULL_MATRIX` —
  `TYPE : TSP` mis-solves explicit asymmetric matrices and reports a wrong
  length). Label validity gate: LKH's reported length must equal the forward
  or backward tour sum on the raw matrix (LKH solves the ATSP optimally on
  these small instances — verified against exact Held-Karp — but lists the
  tour in either orientation, so the sum in the listed orientation is not
  guaranteed to match). If the length matches the backward sum, the persisted
  tour is reversed and re-written, so `{i}.tour` always carries the tour in
  the orientation that matches LKH's reported length. Tours are persisted as
  `{i}.tour` next to the instance; labeling is idempotent (existing tours are
  parsed and re-asserted, not re-run).
- Builds the directed k-NN candidate graph (out-neighbors by distance row) and
  one PyG graph per instance: x_i = D[i, :] / 100, candidate edges (i -> j) for
  each of i's k out-neighbors, y = 1 iff (i, j) is in the LKH tour.

Read the instance files manually (line 1 = n), never `np.loadtxt` the whole
file (ragged column parse error).
"""

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TDDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Repo layout: this file lives in src/tsp_heatmap/; `gen` lives in src/.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gen import gen_tsp  # noqa: E402

log = logging.getLogger("tsp_heatmap.dataset")

DATA_ROOT = REPO_ROOT / "data" / "tsp_heatmap"
SCRATCH_DIR = DATA_ROOT / "scratch"
LKH_DIR = REPO_ROOT / "data" / "LKH-3.0.14"
LKH_BIN = LKH_DIR / "LKH"

SPLIT_NAMES = ("train", "val", "test")


class HeatmapDataset(TDDataset):
    """Wraps a list of per-instance PyG graphs so the loader yields one graph per item."""

    def __init__(self, graphs):
        self.graphs = list(graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def read_instance(path):
    """gen_tsp format: first line n, then a full n x n integer matrix."""
    with open(path) as f:
        n = int(f.readline().strip())
        rows = [ln.strip() for ln in f]
        rows = [ln for ln in rows if ln]
    D = np.array([list(map(int, ln.split())) for ln in rows])
    assert D.shape == (n, n), f"{path}: matrix {D.shape} != ({n}, {n})"
    return D


def out_neighbors(D, k):
    """Directed k-NN (out-neighbors): for each i, the k smallest D[i, :], i excluded."""
    n = D.shape[0]
    arange = np.arange(n)
    nn = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        others = np.setdiff1d(arange, [i], assume_unique=True)
        nn[i] = others[np.argpartition(D[i, others], k - 1)[:k]]
    return nn


def write_tsp_explicit(path, D, name):
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


def write_par(path, tsp_path, tour_path):
    with open(path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
        f.write("RUNS = 1\n")
        f.write("MAX_TRIALS = 1000\n")
        f.write("SEED = 1\n")


def _parse_lkh_tour(tour_path):
    """Parse LKH's tour file: (length, 0-based tour). Validates a permutation of 1..n."""
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


def _tour_edges(tour):
    n = len(tour)
    return [(tour[t], tour[(t + 1) % n]) for t in range(n)]


def _check_tour_integrity(D, tour, label):
    """A valid label is a Hamiltonian cycle: permutation of 0..n-1 with no
    repeated directed edge. Orientation-independent (forward == backward here)."""
    n = D.shape[0]
    assert len(tour) == n and set(tour) == set(range(n)), f"{label}: not a permutation of 0..{n - 1}"
    edges = _tour_edges(tour)
    assert len(set(edges)) == n, f"{label}: repeated directed edge"


def _written_length(tour_file):
    with open(tour_file) as f:
        m = re.search(r"Length\s*=\s*(\d+)", f.read())
    return int(m.group(1)) if m else None


def _write_lkh_tour(src, dst, new_tour):
    """Rewrite an LKH tour file with `new_tour` (0-based), preserving the
    NAME / COMMENT header lines; keeps LKH's header layout (TOUR_SECTION then
    one node per line, then -1, EOF)."""
    lines = src.read_text().splitlines()
    if "TOUR_SECTION" in lines:
        header = lines[: lines.index("TOUR_SECTION")]
    else:  # pre-TOUR_SECTION layout: header ends at the DIMENSION line
        header = lines[: lines.index(next(ln for ln in lines if ln.startswith("DIMENSION")))]
    out = header + ["TOUR_SECTION"] + [str(v + 1) for v in new_tour] + ["-1", "EOF"]
    dst.write_text("\n".join(out) + "\n")


def label_instance(D, inst_file, tour_file):
    """Ensure {i}.tour exists and is valid. Returns (tour, dt_wall_s, reused).

    tour is the Hamiltonian cycle in the orientation that matches LKH's
    reported length (backward if the length equals the backward sum).
    """
    if tour_file.exists():
        lkh_len, tour = _parse_lkh_tour(tour_file)
        _check_tour_integrity(D, tour, str(tour_file))
        fwd = int(sum(D[a, b] for a, b in _tour_edges(tour)))
        assert lkh_len == fwd, (
            f"{tour_file}: LKH length {lkh_len} != forward tour sum {fwd} "
            f"(tour not persisted in the length-matching orientation)")
        return tour, 0.0, True

    n = D.shape[0]
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    tsp_path = SCRATCH_DIR / f"{inst_file.stem}.tsp"
    par_path = SCRATCH_DIR / f"{inst_file.stem}.par"
    tour_tmp = tour_file.with_suffix(".tour.tmp")
    write_tsp_explicit(str(tsp_path), D, inst_file.stem)
    write_par(str(par_path), str(tsp_path), str(tour_tmp))
    t0 = time.perf_counter()
    proc = subprocess.run([str(LKH_BIN), str(par_path)], cwd=str(LKH_DIR),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=180)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"LKH failed on {inst_file}:\n{proc.stdout.decode(errors='replace')}")
    lkh_len, tour = _parse_lkh_tour(tour_tmp)
    _check_tour_integrity(D, tour, str(tour_tmp))
    fwd = int(sum(D[a, b] for a, b in _tour_edges(tour)))
    bwd = int(sum(D[b, a] for a, b in _tour_edges(tour)))
    if lkh_len == fwd:
        pass
    elif lkh_len == bwd:
        tour = list(reversed(tour))
        _write_lkh_tour(tour_tmp, tour_tmp, tour)
    else:
        raise RuntimeError(
            f"{tour_tmp}: LKH length {lkh_len} matches neither forward sum {fwd} "
            f"nor backward sum {bwd} — LKH output invalid for this instance")
    tour_tmp.replace(tour_file)
    assert _written_length(tour_file) == lkh_len, (
        f"{tour_file}: Length line corrupted on rewrite")
    tsp_path.unlink(missing_ok=True)
    par_path.unlink(missing_ok=True)
    return tour, dt, False


def build_graph(D, tour, k):
    """One PyG graph: node features, directed k-NN candidate edges, per-edge tour labels."""
    n = D.shape[0]
    nn = out_neighbors(D, k)
    tour_edges = set(_tour_edges(tour))
    src, dst, y = [], [], []
    for i in range(n):
        for j in nn[i]:
            j = int(j)
            src.append(i)
            dst.append(j)
            y.append(1.0 if (i, j) in tour_edges else 0.0)
    x = torch.from_numpy((D.astype(np.float32) / 100.0))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.float32))


def tour_edge_coverage(D, tour, k):
    """Fraction of directed tour edges (i, j) with j among i's k out-neighbors.

    Per-instance recall upper bound: recall can never exceed it.
    """
    nn = out_neighbors(D, k)
    edges = _tour_edges(tour)
    nn_sets = [set(row.tolist()) for row in nn]
    hits = sum(1 for i, j in edges if j in nn_sets[i])
    return hits / len(edges)


def split_indices(cfg):
    start = 0
    out = {}
    for name in SPLIT_NAMES:
        k = cfg[f"{name}_n"]
        out[name] = range(start, start + k)
        start += k
    return out


def build_splits(cfg):
    """Generate + label all instances of all splits; build per-instance PyG graphs.

    Returns ({split: [Data, ...]}, meta) where meta carries the per-split
    k-NN coverage lists and the LKH labeling summary.
    """
    n, k = cfg["n"], cfg["k"]
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    graphs, coverages = {}, {}
    lkh = {"run": 0, "reused": 0, "wall_total_s": 0.0}
    for name in SPLIT_NAMES:
        split_dir = DATA_ROOT / name
        split_dir.mkdir(parents=True, exist_ok=True)
        gs, covs = [], []
        for i in split_indices(cfg)[name]:
            inst_file = split_dir / f"{i}.txt"
            tour_file = split_dir / f"{i}.tour"
            if not inst_file.exists():
                gen_tsp(n, seed=int(i), path=str(inst_file))
            D = read_instance(inst_file)
            tour, dt, reused = label_instance(D, inst_file, tour_file)
            lkh["reused" if reused else "run"] += 1
            lkh["wall_total_s"] += dt
            gs.append(build_graph(D, tour, k))
            covs.append(tour_edge_coverage(D, tour, k))
        graphs[name] = gs
        coverages[name] = covs
        log.info("split %-5s: %d instances (seeds %d..%d), LKH kNN(%d) coverage "
                 "mean %.4f", name, len(gs),
                 split_indices(cfg)[name].start, split_indices(cfg)[name].stop - 1,
                 k, float(np.mean(covs)))
    return graphs, {"coverage": coverages, "lkh": lkh}


def get_dataloaders(cfg):
    """Build (and persist) the dataset. Returns ({split: HeatmapDataset}, meta).

    Callers wrap their own DataLoader (train shuffle, batch size) on the
    returned dataset objects.
    """
    t0 = time.perf_counter()
    graphs, meta = build_splits(cfg)
    meta["build_time_s"] = time.perf_counter() - t0
    datasets = {name: HeatmapDataset(gs) for name, gs in graphs.items()}
    lkh = meta["lkh"]
    avg_lkh = lkh["wall_total_s"] / lkh["run"] if lkh["run"] else 0.0
    log.info(
        "dataset: n=%d k=%d | train/val/test = %d/%d/%d | %d candidate edges/instance | "
        "LKH: %d run (avg %.3f s, %.1f s total), %d reused | build %.1f s",
        cfg["n"], cfg["k"], cfg["train_n"], cfg["val_n"], cfg["test_n"],
        cfg["n"] * cfg["k"], lkh["run"], avg_lkh, lkh["wall_total_s"],
        lkh["reused"], meta["build_time_s"])
    return datasets, meta


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(message)s",
                        datefmt="%H:%M")
    cfg = {"n": 20, "k": 8, "train_n": 10, "val_n": 5, "test_n": 5, "batch_size": 64}
    datasets, meta = get_dataloaders(cfg)
    for name in SPLIT_NAMES:
        d = datasets[name]
        g0 = d[0]
        log.info("%s: %d graphs | graph0 nodes=%d edges=%d pos=%d",
                 name, len(d), g0.x.shape[0], g0.edge_index.shape[1],
                 int(g0.y.sum()))
    return datasets, meta


if __name__ == "__main__":
    main()
