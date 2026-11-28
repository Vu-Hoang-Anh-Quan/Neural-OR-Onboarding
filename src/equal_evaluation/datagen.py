"""Build the two test sets for the equal-budget evaluation (run once by the
orchestrator; everything under `results/equal_evaluation/` is reused on resume).

(a) CVRP-50 set: 1000 fixed instances generated from the RL4CO CVRP-50
    distribution. Instance #i is drawn with `torch.manual_seed(1234 + i)`
    immediately before `generator([1])` (global torch RNG — the generator has no
    per-instance hook), serialized as an EUC_2D `.vrp` with raw demands
    (`round(demand * CAPACITY)` on the float32 td demand — NOT `round(float32
    raw)`, which breaks for the large-capacity X-scale instances) and
    `CAPACITY : 40`, coordinates on the x1e6 integer scale. The generator tds are
    kept as a torch.save so the POMO eval can be rehydrated without regeneration
    (regenerating 500 more instances per X-scale run would cost ~2 min of RNG).
(b) X set: the 10 `data/chosen/*.vrp` (+ `.sol` best-known-cost) instances,
    recorded with name / dim / cap / demmax / bks. X .vrp files keep the [0,1000]
    integer coordinates and are NOT rescaled.

Scale constant (one place): CVRP-50 .vrp coordinates = td * 1e6 (nint scale,
float32 td). POMO reward L_cmp for CVRP-50 instances = (-reward) * 1e6.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvrp.cvrp_utils import load  # noqa: E402  (independent nint-scale parse, shared)

from rl4co.envs.routing.cvrp.generator import CVRPGenerator  # noqa: E402

# --- constants (orchestrator re-exports its own copy) -------------------------
GENERATOR_SEED_OFFSET = 1234     # torch.manual_seed(1234 + i) before each gen([1])
CVRP_SCALE = 1e6                 # one place: CVRP-50 .vrp coordinate scale (nint convention)
CAPACITY = 40                    # Kool et al. table capacity for num_loc=50 -> raw scalar

GEN_TD_FILENAME = "cvrp50_generator_tds.pt"   # kept for resume; ~30 MB
RECORD_CSV = "x_records.csv"

# X ground truth from the spec appendix (independent check against the .sol files):
# name -> (dim, cap, demmax, bks)
X_EXPECTED = {
    "X-n101-k25": (101, 206, 100, 27591),
    "X-n153-k22": (153, 144, 98, 21220),
    "X-n204-k19": (204, 836, 100, 19565),
    "X-n251-k28": (251, 69, 10, 38684),
    "X-n275-k28": (275, 10, 1, 21245),
    "X-n298-k31": (298, 55, 10, 34231),
    "X-n351-k40": (351, 436, 100, 25896),
    "X-n401-k29": (401, 745, 100, 66154),
    "X-n449-k29": (449, 777, 100, 55233),
    "X-n502-k39": (502, 13, 1, 69226),
}


def _raw_demands_from_td(td, capacity: float) -> list[int]:
    """Raw integer demands from a generator td (td["demand"] [1, N] normalized).

    round(td_demand * capacity), where capacity = the Kool-table raw scalar
    (40 for num_loc=50): the td demand is exactly raw/capacity in float32
    (generator side), so the product lands within half-a-float of the integer
    raw demand and .round() recovers it exactly.
    """
    norm = td["demand"][0]
    return [int(v) for v in (norm * capacity).round().tolist()]


def generate_cvrp50(n: int, out_dir: Path, log) -> dict:
    """Generate n CVRP-50 instances; write .vrp files + generator-tds pt file."""
    generator = CVRPGenerator(num_loc=50)
    vrp_dir = out_dir / "cvrp50"
    vrp_dir.mkdir(parents=True, exist_ok=True)
    tds = []
    for i in range(n):
        torch.manual_seed(GENERATOR_SEED_OFFSET + i)
        td_batched = generator([1])                      # [1, ...] batched gen-td
        assert tuple(td_batched.batch_size) == (1,)
        tds.append(td_batched[0])                       # de-batched: locs [50,2], depot [2], demand [50], capacity [1]
        raw = _raw_demands_from_td(td_batched, CAPACITY)
        # td coords are float32 in [0,1]; the .vrp is written at the x1e6 nint scale
        # (one shared CVRP_SCALE constant), NOT [0,1] (which would round to 0/1).
        depot = [int(round(c * CVRP_SCALE)) for c in td_batched["depot"][0].tolist()]
        locs = [[int(round(c * CVRP_SCALE)) for c in row.tolist()] for row in td_batched["locs"][0]]
        path = vrp_dir / f"cvrp50_{i:04d}.vrp"
        with open(path, "w") as f:
            f.write("NAME : CVRP-50 fixed instance #%04d (RL4CO CVRP-50 distribution)\n" % i)
            f.write("TYPE : CVRP\n")
            f.write("DIMENSION : 51\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write(f"CAPACITY : {CAPACITY}\n")
            f.write("NODE_COORD_SECTION\n")
            f.write(f"1\t{depot[0]}\t{depot[1]}\n")
            for k, (x, y) in enumerate(locs, start=2):
                f.write(f"{k}\t{x}\t{y}\n")
            f.write("DEMAND_SECTION\n")
            f.write("1\t0\n")
            for k, d in enumerate(raw, start=2):
                f.write(f"{k}\t{d}\n")
            f.write("ROUTE_SECTION\n")
            f.write("1\n1\n")
        # Independent parse-back check: the nint distance matrix must be what the
        # solver subprocesses (and the reference) will see.
        # parse-back: round (not truncate) — 1e6-scaled float32 coords are not
        # exactly representable, so e.g. 488223.9999... must not truncate to 488223
        inst = load(str(path))
        assert inst["Q"] == CAPACITY
        got_demand = [int(v) for v in np.round(inst["demand"]).astype(float).tolist()]
        assert got_demand[1:] == raw, f"instance {i}: demand parse-back mismatch"
        got_depot = [int(v) for v in np.round(inst["xy"][0]).tolist()]
        assert got_depot == depot, f"instance {i}: depot parse-back mismatch"
    full = torch.stack([t["locs"] for t in tds], 0)          # [n, 50, 2]
    depots = torch.stack([t["depot"] for t in tds], 0)        # [n, 2]
    demands = torch.stack([t["demand"] for t in tds], 0)      # [n, 50]
    capacities = torch.stack([t["capacity"] for t in tds], 0) # [n, 1]
    gen_td = TensorDict(
        {"locs": full, "depot": depots, "demand": demands, "capacity": capacities},
        batch_size=[n],
    )
    torch.save(gen_td, out_dir / GEN_TD_FILENAME)
    log.info("datagen cvrp50: %d instances -> %s (+ %s, generator tds kept)",
             n, vrp_dir, GEN_TD_FILENAME)
    return {"tds": gen_td}


def _bks_from_sol(path: Path) -> int:
    m = re.search(r"Cost\s+(\d+)", path.read_text())
    assert m, f"no 'Cost <int>' line in {path}"
    return int(m.group(1))


def load_x_set(results_dir: Path, chosen_dir: Path, log) -> list[dict]:
    """Parse the 10 X instances (keep [0,1000] coords; no rescale here).

    Returns per-instance records: name, file, bks, n (nodes), cap, demmax,
    customers_only_xy (n-1,2) int tensor, depot, raw_demands int tensor (n-1).
    Cross-checked against the spec appendix ground truth (dim/cap/demmax/bks).
    """
    (results_dir / RECORD_CSV).parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for name, (dim, cap, demmax, bks) in X_EXPECTED.items():
        vrp = chosen_dir / f"{name}.vrp"
        sol = chosen_dir / f"{name}.sol"
        assert vrp.is_file() and sol.is_file(), f"missing {vrp} or {sol}"
        inst = load(str(vrp))
        assert len(inst["demand"]) == dim, f"{name}: expected dim {dim}, got {len(inst['demand'])}"
        assert inst["Q"] == cap, f"{name}: expected cap {cap}, got {inst['Q']}"
        m = int(inst["demand"][1:].max())
        assert m == demmax, f"{name}: expected demmax {demmax}, got {m}"
        bks_actual = _bks_from_sol(sol)
        assert bks_actual == bks, f"{name}: .sol BKS {bks_actual} != appendix {bks}"
        rec = {
            "name": name,
            "file": str(vrp),
            "bks": bks,
            "n": dim,                       # nodes incl. depot
            "cap": cap,
            "demmax": demmax,
            "xy": torch.tensor(inst["xy"], dtype=torch.long),          # [n, 2] int [0,1000]
            "demands": torch.tensor(inst["demand"][1:], dtype=torch.long),  # [n-1] int raw
        }
        recs.append(rec)
    # records CSV: name, file, bks, n, cap, demmax (the spec's record list)
    with open(results_dir / RECORD_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "file", "bks", "n", "cap", "demmax"])
        for r in recs:
            w.writerow([r["name"], r["file"], r["bks"], r["n"], r["cap"], r["demmax"]])
    log.info("datagen x: %d instances from %s (BKS from .sol; appendix cross-check OK)",
             len(recs), chosen_dir)
    return recs
