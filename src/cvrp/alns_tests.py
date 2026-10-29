"""Quick verification of the ALNS (run from src/cvrp/: python alns_tests.py).

Keeps total runtime small: a small generated instance for the main checks and
one short run on a real Chosen instance.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))        # for gen
sys.path.insert(0, os.path.join(ROOT, "src", "cvrp"))  # for cvrp_utils, alns, ...

from gen import gen_cvrp
from cvrp_utils import load, check, cost
from constructive_heuristics import sweep
from alns import alns_solve, PAIRS

# =====================================================================
# 1. generated instance: n=30 customers, capacity 50.
#    Check structure (depot excluded, full customer coverage), feasibility
#    via check(), and that ALNS never does worse than the sweep baseline.
# =====================================================================
gen_path = os.path.join(ROOT, "data", "cvrp", "alns_test_n30_s0.vrp")
gen_cvrp(n=30, capacity=50, seed=0, path=gen_path)
inst = load(gen_path)
n_cust = len(inst["demand"]) - 1

baseline = sweep(inst)
base_cost = cost(baseline, inst["D"])
res = alns_solve(inst, time_limit=3.0, seed=1)
ok, msg = check(res, inst)
assert ok, msg
flat = [c for r in res for c in r]
assert len(flat) == len(set(flat)) == n_cust and 0 not in flat, flat
res_cost = cost(res, inst["D"])
assert res_cost <= base_cost, (res_cost, base_cost)
print(f"generated instance ok: sweep={base_cost}, alns={res_cost} (3 s budget)")

# =====================================================================
# 2. tiny time limit (< half-second reserve): the loop may run zero
#    iterations and must still return the feasible sweep solution.
#    Two runs with the same seed must be identical (no iterations at all
#    means the RNG state never moves).
# =====================================================================
r1 = alns_solve(inst, time_limit=0.1, seed=7)
r2 = alns_solve(inst, time_limit=0.1, seed=7)
ok, msg = check(r1, inst)
assert ok, msg
assert r1 == r2, "zero-iteration runs with the same seed should match"
print("tiny time limit ok (returns feasible sweep, deterministic)")

# =====================================================================
# 3. wall clock: the search must stop when the budget is up (give it a
#    generous slack, but it must not blow up).
# =====================================================================
t0 = time.perf_counter()
res2 = alns_solve(inst, time_limit=1.0, seed=2)
dt = time.perf_counter() - t0
ok, msg = check(res2, inst)
assert ok, msg
assert dt < 1.0 + 0.5, dt
print(f"wall clock ok: limit=1.0 s, actual={dt:.3f} s")

# =====================================================================
# 4. real instance: X-n101-k25, short budget. Same invariants as (1).
# =====================================================================
from cvrp_utils import load as _load
inst_real = _load(os.path.join(ROOT, "data", "Chosen", "X-n101-k25.vrp"))
base_real = cost(sweep(inst_real), inst_real["D"])
res_real = alns_solve(inst_real, time_limit=2.0, seed=3)
ok, msg = check(res_real, inst_real)
assert ok, msg
real_cost = cost(res_real, inst_real["D"])
assert len(PAIRS) == 6, len(PAIRS)
assert real_cost <= base_real, (real_cost, base_real)
print(f"real instance ok: sweep={base_real}, alns={real_cost} (2 s budget)")

print("ALL OK")
