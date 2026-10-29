"""Quick verification of repairers (run from src/cvrp/)."""
import numpy as np

from repairers import greedy_insertion, regret_2_insertion, _insertion_slots
from cvrp_utils import check, cost

# =====================================================================
# 1. greedy_insertion: hand-built D, costs traced by hand.
#
#   D =                     depot far from 1 (5), close to 2 (3), 3 (4);
#   [0  5  3  4]            1-2 = 1-3 = 2-3 = 1 (a tight triangle).
#   [5  0  1  1]
#   [3  1  0  1]
#   [4  1  1  0]
#
# Start empty, insert {1,2,3}. Greedy picks the globally cheapest slot each
# step:
#   step 1: only new-route slots.  2*5=10, 2*3=6, 2*4=8  -> insert 2  => [[2]]
#   step 2: c1 best 3, c3 best 2   -> insert 3 at pos0 => [[3,2]]
#   step 3: c1 best 1 (between 3 and 2) => [[3,1,2]]
#
# Final: [[3,1,2]], cost = d03+d31+d12+d20 = 4+1+1+3 = 9.
# =====================================================================
D1 = np.array([
    [0, 5, 3, 4],
    [5, 0, 1, 1],
    [3, 1, 0, 1],
    [4, 1, 1, 0],
], dtype=int)
inst1 = {"D": D1, "demand": np.array([0, 1, 1, 1]), "Q": 10}
out = greedy_insertion(inst1, [], [1, 2, 3])
assert out == [[3, 1, 2]], out
assert cost(out, D1) == 9, cost(out, D1)
print("greedy hand-built ok (step-by-step cheapest slot)")

# =====================================================================
# 2. regret_2_insertion vs greedy: a D that distinguishes the two.
#
#   D =
#   [0    1   1   1]
#   [1    0  100   1]     1-2 is huge (100); everything else is 1.
#   [1  100   0   1]
#   [1    1   1   0]
#
# Start with routes=[[1]], insert {2,3}.
#   c=2: slots  pos0=100, pos1=100, new=2   -> best 2, second 100, regret 98
#   c=3: slots  pos0=1,   pos1=1,   new=2   -> best 1, second 1,   regret 0
#   regret_2 picks the highest regret (c=2, regret 98) -> opens new route.
#   greedy  picks the lowest best cost (c=3, best 1)   -> into route [1].
#
# regret_2 => [[3,1],[2]]  cost 5
# greedy   => [[2,3,1]]    cost 4
# =====================================================================
D2 = np.array([
    [0,   1,   1,  1],
    [1,   0, 100,  1],
    [1, 100,   0,  1],
    [1,   1,   1,  0],
], dtype=int)
inst2 = {"D": D2, "demand": np.array([0, 1, 1, 1]), "Q": 10}

g = greedy_insertion(inst2, [[1]], [2, 3])
assert g == [[2, 3, 1]], g
assert cost(g, D2) == 4, cost(g, D2)

in_routes = [[1]]
snapshot = [list(r) for r in in_routes]
r2 = regret_2_insertion(inst2, in_routes, [2, 3])
assert r2 == [[3, 1], [2]], r2
assert cost(r2, D2) == 5, cost(r2, D2)
assert g != r2, "regret_2 should differ from greedy here"
assert in_routes == snapshot, "input routes were mutated"
print("regret_2 vs greedy distinguishable ok (highest regret first)")

# =====================================================================
# 3. tie-breaking: identical regret AND identical best cost -> lowest index.
#
# Two customers, empty routes, equal depot distance (5). Each has exactly one
# feasible slot (a new route) => regret inf for both, best cost 10 for both.
# The still-tied case resolves to the lowest customer index (1 goes in first,
# becoming the backbone; 2 is then slotted in front of it).
#   D =
#   [0 5 5]
#   [5 0 1]
#   [5 1 0]
# Final: [[2,1]], cost = 5+1+5 = 11.
# =====================================================================
D3 = np.array([
    [0, 5, 5],
    [5, 0, 1],
    [5, 1, 0],
], dtype=int)
inst3 = {"D": D3, "demand": np.array([0, 1, 1]), "Q": 10}
out3 = regret_2_insertion(inst3, [], [1, 2])
assert out3 == [[2, 1]], out3   # 1 inserted first (lower index), 2 in front
assert cost(out3, D3) == 11, cost(out3, D3)
print("regret tie (regret + best cost + index) ok")

# =====================================================================
# 4. capacity-forced new route.
# Q = 2. Cust 1 has demand 2 (fills [1]); cust 2 and 3 have demand 1.
#   c=2 cannot enter [1] (load 2), so it must open a new route (single
#   feasible slot); c=3 then fits into that new route (1+1 = 2 <= Q).
#   D: all off-diagonal 1.
# Final: [[1],[3,2]], cost = 2 + 3 = 5.
# =====================================================================
D4 = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
], dtype=int)
demand4 = np.array([0, 2, 1, 1])
inst4 = {"D": D4, "demand": demand4, "Q": 2}
out4 = greedy_insertion(inst4, [[1]], [2, 3])
assert out4 == [[1], [3, 2]], out4
assert cost(out4, D4) == 5, cost(out4, D4)
out4b = regret_2_insertion(inst4, [[1]], [2, 3])
assert out4b == [[1], [3, 2]], out4b
ok, msg = check(out4, inst4)
assert ok, (msg, out4)
print("capacity-forced new route ok")

# =====================================================================
# 5. _insertion_slots structure (the load-bearing invariants).
#    (a) new-route slot uses route_idx == len(routes); delta == 2*d(0,c).
#        inst1, routes=[[2]], c=3: in-route pos0/pos1 delta 2,2 (route 0),
#        new slot (8, 1, 0) with route_idx = len([[2]]) = 1.
#    (b) a full route yields no in-route slots; only the new-route one.
#        inst4, routes=[[1]] (load 2 == Q), c=2: only (2, 1, 0).
# =====================================================================
s5a = _insertion_slots(inst1, [[2]], 3)
assert sorted(s5a) == [(2, 0, 0), (2, 0, 1), (8, 1, 0)], s5a
s5b = _insertion_slots(inst4, [[1]], 2)
assert s5b == [(2, 1, 0)], s5b
print("_insertion_slots structure ok")

# =====================================================================
# 6. real instance round trip: sweep -> destroy (two stages) -> repair.
#    The kept partial routes and the combined removed list must partition all
#    customers; both repairers then pass check() and keep cost finite.
# =====================================================================
from cvrp_utils import load
from constructive_heuristics import sweep
from destroyers import worst_removal, random_removal

inst_real = load("../../data/Chosen/X-n101-k25.vrp")
n_total = len(inst_real["demand"]) - 1
full = sweep(inst_real)
ok_full, _ = check(full, inst_real)
assert ok_full, "sweep should be infeasible"

r1 = worst_removal(inst_real, full, k=25)
r2 = random_removal(r1["routes"], k=15)
partial = r2["routes"]
removed = list(r1["removed"]) + list(r2["removed"])

kept = set(c for r in partial for c in r)
rem = set(removed)
assert not (kept & rem), "kept and removed must be disjoint"
assert kept | rem == set(range(1, n_total + 1)), (len(kept), len(rem))

for name, fn in (("greedy", greedy_insertion), ("regret_2", regret_2_insertion)):
    completed = fn(inst_real, partial, removed)
    ok, msg = check(completed, inst_real)
    assert ok, (name, msg, completed)
    c = cost(completed, inst_real["D"])
    assert isinstance(c, int) and c >= 0, (name, c)
    print(f"real instance round trip ({name}) ok: n={n_total}, cost={c}")

print("ALL OK")
