"""Quick verification of destroyers (run from src/cvrp/)."""
import numpy as np
from destroyers import random_removal, worst_removal, related_removal

# --- random_removal: dict return; removing ALL customers of one route drops that route ---
routes = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
for seed in range(200):
    np.random.seed(seed)
    res = random_removal(routes, k=2)
    kept = [c for r in res["routes"] for c in r]
    assert len(kept) + len(res["removed"]) == 9
    assert not (set(kept) & set(res["removed"]))
    assert all(r != [] for r in res["routes"]), "empty route leaked into output"
    # each returned route must be a subsequence of its original route
    for r in res["routes"]:
        assert any(
            (lambda it=iter(_o): all(c in it for c in r))(_o) for _o in routes
        ), f"{r} is not a subsequence of any original route"

# forced: k equal to the length of one route, so a whole-route removal can occur
routes2 = [[1, 2], [3, 4], [5, 6, 7, 8]]
found_full_removal = False
for seed in range(200):
    np.random.seed(seed)
    res = random_removal(routes2, k=2)
    removed = set(res["removed"])
    for o in routes2:
        if set(o) <= removed and len(o) == 2:  # a whole 2-customer route removed
            found_full_removal = True
            for r in res["routes"]:
                assert set(o).isdisjoint(r) and o not in res["routes"], res
assert found_full_removal, "never hit a seed where a whole route is fully removed"
print("random_removal ok (whole-route removal drops the route)")

# --- worst_removal: hand-built D so contributions are verifiable ---
D = np.array([
    [0, 10, 10, 10, 10, 10],
    [10, 0, 1, 1, 1, 1],
    [10, 1, 0, 1, 1, 1],
    [10, 1, 1, 0, 1, 1],
    [10, 1, 1, 1, 0, 1],
    [10, 1, 1, 1, 1, 0],
], dtype=int)
inst = {"D": D, "demand": np.array([0, 1, 1, 1, 1, 1]), "Q": 10}
routes = [[1, 2, 3], [4]]
# contributions:
#   1 (first)     : d01 + d12 - d02 = 1
#   2 (middle)    : d12 + d23 - d13 = 1
#   3 (last)      : d23 + d30 - d20 = 1
#   4 (only cust. of its route): 2*d04 = 20  (a = b = 0, d_ab = 0)
res = worst_removal(inst, routes, k=1)
assert res["removed"] == [4] and res["routes"] == [[1, 2, 3]], res
res = worst_removal(inst, routes, k=3)
assert res["routes"] == [[3]], res
assert res["removed"] == [1, 2, 4], res  # top-3 set {4,1,2}, returned in route order
res = worst_removal(inst, [[1, 2, 3], [4, 5]], k=4)  # k = num_customers - 1 boundary
assert res["removed"] == [1, 2, 3, 4], res  # 1,2,3 (contrib 1) before 4 (20)? no: 4(20), then 1,2,3
assert res["routes"] == [[5]], res
print("worst_removal ok (boundary + depot-neighbour + sole-customer cases)")

# --- worst_removal on a real instance ---
from cvrp_utils import load
inst2 = load("../../data/Chosen/X-n101-k25.vrp")
r = [[c] for c in range(1, 6)]
res = worst_removal(inst2, r, k=4)
assert len(res["removed"]) == 4 and sum(len(x) for x in res["routes"]) == 1
print("worst_removal on real instance ok")

# --- related_removal: invariants, hand-built D ---
# In this D the depot (0) is far (10) from every customer, and all customer
# pairs are distance 1, so no distance-based assertion is possible; we instead
# check structural invariants that must hold for every seed, k < num_customers:
#   * kept + removed partitions all customers (5 total here);
#   * the seed is among the removed;
#   * no route is empty and every output route is a subsequence of an input
#     route (customers keep their within-route order);
#   * exactly k distinct customers are removed.
inst3 = {"D": D, "demand": np.array([0, 1, 1, 1, 1, 1]), "Q": 10}
r3 = [[1, 2, 3], [4, 5]]
for S in range(200):
    np.random.seed(S)
    res = related_removal(inst3, r3, k=3)
    kept = [c for ro in res["routes"] for c in ro]
    assert len(kept) + len(res["removed"]) == 5
    assert res["seed"] in res["removed"]
    assert not (set(kept) & set(res["removed"]))
    assert all(ro != [] for ro in res["routes"])
    # each output route is a subsequence of an original route
    for ro in res["routes"]:
        assert any(
            (lambda it=iter(o): all(c in it for c in ro))(o) for o in r3
        )
    # exactly k distinct customers removed, all valid customer indices
    assert len(set(res["removed"])) == 3, res
    assert 1 <= res["seed"] <= 5
print("related_removal invariants ok")

# exact precomputed cases (seed + rank fixed by the RNG draw sequence)
# With D6 every pair of distinct customers is equidistant (d=1), so with
# customers [1,2,3] the ranking by distance to the seed is just the natural
# order of the two non-seed customers; rank r_ (0-indexed, m=2) picks the
# r_-th of them. Precompute (seed, rank) for a few RNG seeds, then assert
# related_removal removes exactly {seed, others[rank]} and keeps the rest.
D6 = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
], dtype=int)
inst6 = {"D": D6, "demand": np.array([0, 1, 1, 1]), "Q": 6}
flat6 = [1, 2, 3]
pre = {}
for S in (1, 2, 3, 5, 8, 13):
    np.random.seed(S)
    s = int(np.random.choice(flat6))
    u = np.random.random()
    pre[S] = (s, int(np.floor(u ** 2 * (len(flat6) - 1))))  # m = 2
seen_ranks = set()
for S, (s, r_) in pre.items():
    np.random.seed(S)
    res = related_removal(inst6, [[c] for c in flat6], k=2)
    others = [c for c in flat6 if c != s]
    expected_removed = {s, others[r_]}
    assert res["seed"] == s, (res, S)
    assert set(res["removed"]) == expected_removed, (res, S, expected_removed)
    assert res["routes"] == [[c] for c in flat6 if c not in expected_removed]
    seen_ranks.add(r_)
    print(f"related_removal S={S}: seed={s} rank={r_} -> removed={res['removed']}")
assert {0, 1} <= seen_ranks, f"ranks not covered: {seen_ranks}"
print("related_removal exact case ok")

# k = 3 exact case exercising the seeded chain (the 2nd removal is ranked by
# the 1st removed customer, not the original seed). Distances are chosen so
# the old single-ranking semantics would remove {1,2,3} here, while the chain
# removes {1,2,4}:
#     D[1]=[1,0,1,5,9]  D[2]=[1,1,0,9,2]  D[3]=[1,5,9,0,1]  D[4]=[1,9,2,1,0]
D4 = np.array([
    [0, 1, 1, 1, 1],
    [1, 0, 1, 5, 9],
    [1, 1, 0, 9, 2],
    [1, 5, 9, 0, 1],
    [1, 9, 2, 1, 0],
], dtype=int)
inst4 = {"D": D4, "demand": np.array([0, 1, 1, 1, 1]), "Q": 10}
np.random.seed(43)
res = related_removal(inst4, [[1, 2], [3, 4]], k=3)
# rng: seed=1 (choice); u1 ~ 0.4969 -> r1 = floor(0.4969^2 * 3) = 0, pool
# {2,3,4} ranked [2,3,4] (d = 1,5,9) -> remove 2; u2 ~ 0.1029 -> r2 = 0,
# pool {3,4} ranked by d to 2: [4,3] (d = 2,9) -> remove 4.
assert res["seed"] == 1, res
assert res["removed"] == [1, 2, 4], res
assert res["routes"] == [[3]], res
print("related_removal k=3 chain ok")
print("ALL OK")
