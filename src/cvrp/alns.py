"""ALNS (Adaptive Large Neighborhood Search) for CVRP.

The search is a destroy-repair loop: at each iteration, pick one of the
(destroyer, repairer) pairs adaptively (probability proportional to the
sigmoid of the pair's score), destroy k customers from the current solution,
repair with the pair's repairer, and decide whether to keep the new solution
with a simulated-annealing rule. The pair's score is then updated as an
exponential moving average of the last delta.

Public entry point: alns_solve(inst, time_limit, seed).

Sign convention (per spec): delta = cost(current) - cost(new), so
delta > 0 means the new solution is an improvement and delta < 0 means a
worsening move. The annealing acceptance probability for a worsening move
(i.e. when the move is not better than current) is exp(delta / temp), which
is < 1 in that case.
"""
import math
import time

import numpy as np

from cvrp_utils import check, cost
from constructive_heuristics import sweep
from destroyers import random_removal, worst_removal, related_removal
from repairers import greedy_insertion, regret_2_insertion

# Constant p in the score update s_i = p * delta + (1 - p) * s_i.
_P = 0.5
# Stop the search once the remaining budget drops below this many seconds.
_RESERVE = 0.5
# Destroy/repair pairs (3 destroyers x 2 repairers). random_removal has a
# different signature (no inst), so it is wrapped in _destroy below.
PAIRS = [
    (random_removal, greedy_insertion),
    (random_removal, regret_2_insertion),
    (worst_removal, greedy_insertion),
    (worst_removal, regret_2_insertion),
    (related_removal, greedy_insertion),
    (related_removal, regret_2_insertion),
]


def _avg_customer_distance(inst):
    """Mean distance over unordered pairs of distinct customers (depot excluded)."""
    C = np.asarray(inst["D"][1:, 1:], dtype=float)
    n = C.shape[0]
    if n < 2:
        return 0.0
    np.fill_diagonal(C, 0.0)
    return float(C.sum()) / (n * (n - 1) / 2.0)  # # of unordered pairs


def _sigmoid(x):
    """Logistic function, computed without overflow for any finite x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)  # < 1, so no overflow
    return e / (1.0 + e)


def _pick_pair(scores):
    """Choose a pair index i with probability sigmoid(s_i) / sum_j sigmoid(s_j)."""
    w = np.array([_sigmoid(s) for s in scores])
    total = w.sum()
    if total <= 0.0:
        # Every score was below ~ -745, far out in the underflow tail: fall
        # back to a uniform draw (the true probabilities are indistinguishable
        # from uniform at that scale anyway).
        w = np.full(len(scores), 1.0)
        total = float(len(scores))
    return int(np.random.choice(len(scores), p=w / total))


def _destroy(inst, fn, routes, k):
    """Call a destroyer; random_removal does not take inst."""
    if fn is random_removal:
        return fn(routes, k)
    return fn(inst, routes, k)


def alns_solve(inst, time_limit, seed):
    """Solve a CVRP instance with ALNS for up to time_limit seconds.

    Args:
        inst: dict returned by cvrp_utils.load() (needs "D", "demand", "Q").
        time_limit: budget in seconds. The search checks the remaining
            budget at the top of every iteration, so it stops as soon as
            fewer than 0.5 s remain and returns the best solution found so
            far. If no routes could be generated at all, an error is raised
            instead (sweep is called first unconditionally, so this only
            triggers on a broken instance).
        seed: int, seeds the global numpy RNG (the destroyers draw from it
            too). Same seed + same number of iterations => same sequence.

    Returns:
        The best routes found (list of routes, each a list of customer
        indices, excluding the depot), guaranteed to pass cvrp_utils.check.
    """
    np.random.seed(seed)
    t0 = time.perf_counter()
    elapsed = lambda: time.perf_counter() - t0

    # Initial solution: sweep (always produced; the only "no routes" path).
    best = sweep(inst)
    if not best:
        raise RuntimeError("alns_solve: no routes could be generated")
    cur = [list(r) for r in best]
    best_cost = cost(best, inst["D"])
    cur_cost = best_cost
    avg_dist = _avg_customer_distance(inst)
    scores = [0.0] * len(PAIRS)

    n_cust = len(inst["demand"]) - 1
    k_lo = max(1, int(math.ceil(0.10 * n_cust)))
    k_hi = max(k_lo, int(math.floor(0.30 * n_cust)))

    while time_limit - elapsed() >= _RESERVE:
        i = _pick_pair(scores)
        destroy, repair = PAIRS[i]
        k = int(np.random.randint(k_lo, k_hi + 1))

        out = _destroy(inst, destroy, cur, k)
        new = repair(inst, out["routes"], out["removed"])
        new_cost = cost(new, inst["D"])
        delta = cur_cost - new_cost  # > 0: improvement, < 0: worsening

        if new_cost < best_cost:
            best, best_cost = [list(r) for r in new], new_cost
        if new_cost < cur_cost:
            cur, cur_cost = new, new_cost
        else:
            # Simulated annealing; temperature from the spec, evaluated at
            # the moment of consideration so it cools as the budget runs out.
            temp = k * avg_dist / 2.0 * (1.0 - elapsed() / time_limit)
            if temp > 0 and np.random.random() < math.exp(delta / temp):
                cur, cur_cost = new, new_cost

        scores[i] = _P * delta + (1 - _P) * scores[i]

    ok, msg = check(best, inst)
    assert ok, f"alns_solve returned infeasible solution: {msg}"
    return best


if __name__ == "__main__":
    import sys
    from cvrp_utils import load
    alns_solve(load(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
