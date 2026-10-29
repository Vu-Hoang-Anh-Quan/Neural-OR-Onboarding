"""Simple constructive heuristics for CVRP."""
import numpy as np
from cvrp_utils import check


def sweep(inst):
    """Sweep heuristic.

    Args:
        inst: dict returned by cvrp_utils.load().

    Returns:
        A readable route (list of routes, each route a list of customer
        indices excluding the depot), as implied by cvrp_utils.
    """
    n = len(inst["demand"]) - 1
    customers = np.arange(1, n + 1)
    Q = inst["Q"]
    assert inst["demand"][1:].max() <= Q, "customer demand exceeds capacity Q"

    # 1. Sort customers by atan2 of their coords relative to the depot.
    rel = inst["xy"][1:] - inst["xy"][0]
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    order = customers[np.argsort(angles)]  # keep original customer indices

    # 2-3. Sweep in sorted order; keep adding customers to the current route
    # while it stays feasible, otherwise start a new route.
    routes = []
    current = []
    load = 0
    for c in order:
        if inst["demand"][c] > Q or load + inst["demand"][c] > Q:
            routes.append(current)
            current, load = [], 0
        current.append(int(c))
        load += inst["demand"][c]
    routes.append(current)

    # 5. Solution must pass the feasibility check before returning.
    ok, msg = check(routes, inst)
    assert ok, f"sweep produced infeasible solution: {msg}"
    return routes
