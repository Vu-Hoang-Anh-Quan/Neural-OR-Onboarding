"""Repair operators for CVRP: re-insert removed customers into partial routes.

Both functions take a partial solution (a list of routes that may be missing
customers) and the list of removed customers, and return the completed routes.
An insertion position is either inside an existing route (between two
consecutive customers, or at either end next to the depot) or in a brand-new
singleton route.
"""
import numpy as np

from cvrp_utils import check


def _insertion_slots(inst, routes, c):
    """Feasible insertion slots for customer c.

    Returns a list of (delta, route_idx, pos) where delta is the increase of
    the total cost caused by inserting c in that slot. route_idx == len(routes)
    marks the special "open a new route" slot (pos is 0 there). A slot inside
    a route is feasible iff the route has enough remaining capacity for c; the
    new-route slot is feasible iff demand[c] <= Q.
    """
    D = inst["D"]
    Q = inst["Q"]
    q = int(inst["demand"][c])
    slots = []
    for ri, r in enumerate(routes):
        if int(inst["demand"][r].sum()) + q > Q:
            continue
        n = len(r)
        for pos in range(n + 1):
            a = r[pos - 1] if pos > 0 else 0
            b = r[pos] if pos < n else 0
            slots.append((int(D[a, c] + D[c, b] - D[a, b]), ri, pos))
    if q <= Q:
        slots.append((int(D[0, c] + D[c, 0]), len(routes), 0))
    return slots


def _apply_insert(routes, route_idx, pos, c):
    if route_idx == len(routes):
        routes.append([c])
    else:
        routes[route_idx].insert(pos, c)


def _precondition(inst):
    demand = np.asarray(inst["demand"], dtype=int)
    assert demand[1:].max() <= inst["Q"], "a customer has demand > capacity"


def greedy_insertion(inst, routes, removed):
    """Greedy (cheapest) insertion repair.

    At each step, compute the cheapest insertion slot (lowest cost increase)
    for every customer not yet inserted, and insert the customer whose
    cheapest slot has the lowest cost (ties: lowest customer index).

    Args:
        inst: dict returned by cvrp_utils.load().
        routes: partial routes (list of routes, each a list of customer
            indices excluding the depot). Not mutated.
        removed: customer indices to insert back; must be disjoint from the
            customers currently in routes.

    Returns:
        Completed routes (list of routes) that pass cvrp_utils.check.
    """
    _precondition(inst)
    routes = [list(r) for r in routes if r]
    to_insert = [int(c) for c in removed]
    assert len(to_insert) == len(set(to_insert)), "duplicate customers in removed"

    while to_insert:
        best = None  # (delta, c, (route_idx, pos))
        for c in to_insert:
            slots = _insertion_slots(inst, routes, c)
            assert slots, f"customer {c} has no feasible insertion slot"
            delta, ri, pos = min(slots, key=lambda s: s[0])
            if best is None or (delta, c) < (best[0], best[1]):
                best = (delta, c, (ri, pos))
        _, c, (ri, pos) = best
        _apply_insert(routes, ri, pos, c)
        to_insert.remove(c)

    ok, msg = check(routes, inst)
    assert ok, f"greedy_insertion produced infeasible solution: {msg}"
    return routes


def regret_2_insertion(inst, routes, removed):
    """Regret-2 insertion repair.

    At each step, find the two cheapest distinct insertion slots for every
    customer not yet inserted. The customer's regret is second_best - best;
    a customer with exactly one feasible slot has regret inf (it has no
    alternative and must be placed now). The customer with the highest regret
    is inserted; ties on regret go to the lowest best cost, then to the lowest
    customer index.

    Args:
        inst: dict returned by cvrp_utils.load().
        routes: partial routes (list of routes, each a list of customer
            indices excluding the depot). Not mutated.
        removed: customer indices to insert back; must be disjoint from the
            customers currently in routes.

    Returns:
        Completed routes (list of routes) that pass cvrp_utils.check.
    """
    _precondition(inst)
    routes = [list(r) for r in routes if r]
    to_insert = [int(c) for c in removed]
    assert len(to_insert) == len(set(to_insert)), "duplicate customers in removed"

    while to_insert:
        best = None  # (key, c, (delta, route_idx, pos))
        for c in to_insert:
            slots = sorted(_insertion_slots(inst, routes, c), key=lambda s: s[0])
            assert slots, f"customer {c} has no feasible insertion slot"
            regret = float("inf") if len(slots) < 2 else slots[1][0] - slots[0][0]
            key = (-regret, slots[0][0], c)
            if best is None or key < best[0]:
                best = (key, c, slots[0])
        _, c, (delta, ri, pos) = best
        _apply_insert(routes, ri, pos, c)
        to_insert.remove(c)

    ok, msg = check(routes, inst)
    assert ok, f"regret_2_insertion produced infeasible solution: {msg}"
    return routes
