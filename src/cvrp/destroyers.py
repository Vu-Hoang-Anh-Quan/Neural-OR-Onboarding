"""Destroy operators for CVRP.

The RNG is assumed to be seeded in another file (e.g., np.random.seed(...)
in the driver script), so this module only draws from the global state.
"""
import numpy as np

# Hardcoded exponent for the rank draw in related_removal.
_RELATED_P = 2


def random_removal(routes, k):
    """Randomly remove k customers from the given routes.

    Args:
        routes: list of routes, each route a list of customer indices
            (excluding the depot), served in that order.
        k: int, number of customers to remove; must satisfy k < num_customers.

    Returns:
        dict with keys:
            "routes": routes after removal (removed customers are simply
                dropped, not replaced; routes that lose all their customers
                are dropped entirely).
            "removed": list of the removed customer indices, in the order
                they appeared across the routes.
    """
    flat = [c for r in routes for c in r]
    assert k < len(flat), f"k={k} must be < num_customers={len(flat)}"
    picked = np.random.choice(len(flat), size=k, replace=False)
    to_remove = set(flat[i] for i in picked)
    new_routes = [[c for c in r if c not in to_remove] for r in routes]
    removed = [c for r in routes for c in r if c in to_remove]
    return {"routes": [r for r in new_routes if r], "removed": removed}


def worst_removal(inst, routes, k):
    """Remove the k customers with the largest removal contribution.

    The contribution of customer i, served between neighbours a and b
    (depot 0 when i is at the route boundary) is

        s_i = d(a,i) + d(i,b) - d(a,b)

    Boundaries are handled by the same formula: for the first/last customer
    of a route, a or b is the depot; for the only customer of a route,
    a = b = 0, so s_i = 2*d(0,i) (d(0,0) = 0), i.e. the full cost of the
    route it alone serves.

    Args:
        inst: dict returned by cvrp_utils.load() (needs "D").
        routes: list of routes, each route a list of customer indices
            (excluding the depot), served in that order.
        k: int, number of customers to remove; must satisfy k < num_customers.

    Returns:
        dict with keys:
            "routes": routes after removal (routes emptied by the removals
                are dropped entirely).
            "removed": the removed customer indices, in route order.
    """
    flat = [c for r in routes for c in r]
    assert k < len(flat), f"k={k} must be < num_customers={len(flat)}"
    D = inst["D"]
    # Contribution of every customer, computed once.
    contrib = {}
    for r in routes:
        n = len(r)
        for j, i in enumerate(r):
            a = r[j - 1] if j > 0 else 0
            b = r[j + 1] if j < n - 1 else 0
            contrib[int(i)] = int(D[a, i] + D[i, b] - D[a, b])
    top = sorted(contrib, key=contrib.get, reverse=True)[:k]
    to_remove = set(top)
    new_routes = [[c for c in r if c not in to_remove] for r in routes]
    removed = [c for r in routes for c in r if c in to_remove]
    return {"routes": [r for r in new_routes if r], "removed": removed}


def related_removal(inst, routes, k):
    """Remove k customers related to a randomly drawn seed customer.

    A seed customer i is chosen uniformly at random from all customers and
    counts as the first removal. Then k - 1 further removals are drawn: at
    each step, all not-yet-removed customers are ranked by distance to the
    current seed (lowest first; the depot is excluded), and one customer is
    removed at rank r = floor(u^p * m) (0-indexed), where u ~ U(0,1), p is
    the hardcoded constant _RELATED_P, and m is the current number of
    not-yet-removed customers. The customer just removed becomes the seed
    for the next step's ranking, so the removals form a chain hopping from
    seed to nearby customer.

    Args:
        inst: dict returned by cvrp_utils.load() (needs "D").
        routes: list of routes, each route a list of customer indices
            (excluding the depot), served in that order.
        k: int, number of customers to remove; must satisfy k < num_customers.

    Returns:
        dict with keys:
            "routes": routes after removal (routes emptied by the removals
                are dropped entirely).
            "removed": the removed customer indices, in route order.
            "seed": the first seed customer, chosen uniformly at random.
    """
    flat = [c for r in routes for c in r]
    assert k < len(flat), f"k={k} must be < num_customers={len(flat)}"
    D = inst["D"]

    # The first seed customer is removed first; it counts toward k.
    seed0 = int(np.random.choice(flat))
    seed = seed0
    to_remove = {seed}
    # The removals form a chain: at each step, rank the not-yet-removed
    # customers by distance to the current seed (lowest first), draw a rank,
    # and the drawn customer becomes the seed for the next step.
    for _ in range(k - 1):
        pool = [c for c in flat if c not in to_remove]
        ranked = sorted(pool, key=lambda c: D[seed, c])
        m = len(ranked)
        u = np.random.random()
        r = int(np.floor(u ** _RELATED_P * m))
        seed = int(ranked[r])
        to_remove.add(seed)

    new_routes = [[c for c in r if c not in to_remove] for r in routes]
    removed = [c for r in routes for c in r if c in to_remove]
    return {"routes": [r for r in new_routes if r], "removed": removed, "seed": seed0}
