import numpy as np, vrplib

def gen_cvrp(n=100, capacity=50, seed=0, path=None):
    rng = np.random.default_rng(seed)
    xy = rng.integers(0, 1001, size=(n + 1, 2))          # node 0 = depot
    demand = rng.integers(1, 10, size=n + 1); demand[0] = 0
    path = path or f"rand-n{n}-s{seed}.vrp"
    vrplib.write_instance(path, {
        "NAME": path[:-4], "TYPE": "CVRP", "DIMENSION": n + 1, "CAPACITY": capacity,
        "EDGE_WEIGHT_TYPE": "EUC_2D", "NODE_COORD_SECTION": xy,
        "DEMAND_SECTION": demand, "DEPOT_SECTION": np.array([1, -1])})
    return path
