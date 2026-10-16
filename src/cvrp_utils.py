"""Tiện ích CVRP: đọc instance, ma trận khoảng cách, cost, kiểm tra khả thi.
Quy ước: node 0 là depot; lời giải là list route, mỗi route là list khách (không chứa depot)."""
import numpy as np, vrplib

def load(path):
    inst = vrplib.read_instance(path)                      # tự tính edge_weight cho EUC_2D
    D = np.floor(inst["edge_weight"] + 0.5).astype(int)    # quy ước nint của CVRPLIB
    return {"name": inst["name"], "D": D, "demand": np.asarray(inst["demand"]),
            "Q": int(inst["capacity"]), "xy": np.asarray(inst["node_coord"])}

def route_cost(r, D):
    return int(D[0, r[0]] + sum(D[a, b] for a, b in zip(r, r[1:])) + D[r[-1], 0]) if r else 0

def cost(routes, D):
    return sum(route_cost(r, D) for r in routes)

def check(routes, inst):
    """(khả_thi, thông_báo): mỗi khách đúng một lần, không route nào vượt tải."""
    n = len(inst["demand"]) - 1
    if sorted(c for r in routes for c in r) != list(range(1, n + 1)):
        return False, "thiếu hoặc lặp khách hàng"
    for r in routes:
        if inst["demand"][r].sum() > inst["Q"]:
            return False, f"route vượt tải: {inst['demand'][r].sum()} > {inst['Q']}"
    return True, "ok"

def gap(c, ref):
    return 100.0 * (c - ref) / ref
