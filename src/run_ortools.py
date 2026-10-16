import sys, time
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from cvrp_utils import load, cost, check

path, sec = sys.argv[1], int(sys.argv[2])
inst = load(path); D, dem, Q = inst["D"], inst["demand"], inst["Q"]
n_veh = len(dem) - 1
mgr = pywrapcp.RoutingIndexManager(len(dem), n_veh, 0)
rm = pywrapcp.RoutingModel(mgr)
dist_cb = rm.RegisterTransitCallback(lambda i, j: int(D[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
rm.SetArcCostEvaluatorOfAllVehicles(dist_cb)
dem_cb = rm.RegisterUnaryTransitCallback(lambda i: int(dem[mgr.IndexToNode(i)]))
rm.AddDimensionWithVehicleCapacity(dem_cb, 0, [Q] * n_veh, True, "Capacity")
p = pywrapcp.DefaultRoutingSearchParameters()
p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
p.time_limit.seconds = sec
t = time.perf_counter(); sol = rm.SolveWithParameters(p)
routes = []
for v in range(n_veh):
    i, r = rm.Start(v), []
    while not rm.IsEnd(i):
        node = mgr.IndexToNode(i)
        if node: r.append(node)
        i = sol.Value(rm.NextVar(i))
    if r: routes.append(r)
print(f"ortools,{path},{sec},-,{cost(routes, D)},{check(routes, inst)[0]},{time.perf_counter()-t:.1f}")
