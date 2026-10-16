import sys, time
from pyvrp import read, solve
from pyvrp.stop import MaxRuntime

path, sec, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
data = read(path, round_func="round")                     # cùng quy ước làm tròn với CVRPLIB
t = time.perf_counter()
res = solve(data, stop=MaxRuntime(sec), seed=seed)
print(f"pyvrp,{path},{sec},{seed},{res.cost()},{res.is_feasible()},{time.perf_counter()-t:.1f}")
