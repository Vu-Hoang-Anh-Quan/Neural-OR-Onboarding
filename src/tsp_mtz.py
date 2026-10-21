import numpy as np
from ortools.linear_solver import pywraplp

def solve_tsp_mtz(D, time_limit_s=60):
    n = len(D) # [n, n]
    s = pywraplp.Solver.CreateSolver("SCIP")
    x = {(i, j): s.BoolVar(f"x_{i}_{j}") for i in range(n) for j in range(n) if i != j}
    u = {i: s.NumVar(1, n - 1, f"u_{i}") for i in range(1, n)}
    for i in range(n):
        s.Add(sum(x[i, j] for j in range(n) if j != i) == 1)
        s.Add(sum(x[j, i] for j in range(n) if j != i) == 1)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s.Add(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)
    s.Minimize(sum(D[i][j] * x[i, j] for (i, j) in x))
    s.SetTimeLimit(time_limit_s * 1000)
    status = s.Solve()
    tour, cur = [0], 0
    for _ in range(n - 1):
        cur = next(j for j in range(n) if j != cur and x[cur, j].solution_value() > 0.5)
        tour.append(cur)
    return s.Objective().Value(), tour, status == pywraplp.Solver.OPTIMAL

if __name__ == "__main__":
    rng = np.random.default_rng(0); xy = rng.random((10, 2))
    print(solve_tsp_mtz(np.linalg.norm(xy[:, None] - xy[None], axis=-1)))
