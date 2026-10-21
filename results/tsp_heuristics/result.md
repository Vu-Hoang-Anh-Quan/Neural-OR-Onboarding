# TSP heuristics results

Random uniform TSP instances in `[0,1]^2`, generated with a single NumPy RNG seeded `42` (only this seed is used); 20 instances per n (instance index 0..19).

- Heuristics: **NN** = nearest neighbor (start 0); **2opt** = first-improvement 2-opt; **Oropt** = alternating 2-opt + Or-opt (1-3 node segments) until no improvement.
- Reference for the gap: **MIP optimum** (MTZ / SCIP, from `tsp_mtz.py`, 60s limit) for n = 12; **LKH-3** (`data/LKH-3.0.14`) for the other n.
- Costs are Euclidean tour lengths on `[0,1]^2`. LKH is run on integer coordinates scaled by 1000000 (TSPLIB `EUC_2D`, `nint` rounding); its cost is divided back by the scale.
- Gap (%) = `100 * (heuristic - reference) / reference`.
- Time (s) is the wall-clock time of the full NN + 2-opt + Or-opt pipeline.
- Environment: Python 3.11.17, NumPy 2.4.6, OR-Tools 9.15.6755, LKH-3.0.14.
- Hardware: Ryzen 9 7900X 8 cores, no GPU used.

| n | NN gap | NN+2opt gap | NN+2opt+Oropt gap | time (s) |
|---|--------|-------------|-------------------|----------|
| 12 | 10.79 | 0.98 | 0.40 | 0.0003 |
| 20 | 17.89 | 3.45 | 0.64 | 0.0014 |
| 50 | 23.32 | 5.82 | 3.46 | 0.0102 |
| 100 | 26.47 | 5.50 | 3.27 | 0.0644 |
| 200 | 25.82 | 6.94 | 3.98 | 0.3601 |

Gaps are averaged over the 20 instances of each n. Per-instance values (costs, per-stage times, MIP optimality flags) are in `log.csv`.
