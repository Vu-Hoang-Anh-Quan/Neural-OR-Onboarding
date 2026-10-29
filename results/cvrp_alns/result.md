# ALNS results (CVRP)

ALNS (`src/cvrp/alns.py`) on the 10 chosen CVRPLIB instances (`data/chosen`), 60s per run, 3 random seeds (4, 6, 8).
Reference for the gap: best-known solution (BKS) from each instance's `.sol` file (a reference, not a proof of optimality).
Costs use the CVRPLIB `nint` rounding convention (`cvrp_utils.load`).
Environment: Python 3.11.17, NumPy 2.4.6.

| instance | BKS | avg cost ± std | gap % |
|----------|-----|----------------|-------|
| X-n101-k25 | 27591 | 31182.0 ± 198.8 | 13.02 |
| X-n153-k22 | 21220 | 24116.0 ± 282.2 | 13.65 |
| X-n204-k19 | 19565 | 21848.0 ± 51.0 | 11.67 |
| X-n251-k28 | 38684 | 43475.0 ± 257.1 | 12.38 |
| X-n275-k28 | 21245 | 23643.3 ± 107.0 | 11.29 |
| X-n298-k31 | 34231 | 41271.7 ± 393.5 | 20.57 |
| X-n351-k40 | 25896 | 30670.7 ± 279.1 | 18.44 |
| X-n401-k29 | 66154 | 72498.0 ± 503.0 | 9.59 |
| X-n449-k29 | 55233 | 64884.3 ± 338.8 | 17.47 |
| X-n502-k39 | 69226 | 72653.3 ± 276.8 | 4.95 |

## Hardware specification
CPU: Ryzen 9 7900 X
GPU: Not used