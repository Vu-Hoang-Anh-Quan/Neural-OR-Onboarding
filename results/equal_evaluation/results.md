# Equal-budget evaluation — POMO vs PyVRP vs OR-Tools (CVRP-50 + X set)

## Environment

- Python 3.11.17
- torch 2.11.0+cu128 (CUDA available: True)
- rl4co 0.7.0
- lightning 2.6.6
- tensordict 0.15.6
- pyvrp 0.14.0
- ortools 9.15.6755
- vrplib 2.2.0
- matplotlib 3.11.2
- numpy 2.4.6

## Hardware

- platform Linux-7.0.0-44-generic-x86_64-with-glibc2.43
- nproc 16
- `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader`: NVIDIA GeForce RTX 5060 Ti, 16311 MiB, 15835 MiB
- device used: **cuda**

## Tables (gap vs reference, %)

### CVRP-50 set (1000 instances × 3 passes)

| method | greedy (0.03 s) | sampling/1280 (0.05 s) | 1 s | 10 s |
|---|---|---|---|---|
| POMO | +20.72 ± 5.95 | +12.94 ± 2.56 | — | — |
| PyVRP | +3.19 ± 2.79 | +2.76 ± 2.67 | +0.07 ± 0.25 | +0.01 ± 0.10 |
| OR-Tools | — | — | +5.87 ± 3.68 | +2.95 ± 2.20 |

- Empty cells (no committed row; logged): CVRP-50 ['POMO @ 1 s', 'POMO @ 10 s', 'OR-Tools @ 0.03 s', 'OR-Tools @ 0.05 s']; X set ['POMO @ 1 s', 'POMO @ 10 s', 'OR-Tools @ 0.03 s', 'OR-Tools @ 0.05 s']. (OR-Tools sub-second cells can be empty on rc=1 `AttributeError`.)

### X set (10 instances × 3 passes)

| method | greedy (0.03 s) | sampling/1280 (0.05 s) | 1 s | 10 s |
|---|---|---|---|---|
| POMO | +87.11 ± 62.37 | +79.91 ± 41.54 | — | — |
| PyVRP | +5.20 ± 1.87 | +4.85 ± 1.73 | +2.68 ± 1.20 | +0.78 ± 0.51 |
| OR-Tools | — | — | +10.70 ± 6.19 | +7.58 ± 4.39 |

*Plots: `gap_time_cvrp50.png`, `gap_time_x.png`.*

## Configuration

- Columns: the 3 measured POMO-mode times (greedy = 0.03 s, dihedral-8 = 0.03 s, sampling/1280 = 0.05 s), then the fixed `1 s` and `10 s`. POMO fills only its measured columns (its budget is the decode time, not wall-clock; its 1 s / 10 s cells are blank and drawn as a dotted horizontal extension in the plots).
- References: CVRP-50 → per-instance PyVRP 60 s (seed = 1); X set → BKS from the `.sol` `Cost` lines.
- Cell = mean ± population std (ddof=0) of the pooled per-instance gap values (1000 × 3 passes / 10 × 3 passes).
- **±1 note** (once): POMO tours are evaluated in [0,1] float and scaled ×1e6 / ×1000 to the coordinate scale; the solver costs use the nint integer convention. A ±1 unit offset between the two is expected, not a bug.
- **OR-Tools std ≈ 0**: the CLI has no seed argument, so its re-runs are identical deterministic solves; the pooled std is ~0 by construction.
- **POMO-on-X off-distribution caveat**: POMO was trained on [0,1]² uniform demand-1..10 Q=40; the X instances are far off-distribution, so its large X gaps (e.g. +260 % greedy on n502) report genuine distribution shift, not a bug — the PyVRP ≈ BKS cross-check is the X table's validity anchor.
- **POMO greedy / dihedral passes 2-3 are byte-identical repeats**: those modes are deterministic per instance, so their 3 passes pool as repeats (only the sampling pass varies, seeded 1000/1001/1002).

### Seed summary

| Where | Seed(s) | Feeds |
|---|---|---|
| CVRP-50 instance #i | `torch.manual_seed(1234 + i)` | `CVRPGenerator(num_loc=50)([1])` per instance (global torch RNG) |
| POMO sampling budget, pass s | `torch.manual_seed(1000 + s)`, s ∈ {0,1,2} | the sampling start-action RNG |
| PyVRP budget runs | int `seed` arg, 3 seeds (1, 2, 3) | `run_pyvrp.py` CLI → `pyvrp.solve(seed=...)` |
| OR-Tools budget runs | none (CLI has no seed arg) | 3 identical GLS re-runs; std ≈ 0 (documented, not a bug) |
| CVRP-50 reference (PyVRP 60 s) | single fixed seed = 1 | per-instance reference target, not the randomized object |