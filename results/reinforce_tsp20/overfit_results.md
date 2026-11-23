# REINFORCE TSP-20 — overfit diagnostic

Both variants replay the SAME 5 fixed instances (`inst_42001` … `inst_42005`, ``data/reinforce_tsp20/overfit/``) as every batch — effective batch = 5 tours per Adam step, ``--batch-size`` ignored — under the same one-step-per-batch batched REINFORCE rules as the spec run (spec relaxation of 2026-11-22): epochs = 50, batches/epoch = 100, lr = 0.001, B = 0.5, policy seeds A=0 / B=1000000, device = cuda. Adam steps per variant = 5000 (25000 tours). Judged against these 5 instances' references (LKH mean 165.0, NN mean 314.4, NN+2-opt mean 253.4); the P4 probe reference curve reached a per-tour mean of 601.4 at 10k tours (lr 1e-4).

| instance | LKH-3 | NN | NN+2-opt | init greedy | moving_avg greedy | greedy_rollout greedy |
|---|---|---|---|---|---|---|
| inst_42001 | 176 | 421 | 286 | 970 | 964 | 880 |
| inst_42002 | 152 | 334 | 271 | 776 | 906 | 1033 |
| inst_42003 | 170 | 266 | 266 | 1130 | 891 | 857 |
| inst_42004 | 165 | 223 | 223 | 849 | 1194 | 1097 |
| inst_42005 | 162 | 328 | 221 | 1026 | 1259 | 1063 |
| **mean** | 165.0 | 314.4 | 253.4 | 950.2 | **1042.8** | **986.0** |

- **moving_avg**: final-epoch mean return -1042.8 ± 0.0; best-epoch mean -914.2 (epoch 7); greedy mean over the 5 instances 1042.8 vs LKH mean 165.0.
- **greedy_rollout**: final-epoch mean return -986.1 ± 2.1; best-epoch mean -916.4 (epoch 7); greedy mean over the 5 instances 986.0 vs LKH mean 165.0, promotions = 3 over 50 epochs, final L_base = 855.0.

## Runtimes

| stage | wall (s) |
|---|---|
| train moving_avg | 110.7 |
| train greedy_rollout | 198.2 |
| LKH-3 labels (5 instances) | 0.1 |

## Environment

- 2.11.0+cu128 (CUDA available: True), device used: cuda; Python 3.11.17

## Curves

- `moving_avg_overfit_returns_curve.png` — epoch mean return over the replayed batches (band = ±1 std over the epoch's batch returns).
- `greedy_rollout_overfit_returns_curve.png` — epoch mean return over the replayed batches (band = ±1 std over the epoch's batch returns).