# TSP heatmap GNN results (random ATSP, n = 20, k-NN k = 8)

Supervised edge classifier (GNN, hidden 64) for the NC20/NC10 'heatmap' idea: for each random asymmetric TSP instance (LKH-3 tour as label), one PyG graph per instance whose edges are the directed k nearest out-neighbors (k = 8, ~160 candidate edges) and whose node features are the distance-matrix rows D[i, :]/100. The model classifies each candidate directed edge (i, j) as in-tour or not (positive rate 20/160 = 12.5%).
Instances: 1000 train / 1000 val / 1000 test, one global index i with seed = i (no shared seeds). Model selection on val top-20 F1 (fixed budget; the @0.5 threshold F1 is degenerate here — see Interpretation in markdown/tsp_heatmap_instructions.md); test evaluated once on the best-val checkpoint.
Environment: Python 3.11.17, torch 2.11.0+cu128, PyG 2.8.1, NumPy 2.4.6.

## Test metrics (n = 1000 instances)

| metric | avg ± std | Q1 | median | Q3 |
|--------|-----------|----|--------|----|
| precision (p ≥ 0.5) | avg 0.0000 ± 0.0000 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| recall (p ≥ 0.5) | avg 0.0000 ± 0.0000 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| F1 (p ≥ 0.5) | avg 0.0000 ± 0.0000 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| precision (top-20) | avg 0.1944 ± 0.0323 | Q1 0.2000 | median 0.2000 | Q3 0.2000 |
| recall (top-20) | avg 0.1946 ± 0.0321 | Q1 0.2000 | median 0.2000 | Q3 0.2000 |
| F1 (top-20, fixed budget) | avg 0.1945 ± 0.0322 | Q1 0.2000 | median 0.2000 | Q3 0.2000 |
| k-NN tour-edge coverage | avg 0.9984 ± 0.0088 | Q1 1.0000 | median 1.0000 | Q3 1.0000 |

Per-instance test: min precision 0.0000 / min recall 0.0000 / min F1 0.0000 / min top-20 precision 0.0500 / min top-20 recall 0.0500 / min top-20 F1 0.0500 / coverage range [0.9500, 1.0000].

## Training runtimes

| quantity | value |
|----------|-------|
| LKH-3 labeling | 0 run, 3000 reused; 0.000 s/instance averaged over runs |
| dataset build (gen + label + k-NN) | 2.0 s |
| training | 294.9 s over 500 epochs, avg 0.039 s/epoch |
| per-epoch val inference (over 500 epochs, incl.) | 275.4 s |

## Test inference runtime

| quantity | value |
|----------|-------|
| test inference | 0.6 s over 1000 instances, avg 0.0006 s/instance |

## Configuration

| key | value |
|-----|-------|
| n | 20 |
| k | 8 |
| train_n | 1000 |
| val_n | 1000 |
| test_n | 1000 |
| hidden | 64 |
| dropout | 0.1 |
| epochs | 500 |
| batch_size | 256 |
| lr | 0.001 |
| weight_decay | 0 |
| seed | 42 |
| device | cuda |
| topk | 20 |

## Hardware specification
CPU: Ryzen 7 9700X
GPU: RTX 5060 Ti
