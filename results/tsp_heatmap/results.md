# TSP heatmap GNN results (random ATSP, n = 20, k-NN k = 8)

Supervised edge classifier (GNN, hidden 64) for the NC20/NC10 'heatmap' idea: for each random asymmetric TSP instance (LKH-3 tour as label), one PyG graph per instance whose edges are the directed k nearest out-neighbors (k = 8, ~160 candidate edges) and whose node features are the distance-matrix rows D[i, :]/100. The model classifies each candidate directed edge (i, j) as in-tour or not (positive rate 20/160 = 12.5%).
Instances: 1000 train / 1000 val / 1000 test, one global index i with seed = i (no shared seeds). Model selection on val F1 (threshold 0.5); test evaluated once on the best-val checkpoint.
Environment: Python 3.11.17, torch 2.15.0+cpu, PyG 2.8.1, NumPy 2.4.6.

## Test metrics (n = 1000 instances)

| metric | avg ± std | Q1 | median | Q3 |
|--------|-----------|----|--------|----|
| precision (p ≥ 0.5) | avg 0.0065 ± 0.0788 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| recall (p ≥ 0.5) | avg 0.0004 ± 0.0042 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| F1 (p ≥ 0.5) | avg 0.0007 ± 0.0079 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| precision (top-20) | avg 0.1952 ± 0.0317 | Q1 0.2000 | median 0.2000 | Q3 0.2000 |
| recall (top-20) | avg 0.1954 ± 0.0315 | Q1 0.2000 | median 0.2000 | Q3 0.2000 |
| k-NN tour-edge coverage | avg 0.9984 ± 0.0088 | Q1 1.0000 | median 1.0000 | Q3 1.0000 |

Per-instance test: min precision 0.0000 / min recall 0.0000 / min F1 0.0000 / min top-20 precision 0.1000 / min top-20 recall 0.1000 / coverage range [0.9500, 1.0000].

## Training runtimes

| quantity | value |
|----------|-------|
| LKH-3 labeling | 0 run, 3000 reused; 0.000 s/instance averaged over runs |
| dataset build (gen + label + k-NN) | 2.0 s |
| training | 192.5 s over 500 epochs, avg 0.139 s/epoch |
| per-epoch val inference (over 500 epochs, incl.) | 122.8 s |

## Test inference runtime

| quantity | value |
|----------|-------|
| test inference | 0.2 s over 1000 instances, avg 0.0002 s/instance |

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
| batch_size | 64 |
| lr | 0.001 |
| weight_decay | 0.0 |
| seed | 42 |
| device | cpu |
| topk | 20 |

## Hardware specification
CPU: Ryzen 7 9700X
GPU: Not used (CPU-only torch)
