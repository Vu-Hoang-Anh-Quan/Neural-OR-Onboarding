# TSP heatmap GNN results (random ATSP, n = 20, k-NN k = 8)

Supervised edge classifier (GNN, hidden 64) for the NC20/NC10 'heatmap' idea: for each random asymmetric TSP instance (LKH-3 tour as label), one PyG graph per instance whose edges are the directed k nearest out-neighbors (k = 8, ~160 candidate edges) and whose node features are the distance-matrix rows D[i, :]/100. The model classifies each candidate directed edge (i, j) as in-tour or not (positive rate 20/160 = 12.5%).
Instances: 40 train / 20 val / 20 test, one global index i with seed = i (no shared seeds). Model selection on val top-20 F1 (fixed budget; the @0.5 threshold F1 is degenerate here — see Interpretation in markdown/tsp_heatmap_instructions.md); test evaluated once on the best-val checkpoint.
Environment: Python 3.11.17, torch 2.11.0+cu128, PyG 2.8.1, NumPy 2.4.6.

## Test metrics (n = 20 instances)

| metric | avg ± std | Q1 | median | Q3 |
|--------|-----------|----|--------|----|
| precision (p ≥ 0.5) | avg 0.0000 ± 0.0000 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| recall (p ≥ 0.5) | avg 0.0000 ± 0.0000 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| F1 (p ≥ 0.5) | avg 0.0000 ± 0.0000 | Q1 0.0000 | median 0.0000 | Q3 0.0000 |
| precision (top-20) | avg 0.1300 ± 0.0292 | Q1 0.1000 | median 0.1500 | Q3 0.1500 |
| recall (top-20) | avg 0.1307 ± 0.0292 | Q1 0.1000 | median 0.1500 | Q3 0.1500 |
| F1 (top-20, fixed budget) | avg 0.1303 ± 0.0292 | Q1 0.1000 | median 0.1500 | Q3 0.1500 |
| k-NN tour-edge coverage | avg 0.9950 ± 0.0150 | Q1 1.0000 | median 1.0000 | Q3 1.0000 |

Per-instance test: min precision 0.0000 / min recall 0.0000 / min F1 0.0000 / min top-20 precision 0.1000 / min top-20 recall 0.1000 / min top-20 F1 0.1000 / coverage range [0.9500, 1.0000].

## Training runtimes

| quantity | value |
|----------|-------|
| LKH-3 labeling | 40 run, 40 reused; 0.016 s/instance averaged over runs |
| dataset build (gen + label + k-NN) | 0.7 s |
| training | 0.1 s over 3 epochs, avg 0.012 s/epoch |
| per-epoch val inference (over 3 epochs, incl.) | 0.0 s |

## Test inference runtime

| quantity | value |
|----------|-------|
| test inference | 0.0 s over 20 instances, avg 0.0003 s/instance |

## Configuration

| key | value |
|-----|-------|
| n | 20 |
| k | 8 |
| train_n | 40 |
| val_n | 20 |
| test_n | 20 |
| hidden | 64 |
| dropout | 0.1 |
| epochs | 3 |
| batch_size | 64 |
| lr | 0.001 |
| weight_decay | 0.1 |
| seed | 42 |
| device | cpu |
| topk | 20 |

## Hardware specification
CPU: Ryzen 7 9700X
GPU: RTX 5060 Ti
