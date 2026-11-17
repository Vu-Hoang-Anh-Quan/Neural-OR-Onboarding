# TSP heatmap GNN — plan & handoff notes

Goal: build a k-NN (k=8) edge heatmap with a GNN on random TSP-20, using LKH-3 tours as labels; classify each candidate edge "in tour or not". Report precision/recall (avg ± std, Q1, Q3, median) and the fraction of true tour edges that fall inside the top-k graph (the "heatmap" idea of NC20 / NC10).
Run from repo root: `conda activate nor && python src/tsp_heatmap/orchestrator.py [--train-n 1000 --val-n 1000 --test-n 1000] [--n 20] [--k 8] [--hidden 64] [--epochs 50] [--batch-size 64] [--lr 1e-3] [--seed 42] [--device auto]`

## Overview & Invariants
- Env is `nor` (Python 3.11). Needs `torch` + `torch_geometric` (CPU torch is fine; n=20 is tiny). Record versions in the Result section.
- Instances are **random asymmetric TSP-20** from `gen.tsp` in `src/gen.py`: `gen_tsp(n=20, seed, path)` writes a plain file — line 1 is `n`, then a full n×n **asymmetric** integer matrix (1..100, zero diagonal). **Read it manually** (`n = int(f.readline())` then parse rows); do **not** `np.loadtxt` the whole file (column-count error).
- **Labels = LKH-3 tours.** LKH needs a TSPLIB file; ⚠️ on asymmetric data the type must be `TYPE : ATSP` (NOT `TSP` — TSP misreads an explicit matrix and reports a wrong length even when the tour is plausible). Also set `EDGE_WEIGHT_TYPE : EXPLICIT` and `EDGE_WEIGHT_FORMAT : FULL_MATRIX`. par file: `PROBLEM_FILE = <abs .tsp>`, `OUTPUT_TOUR_FILE = <abs .tour>`, `RUNS = 1`, `MAX_TRIALS = 1000`, `SEED = 1`. Invoke `subprocess.run([LKH_BIN, par], cwd=<data/LKH-3.0.14>)`. Parse length with `re.search(r"Length\s*=\s*(\d+)", tour_raw)`; tour = ints after the `TOUR_SECTION` line until `-1` (one per line, 1-based → 0-based); directed tour edges = `(tour[t], tour[(t+1)%n])`. **Label gate:** the tour must be a permutation of 0..n-1 with no repeated directed edge, and `lkh_len` must equal the **forward or backward** tour sum — probe (Held-Karp vs LKH, 12 cases n≤10) shows LKH's reported length is exact in every case, but the orientation in which the tour is listed is arbitrary (matched neither orientation: 0/12). If `lkh_len == bwd`, reverse the tour and re-write the tour file so `{i}.tour` is persisted in the length-matching orientation (keeps the reuse path simple: re-gate is `fwd == lkh_len`). Binary `data/LKH-3.0.14/LKH` is the LKH-3 build.
- **k-NN graph is directed (out-neighbors).** For node i, out-neighbors = the k=8 smallest `D[i, :]` excluding self. Candidate edge set = these directed (i → j) edges, ~160 per instance. Node features = row of D, `x_i = D[i, :]/100` (20-dim).
- **top-k tour-edge coverage** (the "fraction of real tour edges in the k-NN graph"): per instance, fraction of the 20 directed tour edges (i, j) whose j is among i's k=8 out-neighbors; reported as avg ± std, Q1, Q3, median over the split. This is the **upper bound on recall** — recall can never exceed it.
- Precision/recall: candidate pool = k-NN edges. Primary = threshold 0.5 on the sigmoid heatmap edge probabilities. Secondary = fixed-budget top-20 predicted edges per instance (matches tour size 20). Both reported as avg ± std, Q1, Q3, median over instances.
- Model selection on **val top-20 F1** only (fixed-budget metric — see the "model selection" note below for why the @0.5 threshold F1 must not be used); **test** evaluated once at the end. Best-val checkpoint → `src/tsp_heatmap/checkpoints/tsp_heatmap.pt` (mirrors `src/cora/checkpoints/cora_gcn.pt`). **Model-selection note (why not @0.5 F1):** the trained model's sigmoid probabilities sit near the 12.5% class prior (mean ≈ 0.11–0.14, max ≈ 0.2 even at 500 epochs), so almost no edge crosses 0.5 and the per-instance @0.5 P/R/F1 are 0.0 (fallback) for essentially every epoch of every run. With a `best = -inf` and strict `>` comparison, a degenerate selection metric therefore keeps the **first** epoch's checkpoint — a 50-epoch rerun on 2026-11-17 did exactly this: val F1@0.5 stayed 0.0000 for all 50 epochs, the epoch-1 (least-trained) checkpoint was saved as "best" and it **overwrote** the good epoch-474 checkpoint plus the definitive `results.md`. Fix: select on val top-20 F1 (never degenerate — the top-20 set is always exactly 20 edges; metric is well above the 0.125 chance floor from a few epochs on and tracks training progress); the @0.5 threshold F1 is still logged per epoch and reported in the results, it just no longer drives selection.
- Layout: all task files live in `src/tsp_heatmap/` (no `__init__.py`, not a package):
  - `src/tsp_heatmap/dataset.py` — generate 1000/1000/1000 TSP-20 into `data/tsp_heatmap/{train,val,test}/<i>.txt` (one global index i ∈ [0,3000), split by range; seed = i so splits never collide), run LKH → persist each tour (skip re-labeling if already present, i.e. idempotent), build the directed k-NN graph + one PyG `Data` per instance (features, edge_index, per-edge label), return `{split: HeatmapDataset}` (plain list of `Data`; `InMemoryDataset` is unusable) + metadata — `train.py` wraps the train set in the PyG `DataLoader` itself.
  - `src/tsp_heatmap/model.py` — GNN (hidden 64) node encoder over the k-NN graph + an edge head producing one logit per candidate (i, j) edge; forward returns edge logits.
  - `src/tsp_heatmap/train.py` — training loop (BCE on candidate edges), best-val by top-20 F1 (threshold-0.5 F1 is degenerate here, see invariant note above), save checkpoint, log config/versions/per-epoch.
  - `src/tsp_heatmap/evaluate.py` — threshold-0.5 and top-20 precision/recall + top-k coverage, each as avg ± std / Q1 / Q3 / median, plus measured inference wall time.
  - `src/tsp_heatmap/orchestrator.py` — the only `setup_logging(log_dir)` caller (root `basicConfig` wins once); bootstraps `sys.path` with the `src/` dir (parents[1]) to reach `gen` and `utils/logging_utils`; owns the CLI; drives dataset → train → evaluate → writes `results/tsp_heatmap/results.md`.
- Logging: `src/utils/logging_utils.py` → `setup_logging("results/tsp_heatmap")` = FileHandler + StreamHandler → `results/tsp_heatmap/log.txt` **mirrored to the terminal**. Log the date, python/package versions, full config, LKH labeling summary (count, avg wall), per-epoch train/val, and all test metrics + runtimes. All other modules use **named loggers** (`logging.getLogger("tsp_heatmap.dataset")`, …).
- Record **average training and inference runtimes separately** (LKH labeling avg/instance + total; train total + avg/epoch; test inference total). Hardware to record: **CPU Ryzen 7 9700 X, GPU Not used** (run is CPU-only: `torch 2.15.0+cpu`, CUDA not built — step 7 is the open GPU option).

## Step-by-step workflow
- [x] 1. Write this plan.
- [x] 2. POC: LKH-3 round-trip on `gen_tsp` ATSP/EXPLICIT matrices — 5/5 instances (n=20) satisfy `lkh_len == recomputed`; avg k-NN(8) tour-edge coverage 0.980; avg LKH wall ~0.01 s/instance. (See `#2` notes: `TYPE : ATSP` is the fix for the asymmetry.) Later probe (n ≤ 10 × 4 seeds, exact Held-Karp): `lkh_len == opt` in 12/12; listed-tour orientation matches the forward sum 6/12 and the backward sum 6/12 — hence the orientation-robust gate in `dataset.label_instance` (line 8/9 above).
- [x] 3. Write `src/tsp_heatmap/dataset.py` + `src/tsp_heatmap/model.py` per the invariants (gen + LKH label + k-NN + PyG data + GNN edge classifier).
- [x] 4. Write `src/tsp_heatmap/train.py` + `src/tsp_heatmap/evaluate.py` (train loop, F1 selection, metrics + runtimes).
- [x] 5. Write `src/tsp_heatmap/orchestrator.py` (CLI, `sys.path` bootstrap, single `setup_logging`, orchestration, `results.md` writer).
- [x] 6. Smoke test: tiny split — gen → LKH → label → data → train → checkpoint written, log lines on terminal and in `results/tsp_heatmap/log.txt`; all passed (see log blocks 15:34 / 21:01 in `results/tsp_heatmap/log.txt`).
- [ ] 7. (Open, user informed) GPU: `nor` has `torch 2.15.0+cpu` even though the RTX 5060 Ti is attached (~13.7 GB VRAM already in use by something). Fix if GPU wanted: `conda activate nor && pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132`. CPU is perfectly fine for n=20, so this is optional. **Told the user; not run unless asked.**
- [x] 8. Full run (definitive: 1000/1000/1000, 500 epochs, lr 1e-3, run of 22:48) — best-val checkpoint saved (val F1 0.0008 @ epoch 474); test metrics + top-k coverage + runtimes logged + test evaluated once (see Result section).
- [x] 9. Fill in the Result section below and write `results/tsp_heatmap/results.md` (metrics, runtimes, hardware, versions).

## Result
Definitive run: 1000/1000/1000 (train/val/test), 500 epochs, Adam(lr=1e-3, wd=0), bs=64, seed=42, device=cpu. Environment: Python 3.11.17, torch 2.15.0+cpu, PyG 2.8.1, NumPy 2.4.6. Hardware: CPU Ryzen 7 9700 X, GPU Not used (run is CPU-only).

### Test metrics (1000 instances, best-val checkpoint: val F1 0.0008 @ epoch 474)
| metric | avg ± std | median |
|--------|-----------|--------|
| precision @0.5 | 0.0065 ± 0.0788 | 0.0 |
| recall @0.5 | 0.0004 ± 0.0042 | 0.0 |
| F1 @0.5 | 0.0007 ± 0.0079 | 0.0 |
| precision top-20 | 0.1952 ± 0.0317 | 0.200 |
| recall top-20 | 0.1954 ± 0.0315 | 0.200 |
| k-NN tour-edge coverage | 0.9984 ± 0.0088 | 1.0 |

Chance top-20 precision = 20/160 = 0.125; the k-NN graph covers 99.84% of true tour edges — recall is bounded by coverage, not by the model.

### Interpretation
The model is stuck near the class prior at threshold 0.5 but carries a **real, above-chance ranking signal**:
- Training: loss 0.5146 (ep 1) → ~0.372 plateau (ep ~100) → 0.3711 (ep 500). Prior-only BCE 0.3764 (empirical val rate 0.1248): the model goes below the prior but only by ~0.005.
- Best-val checkpoint (val F1 @0.5): 0.0008 (only 20 val edges predicted ≥ 0.5, few of them correct) — threshold metrics are nearly zero by construction.
- Val probes on the same checkpoint (diag, 160,000 val edges): mean sigmoid on tour edges 0.1371 vs 0.1250 on non-tour (ratio 1.096); correlation(y, p) = 0.109 (epoch-1 checkpoint had 0.068 → ~40% relative improvement); top-20 val precision 0.1944, consistent with the 0.1952 reported on test.
- Conclusion: with hidden-64 GCN on k=8 out-NN graphs, the learnable signal is a weak preference for tour edges, visible only in ranking (top-20) and separation, not in calibrated probabilities. Next levers if pursued: longer training / higher lr, label smoothing, or a calibration step before reporting @0.5 metrics.

### Runtimes (run of 22:48, wall 194.8 s)
| quantity | value |
|----------|-------|
| dataset build (LKH: 0 run, all 3000 reused) | 2.0 s |
| training (500 epochs, CPU) | 192.5 s (0.385 s/epoch incl. per-epoch val inference) |
| test inference (1000 instances) | 0.2 s |

### Files
- `results/tsp_heatmap/results.md` — formal metrics + runtimes (written by `orchestrator.py` at the end of the run; overwritten on rerun).
- `results/tsp_heatmap/log.txt` — appended per-run history (smoke 15:34, small full 21:47, definitive 22:48).
- `src/tsp_heatmap/checkpoints/tsp_heatmap.pt` — best-val checkpoint (epoch 474). **`src/tsp_heatmap/checkpoints/` is gitignored** — the ckpt ships only locally.
- Open items: GPU swap (step 7, needs user go-ahead); optional follow-up with higher lr / more epochs if the @0.5 degradation is to be reduced; `results/tsp_heatmap/` files changed → user reviews and commits.
Log: `results/tsp_heatmap/log.txt` · Checkpoint: `src/tsp_heatmap/checkpoints/tsp_heatmap.pt`
