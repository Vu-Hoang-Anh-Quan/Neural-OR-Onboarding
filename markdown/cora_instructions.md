# Cora GCN — plan & handoff notes

Goal: classify Cora node topics with a 2-layer GCN (standard Planetoid split); log everything; report test accuracy.
Run from repo root: `conda activate nor && python src/cora/train.py [--epochs 200] [--lr 0.01] [--seed 42] [--device auto]`

## Overview & Invariants
- Env is `nor` (Python 3.11). Needs `torch` + `torch_geometric` (CPU torch is fine; Cora is tiny). Record versions in the Result section (and keep in sync below).
- No manual Cora install: first run downloads raw files (~2 MB, internet once) into `data/cora/`, which is the Planetoid root — `data/cora/` holds both raw and processed. All else in `data/` is gitignored, same as the rest.
- Canonical Kipf split via masks: 140 train / 500 val / 1000 test. Model is selected on **val** accuracy only; **test** is evaluated once at the end.
- Layout (updated by user): all task files live in `src/cora/`:
  - `src/cora/dataset.py` — Planetoid load + preprocess (dense features, row L2-norm) + train/val/test DataLoaders. PyG's `DataLoader` rejects `batch_size=-1`, so each loader is `DataLoader([data], batch_size=1)` → one batch-per-epoch = the whole graph; split chosen by mask at train/eval time.
  - `src/cora/model.py` — `GCN(num_features=1433, hidden=64, num_classes=7)`; forward returns logits. ⚠️ Store the drop rate as `self.dropout_p`, never `self.dropout` (shadows `nn.Module.dropout` and crashes in forward).
  - `src/cora/train.py` — training loop; best-val checkpoint → `src/cora/checkpoints/cora_gcn.pt`; logs → `results/cora/log.txt` (also printed to terminal via StreamHandler). Bootstraps `sys.path` with the `src/` dir (parents[1]) to reach `utils/logging_utils`.
  - `src/utils/logging_utils.py` — `setup_logging(log_dir)` = FileHandler + StreamHandler; do not reinvent logging
- Defaults: Adam lr 0.01, weight_decay 5e-4, 200 epochs, seed 42, cross-entropy loss.
- Repo is not a package (no `__init__.py`): `src/cora/train.py` bootstraps `sys.path` with the `src/` dir at the top of the file. Don't restructure into packages.

## Step-by-step workflow
- [x] 1. Write this plan.
- [x] 2. Install deps in `nor`: `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install torch_geometric` (torch 2.15.0+cpu, torch_geometric 2.8.1).
- [x] 3. Write `src/cora/dataset.py`, `src/cora/model.py`, `src/cora/train.py` per the invariants above.
- [x] 4. Smoke test: `python src/cora/train.py --epochs 1` — checked download, log lines (terminal + `results/cora/log.txt`), checkpoint file created.
- [x] 5. Full run: `python src/cora/train.py` — `FINAL TEST ACCURACY` logged; `src/cora/checkpoints/cora_gcn.pt` exists.
- [x] 6. Fill in the Result section below.

## Result
test acc = **0.8090** (best val acc = 0.8060 @ epoch 24, seed 42, Adam lr 0.01, wd 5e-4, 200 epochs, CPU) · python 3.11.17 · torch 2.15.0+cpu · torch_geometric 2.8.1
Log: `results/cora/log.txt` · Checkpoint: `src/cora/checkpoints/cora_gcn.pt`
