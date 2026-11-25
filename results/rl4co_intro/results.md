# RL4CO intro — AM & POMO on TSP-50 / CVRP-50

## Header

GPU gate (mandatory pre-run `nvidia-smi`): free 1.0 / 15.9 GiB, 14.5 GiB used by: /usr/local/lib/ollama/llama-server [56244].
Device: `--device cpu` → **cpu** ('auto' resolves cuda-if-available, else cpu). Batch size: **512** (library default; only an observed OOM would justify lowering it — none occurred).

What was trained: Attention Model (AM, Kool et al. 2019 REINFORCE, rollout baseline) and POMO (Kwon et al. 2020, `shared` baseline, multi-start over all feasible start nodes, 8× dihedral on val/test) from the `rl4co` 0.7.0 library, on TSP-50 (`TSPEnv` + `TSPGenerator(num_loc=50)`) and CVRP-50 (`CVRPEnv` + `CVRPGenerator(num_loc=50)`, capacity 40), with the **library-default hyperparameters**: optimizer Adam lr=1e-4 (**`lr_scheduler=None`** — no LR scheduler, the library's real default; none added), gradient clip 1.0, precision 16-mixed (requested; on CPU Lightning resolves `16-mixed` to `bf16-mixed` — fp16 AMP is unsupported on CPU, so the runtime AMP dtype here is bfloat16), batch 512, 10 epochs × `ceil(5000/512)` optimizer steps, `max_steps=500` as a total-step ceiling. Reward = **−objective** (negative tour / route length), so returns rise toward 0; **length = −return**. Checkpoint protocol: best epoch-mean **training** return (per-step `train/reward` values from the callback's `outputs` capture, equal-weight mean; see checkpoint protocol below). Data sizes are the only deliberate deviations from the library defaults (see Configuration); nothing else was tuned.

Checkpoint protocol: the per-run Lightning `Callback` captures per-batch `train/reward` / `val/reward` (already the batch-mean tensors from `RL4COLitModule.log_metrics`); at each epoch it writes the epoch mean/std to the curve CSV and saves the **best-peak-epoch** checkpoint — metric = epoch-mean **training** return (equal-weight mean of that epoch's per-step `train/reward`), epoch 1 always saves, later epochs only on strict improvement. Checkpoints are state-dict-only `{"epoch", "global_step", "state_dict"}` (rl4co's `load_from_checkpoint` is broken on torch 2.11.0: `weights_only=None` → `weights_only=True` cannot unpickle hparams with pickled envs/baselines); evaluation reloads them into a fresh model with identical env/kwargs via `model.setup(stage="fit")` (which already runs `post_setup_hook()`) + strict `load_state_dict` — **never** the live end-of-training weights.

Evaluation regime: one fixed test set of `test_data_size`=500 instances generated once from the env's generator (run seed `--seed`, slices of 100), decoded with `model(td, **{'phase': 'test', 'decode_type': 'greedy', 'num_starts': 1})` — **single-start greedy, no dihedral augmentation, for BOTH variants**. This is deliberately **not** POMO's default inference regime (multi-start over all feasible start nodes, 8× dihedral on val/test); it keeps the AM/POMO test comparison apples-to-apples.

## Best results

Columns: best epoch (chosen by epoch-mean training return) | train return mean ± std at that epoch → length ≈ −mean | val return mean ± std at that epoch → length | test greedy single-start mean ± std (length) | fit wall time (s).

| variant | problem | best epoch | train (mean ± std) → length | val (mean ± std) → length | test greedy (mean ± std) | fit wall (s) |
|---|---|---|---|---|---|---|
| AM | TSP-50 | 10 | -11.7299 ± 0.7468 → 11.73 | -9.0953 ± 0.0000 → 9.10 | 9.0845 ± 0.5644 | 110 |
| AM | CVRP-50 | 10 | -18.6350 ± 0.2600 → 18.64 | -15.7954 ± 0.0000 → 15.80 | 15.9063 ± 1.7734 | 122 |
| POMO | TSP-50 | 10 | -7.5768 ± 0.0252 → 7.58 | -7.0260 ± 0.0000 → 7.03 | 6.7978 ± 0.4053 | 534 |
| POMO | CVRP-50 | 10 | -13.4894 ± 0.0680 → 13.49 | -12.6456 ± 0.0000 → 12.65 | 12.4429 ± 1.4113 | 685 |

## Configuration

| key | value |
|---|---|
| epochs | 10 |
| max_steps (total-step ceiling) | 500 |
| batch size | 512 (library default) |
| train/val/test sizes | 5000 / 500 / 500 — **only deliberate deviation** from library defaults 100 000 / 10 000 / 10 000 |
| optimizer | Adam, lr=1e-4 (library default) |
| lr scheduler | none (library default) |
| gradient clip | 1.0 (library default) |
| precision | 16 (library default) (requested; Lightning resolves `16-mixed` to `bf16-mixed` at runtime on CPU) |
| seed | 42 |
| policies | AM / POMO library policy defaults (POMO: `num_starts=None` → all feasible starts, 8× dihedral on val/test, SharedBaseline; AM: default warmup rollout baseline) |

## Environment

- Python 3.11.17
- torch 2.11.0+cu128 (CUDA available: True)
- lightning 2.6.6
- rl4co 0.7.0
- matplotlib 3.11.2

## Hardware

CPU: AMD Ryzen 7 9700X 8-Core Processor — 16 logical cores
GPU 0: NVIDIA GeForce RTX 5060 Ti — 15.3 / 15.5 GiB free
- device used: **cpu**

## Plots

- `tsp50_training_returns.png` — per-epoch mean of the per-step `train/reward` values (equal-weight mean over the epoch's steps), both variants
- `cvrp50_training_returns.png` — same for CVRP-50
- band = ±1 population std of that epoch's per-step returns (AM `tab:blue`, POMO `tab:orange`); curves: `curves/<variant>_<problem>50_curve.csv`

## Checkpoints

- `src/rl4co_intro/checkpoints/am_tsp50_best.pt` (state-dict-only `epoch` / `global_step` / `state_dict`)
- `src/rl4co_intro/checkpoints/am_cvrp50_best.pt` (state-dict-only `epoch` / `global_step` / `state_dict`)
- `src/rl4co_intro/checkpoints/pomo_tsp50_best.pt` (state-dict-only `epoch` / `global_step` / `state_dict`)
- `src/rl4co_intro/checkpoints/pomo_cvrp50_best.pt` (state-dict-only `epoch` / `global_step` / `state_dict`)
