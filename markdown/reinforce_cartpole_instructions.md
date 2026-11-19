# REINFORCE CartPole-v1 — plan & handoff notes

Goal: implement vanilla REINFORCE (policy gradient, on-policy, full episodes, reward-to-go) for CartPole-v1 in **two variants** — (a) **no baseline** and (b) a **moving-average baseline** — 5 seeds per variant; plot episode return (mean ± std over the 5 seeds) per variant; write CSV + results.md + one PNG. Report for each variant the **best result, at which episode, and the hardware used**.
Run from repo root: `conda activate nor && python src/reinforce_cartpole/orchestrator.py [--episodes 1000] [--seeds 0 1 2 3 4] [--lr 0.01] [--baseline 0.5] [--device auto]`

## Overview & Invariants
- Env is `nor` (Python 3.11.17). Already installed: `gymnasium 1.4.0`, `torch 2.11.0+cu128`, `numpy 2.4.6`, `matplotlib 3.11.2` (all in `requirements.txt`). Record versions in the Result section.
- Algorithm invariants (all versions, non-negotiable):
  - **gamma = 1.0** (no discount — episodes last at most 500 steps).
  - **Reward-to-go:** log-probability of the action taken at step t is weighted by `G_t = sum_{t' >= t} R_{t'}` (sum of the remaining rewards from t to the end of the episode, **including** `R_t`).
  - **Truncation:** `truncated=True` episodes (survived the default `max_length = 500`) are **kept as-is**: the episode ends there, its reward-to-go is computed and its gradient is **not** zeroed or sign-flipped. Every real step of the episode pays `R = +1.0`, including the final step — gymnasium's `TimeLimit` wrapper only flags `truncated=True` on step 500 and adds **no** terminal bonus — so a surviving 500-step episode has return exactly 500. `terminated` (pole fell) episodes are kept too. The training loop must **never** call `step()` after a `terminated=True` return (post-termination steps return 0.0 and are a no-op).
  - Baseline (variant b) updates **per episode, after the episode ends**: `baseline = baseline * (1 - B) + G_0 * B` with `B = 0.5` (CLI constant, default 0.5), `G_0` = the episode's total return (reward-to-go from the first step). Initial `baseline = 0.0`. The baseline is subtracted from reward-to-go in the loss, so it is a per-return constant: `(G_t - baseline)` for every step t of the episode.
  - Gradient: single-episode full batch, `policy.zero_grad()` → `loss.backward()` → `optimizer.step()` **every step of every episode** (classic REINFORCE).
  - Seeds: 5 per variant (default `0..4`); **every torch operation of an episode run is reproducible from `torch.manual_seed(seed)` set right before the loop for that seed** — each (variant, seed) run is independent so the same index across variants/seeds is comparable.
- Model (identical for both variants): `nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(dim=-1))` — obs 4-dim → 2 action probabilities; act with `Categorical(distribution).sample()`; log-prob = `d.log_prob(action)` (float32). Loss = `-(logprob * (G_t - baseline)).sum()` (positive = maximize), i.e. standard REINFORCE on reward-to-go.
- Env: `gym.make("CartPole-v1")` — default (non sutton-barto) reward: `+1.0` for every real step of the episode (the terminal step included), so **return = episode length** (≤ 500). Episodes end by termination (pole fell / cart hit the edge) or truncation (500 steps). Obs/action float32 tensors. No env modification anywhere: no `render_mode`, no custom `max_length`.
- Layout (no `__init__.py`, not a package — same rule as `src/cora/` / `src/tsp_heatmap/`):
  - `src/reinforce_cartpole/policy.py` — the MLP policy class (returns a `Categorical` or action+logprob).
  - `src/reinforce_cartpole/train.py` — `run_policy(seed, episodes, lr, baseline_b)` (runs ONE (variant, seed) full pass, logs a short summary line; returns per-episode arrays) and `train(variants=..., seeds=..., ...)` that loops all (variant, seed) runs and returns the collected data dict.
  - `src/reinforce_cartpole/orchestrator.py` — **the only entry point** and the only `setup_logging(log_dir)` caller (root `basicConfig` wins once); bootstraps `sys.path` with the `src/` dir (parents[1]) to reach `utils/logging_utils`; owns the CLI (`argparse`); drives train → writes CSV → plots the PNG → writes `results/reinforce_cartpole/results.md`; logs date, python/package versions, config.
- **Output shape:** per `(variant, seed)` run a list of rows `(episode_index, variant, seed, return, baseline_after_update, steps)`. Aggregate over the 5 seeds of each variant, per episode index, via numpy: `mean`, `std` (population std `ddof=0`).
- Hardware to record: **CPU Ryzen 7 9700 X, GPU RTX 5060 Ti.** `device auto` maps to `cuda` when `torch.cuda.is_available()` (it is — CUDA is present here), else `cpu`. Whatever value the run actually used must be recorded in results.md and the Result section (it should match `torch.cuda.is_available()` at runtime).

## Step-by-step workflow
- [x] 1. Write this plan.
- [ ] 2. Write `src/reinforce_cartpole/policy.py` per invariants.
- [ ] 3. Write `src/reinforce_cartpole/train.py` per invariants (both variants; reward-to-go; truncation kept; `steps_to_next` bookkeeping only in the trainer, not the env).
- [ ] 4. Write `src/reinforce_cartpole/orchestrator.py` (CLI, `sys.path` bootstrap, single `setup_logging("results/reinforce_cartpole")`, orchestration):
  - `results/reinforce_cartpole/training.csv` — columns: `variant,seed,episode,return,baseline,steps` (one row per episode of every run; variant ∈ {`no_baseline`, `moving_avg`}).
  - `results/reinforce_cartpole/returns_curve.png` — matplotlib: x = episode (1..E); **one line per variant** — `no_baseline` in blue, `moving_avg` in red — solid line = mean over 5 seeds, shaded `fill_between` band = mean ± 1 std (same color); legend, axis labels (`Episode`, `Return`), title; `dpi=150`.
  - `results/reinforce_cartpole/results.md` — for each variant: **best mean return over the 5 seeds and the episode index where it occurs**, full config (episodes, lr, B, seeds), hardware line (CPU Ryzen 7 9700 X, GPU RTX 5060 Ti) plus the device actually used (`auto` → `cuda` here), versions.
  - Default run: **1000 episodes** per (variant, seed) run, 5 seeds → 10,000 episodes total.
  - `src/reinforce_cartpole/checkpoints/` + `.pt` checkpoints are **not needed** and must not be invented; do not create a checkpoints dir for this task.
- [ ] 5. Smoke test: `python src/reinforce_cartpole/orchestrator.py --episodes 50 --seeds 0 1` — verify CSV + PNG + results.md are written, log lines appear on the terminal and in `results/reinforce_cartpole/log.txt`, episode returns rise over time on both variants (sanity), and no baseline NaN/inf.
- [ ] 6. Full run: `python src/reinforce_cartpole/orchestrator.py` (defaults: 1000 episodes, seeds 0..4, lr 0.01, B 0.5).
- [ ] 7. Fill in the Result section below and finalize `results/reinforce_cartpole/results.md`.

## Result
(To be filled after the full run: best mean return + episode index per variant, runtimes, hardware, versions.)