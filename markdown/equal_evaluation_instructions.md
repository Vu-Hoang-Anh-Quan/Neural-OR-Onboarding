# Multi-agent handoff: `equal_evaluation`

## Goal

Equally-budgeted evaluation of three CVRP solvers — **POMO** (RL4CO, the
`rl4co_intro` task's trained `pomo_cvrp50_best.pt` checkpoint), **PyVRP** and
**OR-Tools** — on two test sets: (a) **1000 fixed CVRP-50 instances** generated
from the RL4CO CVRP-50 distribution, and (b) **10 X instances** from
`data/chosen/` — under **five ascending per-instance time budgets**: the three
POMO-decode times measured empirically, then fixed 1 s and 10 s. Deliverable:
two `results.md` tables (one per test set) of **mean gap ± std over
instances**, gap to per-instance PyVRP-60 s (CVRP-50 set) or to the BKS
(X set), plus two gap–time plots with ±1 std envelopes.

**Scope: write and validate the evaluation code only, in `src/equal_evaluation/`,
results under `results/equal_evaluation/` — then STOP before the full run.**
Do not execute the full evaluation (≈17 h serial, dominated by the 1000 × 60 s
PyVRP reference solve runs); the run gate below is what authorizes it. Only the
cheap smoke test in § Smoke gate may be run.

**Run command after this task**: `conda activate nor && python src/equal_evaluation/orchestrator.py`
(no arguments — all knobs are constants at the top of the orchestrator; keep them there).

## Overview & invariants

### Roles (multi-file layout — do NOT collapse this into one file; single-file is the anti-pattern to avoid here, unlike `rl4co_intro`)

| File | Job |
|---|---|
| `src/equal_evaluation/orchestrator.py` | Entry point; constants (`ROOT`, `SEEDS=[1,2,3]`, `POMO_SEED_PASS = 1000`, time limits, sizes, seeds); phase orchestration + resume logic; writes `log.txt` via `setup_logging("results/equal_evaluation")` |
| `src/equal_evaluation/datagen.py` | Build both test sets once: (a) 1000 CVRP-50 instances by RL4CO `CVRPGenerator` seeding (below), serialized as EUC_2D `.vrp` files with raw demands + `CAPACITY : 40` at **×1e6 integer scale** (scale constant in one place), kept as a generator td `torch.save`; (b) load the 10 `data/chosen/*.vrp` (+ `.sol` BKS); per-instance record list (name, n, cap, demmax, bks) |
| `src/equal_evaluation/pomo_run.py` | Load POMO once (`rl4co_intro/checkpoints/pomo_cvrp50_best.pt`); build per-instance eval envs; CUDA warm-up; run the three POMO budgets per instance × 3 seed passes via `rl4co.tasks.eval.evaluate_policy`; return per-instance tours + rewards + wall times, and write a per-instance CSV + the 3 measured POMO times to the results dir |
| `src/equal_evaluation/solvers_cli.py` | For each instance file, each time limit in the 5-budget list, each of 3 seeds: `subprocess.run` the existing `src/run_pyvrp.py` (CLI) and `src/run_ortools.py` (CLI) — **no re-implementation** — parse their one-line CSV stdout (`pyvrp,<path>,<sec>,<seed>,<cost>,<feasible>,<wall>` / `ortools,<path>,<sec>,-,<cost>,<feasible>,<wall>`; re-read both scripts' exact stdout format before writing the parser); assert `feasible == True`, else log + continue. **OR-Tools sub-second fix (only allowed shared-file change)**: `run_ortools.py` currently does `sec = int(sys.argv[2])` then `p.time_limit.seconds = sec` — a measured POMO budget is sub-second (ms-scale), so the int parse would crash (or round to a useless 0 s). Make the minimal backward-compatible edit: parse `sec = float(sys.argv[2])` and set `p.time_limit.seconds = int(sec)` **and** `p.time_limit.nanos = int(round(sec - int(sec)) * 1e9)` (the protobuf `Duration` fields — the standard way to express sub-second limits in OR-Tools). Existing integer-second calls (1, 10, 60) behave identically. No new wrapper script — the CLI is still `python src/run_ortools.py <file> <sec>`. The reference run also lives here: `run_pyvrp.py <file> 60 <seed=1>` per instance (single fixed seed — justified below) |
| `src/equal_evaluation/aggregate.py` | Gap computation (exact recipes + scaling below), per-instance table, instance-mean ± std aggregation across (instances × seeds), both test sets; write `results.md`; render the two gap–time PNGs (specs below) |

Import-path rule (house pattern, same as `src/run_ortools.py`): every script in
`src/equal_evaluation/` does

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logging_utils import setup_logging   # logging convention: FileHandler + StreamHandler, log dirs under results/<task>/
```

(`utils/` lives under `src/`; importing it also lands `src/` on the path for the
`cvrp` package used by `run_ortools.py`'s imports — which the subprocesses see
anyway since they are launched with `cwd=ROOT`.)

All code + comments in **English**. No `__init__.py`.

### Where the numbers come from (exact, verified in rl4co 0.7.0)

- **Data flow (verify before coding — this is the #1 bug source).**
  `evaluate_policy(env, policy, dataset, method=..., auto_batch_size=False,
  batch_size=1)` takes a **generator-format** `TensorDictDataset`; its
  `EvalBase.__call__` does `env.reset(td)` per batch internally, building the
  *state* td: `locs [1, N+1, 2]` with **depot at index 0** (customers 1..N),
  `demand [1, N]`, `vehicle_capacity` = the **generator's** `vehicle_capacity`
  attribute (default 1.0 — the td's `capacity` key is never read by
  `get_action_mask`). Calling `policy(td, ...)` *directly* on a generator td
  fails in POMO `init_embed` ("Expected size 500 but got size 501") — that is
  why the direct-call note in `rl4co_intro` is about the state td; here the
  production path is always `evaluate_policy` (no direct policy calls).
  Env construction:

  ```python
  from rl4co.envs.routing import CVRPEnv, CVRPGenerator
  env = CVRPEnv(generator=CVRPGenerator(num_loc=N))   # N = customers excl. depot (CVRP-50 → N=50; X-n502-k39 → N=501)
  ```

  `CVRPGenerator(num_loc=N)` for N not in its capacity table (e.g. N = 501 for
  X instances) logs a benign warning ("closest capacity") — harmless, the td
  is hand-built anyway; **never** pass `vehicle_capacity=Q`: `_reset` then
  stamps `vehicle_capacity=Q` into the state td while `demand` stays
  normalized (sum ≤ 1), so `get_action_mask`'s
  `demand + used > vehicle_capacity` check stops firing at all → silently
  infeasible route loads.
  The generator-td `d` has `d["locs"] [1, N, 2]` (customers **only**, no
  depot), `d["depot"] [1, 2]`, `d["demand"] [1, N]`
  **capacity-normalized** (the generator stores `demand / self.capacity` —
  the Kool-et-al table value, *not* `vehicle_capacity`; do **not**
  re-normalize), `d["capacity"]` `[1, 1]` the raw scalar (40.0 for CVRP-50). The dataset to feed is:

  ```python
  from tensordict import TensorDict
  from rl4co.data.dataset import TensorDictDataset
  d = gen([1])                     # batched gen-td for one instance
  ds = TensorDictDataset(d)
  ```

  `env.reset(d)` then produces the state td (`locs [1, N+1, 2]` depot-first,
  `vehicle_capacity` from `gen.vehicle_capacity`, default **1.0**).

- **Reward**: env reward = **−objective** = raw float EUC_2D tour length on
  **[0,1] coordinates**. `out["rewards"]` from `evaluate_policy` is
  best-of-N per instance (best start node / best sampled tour). Convert:
  **`L_cmp = (−reward) × 1e6`** for CVRP-50 instances (written at ×1e6 scale),
  **`L_cmp = (−reward) × 1000`** for X instances (BKS on [0,1000] scale).
  Both references are nint-scale (`floor(x+0.5)`, same as
  `cvrp_utils.route_cost`), so POMO is compared slightly favorably vs the
  PyVRP/OR-Tools floats; note the ±1-unit nint-vs-float rounding offset in
  `results.md` once.

- **The three POMO budgets = the three `evaluate_policy` methods** (from
  `rl4co/tasks/eval.py`, 0.7.0):
  - `method="greedy"` → `GreedyEval` → `policy(td, decode_type="greedy",
    num_starts=0)` — deterministic; single forward.
  - `method="augment_dihedral_8"` → `AugmentationEval(num_augment=8,
    force_dihedral_8=True)` → **best of 8 dihedral copies**, single start.
  - `method="sampling"` → `SamplingEval(samples=1280, softmax_temp=1.0)` →
    `policy(td, decode_type="sampling", num_starts=1280, temperature=1.0,
    top_p=0.0, top_k=0, multisample=True, softmax_temp=1.0,
    select_best=True, select_start_nodes_fn=sample_n_random_actions)`.
  Pass `auto_batch_size=False, batch_size=1` (per-instance eval). The
  **measured wall time per mode** (warm-up done, `torch.cuda.synchronize()`
  around the call, median of ≥ 3 warm instances) is the budget for that mode
  in the 5-column table; if a measured time lands within ~1% of a fixed
  column (1 s / 10 s), drop it into that column and note it.

  ⚠ **Do NOT call `policy(td, decode_type="augment_dihedral_8")` directly** —
  the AM model dispatch rejects it (warns "Unknown decode type … Defaulting to
  Sampling", then asserts). All three budgets go through `evaluate_policy`
  with `method=...` and the kwargs above; that is the production path.

- **POMO seed passes (exactly where each seed feeds)**: budget 3
  (`sampling`) draws the start-action samples from the **global torch RNG**:
  3 passes `pass s ∈ {0,1,2}` set `torch.manual_seed(POMO_SEED_PASS + s)`
  (`POMO_SEED_PASS = 1000`) immediately before each per-instance sampling
  call. Budgets 1–2 are deterministic → runs 2, 3 are byte-identical repeats
  (std ≈ 0; document that). Do **not** touch the global seed during instance
  generation (generation seeds are `torch.manual_seed(1234 + i)`, set just
  before each `gen([1])`; the generator has no per-instance `torch.Generator`
  hook — `torch.distributions.Uniform.sample` takes no `generator=` kwarg, global
  RNG only).

- **X tds (10 instances in `data/chosen/`)**: hand-built per the verified
  n=502 gate recipe (appendix) from each `.vrp` (parse via `vrplib` —
  `cvrp_utils.load` also gives it, but you only need `node_coord` / `demand` /
  `capacity`): coordinates ÷ 1000 (X files are [0,1000] → [0,1]);
  `demand = raw_demand / cap` (rl4co convention — the env's capacity check
  `demand + used > 1.0` operates in this normalized space; `get_action_mask`
  never reads the `capacity` key). The X-td for one instance of dim n
  (N = n−1 customers, matching `CVRPGenerator(num_loc=N)` and the data-flow
  section) is: `locs [1, N, 2]`, `depot [1, 2]`, `demand [1, N]`
  (normalized), `capacity = cap` (informational). Independent exact re-check
  of the tours at the end (same as the n502 gate): all N customers visited
  exactly once, every route's load ≤ cap, exact tour length matches env
  reward to 4 decimals (n502: diff 0.0000).

### Gap + mean/SD (exactly)

- Per-instance per-seed: `gap% = 100 × (cost − ref) / ref` (positive = worse).
  POMO `cost` = `L_cmp` above (plain float on the instance scale — raw tour
  length × 1e6 or × 1000, *not* nint);
  PyVRP cost = its printed float (already that scale); OR-Tools cost = its
  printed int (nint, exact).
- **Cell** = mean ± std (population std, ddof=0) of the **pooled per-instance
  gap values** — (1000 instances × 3 seed passes) for the CVRP-50 set,
  (10 instances × 3 passes) for the X set. POMO's 3 passes enter the pool
  (repeats give the real sampling std); PyVRP/OR-Tools 3 seeds enter too.
- Reference columns: **X set** → BKS from `.sol` `Cost <int>`; **CVRP-50
  set** → per-instance **PyVRP 60 s** cost (the per-instance reference
  target, not the randomized object: a single fixed seed — `seed=1` — is
  correct and documented; 3 seeds would triple ≈16.7 h of serial solves with
  no change to the reference semantics).

### Table + plot specs

- `results.md` — two tables (one per test set), 3 rows (POMO / PyVRP /
  OR-Tools) × 5 ascending budget columns. Column headers: the **3 measured
  POMO-mode times** (label each with mode name + measured seconds — e.g.
  `greedy (0.38 s)`, `dihedral-8 (0.33 s)`, `sampling/1280 (0.51 s)`), then
  **1 s**, **10 s**. Cell = `mean ± std` (gap %, 2 decimals) of the pooled
  per-instance gaps (3 seeds × instances; see §Gap).
  - **POMO fills only its 3 measured-budget columns** (that is what its
    budgets are); its 1 s / 10 s cells are **left blank** — POMO is
    fixed-workload, not wall-clock-limited, and all 3 measured times sit
    well under 1 s, so the 1 s / 10 s budgets would buy it nothing. Say so
    in a table footnote (and draw the horizontal dotted extension in the
    plot instead).
  - **PyVRP fills all 5 columns**: re-run the CLI at each of the 3 measured
    (sub-second, fractional — `run_pyvrp.py` already takes `float(sec)`) and
    the 2 fixed budgets.
  - **OR-Tools fills all 5 columns** too: its 3 sub-second columns need the
    §solvers_cli `int(sec)` → nanos fix; 1 s / 10 s are its native `int(sec)`.
  - Measured POMO times = median per-instance wall over the **first 50
    CVRP-50 instances** per budget (after warm-up), rounded to ~2 decimals,
    same on both tables (one measurement, shared column headers).
- **Two PNGs** (`gap_time_cvrp50.png`, `gap_time_x.png`), one per table:
  - x = budget seconds, **log scale**; 5 categorical ticks labeled with
    measured POMO times / 1 s / 10 s.
  - 3 lines (POMO / PyVRP / OR-Tools), **default matplotlib colors** (free
    colors, no palette tweaks), `fill_between` = mean ± 1 std band.
  - **POMO line starts at its own first point** (its measured greedy time),
    not at the x-axis minimum; extend a horizontal dotted line from POMO's
    last point to the right edge if all 3 POMO times < 10 s (note "POMO
    budget is decode-time, not wall-clock" in the caption).
  - `results.md` also gets the discrete **Environment + Hardware** section
    (versions: Python, torch, rl4co, lightning, tensordict, pyvrp, ortools,
    vrplib, matplotlib, numpy; `nproc` + `nvidia-smi` free/used) — house
    format, see `results/rl4co_intro/results.md` `## Environment` / `## Hardware`.

### Seed summary (write this into `results.md` § Configuration)

| Where | Seed(s) | Feeds |
|---|---|---|
| CVRP-50 instance #i | `torch.manual_seed(1234 + i)` | `CVRPGenerator(num_loc=50)([1])` per instance (global torch RNG; no per-instance `torch.Generator` hook in the generator) |
| POMO sampling budget, pass s | `torch.manual_seed(POMO_SEED_PASS + s)`, s ∈ {0,1,2} | the sampling start-action RNG |
| PyVRP budget runs | int `seed` arg, 3 seeds (e.g. 1, 2, 3) | `run_pyvrp.py` CLI → `pyvrp.solve(seed=...)` |
| OR-Tools budget runs | none (CLI has no seed arg) | 3 identical GLS re-runs; std ≈ 0 — document as such |
| CVRP-50 reference (PyVRP 60 s) | single fixed seed = 1 | the reference target; see § invariants |

## Mandatory pre-run gate: `nvidia-smi`

- POMO runs on GPU (move the loaded model to `cuda`; `torch.cuda.is_available()`
  is True here). Before any GPU phase: print full `nvidia-smi`; require
  `CUDA available: True` and ≥ ~2 GiB free on the RTX 5060 Ti 16 GB. Do NOT
  set a high free-memory bar for this box: ollama holds ~14.9 GiB of the
  16 GiB, and with ~1.4 GiB free the n=502 gate (N = 501, 1280-sample
  sampling — ~10× the decode steps of POMO-cvrp50) ran fine, so ~1–2 GiB
  free is proven-good headroom and a hard 6 GiB gate would refuse a known
  working state. If free < ~1 GiB → stop and ask the user (ollama can be
  asked to free memory; note it if it is).
- Also record the `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader`
  line at the start of the log and in `results.md` § Hardware.

## Workflow checklist

- [ ] `conda activate nor`; `python -c "import torch,rl4co; print(torch.cuda.is_available())"` → True; versions via `importlib.metadata.version` (pyvrp/vrplib have no `__version__` attr — use `importlib.metadata`).
- [ ] Generate the 1000 CVRP-50 instances once (`torch.manual_seed(1234 + i)` per i, serialized at ×1e6; verify a `.vrp` parses via `cvrp_utils.load`; one X instance n502: `run_pyvrp.py <file.vrp> 2 1` ≈ BKS 69 226 ±1 % (BKS/scale cross-check); and a PyVRP-60 s run on CVRP-50 instance #0 gives a sane reference).
- [ ] **Smoke gate** (cheap, ≤ ~5 min, must exit 0 end-to-end): pick **CVRP-50 instances #0 and #1 + X instance n502-k39**; run POMO all 3 budgets (per-instance, `evaluate_policy`, as in the n=502 gate script — that verified recipe is the template for `pomo_run.py`) + 1 s and 10 s `run_pyvrp.py`/`run_ortools.py` CLI calls per solver; assert: (i) POMO tours pass the independent feasibility recheck (all customers once, route load ≤ Q) and `-reward` matches a recompute of the tour length to 4 decimals; (ii) solver stdout parses and `feasible=True`; (iii) POMO-on-X gaps are huge (+260 % greedy / +40 % dihedral-8 on n502 is expected distribution-shift, not a bug — the gate evidence is in the doc appendix) but tours are feasible.
- [ ] Run the 3 POMO seed passes on all 1000 + all 10 X instances (per-instance, CUDA warm-up first, `torch.cuda.synchronize()` around each timed call — the helper's `inference_time` is **total only**, never divide it; wall = your timing).
- [ ] Run PyVRP 3 seeds × 5 time limits + OR-Tools 3 × 5 time limits on both sets via the existing CLI scripts (subprocess; parse CSV line; log the one-line stdout per run to `log.txt`).
- [ ] Run the reference: `run_pyvrp.py <file> 60 1` for all 1000 CVRP-50 instances (single seed — the reference target, see invariants; ≈ 16.7 h serial; **resume-friendly**: write each instance's cost to a per-instance JSON/CSV row file as it completes so a crashed run restarts only the missing ones; this is the single biggest wall-clock cost — do it first in the background if the user approves).
- [ ] Aggregate: per-instance gaps → pool (instances × seed passes) → mean ± std per (method × budget) cell, both tables; write `results.md` (both tables + Environment/Hardware/Configuration/seed-summary sections + the ±1 nint-vs-float note + the OR-Tools std≈0 note + POMO-on-X off-distribution caveat); render the 2 PNGs per the plot spec (log x, 5 ticks, 3 free-color lines, ±1 std `fill_between`, POMO line starts at its first point, dotted extension if all POMO < 10 s).
- [ ] **Stop.** Commit is the user's — do not `git commit`.

## Appendix: n=502 X-gate evidence (off-distribution caveat for POMO-on-X)

Verified on the production path (`evaluate_policy`, per-instance, batch
size 1, `augment_dihedral_8` via `AugmentationEval(num_augment=8,
force_dihedral_8=True)`, sampling via `SamplingEval(samples=1280,
softmax_temp=1.0)` — all through the library, no hand-rolled decode):

| mode | length (unit) | ×1000 | gap vs BKS 69 226 | wall | feasible |
|---|---|---|---|---|---|
| greedy | 249.3743 | 249 374 | **+260 %** | ~0.5 s | yes (501/501, load ≤ 13) |
| augment_dihedral_8 | 96.9856 | 96 986 | **+40 %** | ~0.4 s | yes |
| sampling (1280) | 189.8365 | 189 837 | **+174 %** | ~0.5 s | yes |

(The 256-sample override from an earlier gate run is not the budget; the
canonical 1280 is what the table uses.) The large gaps are genuine
distribution shift (POMO was trained on [0,1]² uniform demand-1..10 Q=40;
X-n502-k39 has Q=13, demand up to 100, and a corner depot at (0,0)) — **not a
bug**; tours pass an independent exact recheck (all customers exactly once,
route load ≤ 13, exact tour length matches the env reward to 4 decimals:
diff 0.0000). The X table therefore reports POMO-on-X with a **prominent
off-distribution caveat**; the BKS cross-check (a 2 s PyVRP run ≈ BKS ±1 %)
is the X table's validity anchor.

### X ground truth (`data/chosen/`, `name dim cap demmax bks`)

| name | dim | cap | demmax | bks |
|---|---|---|---|---|
| n101-k25 | 101 | 206 | 100 | 27591 |
| n153-k22 | 153 | 144 | 98 | 21220 |
| n204-k19 | 204 | 836 | 100 | 19565 |
| n251-k28 | 251 | 69 | 10 | 38684 |
| n275-k28 | 275 | 10 | 1 | 21245 |
| n298-k31 | 298 | 55 | 10 | 34231 |
| n351-k40 | 351 | 436 | 100 | 25896 |
| n401-k29 | 401 | 745 | 100 | 66154 |
| n449-k29 | 449 | 777 | 100 | 55233 |
| n502-k39 | 502 | 13 | 1 | 69226 |

Depot = node "0" / index 0; BKS in the `.sol`'s `Cost <n>` line. Uniform
demand-1 instances are **n275 and n502 only** (the rest are mixed demand).

### Hardware (current box)

CPU: AMD Ryzen 7 9700X 8-Core (16 logical cores, `nproc` = 16). GPU: NVIDIA
GeForce RTX 5060 Ti 16 GB (`nvidia-smi` free/used at run time). Python 3.11.17,
torch 2.11.0+cu128, rl4co 0.7.0, lightning 2.6.6, tensordict 0.15.6, pyvrp
0.14.0, ortools 9.15.6755, vrplib 2.2.0, matplotlib 3.11.2, numpy 2.4.6.

### Resumability + sanity

- **Resume-friendly everywhere**: the 1000-instance reference (`≈ 16.7 h`
  serial — the single biggest wall-clock cost) writes a per-instance
  "done" row to a JSON/CSV after each instance; a crashed run restarts only
  the missing ones. Same pattern for the POMO per-instance pass (the 3
  seed passes are independent — a crash in pass 2 doesn't redo pass 1).
- **Sanity checks to eyeball before trusting the table**:
  - POMO CVRP-50 gap vs the PyVRP-60 s reference should be in a sane range
    (POMO on-distribution; expect single-digit % if the reference is near-optimal,
    or slightly worse if 60 s PyVRP is tight — both fine; > 20 % means a bug).
  - X BKS cross-check: a 2 s `run_pyvrp.py` on one X instance ≈ BKS ±1 %
    (n502: PyVRP ≈ 69 906 = BKS ×1.0098 — same scale, confirmed).
  - POMO measured times should be near-monotonic across modes (greedy ≤
    dihedral-8 ≤ sampling) — non-monotonic walls in the gate are first-call
    CUDA warmup, hence the warm-up step.
  - OR-Tools std ≈ 0 (documented; not a bug).
