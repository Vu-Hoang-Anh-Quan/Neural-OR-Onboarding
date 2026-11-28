"""POMO evaluation for the equal-budget evaluation (rl4co_intro checkpoint).

House recipe (re-verified against `src/rl4co_intro/orchestrator.py`): fresh
`CVRPEnv(generator=CVRPGenerator(num_loc=N))` +
`POMO(env, train_data_size=1, val_data_size=1, test_data_size=1, batch_size=1)`
(the default 6-layer `AttentionModelPolicy`, the same default the training task
used, so the strict load below is valid), `model.setup(stage="fit")` (safe
pre-load: `generate_default_data` defaults False — no data-dir writes), then
`load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"], strict=True)`
and `.eval().cuda()`.

Eval is the production path only: per-instance
`evaluate_policy(env, model.policy, dataset, method=..., auto_batch_size=False,
batch_size=1)` where `dataset = TensorDictDataset(gen_td_batch1)` — the helper
builds its own DataLoader (`dataset.collate_fn`) and does `env.reset(td)` per
batch internally. NEVER `policy(td, decode_type="augment_dihedral_8")`
directly (the AM dispatch rejects it). The three spec budgets:
  - method="greedy"             deterministic, single forward
  - method="augment_dihedral_8" best of 8 dihedral copies, single start
  - method="sampling"           SamplingEval: samples=1280, softmax_temp=1.0,
                                best-of-N reward from out["reward"]

Seeding: pass s of the sampling budget sets `torch.manual_seed(POMO_SEED_PASS +
s)` immediately before that per-instance sampling call (the start-action
samples draw from the global torch RNG). greedy/augment are deterministic, so
passes 1 and 2 repeat them byte-identically (documented).

CUDA: warm-up pass first, then `torch.cuda.synchronize()` around each timed
call (the helper's `inference_time` covers the whole dataloader and is never
used — nor divided). Budget times = median of the per-instance walls over the
first 50 CVRP-50 instances per budget, after warm-up.

Every tour gets an independent (non-tensor) recheck before its row is kept:
all customers visited exactly once, every route load <= Q (normalized <= 1.0),
and the raw recompute of the closed tour length matches -reward to 4 decimals
(REWARD_TOL on the [0,1] float32 scale; probe-verified diffs: 9.7e-7 CVRP-50,
6.5e-5 on X n502).

Outputs (append-only CSV rows — crash/resume safe; a row is committed only
after its recheck passed):
  <results>/pomo_cvrp50.csv  id, pass, mode, reward, wall
  <results>/pomo_x.csv       id, pass, mode, reward, wall     (X set: id=name)
  <results>/budgets.csv      mode, measured_median_s, column_time_s, merged_into_fixed
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datagen  # noqa: E402  (sibling: GEN_TD_FILENAME / load_x_set / X records)

from rl4co.data.dataset import TensorDictDataset  # noqa: E402
from rl4co.envs import CVRPEnv  # noqa: E402
from rl4co.envs.routing.cvrp.generator import CVRPGenerator  # noqa: E402
from rl4co.models import POMO  # noqa: E402
from rl4co.tasks.eval import evaluate_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "src" / "rl4co_intro" / "checkpoints" / "pomo_cvrp50_best.pt"

MODES = ("greedy", "augment_dihedral_8", "sampling")  # spec's 3 POMO budgets
POMO_SEED_PASS = 1000       # sampling pass s -> torch.manual_seed(1000 + s)
REWARD_TOL = 5e-4           # 4-decimal tolerance on [0,1] float32 tour length
DE = 1e-5                   # float tolerance for the route-load check
WARMUP_CALLS = 2            # warm-up calls per mode before timed calls
MEASURE_LIMIT = 50          # median wall over the first 50 CVRP-50 instances
FIXED_BUDGETS = (1.0, 10.0)  # fixed columns; merge a measured time within ~1%

POMO_CVRP50_CSV = "pomo_cvrp50.csv"
POMO_X_CSV = "pomo_x.csv"
BUDGETS_CSV = "budgets.csv"


# --- independent tour recheck (template: _probe.raw_tour_len) -----------------

def recheck_tour(actions, reward, xy, demand01, ctx):
    """Verify a tour independently of the env reward (no tensor math on actions).

    `actions` = full policy tour over customer indices with depot returns (0);
    `xy` = [0,1] float coords, index 0 = depot; `demand01` = normalized demands
    (index 0 = 0.0, customers in (0, 1]). Returns L (raw closed tour length).
    Raises AssertionError on coverage / capacity / length failures.
    """
    n_cust = len(xy) - 1
    vis = [a for a in actions if a != 0]
    if len(vis) != n_cust or sorted(vis) != list(range(1, n_cust + 1)):
        raise AssertionError(f"{ctx}: coverage failed (each customer exactly once)")
    routes, cur = [], []
    for a in actions:
        if a == 0:
            if cur:
                routes.append(cur)
                cur = []
        else:
            cur.append(a)
    if cur:
        routes.append(cur)
    p0 = [p[0] for p in xy]
    p1 = [p[1] for p in xy]
    locs2 = torch.stack(
        [torch.tensor(p0, dtype=torch.float64), torch.tensor(p1, dtype=torch.float64)], -1
    )

    def d(i, j):
        return float((locs2[i] - locs2[j]).square().sum().sqrt())

    L = 0.0
    for r in routes:
        seq = [0] + r + [0]
        L += sum(d(seq[k], seq[k + 1]) for k in range(len(seq) - 1))
        load = sum(demand01[c] for c in r)
        if load > 1.0 + DE:
            raise AssertionError(f"{ctx}: route load {load:.4f} > 1.0 (Q exceeded)")
    diff = abs(reward + L)
    if diff >= REWARD_TOL:
        raise AssertionError(f"{ctx}: |reward + L| = {diff:.2e} >= {REWARD_TOL}")
    return L, routes


# --- model build (house recipe) -----------------------------------------------

def build_model(num_loc: int):
    """Fresh env + POMO + strict load of the trained state dict; on cuda."""
    gen = CVRPGenerator(num_loc=num_loc)
    env = CVRPEnv(generator=gen)
    model = POMO(
        env,
        train_data_size=1, val_data_size=1, test_data_size=1,
        batch_size=1,
    )
    model.setup(stage="fit")  # safe pre-load (generate_default_data=False)
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().cuda()
    return env, model


def make_gen_dataset(td1):
    """Batch-1 generator td -> the dataset evaluate_policy expects (batch-1 only:
    zero-padded collate of several generator tds would corrupt the batch dim)."""
    return TensorDictDataset(td1)


# --- eval plumbing ------------------------------------------------------------

def _eval(env, model, td1, method, seed=None):
    """One per-instance evaluate_policy call (batch 1). Returns (actions, reward, wall)."""
    if method == "sampling":
        # Sampling start-action RNG is the global torch RNG: pass-level seed goes
        # here, right before the call.
        torch.manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = evaluate_policy(
        env,
        model.policy,               # the policy module, not the RL4COLitModule
        make_gen_dataset(td1),
        method=method,
        auto_batch_size=False,
        batch_size=1,
        progress=False,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    # batch-1: no padding ambiguity; actions [1, padded_T], rewards [1]
    actions = res["actions"][0].long().tolist()
    reward = float(res["rewards"][0])
    return actions, reward, wall


def warmup(env, model, td1, log):
    for method in MODES:
        for _ in range(WARMUP_CALLS):
            _eval(env, model, td1, method,
                  seed=POMO_SEED_PASS if method == "sampling" else None)
    torch.cuda.synchronize()
    log.info("pomo: CUDA warm-up done (%d warm-up calls x %d modes)", WARMUP_CALLS, len(MODES))


# --- resumability helpers -------------------------------------------------------

def _committed(path: Path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {(row["id"], row["pass"], row["mode"]) for row in csv.DictReader(f)}


def append_row(path: Path, header, row: dict):
    first = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if first:
            w.writeheader()
        w.writerow(row)
        f.flush()


def write_budgets(res_dir: Path, measured: dict, log):
    """Decide the 3 measured column times (median wall, rounded 2 dp; merge into
    the 1 s / 10 s column when within ~1%). Overwrites budgets.csv atomically."""
    rows = []
    for mode in MODES:
        t = round(measured[mode], 2)
        merged = ""
        for c in FIXED_BUDGETS:
            if abs(t - c) / c < 0.01:
                t, merged = c, ("1s" if c == 1.0 else "10s")
                log.warning(
                    "pomo: %s measured %.4f s within ~1%% of the %s s column "
                    "- merged into it", mode, measured[mode], c)
                break
        rows.append({
            "mode": mode,
            "measured_median_s": f"{measured[mode]:.4f}",
            "column_time_s": f"{t:g}",
            "merged_into_fixed": merged,
        })
    with open(res_dir / BUDGETS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "measured_median_s",
                                          "column_time_s", "merged_into_fixed"])
        w.writeheader()
        w.writerows(rows)
    log.info("pomo: budgets -> %s (%s)", BUDGETS_CSV,
             ", ".join(f"{r['mode']}={r['column_time_s']}s" for r in rows))


def ensure_budgets(res_dir: Path, log):
    """budgets.csv must exist before the solver phases; rebuild it from the
    pass-0 CVRP-50 walls if a crash left it behind."""
    if (res_dir / BUDGETS_CSV).exists():
        return
    path = res_dir / POMO_CVRP50_CSV
    if not path.exists():
        raise RuntimeError("budgets.csv missing and no pass-0 walls yet - run the POMO phase first")
    walls = {m: [] for m in MODES}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["pass"] == "0" and row["wall"]:
                walls[row["mode"]].append(float(row["wall"]))
    for m in MODES:
        if len(walls[m]) < MEASURE_LIMIT:
            raise RuntimeError(
                f"budgets missing and only {len(walls[m])}/{MEASURE_LIMIT} pass-0 walls "
                f"for {m} - finish the POMO CVRP-50 phase first")
    write_budgets(res_dir, {m: float(np.median(walls[m])) for m in MODES}, log)


# --- the two sets ---------------------------------------------------------------

def run_cvrp50(res_dir: Path, tds, log) -> None:
    """3 seed passes x N CVRP-50 instances x 3 modes; pass 0 also measures."""
    out = res_dir / POMO_CVRP50_CSV
    header = ["id", "pass", "mode", "reward", "wall"]
    committed = _committed(out)
    n = tds.batch_size[0]
    log.info("pomo cvrp50: %d instances, %d committed rows (resume); model load",
             n, len(committed))
    env, model = build_model(50)
    walls = {m: [] for m in MODES}
    for pass_id in range(3):
        for i in range(n):
            td1 = tds[i:i + 1]          # batch [1]
            xy = ([tuple(map(float, td1["depot"][0].tolist()))]
                  + [tuple(map(float, p)) for p in td1["locs"][0]])
            demand01 = [0.0] + [float(v) for v in td1["demand"][0].tolist()]
            for mode in MODES:
                k = (str(i), str(pass_id), mode)
                if k in committed:
                    continue
                seed = POMO_SEED_PASS + pass_id if mode == "sampling" else None
                try:
                    actions, reward, wall = _eval(env, model, td1, mode, seed=seed)
                    recheck_tour(actions, reward, xy, demand01,
                                 ctx=f"cvrp50 #{i} pass{pass_id} {mode}")
                except Exception as e:
                    # invalid/infeasible tour (env check_solution_validity assert)
                    # or failed recheck: no row, logged; key stays uncommitted
                    # so a resume retries it.
                    log.warning("pomo cvrp50: cvrp50 #%d pass %d %s: no row "
                                "committed (%s: %s)", i, pass_id, mode,
                                type(e).__name__, e)
                    continue
                append_row(out, header,
                           {"id": str(i), "pass": str(pass_id), "mode": mode,
                            "reward": f"{reward:.6f}", "wall": f"{wall:.4f}"})
                committed.add(k)
                if pass_id == 0 and len(walls[mode]) < MEASURE_LIMIT:
                    walls[mode].append(wall)
            del td1, xy, demand01
        log.info("pomo cvrp50: pass %d/3 done (rechecks: coverage + load<=Q + |r+L|<%.0e)",
                 pass_id + 1, REWARD_TOL)
        del env, model
        torch.cuda.empty_cache()
        env, model = build_model(50)
    if all(len(walls[m]) >= MEASURE_LIMIT for m in MODES):
        medians = {m: float(np.median(walls[m])) for m in MODES}
        g, d, s = (medians[m] for m in MODES)
        monotonic = "OK" if g <= d <= s else "NON-MONOTONIC - check warmup (expected after warmup)"
        log.info("pomo cvrp50: measured medians greedy %.3f / dihedral-8 %.3f / sampling %.3f (monotonic %s)",
                 g, d, s, monotonic)
        write_budgets(res_dir, medians, log)
    del env, model
    torch.cuda.empty_cache()


def run_x(res_dir: Path, recs, log) -> None:
    """3 seed passes x the X record list; fresh env+model per instance+pass
    (per-N generator, as the n=502 gate recipe)."""
    out = res_dir / POMO_X_CSV
    header = ["id", "pass", "mode", "reward", "wall"]
    committed = _committed(out)
    log.info("pomo x: %d instances, %d committed rows (resume)", len(recs), len(committed))
    for pass_id in range(3):
        for rec in recs:
            name = rec["name"]
            N = rec["n"] - 1
            Q = float(rec["cap"])
            env, model = build_model(N)
            # Hand-built X td (the verified n=502 gate recipe): [0,1000] -> [0,1],
            # demand normalized by Q (informational raw capacity key kept).
            td1 = TensorDict({
                "locs": (rec["xy"][1:] / 1000.0).to(torch.float32)[None],      # [1, N, 2]
                "depot": (rec["xy"][0] / 1000.0).to(torch.float32)[None],      # [1, 2]
                "demand": (rec["demands"].to(torch.float32) / Q)[None],         # [1, N]
                "capacity": torch.tensor([[Q]]),                               # info
            }, batch_size=[1])
            xy = [tuple(map(float, p)) for p in (rec["xy"] / 1000.0).tolist()]
            demand01 = [0.0] + [float(v) / Q for v in rec["demands"].tolist()]
            for mode in MODES:
                k = (name, str(pass_id), mode)
                if k in committed:
                    continue
                seed = POMO_SEED_PASS + pass_id if mode == "sampling" else None
                try:
                    actions, reward, wall = _eval(env, model, td1, mode, seed=seed)
                    recheck_tour(actions, reward, xy, demand01,
                                 ctx=f"x {name} pass{pass_id} {mode}")
                except Exception as e:
                    # invalid/infeasible tour (env check_solution_validity assert)
                    # or failed recheck: no row, logged; key stays uncommitted
                    # so a resume retries it.
                    log.warning("pomo x: %s pass %d %s: no row committed (%s: %s)",
                                name, pass_id, mode, type(e).__name__, e)
                    continue
                append_row(out, header,
                           {"id": name, "pass": str(pass_id), "mode": mode,
                            "reward": f"{reward:.6f}", "wall": f"{wall:.4f}"})
                committed.add(k)
            log.info("pomo x: %s pass %d/3 done (rechecks passed)", name, pass_id + 1)
            del env, model, td1, xy, demand01
        torch.cuda.empty_cache()
        log.info("pomo x: pass %d/3 done", pass_id + 1)


def ensure_pomo(res_dir: Path, n_cvrp: int, x_names: list, log) -> None:
    """Run/finish the POMO phase (resume-safe entry)."""
    tds = torch.load(res_dir / datagen.GEN_TD_FILENAME, weights_only=False)
    assert tds.batch_size[0] >= n_cvrp, "generator tds shorter than requested CVRP-50 set"
    run_cvrp50(res_dir, tds, log)
    full_recs = [r for r in datagen.load_x_set(res_dir, ROOT / "data" / "chosen", log)
                if r["name"] in x_names]
    if not full_recs or {r["name"] for r in full_recs} != set(x_names):
        raise RuntimeError(f"X records mismatch: {[r['name'] for r in full_recs]}")
    run_x(res_dir, full_recs, log)
