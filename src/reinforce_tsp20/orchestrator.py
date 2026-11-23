"""Orchestrator for the REINFORCE TSP-20 (attention) solver — the only script
you run.

Spec run (default):
    conda activate nor && python src/reinforce_tsp20/orchestrator.py \
        [--epochs 100] [--batches-per-epoch 100] [--batch-size 512] \
        [--lr 1e-3] [--n 20] [--val-n 1000] [--test-n 1000] \
        [--baseline-b 0.5] [--device auto] [--relabel]

Overfit diagnostic (replays the 5 fixed instances of
``data/reinforce_tsp20/overfit/`` as EVERY batch — effective batch = 5,
``--batch-size`` ignored — under the same one-step-per-batch rules; writes
``overfit_results.md`` + ``<variant>_overfit_returns_curve.png`` and touches
no eval data):
    conda activate nor && python src/reinforce_tsp20/orchestrator.py \
        --overfit [--epochs 20] [--batches-per-epoch 100] [--lr 1e-3]

The default lr is 1e-3 (NOT the instructions file's 1e-4): the spec's
1e-4 assumed the spec's mandatory one-update-per-tour update, which is
relaxed (2026-11-22) to one `optimizer.step()` per 512-tour batch — the
10x-larger default compensates for 512x fewer optimizer steps per instance.

Pipeline: prepare the fixed eval instances + LKH-3 reference labels
(``dataset.py``) -> train one policy per variant (``train.py``), with the
checkpoint protocol chosen by ``--use_eval`` (default False: keep, per
variant, ONE checkpoint by best epoch-mean TRAINING return, re-saving
``src/reinforce_tsp20/checkpoints/<variant>_best_train_return.pt`` on every
strict improvement; with --use_eval: evaluate the live policy on the VAL
split at the end of every epoch — greedy + 128-sampled rows — and keep the
two simultaneous best checkpoints of each kind) -> final evaluation of all
methods on val + test, with the four neural rows coming from those
checkpoints (with --use_eval False, both of a variant's rows use its
single checkpoint) ->
draw one returns-curve PNG per variant (inline) -> write ``results.md``
(inline). Logs to ``results/reinforce_tsp20/log.txt`` (mirrored to the
terminal).
"""

import argparse
import contextlib
import datetime
import io
import logging
import sys
import time
from pathlib import Path

# Repo structure: this file lives in src/reinforce_tsp20/.  Python auto-adds
# the script dir, but utils/ (and the task modules) live in src/ and
# src/reinforce_tsp20/ respectively -- add src/ too (parents[1]).
SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402
from heuristics import nearest_neighbor, two_opt, tour_length  # noqa: E402

import train as train_mod  # noqa: E402
import dataset as dataset_mod  # noqa: E402

matplotlib.use("Agg")  # file output only, never opens a window
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("reinforce_tsp20.orchestrator")

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "results" / "reinforce_tsp20"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
RESULTS_MD = LOG_DIR / "results.md"

HARDWARE = ("Ryzen 7 9700 X", "RTX 5060 Ti")
VARIANT_COLORS = {"moving_avg": "#1f77b4", "greedy_rollout": "#d62728"}
GAMMA = 1.0  # fixed by the task invariants -- no discount
EVAL_SAMPLES = 128  # "128-sampled" eval rows: full tours per instance


def pick_device(name):
    name = str(name)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


# ------------------------------------------------------------------ eval rows
def _stats(lengths, lkh_mean, force_zero_std):
    """Per-method summary stats over the per-instance lengths of one split.

    mean, std (population, ddof=0), q1/median/q3 (np.percentile linear
    default) and ``gap vs LKH-3`` = mean - mean(LKH-3 on the same split).
    Neural greedy rows report std = 0.0 by construction (no sampling); every
    other row reports the population std across the instances.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    mean = float(np.mean(lengths))
    if force_zero_std:
        std = 0.0
    else:
        std = float(np.std(lengths, ddof=0))
    q1, median, q3 = np.percentile(lengths, [25.0, 50.0, 75.0])
    return {
        "mean": mean,
        "std": std,
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "gap": mean - float(lkh_mean),
    }


def _nn_lengths(D_list):
    out = np.empty(len(D_list))
    for i, D in enumerate(D_list):
        out[i] = tour_length(nearest_neighbor(D, start=0), D)
    return out


def _nn2opt_lengths(D_list):
    # `two_opt` historically printed one stray line on an asymmetric input;
    # keep stdout clean regardless (it does not affect the returned tour).
    with contextlib.redirect_stdout(io.StringIO()):
        out = np.empty(len(D_list))
        for i, D in enumerate(D_list):
            tour = two_opt(list(nearest_neighbor(D, start=0)), D)
            out[i] = tour_length(tour, D)
    return out


def _greedy_lengths(policy, D_list, device):
    """One deterministic greedy rollout per instance -> tour length."""
    out = np.empty(len(D_list))
    for i, D in enumerate(D_list):
        Dt = torch.from_numpy(D.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            L = policy.rollout(Dt, mode="greedy")["length"]
        out[i] = float(L[0])
    return out


def _sampled_min_lengths(policy, gen, D_list, device, n_samples):
    """n_samples sampled full tours per instance -> per-instance min length.

    The tours are n_samples *independent* full rollouts, each drawn B=1 from
    the given generator, so the draws consumed are exactly n_samples * 19
    uniforms per instance (19 = n - 1 choice steps) -- fully reproducible once
    the generator is seeded.
    """
    n = len(D_list)
    out = np.empty(n)
    for i, D in enumerate(D_list):
        Dt = torch.from_numpy(D.astype(np.float32)).unsqueeze(0).to(device)
        best = float("inf")
        with torch.no_grad():
            for _ in range(n_samples):
                L = policy.rollout(Dt, mode="sample", generator=gen)["length"]
                if L[0].item() < best:
                    best = L[0].item()
        out[i] = best
    return out


def _to_Dt(D_list, device):
    """Stack a list of (n, n) matrices into a float32 (B, n, n) tensor on
    `device` — the same conversion as training (`train._to_D`: float64 ->
    float32), so eval sees exactly the numeric inputs training sees."""
    arr = np.stack([np.asarray(x, dtype=np.float64) for x in D_list])
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def chunked_greedy_lengths(policy, D_list, device, chunk=128):
    """Greedy tour length per instance of D_list, in `chunk`-instance batches
    of a single no-grad rollout (peak GPU memory bounded — the per-epoch eval
    hook runs while the training weights + optimizer state are resident, and
    ollama occupies most of the card; the chunk defaults were picked from the
    2026-11-22 GPU-memory benchmark: rollout rows=512 -> peak 291 MiB,
    rows=128 fits well below the ~1 GB left free)."""
    D_list = list(D_list)
    m = len(D_list)
    out = np.empty(m)
    for s in range(0, m, chunk):
        block = D_list[s:s + chunk]
        Dt = _to_Dt(block, device)
        with torch.no_grad():
            out[s:s + len(block)] = policy.rollout(
                Dt, mode="greedy")["length"].cpu().numpy()
    return out.astype(np.float64)


def chunked_sampled_min_lengths(policy, D_list, n_samples, gen, device,
                                rows_per_call=1024):
    """Per-instance MIN of n_samples independent sampled tour lengths for
    each instance of D_list, in no-grad rollout calls of at most
    `rows_per_call` rows (rows_per_call // n_samples instances per call at
    the defaults: 8 val instances x 128 samples = 1024 rows — chosen from
    the 2026-11-22 GPU-memory benchmark: rows=1024 -> 22 ms / peak 573 MiB,
    rows=2048 OOMs on the ~1 GB ollama leaves).

    Each call tensor repeats the call's instances n_samples times in order
    (rollout r // n_samples of instance i), so the per-instance min over its
    own rows is the min of n_samples sampled tours of that instance —
    statistically the same quantity as the final-eval B=1-per-instance
    helper (a different draw schedule, not a bit-identical draw sequence:
    one vectorized `torch.rand` per row per step within the call), fully
    reproducible for fixed weights + generator seed.
    """
    D_list = list(D_list)
    m = len(D_list)
    out = np.empty(m)
    per_call = max(1, rows_per_call // n_samples)
    for s in range(0, m, per_call):
        block = D_list[s:s + per_call]
        reps = [D for D in block for _ in range(n_samples)]
        Dt = _to_Dt(reps, device)
        with torch.no_grad():
            L = policy.rollout(Dt, mode="sample",
                               generator=gen)["length"].cpu().numpy()
        L = L.reshape(len(block), n_samples)
        out[s:s + len(block)] = L.min(axis=1)
    return out.astype(np.float64)


def make_epoch_end_eval(variant, vi, val_D, n_samples, device, log):
    """Build the end-of-epoch eval callback for ONE variant (the
    --use_eval=True protocol; the signature matches
    `train._train_variant_body`'s hook slot: called as
    ``fn(variant, vi, epoch, policy, epoch_mean=...)`` at the end of every
    epoch).

    Every epoch, on the LIVE policy weights, it evaluates on the val split
    (instances `val_D`; selection set — the test split stays held out for
    the final report):
      - greedy row: greedy tour length per val instance, chunked <= 128
        instances per rollout, `torch.no_grad`;
      - 128-sampled row: per-instance min of n_samples independent sampled
        tours, same chunking, `torch.no_grad`, drawn from a DEDICATED eval
        generator (deterministic seed from the policy seed, re-seeded once
        per epoch before the sampled pass) — never the training generator,
        so the training math, RNG state and draw sequence are untouched.
    Both rows are logged. The policy's state_dict (raw, no optimizer state)
    is saved to `CKPT_DIR/<variant>_best_greedy.pt` / `<variant>_best_128.pt`
    whenever that row is a new all-epoch best (epoch 1 always writes both).
    The two best checkpoints are the SOLE source of the four neural rows in
    the final report (greedy row <- best_greedy, sampled row <- best_128).
    """
    ckpt_greedy = CKPT_DIR / f"{variant}_best_greedy.pt"
    ckpt_128 = CKPT_DIR / f"{variant}_best_128.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best = {"greedy": None, "sampled": None}
    policy_seed = vi * 1_000_000
    gen = torch.Generator(device=str(device))

    def _call(variant, vi, e, policy, epoch_mean=None):
        # epoch_mean is not needed here (the hook decides by val metrics);
        # the kwarg is part of the hook slot's call signature.
        t0 = time.perf_counter()
        glen = chunked_greedy_lengths(policy, val_D, device)
        # Dedicated eval generator: deterministic per (variant, epoch),
        # seeded from the policy seed; NEVER the training generator.
        gen.manual_seed(policy_seed * 1_000_003 + e)
        slen = chunked_sampled_min_lengths(policy, val_D, n_samples, gen,
                                           device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gmean = float(np.mean(glen))
        smean = float(np.mean(slen))
        log.info("[%s] epoch %3d eval (val) | greedy mean %.1f | "
                 "128-sampled mean %.1f | wall %.2f s",
                 variant, e, gmean, smean, time.perf_counter() - t0)
        # Best-of-so-far checkpoints: raw state_dict; the final report loads
        # them into a fresh TSPPolicy (see load_best_checkpoints).
        if best["greedy"] is None or gmean < best["greedy"]:
            best["greedy"] = gmean
            torch.save(policy.state_dict(), str(ckpt_greedy))
            log.info("  [%s] best-greedy checkpoint saved (val mean %.1f, "
                     "epoch %d) -> %s",
                     variant, gmean, e, ckpt_greedy.name)
        if best["sampled"] is None or smean < best["sampled"]:
            best["sampled"] = smean
            torch.save(policy.state_dict(), str(ckpt_128))
            log.info("  [%s] best-128 checkpoint saved (val mean %.1f, "
                     "epoch %d) -> %s",
                     variant, smean, e, ckpt_128.name)

    return _call


def make_best_train_return_hook(variant, device, log):
    """Build the end-of-epoch checkpointing callback for ONE variant (the
    --use_eval=False protocol; the signature matches
    `train._train_variant_body`'s hook slot: called as
    ``fn(variant, vi, epoch, policy, epoch_mean=...)`` at the end of every
    epoch).

    No evaluation on any split: it decides by the epoch's MEAN TRAINING
    return, ``epoch_mean`` — the mean over the epoch's batch returns, which
    train already computed and logged, passed in as the hook kwarg (return =
    -tour length, so higher = better).  Whenever that mean is a new all-epoch
    best (epoch 1 always writes; strict improvement only), the LIVE policy's
    raw ``state_dict`` (no optimizer state) is saved to
    ``CKPT_DIR/<variant>_best_train_return.pt`` — ONE checkpoint file per
    variant, overwritten in place.  No generators are created, nothing is
    sampled: the training math, RNG state and draw sequence are untouched.
    The single checkpoint is the SOLE source of BOTH of the variant's rows
    in the final report (see ``load_best_train_return_checkpoints``): the
    greedy row and the 128-sampled row are both computed from these weights.
    """
    ckpt = CKPT_DIR / f"{variant}_best_train_return.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best = {"return": None}  # mut in the closure (no rebinding)

    def _call(variant, vi, e, policy, epoch_mean=None):
        if epoch_mean is None:
            raise RuntimeError(
                f"best-training-return hook for {variant}: no epoch_mean "
                f"passed by the train loop (epoch {e})")
        if best["return"] is None or epoch_mean > best["return"]:
            best["return"] = epoch_mean
            torch.save(policy.state_dict(), str(ckpt))
            log.info(
                "  [%s] epoch %d best training return %.2f -> checkpoint "
                "saved -> %s",
                variant, e, epoch_mean, ckpt.name)

    return _call


def load_best_train_return_checkpoints(n, device):
    """Load each variant's ONE best-training-return checkpoint (the
    --use_eval=False protocol) into a FRESH `TSPPolicy` instance (raw
    state_dict -> `TSPPolicy(n)` + `load_state_dict`).

    Returns ``{variant: {"greedy": pol, "128": pol}}`` where BOTH slots
    reference the SAME loaded policy: the greedy row and the 128-sampled row
    of the final report are both computed from the single checkpoint (the
    128-sampled row's rows still draw their 128 tours from the eval-phase
    generator, exactly as in the two-checkpoint protocol)."""
    from policy import TSPPolicy  # local: same module name as train.py has
    out = {}
    for variant in train_mod.VARIANTS:
        path = CKPT_DIR / f"{variant}_best_train_return.pt"
        pol = TSPPolicy(n=n).to(device)
        sd = torch.load(str(path), map_location=device)
        pol.load_state_dict(sd)
        out[variant] = {"greedy": pol, "128": pol}
        log.info("[%s] final report policies loaded from %s "
                 "(both rows from this single checkpoint)",
                 variant, path.name)
    return out


def load_best_checkpoints(n, device):
    """Load each variant's two best checkpoints into FRESH `TSPPolicy`
    instances (raw state_dict -> `TSPPolicy(n)` + `load_state_dict`).

    Returns ``{variant: {"greedy": policy_g, "128": policy_s}}`` — the ONLY
    policies used for the four neural rows of the final report
    (greedy row <- best_greedy checkpoint, sampled row <- best_128
    checkpoint)."""
    from policy import TSPPolicy  # local: same module name as train.py has
    out = {}
    for variant in train_mod.VARIANTS:
        p_g = TSPPolicy(n=n).to(device)
        p_s = TSPPolicy(n=n).to(device)
        sd_g = torch.load(str(CKPT_DIR / f"{variant}_best_greedy.pt"),
                          map_location=device)
        sd_s = torch.load(str(CKPT_DIR / f"{variant}_best_128.pt"),
                          map_location=device)
        p_g.load_state_dict(sd_g)
        p_s.load_state_dict(sd_s)
        out[variant] = {"greedy": p_g, "128": p_s}
        log.info("[%s] final report policies loaded from %s / %s",
                 variant,
                 (CKPT_DIR / f"{variant}_best_greedy.pt").name,
                 (CKPT_DIR / f"{variant}_best_128.pt").name)
    return out


def evaluate_split(split, eval_data, device, n_samples, gens, log, ckpt_pols):
    """Run all 7 methods on ONE split's instances; return (rows, split wall).

    Rows are in the fixed order:
        NN, NN+2opt, LKH-3, <A> greedy, <A> 128-sampled,
        <B> greedy, <B> 128-sampled.
    Each row carries its summary stats + the wall-clock seconds for that method
    over the split's instances.  The LKH-3 row's wall is the labeling wall
    already measured for this split by ``dataset.build_eval``.

    The four neural rows use ``ckpt_pols[variant]`` — the checkpoint-loaded
    policies from training (the two best val checkpoints via
    ``load_best_checkpoints`` with --use_eval; or the single
    best-training-return checkpoint, mapped to BOTH the greedy and the
    "128" slots, via ``load_best_train_return_checkpoints``
    without it) — NOT the live end-of-training weights: the report is
    checkpoint-selected, per the checkpoint protocol in effect.
    """
    t_split = time.perf_counter()
    data = eval_data[split]
    D_list = data["D"]
    lkh = np.asarray(data["lkh_len"], dtype=np.float64)
    lkh_mean = float(lkh.mean())
    rows = []

    t0 = time.perf_counter()
    rows.append(("NN", _stats(_nn_lengths(D_list), lkh_mean, False),
                 time.perf_counter() - t0))
    t0 = time.perf_counter()
    rows.append(("NN+2opt", _stats(_nn2opt_lengths(D_list), lkh_mean, False),
                 time.perf_counter() - t0))
    rows.append(("LKH-3", _stats(lkh, lkh_mean, False),
                 eval_data["lkh_wall_s"][split]))

    for variant in train_mod.VARIANTS:
        # checkpoint-selected policies (see docstring).
        pol_g = ckpt_pols[variant]["greedy"]
        pol_s = ckpt_pols[variant]["128"]
        t0 = time.perf_counter()
        glen = _greedy_lengths(pol_g, D_list, device)
        rows.append((f"{variant} (greedy)", _stats(glen, lkh_mean, True),
                     time.perf_counter() - t0))
        t0 = time.perf_counter()
        slen = _sampled_min_lengths(pol_s, gens[variant], D_list, device,
                                    n_samples)
        rows.append((f"{variant} ({n_samples}-sampled)",
                     _stats(slen, lkh_mean, False), time.perf_counter() - t0))

    wall = time.perf_counter() - t_split
    for method, st, mwall in rows:
        log.info("  [%s] %-24s mean %.1f std %.1f q1 %.1f med %.1f q3 %.1f "
                 "gap %+.1f wall %.3f s",
                 split, method, st["mean"], st["std"], st["q1"],
                 st["median"], st["q3"], st["gap"], mwall)
    return rows, wall


def plot_variant_returns(variant, res, path):
    """One PNG per variant: x = epoch 1..E, y = epoch mean return, band =
    mean +/- 1 std over the epoch's batch returns (population std)."""
    epochs = res["epochs"]
    x = np.array([e["epoch"] for e in epochs])
    mean = np.array([e["mean"] for e in epochs])
    std = np.array([e["std"] for e in epochs])
    best = max(epochs, key=lambda e: e["mean"])
    color = VARIANT_COLORS[variant]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.plot(x, mean, color=color, lw=1.4,
            label="epoch mean return (±1 std over batch returns)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("epoch mean return (negative tour length)")
    ax.set_xlim(1, max(len(epochs), 1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        f"{variant} — best-epoch mean {best['mean']:.1f} (at epoch "
        f"{best['epoch']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt_table(rows):
    """Markdown rows for one split's eval table."""
    lines = [
        "| method | mean | std | q1 | median | q3 | gap vs LKH-3 | wall (s) |",
        "|--------|------|-----|----|--------|----|--------------|---------|",
    ]
    for method, st, wall in rows:
        lines.append(
            f"| {method} | {st['mean']:.1f} | {st['std']:.1f} | {st['q1']:.1f} "
            f"| {st['median']:.1f} | {st['q3']:.1f} | {st['gap']:+.1f} "
            f"| {wall:.3f} |")
    return "\n".join(lines)


def write_results_md(cfg, res, val_rows, test_rows, runtimes, env_line):
    """Write results/reinforce_tsp20/results.md (all 7 sections)."""
    base = cfg["base"]
    epA = res["moving_avg"]["epochs"][-1]
    epB = res["greedy_rollout"]["epochs"][-1]
    pb = res["greedy_rollout"]
    use_eval = bool(cfg.get("use_eval"))

    # The checkpoint-protocol sentence for the header paragraph: which
    # checkpoints exist and which rows come from which weights.
    if use_eval:
        ckpt_protocol = (
            "After EVERY epoch, the live policy of each variant is evaluated "
            "on the "
            f"{cfg['val_n']} VAL instances (greedy tour length, and the "
            "per-instance min of 128 independent sampled tours), and the "
            "two SIMULTANEOUS best checkpoints per variant are kept (best "
            "by val greedy mean, and best by val 128-sampled mean — "
            "independent).  The neural rows below are "
            f"evaluated on the {cfg['val_n']} val + {cfg['test_n']} test "
            "instances AGAINST NN, NN+2-opt, and LKH-3 (reference, RUNS = 1, "
            "SEED = 1), and come from those checkpoints ONLY: each variant's "
            "greedy row uses its best-greedy checkpoint, its 128-sampled "
            "row its best-128 checkpoint (the end-of-training weights are "
            "never used for the reported rows).")
    else:
        ckpt_protocol = (
            "At the end of EVERY epoch, each variant keeps a checkpoint by "
            "best epoch-mean TRAINING return (the mean over the epoch's "
            f"{cfg['batches_per_epoch']} batch returns; return = -tour "
            "length, so higher = better — no evaluation on any split, no "
            "generators): the live policy's raw state_dict is re-saved to "
            "`<variant>_best_train_return.pt` whenever that epoch mean is a "
            "new all-epoch best (epoch 1 always writes; strict improvement "
            "only).  The neural rows below are "
            f"evaluated on the {cfg['val_n']} val + {cfg['test_n']} test "
            "instances AGAINST NN, NN+2-opt, and LKH-3 (reference, RUNS = 1, "
            "SEED = 1), and come from that ONE checkpoint file per variant "
            "ONLY: BOTH of a variant's rows (the greedy row and the "
            "128-sampled row) are computed from that same checkpoint (the "
            "end-of-training weights are never used for the reported "
            "rows).")

    def _ckpt_row(variant):
        # The Runtimes "checkpoints" row for one variant, per protocol.
        if use_eval:
            rel_g = (CKPT_DIR / f"{variant}_best_greedy.pt").relative_to(
                REPO_ROOT)
            rel_s = (CKPT_DIR / f"{variant}_best_128.pt").relative_to(
                REPO_ROOT)
            return (
                f"| checkpoints, {variant} | `{rel_g.as_posix()}` "
                "(best val greedy), "
                f"`{rel_s.as_posix()}` (best val 128-sampled) |")
        rel_t = (CKPT_DIR / f"{variant}_best_train_return.pt").relative_to(
            REPO_ROOT)
        return (
            f"| checkpoints, {variant} | `{rel_t.as_posix()}` "
            "(best TRAINING-epoch mean return; both the greedy and the "
            "128-sampled rows are computed from this single checkpoint) |")

    lines = []
    # 1. Header paragraph
    lines += [
        "# REINFORCE TSP-20 (attention) results",
        "",
        "Vanilla REINFORCE (on-policy policy gradient, gamma = 1.0, reward-to-go) "
        f"on random asymmetric TSP-{cfg['n']} instances (n = {cfg['n']}; integer matrices from "
        "`gen_tsp`; reward per step = -distance(step), full-tour return = -tour "
        "length, no terminal bonus), solved by an attention/Transformer policy: "
        "an encoder of 3 blocks (each = 8-head multi-head self-attention, dim 64, "
        "dim per head 8, + FFN 64 -> 256 -> 64 (GELU), residual and pre-layer-norms) "
        "over the n node embeddings (x_i = D[i, :] / 100 -> 64), plus an "
        "averaging-node decoder (one special query = the mean of the start-node and "
        "current-node encodings, attending over all unvisited nodes; one 8-head dim-64 "
        f"attention layer + FFN 64 -> 256 -> 64 + linear head to n logits, visited "
        f"nodes masked to -inf).  Two baseline variants, one policy per variant: "
        f"(A) `moving_avg` — baseline b <- b*(1-B) + G_epoch*B, B = {cfg['baseline_b']}, "
        "initial 0.0, updated once per epoch after its "
        f"{cfg['batches_per_epoch']} batches, where G_epoch is the epoch-mean of the "
        "epoch's batch returns, subtracted as the per-return (G_t - b); (B) "
        "`greedy_rollout` — a frozen greedy copy (identical initial weights, no "
        "gradient updates except promotion) whose per-batch REINFORCE baseline is "
        f"-L_base, L_base = the frozen copy's mean greedy tour length over the batch's "
        f"{cfg['batch_size']} instances (so the loss uses G_t + L_base); at the end of "
        "each epoch the frozen weights are promoted (baseline_theta <- cur_theta) iff "
        "the current policy's epoch-mean greedy length L_cur over ALL of the epoch's "
        "instances satisfies L_cur <= ceil(0.95 * L_base).  UPDATE (spec relaxation, "
        "2026-11-22: the spec's mandatory 'one update per tour' is relaxed to one "
        "update per batch — see the instructions file's Deviations section): each "
        "batch's "
        f"{cfg['batch_size']} tours are sampled under the batch-start weights, and the "
        "batch receives ONE REINFORCE update "
        "(zero_grad -> loss.backward -> optimizer.step once per batch, no gradient "
        "accumulation; reward-to-go G_t = sum_{t' >= t} R_{t'} per tour; loss = the "
        "batch sum -sum_b sum_t logprob_{b,t} * (G_{b,t} - baseline), i.e. the same "
        f"REINFORCE sum as per-tour updates).  {cfg['epochs']} epochs x "
        f"{cfg['batches_per_epoch']} batches x batch-size {cfg['batch_size']} fresh "
        f"instances, Adam lr = {cfg['lr']}, policy_seed = variant_index * 1_000_000 "
        f"(0 = moving_avg, {1_000_000} = greedy_rollout), device = {cfg['device']}.  "
        f"{ckpt_protocol}",
        f"Environment: {env_line}.",
        "",
        "## Training summary",
        "",
        "| variant | policy_seed | final epoch mean ± std (over batch returns) | "
        "promotions | promotion epochs | final L_base | training wall (s) |",
        "|---------|-------------|----------------------------------------------|"
        "------------|------------------|--------------|-------------------|",
    ]
    ma = res["moving_avg"]
    lines.append(
        f"| moving_avg | {ma['policy_seed']} | {epA['mean']:.1f} ± {epA['std']:.1f} "
        f"| n/a | n/a | n/a | {ma['wall_s']:.1f} |"
    )
    lines.append(
        f"| greedy_rollout | {pb['policy_seed']} | {epB['mean']:.1f} ± "
        f"{epB['std']:.1f} | {pb['promotions']} | "
        f"{pb['promotion_epochs']} | {pb['final_l_base']:.1f} "
        f"| {pb['wall_s']:.1f} |"
    )

    # 3. Results (val) / Results (test)
    lines += [
        "",
        "## Results (val)",
        "",
        _fmt_table(val_rows),
        "",
        "## Results (test)",
        "",
        _fmt_table(test_rows),
        "",
        "## Runtimes",
        "",
        "| quantity | value |",
        "|----------|-------|",
        f"| training, moving_avg | {ma['wall_s']:.1f} s |",
        f"| training, greedy_rollout | {pb['wall_s']:.1f} s |",
        f"| LKH labeling, val | {runtimes['lkh']['val']:.3f} s |",
        f"| LKH labeling, test | {runtimes['lkh']['test']:.3f} s |",
        f"| LKH labeling, total | {runtimes['lkh']['total']:.3f} s |",
        f"| eval, val | {runtimes['eval']['val']:.3f} s |",
        f"| eval, test | {runtimes['eval']['test']:.3f} s |",
        f"| total | {runtimes['total']:.1f} s |",
    ]
    for variant in train_mod.VARIANTS:
        lines.append(_ckpt_row(variant))
    lines += [
        "",
        "## Configuration",
        "",
        "| key | value |",
        "|-----|-------|",
        f"| n | {cfg['n']} |",
        f"| epochs | {cfg['epochs']} |",
        f"| batches per epoch | {cfg['batches_per_epoch']} |",
        f"| batch size | {cfg['batch_size']} |",
        f"| lr | {cfg['lr']} |",
        f"| gamma | {cfg['gamma']} |",
        f"| B (moving_avg baseline const) | {cfg['baseline_b']} |",
        f"| policy seeds | 0 (moving_avg), 1000000 (greedy_rollout) |",
        f"| device | {cfg['device']} |",
        f"| val_n | {cfg['val_n']} |",
        f"| test_n | {cfg['test_n']} |",
        f"| eval base offset (BASE) | {cfg['base']} |",
        f"| relabel | {cfg['relabel']} |",
        f"| use_eval | {cfg['use_eval']} |",
        "",
        "## Environment",
        "",
        env_line + ".",
        "",
        "## Hardware specification",
        f"CPU: {HARDWARE[0]}",
        f"GPU: {HARDWARE[1]}",
        f"Device used (resolved --device={cfg['device_arg']}): {cfg['device']}",
        f"eval base offset (BASE): {base}",
        "",
    ]
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines))


# ------------------------------------------------------------------ overfit
def _run_overfit(n, epochs, batches_per_epoch, lr, baseline_b, device):
    """Overfit diagnostic: both variants replay the SAME 5 fixed instances
    from ``data/reinforce_tsp20/overfit/`` as EVERY batch, then we report the
    per-instance greedy tour lengths against the LKH-3 / NN / NN+2-opt
    references for those 5 instances.  Writes ``overfit_results.md`` and one
    ``<variant>_overfit_returns_curve.png`` per variant (the spec-run
    ``results.md`` / ``*_returns_curve.png`` are untouched).
    """
    log = logging.getLogger("reinforce_tsp20.orchestrator")
    from policy import TSPPolicy  # noqa: E402

    data_root = dataset_mod.DATA_ROOT
    inst_dir = data_root / "overfit"
    inst_files = sorted(inst_dir.glob("inst_*.txt"))
    if len(inst_files) != 5:
        raise RuntimeError(
            f"expected 5 overfit instances in {inst_dir}, got "
            f"{len(inst_files)}: {[p.name for p in inst_files]!r}")
    D_fixed = np.stack([dataset_mod.read_instance(p) for p in inst_files])
    D_fixed = np.asarray(D_fixed, dtype=np.float64)
    names = [p.stem for p in inst_files]

    t_lkh0 = time.perf_counter()
    lkh = []
    for i, D in enumerate(D_fixed):
        lkh_len, _tour, _dt = dataset_mod._run_lkh(D, seed=int(names[i][-4:]))
        lkh.append(lkh_len)
    lkh_wall = time.perf_counter() - t_lkh0
    lkh = np.asarray(lkh, dtype=np.float64)
    nn = _nn_lengths([D_fixed[i] for i in range(len(inst_files))])
    nn2opt = _nn2opt_lengths([D_fixed[i] for i in range(len(inst_files))])

    torch.manual_seed(0)  # determinism for the init-greedy row only
    init_policy = TSPPolicy(n=n).to(device)
    init_g = _greedy_lengths(init_policy, [D_fixed[i] for i in range(len(inst_files))],
                             device)
    del init_policy

    t0 = time.perf_counter()
    trained = train_mod.train_variants_overfit(
        n=n, epochs=epochs, batches_per_epoch=batches_per_epoch, lr=lr,
        baseline_b=baseline_b, device=device, log=log, D_fixed=D_fixed)
    # Drop the non-variant payload key; the variant order matches VARIANTS.
    variants = [v for v in trained if v != "_D_fixed"]
    res = {v: trained[v] for v in variants}
    train_wall = {v: res[v]["wall_s"] for v in variants}

    greedy = {}
    for variant in variants:
        if device.type == "cuda":
            torch.cuda.synchronize()
        greedy[variant] = _greedy_lengths(
            res[variant]["policy"],
            [D_fixed[i] for i in range(len(inst_files))], device)

    OVERFIT_MD = LOG_DIR / "overfit_results.md"
    lines = []
    lines += [
        "# REINFORCE TSP-20 — overfit diagnostic",
        "",
        f"Both variants replay the SAME 5 fixed instances "
        f"(`{names[0]}` … `{names[-1]}`, ``data/reinforce_tsp20/overfit/``) as "
        f"every batch — effective batch = {len(inst_files)} tours per Adam step, "
        f"``--batch-size`` ignored — under the same one-step-per-batch batched "
        f"REINFORCE rules as the spec run (spec relaxation of 2026-11-22): "
        f"epochs = {epochs}, batches/epoch = {batches_per_epoch}, "
        f"lr = {lr:g}, B = {baseline_b:g}, "
        f"policy seeds A=0 / B=1000000, device = {str(device)}. "
        f"Adam steps per variant = {epochs * batches_per_epoch} "
        f"({epochs * batches_per_epoch * len(inst_files)} tours). "
        f"Judged against these 5 instances' references "
        f"(LKH mean {lkh.mean():.1f}, NN mean {float(nn.mean()):.1f}, "
        f"NN+2-opt mean {float(nn2opt.mean()):.1f}); the P4 probe reference "
        f"curve reached a per-tour mean of 601.4 at 10k tours (lr 1e-4).",
        "",
    ]
    header = ("| instance | LKH-3 | NN | NN+2-opt | init greedy | "
              + " | ".join(f"{v} greedy" for v in variants) + " |")
    lines += [
        header,
        "|---" * (5 + len(variants)) + "|",
    ]
    for i, name in enumerate(names):
        row = (f"| {name} | {int(lkh[i])} | {int(nn[i])} | "
               f"{int(nn2opt[i])} | {init_g[i]:.0f} | "
               + " | ".join(f"{greedy[v][i]:.0f}" for v in variants)
               + " |")
        lines.append(row)
    means = (f"| **mean** | {lkh.mean():.1f} | {float(nn.mean()):.1f} "
             f"| {float(nn2opt.mean()):.1f} | {float(init_g.mean()):.1f} | "
             + " | ".join(
                 f"**{float(greedy[v].mean()):.1f}**" for v in variants)
             + " |")
    lines += [means, ""]
    for variant in variants:
        e = res[variant]["epochs"]
        best = max(e, key=lambda x: x["mean"])
        prom_txt = ""
        if variant == "greedy_rollout":
            prom_txt = (
                f", promotions = {res[variant]['promotions']} over {epochs} "
                f"epochs, final L_base = {res[variant]['final_l_base']:.1f}")
        lines += [
            f"- **{variant}**: final-epoch mean return "
            f"{e[-1]['mean']:.1f} ± {e[-1]['std']:.1f}; best-epoch mean "
            f"{best['mean']:.1f} (epoch {best['epoch']}); greedy mean over the "
            f"5 instances {float(greedy[variant].mean()):.1f} vs LKH mean "
            f"{lkh.mean():.1f}{prom_txt}.",
        ]
    lines += [
        "",
        "## Runtimes",
        "",
        "| stage | wall (s) |",
        "|---|---|",
    ]
    for variant in variants:
        lines.append(f"| train {variant} | {train_wall[variant]:.1f} |")
    lines += [
        f"| LKH-3 labels (5 instances) | {lkh_wall:.1f} |",
        "",
        "## Environment",
        "",
        f"- {torch.__version__} (CUDA available: {torch.cuda.is_available()}), "
        f"device used: {str(device)}; Python {sys.version.split()[0]}",
        "",
        "## Curves",
        "",
    ]
    for variant in variants:
        lines.append(
            f"- `{variant}_overfit_returns_curve.png` — epoch mean return over "
            f"the replayed batches (band = ±1 std over the epoch's batch "
            f"returns).")
    OVERFIT_MD.write_text("\n".join(lines))
    log.info("overfit results written to %s", OVERFIT_MD)

    for variant in variants:
        path = LOG_DIR / f"{variant}_overfit_returns_curve.png"
        plot_variant_returns(variant, res[variant], path)
        log.info("overfit plot written to %s", path)

    total = time.perf_counter() - t0
    for variant in variants:
        log.info("overfit done [%s]: greedy mean %.1f vs LKH mean %.1f",
                 variant, float(greedy[variant].mean()), lkh.mean())
    log.info("overfit total wall: %.1f s (training %s)",
             total, {v: f"{w:.1f}" for v, w in train_wall.items()})


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batches-per-epoch", type=int, default=100,
                        dest="batches_per_epoch")
    parser.add_argument("--batch-size", type=int, default=512,
                        dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam lr (default 1e-3: the spec's 1e-4 assumed the "
                             "mandatory per-tour update; with the relaxed one "
                             "optimizer.step() per 512-tour batch the default is "
                             "10x larger to compensate)")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--val-n", type=int, default=1000, dest="val_n")
    parser.add_argument("--test-n", type=int, default=1000, dest="test_n")
    parser.add_argument("--baseline-b", type=float, default=0.5,
                        dest="baseline_b")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--relabel", action="store_true",
                        help="recompute LKH-3 labels even if the CSVs exist")
    parser.add_argument("--use_eval", action="store_true",
                        help="checkpoint protocol (default off): ON = the "
                             "end-of-epoch VAL eval of the live policy "
                             "(greedy + 128-sampled rows, logged) keeping "
                             "the TWO simultaneous best checkpoints per "
                             "variant (best-greedy / best-128, one per "
                             "row); OFF (default) = the simpler protocol: "
                             "ONE checkpoint per variant, kept by best "
                             "epoch-mean TRAINING return (the number "
                             "already logged per epoch — no eval, no "
                             "generators), used for BOTH that variant's "
                             "reported rows")
    parser.add_argument("--overfit", action="store_true",
                        help="overfit diagnostic: replay the 5 fixed instances "
                             "from data/reinforce_tsp20/overfit/ as EVERY "
                             "batch (effective batch = 5, --batch-size not "
                             "used) under the same one-step-per-batch REINFORCE "
                             "rules; report per-instance greedy lengths vs "
                             "LKH-3/NN/NN+2-opt to overfit_results.md + "
                             "<variant>_overfit_returns_curve.png")
    args = parser.parse_args()

    setup_logging(LOG_DIR)
    log = logging.getLogger("reinforce_tsp20.orchestrator")

    device = pick_device(args.device)
    torch_ver = torch.__version__
    env_line = (
        f"Python {sys.version.split()[0]}, torch {torch_ver} "
        f"(CUDA available: {torch.cuda.is_available()}), "
        f"NumPy {np.__version__}, matplotlib {matplotlib.__version__}"
    )
    cfg = {
        "n": args.n, "epochs": args.epochs,
        "batches_per_epoch": args.batches_per_epoch,
        "batch_size": args.batch_size, "lr": args.lr, "gamma": GAMMA,
        "baseline_b": args.baseline_b, "device": str(device),
        "device_arg": args.device, "val_n": args.val_n,
        "test_n": args.test_n, "relabel": args.relabel,
        "use_eval": args.use_eval,
    }

    log.info("run started | %s | device=%s | %s",
             datetime.datetime.now().isoformat(timespec="seconds"), device,
             env_line)
    for key, val in cfg.items():
        log.info("cfg %-18s = %s", key, val)

    if args.overfit:
        _run_overfit(args.n, args.epochs, args.batches_per_epoch, args.lr,
                     args.baseline_b, device)
        return

    t_total = time.perf_counter()

    # ---- 1. eval instances + LKH-3 labels
    eval_data = dataset_mod.build_eval(args.n, args.val_n, args.test_n,
                                       relabel=args.relabel)
    cfg["base"] = eval_data["base"]

    # ---- 2. train both variants (checkpoint protocol per --use_eval)
    # ONE hook per variant runs at the end of every epoch with the LIVE
    # policy weights. --use_eval (ON): end-of-epoch VAL eval (greedy +
    # 128-sampled rows, logged; dedicated policy_seed-derived generators +
    # chunked no-grad rollouts) keeping the two simultaneous best
    # checkpoints in CKPT_DIR. Otherwise: checkpointing by best epoch-mean
    # TRAINING return (the mean train already computed; no eval, no
    # generators, no extra computation) keeping ONE checkpoint per
    # variant.  Either way the training math, RNG and draw sequence are
    # untouched.
    hooks = {}
    for vi, variant in enumerate(train_mod.VARIANTS):
        if args.use_eval:
            hooks[variant] = make_epoch_end_eval(
                variant, vi, eval_data["val"]["D"], EVAL_SAMPLES, device,
                log)
        else:
            hooks[variant] = make_best_train_return_hook(variant, device,
                                                         log)

    def _hook(variant, vi, e, policy, epoch_mean=None):
        hooks[variant](variant, vi, e, policy, epoch_mean=epoch_mean)

    res = train_mod.train_variants(
        n=args.n, epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch, batch_size=args.batch_size,
        lr=args.lr, baseline_b=args.baseline_b, device=device, log=log,
        epoch_end_eval=_hook)

    # ---- 3. evaluate (all methods, both splits)
    # The four neural rows come from the variants' checkpoints saved during
    # step 2, loaded into FRESH TSPPolicy instances — not from the live
    # end-of-training weights: with --use_eval that's the two best val
    # checkpoints (one per row); without it, the one best-training-return
    # checkpoint, mapped to BOTH of the variant's slots.
    if args.use_eval:
        ckpt_pols = load_best_checkpoints(args.n, device)
    else:
        ckpt_pols = load_best_train_return_checkpoints(args.n, device)
    # Drop the live policies now that their checkpoints are the report's
    # source; free their GPU memory before the eval allocations.
    for v in res:
        res[v].pop("policy", None)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # One generator per variant, re-seeded ONCE with the policy seed at the
    # start of the eval phase so the 128-sampled rows are reproducible.
    gens = {v: torch.Generator(device=str(device))
                 .manual_seed(res[v]["policy_seed"]) for v in res}
    log.info("evaluation phase (all methods, both splits)")
    val_rows, eval_val_wall = evaluate_split(
        "val", eval_data, device, EVAL_SAMPLES, gens, log,
        ckpt_pols=ckpt_pols)
    test_rows, eval_test_wall = evaluate_split(
        "test", eval_data, device, EVAL_SAMPLES, gens, log,
        ckpt_pols=ckpt_pols)
    eval_wall = {"val": eval_val_wall, "test": eval_test_wall}
    eval_total = eval_val_wall + eval_test_wall
    log.info("eval done: val wall %.2f s, test wall %.2f s (total %.2f s)",
             eval_val_wall, eval_test_wall, eval_total)

    runtimes = {
        "train": {v: res[v]["wall_s"] for v in res},
        "lkh": eval_data["lkh_wall_s"],
        "eval": eval_wall,
    }
    runtimes["total"] = (sum(runtimes["train"].values())
                         + runtimes["lkh"]["total"] + eval_total)

    # ---- 4. plots (one PNG per variant, inline)
    for variant in res:
        path = LOG_DIR / f"{variant}_returns_curve.png"
        plot_variant_returns(variant, res[variant], path)
        log.info("plot written to %s", path)

    # ---- 5. results.md (inline)
    write_results_md(cfg, res, val_rows, test_rows, runtimes, env_line)
    log.info("results written to %s", RESULTS_MD)

    total_s = time.perf_counter() - t_total
    log.info("total wall: %.1f s (pipeline sum %.1f s)", total_s,
             runtimes["total"])


if __name__ == "__main__":
    main()
