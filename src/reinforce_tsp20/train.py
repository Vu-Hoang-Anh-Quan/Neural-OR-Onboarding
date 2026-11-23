"""REINFORCE training for the REINFORCE TSP-20 solver — one policy per variant.

Trains the two baseline variants of vanilla REINFORCE (gamma = 1.0, reward =
-distance(step) on the original integer matrix, return = -tour length,
full reward-to-go) on the attention policy defined in policy.py.  Per
variant: one freshly initialized policy, its own Adam optimizer (lr), its
own torch.Generator for action sampling and its own baseline state —
variants are trained strictly independently.

Batch instances are generated fresh for every (epoch e, batch j) with seed
    policy_seed * 100000 + (e - 1) * 100 + j
(100 always in the formula) and each batch's per-instance seeds are
    batch_seed * 10000 + b
(b = 0 .. batch_size - 1): deterministic within the batch seed and distinct
across all (e, j, b) triples, and never persisted.

Each epoch consists of `batches_per_epoch` fresh instances-batches.  The
policy samples one complete tour per instance of a batch (all B = 512 tours
in parallel under the batch-start weights), and the batch receives exactly
ONE REINFORCE update — the spec's "one update per tour" is relaxed,
2026-11-22 (see the instructions file's Deviations section): `zero_grad()`
-> `loss.backward()` -> `optimizer.step()` once per batch, where the loss
is the REINFORCE sum over the batch, `-sum_b sum_t logprob_{b,t} *
(G_{b,t} + scalar)` with each tour's own reward-to-go
G_t = sum_{t' >= t} R_{t'} (gamma = 1.0, no discount); scalar = -baseline
(variant A) / +L_base (variant B), constant across the batch.  Semantics vs
spec: 512 Adam steps per batch become 1 (the orchestrator's default lr is
raised 1e-4 -> 1e-3 to compensate), all of a batch's tours are on-policy
w.r.t. the batch-start weights only, and the RNG draw mapping changes
(one (B,1) draw per step instead of one draw per tour).

Recorded per-(variant, epoch) stats are computed over the epoch's *batch
returns*: each batch return is the mean of that batch's tour returns, and
the epoch mean/±std are over those batch means (population std, ddof = 0).
Variant A's baseline update uses the same epoch mean.

Variant A (moving_avg): moving-average baseline `b <- b * (1 - B) +
G_epoch * B` (B = baseline_b, initial 0.0), updated once per epoch after
its last batch, where G_epoch is the mean of the epoch's batch returns;
b is subtracted as a per-return constant from every reward-to-go of the
epoch (the loss uses G_t - b).

Variant B (greedy_rollout): a frozen greedy-policy copy of the variant's
policy (identical initial weights, kept up-to-date only by epoch-end
promotions).  For each batch, the baseline is the frozen policy's mean
greedy tour length L_base over that batch's fresh instances (computed with
no gradient), added as a per-return constant to every reward-to-go of the
batch (the loss uses G_t + L_base).  At the end of each epoch, the
promotion rule is applied: with L_base and L_cur the mean greedy tour
lengths of the frozen and current policy respectively over ALL of the
epoch's fresh instances (recomputed, no gradient), the frozen policy
adopts the current policy's weights iff
    L_cur <= ceil(0.95 * L_base)
and the promotion is recorded.  The final L_base recorded at the end of
training is the frozen copy's mean greedy length over the last epoch's
instances, as the frozen copy stands at the end of training.

The eval generator used for the "128-sampled" eval rows is NOT used for
training sampling (training uses the per-variant training generator above).
"""

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Repo layout: this file lives in src/reinforce_tsp20/; `gen` lives in src/.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gen import gen_tsp  # noqa: E402
from policy import TSPPolicy  # noqa: E402
from dataset import read_instance, DATA_ROOT  # noqa: E402

VARIANTS = ("moving_avg", "greedy_rollout")

TRAIN_SCRATCH = DATA_ROOT / "train_scratch"


# ------------------------------------------------------------------ instances
def _gen_instances(batch_seed, batch_size, n, scratch=TRAIN_SCRATCH):
    """Fresh batch_size instances for (epoch, batch) seed `batch_seed`.

    Written to scratch files only so they can be read back via read_instance
    (gen_tsp only returns the path); nothing is persisted.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    Ds = []
    for b in range(batch_size):
        s = batch_seed * 10000 + b
        p = scratch / f"tr_{s}.txt"
        gen_tsp(n, seed=s, path=str(p))
        Ds.append(read_instance(p))
        p.unlink()
    return Ds


# ----------------------------------------------------------------- REINFORCE
def _reinforce_batch(policy, optimizer, D, generator, scalar, timings=None):
    """Batched REINFORCE: sample ALL rows of D as tours under the current
    weights, then ONE update for the whole batch (SPEC RELAXATION, 2026-11-22:
    one `optimizer.step()` per batch instead of the spec's one per tour).

    All B = D.shape[0] tours are sampled with the batch-start weights
    (parallel rollout, per-step `torch.rand(B, 1, generator=...)`), the
    reward-to-go is computed per tour (`G_{b,t} = sum_{t' >= t} R_{b,t'}`,
    including R_t and the closing return edge), and the loss is the REINFORCE
    sum over the batch: `-sum_b sum_t logprob_{b,t} * (G_{b,t} + scalar)`,
    gradient = one full-batch `zero_grad()` -> `loss.backward()` ->
    `optimizer.step()` at the end of the batch (no gradient accumulation).
    `scalar` is the constant advantage shift, constant across the batch
    (variant A: -baseline; variant B: +L_base, L_base = the batch's
    frozen-greedy mean).  Semantics vs spec: 512 Adam steps become 1; the
    batch is all-on-policy w.r.t. the batch-start weights only; the RNG
    draw mapping changes (B draws/step instead of one per tour).  The
    within-batch loss is the same REINFORCE sum as for per-tour updates, so
    the per-step (G_t + scalar) form is unchanged.

    Returns the batch's per-instance returns = negative tour lengths
    (np.float64, (B,)).

    Timing (purely diagnostic, no behavior change): if `timings` is a dict
    it accumulates wall seconds for the batch into
    rollout_s / loss_bwd_s / step_s / sync_s / n; the only host<->device
    sync is the final returns read-out, which also absorbs device-side
    completion of the queued rollout/gradient work since the previous
    batch (GPU work is launched asynchronously), like before.
    """
    B = D.shape[0]
    t0 = time.perf_counter()
    out = policy.rollout(D, mode="sample", generator=generator)
    logps = out["logprobs"]                           # n-1 tensors (B,)
    rewards = out["rewards"]                          # (B, n-1)
    t1 = time.perf_counter()
    # Vectorized reward-to-go: G_t = sum_{t' >= t} R_{t'} (including R_t)
    # = reversed cumulative sum of the rewards.
    rtg = torch.flip(
        torch.cumsum(torch.flip(rewards, dims=[1]), dim=1), dims=[1])
    loss = -torch.stack(
        [logps[t] * (rtg[:, t] + scalar) for t in range(len(logps))]
    ).sum(dim=1).sum()
    optimizer.zero_grad()
    loss.backward()
    t2 = time.perf_counter()
    optimizer.step()
    t3 = time.perf_counter()
    rets = out["return"].cpu().numpy().astype(np.float64)
    t4 = time.perf_counter()
    if timings is not None:
        timings["rollout_s"] += t1 - t0
        timings["loss_bwd_s"] += t2 - t1
        timings["step_s"] += t3 - t2
        timings["sync_s"] += t4 - t3
        timings["n"] += B
    return rets


def _greedy_lengths(policy, D):
    """No-grad greedy tour length per row of D, as np.float64 (B,)."""
    with torch.no_grad():
        return policy.rollout(D, mode="greedy")["length"].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------- training
def _train_variant_body(variant, vi, n, epochs, batches_per_epoch, lr,
                        baseline_b, device, log, make_batch, timings=None,
                        epoch_end_eval=None):
    """Run one variant's epoch loop; return its results dict.

    ``make_batch(j, e)`` is the ONLY thing that differs between the spec run
    and the overfit run: it returns the numpy ``(batch_size, n, n)`` distance
    matrices for batch ``j`` of epoch ``e`` (1-based).  Everything else — the
    policy init, the batched REINFORCE update (``_reinforce_batch`` — one
    ``optimizer.step()`` per batch, spec relaxation of 2026-11-22), the
    moving-average baseline (variant A) and the frozen-greedy-copy baseline +
    end-of-epoch promotion (variant B) — is shared verbatim with the spec run,
    so the overfit trains under the exact same rules.

    ``epoch_end_eval`` is an optional ``fn(variant, vi, epoch, policy,
    epoch_mean=...)`` invoked once at the end of each epoch (after the
    timing log, still inside the epoch loop) on the live policy.
    ``epoch_mean`` is the epoch's mean of its batch returns (the number
    already recorded in the epoch log line) — a keyword arg the
    checkpointing hooks can use without recomputing anything.  ``None``
    (the overfit path's default) means the hook never runs.

    Timing (purely diagnostic): if ``timings`` is a dict it accumulates wall
    seconds for the phase checkpoints around each batch — gen (make_batch),
    h2d (_to_D), train (_reinforce_batch; plus per-batch rollout/loss_bwd/
    step/sync counters filled by it), and greedy_batch (variant B's
    per-batch frozen-copy pass) — and appends an aggregated row per epoch.
    Each timed batch then ends with a CUDA device sync so the recorded
    section walls are wall-clock (with the sync counted in the last section
    of the batch); with ``timings=None`` there is no extra sync and no
    overhead — the spec path is unaffected.
    """
    policy_seed = vi * 1_000_000
    t0 = time.perf_counter()
    log.info("[%s] training start (policy_seed=%d, device=%s)",
             variant, policy_seed, device)
    torch.manual_seed(policy_seed)
    policy = TSPPolicy(n=n).to(device)
    generator = torch.Generator(device=str(device)).manual_seed(policy_seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    frozen = None
    baseline = 0.0
    promotions = []
    epoch_stats = []

    def _to_D(arr):
        return torch.from_numpy(
            np.asarray(arr, dtype=np.float64).astype(np.float32)).to(device)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for e in range(1, epochs + 1):
        batch_returns = []   # per-batch mean of that batch's tour returns
        base_lengths = []    # variant B: per-instance frozen greedy lengths
        t_snap = dict(timings) if timings is not None else None
        for j in range(batches_per_epoch):
            if timings is None:
                D = _to_D(make_batch(j, e))
                if variant == "moving_avg":
                    rets = _reinforce_batch(policy, optimizer, D, generator,
                                             scalar=-baseline)
                else:
                    if frozen is None:
                        frozen = TSPPolicy(n=n).to(device)
                        frozen.load_state_dict(policy.state_dict())
                        for p in frozen.parameters():
                            p.requires_grad_(False)
                    base = _greedy_lengths(frozen, D)   # (B,) per-instance
                    base_lengths.append(base)
                    rets = _reinforce_batch(
                        policy, optimizer, D, generator,
                        scalar=float(base.mean()))
            else:
                batch_timings = {
                    "gen_s": 0.0, "h2d_s": 0.0, "train_s": 0.0,
                    "train_wall_s": 0.0, "greedy_s": 0.0,
                    "rollout_s": 0.0, "loss_bwd_s": 0.0,
                    "step_s": 0.0, "sync_s": 0.0, "n": 0}
                t_a = time.perf_counter()
                D_np = make_batch(j, e)
                t_b = time.perf_counter()
                D = _to_D(D_np)
                t_c = time.perf_counter()
                if variant == "moving_avg":
                    ts = time.perf_counter()
                    rets = _reinforce_batch(
                        policy, optimizer, D, generator,
                        scalar=-baseline, timings=batch_timings)
                    batch_timings["train_wall_s"] += \
                        time.perf_counter() - ts
                else:
                    if frozen is None:
                        frozen = TSPPolicy(n=n).to(device)
                        frozen.load_state_dict(policy.state_dict())
                        for p in frozen.parameters():
                            p.requires_grad_(False)
                    ts = time.perf_counter()
                    base_lengths.append(_greedy_lengths(frozen, D))  # no-grad
                    batch_timings["greedy_s"] += time.perf_counter() - ts
                    ts = time.perf_counter()
                    rets = _reinforce_batch(
                        policy, optimizer, D, generator,
                        scalar=float(base_lengths[-1].mean()),
                        timings=batch_timings)
                    batch_timings["train_wall_s"] += \
                        time.perf_counter() - ts
                t_d = time.perf_counter()
                _sync()
                t_e = time.perf_counter()
                batch_timings["gen_s"] += t_b - t_a
                batch_timings["h2d_s"] += t_c - t_b
                batch_timings["train_s"] = batch_timings["rollout_s"] + \
                    batch_timings["loss_bwd_s"] + batch_timings["step_s"] + \
                    batch_timings["sync_s"]
                batch_timings["wall_s"] = t_e - t_a
                for k in ("gen_s", "h2d_s", "greedy_s", "train_s",
                          "train_wall_s", "rollout_s", "loss_bwd_s",
                          "step_s", "sync_s", "n"):
                    timings[k] = timings.get(k, 0.0) + \
                        batch_timings[k]
                timings.setdefault("wall_s", 0.0)
                timings["wall_s"] += batch_timings["wall_s"]
                if j % 25 == 0:
                    log.info(
                        "[%s] e%d j%03d: gen %6.2f s | h2d %5.3f | "
                        "greedy_batch %7.2f (frozen no-grad unroll) | "
                        "train %7.2f (tours %d, one Adam step for the "
                        "batched REINFORCE update: rollout %7.2f / "
                        "loss+bwd %7.2f / opt.step %6.2f / "
                        "return-sync %6.2f) | batch wall %7.2f s",
                        variant, e, j, batch_timings["gen_s"],
                        batch_timings["h2d_s"], batch_timings["greedy_s"],
                        batch_timings["train_s"], batch_timings["n"],
                        batch_timings["rollout_s"],
                        batch_timings["loss_bwd_s"],
                        batch_timings["step_s"], batch_timings["sync_s"],
                        batch_timings["wall_s"])
            batch_returns.append(float(rets.mean()))

        # Epoch stats over the epoch's batch returns (population std).
        mean_ = float(np.mean(batch_returns))
        std_ = float(np.std(batch_returns, ddof=0))
        if variant == "moving_avg":
            baseline = baseline * (1.0 - baseline_b) + mean_ * baseline_b
            epoch_stats.append(
                dict(epoch=e, mean=mean_, std=std_, baseline=baseline))
            log.info("[%s] epoch %d/%d: mean return %.2f ± %.2f, "
                     "baseline %.2f",
                     variant, e, epochs, mean_, std_, baseline)
        else:
            # End-of-epoch promotion over ALL of the epoch's instances
            # (the per-batch L_base arrays cover them exactly).
            l_base_e = float(np.concatenate(base_lengths).mean())
            t_promo = time.perf_counter()
            cur_lens = []
            for j in range(batches_per_epoch):
                D = _to_D(make_batch(j, e))
                cur_lens.append(_greedy_lengths(policy, D))
            l_cur_e = float(np.concatenate(cur_lens).mean())
            _sync()
            t_promo = time.perf_counter() - t_promo
            if timings is not None:
                timings["promo_recompute_s"] = \
                    timings.get("promo_recompute_s", 0.0) + t_promo
            threshold = math.ceil(0.95 * l_base_e)
            promoted = l_cur_e <= threshold
            if promoted:
                frozen.load_state_dict(policy.state_dict())
                promotions.append(e)
            epoch_stats.append(
                dict(epoch=e, mean=mean_, std=std_, baseline=None,
                     l_base=l_base_e, l_cur=l_cur_e,
                     promoted=promoted))
            log.info("[%s] epoch %d/%d: mean return %.2f ± %.2f, "
                     "L_base %.2f, L_cur %.2f, threshold %.0f, "
                     "promoted %s, promotions so far %d",
                     variant, e, epochs, mean_, std_, l_base_e, l_cur_e,
                     threshold, promoted, len(promotions))

        if timings is not None:
            _dl = (lambda k: float(timings.get(k, 0.0)) -
                   float(t_snap.get(k, 0.0)))
            log.info("[%s] epoch %d timing | gen(make_batch) %.2f s | "
                     "h2d %.2f s | greedy_batch %.2f s | train %.2f s "
                     "(tours %d: rollout %.2f / loss+bwd %.2f / "
                     "step %.2f / ret_sync %.2f) | promo_recompute %.2f s "
                     "| epoch wall %.2f s",
                     variant, e, _dl("gen_s"), _dl("h2d_s"),
                     _dl("greedy_s"), _dl("train_s"),
                     int(_dl("n")), _dl("rollout_s"),
                     _dl("loss_bwd_s"), _dl("step_s"),
                     _dl("sync_s"), _dl("promo_recompute_s"),
                     _dl("wall_s"))

        # End-of-epoch hook (default off: the overfit path passes nothing).
        # Runs AFTER the timing log above so that block stays pure-training
        # accounting; the hook owns its own RNG and any extra GPU memory it
        # allocates. ``epoch_mean`` is the epoch's mean of batch returns (the
        # number already recorded above), passed so the checkpointing hook
        # (the --use_eval=False protocol) can compare it against the
        # all-epoch best without recomputing anything.
        if epoch_end_eval is not None:
            epoch_end_eval(variant, vi, e, policy, epoch_mean=mean_)

    final_l_base = None
    if variant == "greedy_rollout":
        # Final L_base: the frozen copy as it stands at the end of training,
        # over the last epoch's instances.
        t_final = time.perf_counter()
        lens = []
        for j in range(batches_per_epoch):
            D = _to_D(make_batch(j, epochs))
            lens.append(_greedy_lengths(frozen, D))
        _sync()
        final_l_base = float(np.concatenate(lens).mean())
        if timings is not None:
            timings["final_lbase_s"] = time.perf_counter() - t_final

    return {
        "policy": policy,
        "policy_seed": policy_seed,
        "epochs": epoch_stats,
        "wall_s": time.perf_counter() - t0,
        "promotions": len(promotions),
        "promotion_epochs": promotions,
        "final_l_base": final_l_base,
    }


def train_variants(n, epochs, batches_per_epoch, batch_size, lr, baseline_b,
                   device, log, epoch_end_eval=None):
    """Train both variants on freshly generated instances; per-variant
    results (policy + stats).  This is the spec run path.

    ``epoch_end_eval`` is an optional ``fn(variant, vi, epoch, policy,
    epoch_mean=...)`` invoked at the end of every epoch (after the per-epoch
    timing log) on the live policy, where ``epoch_mean`` is the epoch's mean
    of its batch returns.  The default ``None`` means the hook never runs,
    keeping the no-eval behavior (e.g. the overfit path) unchanged."""
    device = torch.device(device)
    results = {}
    for vi, variant in enumerate(VARIANTS):
        policy_seed = vi * 1_000_000

        def make_batch(j, e, _ps=policy_seed, _bs=batch_size, _n=n):
            # Fresh instances for (epoch e, batch j): batch_seed scheme from
            # the task (100 always in the formula).
            batch_seed = _ps * 100000 + (e - 1) * 100 + j
            return np.stack(_gen_instances(batch_seed, _bs, _n))

        res = _train_variant_body(variant, vi, n, epochs,
                                  batches_per_epoch, lr, baseline_b, device,
                                  log, make_batch, timings={},
                                  epoch_end_eval=epoch_end_eval)
        log.info("[%s] training done in %.1f s (final epoch mean %.2f)",
                 variant, res["wall_s"], res["epochs"][-1]["mean"])
        results[variant] = res
    return results


def train_variants_overfit(n, epochs, batches_per_epoch, lr, baseline_b,
                           device, log, D_fixed):
    """Overfit test: train both variants on the SAME fixed set of instances
    (``D_fixed``, numpy ``(B, n, n)``) — every batch replays those instances —
    under the exact same batched REINFORCE + variant rules as
    ``train_variants`` (one `optimizer.step()` per batch, spec relaxation
    of 2026-11-22).  Returns per-variant results shaped like
    ``train_variants`` plus the fixed instance set, so the caller can report
    per-instance greedy tour lengths against the LKH reference.
    """
    device = torch.device(device)
    D_fixed = np.asarray(D_fixed, dtype=np.float64)
    results = {}
    for vi, variant in enumerate(VARIANTS):
        policy_seed = vi * 1_000_000

        def make_batch(j, e, _D=D_fixed):
            # Overfit: every (e, j) batch is the same fixed instances.
            return _D

        res = _train_variant_body(variant, vi, n, epochs,
                                  batches_per_epoch, lr, baseline_b, device,
                                  log, make_batch)
        log.info("[%s] overfit training done in %.1f s (final epoch mean %.2f)",
                 variant, res["wall_s"], res["epochs"][-1]["mean"])
        results[variant] = res
    results["_D_fixed"] = D_fixed
    return results
