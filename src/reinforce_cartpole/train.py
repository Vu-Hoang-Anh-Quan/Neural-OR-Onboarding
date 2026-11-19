"""REINFORCE training for CartPole-v1 — 10 distinct policies (2 variants x 5
seeds), each trained independently for 100 epochs of 10 fresh env episodes.

Policy model (markdown/reinforce_cartpole_instructions.md): a policy is a
(variant, seed) pair; the 10 policies are all distinct (distinct initial
weights, distinct action stream, distinct env-reset stream) and train
independently, back to back (no cross-policy synchronization). Each policy
runs `epochs` epochs; in each epoch it samples `per_epoch` fresh env episodes
("instances") from its own env, taking one classic REINFORCE update per env
episode.

Algorithm invariants (same file):
- gamma = 1.0, reward-to-go G_t = sum_{t' >= t} R_{t'} (includes R_t).
- An env episode is kept as-is in every outcome: pole-fall (terminated) and
  500-step time-limit (truncated) alike. Truncated episodes are *not* zeroed
  or bootstrapped; return of a fully-survived episode is exactly 500.
  step() is never called after terminated=True.
- Baseline (moving_avg variant only) updates **once per epoch, after the
  epoch's env episodes finish**: b <- b * (1 - B) + G_epoch * B, B = 0.5,
  initial b = 0.0, where G_epoch = the mean of the epoch's per_epoch env
  episode returns. Each of the 5 moving_avg policies has its own baseline.
- Classic REINFORCE, per-env-episode full batch: each env episode accumulates
  one loss, -(sum_t logprob_t * (G_t - baseline)), and the update triplet
  zero_grad -> backward -> optimizer.step is taken at the end of that episode.
- Reproducibility: policy (variant index vi, seed s) is fully determined by
  policy_seed = vi * 1_000_000 + s — initial weights set with
  torch.manual_seed(policy_seed), the action torch.Generator seeded with the
  same key, and its env reset with seed
  policy_seed * 100000 + (e - 1) * per_epoch + i for instance i of epoch e
  (e 1-based, i 0-based), so every (policy, epoch, instance) gets a distinct
  reset seed.
"""

import logging
import time

import gymnasium as gym
import numpy as np
import torch

import policy as policy_mod

log = logging.getLogger("reinforce_cartpole.train")

VARIANTS = ("no_baseline", "moving_avg")

# Distinct per-variant block so no two policies (any variant, any seed) share
# a policy seed: vi * 1_000_000 + s.
_POLICY_SEED_OFFSET = 1_000_000


def run_episode(model, optimizer, env, obs, device, generator, baseline):
    """Run ONE full env episode under `model` and update the optimizer.

    `obs` is the observation of the just-reset env (the info part of the
    reset tuple is dropped before the call); the episode ends on the first
    terminated/truncated step (never stepped past). The update is the
    classic REINFORCE full-episode batch (gamma = 1.0): reward-to-go
    G_t = sum_{t' >= t} R_{t'}; loss = -sum_t logprob_t * (G_t - baseline);
    zero_grad -> backward -> optimizer.step.

    Returns (returns_g, steps): returns_g = G_0 (the total episode return =
    episode length in this environment), steps = number of real steps taken.
    """
    rewards = []
    logprobs = []
    obs_t = obs
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, logprob = model.act(obs_t, device, generator)
        obs_t, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(float(reward))
        logprobs.append(logprob)

    loss_terms = []
    g = 0.0
    for r_t, logprob in zip(rewards[::-1], logprobs[::-1]):
        g += r_t
        loss_terms.append(-logprob * (g - baseline))
    model.zero_grad()
    torch.stack(loss_terms).sum().backward()
    optimizer.step()

    return g, len(rewards)


def _train_policy(variant, seed, policy_seed, env, model, optimizer,
                  generator, epochs, per_epoch, baseline_b, device):
    """Train one policy for `epochs` epochs of `per_epoch` fresh env episodes.

    Within an epoch the 10 instances are sampled back to back: env reset with
    a distinct seed (see module docstring), a full env episode run under the
    current weights, one end-of-episode REINFORCE update. After the epoch's
    instances the moving_avg baseline is updated with the epoch-mean return
    (per invariants).

    Returns (means, stds, baseline_trace, steps_means), each (epochs,) arrays:
    per-epoch mean return, per-epoch population std (ddof=0) over the
    per_epoch instances, the baseline value after the epoch's update, and the
    mean env-episode length in the epoch.
    """
    means = np.zeros(epochs, dtype=np.float64)
    stds = np.zeros(epochs, dtype=np.float64)
    baseline_trace = np.zeros(epochs, dtype=np.float64)
    steps_means = np.zeros(epochs, dtype=np.float64)
    baseline = 0.0
    use_baseline = variant == "moving_avg"
    for e in range(1, epochs + 1):
        returns_e = []
        steps_e = []
        for i in range(per_epoch):
            reset_seed = policy_seed * 100000 + (e - 1) * per_epoch + i
            obs = env.reset(seed=reset_seed)[0]
            g, steps = run_episode(model, optimizer, env, obs, device,
                                   generator, baseline)
            returns_e.append(g)
            steps_e.append(steps)
        mean_e = float(np.mean(returns_e))
        std_e = float(np.std(returns_e, ddof=0))
        if use_baseline:
            baseline = baseline * (1.0 - baseline_b) + mean_e * baseline_b
        means[e - 1] = mean_e
        stds[e - 1] = std_e
        baseline_trace[e - 1] = baseline
        steps_means[e - 1] = float(np.mean(steps_e))
        log.info("policy %s seed %d | epoch %3d/%3d | g %.1f ± %.1f",
                 variant, seed, e, epochs, mean_e, std_e)
    return means, stds, baseline_trace, steps_means


def train(variants=VARIANTS, seeds=(0, 1, 2, 3, 4), epochs=100, per_epoch=10,
          lr=0.01, baseline_b=0.5, device=torch.device("cpu")):
    """Train all (variant, seed) policies independently, back to back: each
    policy gets its own env, model, optimizer, baseline, and action
    generator, then runs `epochs` epochs of `per_epoch` fresh env episodes.

    Returns a data dict for the orchestrator:
      variants, seeds, epochs, per_epoch, lr, baseline_b,
      runs: {variant: {seed: {"mean" (E,), "std" (E,), "baseline" (E,),
                              "steps_mean" (E,)}}},
      policy_stats: {variant: {seed: {"best": {"g", "g_std", "epoch"},
                                      "final_g", "final_g_std"}}},
      best: {variant: {"seed", "epoch", "g", "g_std"}} — the variant's
      best-seed policy (the policy with the highest best-epoch mean g) and
      that policy's best epoch (the epoch with the highest per-epoch mean g,
      earliest one on a tie),
      runtimes: {variant: seconds} (summed over that variant's policies, all
      their epochs).
    """
    runs = {v: {} for v in variants}
    policy_stats = {v: {} for v in variants}
    runtimes = {v: 0.0 for v in variants}
    best = {v: {} for v in variants}
    for vi, variant in enumerate(variants):
        for s in seeds:
            policy_seed = vi * _POLICY_SEED_OFFSET + s
            env = gym.make("CartPole-v1")
            torch.manual_seed(policy_seed)  # distinct init weights per policy
            model = policy_mod.CartPolePolicy().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            generator = torch.Generator(device=device).manual_seed(policy_seed)
            t0 = time.perf_counter()
            means, stds, baseline_trace, steps_means = _train_policy(
                variant, s, policy_seed, env, model, optimizer, generator,
                epochs, per_epoch, baseline_b, device)
            elapsed = time.perf_counter() - t0
            env.close()
            runtimes[variant] += elapsed
            runs[variant][s] = {
                "mean": means, "std": stds,
                "baseline": baseline_trace, "steps_mean": steps_means,
            }
            best_idx = int(np.argmax(means))  # earliest epoch on a tie
            policy_stats[variant][s] = {
                "best": {"g": float(means[best_idx]),
                         "g_std": float(stds[best_idx]),
                         "epoch": best_idx + 1},
                "final_g": float(means[-1]),
                "final_g_std": float(stds[-1]),
            }
            log.info(
                "policy %s seed %d done | %d epochs in %.1f s | best g %.1f "
                "at epoch %d | final g %.1f ± %.1f",
                variant, s, epochs, elapsed, means[best_idx], best_idx + 1,
                means[-1], stds[-1],
            )
        best_seed = max(seeds, key=lambda s: policy_stats[variant][s]["best"]["g"])
        bs = policy_stats[variant][best_seed]["best"]
        best[variant] = {"seed": best_seed, "epoch": bs["epoch"],
                         "g": bs["g"], "g_std": bs["g_std"]}

    return {"variants": list(variants), "seeds": list(seeds),
            "epochs": epochs, "per_epoch": per_epoch,
            "lr": lr, "baseline_b": baseline_b,
            "runs": runs, "policy_stats": policy_stats, "best": best,
            "runtimes": runtimes}
