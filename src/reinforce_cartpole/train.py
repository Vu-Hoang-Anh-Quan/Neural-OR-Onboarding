"""REINFORCE training for CartPole-v1 — both variants, one full pass per
(variant, seed) run. Called by orchestrator.py (the only setup_logging caller),
so this module only uses its named logger.

Algorithm invariants (see markdown/reinforce_cartpole_instructions.md):
- gamma = 1.0, reward-to-go G_t = sum_{t' >= t} R_{t'} (includes R_t).
- Episodes are kept as-is in every outcome: pole-fall (terminated) and
  500-step time-limit (truncated) alike. Truncated episodes are *not*
  zeroed or bootstrapped; return of a fully-survived episode is exactly 500.
  step() is never called after terminated=True.
- Baseline (moving_avg variant only) updates per episode AFTER the episode
  ends: b <- b * (1 - B) + G_0 * B, B = 0.5, initial b = 0.0. The baseline
  is subtracted from the reward-to-go of every step of the episode.
- Classic REINFORCE, full-episode batch: the episode's steps accumulate one
  loss, S_t = sum over t of -(logprob_t * (G_t - baseline)), and the update
  triplet zero_grad -> backward -> optimizer.step is taken at the end of
  EVERY episode (one update per episode, over the full-episode batch).
- Reproducibility: torch.manual_seed(seed) is set right before the episode
  loop of every (variant, seed) run; each run is independent of the others,
  so the same episode index is comparable across variants and seeds.
"""

import logging
import time

import gymnasium as gym
import numpy as np
import torch

import policy as policy_mod

log = logging.getLogger("reinforce_cartpole.train")

VARIANTS = ("no_baseline", "moving_avg")


def run_policy(seed, episodes, lr, baseline_b, variant, device):
    """Run ONE full (variant, seed) pass and return per-episode arrays.

    Returns a dict with keys: episodes (int), returns (float64), baseline
    (float64 — the per-episode moving-average values; all 0 for no_baseline),
    steps (int — real steps per episode, 1..500), run_time_s (wall).
    """
    env = gym.make("CartPole-v1")
    torch.manual_seed(seed)

    net = policy_mod.CartPolePolicy().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    baseline = 0.0

    returns = np.zeros(episodes, dtype=np.float64)
    steps = np.zeros(episodes, dtype=np.int64)
    baseline_after = np.zeros(episodes, dtype=np.float64)

    t0 = time.perf_counter()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed)  # seed the env per episode for determinism
        episode_rewards = []
        episode_logprobs = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, logprob = net.act(obs, device)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_rewards.append(float(reward))
            episode_logprobs.append(logprob)

        # Reward-to-go with gamma = 1.0: G_t = R_t + R_{t+1} + ... (R in {0, 1},
        # so G_t is the episode length minus t). Accumulate the gradient across
        # all steps and step the optimizer once per episode (classic REINFORCE).
        policy_loss_terms = []
        g = 0.0
        for r_t, logprob in zip(episode_rewards[::-1], episode_logprobs[::-1]):
            g += r_t
            policy_loss_terms.append(-logprob * (g - baseline))  # maximize return
        net.zero_grad()
        total_loss = torch.sum(torch.stack(policy_loss_terms))
        total_loss.backward()
        optimizer.step()

        returns[ep] = g  # = G_0 = episode length (return)
        steps[ep] = len(episode_rewards)
        # Baseline updates per episode AFTER the episode ends, from G_0.
        if variant == "moving_avg":
            baseline = baseline * (1.0 - baseline_b) + returns[ep] * baseline_b
        baseline_after[ep] = baseline

    run_time_s = time.perf_counter() - t0
    env.close()
    log.info(
        "variant=%s seed=%d done | %d episodes in %.1f s | "
        "best return %.0f (ep %+d), final return %.0f, final baseline %.1f",
        variant, seed, ep + 1, run_time_s,
        returns.max(), int(np.argmax(returns)) + 1, returns[-1],
        (baseline if variant == "moving_avg" else 0.0),
    )
    return {
        "episodes": episodes,
        "returns": returns,
        "baseline": baseline_after,
        "steps": steps,
        "run_time_s": run_time_s,
    }


def train(variants=VARIANTS, seeds=(0, 1, 2, 3, 4), episodes=1000,
          lr=0.01, baseline_b=0.5, device=torch.device("cpu")):
    """Loop over all (variant, seed) runs; return the collected data dict.
    Each (variant, seed) run is independent, so the same episode index is
    comparable across variants and seeds."""
    runs = {}
    for variant in variants:
        rows = {}
        for seed in seeds:
            res = run_policy(seed, episodes, lr, baseline_b, variant, device)
            rows[seed] = res
        runs[variant] = rows
    return {"runs": runs, "episodes": episodes, "lr": lr,
            "baseline_b": baseline_b, "seeds": list(seeds)}
