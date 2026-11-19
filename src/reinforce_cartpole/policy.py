"""MLP policy for REINFORCE on CartPole-v1 (shared by both variants).

Architecture (fixed by the task invariants, markdown/reinforce_cartpole_instructions.md):
obs (4-dim) -> Linear(4, 64) -> ReLU -> Linear(64, 2) -> Softmax -> action probs.
Acting samples from a Categorical distribution using the caller's torch.Generator,
which both yields a proper per-step log-probability (float32, graph reaching the
weights) and keeps each policy's action stream reproducible from its own seed,
independent of every other policy's episodes.
"""

import torch
import torch.nn as nn


class CartPolePolicy(nn.Module):
    """Two-layer MLP returning a Categorical over the 2 CartPole actions."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

    def act(self, obs, device, generator):
        """Sample one action for a single obs.

        obs: (4,) float array from the env. Returns (action, logprob), both
        float32 on `device`: the sampled action (consumed via .item() for
        env.step) and its log-prob, which carries the parameter graph so
        gradients flow (one end-of-episode update per env episode, per
        invariants). This torch build's Categorical.sample() takes no
        generator, so the action is drawn from the policy's generator via CDF
        sampling (exact Categorical sampling, equivalent to
        multinomial(generator=...)); log-prob = Categorical(probs).log_prob(action),
        which carries the parameter graph for the end-of-episode update.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        probs = self.net(obs_t)
        dist = torch.distributions.Categorical(probs)
        u = torch.rand((), generator=generator, dtype=torch.float32, device=device)
        with torch.no_grad():
            cdf = probs.double().cumsum(dim=-1)
        action = (cdf >= u.double()).long().argmax()  # first bin with F_k >= u
        return action, dist.log_prob(action)
