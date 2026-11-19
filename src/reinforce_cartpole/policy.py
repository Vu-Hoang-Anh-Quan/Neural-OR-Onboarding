"""MLP policy for REINFORCE on CartPole-v1 (shared by both variants).

Architecture (fixed by the task invariants, markdown/reinforce_cartpole_instructions.md):
obs (4-dim) -> Linear(4, 64) -> ReLU -> Linear(64, 2) -> Softmax -> action probs.
Acting samples from a Categorical distribution so the trainer gets a proper
per-step log-probability (float32) whose graph reaches the weights.
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

    def act(self, obs, device):
        """Sample one action for a single obs.

        obs: (4,) numpy/float array from the env; moved to `device` here
        (the env always produces CPU tensors).
        Returns (action, logprob), both float32 tensors on `device`: the
        sampled action (consumed via .item() for env.step) and its log-prob,
        which carries the parameter graph so gradients flow (one full-episode
        update per episode, per the invariants).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        probs = self.net(obs_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)
