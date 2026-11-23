"""The TSP-20 attention policy: encoder + averaging-node decoder.

A policy is one (variant, run) pair.  It exposes:
- ``encode(D)``  : scaled distance rows -> encoded node set H (B, n, d).
- ``act(obs...)``: one step — given the encoded node set, the start node, the
  current node and the visited mask, return (action, logprob, probs).  The
  action is either drawn from the CDF through the policy's ``torch.Generator``
  (``mode='sample'``) or taken as the argmax over unvisited nodes
  (``mode='greedy'``).
- ``rollout(...)``: unroll a full n-step tour (argmax or sampled), returning the
  tour, the per-step reward-to-go inputs, the per-step log-probs (n-1 choice
  steps) and the total tour length (the closing return edge is included).

Tour convention: the tour starts at a fixed start node (0), visits the other
n-1 nodes in the policy-chosen order (n-1 choice steps), then closes with the
forced return edge to the start — so a full tour is exactly n steps / n edges,
the return is the negative tour length, and there is no terminal bonus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import attention as att


class TSPPolicy(nn.Module):
    def __init__(self, n=20, d_model=64, nhead=8, num_layers=3, ffn_dim=256):
        super().__init__()
        self.n = n
        self.d_model = d_model
        self.encoder = att.Encoder(n, d_model, nhead, num_layers, ffn_dim)
        self.decoder = att.Decoder(d_model, nhead, ffn_dim, n)
        self._start_default = 0

    # ------------------------------------------------------------------ encode
    def encode(self, D):
        """D (B, n, n) full distance matrix -> H (B, n, d_model) encoded nodes."""
        X = D / 100.0
        return self.encoder(X)

    # ------------------------------------------------------------------- query
    def _query(self, H, start, cur):
        """Averaging-node query: mean of the start- and current-node encodings.

        ``start``: int (fixed) or (B,) long tensor; ``cur``: int or (B,) long
        tensor.  Returns (B, 1, d_model).
        """
        B = H.shape[0]
        d = self.d_model
        if isinstance(start, int):
            Hs = H[:, start, :].unsqueeze(1)
        else:
            Hs = H.gather(1, start.view(B, 1, 1).expand(B, 1, d))
        if isinstance(cur, int):
            Hc = H[:, cur, :].unsqueeze(1)
        else:
            Hc = H.gather(1, cur.view(B, 1, 1).expand(B, 1, d))
        return 0.5 * (Hs + Hc)

    # --------------------------------------------------------------------- act
    def act(self, H, start, cur, visited, mode="sample", generator=None):
        """One step.  Returns (action (B,), logprob (B,), probs (B, n))."""
        q = self._query(H, start, cur)                    # (B, 1, d)
        logits, probs = self.decoder(q, H, visited)       # (B, n), (B, n)
        B = H.shape[0]
        if mode == "sample":
            # CDF draw through the policy's own torch.Generator (one uniform
            # per row per step).  probs.sum(-1) is ~1 to rounding, so the
            # cumulative row sum is normalised by its last entry to make the
            # final column exactly 1.0 — the draw then always lands on or
            # before the last unvisited column (no all-false mask possible).
            # Selection probability of each node equals its probability —
            # identical to Categorical sampling.
            u = torch.rand(B, 1, device=probs.device, generator=generator)
            c = probs.cumsum(-1) / probs.sum(-1, keepdim=True)  # (B, n), c[:,-1] = 1
            action = (c >= u).to(torch.long).argmax(dim=1)      # (B,)
        else:  # greedy: argmax over unvisited (visited logits are -inf)
            action = logits.argmax(dim=1)
        logprob = Categorical(probs=probs).log_prob(action)
        return action, logprob, probs

    # --------------------------------------------------------------- rollout
    def rollout(self, D, mode="sample", generator=None, start=0):
        """Unroll a full tour per row of D (B, n, n).

        Returns a dict:
          tour      (B, n) long  — start node first, then the n-1 chosen nodes
          logprobs  list of (n-1) tensors (B,) — one per choice step
          rewards   (B, n) float — R[0..n-2] move rewards, R[n-1] return edge
          length    (B,) float   — total tour length (sum of all n edges)
          return    (B,) float   — -length (the REINFORCE return)
        """
        n = self.n
        B = D.shape[0]
        dev = D.device
        H = self.encode(D)                                # (B, n, d)
        visited = torch.zeros(B, n, dtype=torch.bool, device=dev)
        if isinstance(start, int):
            start_t = torch.full((B,), start, dtype=torch.long, device=dev)
        else:
            start_t = start.to(dev)
        visited[torch.arange(B, device=dev), start_t] = True
        cur = start_t.clone()
        tour = torch.zeros(B, n, dtype=torch.long, device=dev)
        tour[:, 0] = start_t
        logprobs, reward_moves = [], []
        for t in range(n - 1):                            # n-1 choice steps
            action, logprob, _ = self.act(H, start_t, cur, visited, mode,
                                          generator)
            logprobs.append(logprob)
            Dcur = torch.gather(D, 1, cur.view(B, 1, 1).expand(B, 1, n))  # (B,1,n)
            Dcur = Dcur.squeeze(1)
            dist = torch.gather(Dcur, 1, action.view(B, 1)).squeeze(1)  # (B,)
            reward_moves.append(-dist)
            # Fresh tensor for the next step: the mask passed to act() is
            # saved in the autograd graph, so it must never be modified in
            # place afterwards (the clone is updated, the saved copy is not).
            visited = visited.clone()
            visited[torch.arange(B, device=dev), action] = True
            cur = action
            tour[:, t + 1] = action
        # closing return edge (forced back to the start)
        Dcur = torch.gather(D, 1, cur.view(B, 1, 1).expand(B, 1, n)).squeeze(1)
        dist_ret = torch.gather(Dcur, 1, start_t.view(B, 1)).squeeze(1)  # (B,)
        rewards = torch.stack(reward_moves + [-dist_ret], dim=1)  # (B, n)
        length = -rewards.sum(dim=1)
        return {
            "tour": tour,
            "logprobs": logprobs,
            "rewards": rewards,
            "length": length,
            "return": -length,
        }
