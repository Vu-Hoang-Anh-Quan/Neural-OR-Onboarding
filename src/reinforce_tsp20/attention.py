"""All attention / Transformer code for the REINFORCE TSP-20 solver.

Nothing else in the task defines attention layers; the policy in ``policy.py``
composes exactly these modules:

- input projection  (Linear n -> d_model): each node's feature row
  ``x_i = D[i, :] / 100`` (n-dim) is mapped to dim d_model;
- 3 encoder blocks: each = [multi-head self-attention (8 heads, dim 64) +
  FFN 64 -> 256 -> 64 (GELU)] with residual connections and pre-layer-norms;
- the "averaging-node" decoder: given a single query (the start + current-node
  aggregation) it attends over all unvisited encoded nodes (visited nodes
  masked out via an additive -inf bias / key-padding mask), then one
  attention layer (8 heads, dim 64) + FFN (64 -> 256 -> 64) + linear head to
  n logits with the visited mask folded in.  Log-attention weights are not
  stored; only the next-node distribution is returned.

Masking convention: a *visited* mask (bool, True = already visited / must-not
be selected).  Mask helpers live here: ``key_bias`` (per-key additive bias,
(n,) or (B, n)) and the per-instance ``masked_fill`` inside ``_mha``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def key_bias(visited):
    """Per-key additive bias: 0.0 where attendable, -inf on ``visited`` nodes.

    ``visited`` (n, bool) or (B, n, bool) -> bias of the same leading shape.
    This is the additive form of the key-padding-mask / -inf-in-softmax
    convention; either this bias or ``masked_fill`` in ``_mha`` may be used.
    """
    bias = torch.zeros(visited.shape, device=visited.device,
                      dtype=torch.get_default_dtype())
    return bias.masked_fill(visited.to(torch.bool), float("-inf"))


def _mha(q, kv, nhead, visited=None):
    """Multi-head attention.  Q from ``q`` (B, Lq, d), K/V from ``kv``
    (B, nk, d).  ``visited`` (B, nk, bool) masks key columns (per-instance
    -inf before the softmax).  Pre-LN is applied by the caller before
    invoking this; the raw LN-normalised Q/K/V are expected.  Returns
    (out (B, Lq, d), attn (B, Lq, nk)).
    """
    B, Lq, d = q.shape
    nk = kv.shape[1]
    h = nhead
    hd = d // h
    qh = q.view(B, Lq, h, hd).permute(0, 2, 1, 3) * (1.0 / math.sqrt(hd))
    kh = kv.view(B, nk, h, hd).permute(0, 2, 1, 3)
    vh = kv.view(B, nk, h, hd).permute(0, 2, 1, 3)
    scores = torch.matmul(qh, kh.transpose(-1, -2))       # (B, h, Lq, nk)
    if visited is not None:
        scores = scores.masked_fill(
            visited.to(torch.bool).unsqueeze(1).unsqueeze(2),
            float("-inf"))                                # (B,1,1,nk) broadcast
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, vh).permute(0, 2, 1, 3).reshape(B, Lq, d)
    return out, attn


class FFN(nn.Module):
    """Two-layer feed-forward: dim -> ffn_dim (GELU) -> dim."""

    def __init__(self, d_model=64, ffn_dim=256):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class EncoderBlock(nn.Module):
    """Pre-LN encoder block: residual self-attention + residual FFN over all
    n nodes (no mask)."""

    def __init__(self, d_model=64, nhead=8, ffn_dim=256):
        super().__init__()
        self.ln_a = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.nhead = nhead
        self.ffn = FFN(d_model, ffn_dim)

    def forward(self, x):
        h = self.ln_a(x)
        attn, _ = _mha(h, h, self.nhead)
        x = x + attn
        x = x + self.ffn(self.ln_f(x))
        return x


class Encoder(nn.Module):
    """Input projection (n -> d_model) + 3 encoder blocks.

    ``forward(X)`` with X (B, n, n) = the scaled distance rows
    ``x_i = D[i, :] / 100``; returns (B, n, d_model).
    """

    def __init__(self, n=20, d_model=64, nhead=8, num_layers=3, ffn_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(n, d_model)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, nhead, ffn_dim) for _ in range(num_layers)])

    def forward(self, X):
        h = self.input_proj(X)
        for blk in self.blocks:
            h = blk(h)
        return h


class Decoder(nn.Module):
    """Averaging-node decoder (action head).

    ``forward(q, H, visited)``: ``q`` (B, 1, d) = start- + current-node
    aggregation; ``H`` (B, n, d) = encoded node set (re-read unchanged every
    step); ``visited`` (B, n, bool) marks nodes that must be masked out.
    Returns ``(logits, probs)`` both (B, n), with the visited mask folded into
    the logits (-inf before the softmax).
    """

    def __init__(self, d_model=64, nhead=8, ffn_dim=256, n_nodes=20):
        super().__init__()
        self.nhead = nhead
        self.n_nodes = n_nodes
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, ffn_dim)
        self.head = nn.Linear(d_model, n_nodes)

    def forward(self, q, H, visited):
        visited = visited.to(torch.bool)
        qn = self.ln_q(q)
        Hn = self.ln_kv(H)
        attn, _ = _mha(qn, Hn, self.nhead, visited=visited)  # (B,1,n)
        agg = q + attn                                     # (B,1,d)
        agg = agg + self.ffn(self.ln_f(agg))
        logits = self.head(agg).squeeze(1)                # (B, n)
        logits = logits.masked_fill(visited, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return logits, probs
