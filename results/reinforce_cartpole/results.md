# REINFORCE CartPole-v1 results

Vanilla REINFORCE (on-policy policy gradient, gamma = 1.0, reward-to-go) on CartPole-v1, two variants — `no_baseline` and `moving_avg` (moving-average baseline updated once per epoch, b <- b*(1-B) + G_epoch*B, B = 0.5, initial 0.0) — with 5 seed policies per variant: 10 distinct policies, each trained independently for 50 epochs of 10 fresh env episodes ("instances"). A policy's epoch return = mean of its 10 instance returns of that epoch (std = population std, ddof = 0). Model: Linear(4,64) -> ReLU -> Linear(64,2) -> Softmax; Adam lr = 0.01; one optimizer update per env episode over its full-episode batch (loss = -sum_t logprob_t * (G_t - baseline), G_t = sum_{t' >= t} R_{t'}). CartPole-v1 rewards +1.0 per step (the terminal step included), so each episode return = its length (<= 500); truncated (500-step) episodes are kept as-is, no bootstrap or bonus.
Environment: Python 3.11.17, torch 2.11.0+cu128, gymnasium 1.4.0, NumPy 2.4.6, matplotlib 3.11.2.

## Best results per variant (best of the 5 seed policies)

| variant | best seed | best-g (mean ± std) | at epoch | final g (mean ± std) |
|---------|-----------|---------------------|----------|----------------------|
| no_baseline | 0 | 500.0 ± 0.0 | 35 | 466.5 ± 100.5 |
| moving_avg | 0 | 500.0 ± 0.0 | 22 | 500.0 ± 0.0 |

All per-epoch lines of `returns_curve.png` are the mean ± 1 std of the 10 instance returns of the best-seed policy of each variant; 'best seed' = the policy with the highest best-epoch mean g; 'best epoch' = the epoch with the highest per-epoch mean g of that policy.

## Runtimes

| quantity | value |
|----------|-------|
| training, no_baseline (5 seed policies x 50 epochs) | 247.0 s |
| training, moving_avg (5 seed policies x 50 epochs) | 277.8 s |
| total (all 10 policies) | 525.3 s |

## Configuration

| key | value |
|-----|-------|
| epochs | 50 |
| per_epoch | 10 |
| lr | 0.01 |
| baseline_b | 0.5 |
| seeds | [0, 1, 2, 3, 4] |
| gamma | 1.0 |
| variants | ['no_baseline', 'moving_avg'] |
| device | cuda |

## Environment

Python 3.11.17, torch 2.11.0+cu128, gymnasium 1.4.0, NumPy 2.4.6, matplotlib 3.11.2.

## Hardware specification
CPU: Ryzen 7 9700 X
GPU: RTX 5060 Ti
Device used (resolved --device=auto): cuda
