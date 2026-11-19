# REINFORCE CartPole-v1 results

Vanilla REINFORCE (on-policy policy gradient, gamma = 1.0, reward-to-go) on CartPole-v1, two variants: `no_baseline` and `moving_avg` (per-episode moving-average baseline, B = 0.5, initial baseline 0.0). 50 episodes per (variant, seed) run, seeds [0, 1] — same episode index comparable across variants/seeds. Model: Linear(4,64) -> ReLU -> Linear(64,2) -> Softmax; Adam lr = 0.01; one optimizer update per episode over the full-episode batch (loss = -sum_t logprob_t * (G_t - baseline), G_t = sum_{t' >= t} R_{t'}). CartPole-v1 rewards +1.0 per step (the terminal step included), so return = episode length (<= 500); truncated (500-step) episodes are kept as-is, no bootstrap or bonus.
Environment: Python 3.11.17, torch 2.11.0+cu128, gymnasium 1.4.0, NumPy 2.4.6, matplotlib 3.11.2.

## Best results per variant

| variant | best mean return | at episode | mean return at final episode |
|---------|------------------|------------|------------------------------|
| no_baseline | 138.5 | 36 | 57.0 |
| moving_avg | 107.0 | 26 | 28.5 |

Best mean return = max over episode index of the per-episode mean over the 2 seeds (std ddof = 0); curve in `returns_curve.png` (mean line, filled band = mean ± 1 std).

## Runtimes

| quantity | value |
|----------|-------|
| training, no_baseline (2 x 50 episodes) | 0.7 s total, 0.4 s/run avg |
| training, moving_avg (2 x 50 episodes) | 0.6 s total, 0.3 s/run avg |
| total (all 4 runs) | 1.9 s |

## Configuration

| key | value |
|-----|-------|
| episodes | 50 |
| lr | 0.01 |
| baseline_b | 0.5 |
| seeds | [0, 1] |
| gamma | 1.0 |
| variants | ['no_baseline', 'moving_avg'] |
| device | cpu |

## Environment

Python 3.11.17, torch 2.11.0+cu128, gymnasium 1.4.0, NumPy 2.4.6, matplotlib 3.11.2.

## Hardware specification
CPU: Ryzen 7 9700 X
GPU: RTX 5060 Ti
Device used (resolved --device=cpu): cpu
