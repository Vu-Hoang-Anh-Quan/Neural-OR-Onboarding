# REINFORCE TSP-20 (attention) results

Vanilla REINFORCE (on-policy policy gradient, gamma = 1.0, reward-to-go) on random asymmetric TSP-20 instances (n = 20; integer matrices from `gen_tsp`; reward per step = -distance(step), full-tour return = -tour length, no terminal bonus), solved by an attention/Transformer policy: an encoder of 3 blocks (each = 8-head multi-head self-attention, dim 64, dim per head 8, + FFN 64 -> 256 -> 64 (GELU), residual and pre-layer-norms) over the n node embeddings (x_i = D[i, :] / 100 -> 64), plus an averaging-node decoder (one special query = the mean of the start-node and current-node encodings, attending over all unvisited nodes; one 8-head dim-64 attention layer + FFN 64 -> 256 -> 64 + linear head to n logits, visited nodes masked to -inf).  Two baseline variants, one policy per variant: (A) `moving_avg` — baseline b <- b*(1-B) + G_epoch*B, B = 0.5, initial 0.0, updated once per epoch after its 100 batches, where G_epoch is the epoch-mean of the epoch's batch returns, subtracted as the per-return (G_t - b); (B) `greedy_rollout` — a frozen greedy copy (identical initial weights, no gradient updates except promotion) whose per-batch REINFORCE baseline is -L_base, L_base = the frozen copy's mean greedy tour length over the batch's 512 instances (so the loss uses G_t + L_base); at the end of each epoch the frozen weights are promoted (baseline_theta <- cur_theta) iff the current policy's epoch-mean greedy length L_cur over ALL of the epoch's instances satisfies L_cur <= ceil(0.95 * L_base).  UPDATE (spec relaxation, 2026-11-22: the spec's mandatory 'one update per tour' is relaxed to one update per batch — see the instructions file's Deviations section): each batch's 512 tours are sampled under the batch-start weights, and the batch receives ONE REINFORCE update (zero_grad -> loss.backward -> optimizer.step once per batch, no gradient accumulation; reward-to-go G_t = sum_{t' >= t} R_{t'} per tour; loss = the batch sum -sum_b sum_t logprob_{b,t} * (G_{b,t} - baseline), i.e. the same REINFORCE sum as per-tour updates).  1000 epochs x 100 batches x batch-size 512 fresh instances, Adam lr = 0.0001, policy_seed = variant_index * 1_000_000 (0 = moving_avg, 1000000 = greedy_rollout), device = cuda.  At the end of EVERY epoch, each variant keeps a checkpoint by best epoch-mean TRAINING return (the mean over the epoch's 100 batch returns; return = -tour length, so higher = better — no evaluation on any split, no generators): the live policy's raw state_dict is re-saved to `<variant>_best_train_return.pt` whenever that epoch mean is a new all-epoch best (epoch 1 always writes; strict improvement only).  The neural rows below are evaluated on the 1000 val + 1000 test instances AGAINST NN, NN+2-opt, and LKH-3 (reference, RUNS = 1, SEED = 1), and come from that ONE checkpoint file per variant ONLY: BOTH of a variant's rows (the greedy row and the 128-sampled row) are computed from that same checkpoint (the end-of-training weights are never used for the reported rows).
Environment: Python 3.11.17, torch 2.11.0+cu128 (CUDA available: True), NumPy 2.4.6, matplotlib 3.11.2.

## Training summary

| variant | policy_seed | final epoch mean ± std (over batch returns) | promotions | promotion epochs | final L_base | training wall (s) |
|---------|-------------|----------------------------------------------|------------|------------------|--------------|-------------------|
| moving_avg | 0 | -356.0 ± 3.1 | n/a | n/a | n/a | 10689.9 |
| greedy_rollout | 1000000 | -357.8 ± 3.2 | 19 | [38, 46, 49, 51, 54, 59, 65, 74, 86, 101, 123, 147, 181, 216, 249, 302, 400, 595, 931] | 358.4 | 25825.6 |

## Results (val)

| method | mean | std | q1 | median | q3 | gap vs LKH-3 | wall (s) |
|--------|------|-----|----|--------|----|--------------|---------|
| NN | 322.3 | 61.8 | 276.0 | 320.0 | 362.0 | +148.9 | 0.037 |
| NN+2opt | 293.8 | 53.4 | 255.0 | 292.0 | 330.0 | +120.4 | 0.331 |
| LKH-3 | 173.4 | 27.1 | 154.0 | 172.0 | 191.0 | +0.0 | 0.000 |
| moving_avg (greedy) | 355.1 | 0.0 | 307.8 | 351.0 | 400.0 | +181.7 | 7.980 |
| moving_avg (128-sampled) | 271.1 | 46.9 | 238.0 | 268.0 | 301.0 | +97.7 | 1138.686 |
| greedy_rollout (greedy) | 360.2 | 0.0 | 314.0 | 357.0 | 404.0 | +186.9 | 7.946 |
| greedy_rollout (128-sampled) | 273.2 | 49.2 | 239.0 | 269.0 | 306.0 | +99.8 | 1138.097 |

## Results (test)

| method | mean | std | q1 | median | q3 | gap vs LKH-3 | wall (s) |
|--------|------|-----|----|--------|----|--------------|---------|
| NN | 315.7 | 61.5 | 272.0 | 316.0 | 356.0 | +144.5 | 0.035 |
| NN+2opt | 289.1 | 52.2 | 252.0 | 288.0 | 325.0 | +117.9 | 0.320 |
| LKH-3 | 171.2 | 26.0 | 153.0 | 170.0 | 188.0 | +0.0 | 0.000 |
| moving_avg (greedy) | 347.7 | 0.0 | 306.0 | 347.0 | 391.0 | +176.5 | 7.974 |
| moving_avg (128-sampled) | 268.2 | 48.2 | 233.8 | 266.0 | 303.0 | +97.0 | 1139.962 |
| greedy_rollout (greedy) | 355.0 | 0.0 | 304.0 | 353.0 | 406.0 | +183.7 | 7.984 |
| greedy_rollout (128-sampled) | 271.0 | 49.8 | 236.0 | 268.0 | 301.0 | +99.7 | 1135.353 |

## Runtimes

| quantity | value |
|----------|-------|
| training, moving_avg | 10689.9 s |
| training, greedy_rollout | 25825.6 s |
| LKH labeling, val | 0.000 s |
| LKH labeling, test | 0.000 s |
| LKH labeling, total | 0.000 s |
| eval, val | 2293.077 s |
| eval, test | 2291.627 s |
| total | 41100.3 s |
| checkpoints, moving_avg | `src/reinforce_tsp20/checkpoints/moving_avg_best_train_return.pt` (best TRAINING-epoch mean return; both the greedy and the 128-sampled rows are computed from this single checkpoint) |
| checkpoints, greedy_rollout | `src/reinforce_tsp20/checkpoints/greedy_rollout_best_train_return.pt` (best TRAINING-epoch mean return; both the greedy and the 128-sampled rows are computed from this single checkpoint) |

## Configuration

| key | value |
|-----|-------|
| n | 20 |
| epochs | 1000 |
| batches per epoch | 100 |
| batch size | 512 |
| lr | 0.0001 |
| gamma | 1.0 |
| B (moving_avg baseline const) | 0.5 |
| policy seeds | 0 (moving_avg), 1000000 (greedy_rollout) |
| device | cuda |
| val_n | 1000 |
| test_n | 1000 |
| eval base offset (BASE) | 1000000000 |
| relabel | False |
| use_eval | False |

## Environment

Python 3.11.17, torch 2.11.0+cu128 (CUDA available: True), NumPy 2.4.6, matplotlib 3.11.2.

## Hardware specification
CPU: Ryzen 7 9700 X
GPU: RTX 5060 Ti
Device used (resolved --device=auto): cuda
eval base offset (BASE): 1000000000

## Notes
Even after 1000 epochs, the two variants haven't converged yet, as you can observe in the two returns curve `.png` files. However, the results is acceptable, and the author do not have the computational resources to run another 2000 epochs training.