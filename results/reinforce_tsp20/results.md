# REINFORCE TSP-20 (attention) results

Vanilla REINFORCE (on-policy policy gradient, gamma = 1.0, reward-to-go) on random asymmetric TSP-20 instances (n = 20; integer matrices from `gen_tsp`; reward per step = -distance(step), full-tour return = -tour length, no terminal bonus), solved by an attention/Transformer policy: an encoder of 3 blocks (each = 8-head multi-head self-attention, dim 64, dim per head 8, + FFN 64 -> 256 -> 64 (GELU), residual and pre-layer-norms) over the n node embeddings (x_i = D[i, :] / 100 -> 64), plus an averaging-node decoder (one special query = the mean of the start-node and current-node encodings, attending over all unvisited nodes; one 8-head dim-64 attention layer + FFN 64 -> 256 -> 64 + linear head to n logits, visited nodes masked to -inf).  Two baseline variants, one policy per variant: (A) `moving_avg` — baseline b <- b*(1-B) + G_epoch*B, B = 0.5, initial 0.0, updated once per epoch after its 100 batches, where G_epoch is the epoch-mean of the epoch's batch returns, subtracted as the per-return (G_t - b); (B) `greedy_rollout` — a frozen greedy copy (identical initial weights, no gradient updates except promotion) whose per-batch REINFORCE baseline is -L_base, L_base = the frozen copy's mean greedy tour length over the batch's 512 instances (so the loss uses G_t + L_base); at the end of each epoch the frozen weights are promoted (baseline_theta <- cur_theta) iff the current policy's epoch-mean greedy length L_cur over ALL of the epoch's instances satisfies L_cur <= ceil(0.95 * L_base).  UPDATE (spec relaxation, 2026-11-22: the spec's mandatory 'one update per tour' is relaxed to one update per batch — see the instructions file's Deviations section): each batch's 512 tours are sampled under the batch-start weights, and the batch receives ONE REINFORCE update (zero_grad -> loss.backward -> optimizer.step once per batch, no gradient accumulation; reward-to-go G_t = sum_{t' >= t} R_{t'} per tour; loss = the batch sum -sum_b sum_t logprob_{b,t} * (G_{b,t} - baseline), i.e. the same REINFORCE sum as per-tour updates).  100 epochs x 100 batches x batch-size 512 fresh instances, Adam lr = 0.0001, policy_seed = variant_index * 1_000_000 (0 = moving_avg, 1000000 = greedy_rollout), device = cuda.  At the end of EVERY epoch, each variant keeps a checkpoint by best epoch-mean TRAINING return (the mean over the epoch's 100 batch returns; return = -tour length, so higher = better — no evaluation on any split, no generators): the live policy's raw state_dict is re-saved to `<variant>_best_train_return.pt` whenever that epoch mean is a new all-epoch best (epoch 1 always writes; strict improvement only).  The neural rows below are evaluated on the 1000 val + 1000 test instances AGAINST NN, NN+2-opt, and LKH-3 (reference, RUNS = 1, SEED = 1), and come from that ONE checkpoint file per variant ONLY: BOTH of a variant's rows (the greedy row and the 128-sampled row) are computed from that same checkpoint (the end-of-training weights are never used for the reported rows).
Environment: Python 3.11.17, torch 2.11.0+cu128 (CUDA available: True), NumPy 2.4.6, matplotlib 3.11.2.

## Training summary

| variant | policy_seed | final epoch mean ± std (over batch returns) | promotions | promotion epochs | final L_base | training wall (s) |
|---------|-------------|----------------------------------------------|------------|------------------|--------------|-------------------|
| moving_avg | 0 | -603.5 ± 5.0 | n/a | n/a | n/a | 989.8 |
| greedy_rollout | 1000000 | -599.7 ± 4.4 | 9 | [38, 46, 49, 51, 54, 59, 65, 74, 86] | 598.9 | 1871.1 |

## Results (val)

| method | mean | std | q1 | median | q3 | gap vs LKH-3 | wall (s) |
|--------|------|-----|----|--------|----|--------------|---------|
| NN | 322.3 | 61.8 | 276.0 | 320.0 | 362.0 | +148.9 | 0.027 |
| NN+2opt | 293.8 | 53.4 | 255.0 | 292.0 | 330.0 | +120.4 | 0.359 |
| LKH-3 | 173.4 | 27.1 | 154.0 | 172.0 | 191.0 | +0.0 | 16.098 |
| moving_avg (greedy) | 576.0 | 0.0 | 507.0 | 576.5 | 642.0 | +402.6 | 7.921 |
| moving_avg (128-sampled) | 409.5 | 61.3 | 366.0 | 407.0 | 453.2 | +236.1 | 1134.824 |
| greedy_rollout (greedy) | 571.0 | 0.0 | 502.0 | 568.5 | 638.0 | +397.7 | 7.864 |
| greedy_rollout (128-sampled) | 406.6 | 60.9 | 364.8 | 406.0 | 448.0 | +233.2 | 1135.732 |

## Results (test)

| method | mean | std | q1 | median | q3 | gap vs LKH-3 | wall (s) |
|--------|------|-----|----|--------|----|--------------|---------|
| NN | 315.7 | 61.5 | 272.0 | 316.0 | 356.0 | +144.5 | 0.027 |
| NN+2opt | 289.1 | 52.2 | 252.0 | 288.0 | 325.0 | +117.9 | 0.357 |
| LKH-3 | 171.2 | 26.0 | 153.0 | 170.0 | 188.0 | +0.0 | 15.760 |
| moving_avg (greedy) | 573.3 | 0.0 | 503.0 | 573.0 | 637.0 | +402.0 | 7.911 |
| moving_avg (128-sampled) | 409.5 | 62.8 | 365.0 | 409.0 | 451.0 | +238.2 | 1143.214 |
| greedy_rollout (greedy) | 574.2 | 0.0 | 505.8 | 574.0 | 641.0 | +403.0 | 7.981 |
| greedy_rollout (128-sampled) | 408.8 | 60.2 | 367.0 | 406.0 | 449.2 | +237.6 | 1139.518 |

## Runtimes

| quantity | value |
|----------|-------|
| training, moving_avg | 989.8 s |
| training, greedy_rollout | 1871.1 s |
| LKH labeling, val | 16.098 s |
| LKH labeling, test | 15.760 s |
| LKH labeling, total | 31.858 s |
| eval, val | 2286.728 s |
| eval, test | 2299.008 s |
| total | 7478.5 s |
| checkpoints, moving_avg | `src/reinforce_tsp20/checkpoints/moving_avg_best_train_return.pt` (best TRAINING-epoch mean return; both the greedy and the 128-sampled rows are computed from this single checkpoint) |
| checkpoints, greedy_rollout | `src/reinforce_tsp20/checkpoints/greedy_rollout_best_train_return.pt` (best TRAINING-epoch mean return; both the greedy and the 128-sampled rows are computed from this single checkpoint) |

## Configuration

| key | value |
|-----|-------|
| n | 20 |
| epochs | 100 |
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
