# GRPO Generation & Forward Pass Mechanics

## The Sequence

1. **Take 1 prompt**: "Janet has 16 eggs..."
2. **Generate 8 different answers** to that same question
3. **Reward functions score all 8 answers**
4. **GRPO compares** which answers got higher rewards
5. **Repeat with gradient accumulation** over 8 such prompt-sets

## Parameters
- `per_device_train_batch_size=1` → 1 prompt at a time
- `num_generations=8` → 8 completions per prompt
- `gradient_accumulation_steps=8` → accumulate over 8 prompts before weight update

## Memory Logic
- "Batch size of 1" = 1 prompt processed per forward pass
- That 1 prompt spawns 8 generations for reward comparison
- Different from regular training where 1 sample = 1 input/output pair



### Sequencing of the algorithm (current workflow: frozen reference on GPU1)
- **Sync reference**: Copy weights `model.state_dict() → inference_model` (GPU1) at each GRPO step via `copy_model_weights()`. The `inference_model` is eval, frozen, and (optionally) compiled from initialization.
- **Rollouts (GPU1)**: Sample `group_size` responses per prompt via HF `generate` (optional stop strings). If using GRPO-Clip, set `output_scores=True`.
- **Rewards/advantages**: Score with `r1_zero_reward_fn`; compute group-normalized advantages per prompt.
- **Old log-probs (GPU1 → GPU0)**: Convert generation scores to per-token log-probs and align with `assemble_old_log_probs`; move tensor to GPU0 and reuse within the step.
- **Train (GPU0)**: Recompute current `log_probs`, compute GRPO/GRPO-Clip loss, accumulate microbatches, clip grads, then `optimizer.step()` per epoch.
- **Validation (optional)**: Resync `inference_model` and run greedy eval.

### Current vs previous workflows
- **Current: two-GPU frozen reference (HF)**
  - **Reference source**: Frozen `inference_model` on GPU1 (HF `generate` with `output_scores=True`).
  - **Old log-probs**: Use generation scores + `assemble_old_log_probs` to align; send tensor to GPU0.
  - **Parallelism**: GPU1 handles rollout + `old_log_probs`; GPU0 trains.
  - **Memory**: Two models (one per GPU), small tensor transfers.

- **Previous: vLLM hot-swapping (two-GPU)**
  - **Reference source**: vLLM engine on GPU1; weights/configs were hot-swapped to stay near on-policy.
  - **Trade-offs**: Higher operational complexity, less direct control over `generate` settings vs HF; required frequent syncing.
  - **Reason for change**: Simplify codepath, remove hot-swapping, unify training/inference via HF while keeping two-device parallelism.

### When the models start to differ within a step
- **Identical** after sync, during rollouts, and during `old_log_probs` compute.
- **Still identical** during microbatch backward passes (we only accumulate grads).
- **First divergence** happens at the end of the first epoch, when `optimizer.step()` updates `model`.
- `inference_model` remains frozen until the next GRPO step.

### Impact of epochs_per_rollout_batch
- **= 1**: r = exp(logp_cur − logp_old) ≈ 1 for the epoch (clipping won’t engage). You still update once; the model diverges only after the epoch and will be used next step.
- **> 1**: After the first optimizer.step, `logp_cur` differs; r moves away from 1 in later epochs; clipping can engage and stabilize updates.

### On-policy vs approximate on-policy
- **On-policy**: Rollouts and `old_log_probs` come from the same snapshot used to act (the synced `inference_model`).
- **Approximate on-policy**: Reusing the same data for multiple epochs is standard PPO/GRPO; importance ratio r and clipping control drift.

### Why learning still happens when r ≈ 1
- The gradient comes from advantages: with r ≈ 1, GRPO-Clip reduces to REINFORCE-with-baseline. Group-normalized advantages push up high-reward responses and down low-reward ones, so you still get improvement.
