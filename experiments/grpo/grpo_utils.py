# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for Group Relative Policy Optimisation (GRPO)."""

from typing import Callable

import torch
from utils.training_utils import masked_normalize


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of responses per question (group).
        advantage_eps: float, small constant to avoid division by zero in normalisation.
        normalize_by_std: bool, if True, divide by the per-group standard deviation;
            otherwise subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            advantages shape (rollout_batch_size,): Group-normalised rewards
                for each rollout response.
            raw_rewards shape (rollout_batch_size,): Unnormalised rewards
                for each rollout response.
            metadata: dict containing statistics to log (e.g. mean, std, max/min of rewards).
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            f"Length mismatch: {len(rollout_responses)} responses vs "
            f"{len(repeated_ground_truths)} ground truths"
        )

    rollout_batch_size = len(rollout_responses)
    if rollout_batch_size % group_size != 0:
        raise ValueError(
            f"rollout_batch_size ({rollout_batch_size}) must be divisible by "
            f"group_size ({group_size})"
        )

    # Compute raw rewards for all responses
    raw_rewards_list = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards_list.append(reward_dict["reward"])

    # Convert to tensor with specified dtype
    raw_rewards = torch.tensor(raw_rewards_list, dtype=dtype)

    # Reshape to groups: (n_groups, group_size)
    n_groups = rollout_batch_size // group_size
    grouped_rewards = raw_rewards.view(n_groups, group_size)

    # Compute per-group statistics
    group_means = grouped_rewards.mean(dim=1, keepdim=True)  # (n_groups, 1)

    if normalize_by_std:
        group_stds = grouped_rewards.std(
            dim=1, keepdim=True, unbiased=True)  # (n_groups, 1)
        # Normalize: (reward - mean) / (std + eps)
        normalized_rewards = (grouped_rewards - group_means) / \
            (group_stds + advantage_eps)
    else:
        # Normalize: reward - mean
        normalized_rewards = grouped_rewards - group_means

    # Flatten back to original shape
    advantages = normalized_rewards.view(-1)

    # Compute minimal metadata
    metadata = {
        "raw_rewards_mean": float(raw_rewards.mean()),
        "raw_rewards_std": float(raw_rewards.std()),
        "advantages_mean": float(advantages.mean()),
        "advantages_std": float(advantages.std()),
    }

    if normalize_by_std:
        group_stds_flat = group_stds.view(-1)
        metadata.update({
            "group_stds_mean": float(group_stds_flat.mean()),
            "group_stds_std": float(group_stds_flat.std()),
        })

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages
    is either the raw reward or an already-normalised advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor shape (batch_size, 1), scalar
            reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor shape (batch_size, sequence_length), logprobs for
            each token.

    Returns:
        torch.Tensor shape (batch_size, sequence_length), the per-token policy-gradient
        loss (to be aggregated across the batch and sequence dimensions in the training loop).
    """
    # Validate input shapes
    batch_size = policy_log_probs.shape[0]
    sequence_length = policy_log_probs.shape[1]

    if raw_rewards_or_advantages.shape != (batch_size, 1):
        raise ValueError(
            f"Expected raw_rewards_or_advantages shape ({batch_size}, 1), "
            f"got {raw_rewards_or_advantages.shape}"
        )

    # Broadcast rewards/advantages over sequence length
    # Shape: (batch_size, 1) -> (batch_size, sequence_length)
    broadcasted_rewards = raw_rewards_or_advantages.expand(
        batch_size, sequence_length)

    # Compute per-token policy gradient loss
    # Policy gradient: -reward * log_prob
    per_token_loss = -broadcasted_rewards * policy_log_probs

    return per_token_loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO-Clip loss per token.

    Implements the loss from equation (33):
    -min(π_θ(a_t|q, o<t) / π_θ_old(a_t|q, o<t) * A_t,
         clip(π_θ(a_t|q, o<t) / π_θ_old(a_t|q, o<t), 1-ε, 1+ε) * A_t)

    Args:
        advantages: torch.Tensor shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor shape (batch_size, sequence_length),
            per-token log probs from the policy being trained.
        old_log_probs: torch.Tensor shape (batch_size, sequence_length),
            per-token log probs from the old policy.
        cliprange: float, clip parameter ε (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: torch.Tensor of shape (batch_size, sequence_length),
                the per-token clipped loss.
            metadata: dict containing clipping statistics for logging.
    """
    # Validate input shapes
    batch_size, sequence_length = policy_log_probs.shape

    if advantages.shape != (batch_size, 1):
        raise ValueError(
            f"Expected advantages shape ({batch_size}, 1), got {advantages.shape}"
        )

    if old_log_probs.shape != (batch_size, sequence_length):
        raise ValueError(
            f"Expected old_log_probs shape ({batch_size}, {sequence_length}), "
            f"got {old_log_probs.shape}"
        )

    # Broadcast advantages over sequence length
    # Shape: (batch_size, 1) -> (batch_size, sequence_length)
    # Compute core ratio/clipping math in float32 for numerical stability.
    model_dtype = policy_log_probs.dtype
    broadcasted_advantages_f32 = advantages.to(torch.float32).expand(batch_size, sequence_length)
    policy_log_probs_f32 = policy_log_probs.to(torch.float32)
    old_log_probs_f32 = old_log_probs.to(torch.float32)

    # Compute importance sampling ratio: π_θ / π_θ_old
    # log(π_θ / π_θ_old) = log(π_θ) - log(π_θ_old)
    log_ratio = policy_log_probs_f32 - old_log_probs_f32
    ratio = torch.exp(log_ratio)

    # Compute unclipped policy gradient term: ratio * A_t
    unclipped_loss = ratio * broadcasted_advantages_f32

    # Compute clipped policy gradient term: clip(ratio, 1-ε, 1+ε) * A_t
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_loss = clipped_ratio * broadcasted_advantages_f32

    # Take minimum (which gives maximum loss since we negate at the end)
    min_loss = torch.min(unclipped_loss, clipped_loss)

    # Apply negative sign for loss (we want to minimize negative reward)
    per_token_loss_f32 = -min_loss
    # Cast back to the model's dtype for downstream reductions/backprop
    per_token_loss = per_token_loss_f32.to(model_dtype)

    # Compute metadata for logging (float32 mean is fine)
    # Track which tokens were clipped
    was_clipped = (clipped_loss < unclipped_loss).float()

    # Keep metadata minimal and scalar-only to avoid logging large tensors downstream
    metadata = {
        "ratio_mean": ratio.mean(),
        "clip_fraction": was_clipped.mean(),
    }

    return per_token_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs: torch.Tensor shape (batch_size, sequence_length),
            per-token log-probabilities from the policy being trained.
        loss_type: str, one of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards: torch.Tensor | None, required if loss_type == "no_baseline";
            shape (batch_size, 1).
        advantages: torch.Tensor | None, required for "reinforce_with_baseline" and "grpo_clip";
            shape (batch_size, 1).
        old_log_probs: torch.Tensor | None, required for "grpo_clip";
            shape (batch_size, sequence_length).
        cliprange: float | None, required for "grpo_clip"; scalar ε used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: torch.Tensor shape (batch_size, sequence_length), per-token loss.
            metadata: dict, statistics from the underlying routine
                (e.g., clip fraction for GRPO-Clip).
    """
    valid_loss_types = ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    if loss_type not in valid_loss_types:
        raise ValueError(
            f"Invalid loss_type '{loss_type}'. Must be one of {valid_loss_types}")

    if loss_type == "no_baseline":
        # Use raw rewards as advantages (no baseline subtraction)
        if raw_rewards is None:
            raise ValueError(
                "raw_rewards is required for loss_type='no_baseline'")

        per_token_loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )

        # No additional metadata (avoid duplicates with step-level reward stats)
        metadata = {}

    elif loss_type == "reinforce_with_baseline":
        # Use pre-computed advantages (group-normalised rewards)
        if advantages is None:
            raise ValueError(
                "advantages is required for loss_type='reinforce_with_baseline'")

        per_token_loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )

        # No additional metadata (avoid duplicates with step-level advantage stats)
        metadata = {}

    elif loss_type == "grpo_clip":
        # Use GRPO-Clip loss with advantages and old policy log probs
        if advantages is None:
            raise ValueError(
                "advantages is required for loss_type='grpo_clip'")
        if old_log_probs is None:
            raise ValueError(
                "old_log_probs is required for loss_type='grpo_clip'")
        if cliprange is None:
            raise ValueError("cliprange is required for loss_type='grpo_clip'")

        per_token_loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )

        # metadata is already provided by compute_grpo_clip_loss

    return per_token_loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.

    Args:
        tensor: torch.Tensor, the data to be averaged.
        mask: torch.Tensor, same shape as tensor; positions with 1 are included in the mean.
        dim: int | None, dimension over which to average. If None, compute the mean over all masked elements.

    Returns:
        torch.Tensor, the masked mean; shape matches tensor.mean(dim) semantics.
    """
    if tensor.shape != mask.shape:
        raise ValueError(
            f"tensor and mask must have the same shape. Got tensor: {tensor.shape}, mask: {mask.shape}")

    # Convert mask to same dtype as tensor for computation
    mask_float = mask.to(dtype=tensor.dtype)

    # Apply mask: set unmasked elements to 0
    masked_tensor = tensor * mask_float

    if dim is None:
        # Compute mean over all masked elements
        num_masked_elements = mask_float.sum()
        if num_masked_elements == 0:
            # Avoid division by zero - return 0 when no elements are masked
            return torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
        return masked_tensor.sum() / num_masked_elements
    else:
        # Compute mean along specified dimension
        # Count number of masked elements along the dimension
        num_masked_elements = mask_float.sum(dim=dim)

        # Handle case where some slices have no masked elements
        # Use torch.where to avoid division by zero
        masked_sum = masked_tensor.sum(dim=dim)
        result = torch.where(
            num_masked_elements > 0,
            masked_sum / num_masked_elements,
            torch.full_like(masked_sum, float('nan'))
        )

        return result


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    seq_loss_reduction: str = "per_example_mean",
    accumulation_denominator: int | None = None,
    *,
    grad_scaler: "torch.amp.GradScaler" | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch for GRPO training.

    Args:
        policy_log_probs: torch.Tensor shape (batch_size, sequence_length),
            per-token log-probabilities from the policy being trained.
        response_mask: torch.Tensor shape (batch_size, sequence_length),
            1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: int, number of microbatches per optimizer step.
        loss_type: str, one of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards: torch.Tensor | None, needed when loss_type == "no_baseline";
            shape (batch_size, 1).
        advantages: torch.Tensor | None, needed when loss_type != "no_baseline";
            shape (batch_size, 1).
        old_log_probs: torch.Tensor | None, required for GRPO-Clip;
            shape (batch_size, sequence_length).
        cliprange: float | None, clip parameter ε for GRPO-Clip.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: scalar tensor, the microbatch loss, adjusted for gradient accumulation.
            metadata: dict with metadata from the underlying loss call and other statistics.
    """
    batch_size, sequence_length = policy_log_probs.shape

    # Validate inputs match expected shapes
    if response_mask.shape != (batch_size, sequence_length):
        raise ValueError(
            f"response_mask shape {response_mask.shape} doesn't match "
            f"policy_log_probs shape {policy_log_probs.shape}"
        )

    # Step 1: Compute per-token policy gradient loss
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # Step 2: Reduce token losses over sequence dimension according to config
    seq_loss_reduction = str(seq_loss_reduction).lower()
    if seq_loss_reduction not in {"per_example_mean", "masked_normalize"}:
        raise ValueError(
            f"Invalid seq_loss_reduction '{seq_loss_reduction}'. "
            "Expected one of {'per_example_mean', 'masked_normalize'}"
        )

    if seq_loss_reduction == "per_example_mean":
        # Average per example over response tokens, then mean over batch
        per_example_loss = masked_mean(
            per_token_loss, response_mask.to(per_token_loss.dtype), dim=1)
        # Guard: if a row has zero masked tokens, masked_mean returns NaN; treat as 0 loss for that example
        per_example_loss = torch.nan_to_num(per_example_loss, nan=0.0)
        batch_loss = per_example_loss.mean()
    else:
        # Sum over all masked tokens across the microbatch, then divide by batch size
        masked_sum = masked_normalize(
            per_token_loss, response_mask, normalize_constant=1.0, dim=None
        )
        batch_loss = masked_sum / float(batch_size)

    # Step 4: Adjust for gradient accumulation steps
    # Scale by 1/denominator so that when we accumulate gradients across
    # multiple microbatches, the final gradient magnitude is correct.
    # Use provided accumulation_denominator for partial groups; otherwise fall back.
    denom = float(accumulation_denominator) if accumulation_denominator is not None else float(
        gradient_accumulation_steps)
    scaled_loss = batch_loss / denom

    # Step 5: Backpropagate gradients (support GradScaler for fp16)
    if grad_scaler is not None:
        grad_scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    # Step 6: Prepare minimal metadata for logging (loss-specific only)
    # Only expose minimal metadata needed by the caller for aggregation
    # Keep any loss-specific fields (e.g., clip_fraction) and the unscaled batch loss
    metadata = {
        **loss_metadata,
        "batch_loss": batch_loss.detach(),
    }

    # Return the scaled loss for logging (this is what was actually used for backprop)
    return scaled_loss.detach(), metadata
