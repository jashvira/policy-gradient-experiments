from __future__ import annotations

import torch
from utils.training_utils import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SFT loss for a microbatch, scale for grad accumulation, and backprop.

    Args:
        policy_log_probs: (batch_size, sequence_length) per-token log-probabilities.
        response_mask: (batch_size, sequence_length) mask for response tokens.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant: constant to divide masked sum by (default 1.0).

    Returns:
        (loss, metadata) where loss is a scalar tensor used for logging and gradients.
    """
    # Negative log-likelihood per token
    nll = -policy_log_probs

    # Sum over masked tokens across the whole microbatch using provided normalization
    masked_sum = masked_normalize(nll, response_mask, normalize_constant=normalize_constant, dim=None)

    # Normalize by batch size and grad accumulation steps (normalize_constant already applied)
    batch_size = policy_log_probs.shape[0]
    denom = float(batch_size) * float(gradient_accumulation_steps)
    loss = masked_sum / denom

    # Backpropagate
    loss.backward()

    metadata = {
        "nll_sum": masked_sum.detach(),
        "num_masked": response_mask.sum().detach(),
        "batch_size": torch.tensor(batch_size),
        "denom": torch.tensor(denom),
    }
    return loss.detach(), metadata


