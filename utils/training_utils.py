"""
Training utilities for SFT and RL training
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from utils.vllm_utils import evaluate_vllm
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy from unnormalized logits in a numerically
    stable way using logsumexp.

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length) with the entropy
        for each next-token prediction.
    """
    normalization_term = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - normalization_term
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant, respecting a boolean mask.

    Args:
        tensor: Tensor to sum and normalize.
        mask: Same shape as tensor; positions with 1/True are included in the sum.
        normalize_constant: Constant to divide the sum by.
        dim: Dimension to sum along; if None, sum over all dimensions.

    Returns:
        Normalized sum with masked-out elements excluded from the sum.
    """
    masked = tensor * mask.to(dtype=tensor.dtype)
    if dim is None:
        masked_sum = masked.sum()
    else:
        masked_sum = masked.sum(dim=dim)
    return masked_sum / normalize_constant


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Tokenize prompts and outputs separately without padding first
    tokenized_prompts = tokenizer(
        prompt_strs, add_special_tokens=False, padding=False, return_tensors=None
    )
    tokenized_outputs = tokenizer(
        output_strs, add_special_tokens=False, padding=False, return_tensors=None
    )

    # Compute lengths for mask creation
    prompt_token_counts = [len(ids) for ids in tokenized_prompts["input_ids"]]
    output_token_counts = [len(ids) for ids in tokenized_outputs["input_ids"]]

    batch_size = len(prompt_token_counts)
    max_sequence_length = max(
        (pl + ol) for pl, ol in zip(prompt_token_counts, output_token_counts)
    ) if batch_size > 0 else 0

    # Preallocate padded tensor and fill row-wise to avoid many small tensor pads
    padded_input_ids = torch.full(
        (batch_size, max_sequence_length), pad_id, dtype=torch.long)
    for row_index, (prompt_id_list, output_id_list) in enumerate(
        zip(tokenized_prompts["input_ids"], tokenized_outputs["input_ids"])
    ):
        concatenated_id_list = prompt_id_list + output_id_list
        if concatenated_id_list:
            concatenated_id_tensor = torch.tensor(
                concatenated_id_list, dtype=torch.long)
            padded_input_ids[row_index, : concatenated_id_tensor.numel(
            )] = concatenated_id_tensor

    # Create response mask using vectorized tensor operations
    position_indices = torch.arange(
        max_sequence_length).unsqueeze(0).expand(batch_size, -1)
    prompt_end_positions = torch.tensor(
        prompt_token_counts, dtype=torch.long).unsqueeze(1)
    output_token_counts_tensor = torch.tensor(
        output_token_counts, dtype=torch.long).unsqueeze(1)
    response_end_positions = prompt_end_positions + output_token_counts_tensor
    response_mask_unshifted = (position_indices >= prompt_end_positions) & (
        position_indices < response_end_positions)

    return {
        "input_ids": padded_input_ids[:, :-1],
        "labels": padded_input_ids[:, 1:],
        "response_mask": response_mask_unshifted[:, 1:],
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    *,
    requires_grad: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities and optional token entropy.

    Args:
        model: PreTrainedModel used for scoring.
        input_ids: Tensor of shape (batch_size, sequence_length).
        labels: Tensor of shape (batch_size, sequence_length).
        return_token_entropy: If True, include per-token entropy.

    Returns:
        dict with keys:
            - "log_probs": Tensor (batch_size, sequence_length)
            - "token_entropy": optional Tensor (batch_size, sequence_length)
    """
    # Enable/disable autograd depending on caller context
    with torch.set_grad_enabled(requires_grad):
        logits = model(
            input_ids, attention_mask=attention_mask).logits  # (B, S, V)

    # Compute log-probs over vocabulary and gather per-position label log-prob
    log_probs_vocab = F.log_softmax(logits, dim=-1)  # (B, S, V)
    per_token_log_probs = torch.gather(
        log_probs_vocab, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, S)

    result: dict[str, torch.Tensor] = {"log_probs": per_token_log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result


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
    masked_sum = masked_normalize(
        nll, response_mask, normalize_constant=normalize_constant, dim=None)

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


def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    *,
    # vLLM (required)
    vllm_model: object,
    vllm_sampling_params: object,
    # logging options
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, object]:
    """Generate responses for prompts and compute logging metrics.

    Logs per-example fields and aggregate statistics:
      - prompt, response, ground_truth
      - reward dict (including format_reward, answer_reward, reward)
      - mean token entropy over response tokens
      - response length in tokens
      - aggregates: avg token entropy, avg length, avg length for correct/incorrect

    Args:
        model: Language model to generate with (already on correct device).
        tokenizer: Tokenizer paired with the model.
        prompts: List of prompt strings.
        ground_truths: List of corresponding ground-truth answers.
        reward_fn: Callable that returns a dict with keys like
            {"reward", "format_reward", "answer_reward"} given (response, ground_truth).
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: nucleus sampling parameter.
        is_correct_fn: Optional function mapping reward dict -> bool for correctness.

    Returns:
        Dict with keys "examples" (list of per-example dicts) and "aggregates" (metrics).
    """
    model.eval()
    device = next(model.parameters()).device

    # Generate with vLLM (via evaluate_vllm) or HF
    responses: List[str]
    vllm_eval_payload: Optional[Dict[str, object]] = None
    # Use the standardized vLLM evaluator to generate and persist results
    vllm_eval_payload = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=vllm_sampling_params,
        output_dir=output_dir or "evaluation_results",
        model_name=model_name or "unknown",
        problem_metadata=None,
        save_full_responses=True,
        write_results=False,
    )
    # Extract generated texts back for entropy/length computation
    results_list = vllm_eval_payload.get(
        "results", [])  # type: ignore[assignment]
    responses = [
        (
            (res.get("full_response") if isinstance(res, dict) else None)
            or ""
        )
        for res in results_list  # type: ignore[union-attr]
    ]

    # Compute token entropy over response tokens using utilities
    tokenization = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer,
    )
    concat_input_ids = tokenization["input_ids"].to(device)
    concat_labels = tokenization["labels"].to(device)
    response_mask = tokenization["response_mask"].to(device)

    with torch.no_grad():
        scoring = get_response_log_probs(
            model=model,
            input_ids=concat_input_ids,
            labels=concat_labels,
            return_token_entropy=True,
        )
    token_entropy = scoring.get("token_entropy")  # (B, S)
    if token_entropy is None:
        # Fallback compute from logits if needed (should not happen)
        logits = model(concat_input_ids).logits
        token_entropy = compute_entropy(logits)

    # Per-example mean entropy over response tokens
    response_mask_f = response_mask.to(dtype=token_entropy.dtype)
    mask_counts = response_mask_f.sum(dim=1).clamp_min(1)
    response_entropy_mean = (
        (token_entropy * response_mask_f).sum(dim=1) / mask_counts).detach().cpu().tolist()

    # Response lengths in tokens
    response_lengths = response_mask.sum(dim=1).detach().cpu().tolist()

    # Rewards and correctness (reused from vLLM payload)
    example_dicts: List[Dict[str, object]] = []
    for idx, (prompt, response, gt, resp_len, resp_ent) in enumerate(zip(
        prompts, responses, ground_truths, response_lengths, response_entropy_mean
    )):
        # Reuse rewards/correctness from vLLM evaluation to avoid recomputation
        vres = vllm_eval_payload["results"][idx]  # type: ignore[index]
        rewards = vres.get("rewards", {}) if isinstance(vres, dict) else {}
        is_correct = bool(vres.get("correct", False)
                          ) if isinstance(vres, dict) else False

        example_dicts.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": gt,
                "reward": rewards.get("reward"),
                "format_reward": rewards.get("format_reward"),
                "answer_reward": rewards.get("answer_reward"),
                "response_token_entropy_mean": resp_ent,
                "response_length_tokens": int(resp_len),
                "is_correct": is_correct,
            }
        )

    # Aggregates (single-pass, no numpy)
    total_len = 0.0
    total_ent = 0.0
    count = len(response_lengths)
    for l, e in zip(response_lengths, response_entropy_mean):
        total_len += float(l)
        total_ent += float(e)
    denom = max(1, count)
    aggregates = {
        "avg_response_length": total_len / denom,
        "avg_response_token_entropy": total_ent / denom,
    }

    # If vLLM evaluation was used, surface its accuracy metrics too
    if vllm_eval_payload is not None:
        try:
            # type: ignore[index]
            overall = vllm_eval_payload["metadata"]["overall_metrics"]
            aggregates.update({
                "answer_accuracy": float(overall.get("answer_accuracy", 0.0)),
                "format_accuracy": float(overall.get("format_accuracy", 0.0)),
                "overall_accuracy": float(overall.get("overall_accuracy", 0.0)),
            })
        except Exception:
            pass

    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "generation_backend": "vllm",
            "model_name": model_name or "unknown",
            "num_examples": len(prompts),
            "sampling": {
                "max_new_tokens": getattr(vllm_sampling_params, "max_tokens", None),
                "temperature": getattr(vllm_sampling_params, "temperature", None),
                "top_p": getattr(vllm_sampling_params, "top_p", None),
            },
        },
        "examples": example_dicts,
        "aggregates": aggregates,
    }

    # Persist merged SFT eval result once here (canonical SFT path)
    out_dir = Path(output_dir or "logs/generations")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (model_name or "sft").replace("/", "_")
    out_path = out_dir / f"{prefix}_generations_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
