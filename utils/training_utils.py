"""
Training utilities for SFT and RL training
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase
# Intentionally avoid importing vLLM here to keep training logging HF-only
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json


def compute_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute per-token entropy from unnormalized logits in a numerically
    stable way using logsumexp.

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.
        eps: Small epsilon for numerical stability.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length) with the entropy
        for each next-token prediction.
    """
    normalization_term = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - normalization_term
    probs = log_probs.exp()
    # Add epsilon to prevent log(0) in case of numerical precision issues
    stable_log_probs = torch.clamp(log_probs, min=torch.log(torch.tensor(eps, device=logits.device, dtype=logits.dtype)))
    return -(probs * stable_log_probs).sum(dim=-1)


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

    # Memory-light per-token log-prob: log_softmax = logits - logsumexp(logits)
    # Avoids materializing the (B, S, V) log-prob tensor
    normalization_term = torch.logsumexp(logits, dim=-1, keepdim=True)  # (B, S, 1)
    selected_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, S)
    per_token_log_probs = selected_logits - normalization_term.squeeze(-1)  # (B, S)

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


def compute_log_probs_for_responses(
    model: torch.nn.Module,
    prompts: list[str],
    responses: list[str],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    torch_dtype: torch.dtype,
    requires_grad: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper to compute log probs for prompt-response pairs.

    Args:
        model: Model to compute log probs with
        prompts: List of prompt strings
        responses: List of response strings (same length as prompts)
        tokenizer: Tokenizer to use
        device: Device to compute on
        torch_dtype: Data type for autocast
        requires_grad: Whether to enable gradients

    Returns:
        (log_probs, response_mask) both on the target device
    """
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)

    with torch.autocast(device_type=device.type, dtype=torch_dtype):
        outputs = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
            requires_grad=requires_grad,
        )

    log_probs = outputs["log_probs"]
    return log_probs, response_mask


def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    write_results: bool = False,
) -> Dict[str, object]:
    """HF-only generation and logging helper for training batches.

    Generates with the HF model, computes response entropy/length, evaluates rewards,
    and returns a dict with examples and aggregates. Optionally writes a JSON artifact.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    # HF generation
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=bool(temperature and temperature > 0),
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    # Determine prompt lengths
    if attn_mask is not None:
        prompt_lens = attn_mask.sum(dim=1).tolist()
    else:
        prompt_lens = (input_ids != pad_id).sum(dim=1).tolist()
    responses: List[str] = []
    for i in range(len(prompts)):
        gen_part = gen_ids[i, prompt_lens[i]:]
        responses.append(tokenizer.decode(gen_part, skip_special_tokens=True))

    # Token entropy and response length
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
    token_entropy = scoring.get("token_entropy")
    if token_entropy is None:
        logits = model(concat_input_ids).logits
        token_entropy = compute_entropy(logits)
    response_mask_f = response_mask.to(dtype=token_entropy.dtype)
    mask_counts = response_mask_f.sum(dim=1).clamp_min(1)
    response_entropy_mean = (
        (token_entropy * response_mask_f).sum(dim=1) / mask_counts
    ).detach().cpu().tolist()
    response_lengths = response_mask.sum(dim=1).detach().cpu().tolist()

    # Rewards and aggregates
    example_dicts: List[Dict[str, object]] = []
    n_format_correct = 0
    n_answer_correct = 0
    n_correct = 0
    for prompt, response, gt, resp_len, resp_ent in zip(
        prompts, responses, ground_truths, response_lengths, response_entropy_mean
    ):
        rewards = reward_fn(response, gt)
        is_correct = bool(rewards.get("reward", 0.0))
        n_format_correct += int(bool(rewards.get("format_reward", 0.0)))
        n_answer_correct += int(bool(rewards.get("answer_reward", 0.0)))
        n_correct += int(is_correct)
        example_dicts.append({
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            "reward": rewards.get("reward"),
            "format_reward": rewards.get("format_reward"),
            "answer_reward": rewards.get("answer_reward"),
            "response_token_entropy_mean": resp_ent,
            "response_length_tokens": int(resp_len),
            "is_correct": is_correct,
        })

    denom = max(1, len(response_lengths))
    aggregates = {
        "avg_response_length": float(sum(response_lengths)) / denom,
        "avg_response_token_entropy": float(sum(response_entropy_mean)) / denom,
        "answer_accuracy": n_answer_correct / denom,
        "format_accuracy": n_format_correct / denom,
        "overall_accuracy": n_correct / denom,
    }

    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "generation_backend": "hf",
            "model_name": model_name or "unknown",
            "num_examples": len(prompts),
            "sampling": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        },
        "examples": example_dicts,
        "aggregates": aggregates,
    }

    if write_results:
        out_dir = Path(output_dir or "logs/generations")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = (model_name or "sft").replace("/", "_")
        out_path = out_dir / f"{prefix}_generations_{stamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    if was_training:
        model.train()

    return result
