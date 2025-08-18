"""
Utilities for inference on a separate device using Hugging Face models.

This module provides helpers to:
- Initialize an inference model from a source model on a target device
- Generate multiple responses per prompt (grouped sampling)
- Greedy generation for evaluation
- Compute simple reward-based validation metrics
"""

from typing import List, Dict, Callable
import copy
import torch


def init_inference_model_from(source_model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Deep-copy a model to the specified device and set it to eval mode."""
    inference_model = copy.deepcopy(source_model).to(device)
    inference_model.eval()
    return inference_model


def generate_grouped_responses(
    inference_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    group_size: int,
    *,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    sampling_temperature: float = 0.0,
    sampling_top_p: float = 1.0,
) -> List[str]:
    """
    Generate `group_size` responses per prompt using HF generation.

    Returns a flattened list of responses with order:
    [p0_r0, p0_r1, ..., p0_r{g-1}, p1_r0, ...].
    """
    inference_model.eval()
    eval_device = next(inference_model.parameters()).device

    with torch.no_grad():
        tokenized = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(eval_device)
        attention_mask = tokenized["attention_mask"].to(eval_device)

        do_sample = sampling_temperature > 0.0
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        sequences = inference_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max(0, min_new_tokens),
            do_sample=do_sample,
            temperature=sampling_temperature if do_sample else None,
            top_p=sampling_top_p if do_sample else None,
            num_return_sequences=group_size,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_lengths = attention_mask.sum(dim=1).repeat_interleave(group_size)
        sequences = sequences.detach().cpu()

        responses: List[str] = []
        for i in range(sequences.shape[0]):
            prompt_len = int(prompt_lengths[i].item())
            generated_tokens = sequences[i][prompt_len:]
            response_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            responses.append(response_text)

    return responses


def greedy_generate_responses(
    inference_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
) -> List[str]:
    """Generate one greedy response per prompt using HF generation."""
    inference_model.eval()
    eval_device = next(inference_model.parameters()).device

    with torch.no_grad():
        tokenized = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(eval_device)
        attention_mask = tokenized["attention_mask"].to(eval_device)

        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        sequences = inference_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_lengths = attention_mask.sum(dim=1)
        sequences = sequences.detach().cpu()
        prompt_lengths = prompt_lengths.detach().cpu()

        responses: List[str] = []
        for i in range(sequences.shape[0]):
            prompt_len = int(prompt_lengths[i].item())
            generated_tokens = sequences[i][prompt_len:]
            response_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            responses.append(response_text)

    return responses


def compute_reward_metrics(
    responses: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
) -> Dict[str, float]:
    """Compute simple accuracy-style metrics from a reward function."""
    format_correct = 0
    answer_correct = 0
    overall_correct = 0

    for response, gt in zip(responses, ground_truths):
        reward = reward_fn(response, gt, fast=True)  # assumes signature compatibility
        format_correct += 1 if float(reward.get("format_reward", 0.0)) > 0.5 else 0
        answer_correct += 1 if float(reward.get("answer_reward", 0.0)) > 0.5 else 0
        overall_correct += 1 if float(reward.get("reward", 0.0)) > 0.5 else 0

    total = len(responses)
    return {
        "val_accuracy": overall_correct / total if total else 0.0,
        "val_format_accuracy": format_correct / total if total else 0.0,
        "val_answer_accuracy": answer_correct / total if total else 0.0,
        "val_total": total,
    }


