"""
Utilities for inference on a separate device using Hugging Face models.

This module provides helpers to:
- Initialize an inference model from a source model on a target device
- Generate multiple responses per prompt (grouped sampling)
- Greedy generation for evaluation
- Compute simple reward-based validation metrics
"""

from typing import List, Dict, Callable, Optional
import copy
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F


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
    stop_strings: Optional[List[str]] = None,
    return_scores: bool = False,
):
    """
    Generate `group_size` responses per prompt using HF generation.

    Returns:
      - If return_scores=False:
          List[str] of responses flattened as [p0_r0, p0_r1, ..., p0_r{g-1}, p1_r0, ...]
      - If return_scores=True:
          (responses: List[str], logprobs_matrix: torch.Tensor (B, T), gen_lens: torch.Tensor (B,))
          where B = batch size (n_prompts * group_size), T = generated steps, gen_lens includes EOS.
    """
    inference_model.eval()
    eval_device = next(inference_model.parameters()).device

    # Use inference_mode to disable autograd and some dispatcher overhead
    with torch.inference_mode():
        tokenized = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(eval_device)
        attention_mask = tokenized["attention_mask"].to(eval_device)

        do_sample = sampling_temperature > 0.0
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        stopping_criteria = _build_stopping_criteria(tokenizer, stop_strings)

        # Common generation parameters
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max(0, min_new_tokens),
            "do_sample": do_sample,
            "temperature": sampling_temperature if do_sample else None,
            "top_p": sampling_top_p if do_sample else None,
            "num_return_sequences": group_size,
            "pad_token_id": pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }

        if return_scores:
            gen_kwargs.update({
                "return_dict_in_generate": True,
                "output_scores": True,
            })
            gen_out = inference_model.generate(**gen_kwargs)
            sequences = gen_out.sequences  # (B, S_total)
            scores = gen_out.scores        # list[T] of (B, V)
        else:
            sequences = inference_model.generate(**gen_kwargs)
            scores = None

        # Common prompt length handling
        prompt_lengths = attention_mask.sum(dim=1)
        if group_size > 1:
            prompt_lengths = prompt_lengths.repeat_interleave(group_size)

        # Process scores if needed
        logprobs_matrix = None
        gen_lens = None
        if return_scores and scores is not None:
            # Build positions to gather chosen ids: (B, T)
            B = sequences.shape[0]
            T = len(scores)
            t_indices = torch.arange(T, device=sequences.device).unsqueeze(0).expand(B, -1)
            positions = prompt_lengths.unsqueeze(1) + t_indices
            positions = positions.clamp(max=sequences.shape[1] - 1)
            chosen_ids = sequences.gather(1, positions)  # (B, T)

            # Memory-efficient per-step log-prob computation without stacking (B, T, V)
            logprobs_matrix = torch.empty((B, T), dtype=torch.float32, device=sequences.device)
            for t, step_scores in enumerate(scores):
                # step_scores: (B, V) unnormalized logits for step t
                denom = torch.logsumexp(step_scores, dim=-1)  # (B,)
                chosen_t = chosen_ids[:, t]
                selected = step_scores.gather(1, chosen_t.unsqueeze(1)).squeeze(1)  # (B,)
                logprobs_matrix[:, t] = selected - denom

            # Compute gen lengths up to and including EOS (or full T if none)
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                gen_lens = torch.full((B,), T, dtype=torch.long, device=sequences.device)
            else:
                eos_mask = (chosen_ids == eos_id)
                has_eos = eos_mask.any(dim=1)
                first_idx = eos_mask.float().argmax(dim=1)
                gen_lens = torch.where(has_eos, first_idx + 1, torch.full_like(first_idx, T))

        # Common response decoding
        responses: List[str] = []
        sequences_cpu = sequences.detach().cpu()
        prompt_lengths_cpu = prompt_lengths.detach().cpu()

        for i in range(sequences.shape[0]):
            prompt_len = int(prompt_lengths_cpu[i].item())
            if return_scores and gen_lens is not None:
                # Use computed generation length for score path
                gen_len = int(gen_lens[i].item())
                generated_tokens = sequences_cpu[i][prompt_len:prompt_len + gen_len]
            else:
                # Use all generated tokens for non-score path
                generated_tokens = sequences_cpu[i][prompt_len:]

            response_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            responses.append(response_text)

    if return_scores:
        return responses, logprobs_matrix.detach().cpu(), gen_lens.detach().cpu()
    else:
        return responses


def greedy_generate_responses(
    inference_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    stop_strings: Optional[List[str]] = None,
) -> List[str]:
    """Generate one greedy response per prompt using HF generation."""
    return generate_grouped_responses(
        inference_model=inference_model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=1,
        max_new_tokens=max_new_tokens,
        min_new_tokens=0,
        sampling_temperature=0.0,
        sampling_top_p=1.0,
        stop_strings=stop_strings,
        return_scores=False,
    )


class TokenSequenceStopCriteria(StoppingCriteria):
    """Stop when any of the provided token sequences appears as a suffix."""

    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.shape[1] == 0:
            return False
        for seq in self.stop_sequences:
            L = len(seq)
            if L == 0 or input_ids.shape[1] < L:
                continue
            tail = input_ids[:, -L:]
            stop_tensor = torch.tensor(seq, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            # Check if any sequence in the batch ends with the stop tokens
            if (tail == stop_tensor).all(dim=1).any().item():
                return True
        return False


def _build_stopping_criteria(tokenizer, stop_strings: Optional[List[str]]):
    if not stop_strings:
        return None
    stop_token_ids: List[List[int]] = []
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) > 0:
            stop_token_ids.append(ids)
    if not stop_token_ids:
        return None
    return StoppingCriteriaList([TokenSequenceStopCriteria(stop_token_ids)])


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




