#!/usr/bin/env python3
"""
vLLM utilities for MATH dataset inference.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from typing import List, Dict, Any, Optional, Callable
from unittest.mock import patch
import torch
from transformers import PreTrainedModel
from utils.drgrpo_grader import extract_answer


def load_vllm_model(model_path: str, **kwargs) -> LLM:
    """
    Load model with vLLM for optimized inference.

    Common optional parameters (pass via kwargs if needed):
    - gpu_memory_utilization: GPU memory fraction to use (default 0.9)
    - max_model_len: Max context length (default: model's max)
    - max_num_seqs: Max concurrent sequences/batch size (default 256)
    - max_num_batched_tokens: Tokens per forward pass (default 512)
    - trust_remote_code: Allow custom model code (some models require this)
    """
    return LLM(model=model_path, **kwargs)


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    """Initialize a vLLM instance on a specific device with patches for single-GPU usage.

    Applies TRL-style patches to ensure world size = 1 and disable profiling assertions.
    Places the model on the specified device and sets a deterministic seed.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch per TRL to ensure single-process behavior and avoid profiling checks
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Load HF policy weights into an existing vLLM LLM instance.

    Mirrors TRL's approach of pushing a policy state_dict into vLLM's model runner.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def create_sampling_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    stop_tokens: Optional[List[str]] = None
) -> SamplingParams:
    """Create sampling parameters for mathematical reasoning."""
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_tokens or ["</answer>"],
        include_stop_str_in_output=True
    )


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    problem_metadata: Optional[List[Dict]] = None,
    save_full_responses: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a language model on prompts and compute rewards.
    Automatically saves results to disk with timestamp.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    total = len(outputs)

    # Initialize metrics
    n_format_correct = 0
    n_format_incorrect = 0
    n_answer_correct = 0
    n_answer_incorrect = 0
    n_correct = 0
    n_incorrect = 0

    # Create structured per-problem results
    problem_results = []
    subject_stats = {}

    for i, output in enumerate(outputs):
        # Calculate reward for this output
        reward = reward_fn(output.outputs[0].text, ground_truths[i])

        # Extract metadata if available
        metadata = problem_metadata[i] if problem_metadata else {}
        subject = metadata.get('subject', 'Unknown')

        # Update metrics
        format_correct = reward.get('format_reward', 0.0)
        answer_correct = reward.get('answer_reward', 0.0)
        correct = reward.get('reward', 0.0)

        n_format_correct += int(format_correct)
        n_format_incorrect += int(not format_correct)
        n_answer_correct += int(answer_correct)
        n_answer_incorrect += int(not answer_correct)
        n_correct += int(correct)
        n_incorrect += int(not correct)

        # Track subject statistics
        if subject not in subject_stats:
            subject_stats[subject] = {'correct': 0, 'total': 0}
        subject_stats[subject]['total'] += 1
        if answer_correct > 0:
            subject_stats[subject]['correct'] += 1

        # Get response text
        response = output.outputs[0].text

        # Create problem result
        problem_result = {
            'id': metadata.get('unique_id', f'problem_{i}'),
            'subject': subject,
            'level': metadata.get('level', 'Unknown'),
            'question': metadata.get('problem', prompts[i].split('User: ')[-1].split('\nAssistant:')[0] if 'User: ' in prompts[i] else 'N/A'),
            'ground_truth': ground_truths[i],
            'r1_format_extracted': response.split("<answer>")[-1].replace("</answer>", "") if "</think> <answer>" in response and "</answer>" in response else None,
            'correct': answer_correct > 0,
            'rewards': reward
        }
        if save_full_responses:
            problem_result['full_response'] = response
        problem_results.append(problem_result)

    # Calculate subject accuracies
    for subject in subject_stats:
        if subject_stats[subject]['total'] > 0:
            subject_stats[subject]['accuracy'] = subject_stats[subject]['correct'] / subject_stats[subject]['total']

    # Extract prompt template (assume all prompts use same template)
    prompt_template = prompts[0].replace(prompts[0].split('User: ')[-1].split('\nAssistant:')[0], '{question}') if prompts else None

    results = {
        'metadata': {
            'model_name': model_name or 'unknown',
            'timestamp': datetime.now().isoformat(),
            'total_samples': total,
            'prompt_template': prompt_template,
            'sampling_params': {
                'temperature': eval_sampling_params.temperature,
                'top_p': eval_sampling_params.top_p,
                'max_tokens': eval_sampling_params.max_tokens,
                'stop': eval_sampling_params.stop
            },
            'overall_metrics': {
                'answer_accuracy': n_answer_correct / total,
                'format_accuracy': n_format_correct / total,
                'overall_accuracy': n_correct / total,
                'n_format_correct': n_format_correct,
                'n_format_incorrect': n_format_incorrect,
                'n_answer_correct': n_answer_correct,
                'n_answer_incorrect': n_answer_incorrect,
                'n_correct': n_correct,
                'n_incorrect': n_incorrect
            }
        },
        'subject_breakdown': subject_stats,
        'results': problem_results
    }

    # Save results to disk
    if output_dir is None:
        output_dir = "evaluation_results"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = model_name.replace("/", "_") if model_name else "model"
    filename = f"{model_prefix}_eval_{timestamp}.json"

    filepath = output_path / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to: {filepath}")

    return results