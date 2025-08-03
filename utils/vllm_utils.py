#!/usr/bin/env python3
"""
vLLM utilities for MATH dataset inference.
"""

from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional, Callable


def load_vllm_model(
    model_path: str,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 2048,
    **kwargs
) -> LLM:
    """Load model with vLLM for optimized inference."""
    return LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        **kwargs
    )


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
    eval_sampling_params: SamplingParams
) -> Dict[str, Any]:
    """
    Evaluate a language model on prompts and compute rewards.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    rewards = [
        reward_fn(output.outputs[0].text, ground_truths[i])
        for i, output in enumerate(outputs)
    ]
    
    answer_rewards = [r.get('answer_reward', 0.0) for r in rewards]
    format_rewards = [r.get('format_reward', 0.0) for r in rewards]
    total = len(rewards)
    
    return {
        'responses': [output.outputs[0].text for output in outputs],
        'rewards': rewards,
        'total_samples': total,
        'answer_accuracy': sum(answer_rewards) / total,
        'format_accuracy': sum(format_rewards) / total,
    }