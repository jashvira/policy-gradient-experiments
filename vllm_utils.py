#!/usr/bin/env python3
"""
vLLM utilities for MATH dataset inference.
Includes generation parameters optimized for mathematical reasoning.
"""

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional


def load_vllm_model(model_path: str, **kwargs) -> LLM:
    """Load model with vLLM for optimized inference."""
    print(f"Loading model with vLLM: {model_path}")
    
    default_config = {
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 2048,
        "trust_remote_code": True
    }
    
    # Override defaults with any provided kwargs
    config = {**default_config, **kwargs}
    
    model = LLM(model=model_path, **config)
    return model


def create_sampling_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    stop_tokens: Optional[List[str]] = None
) -> SamplingParams:
    """
    Create sampling parameters for mathematical reasoning.
    
    Based on Dr. GRPO: stop when the model completes its answer
    https://github.com/sail-sg/understand-r1-zero/blob/...
    """
    if stop_tokens is None:
        stop_tokens = ["</answer>"]
    
    # Based on image: generation hyperparameters with stop tokens
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_tokens,
        include_stop_str_in_output=True  # Include stop string in output
    )
    
    return sampling_params


def format_prompt_with_template(question: str, template_path: str) -> str:
    """Format question using specified prompt template."""
    with open(template_path, 'r') as f:
        template = f.read().strip()
    
    return template.format(question=question)


def batch_generate(
    model: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    batch_size: Optional[int] = None
) -> List[str]:
    """
    Generate responses for a batch of prompts.
    
    Args:
        model: vLLM model instance
        prompts: List of input prompts
        sampling_params: Generation parameters
        batch_size: Optional batch size (vLLM handles batching automatically)
        
    Returns:
        List of generated responses
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    
    outputs = model.generate(prompts, sampling_params)
    
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text)
    
    return responses


def setup_math_inference_pipeline(
    model_path: str,
    template_path: str = "prompts/r1_zero.prompt",
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    **model_kwargs
) -> tuple[LLM, SamplingParams, str]:
    """
    Setup complete inference pipeline for MATH dataset.
    
    Returns:
        Tuple of (model, sampling_params, template)
    """
    model = load_vllm_model(model_path, **model_kwargs)
    
    sampling_params = create_sampling_params(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    with open(template_path, 'r') as f:
        template = f.read().strip()
    
    return model, sampling_params, template


def prepare_tokenizer(model_path: str) -> AutoTokenizer:
    """Load and prepare tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer