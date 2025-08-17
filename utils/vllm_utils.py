#!/usr/bin/env python3
"""
vLLM utilities for MATH dataset inference.
"""

import atexit
import json
import os
import importlib
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

import torch
from transformers import PreTrainedModel
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from utils.drgrpo_grader import extract_answer


def _worker_load_hf_state(self, sd):
    """Worker-side loader used via LLM.collective_rpc on vLLM v1.

    Expects `self` to be a worker object. Finds the HF model and loads `sd`.
    Returns minimal diagnostics.
    """
    model = None
    try:
        if hasattr(self, "model_runner") and hasattr(self.model_runner, "model"):
            model = self.model_runner.model
        elif hasattr(self, "model"):
            model = self.model
    except Exception:
        model = None
    if model is None:
        raise RuntimeError("vLLM worker model handle not found for weight update")
    try:
        res = model.load_state_dict(sd, strict=True)
    except Exception:
        res = model.load_state_dict(sd, strict=False)
    missing = getattr(res, "missing_keys", []) if res is not None else []
    unexpected = getattr(res, "unexpected_keys", []) if res is not None else []
    return {"missing": missing, "unexpected": unexpected}

def cleanup_distributed_process_groups():
    """Clean up any distributed process groups to prevent resource leaks."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        # Silently ignore errors during cleanup
        pass


def ensure_distributed_cleanup():
    """Ensure distributed cleanup happens at program exit."""
    atexit.register(cleanup_distributed_process_groups)


def destroy_vllm_instance(llm_instance: LLM) -> None:
    """Properly destroy a vLLM instance with distributed cleanup."""
    try:
        # First try to clean up the LLM instance
        if hasattr(llm_instance, 'llm_engine'):
            engine = llm_instance.llm_engine
            # Try to get the driver worker to clean up properly
            if hasattr(engine, 'model_executor') and hasattr(engine.model_executor, 'driver_worker'):
                worker = engine.model_executor.driver_worker
                if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
                    # Clear the model to free memory
                    del worker.model_runner.model

        # Clean up the instance itself
        del llm_instance

        # Force cleanup of any distributed state
        cleanup_distributed_process_groups()

        # Force garbage collection and CUDA cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    except Exception:
        # Silently handle any cleanup errors
        pass


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
    # Default to safer settings across vLLM versions
    kwargs.setdefault("trust_remote_code", True)
    kwargs.setdefault("dtype", "auto")
    return LLM(model=model_path, **kwargs)


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    *,
    dtype: Union[str, torch.dtype, None] = "auto",
    trust_remote_code: bool = True,
) -> LLM:
    """Initialize a vLLM instance on a specific device with patches for single-GPU usage.

    Applies TRL-style patches to ensure world size = 1 and disable profiling assertions.
    Places the model on the specified device and sets a deterministic seed.
    """
    # Ensure distributed cleanup is registered for program exit
    ensure_distributed_cleanup()

    vllm_set_random_seed(seed)

    # Monkeypatch per TRL to ensure single-process behavior and avoid profiling checks
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # Make the profiling patch resilient across vLLM versions
    profiling_patch = nullcontext()
    try:
        worker_mod = importlib.import_module("vllm.worker.worker")
        if hasattr(worker_mod, "Worker") and hasattr(
            worker_mod.Worker, "_assert_memory_footprint_increased_during_profiling"
        ):
            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                return_value=None,
            )
    except Exception:
        profiling_patch = nullcontext()

    # Select device by constraining visible GPUs; vLLM selects GPU 0 of the visible set
    if device.startswith("cuda:"):
        # Map 'cuda:N' to only expose that GPU
        try:
            gpu_index = device.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        except Exception:
            pass
    elif device == "cpu":
        # vLLM CPU is not generally supported; caller should have handled fallback
        raise RuntimeError("vLLM does not support CPU inference in this configuration")

    with world_size_patch, profiling_patch:
        llm_kwargs: Dict[str, Any] = {
            "model": model_id,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }
        if isinstance(dtype, torch.dtype):
            # vLLM accepts strings like "bfloat16"; map common torch dtypes
            dtype_map = {
                torch.float16: "float16",
                torch.half: "float16",
                torch.bfloat16: "bfloat16",
                torch.float32: "float32",
            }
            llm_kwargs["dtype"] = dtype_map.get(dtype, "auto")
        elif dtype is not None:
            llm_kwargs["dtype"] = dtype
        return LLM(**llm_kwargs)


def load_policy_into_vLLM_instance_from_checkpoint(
    *,
    policy: PreTrainedModel,
    tokenizer,
    eval_device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    tmp_dir: Optional[Union[str, Path]] = None,
    existing_llm: Optional[LLM] = None,
) -> LLM:
    """
    Load policy weights into the running vLLM engine by directly updating the
    worker's HF model (Brandon's approach, works on pre‑v1 vLLM APIs).

    Steps:
      - Build a clean HF state_dict (unwrap torch.compile; strip '_orig_mod.' prefixes)
      - Reach into llm.llm_engine.model_executor.driver_worker.model_runner.model
        and call load_state_dict on that HF module
      - Reset prefix/KV caches after the swap so new weights are used cleanly

    Returns the same LLM instance (weights updated in-place).
    """
    if existing_llm is None:
        raise ValueError("existing_llm must be provided for direct weight loading")
    
    # Prepare a clean CPU state_dict and strip torch.compile wrappers
    raw_state_dict = policy.state_dict()
    clean_state_dict = {}
    for key, value in raw_state_dict.items():
        clean_key = key
        while clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]
        clean_state_dict[clean_key] = value.detach().to("cpu").contiguous()

    # Try Brandon's direct worker path (pre-v1 API)
    engine = getattr(existing_llm, "llm_engine", None)
    if engine is None:
        raise RuntimeError("vLLM engine not available on LLM instance")
    model = None
    try:
        model = engine.model_executor.driver_worker.model_runner.model
    except Exception:
        # If the path is missing, this version likely doesn't support direct swapping
        raise RuntimeError("Direct worker path not found on this vLLM; use v1 RPC + NCCL broadcast instead")

    # Load state dict onto worker model
    try:
        model.load_state_dict(clean_state_dict, strict=True)
    except Exception:
        model.load_state_dict(clean_state_dict, strict=False)

    # Reset caches
    if hasattr(existing_llm, "reset_prefix_cache"):
        try:
            existing_llm.reset_prefix_cache()
        except Exception:
            pass

    return existing_llm


# Backwards-compat wrapper to avoid breaking older call sites
def load_policy_into_vllm_instance(
    policy: PreTrainedModel,
    llm: LLM,
    *,
    tokenizer=None,
    eval_device: Optional[str] = None,
    seed: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    tmp_dir: Optional[Union[str, Path]] = None,
):
    """
    Load policy weights directly into vLLM instance via state_dict injection.
    
    Returns the same LLM instance (weights updated in-place).
    The tokenizer, eval_device, seed, gpu_memory_utilization, and tmp_dir parameters 
    are kept for backwards compatibility but are no longer used.
    """
    return load_policy_into_vLLM_instance_from_checkpoint(
        policy=policy,
        tokenizer=tokenizer,  # unused but kept for compatibility
        eval_device=eval_device,  # unused but kept for compatibility
        seed=seed,  # unused but kept for compatibility
        gpu_memory_utilization=gpu_memory_utilization or 0.85,  # unused but kept for compatibility
        tmp_dir=tmp_dir,  # unused but kept for compatibility
        existing_llm=llm,
    )


def create_sampling_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    min_tokens: int = 0,
    n: int = 1,
    stop_tokens: Optional[List[str]] = None
) -> SamplingParams:
    """Create sampling parameters for mathematical reasoning."""
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        n=n,
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
    save_full_responses: bool = False,
    write_results: bool = True,
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

    # Save results to disk (optional)
    if write_results:
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