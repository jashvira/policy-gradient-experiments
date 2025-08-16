#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimisation) training script.
Implements Algorithm 3 from the GRPO paper.
"""

import os
# Set memory allocation config before any PyTorch imports
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
from pathlib import Path

# Ensure repository root is on sys.path for absolute imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import atexit
import json
import random
import math
import copy
from typing import List, Dict, Tuple, Any
from datetime import datetime
from pathlib import Path

import torch
import wandb
import numpy as np
import typer

from experiments.grpo.config_schema import GRPOTrainConfig
from experiments.grpo.grpo_utils import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step
)
from utils.vllm_utils import (
    init_vllm, create_sampling_params, load_policy_into_vllm_instance, 
    evaluate_vllm, destroy_vllm_instance
)
from utils.math_data import load_math_train, load_math_validation, format_with_r1_zero_prompt
from utils.training_utils import tokenize_prompt_and_output, get_response_log_probs, compute_entropy
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.model_utils import setup_model_and_tokenizer
from utils.optim_sched_utils import build_adamw, build_warmup_then_scheduler


def setup_wandb(config: GRPOTrainConfig):
    """Initialize wandb logging."""
    resolved_project = os.environ.get("WANDB_PROJECT", config.project)
    resolved_entity = os.environ.get("WANDB_ENTITY", config.wandb_entity)
    
    wandb.init(
        project=resolved_project,
        entity=resolved_entity,
        name=config.run_name,
        config=config.__dict__,
        save_code=True,
    )
    print(f"Initialized wandb: project={resolved_project}, entity={resolved_entity}")


def sample_rollout_batch(
    problems: List[str],
    answers: List[str],
    n_prompts: int,
    group_size: int,
    vllm_model,
    sampling_params,
    seed: int = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Sample a rollout batch: n_prompts questions, each with group_size responses.
    
    Returns:
        Tuple of (all_prompts, all_responses, all_ground_truths) where each list
        has length n_prompts * group_size
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample n_prompts questions
    indices = random.sample(range(len(problems)), min(n_prompts, len(problems)))
    sampled_problems = [problems[i] for i in indices]
    sampled_answers = [answers[i] for i in indices]
    
    # Format prompts with r1_zero format
    prompts = [format_with_r1_zero_prompt(problem) for problem in sampled_problems]
    
    print(f"Generating rollout batch: {n_prompts} prompts × {group_size} responses = {n_prompts * group_size} total")
    
    # Generate group_size responses per prompt
    sampling_params.n = group_size
    outputs = vllm_model.generate(prompts, sampling_params)
    
    # Flatten to get all prompts, responses, and ground truths
    all_prompts = []
    all_responses = []
    all_ground_truths = []
    
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        ground_truth = sampled_answers[i]
        
        # Each output has group_size responses
        for j in range(len(output.outputs)):
            response = output.outputs[j].text
            all_prompts.append(prompt)
            all_responses.append(response)
            all_ground_truths.append(ground_truth)
    
    print(f"Generated {len(all_responses)} responses total")
    return all_prompts, all_responses, all_ground_truths


def train_on_rollout_batch(
    model,
    tokenizer,
    all_prompts: List[str],
    all_responses: List[str],
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    config: GRPOTrainConfig,
    optimizer,
    device: torch.device,
    all_ground_truths: List[str],
) -> Dict[str, float]:
    """
    Train on a rollout batch using the computed advantages.
    
    Returns metrics dictionary.
    """
    model.train()
    
    # Tokenize all prompt-response pairs
    tokenized = tokenize_prompt_and_output(all_prompts, all_responses, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)
    
    # Get current policy log probs and token entropy
    policy_outputs = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=True
    )
    policy_log_probs = policy_outputs["log_probs"]
    
    # Compute token entropy if not returned by get_response_log_probs
    token_entropy = policy_outputs.get("token_entropy")
    if token_entropy is None:
        with torch.no_grad():
            logits = model(input_ids).logits
            token_entropy = compute_entropy(logits)
    
    # Compute train rewards for logging
    train_rewards = {"format": [], "answer": [], "total": []}
    for response, ground_truth in zip(all_responses, all_ground_truths):
        reward_dict = r1_zero_reward_fn(response, ground_truth, fast=True)
        train_rewards["format"].append(reward_dict.get("format_reward", 0.0))
        train_rewards["answer"].append(reward_dict.get("answer_reward", 0.0))
        train_rewards["total"].append(reward_dict.get("reward", 0.0))
    
    # Split into microbatches and train
    batch_size = len(all_prompts)
    micro_batch_size = config.micro_train_batch_size
    total_loss = 0.0
    step_count = 0
    all_metadata = {}
    total_clip_fraction = 0.0
    total_entropy = 0.0
    
    # Zero gradients at start
    optimizer.zero_grad()
    
    for start_idx in range(0, batch_size, micro_batch_size):
        end_idx = min(start_idx + micro_batch_size, batch_size)
        
        # Extract microbatch
        micro_policy_log_probs = policy_log_probs[start_idx:end_idx]
        micro_response_mask = response_mask[start_idx:end_idx]
        micro_advantages = advantages[start_idx:end_idx].unsqueeze(1)  # (micro_batch, 1)
        micro_old_log_probs = old_log_probs[start_idx:end_idx] if old_log_probs is not None else None
        micro_token_entropy = token_entropy[start_idx:end_idx]
        
        # Ensure gradients are enabled for microbatch
        micro_policy_log_probs.requires_grad_(True)
        
        # Run microbatch train step
        loss, metadata = grpo_microbatch_train_step(
            policy_log_probs=micro_policy_log_probs,
            response_mask=micro_response_mask,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            loss_type=config.loss_type,
            raw_rewards=None,  # We use advantages, not raw rewards
            advantages=micro_advantages,
            old_log_probs=micro_old_log_probs,
            cliprange=config.cliprange if config.loss_type == "grpo_clip" else None,
        )
        
        total_loss += loss.item()
        step_count += 1
        
        # Accumulate metadata from first microbatch for logging
        if start_idx == 0:
            all_metadata = {k: v.item() if torch.is_tensor(v) else v for k, v in metadata.items()}
            
        # Accumulate clip fraction if using grpo_clip
        if config.loss_type == "grpo_clip" and "clip_fraction" in metadata:
            total_clip_fraction += metadata["clip_fraction"]
            
        # Accumulate token entropy
        masked_entropy = micro_token_entropy * micro_response_mask.float()
        avg_entropy = masked_entropy.sum() / micro_response_mask.sum().clamp_min(1)
        total_entropy += avg_entropy.item()
    
    # Take optimizer step after accumulating gradients and capture gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
    optimizer.step()
    
    avg_loss = total_loss / step_count
    avg_clip_fraction = total_clip_fraction / step_count if config.loss_type == "grpo_clip" else 0.0
    avg_entropy = total_entropy / step_count
    
    return {
        "train_loss": avg_loss,
        "grad_norm": grad_norm.item(),
        "token_entropy": avg_entropy,
        "clip_fraction": avg_clip_fraction,
        "train_format_reward": np.mean(train_rewards["format"]),
        "train_answer_reward": np.mean(train_rewards["answer"]),
        "train_total_reward": np.mean(train_rewards["total"]),
        "num_microbatches": step_count,
        **all_metadata
    }


def run_grpo_step(
    model,
    tokenizer,
    old_model,
    problems: List[str],
    answers: List[str],
    config: GRPOTrainConfig,
    vllm_model,
    sampling_params,
    optimizer,
    device: torch.device,
    step: int,
) -> Dict[str, Any]:
    """
    Run a single GRPO step following Algorithm 3.
    
    Returns metrics dictionary.
    """
    print(f"\n=== GRPO Step {step + 1}/{config.n_grpo_steps} ===")
    
    # Sample rollout batch (Algorithm 3, Line 5)
    all_prompts, all_responses, all_ground_truths = sample_rollout_batch(
        problems=problems,
        answers=answers,
        n_prompts=config.n_prompts_per_rollout_batch,
        group_size=config.group_size,
        vllm_model=vllm_model,
        sampling_params=sampling_params,
        seed=config.seed + step,
    )
    
    # Compute rewards (Algorithm 3, Line 6)
    print("Computing rewards...")
    raw_rewards = []
    for response, ground_truth in zip(all_responses, all_ground_truths):
        reward_dict = r1_zero_reward_fn(response, ground_truth, fast=True)
        raw_rewards.append(reward_dict["reward"])
    
    # Compute advantages with group normalization (Algorithm 3, Line 7)
    print("Computing group-normalized advantages...")
    advantages, raw_rewards_tensor, reward_metadata = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=all_responses,
        repeated_ground_truths=all_ground_truths,
        group_size=config.group_size,
        advantage_eps=config.advantage_eps,
        normalize_by_std=config.use_std_normalization,
    )
    
    # Get old policy log probs if needed for grpo_clip
    old_log_probs = None
    if config.loss_type == "grpo_clip":
        print("Computing old policy log probs...")
        tokenized = tokenize_prompt_and_output(all_prompts, all_responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        
        with torch.no_grad():
            old_outputs = get_response_log_probs(
                model=old_model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False
            )
            old_log_probs = old_outputs["log_probs"].detach()  # Ensure no gradients
    
    # Training loop (Algorithm 3, Lines 8-9)
    print(f"Training for {config.epochs_per_rollout_batch} epochs...")
    epoch_metrics = []
    
    for epoch in range(config.epochs_per_rollout_batch):
        epoch_metrics_dict = train_on_rollout_batch(
            model=model,
            tokenizer=tokenizer,
            all_prompts=all_prompts,
            all_responses=all_responses,
            advantages=advantages,
            old_log_probs=old_log_probs,
            config=config,
            optimizer=optimizer,
            device=device,
            all_ground_truths=all_ground_truths,
        )
        epoch_metrics.append(epoch_metrics_dict)
    
    # Update old policy (Algorithm 3, Line 10 - implicitly, we'll copy weights)
    # Note: For efficiency, we update old_model at the end of each step
    
    # Aggregate metrics
    metrics = {
        "step": step + 1,
        "rollout_batch_size": len(all_responses),
        "n_prompts": config.n_prompts_per_rollout_batch,
        "group_size": config.group_size,
        **reward_metadata,
        "avg_train_loss": np.mean([m["train_loss"] for m in epoch_metrics]),
    }
    
    # Add first epoch's detailed metrics
    if epoch_metrics:
        for key, value in epoch_metrics[0].items():
            if key not in metrics:
                metrics[f"epoch0_{key}"] = value
    
    return metrics


def copy_model_weights(source_model, target_model):
    """Copy weights from source_model to target_model."""
    target_model.load_state_dict(source_model.state_dict())


def evaluate_on_validation(
    vllm_model,
    val_problems: List[str],
    val_answers: List[str],
    config: GRPOTrainConfig,
    step: int,
) -> Dict[str, float]:
    """Run evaluation on validation set using existing evaluate_vllm."""
    print("Running validation evaluation...")
    
    # Sample subset for evaluation (use val_samples instead of val_rollout_batch_size)
    n_val_samples = min(config.val_samples, len(val_problems))
    indices = random.sample(range(len(val_problems)), n_val_samples)
    
    val_prompts = [format_with_r1_zero_prompt(val_problems[i]) for i in indices]
    val_ground_truths = [val_answers[i] for i in indices]
    
    # Create evaluation sampling params (greedy)
    eval_sampling_params = create_sampling_params(
        temperature=0.0,  # Greedy for evaluation
        top_p=1.0,
        max_tokens=config.sampling_max_tokens,
        min_tokens=1,
        n=1,
    )
    
    # Use existing evaluate_vllm function
    results = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=val_prompts,
        ground_truths=val_ground_truths,
        eval_sampling_params=eval_sampling_params,
        output_dir=config.val_log_dir,
        model_name=f"grpo_step_{step}",
        write_results=True,
    )
    
    return {
        "val_accuracy": results["accuracy"],
        "val_format_accuracy": results["format_accuracy"],
        "val_answer_accuracy": results["answer_accuracy"],
        "val_total": results["total"],
    }


def main(
    config: str = typer.Option(..., help="Path to YAML config file"),
    resume: str = typer.Option(None, help="Path to checkpoint to resume from")
):
    """GRPO (Group Relative Policy Optimisation) training for mathematical reasoning."""
    
    # Load and resolve config
    config_obj = GRPOTrainConfig.from_path(config)
    config_obj.resolve()
    
    print(f"GRPO Config: {config_obj}")
    
    # Set random seeds
    torch.manual_seed(config_obj.seed)
    np.random.seed(config_obj.seed)
    random.seed(config_obj.seed)
    
    # Setup wandb
    setup_wandb(config_obj)
    
    # Load data
    print("Loading MATH dataset...")
    train_problems, train_answers = load_math_train()
    val_problems, val_answers = load_math_validation()
    print(f"Loaded {len(train_problems)} training problems, {len(val_problems)} validation problems")
    
    # Setup models and tokenizer
    print("Setting up models...")
    model, tokenizer, device = setup_model_and_tokenizer(
        model_name=config_obj.model_name,
        train_device=config_obj.train_device,
    )
    
    # Create old model copy for GRPO
    old_model = copy.deepcopy(model)
    
    # Setup optimizer
    optimizer = build_adamw(
        model.parameters(),
        lr=config_obj.learning_rate,
        weight_decay=config_obj.weight_decay,
        betas=(config_obj.adam_beta1, config_obj.adam_beta2),
        eps=config_obj.adam_eps,
        fused=config_obj.adam_fused,
    )
    
    # Setup vLLM for generation
    print("Initializing vLLM...")
    vllm_model = init_vllm(
        model_id=config_obj.model_name,
        device=config_obj.eval_device,
        seed=config_obj.seed,
        gpu_memory_utilization=config_obj.gpu_memory_utilization,
    )
    
    sampling_params = create_sampling_params(
        temperature=config_obj.sampling_temperature,
        top_p=config_obj.sampling_top_p,
        max_tokens=config_obj.sampling_max_tokens,
        min_tokens=config_obj.sampling_min_tokens,
        n=1,  # Will be set per call
    )
    
    # Training loop
    print(f"Starting GRPO training for {config_obj.n_grpo_steps} steps...")
    
    try:
        for step in range(config_obj.n_grpo_steps):
            # Load current model weights into vLLM
            load_policy_into_vllm_instance(vllm_model, model)
            
            # Run GRPO step
            step_metrics = run_grpo_step(
                model=model,
                tokenizer=tokenizer,
                old_model=old_model,
                problems=train_problems,
                answers=train_answers,
                config=config_obj,
                vllm_model=vllm_model,
                sampling_params=sampling_params,
                optimizer=optimizer,
                device=device,
                step=step,
            )
            
            # Update old model with current weights
            copy_model_weights(model, old_model)
            
            # Log metrics
            wandb.log(step_metrics, step=step)
            print(f"Step {step + 1} metrics: {step_metrics}")
            
            # Periodic validation
            if (step + 1) % config_obj.val_every_grpo_steps == 0:
                val_metrics = evaluate_on_validation(
                    vllm_model=vllm_model,
                    val_problems=val_problems,
                    val_answers=val_answers,
                    config=config_obj,
                    step=step,
                )
                wandb.log(val_metrics, step=step)
                print(f"Validation metrics: {val_metrics}")
            
            # Periodic checkpoint saving
            if (step + 1) % config_obj.save_every_grpo_steps == 0:
                save_path = Path(config_obj.save_dir) / f"checkpoint_step_{step + 1}"
                save_path.mkdir(parents=True, exist_ok=True)
                
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
    
    finally:
        # Cleanup
        print("Cleaning up vLLM...")
        destroy_vllm_instance(vllm_model)
        
        # Final save
        if config_obj.save_at_end:
            final_save_path = Path(config_obj.save_dir) / "final_model"
            final_save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_save_path)
            tokenizer.save_pretrained(final_save_path)
            print(f"Saved final model to {final_save_path}")
        
        wandb.finish()


if __name__ == "__main__":
    typer.run(main)