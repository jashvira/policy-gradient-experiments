#!/usr/bin/env python3
"""
Expert Iteration training script for MATH dataset.
Heavily reuses utilities from SFT implementation.
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
import argparse
import json
import random
from typing import List, Dict, Tuple

import torch
import wandb

from experiments.ei.config_schema import EITrainConfig
from utils.vllm_utils import (
    init_vllm, create_sampling_params, evaluate_vllm, 
    destroy_vllm_instance, cleanup_distributed_process_groups
)
from utils.math_data import load_math_train, load_math_validation, format_with_r1_zero_prompt
from utils.training_utils import (
    tokenize_prompt_and_output, get_response_log_probs,
    sft_microbatch_train_step, compute_entropy
)
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.model_utils import setup_model_and_tokenizer
from utils.optim_sched_utils import build_adamw, build_warmup_then_scheduler


def sample_batch_from_dataset(
    problems: List[str], 
    answers: List[str], 
    metadata: List[Dict], 
    batch_size: int,
    seed: int = None
) -> Tuple[List[str], List[str], List[Dict]]:
    """Randomly sample a batch of problems from the dataset."""
    if seed is not None:
        random.seed(seed)
    
    indices = random.sample(range(len(problems)), min(batch_size, len(problems)))
    sampled_problems = [problems[i] for i in indices]
    sampled_answers = [answers[i] for i in indices]
    sampled_metadata = [metadata[i] for i in indices]
    
    return sampled_problems, sampled_answers, sampled_metadata


def generate_and_filter_responses(
    vllm_model,
    prompts: List[str],
    ground_truths: List[str],
    G: int,
    sampling_params,
    fast_grading: bool = True
) -> Tuple[List[Dict[str, str]], Dict[str, float]]:
    """Generate G rollouts per prompt and filter correct responses.
    
    Returns:
        Tuple of (filtered_data, stats) where stats includes entropy metrics
    """
    print(f"Generating {G} rollouts for {len(prompts)} prompts (total: {len(prompts) * G} responses)...")
    
    # Set n=G in sampling params for G responses per prompt
    sampling_params.n = G
    
    # Generate with vLLM - will generate G responses per prompt
    outputs = vllm_model.generate(prompts, sampling_params)
    
    # Filter correct responses and collect stats
    filtered_data = []
    total_responses = 0
    correct_responses = 0
    
    for i, output in enumerate(outputs):
        ground_truth = ground_truths[i]
        prompt = prompts[i]
        
        # Each output has G responses
        for j in range(len(output.outputs)):
            response = output.outputs[j].text
            total_responses += 1
            
            # Grade the response
            rewards = r1_zero_reward_fn(response, ground_truth, fast=fast_grading)
            if rewards.get("reward", 0.0) > 0:  # Correct answer
                correct_responses += 1
                filtered_data.append({
                    "prompt": prompt,
                    "response": response
                })
    
    success_rate = correct_responses / max(1, total_responses)
    print(f"Filtered to {len(filtered_data)} correct responses from {total_responses} total (success rate: {success_rate:.2%})")
    
    stats = {
        "total_responses": total_responses,
        "correct_responses": correct_responses,
        "success_rate": success_rate,
    }
    
    return filtered_data, stats


def run_sft_on_filtered_data(
    model, tokenizer, filtered_data: List[Dict[str, str]], 
    config: EITrainConfig, device: torch.device
) -> Dict[str, float]:
    """Run SFT training on filtered data. Returns metrics including entropy."""
    if not filtered_data:
        print("Warning: No filtered data for SFT training!")
        return {"loss": float('inf'), "avg_entropy": 0.0}
    
    print(f"Running SFT on {len(filtered_data)} examples for {config.sft_epochs_per_step} epochs...")
    
    # Create optimizer
    optimizer = build_adamw(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        fused=config.adam_fused,
    )
    
    # Create data batches
    prompts = [ex["prompt"] for ex in filtered_data]
    responses = [ex["response"] for ex in filtered_data]
    
    model.train()
    total_loss = 0.0
    total_entropy = 0.0
    step_count = 0
    
    for epoch in range(config.sft_epochs_per_step):
        # Shuffle data each epoch
        paired_data = list(zip(prompts, responses))
        random.shuffle(paired_data)
        shuffled_prompts, shuffled_responses = zip(*paired_data)
        
        # Process in batches
        for i in range(0, len(shuffled_prompts), config.batch_size):
            batch_prompts = list(shuffled_prompts[i:i + config.batch_size])
            batch_responses = list(shuffled_responses[i:i + config.batch_size])
            
            # Tokenize
            tokenized = tokenize_prompt_and_output(
                prompt_strs=batch_prompts,
                output_strs=batch_responses,
                tokenizer=tokenizer,
            )
            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)
            response_mask = tokenized["response_mask"].to(device)
            attention_mask = tokenized.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Get log probs and entropy
            scores = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,  # Enable entropy computation
                requires_grad=True,
                attention_mask=attention_mask,
            )
            policy_log_probs = scores["log_probs"]
            
            # Track entropy
            if "token_entropy" in scores:
                # Compute masked entropy (only for response tokens)
                token_entropy = scores["token_entropy"]
                masked_entropy = token_entropy * response_mask.float()
                avg_entropy = masked_entropy.sum() / response_mask.sum().clamp_min(1)
                total_entropy += avg_entropy.item()
            
            # Compute loss and backprop
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                normalize_constant=1.0,
            )
            
            # Optimizer step
            if (step_count + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            step_count += 1
    
    avg_loss = total_loss / max(1, step_count)
    avg_entropy = total_entropy / max(1, step_count)
    print(f"SFT completed. Average loss: {avg_loss:.4f}, Average entropy: {avg_entropy:.4f}")
    
    return {"loss": avg_loss, "avg_entropy": avg_entropy}


def run_expert_iteration(config: EITrainConfig):
    """Main Expert Iteration training loop."""
    
    # Setup wandb
    resolved_project = os.environ.get("WANDB_PROJECT", config.project)
    resolved_entity = config.wandb_entity or os.environ.get("WANDB_ENTITY")
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception:
            pass
    
    wandb.init(project=resolved_project, name=config.run_name, entity=resolved_entity)
    wandb.config.update(config.to_dict())
    
    # Load data
    print("Loading MATH datasets...")
    train_problems, train_answers, train_metadata = load_math_train()
    val_problems, val_answers, val_metadata = load_math_validation()
    
    # Format validation prompts
    val_prompts = [format_with_r1_zero_prompt(p) for p in val_problems[:config.val_samples]]
    val_gts = val_answers[:config.val_samples]
    
    print(f"Loaded {len(train_problems)} training problems, {len(val_prompts)} validation problems")
    
    # Setup model and tokenizer
    print(f"Loading model: {config.model_name}")
    model, tokenizer, train_device = setup_model_and_tokenizer(
        model_name=config.model_name, 
        train_device=config.train_device
    )
    
    # Validate eval device
    if config.eval_device == config.train_device:
        raise RuntimeError("eval_device must be different from train_device for vLLM")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Expert Iteration loop
    for ei_step in range(config.n_ei_steps):
        print(f"\n=== Expert Iteration Step {ei_step + 1}/{config.n_ei_steps} ===")
        
        # Sample batch of training problems
        sampled_problems, sampled_answers, _ = sample_batch_from_dataset(
            train_problems, train_answers, train_metadata,
            config.batch_size_per_ei_step,
            seed=config.seed + ei_step
        )
        
        # Format prompts
        sampled_prompts = [format_with_r1_zero_prompt(p) for p in sampled_problems]
        
        # Save current model for vLLM
        tmp_model_dir = checkpoint_dir / f"tmp_ei_step_{ei_step}"
        tmp_model_dir.mkdir(exist_ok=True)
        model.save_pretrained(tmp_model_dir)
        tokenizer.save_pretrained(tmp_model_dir)
        
        # Initialize vLLM for generation
        print(f"Initializing vLLM on {config.eval_device}...")
        vllm_model = init_vllm(
            model_id=str(tmp_model_dir),
            device=config.eval_device,
            seed=config.seed,
            gpu_memory_utilization=config.vllm_gpu_mem_utilization,
        )
        
        # Create sampling params
        # Use default stop at </answer> but we need to handle second answer tag in post-processing
        sampling_params = create_sampling_params(
            temperature=config.sampling_temperature,
            top_p=config.sampling_top_p,
            max_tokens=config.sampling_max_tokens,
            stop_tokens=["</answer>"]  # This will stop at first </answer>
        )
        sampling_params.min_tokens = config.sampling_min_tokens
        # Note: n=G is set inside generate_and_filter_responses
        
        try:
            # Generate and filter responses
            filtered_data, gen_stats = generate_and_filter_responses(
                vllm_model, sampled_prompts, sampled_answers, 
                config.G, sampling_params
            )
            
            # Log generation stats
            wandb.log({
                f"ei_step_{ei_step}/generation_success_rate": gen_stats["success_rate"],
                f"ei_step_{ei_step}/num_filtered_examples": len(filtered_data),
                f"ei_step_{ei_step}/num_sampled_problems": len(sampled_prompts),
                f"ei_step_{ei_step}/total_responses_generated": gen_stats["total_responses"],
                "ei_step": ei_step,
            })
            
        finally:
            # Clean up vLLM
            destroy_vllm_instance(vllm_model)
            # Clean up tmp model
            import shutil
            shutil.rmtree(tmp_model_dir, ignore_errors=True)
        
        # Run SFT on filtered data
        if filtered_data:
            sft_metrics = run_sft_on_filtered_data(
                model, tokenizer, filtered_data, config, train_device
            )
            wandb.log({
                f"ei_step_{ei_step}/sft_loss": sft_metrics["loss"],
                f"ei_step_{ei_step}/sft_avg_entropy": sft_metrics["avg_entropy"],
                "ei_step": ei_step,
            })
        else:
            # Log zero entropy if no data
            wandb.log({
                f"ei_step_{ei_step}/sft_loss": float('inf'),
                f"ei_step_{ei_step}/sft_avg_entropy": 0.0,
                "ei_step": ei_step,
            })
        
        # Validation - run at every EI step for proper accuracy curves
        print(f"Running validation at EI step {ei_step + 1}...")
            
            # Save model for validation
            val_model_dir = checkpoint_dir / f"ei_step_{ei_step}_model"
            val_model_dir.mkdir(exist_ok=True)
            model.save_pretrained(val_model_dir)
            tokenizer.save_pretrained(val_model_dir)
            
            # Initialize vLLM for validation
            val_vllm = init_vllm(
                model_id=str(val_model_dir),
                device=config.eval_device,
                seed=config.seed,
                gpu_memory_utilization=config.vllm_gpu_mem_utilization,
            )
            
            val_sampling_params = create_sampling_params(
                temperature=config.val_temperature,
                top_p=config.val_top_p,
                max_tokens=config.max_new_tokens,
                stop_tokens=["</answer>"]
            )
            
            try:
                # Run validation
                val_results = evaluate_vllm(
                    vllm_model=val_vllm,
                    reward_fn=r1_zero_reward_fn,
                    prompts=val_prompts,
                    ground_truths=val_gts,
                    eval_sampling_params=val_sampling_params,
                    output_dir=str(checkpoint_dir / "eval_results"),
                    model_name=f"{config.run_name}_ei_step_{ei_step}",
                    save_full_responses=False,
                    write_results=True,
                )
                
                # Log validation metrics
                metrics = val_results.get("metadata", {}).get("overall_metrics", {})
                wandb.log({
                    f"ei_step_{ei_step}/val_answer_accuracy": metrics.get("answer_accuracy", 0.0),
                    f"ei_step_{ei_step}/val_format_accuracy": metrics.get("format_accuracy", 0.0),
                    f"ei_step_{ei_step}/val_overall_accuracy": metrics.get("overall_accuracy", 0.0),
                    "ei_step": ei_step,
                })
                
                print(f"Validation accuracy: {metrics.get('answer_accuracy', 0.0):.4f}")
                
            finally:
                destroy_vllm_instance(val_vllm)
        
        # Save checkpoint
        if config.save_checkpoints:
            step_checkpoint_dir = checkpoint_dir / f"ei_step_{ei_step}_final"
            model.save_pretrained(step_checkpoint_dir)
            tokenizer.save_pretrained(step_checkpoint_dir)
            print(f"Saved checkpoint to {step_checkpoint_dir}")
    
    print("\nExpert Iteration training completed!")


def main():
    # Register cleanup
    atexit.register(cleanup_distributed_process_groups)
    
    parser = argparse.ArgumentParser(description="Expert Iteration training on MATH dataset")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = EITrainConfig.from_path(config_path)
    config = config.resolve(repo_root=_REPO_ROOT)
    
    print(f"Starting Expert Iteration with config: {config_path}")
    print(f"Model: {config.model_name}")
    print(f"EI steps: {config.n_ei_steps}, G={config.G}, batch_size={config.batch_size_per_ei_step}")
    
    run_expert_iteration(config)


if __name__ == "__main__":
    main()