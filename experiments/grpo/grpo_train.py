#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimisation) training script.
Implements Algorithm 3 from the GRPO paper.
"""

from utils.inference_utils import (
    init_inference_model_from,
    generate_grouped_responses,
    greedy_generate_responses,
    compute_reward_metrics,
)
from utils.optim_sched_utils import build_adamw, build_warmup_then_scheduler
from utils.model_utils import setup_model_and_tokenizer
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.training_utils import tokenize_prompt_and_output, get_response_log_probs, compute_entropy, compute_log_probs_for_responses
from utils.math_data import load_math_train, load_math_validation, format_with_r1_zero_prompt
from experiments.grpo.grpo_utils import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step
)
from experiments.grpo.config_schema import GRPOTrainConfig
import typer
import numpy as np
import wandb
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
import copy
import math
import random
import json
import atexit
from pathlib import Path
import sys
import os
# Set memory allocation config and logging options before any PyTorch imports
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# Reduce verbose PyTorch logging and error verbosity
# Use valid TORCH_LOGS syntax: optional +/- prefix, no equals. "-inductor" => ERROR level.
os.environ.setdefault('TORCH_LOGS', '-inductor')  # Quiet inductor logs and avoid empty string bug
os.environ.setdefault('PYTORCH_DISABLE_STACK_TRACE', '1')  # Reduce stack trace verbosity


# Ensure repository root is on sys.path for absolute imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Suppress massive tensor dumps and model architecture in error messages to keep logs clean
def suppress_large_tensor_repr():
    """Prevent massive tensor dumps and verbose model architecture in error messages to keep logs clean."""
    # Store originals
    original_tensor_repr = torch.Tensor.__repr__
    original_tensor_str = torch.Tensor.__str__
    original_module_repr = torch.nn.Module.__repr__

    def clean_tensor_repr(self):
        if self.numel() > 10:  # Smaller threshold
            shape_str = "x".join(map(str, self.shape))
            return f"tensor({shape_str}, dtype={self.dtype}, device={self.device})"
        return original_tensor_repr(self)

    def clean_tensor_str(self):
        if self.numel() > 10:
            shape_str = "x".join(map(str, self.shape))
            return f"tensor({shape_str}, dtype={self.dtype}, device={self.device})"
        return original_tensor_str(self)

    def clean_module_repr(self):
        """Show just class name instead of full architecture for large models."""
        class_name = self.__class__.__name__
        # For large models, just show the class name
        if hasattr(self, 'config') and hasattr(self.config, 'num_hidden_layers'):
            return f"{class_name}(layers={self.config.num_hidden_layers})"
        elif len(list(self.parameters())) > 100:  # Large model heuristic
            return f"{class_name}(...)"
        return original_module_repr(self)

    # Override both repr and str methods
    torch.Tensor.__repr__ = clean_tensor_repr
    torch.Tensor.__str__ = clean_tensor_str
    torch.nn.Module.__repr__ = clean_module_repr

    # Note: numpy arrays are immutable types, so we can't patch them
    # but torch tensor patching is sufficient for our use case

# Set up clean logging once at module level
suppress_large_tensor_repr()

# Configure cleaner error handling and warnings
import warnings
# Filter out common but non-critical warnings that clutter logs
warnings.filterwarnings('ignore', message='.*TensorFloat32.*')
warnings.filterwarnings('ignore', message='.*flash_attention.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor.*')
# Suppress LaTeX macro warnings from grader
warnings.filterwarnings('ignore', message='.*macro.*failed its substitution.*')
warnings.filterwarnings('ignore', message='.*Error in configuration.*')


# vLLM removed; we will use plain Hugging Face generation on a separate inference model/device


def setup_wandb(config: GRPOTrainConfig):
    """Initialize wandb logging."""
    resolved_project = os.environ.get("WANDB_PROJECT", config.project)
    resolved_entity = os.environ.get("WANDB_ENTITY", config.wandb_entity)

    # Skip wandb initialization if project is None or empty
    if not resolved_project:
        print("Wandb disabled (project is None or empty)")
        return

    # Add timestamp to run name to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_name = f"{config.run_name}_{timestamp}"

    wandb.init(
        project=resolved_project,
        entity=resolved_entity,
        name=unique_run_name,
        config=config.__dict__,
        save_code=True,
    )
    print(
        f"Initialized wandb: project={resolved_project}, entity={resolved_entity}, run_name={unique_run_name}")


def sample_rollout_batch(
    problems: List[str],
    answers: List[str],
    n_prompts: int,
    group_size: int,
    inference_model,
    tokenizer,
    config: GRPOTrainConfig,
    seed: int = None,
    stop_strings: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Sample a rollout batch: n_prompts questions, each with group_size responses.

    Returns:
        Tuple of (all_prompts, all_responses, all_ground_truths)
        where all lists have length n_prompts * group_size.
    """
    if seed is not None:
        random.seed(seed)

    # Sample n_prompts questions
    indices = random.sample(range(len(problems)),
                            min(n_prompts, len(problems)))
    sampled_problems = [problems[i] for i in indices]
    sampled_answers = [answers[i] for i in indices]

    # Format prompts with r1_zero format
    prompts = [format_with_r1_zero_prompt(
        problem) for problem in sampled_problems]

    print(
        f"Generating rollout batch: {n_prompts} prompts × {group_size} responses = {n_prompts * group_size} total")

    # Generate grouped responses via utility (no output_scores for memory efficiency)
    responses = generate_grouped_responses(
        inference_model=inference_model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=group_size,
        max_new_tokens=config.sampling_max_tokens,
        min_new_tokens=getattr(config, "sampling_min_tokens", 0),
        sampling_temperature=config.sampling_temperature,
        sampling_top_p=config.sampling_top_p,
        stop_strings=stop_strings,
        return_scores=False,
    )

    # Build flattened results aligned with responses
    all_prompts: List[str] = []
    all_responses: List[str] = []
    all_ground_truths: List[str] = []
    for prompt_text, answer in zip(prompts, sampled_answers):
        for _ in range(group_size):
            all_prompts.append(prompt_text)
            all_responses.append(responses[len(all_responses)])
            all_ground_truths.append(answer)

    print(f"Generated {len(all_responses)} responses total")

    # Clean up CUDA cache after generation to reduce memory fragmentation
    from utils.memory_utils import cleanup_cuda_cache, get_model_device
    cleanup_cuda_cache(device=get_model_device(inference_model))

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

    Computes policy log-probs per microbatch to avoid OOM on large batches.

    Returns metrics dictionary.
    """
    model.train()

    # Tokenize once on CPU to avoid holding large tensors on GPU
    tokenized = tokenize_prompt_and_output(all_prompts, all_responses, tokenizer)
    input_ids_cpu = tokenized["input_ids"]
    labels_cpu = tokenized["labels"]
    response_mask_cpu = tokenized["response_mask"]

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

        # Move microbatch slice to device and compute policy log-probs
        micro_input_ids = input_ids_cpu[start_idx:end_idx].to(device)
        micro_labels = labels_cpu[start_idx:end_idx].to(device)
        micro_response_mask = response_mask_cpu[start_idx:end_idx].to(device)

        # Compute policy log-probs for this microbatch with gradients
        with torch.autocast(device_type=device.type, dtype=config.torch_dtype):
            policy_outputs = get_response_log_probs(
                model=model,
                input_ids=micro_input_ids,
                labels=micro_labels,
                return_token_entropy=False,
                requires_grad=True,
            )
        micro_policy_log_probs = policy_outputs["log_probs"]

        # Extract other microbatch tensors
        micro_advantages = advantages[start_idx:end_idx].unsqueeze(1)  # (micro_batch, 1)
        micro_old_log_probs = old_log_probs[start_idx:end_idx] if old_log_probs is not None else None

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
            all_metadata = {k: v.item() if torch.is_tensor(
                v) else v for k, v in metadata.items()}

        # Accumulate clip fraction if using grpo_clip
        if config.loss_type == "grpo_clip" and "clip_fraction" in metadata:
            total_clip_fraction += metadata["clip_fraction"]

        # Skip token entropy computation to save memory
        total_entropy += 0.0

    # Take optimizer step after accumulating gradients and capture gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=config.grad_clip)
    optimizer.step()

    avg_loss = total_loss / step_count
    avg_clip_fraction = total_clip_fraction / \
        step_count if config.loss_type == "grpo_clip" else 0.0
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
    problems: List[str],
    answers: List[str],
    config: GRPOTrainConfig,
    inference_model,
    optimizer,
    device: torch.device,
    step: int,
) -> Dict[str, Any]:
    """
    Run a single GRPO step following Algorithm 3.

    Returns metrics dictionary.
    """
    from utils.memory_utils import log_all_gpu_memory

    print(f"\n=== GRPO Step {step + 1}/{config.n_grpo_steps} ===")
    log_all_gpu_memory(f"Step {step + 1} - Start")

    # Sample rollout batch (Algorithm 3, Line 5)
    stop_strings = ["</answer>"] if hasattr(config, "use_answer_stop") and config.use_answer_stop else None
    all_prompts, all_responses, all_ground_truths = sample_rollout_batch(
        problems=problems,
        answers=answers,
        n_prompts=config.n_prompts_per_rollout_batch,
        group_size=config.group_size,
        inference_model=inference_model,
        tokenizer=tokenizer,
        config=config,
        seed=config.seed + step,
        stop_strings=stop_strings,
    )
    log_all_gpu_memory(f"Step {step + 1} - After Rollout")

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
        dtype=config.torch_dtype,
    )
    # Ensure advantages are on the same device and dtype as the training tensors
    advantages = advantages.to(device=device, dtype=config.torch_dtype)
    log_all_gpu_memory(f"Step {step + 1} - After Advantages")

    # Get old policy log probs if needed for grpo_clip
    old_log_probs = None
    if config.loss_type == "grpo_clip":
        # Post-hoc old log-probs: forward pass on frozen inference model (no output_scores during generation)
        eval_device = next(inference_model.parameters()).device
        old_log_probs, _ = compute_log_probs_for_responses(
            model=inference_model,
            prompts=all_prompts,
            responses=all_responses,
            tokenizer=tokenizer,
            device=eval_device,
            torch_dtype=config.torch_dtype,
            requires_grad=False
        )
        old_log_probs = old_log_probs.to(device=device, dtype=config.torch_dtype).detach()

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

    log_all_gpu_memory(f"Step {step + 1} - After Training")

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

    log_all_gpu_memory(f"Step {step + 1} - End")

    # Add memory stats to metrics for all GPUs
    from utils.memory_utils import log_gpu_memory_with_roles
    log_gpu_memory_with_roles(device, inference_model, metrics)

    return metrics


def copy_model_weights(source_model, target_model):
    """
    Copy weights from `source_model` to `target_model` in-place.

    This implementation avoids constructing large intermediate tensors on the
    destination device (GPU1) by performing parameter-wise device-to-device
    copies into the existing storages of `target_model`. This greatly reduces
    peak memory usage compared to calling `load_state_dict` on a full
    state_dict when models live on different GPUs.
    """
    import torch

    # Fast path: parameter-wise copy with no gradient tracking
    with torch.no_grad():
        # Build lookups for source parameters and buffers by name
        src_params = dict(source_model.named_parameters())
        src_buffers = dict(source_model.named_buffers())

        # Copy parameters
        for name, tgt_param in target_model.named_parameters():
            src_param = src_params.get(name, None)
            if src_param is None:
                continue  # Skip unexpected/missing entries safely
            # Copy directly into the existing storage on the target device
            tgt_param.detach().copy_(src_param.detach(), non_blocking=True)

        # Copy buffers (e.g., running stats)
        for name, tgt_buf in target_model.named_buffers():
            src_buf = src_buffers.get(name, None)
            if src_buf is None:
                continue
            tgt_buf.detach().copy_(src_buf.detach(), non_blocking=True)

    # Optional: help the allocator by clearing small, now-unused temps
    try:
        if torch.cuda.is_available():
            # Synchronize target device to finalize copies before cleanup
            torch.cuda.synchronize(device=next(target_model.parameters()).device)
            torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate_on_validation(
    inference_model,
    tokenizer,
    val_problems: List[str],
    val_answers: List[str],
    config: GRPOTrainConfig,
    step: int,
) -> Dict[str, float]:
    """Run evaluation on validation set using plain HF generation (greedy)."""
    from utils.memory_utils import log_all_gpu_memory, cleanup_cuda_cache, get_model_device

    log_all_gpu_memory(f"Validation Step {step + 1} - Start")

    n_val_samples = min(config.val_samples, len(val_problems))
    indices = random.sample(range(len(val_problems)), n_val_samples)

    prompts = [format_with_r1_zero_prompt(val_problems[i]) for i in indices]
    ground_truths = [val_answers[i] for i in indices]

    # Batched generation to cap KV-cache + logits memory on eval device
    batch_size = max(1, getattr(config, "val_rollout_batch_size", 64))
    print(
        f"Generating responses for {n_val_samples} validation samples (batch_size={batch_size})..."
    )
    responses: List[str] = []
    for start_idx in range(0, len(prompts), batch_size):
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_responses = greedy_generate_responses(
            inference_model=inference_model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=config.sampling_max_tokens,
        )
        responses.extend(batch_responses)
        # Trim allocator fragmentation between batches
        cleanup_cuda_cache(device=get_model_device(inference_model))

    print("Computing validation metrics...")
    metrics = compute_reward_metrics(
        responses=responses,
        ground_truths=ground_truths,
        reward_fn=lambda r, g, fast=True: r1_zero_reward_fn(r, g, fast=fast),
    )

    # Final cleanup after validation to reduce memory fragmentation
    cleanup_cuda_cache(device=get_model_device(inference_model))

    log_all_gpu_memory(f"Validation Step {step + 1} - End")
    return metrics


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
    train_problems, train_answers, _ = load_math_train()
    val_problems, val_answers, _ = load_math_validation()
    print(
        f"Loaded {len(train_problems)} training problems, {len(val_problems)} validation problems")

    # Setup models and tokenizer
    print("Setting up models...")
    model, tokenizer, device = setup_model_and_tokenizer(
        model_name=config_obj.model_name,
        train_device=config_obj.train_device,
        torch_dtype=config_obj.torch_dtype,
    )

    # Setup a separate inference model on eval device (no hot-swapping)
    print("Initializing inference model on eval device...")
    eval_device = torch.device(config_obj.eval_device)
    # Initialize inference/reference model from the uncompiled training model
    inference_model = init_inference_model_from(model, eval_device)
    for p in inference_model.parameters():
        p.requires_grad_(False)
    # Optionally compile inference model for faster forward passes
    try:
        inference_model = torch.compile(inference_model)
    except Exception:
        pass

    # Optional: compile model for faster training (compile after creating inference copy)
    try:
        model = torch.compile(model)
    except Exception:
        pass

    # Log memory after model setup
    from utils.memory_utils import log_all_gpu_memory
    log_all_gpu_memory("After Model Setup")

    # Setup optimizer
    optimizer = build_adamw(
        model.parameters(),
        lr=config_obj.learning_rate,
        weight_decay=config_obj.weight_decay,
        betas=(config_obj.adam_beta1, config_obj.adam_beta2),
        eps=config_obj.adam_eps,
        fused=config_obj.adam_fused,
    )

    # Training loop
    print(f"Starting GRPO training for {config_obj.n_grpo_steps} steps...")

    # Log initial memory state and reset peak stats
    from utils.memory_utils import log_all_gpu_memory, reset_peak_memory_stats
    reset_peak_memory_stats()
    log_all_gpu_memory("Training Start")

    try:
        for step in range(config_obj.n_grpo_steps):
            # GRPO Algorithm: First copy current policy to reference (inference) model
            copy_model_weights(model, inference_model)

            # Run GRPO step
            step_metrics = run_grpo_step(
                model=model,
                tokenizer=tokenizer,
                problems=train_problems,
                answers=train_answers,
                config=config_obj,
                inference_model=inference_model,
                optimizer=optimizer,
                device=device,
                step=step,
            )

            # Log metrics
            if wandb.run is not None:
                wandb.log(step_metrics, step=step)
            print(f"Step {step + 1} metrics: {step_metrics}")

            # Periodic validation
            if (step + 1) % config_obj.val_every_grpo_steps == 0:
                print("Running validation evaluation...")
                # Sync inference model with latest weights for evaluation
                copy_model_weights(model, inference_model)
                val_metrics = evaluate_on_validation(
                    inference_model=inference_model,
                    tokenizer=tokenizer,
                    val_problems=val_problems,
                    val_answers=val_answers,
                    config=config_obj,
                    step=step,
                )
                if wandb.run is not None:
                    wandb.log(val_metrics, step=step)
                print(f"Validation metrics: {val_metrics}")

            # Periodic checkpoint saving
            if (step + 1) % config_obj.save_every_grpo_steps == 0:
                save_path = Path(config_obj.save_dir) / \
                    f"checkpoint_step_{step + 1}"
                save_path.mkdir(parents=True, exist_ok=True)

                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    finally:
        # Cleanup
        # Final save
        if config_obj.save_at_end:
            final_save_path = Path(config_obj.save_dir) / "final_model"
            final_save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_save_path)
            tokenizer.save_pretrained(final_save_path)
            print(f"Saved final model to {final_save_path}")

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    typer.run(main)
