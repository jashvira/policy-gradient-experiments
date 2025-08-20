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
    assemble_old_log_probs,
)
from utils.optim_sched_utils import build_adamw, build_warmup_then_scheduler
from utils.model_utils import setup_model_and_tokenizer
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.training_utils import tokenize_prompt_and_output, get_response_log_probs, compute_entropy
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
os.environ.setdefault('TORCH_LOGS', '')  # Disable verbose torch logs
os.environ.setdefault('PYTORCH_DISABLE_STACK_TRACE', '1')  # Reduce stack trace verbosity


# Ensure repository root is on sys.path for absolute imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Suppress massive tensor dumps in error messages to keep logs clean
def suppress_large_tensor_repr():
    """Prevent massive tensor dumps in error messages to keep logs clean."""
    # Store originals
    original_tensor_repr = torch.Tensor.__repr__
    original_tensor_str = torch.Tensor.__str__
    
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
    
    # Override both repr and str methods
    torch.Tensor.__repr__ = clean_tensor_repr
    torch.Tensor.__str__ = clean_tensor_str
    
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


# vLLM removed; we will use plain Hugging Face generation on a separate inference model/device


def setup_wandb(config: GRPOTrainConfig):
    """Initialize wandb logging."""
    resolved_project = os.environ.get("WANDB_PROJECT", config.project)
    resolved_entity = os.environ.get("WANDB_ENTITY", config.wandb_entity)

    # Skip wandb initialization if project is None or empty
    if not resolved_project:
        print("Wandb disabled (project is None or empty)")
        return

    wandb.init(
        project=resolved_project,
        entity=resolved_entity,
        name=config.run_name,
        config=config.__dict__,
        save_code=True,
    )
    print(
        f"Initialized wandb: project={resolved_project}, entity={resolved_entity}")


def sample_rollout_batch(
    problems: List[str],
    answers: List[str],
    n_prompts: int,
    group_size: int,
    inference_model,
    tokenizer,
    config: GRPOTrainConfig,
    seed: int = None,
    use_output_scores: bool = False,
    stop_strings: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sample a rollout batch: n_prompts questions, each with group_size responses.

    Returns:
        Tuple of (all_prompts, all_responses, all_ground_truths, old_logprobs_matrix, gen_lens)
        where the first three lists have length n_prompts * group_size, and the last two
        tensors are optional (only present when use_output_scores=True).
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

    # Generate grouped responses via utility
    responses_out = generate_grouped_responses(
        inference_model=inference_model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=group_size,
        max_new_tokens=config.sampling_max_tokens,
        min_new_tokens=getattr(config, "sampling_min_tokens", 0),
        sampling_temperature=config.sampling_temperature,
        sampling_top_p=config.sampling_top_p,
        stop_strings=stop_strings,
        return_scores=use_output_scores,
    )
    old_logprobs_matrix: Optional[torch.Tensor] = None
    gen_lens: Optional[torch.Tensor] = None
    if use_output_scores:
        responses, old_logprobs_matrix, gen_lens = responses_out
    else:
        responses = responses_out

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

    return all_prompts, all_responses, all_ground_truths, old_logprobs_matrix, gen_lens


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
    tokenized = tokenize_prompt_and_output(
        all_prompts, all_responses, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)

    # Get current policy log probs under autocast (no token entropy for memory safety)
    with torch.autocast(device_type=device.type, dtype=config.torch_dtype):
        policy_outputs = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
        )
    policy_log_probs = policy_outputs["log_probs"]

    # Skip computing token entropy for now to avoid BxSxV allocations
    token_entropy = torch.zeros_like(policy_log_probs)

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
        micro_advantages = advantages[start_idx:end_idx].unsqueeze(
            1)  # (micro_batch, 1)
        micro_old_log_probs = old_log_probs[start_idx:
                                            end_idx] if old_log_probs is not None else None
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
            all_metadata = {k: v.item() if torch.is_tensor(
                v) else v for k, v in metadata.items()}

        # Accumulate clip fraction if using grpo_clip
        if config.loss_type == "grpo_clip" and "clip_fraction" in metadata:
            total_clip_fraction += metadata["clip_fraction"]

        # Accumulate token entropy
        masked_entropy = micro_token_entropy * micro_response_mask.float()
        avg_entropy = masked_entropy.sum() / micro_response_mask.sum().clamp_min(1)
        total_entropy += avg_entropy.item()

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
    use_output_scores = (config.loss_type == "grpo_clip")
    stop_strings = ["</answer>"] if hasattr(config, "use_answer_stop") and config.use_answer_stop else None
    all_prompts, all_responses, all_ground_truths, old_logprobs_matrix, gen_lens = sample_rollout_batch(
        problems=problems,
        answers=answers,
        n_prompts=config.n_prompts_per_rollout_batch,
        group_size=config.group_size,
        inference_model=inference_model,
        tokenizer=tokenizer,
        config=config,
        seed=config.seed + step,
        use_output_scores=use_output_scores,
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
    if config.loss_type == "grpo_clip" and old_logprobs_matrix is not None:
        # Build old_log_probs tensor from vectorized logprobs_matrix aligned to response spans
        tokenized = tokenize_prompt_and_output(
            all_prompts, all_responses, tokenizer)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        old_lp = assemble_old_log_probs(
            logprobs_matrix=old_logprobs_matrix,
            gen_lens=gen_lens,
            response_mask=response_mask,
            labels=labels,
            dtype=config.torch_dtype,
        )
        old_log_probs = old_lp.to(device=device, dtype=config.torch_dtype).detach()

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
    return metrics


def copy_model_weights(source_model, target_model):
    """Copy weights from source_model to target_model."""
    target_model.load_state_dict(source_model.state_dict())


def evaluate_on_validation(
    inference_model,
    tokenizer,
    val_problems: List[str],
    val_answers: List[str],
    config: GRPOTrainConfig,
    step: int,
) -> Dict[str, float]:
    """Run evaluation on validation set using plain HF generation (greedy)."""
    from utils.memory_utils import log_all_gpu_memory
    
    print("Running validation evaluation...")
    log_all_gpu_memory(f"Validation Step {step + 1} - Start")

    n_val_samples = min(config.val_samples, len(val_problems))
    indices = random.sample(range(len(val_problems)), n_val_samples)

    prompts = [format_with_r1_zero_prompt(val_problems[i]) for i in indices]
    ground_truths = [val_answers[i] for i in indices]

    responses = greedy_generate_responses(
        inference_model=inference_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=config.sampling_max_tokens,
    )

    metrics = compute_reward_metrics(
        responses=responses,
        ground_truths=ground_truths,
        reward_fn=lambda r, g, fast=True: r1_zero_reward_fn(r, g, fast=fast),
    )

    # Clean up CUDA cache after validation to reduce memory fragmentation
    from utils.memory_utils import cleanup_cuda_cache, get_model_device
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
