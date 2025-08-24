#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimisation) training script.
Implements Algorithm 3 from the GRPO paper.
"""

# Ensure repository root is on sys.path for absolute imports
import os
# Set PyTorch env vars before importing torch
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TORCH_LOGS', '-inductor')
from pathlib import Path
import sys
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
import warnings
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
# Filter out common but non-critical warnings that clutter logs
warnings.filterwarnings('ignore', message='.*TensorFloat32.*')
warnings.filterwarnings('ignore', message='.*flash_attention.*')
warnings.filterwarnings('ignore', category=UserWarning,
                        module='torch._inductor.*')
# Suppress LaTeX macro warnings from grader
warnings.filterwarnings('ignore', message='.*macro.*failed its substitution.*')
warnings.filterwarnings('ignore', message='.*Error in configuration.*')


# vLLM removed; we will use plain Hugging Face generation on a separate inference model/device


def setup_wandb(config: GRPOTrainConfig, config_path: Optional[str] = None):
    """Initialize wandb logging."""
    resolved_project = os.environ.get("WANDB_PROJECT", config.project)
    resolved_entity = os.environ.get("WANDB_ENTITY", config.wandb_entity)

    # Skip wandb initialization if project is None or empty
    if not resolved_project:
        print("Wandb disabled (project is None or empty)")
        return

    # Add timestamp to run name to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Hook model basename and loss type into the run name for easier identification
    try:
        model_base = Path(config.model_name).name
    except Exception:
        model_base = str(config.model_name)

    loss_tag = getattr(config, "loss_type", "")
    # Compose run name: {run_name}_{model}_{loss}_{timestamp}
    unique_run_name = f"{config.run_name}_{model_base}_{loss_tag}_{timestamp}"

    wandb.init(
        project=resolved_project,
        entity=resolved_entity,
        name=unique_run_name,
        config=config.__dict__,
        save_code=True,
    )
    # Persist the unique run name back into the config for downstream use (e.g., checkpoint paths)
    try:
        config.run_name = unique_run_name
    except Exception:
        pass

    # Log the original YAML config file as an artifact before training starts
    try:
        if config_path and os.path.exists(config_path) and wandb.run is not None:
            import hashlib
            # Compute a short content hash to help dedup/versioning
            cfg_bytes = Path(config_path).read_bytes()
            cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()
            short_sha = cfg_sha[:12]

            # Name matches existing convention: train_config (versioned by W&B)
            artifact = wandb.Artifact(
                name="train_config",
                type="config",
                metadata={
                    "run_name": unique_run_name,
                    "model_name": getattr(config, "model_name", None),
                    "loss_type": getattr(config, "loss_type", None),
                    "sha256": cfg_sha,
                },
            )
            # Preserve original filename (e.g., grpo_clip.yaml)
            artifact.add_file(str(config_path), name=Path(config_path).name)
            wandb.log_artifact(artifact, aliases=[
                               "latest", f"run:{unique_run_name}", f"sha256:{short_sha}"])
            # Also save the file into the run's file set for convenience
            try:
                wandb.save(str(config_path))
            except Exception:
                pass
    except Exception:
        pass

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
    # Sample problems and format prompts in one pass
    prompts, sampled_answers = zip(*(
        (format_with_r1_zero_prompt(problems[i]), answers[i]) for i in indices
    ))
    prompts, sampled_answers = list(prompts), list(sampled_answers)

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
    *,
    inference_model,
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
    tokenized = tokenize_prompt_and_output(
        all_prompts, all_responses, tokenizer)
    input_ids_cpu = tokenized["input_ids"]
    labels_cpu = tokenized["labels"]
    response_mask_cpu = tokenized["response_mask"]

    # Old policy log-probs will be computed per-microbatch on the eval device for grpo_clip

    # Compute train rewards for logging
    train_rewards = {"format": [], "answer": [], "total": []}
    for response, ground_truth in zip(all_responses, all_ground_truths):
        reward_dict = r1_zero_reward_fn(response, ground_truth, fast=True)
        train_rewards["format"].append(reward_dict.get("format_reward", 0.0))
        train_rewards["answer"].append(reward_dict.get("answer_reward", 0.0))
        train_rewards["total"].append(reward_dict.get("reward", 0.0))

    # Split into microbatches and train
    num_rollout_examples = len(all_prompts)
    microbatchsize_examples = config.micro_train_batch_size
    sum_scaled_loss = 0.0
    sum_true_batch_loss = 0.0  # mean per-example masked loss before GA scaling
    processed_microbatches = 0
    summarized_metadata = {}
    sum_clip_fraction = 0.0
    sum_entropy = 0.0

    # Zero gradients at start
    optimizer.zero_grad()

    # Microbatch accumulation with per-group optimizer steps
    total_microbatches = (
        num_rollout_examples + microbatchsize_examples - 1) // microbatchsize_examples
    optimizer_update_count = 0
    sum_grad_norm = 0.0

    for microbatch_index in range(total_microbatches):
        slice_start = microbatch_index * microbatchsize_examples
        slice_end = min(slice_start + microbatchsize_examples,
                        num_rollout_examples)

        # Determine target denominator for this accumulation group
        group_start_index = microbatch_index - \
            (microbatch_index % config.gradient_accumulation_steps)
        group_size_for_accum = min(
            config.gradient_accumulation_steps,
            total_microbatches - group_start_index,
        )

        # Move microbatch slice to device and compute policy log-probs
        micro_input_ids = input_ids_cpu[slice_start:slice_end].to(device)
        micro_labels = labels_cpu[slice_start:slice_end].to(device)
        micro_response_mask = response_mask_cpu[slice_start:slice_end].to(
            device)

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
        micro_advantages = advantages[slice_start:slice_end].unsqueeze(1)
        micro_old_log_probs = None
        if config.loss_type == "grpo_clip":
            eval_device = next(inference_model.parameters()).device
            ids_ev = input_ids_cpu[slice_start:slice_end].to(
                eval_device, non_blocking=True)
            lbl_ev = labels_cpu[slice_start:slice_end].to(
                eval_device, non_blocking=True)
            # Ensure dropout/etc. are disabled exactly at use-site
            _was_training_ref = inference_model.training
            inference_model.eval()
            try:
                old_scores = get_response_log_probs(
                    model=inference_model,
                    input_ids=ids_ev,
                    labels=lbl_ev,
                    return_token_entropy=False,
                    requires_grad=False,
                )
            finally:
                if _was_training_ref:
                    inference_model.train()
            micro_old_log_probs = old_scores["log_probs"].to(
                device=device, dtype=config.torch_dtype, non_blocking=True
            )
            del ids_ev, lbl_ev, old_scores
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Run microbatch train step (scale by group_total)
        loss, metadata = grpo_microbatch_train_step(
            policy_log_probs=micro_policy_log_probs,
            response_mask=micro_response_mask,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            loss_type=config.loss_type,
            raw_rewards=None,
            advantages=micro_advantages,
            old_log_probs=micro_old_log_probs,
            cliprange=config.cliprange if config.loss_type == "grpo_clip" else None,
            seq_loss_reduction=getattr(
                config, "seq_loss_reduction", "per_example_mean"),
            accumulation_denominator=int(group_size_for_accum),
        )

        sum_scaled_loss += loss.item()
        # Aggregate unscaled batch loss from metadata
        if "batch_loss" in metadata:
            try:
                sum_true_batch_loss += float(metadata["batch_loss"]) if not torch.is_tensor(
                    metadata["batch_loss"]) else float(metadata["batch_loss"].item())
            except Exception:
                pass
        processed_microbatches += 1

        if microbatch_index == 0:
            summarized: dict[str, float | int | bool] = {}
            for k, v in metadata.items():
                if torch.is_tensor(v):
                    if v.numel() == 1:
                        summarized[k] = v.item()
                else:
                    summarized[k] = v
            summarized_metadata = summarized

        if config.loss_type == "grpo_clip" and "clip_fraction" in metadata:
            sum_clip_fraction += metadata["clip_fraction"]

        sum_entropy += 0.0

        # If end of accumulation group or last microbatch, take optimizer step
        is_end_of_group = ((microbatch_index + 1) % config.gradient_accumulation_steps ==
                           0) or ((microbatch_index + 1) == total_microbatches)
        if is_end_of_group:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_update_count += 1
            sum_grad_norm += float(grad_norm.item())
    avg_loss = sum_scaled_loss / max(1, processed_microbatches)
    avg_true_batch_loss = sum_true_batch_loss / max(1, processed_microbatches)
    avg_clip_fraction = sum_clip_fraction / \
        max(1, processed_microbatches) if config.loss_type == "grpo_clip" else 0.0
    avg_entropy = sum_entropy / max(1, processed_microbatches)
    avg_grad_norm = sum_grad_norm / max(1, optimizer_update_count)

    return {
        "train_loss": avg_loss,
        "train_true_loss": avg_true_batch_loss,
        "grad_norm": avg_grad_norm,
        "token_entropy": avg_entropy,
        "clip_fraction": avg_clip_fraction,
        "train_format_reward": np.mean(train_rewards["format"]),
        "train_answer_reward": np.mean(train_rewards["answer"]),
        "train_total_reward": np.mean(train_rewards["total"]),
        "num_microbatches": processed_microbatches,
        "num_optimizer_updates": optimizer_update_count,
        **summarized_metadata
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
    stop_strings = [
        "</answer>"] if hasattr(config, "use_answer_stop") and config.use_answer_stop else None
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

    # Old policy log-probs are computed inside train_on_rollout_batch for grpo_clip

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
            inference_model=inference_model,
            config=config,
            optimizer=optimizer,
            device=device,
            all_ground_truths=all_ground_truths,
        )
        epoch_metrics.append(epoch_metrics_dict)

    log_all_gpu_memory(f"Step {step + 1} - After Training")

    # Update old policy (Algorithm 3, Line 10 - implicitly, we'll copy weights)
    # Note: For efficiency, we update old_model at the end of each step

    # Step-wise metrics only
    metrics = {
        "step": step + 1,
        "rollout_batch_size": len(all_responses),
        "n_prompts": config.n_prompts_per_rollout_batch,
        "group_size": config.group_size,
        **reward_metadata,
    }
    if epoch_metrics:
        # Merge the single epoch's metrics directly (epochs_per_rollout_batch is typically 1)
        metrics.update(epoch_metrics[-1])

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
            torch.cuda.synchronize(device=next(
                target_model.parameters()).device)
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

    # Support selecting the entire validation set via config.val_samples == "all"
    if isinstance(getattr(config, "val_samples", None), str) and str(config.val_samples).lower() == "all":
        n_val_samples = len(val_problems)
        # Deterministic full-order evaluation (no extra shuffling)
        indices = list(range(len(val_problems)))
    else:
        # Default: sample a random subset of size val_samples (bounded by dataset size)
        n_val_samples = min(int(config.val_samples), len(val_problems))
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

    # Save a small sample (first 10) of already-generated responses without extra generation
    try:
        from pathlib import Path as _Path
        from utils.training_utils import write_generations_from_samples as _write_gen

        out_dir = _Path(config.val_log_dir) / _Path(getattr(config,
                                                            "run_name", "grpo")) / f"val_step_{step + 1}"
        sample_n = min(10, len(prompts))
        if sample_n > 0:
            saved = _write_gen(
                prompts=prompts[:sample_n],
                responses=responses[:sample_n],
                ground_truths=ground_truths[:sample_n],
                reward_fn=r1_zero_reward_fn,
                out_dir=str(out_dir),
                model_name=getattr(config, "run_name", "grpo"),
                max_examples=sample_n,
            )
            if saved:
                print(f"Saved {sample_n} validation generations to {saved}")
    except Exception:
        # Non-fatal: continue even if logging fails
        pass

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
    setup_wandb(config_obj, config_path=config)

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
    # Ensure strict eval mode for reference model (no dropout/BN updates)
    inference_model.eval()
    for p in inference_model.parameters():
        p.requires_grad_(False)
    # Optionally compile inference model for faster forward passes
    try:
        inference_model = torch.compile(inference_model)
        # Preserve eval mode after compile wrapping
        inference_model.eval()
    except Exception:
        pass

    # Optional: compile model for faster training (compile after creating inference copy)
    try:
        model = torch.compile(model)
        # Ensure training mode for the trainable model after compile
        model.train()
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

    # Optional pre-training validation for baseline metrics and sanity check
    try:
        print("Running pre-training validation...")
        pre_val_metrics = evaluate_on_validation(
            inference_model=inference_model,
            tokenizer=tokenizer,
            val_problems=val_problems,
            val_answers=val_answers,
            config=config_obj,
            step=-1,
        )
        # Log onto the same keys/series as later evals; use step=-1 so it appears before step 0
        if wandb.run is not None:
            wandb.log(pre_val_metrics, step=-1)
        print(f"Pre-training validation: {pre_val_metrics}")
    except Exception as _e:
        # Non-fatal; continue with training even if pre-eval fails
        print(f"Pre-training validation skipped due to error: {_e}")
    finally:
        # Use unified cleanup utility to release allocator pressure on both devices
        try:
            from utils.memory_utils import cleanup_cuda_cache, get_model_device
            cleanup_cuda_cache(device=get_model_device(inference_model))
            cleanup_cuda_cache(device=device)
        except Exception:
            pass

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
                save_path = Path(config_obj.save_dir) / Path(config_obj.run_name) / \
                    f"checkpoint_step_{step + 1}"
                save_path.mkdir(parents=True, exist_ok=True)

                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    finally:
        # Cleanup
        # Final save
        if config_obj.save_at_end:
            final_save_path = Path(config_obj.save_dir) / \
                Path(config_obj.run_name) / "final_model"
            final_save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(final_save_path)
            tokenizer.save_pretrained(final_save_path)
            print(f"Saved final model to {final_save_path}")

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    typer.run(main)
