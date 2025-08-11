#!/usr/bin/env python3
"""
SFT training script with periodic vLLM validation and wandb logging.
Loads hyperparameters from a YAML config file.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for absolute imports like `utils.*` and `experiments.*`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import math
import os
import torch
import wandb
import argparse
import json
from pathlib import Path
from typing import Union
from experiments.sft.config_schema import SFTTrainConfig
from utils.vllm_utils import init_vllm, create_sampling_params, load_policy_into_vllm_instance, evaluate_vllm
from utils.math_data import load_math_validation, format_with_r1_zero_prompt
from utils.training_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step
)
from utils.drgrpo_grader import r1_zero_reward_fn
from utils.model_utils import setup_model_and_tokenizer
from utils.optim_sched_utils import build_adamw, build_warmup_then_scheduler


def load_sft_data(dataset_path: Union[str, Path]):
    """Load SFT data from a JSONL file (path provided via config)."""
    dataset_path = Path(dataset_path)
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} SFT examples from {dataset_path}")
    return data


def create_data_loader(data, batch_size=4):
    """Yield batches of raw prompts/responses; tokenization is done inside the train loop."""
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        batch = {
            "prompts": [ex["prompt"] for ex in chunk],
            "responses": [ex["response"] for ex in chunk],
        }
        batches.append(batch)
    return batches


def train_model(
    model,
    tokenizer,
    train_batches,
    val_prompts,
    val_answers,
    gradient_accumulation_steps: int,
    lr: float,
    val_every_steps: int,
    max_new_tokens: int,
    project: str,
    run_name: str,
    seed: int,
    grad_clip: float,
    vllm_gpu_mem_utilization: float,
    model_name_for_vllm: str | None,
    wandb_entity: str | None,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    warmup_steps: int,
    warmup_ratio: float | None,
    lr_scheduler: str,
    eval_device: str,
    val_temperature: float,
    val_top_p: float,
    val_log_dir: str,
    adam_fused: bool | None = None,
    eval_before_training: bool = True,
    num_epochs: int = 1,
):
    """Train with grad accumulation, grad clipping, periodic vLLM validation, and wandb logging."""
    # Resolve W&B settings and auto-login from environment if provided
    resolved_project = os.environ.get("WANDB_PROJECT", project)
    resolved_entity = wandb_entity or os.environ.get("WANDB_ENTITY")
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception:
            pass
    wandb.init(project=resolved_project, name=run_name, entity=resolved_entity)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Disable HF generation logging during training batches for now
    enable_train_generation_logging = False

    optimizer = build_adamw(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        fused=adam_fused,
    )
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad()

    # Validate eval device but defer vLLM engine creation to validation time
    if eval_device == "cpu":
        raise RuntimeError(
            "eval_device=cpu is not supported by vLLM. Please set a CUDA device, e.g., cuda:1")
    if eval_device == device.type or (isinstance(device, torch.device) and eval_device == str(device)):
        # Avoid trying to share the same device as training for vLLM
        raise RuntimeError(
            "vLLM eval on the same device as training is not supported. Use a different eval_device.")

    # Compute total optimizer steps for scheduling across all epochs
    steps_per_epoch = math.ceil(len(train_batches) / gradient_accumulation_steps)
    total_update_steps = steps_per_epoch * max(1, int(num_epochs))
    if warmup_ratio is not None and warmup_steps == 0:
        warmup_steps = int(warmup_ratio * total_update_steps)

    scheduler = build_warmup_then_scheduler(
        optimizer=optimizer,
        total_update_steps=total_update_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        after=lr_scheduler,
    )

    # Logging helpers
    def _log_train(loss_tensor: torch.Tensor) -> None:
        wandb.log({
            "train/loss": float(loss_tensor.item() * gradient_accumulation_steps),
            "train/lr": float(optimizer.param_groups[0]["lr"]),
            "train_step": train_step,
        })

    def _log_eval(result: dict) -> None:
        payload = {
            "eval/avg_response_length": result["aggregates"].get("avg_response_length", 0.0),
            "eval/avg_response_token_entropy": result["aggregates"].get("avg_response_token_entropy", 0.0),
            "eval_step": eval_step,
            "eval/global_train_step": train_step,
            "eval/epoch": (train_step / steps_per_epoch) if steps_per_epoch > 0 else 0.0,
            "eval/num_samples": len(val_prompts),
        }
        # If available (from vLLM path), include accuracy scalars
        if "answer_accuracy" in result.get("aggregates", {}):
            payload.update({
                "eval/answer_accuracy": result["aggregates"]["answer_accuracy"],
                "eval/format_accuracy": result["aggregates"]["format_accuracy"],
                "eval/overall_accuracy": result["aggregates"]["overall_accuracy"],
            })
        wandb.log(payload)

    def _finalize_train_step(batch: dict, loss: torch.Tensor) -> None:
        """Complete a training step: clip grads, optimizer step, logging, validation check"""
        nonlocal train_step, eval_step

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_step += 1
        _log_train(loss)

        # Optional: training batch generation+metrics (disabled by default)
        if enable_train_generation_logging:
            train_result = log_generations(
                model=model,
                tokenizer=tokenizer,
                prompts=batch["prompts"],
                ground_truths=batch["responses"],
                reward_fn=r1_zero_reward_fn,
            )
            train_aggregates = train_result.get("aggregates", {})
            wandb.log({
                "train/avg_response_length": train_aggregates.get("avg_response_length", 0.0),
                "train/avg_response_token_entropy": train_aggregates.get("avg_response_token_entropy", 0.0),
                "train/answer_accuracy": train_aggregates.get("answer_accuracy", 0.0),
                "train/format_accuracy": train_aggregates.get("format_accuracy", 0.0),
                "train/overall_accuracy": train_aggregates.get("overall_accuracy", 0.0),
                "train_step": train_step,
            })

        # Periodic validation check
        if train_step % val_every_steps == 0:
            result = _run_validation()
            eval_step += 1
            _log_eval(result)

    # Helper to run validation via vLLM; returns result dict
    def _run_validation() -> dict:
        # vLLM engines are not designed for live weight mutation. Save the
        # current HF model to disk and initialize a fresh vLLM instance from it.
        tmp_ckpt_dir = Path(val_log_dir) / "_tmp_vllm_ckpt"
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Persist model and tokenizer
        tokenizer.save_pretrained(tmp_ckpt_dir)
        model.save_pretrained(tmp_ckpt_dir)

        # Create structured output directory for this validation step
        eval_output_dir = Path("eval_runs") / run_name / f"step_{train_step}"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Spin up a vLLM engine from the saved checkpoint on the eval device
        llm_eval = init_vllm(
            model_id=str(tmp_ckpt_dir),
            device=eval_device,
            seed=seed,
            gpu_memory_utilization=vllm_gpu_mem_utilization,
        )

        sampling = create_sampling_params(
            temperature=val_temperature, top_p=val_top_p, max_tokens=max_new_tokens
        )

        try:
            # Use evaluate_vllm directly - it handles generation, evaluation, and storage
            result = evaluate_vllm(
                vllm_model=llm_eval,
                reward_fn=r1_zero_reward_fn,
                prompts=val_prompts,
                ground_truths=val_answers,
                eval_sampling_params=sampling,
                output_dir=str(eval_output_dir),
                model_name=f"{run_name}_step_{train_step}",
                problem_metadata=None,
                save_full_responses=True,
                write_results=True,  # Store results to disk
            )
            # Return aggregates for wandb logging
            return {"aggregates": result.get("metadata", {}).get("overall_metrics", {})}
        finally:
            # Best-effort cleanup to free memory between validations
            try:
                del llm_eval
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Clean up temporary checkpoint
            try:
                import shutil
                shutil.rmtree(tmp_ckpt_dir)
            except Exception:
                pass

    train_step = 0
    eval_step = 0

    # Optional pre-training evaluation
    if eval_before_training:
        result = _run_validation()
        eval_step += 1
        _log_eval(result)
    for epoch in range(int(max(1, num_epochs))):
        for idx, batch in enumerate(train_batches):
            # Tokenize with response mask
            tokenized = tokenize_prompt_and_output(
                prompt_strs=batch["prompts"],
                output_strs=batch["responses"],
                tokenizer=tokenizer,
            )
            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)
            response_mask = tokenized["response_mask"].to(device)
            attention_mask = tokenized.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Get per-token log-probabilities
            scores = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
                requires_grad=True,
                attention_mask=attention_mask,
            )
            policy_log_probs = scores["log_probs"]

            # Compute masked SFT loss, handle grad accumulation, and backprop
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
            )

            # Clip and step on accumulation boundary
            if (idx + 1) % gradient_accumulation_steps == 0:
                _finalize_train_step(batch, loss)

        # Finalize leftover gradients if the number of microbatches is not
        # divisible by gradient_accumulation_steps. This ensures the last
        # partial accumulation is not dropped.
        remainder = len(train_batches) % gradient_accumulation_steps
        if remainder != 0:
            _finalize_train_step(batch, loss)

    # Always run a final validation and record summary metrics
    try:
        result = _run_validation()
        eval_step += 1
        _log_eval(result)
        aggregates = result.get("aggregates", {})
        if hasattr(wandb, "run") and wandb.run is not None:
            wandb.run.summary["final/answer_accuracy"] = float(aggregates.get("answer_accuracy", 0.0))
            wandb.run.summary["final/format_accuracy"] = float(aggregates.get("format_accuracy", 0.0))
            wandb.run.summary["final/overall_accuracy"] = float(aggregates.get("overall_accuracy", 0.0))
    except Exception:
        pass


def save_model(model, tokenizer, output_dir: str = "./sft_output"):
    """Save trained model and tokenizer"""
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    print(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SFT training with periodic vLLM validation")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional path to YAML config. If omitted, dataclass defaults are used.")
    parser.add_argument("--unique-train-examples", type=str, default=None,
                        help='Override unique_train_examples (int or "all")')
    args = parser.parse_args()

    # Load config strictly from YAML if provided; otherwise use dataclass defaults
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = SFTTrainConfig.from_path(config_path)
    else:
        cfg = SFTTrainConfig()
    repo_root = Path(__file__).resolve().parents[2]
    cfg = cfg.resolve(repo_root=repo_root)

    # Load SFT train data (path from config or default to repo_root/MATH/sft.jsonl)
    sft_data = load_sft_data(cfg.sft_data_path)

    # Setup model
    model, tokenizer, device = setup_model_and_tokenizer(
        model_name=cfg.model_name, train_device=cfg.train_device)

    # Optionally restrict number of unique training examples
    # Apply CLI override for unique_train_examples if provided
    unique_override = args.unique_train_examples
    if unique_override is not None:
        if unique_override.lower() == "all":
            cfg.unique_train_examples = "all"
        else:
            cfg.unique_train_examples = int(unique_override)
    unique_train = cfg.unique_train_examples
    if isinstance(unique_train, int):
        sft_data = sft_data[:unique_train]

    # Create batches
    batch_size = int(cfg.batch_size)
    batches = create_data_loader(sft_data, batch_size)

    # Validation on canonical MATH validation set (r1_zero formatted)
    problems, answers, _ = load_math_validation()
    if isinstance(cfg.val_samples, int):
        selected_problems = problems[: int(cfg.val_samples)]
        selected_answers = answers[: int(cfg.val_samples)]
    else:
        selected_problems = problems
        selected_answers = answers
    val_prompts = [format_with_r1_zero_prompt(p) for p in selected_problems]
    val_answers = selected_answers

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_batches=batches,
        val_prompts=val_prompts,
        val_answers=val_answers,
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        lr=float(cfg.lr),
        val_every_steps=int(cfg.val_every_steps),
        max_new_tokens=int(cfg.max_new_tokens),
        project=str(cfg.project),
        run_name=str(cfg.run_name),
        seed=int(cfg.seed),
        grad_clip=float(cfg.grad_clip),
        vllm_gpu_mem_utilization=float(cfg.vllm_gpu_mem_utilization),
        model_name_for_vllm=str(cfg.model_name_for_vllm),
        wandb_entity=cfg.wandb_entity,
        weight_decay=float(cfg.weight_decay),
        adam_beta1=float(cfg.adam_beta1),
        adam_beta2=float(cfg.adam_beta2),
        adam_eps=float(cfg.adam_eps),
        warmup_steps=int(cfg.warmup_steps),
        warmup_ratio=float(
            cfg.warmup_ratio) if cfg.warmup_ratio is not None else None,
        lr_scheduler=str(cfg.lr_scheduler),
        eval_device=str(cfg.eval_device),
        val_temperature=float(cfg.val_temperature),
        val_top_p=float(cfg.val_top_p),
        val_log_dir=str(cfg.val_log_dir),
        eval_before_training=bool(cfg.eval_before_training),
        num_epochs=int(getattr(cfg, "num_epochs", 1)),
    )

    # Save (optional)
    if bool(getattr(cfg, "save_at_end", False)):
        save_dir = str(cfg.save_dir)
        save_model(model, tokenizer, output_dir=save_dir)


if __name__ == "__main__":
    main()
