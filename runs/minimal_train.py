#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

import verifiers as vf

"""
Minimal GRPO training script (aligned with Verifiers example), adapted for MBPP.

Quickstart (examples):
- Install env: uv run vf-install mbpp_baseline -p environments
- Inference server (optional): CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model Qwen/Qwen2.5-Coder-1.5B
- Train (local HF accelerate): CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml runs/minimal_train.py --model Qwen/Qwen2.5-Coder-1.5B

Hyperparameter choices follow Verifiers training guide [training docs].
"""

# Ensure local environments/ is importable (no install required)
PROJECT_ROOT = Path(__file__).parents[1]
ENVS_DIR = PROJECT_ROOT / "environments"
sys.path.insert(0, str(ENVS_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal GRPO training for MBPP using Verifiers")
    p.add_argument("--model", default=".models/Qwen2.5-Coder-1.5B", help="HF model id or local path")
    p.add_argument("--env-id", default="mbpp_baseline", help="Verifiers environment id")
    p.add_argument("--max-steps", type=int, default=500)
    # Batch config per guide: 8 x 16 gens x 4 accum
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    # Sequence/generation lengths (consider model context window)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=1024)
    # Eval/save cadence
    p.add_argument("--eval-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=250)
    # Logging
    p.add_argument("--wandb-project", type=str, default="mbpp")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B workspace/entity")
    p.add_argument("--wandb-name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Route logging and inference via environment variables that Verifiers/transformers respect
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_name:
        os.environ["WANDB_NAME"] = args.wandb_name

    # Point OpenAI-compatible client to local vLLM by default
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))

    # Environment (load from local package path)
    vf_env = vf.load_environment(env_id=args.env_id)

    # Model and tokenizer (HF/transformers compatible)
    model_name = args.model
    run_name = (args.wandb_name or f"{args.env_id}-grpo_" + model_name.split("/")[-1].lower())

    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    # Training args (start from GRPO defaults and override per guide)
    training_args = vf.grpo_defaults(run_name=run_name)
    # Batch
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.num_generations = args.num_generations
    training_args.gradient_accumulation_steps = args.grad_accum
    # Lengths & sampling
    training_args.max_tokens = args.max_tokens
    training_args.max_seq_len = args.max_seq_len
    training_args.temperature = 1.0
    training_args.top_p = 1.0
    training_args.top_k = None
    # Schedule
    training_args.learning_rate = 1e-6
    training_args.lr_scheduler_type = "constant_with_warmup"
    training_args.warmup_steps = 10
    training_args.max_steps = args.max_steps
    training_args.num_iterations = 1
    training_args.max_grad_norm = 0.01
    # GRPO / KL
    training_args.beta = 0.001
    training_args.sync_ref_model = True
    training_args.ref_model_sync_steps = 100
    training_args.ref_model_mixup_alpha = 0.5
    training_args.loss_type = "dr_grpo"
    training_args.epsilon = 0.2
    training_args.delta = None
    # Async generation (vLLM endpoint via env OPENAI_BASE_URL)
    training_args.num_batches_ahead = 1
    training_args.async_generation_timeout = 300.0
    training_args.max_concurrent = 1024

    # Eval/save
    training_args.eval_strategy = "steps"
    training_args.eval_steps = args.eval_steps
    training_args.per_device_eval_batch_size = 32
    training_args.save_strategy = "steps"
    training_args.save_steps = args.save_steps

    # Logging configuration (W&B enabled; routed via env vars)
    training_args.logging_steps = 1
    training_args.log_completions = True
    training_args.report_to = "wandb"

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        peft_config=vf.lora_defaults(),
    )

    trainer.train()


if __name__ == "__main__":
    main()
