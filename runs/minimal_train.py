#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path
import yaml

import verifiers as vf

"""
Clean GRPO training script with YAML configuration.

Usage:
- CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 runs/minimal_train.py
- CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 runs/minimal_train.py --config configs/custom.yaml
"""

# Ensure local environments/ is importable
PROJECT_ROOT = Path(__file__).parents[1]
ENVS_DIR = PROJECT_ROOT / "environments"
sys.path.insert(0, str(ENVS_DIR))


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_config_to_training_args(config: dict, training_args, run_name: str):
    """Apply configuration to training arguments."""
    # Model and environment
    training_args.run_name = run_name

    # Training hyperparameters
    training_args.max_steps = config['training']['max_steps']
    training_args.learning_rate = config['training']['learning_rate']
    training_args.lr_scheduler_type = config['training']['lr_scheduler_type']
    training_args.warmup_steps = config['training']['warmup_steps']
    training_args.max_grad_norm = config['training']['max_grad_norm']
    training_args.num_iterations = config['training']['num_iterations']

    # Batch configuration
    training_args.per_device_train_batch_size = config['batch']['per_device_train_batch_size']
    training_args.num_generations = config['batch']['num_generations']
    training_args.gradient_accumulation_steps = config['batch']['gradient_accumulation_steps']

    # Generation settings
    training_args.max_seq_len = config['generation']['max_seq_len']
    training_args.max_tokens = config['generation']['max_tokens']
    training_args.temperature = config['generation']['temperature']
    training_args.top_p = config['generation']['top_p']
    training_args.top_k = config['generation']['top_k']

    # GRPO parameters
    training_args.beta = config['grpo']['beta']
    training_args.sync_ref_model = config['grpo']['sync_ref_model']
    training_args.ref_model_sync_steps = config['grpo']['ref_model_sync_steps']
    training_args.ref_model_mixup_alpha = config['grpo']['ref_model_mixup_alpha']
    training_args.loss_type = config['grpo']['loss_type']
    training_args.epsilon = config['grpo']['epsilon']
    training_args.delta = config['grpo']['delta']

    # Async generation
    training_args.num_batches_ahead = config['async_generation']['num_batches_ahead']
    training_args.async_generation_timeout = config['async_generation']['timeout']
    training_args.max_concurrent = config['async_generation']['max_concurrent']

    # Evaluation
    training_args.eval_strategy = config['eval']['strategy']
    training_args.eval_steps = config['eval']['steps']
    training_args.per_device_eval_batch_size = config['eval']['per_device_batch_size']

    # Saving
    training_args.save_strategy = config['save']['strategy']
    training_args.save_steps = config['save']['steps']

    # Logging
    training_args.logging_steps = config['logging']['steps']
    training_args.log_completions = config['logging']['log_completions']
    training_args.report_to = config['logging']['report_to']

    # Gradient checkpointing
    if 'gradient_checkpointing' in config:
        training_args.gradient_checkpointing = config['gradient_checkpointing']
    if 'gradient_checkpointing_kwargs' in config and config['gradient_checkpointing_kwargs'] is not None:
        training_args.gradient_checkpointing_kwargs = config['gradient_checkpointing_kwargs']


def setup_environment_variables(config: dict):
    """Setup environment variables for W&B and API."""
    # W&B configuration
    if config['logging']['wandb_project']:
        os.environ["WANDB_PROJECT"] = config['logging']['wandb_project']
    if config['logging']['wandb_entity']:
        os.environ["WANDB_ENTITY"] = config['logging']['wandb_entity']
    if config['logging']['wandb_name']:
        os.environ["WANDB_NAME"] = config['logging']['wandb_name']

    # vLLM API configuration
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8000/v1")
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training with YAML configuration")
    p.add_argument("--config", default="configs/grpo_default.yaml", help="Path to YAML config file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    # Setup environment variables
    setup_environment_variables(config)

    # Load environment and model
    vf_env = vf.load_environment(env_id=config['model']['env_id'])
    model_name = config['model']['name']
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    # Generate run name
    run_name = (config['logging']['wandb_name'] or
                f"{config['model']['env_id']}-grpo_{model_name.split('/')[-1].lower()}")

    # Setup training arguments
    training_args = vf.grpo_defaults(run_name=run_name)
    apply_config_to_training_args(config, training_args, run_name)

    # Create trainer and start training
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
