#!/usr/bin/env python3
"""
SFT training script with periodic vLLM validation and wandb logging.
Loads hyperparameters from a JSON config file.
"""

import json
import math
import torch
import wandb
import argparse
import json
from pathlib import Path
from typing import Union
from experiments.sft.config_schema import SFTTrainConfig
from utils.vllm_utils import init_vllm, create_sampling_params, load_policy_into_vllm_instance
from utils.training_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    log_generations
)
from experiments.sft.sft_utils import sft_microbatch_train_step
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
    adam_fused: bool | None,
    warmup_steps: int,
    warmup_ratio: float | None,
    lr_scheduler: str,
    eval_device: str,
    val_temperature: float,
    val_top_p: float,
    val_log_dir: str,
):
    """Train with grad accumulation, grad clipping, periodic vLLM validation, and wandb logging."""
    wandb.init(project=project, name=run_name, entity=wandb_entity)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

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

    # Setup vLLM on configured eval device (no automatic CPU fallback)
    llm = init_vllm(model_id=(model_name_for_vllm or run_name), device=eval_device, seed=seed, gpu_memory_utilization=vllm_gpu_mem_utilization)

    # Compute total optimizer steps for scheduling
    total_update_steps = math.ceil(len(train_batches) / gradient_accumulation_steps)
    if warmup_ratio is not None and warmup_steps == 0:
        warmup_steps = int(warmup_ratio * total_update_steps)

    scheduler = build_warmup_then_scheduler(
        optimizer=optimizer,
        total_update_steps=total_update_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        after=lr_scheduler,
    )

    train_step = 0
    eval_step = 0
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

        # Get per-token log-probabilities
        scores = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_token_entropy=False,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_step += 1
            wandb.log({
                "train/loss": float(loss.item() * gradient_accumulation_steps),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train_step": train_step,
            })

            # Periodic validation using vLLM
            if train_step % val_every_steps == 0:
                # Load policy weights into vLLM inference instance
                load_policy_into_vllm_instance(model, llm)
                sampling = create_sampling_params(temperature=val_temperature, top_p=val_top_p, max_tokens=max_new_tokens)
                result = log_generations(
                    model=model,  # used for entropy/log-probs
                    tokenizer=tokenizer,
                    prompts=val_prompts,
                    ground_truths=val_answers,
                    reward_fn=r1_zero_reward_fn,
                    use_vllm=True,
                    vllm_model=llm,
                    vllm_sampling_params=sampling,
                    output_dir=str(val_log_dir),
                    model_name=run_name,
                )
                eval_step += 1
                wandb.log({
                    "eval/avg_response_length": result["aggregates"]["avg_response_length"],
                    "eval/avg_response_token_entropy": result["aggregates"]["avg_response_token_entropy"],
                    "eval_step": eval_step,
                })


def save_model(model, tokenizer, output_dir: str = "./sft_output"):
    """Save trained model and tokenizer"""
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    print(f"Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SFT training with periodic vLLM validation")
    default_cfg_dir = (Path(__file__).parent / "configs").resolve()
    parser.add_argument("--config", type=str, default=None, help="Optional: path to JSON config file (overrides dir/name)")
    parser.add_argument("--config-dir", type=str, default=str(default_cfg_dir), help="Directory containing config JSON files")
    parser.add_argument("--config-name", type=str, default="default", help="Config filename without .json inside config-dir")
    parser.add_argument("--unique-train-examples", type=str, default=None, help='Override unique_train_examples (int or "all")')
    args = parser.parse_args()

    # Load config (file overrides dir/name) into dataclass
    if args.config:
        config_path = Path(args.config)
    else:
        # Prefer YAML if present, else JSON
        yaml_path = Path(args.config_dir) / f"{args.config_name}.yaml"
        yml_path = Path(args.config_dir) / f"{args.config_name}.yml"
        json_path = Path(args.config_dir) / f"{args.config_name}.json"
        if yaml_path.exists():
            config_path = yaml_path
        elif yml_path.exists():
            config_path = yml_path
        else:
            config_path = json_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = SFTTrainConfig.from_path(config_path)
    repo_root = Path(__file__).resolve().parents[2]
    cfg = cfg.resolve(repo_root=repo_root)

    # Load data (path from config or default to repo_root/MATH/sft.jsonl)
    sft_data = load_sft_data(cfg.sft_data_path)

    # Setup model
    model, tokenizer, device = setup_model_and_tokenizer(model_name=cfg.model_name, train_device=cfg.train_device)

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

    val_samples = int(cfg.val_samples)
    val_prompts, val_answers = zip(*((ex["prompt"], ex["response"]) for ex in sft_data[:val_samples]))
    val_prompts, val_answers = list(val_prompts), list(val_answers)

    # Train with periodic eval and wandb
    required_keys = [
        "gradient_accumulation_steps",
        "lr",
        "val_every_steps",
        "max_new_tokens",
        "project",
        "run_name",
        "seed",
        "grad_clip",
        "vllm_gpu_mem_utilization",
        "model_name_for_vllm",
    ]
    missing_keys = [k for k in required_keys if not hasattr(cfg, k)]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

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
        warmup_ratio=float(cfg.warmup_ratio) if cfg.warmup_ratio is not None else None,
        lr_scheduler=str(cfg.lr_scheduler),
        eval_device=str(cfg.eval_device),
        val_temperature=float(cfg.val_temperature),
        val_top_p=float(cfg.val_top_p),
        val_log_dir=str(cfg.val_log_dir),
    )

    # Save
    save_dir = str(cfg.save_dir)
    save_model(model, tokenizer, output_dir=save_dir)


if __name__ == "__main__":
    main()