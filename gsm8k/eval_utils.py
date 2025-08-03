#!/usr/bin/env python3
"""
General evaluation utilities for model inference.
Reusable across different datasets and evaluation tasks.
"""

import torch
from transformers import AutoModelForCausalLM
from vllm import LLM


def prepare_model_for_eval(model, tokenizer_name=None):
    """Prepare already loaded model for evaluation."""
    print("Preparing model for evaluation...")

    # Optimize for inference
    model.eval()
    model.requires_grad_(False)

    # Compile for speed (disabled for debugging)
    print("Compiling model...")
    model = torch.compile(model)

    return model


def load_model_smart(model_path: str, use_vllm: bool = True):
    """Load model - either HF model name or full model checkpoint."""
    print(f"Loading model: {model_path}")

    if use_vllm:
        print("Using vLLM for inference...")
        # Load with vLLM for optimized inference
        model = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            trust_remote_code=True
        )
        return model, model_path
    else:
        # Simple loading - works for both HF model names and full checkpoints
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )

        # Prepare for evaluation
        model = prepare_model_for_eval(model)

        return model, model_path


def collate_fn(batch, tokenizer):
    """Collate function for DataLoader with left padding."""
    prompt_lengths = [len(x["prompt_ids"]) for x in batch]
    max_length = max(prompt_lengths)

    input_ids = torch.full((len(batch), max_length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)

    for i, example in enumerate(batch):
        prompt_len = len(example["prompt_ids"])
        # Left padding
        input_ids[i, -prompt_len:] = torch.tensor(example["prompt_ids"])
        attention_mask[i, -prompt_len:] = True

    return {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "questions": [x["question"] for x in batch],
        "targets": [x["target"] for x in batch]
    }