# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project uses `uv` for Python package management. The main dependencies include:
- PyTorch and transformers for model training
- TRL (Transformers Reinforcement Learning) for GRPO training
- PEFT for LoRA fine-tuning
- HuggingFace datasets for GSM8K dataset loading

## Core Architecture

This is a mathematical reasoning training system using GRPO (Group Relative Policy Optimization) on the GSM8K dataset. The main components are:

### Dataset Processing (`willgist.py:58-73`)
- Loads OpenAI GSM8K dataset with chain-of-thought reasoning
- Formats prompts with XML-structured reasoning format: `<reasoning>...</reasoning><answer>...</answer>`
- Extracts numerical answers from `#### N` format in original dataset

### Reward Functions (`willgist.py:78-122`)
The system uses multiple granular reward functions:
- `correctness_reward_func`: 2.0 for correct answers, 0.0 otherwise
- `int_reward_func`: 0.5 reward for integer answers
- `strict_format_reward_func`: 0.5 for exact XML format compliance
- `soft_format_reward_func`: 0.5 for lenient XML format matching
- `xmlcount_reward_func`: Granular rewards (0.125 each) for proper XML tag usage

### Training Configuration (`willgist.py:133-185`)
- Supports both Llama-3.2-1B-Instruct and Qwen2.5-1.5B-Instruct models
- Uses LoRA with r=16, alpha=64 targeting attention and MLP layers
- GRPO training with 16 generations per batch, gradient accumulation of 4
- Learning rate: 5e-6 with cosine scheduler and 10% warmup

## Common Commands

```bash
# Install dependencies
uv sync

# Run training (requires CUDA GPU)
cd gsm8k
python willgist.py

# The script will automatically:
# - Load and format the GSM8K dataset
# - Initialize the model with LoRA adapters
# - Train using GRPO with multiple reward functions
# - Log to Weights & Biases
```

## Model Output Format

The system expects and rewards responses in this XML format:
```
<reasoning>
Step-by-step mathematical reasoning here
</reasoning>
<answer>
Final numerical answer
</answer>
```

## Training Details

- Uses flash attention 2 for memory efficiency
- Mixed precision training with bfloat16
- Multi-GPU training supported (PEFT disabled for multi-GPU)
- Saves checkpoints every 100 steps to `outputs/` directory
- Wandb integration for experiment tracking