# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This repository contains policy gradient and reinforcement learning experiments, primarily focused on mathematical reasoning using the MATH dataset. Key components:

- `experiments/` - Main experiment implementations
  - `experiments/sft/` - Supervised fine-tuning experiments with YAML configs
  - `experiments/baseline/` - Baseline evaluation scripts
- `utils/` - Core utilities for training, evaluation, and data processing
  - `utils/training_utils.py` - SFT training loops, tokenization, and loss computation
  - `utils/vllm_utils.py` - vLLM integration for fast inference and evaluation
  - `utils/math_data.py` - MATH dataset loading and preprocessing
  - `utils/drgrpo_grader.py` - Mathematical answer grading using symbolic verification
  - `utils/model_utils.py` - Model and tokenizer setup utilities
- `MATH/` - Dataset files (JSONL format for training/validation)
- `tests/` - Comprehensive test suite with snapshot testing

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv sync

# Alternative setup with optional flash-attention
bash scripts/setup_env.sh --flash-attn

# Set up Weights & Biases
bash scripts/setup_wandb.sh <API_KEY> [ENTITY] [PROJECT]
```

### Training Commands
```bash
# SFT training with config file
uv run python experiments/sft/sft_train.py --config experiments/sft/configs/sft_correct_full.yaml

# Override specific parameters
uv run python experiments/sft/sft_train.py --config experiments/sft/configs/unique_256.yaml --unique-train-examples 128

# Baseline evaluation
uv run python experiments/baseline/baseline_eval.py --model_path .models/Qwen2.5-Math-1.5B-Instruct --output_dir evaluation_results
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_sft.py -v
uv run pytest tests/test_grpo.py -v

# Run tests with coverage
uv run pytest --cov=utils --cov=experiments
```

## Training Configuration

Training uses YAML configuration files in `experiments/sft/configs/`. Key parameters:
- `sft_data_path` - Path to JSONL training data
- `unique_train_examples` - Number of training examples ("all" for full dataset)
- `batch_size` / `gradient_accumulation_steps` - Effective batch size control
- `val_every_steps` - Frequency of vLLM validation runs
- `model_name` - Path to base model (typically in `.models/`)

## Architecture Notes

### Multi-Device Training
- Training uses HuggingFace Transformers on one device
- Validation uses vLLM on a separate device to avoid memory conflicts
- Automatic distributed cleanup prevents resource leaks

### Data Pipeline
- JSONL files contain `{"prompt": "...", "response": "..."}` entries
- Tokenization creates response masks for training only on outputs
- Mathematical grading uses symbolic verification (sympy, latex2sympy)

### Evaluation System
- Uses r1_zero prompt format for mathematical reasoning
- vLLM provides fast batch inference for validation
- Results include accuracy metrics and subject breakdowns
- Full evaluation artifacts saved to `eval_runs/` directory

## Model Dependencies

Primary model: Qwen/Qwen2.5-Math-1.5B-Instruct
- Download to `.models/Qwen2.5-Math-1.5B-Instruct` for local use
- Requires CUDA-capable hardware for training
- Flash Attention recommended for memory efficiency