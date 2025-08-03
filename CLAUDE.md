# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project uses `uv` for Python package management. The main dependencies include:
- PyTorch and transformers for model training
- TRL (Transformers Reinforcement Learning) for post-training
- PEFT for LoRA fine-tuning
- HuggingFace datasets for MATH dataset loading

## Core Architecture

This is a mathematical reasoning post-training system on the MATH dataset (Hendrycks et al.). The MATH dataset contains challenging high school and undergraduate level mathematical competition problems.

### Dataset Processing
- Loads MATH dataset with step-by-step solutions
- Covers areas: Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, Precalculus, Calculus
- Problems require multi-step reasoning and formal mathematical notation

### Training Configuration
- Supports various model sizes for post-training experiments
- Uses LoRA fine-tuning for efficient parameter updates
- Post-training techniques for mathematical reasoning enhancement

## Common Commands

```bash
# Install dependencies
uv sync

# Run post-training experiments
python train_math.py

# Evaluate on MATH dataset
python eval_math.py --model <model_path>
```

## Model Output Format

The system works with various mathematical formats including:
- LaTeX mathematical notation
- Step-by-step solution reasoning
- Final numerical or algebraic answers

## Training Details

- Mixed precision training with bfloat16
- Checkpoint saving for model recovery
- WandB integration for experiment tracking
- Support for multi-GPU training

## Dataset Information

MATH dataset (Hendrycks et al.):
- 12,500 challenging competition mathematics problems
- 5 difficulty levels
- 7 subject areas
- Requires advanced mathematical reasoning capabilities