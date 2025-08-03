# GSM8K Mathematical Reasoning with GRPO

GRPO training for mathematical reasoning adapted from Will Brown's research. I've added evaluation tools with vLLM support and batch processing.

## Contents

- `grpo_will.py` - GRPO training (Will Brown's work)
- `utils.py` - XML parsing utilities (Will Brown's work)  
- `eval_gsm8k.py` - Fast evaluation with vLLM/batch inference
- `eval_utils.py` - Model loading and collation utilities

## Evaluation Features

- vLLM backend for accelerated inference
- Batch processing with configurable sizes
- Organized result storage with timestamps
- WandB integration and artifact logging
- Support for both HF models and local checkpoints

## Usage

```bash
python grpo_will.py      # Training
python eval_gsm8k.py --model Qwen/Qwen2.5-0.5B-Instruct --batch_size 512  # Evaluation
```

## Attribution

Training code adapted from Will Brown: https://github.com/willccbb/verifiers
