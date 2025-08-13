# Quick Setup Guide for Policy Gradient Experiments

## Prerequisites
- Ubuntu VM with NVIDIA H100 GPUs
- CUDA 12.6+ drivers installed
- Internet access

## 1. Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

## 2. Clone and Setup Project
```bash
git clone <repo-url> policy-gradient-experiments
cd policy-gradient-experiments
```

## 3. Install Dependencies
```bash
# Install all dependencies including Flash Attention
uv sync
```

## 4. Download Model
```bash
mkdir -p .models
uv run python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
save_path = '.models/Qwen2.5-Math-1.5B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
model.save_pretrained(save_path)
print(f'Model saved to {save_path}')
"
```

## 5. Verify Setup
```bash
# Test Flash Attention
uv run python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"

# Test model loading
uv run python -c "
from utils.model_utils import setup_model_and_tokenizer
model, tokenizer, device = setup_model_and_tokenizer('.models/Qwen2.5-Math-1.5B-Instruct', 'cuda:0')
print(f'Model on {device}, Flash Attention: {getattr(model.config, \"_attn_implementation\", \"unknown\")}')
"
```