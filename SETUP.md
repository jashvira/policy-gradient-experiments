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
# Install base dependencies (excluding torch/flash-attn)
uv sync

# Install PyTorch and Flash Attention with precompiled wheels
# NOTE: These are excluded from pyproject.toml to use specific precompiled versions
source ~/.local/bin/env
uv pip install torch==2.7.1
uv pip install --no-build-isolation flash-attn==2.8.0.post2
```

**Why this approach works:**
- **Precompiled wheels**: PyTorch 2.7.1 and Flash Attention 2.8.0.post2 have official precompiled Linux wheels
- **No compilation**: Avoids lengthy compilation and build errors
- **Dependency order**: Installing torch first provides build dependencies for flash-attn
- **Build isolation**: `--no-build-isolation` allows flash-attn to use the pre-installed torch

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
# Test Flash Attention installation
source ~/.local/bin/env
uv run python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
# Expected output: Flash Attention: 2.8.0.post2

# Test PyTorch installation
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected output: PyTorch: 2.7.1+cu121

# Test model loading with Flash Attention
uv run python -c "
from utils.model_utils import setup_model_and_tokenizer
model, tokenizer, device = setup_model_and_tokenizer('.models/Qwen2.5-Math-1.5B-Instruct', 'cuda:0')
print(f'Model on {device}, Flash Attention: {getattr(model.config, \"_attn_implementation\", \"unknown\")}')
"
# Expected output: Model on cuda:0, Flash Attention: flash_attention_2
```

## Troubleshooting

**If Flash Attention installation fails:**
1. Ensure torch is installed first: `uv pip install torch==2.7.1`  
2. Clear any cached builds: `uv cache clean`
3. Try installing with verbose output: `uv pip install -v --no-build-isolation flash-attn==2.8.0.post2`

**If you get "Could not locate vLLM internal model" errors:**
- This is expected with vLLM v1 architecture changes
- Hot swapping is disabled but training will proceed normally
- Use periodic model saves instead of in-memory weight swapping