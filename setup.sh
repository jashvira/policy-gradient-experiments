#!/usr/bin/env bash
set -euo pipefail

# 1. Install UV if not present
if ! command -v uv &> /dev/null; then
  echo "Installing UV package manager..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc || true
  source $HOME/.local/bin/env || true
fi

# 2. Sync dependencies
echo "Syncing dependencies with uv..."
uv sync

# 3. Install PyTorch and Flash Attention
echo "Installing PyTorch 2.7.1..."
uv pip install torch==2.7.1

echo "Installing Flash Attention 2.8.0.post2..."
uv pip install --no-build-isolation flash-attn==2.8.0.post2

# 4. Download model
MODEL_DIR=".models/Qwen2.5-Math-1.5B"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Downloading Qwen2.5-Math-1.5B model..."
  mkdir -p .models
  uv run python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = 'Qwen/Qwen2.5-Math-1.5B'
save_path = '$MODEL_DIR'
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
else
  echo "Model already downloaded at $MODEL_DIR"
fi

# 5. Verify installations
echo "Verifying Flash Attention installation..."
uv run python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"

echo "Verifying PyTorch installation..."
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo "Verifying model loading with Flash Attention..."
uv run python -c "from utils.model_utils import setup_model_and_tokenizer; model, tokenizer, device = setup_model_and_tokenizer('$MODEL_DIR', 'cuda:0'); print('Model on', device, ', Flash Attention:', getattr(model.config, '_attn_implementation', 'unknown'))"

echo "Setup complete."
