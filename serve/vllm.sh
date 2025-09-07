#!/bin/bash
# Start vLLM server for MBPP evaluation

MODEL_PATH="${MODEL_PATH:-.models/Qwen2.5-Coder-1.5B}"
PORT="${PORT:-8000}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

echo "Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU \
VLLM_NO_USAGE_STATS=1 \
uv run python serve/vllm_py313_fix.py \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --enforce-eager \
  --disable-log-requests
