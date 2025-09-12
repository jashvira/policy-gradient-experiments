#!/bin/bash
# Start vLLM server for multi-GPU inference

MODEL_PATH="${MODEL_PATH:-.models/Qwen2.5-3B}"
PORT="${PORT:-8000}"
GPU="${CUDA_VISIBLE_DEVICES:-0,1,2}"

# Derive data-parallel size from GPU list
IFS=',' read -r -a GPU_ARR <<< "$GPU"
DP_SIZE=${#GPU_ARR[@]}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

echo "Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPU: $GPU"
echo "Data parallel size: $DP_SIZE"

CUDA_VISIBLE_DEVICES=$GPU \
VLLM_NO_USAGE_STATS=1 \
uv run python serve/vllm_py313_fix.py \
  --model "$MODEL_PATH" \
  --port "$PORT" \
  --data-parallel-size "$DP_SIZE" \
  --enforce-eager \
  --disable-log-requests
