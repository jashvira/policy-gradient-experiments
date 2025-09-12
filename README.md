## Quickstart

Minimal, Verifiers-aligned workflow for MBPP with vLLM + SandboxFusion.

### Prerequisites
- Python 3.11+, `uv`, `docker`
- GPU + CUDA drivers for vLLM

### Install dependencies
```bash
uv sync  # installs project deps incl. verifiers[all], vLLM, etc.
```

### Install environment (editable)
```bash
uv run vf-install mbpp_baseline -p environments
```

### Start SandboxFusion (8080)
Use the official server image to avoid building from source.
```bash
docker run -d --rm --privileged -p 8080:8080 \
  --name sandboxfusion volcengine/sandbox-fusion:server-20250609
export SANDBOX_URL=http://localhost:8080
```

Quick check:
```bash
curl -s -X POST "$SANDBOX_URL/run_code" \
  -H 'Content-Type: application/json' \
  -d '{"code":"print(1)","language":"python"}' | python -m json.tool
```

### Start vLLM (8000)
```bash
bash serve/vllm.sh
```

Get the served model name:
```bash
MODEL=$(curl -s http://localhost:8000/v1/models | jq -r '.data[0].id')
echo $MODEL
```

### Run evaluation
```bash
uv run python runs/programmatic_eval.py \
  -b http://localhost:8000/v1 \
  -m "$MODEL" \
  -d mbpp_baseline \
  -n 10 -r 1
```

Alternative (CLI):
```bash
uv run vf-eval mbpp_baseline -b http://localhost:8000/v1 -m "$MODEL" -n 10
```

### Inspect results
```bash
uv run vf-tui
```


