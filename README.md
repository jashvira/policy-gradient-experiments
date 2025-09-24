## Results with Qwen2.5-1.5B



### Training Report
<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; background: #f6f8fa; border-radius: 8px; margin: 1rem 0;">
  <iframe
    src="https://wandb.ai/jashvira-maptek/mbpp/reports/MBPP-hill-climb-with-Qwen2-5-1-5B--VmlldzoxNDUxMTkyNg?accessToken=8d0wu7flekv2ltw3mxpb6nhx3blwaysut4e6j3958kflkc6iu58w17fgn2w6ecxc#train-rewards"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none; border-radius: 8px;"
    title="MBPP Training Results - Qwen2.5-1.5B">
  </iframe>
</div>

**[Complete Results](https://api.wandb.ai/links/jashvira-maptek/5yjovjdt)**

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


