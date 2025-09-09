### Verifiers: minimal workflow (tight)

- **Env module (single file)**
  - `environments/<env_name>/<env_name>.py` exposes `load_environment(...) -> vf.Environment`.
  - `environments/<env_name>/pyproject.toml` (deps incl. `verifiers`; hatch build include for the file).
  - Outputs saved under `environments/<env_name>/outputs/evals/...` (gitignored).

- **Design (what `load_environment` does)**
  - Load HF `Dataset` (e.g., from `datasets/.../*.jsonl`) with a prompt column (or map `question` → messages).
  - Build parser (e.g., `vf.ThinkParser`) and rubric (weighted reward funcs).
  - Return `vf.SingleTurnEnv(dataset=..., eval_dataset=..., system_prompt=..., parser=..., rubric=...)`.
  - Keep configurable: `dataset_split`, `eval_split`, `num_examples`, `data_dir`, `prompt_file`.

- **Install env (editable)**
```bash
vf-install <env_name> -p environments
```

- **Start model server (OpenAI‑compatible, e.g., vLLM)**
```bash
CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 \
uv run vf-vllm --model .models/Qwen2.5-Coder-3B \
  --gpu-memory-utilization 0.95 --enforce-eager --disable-log-requests
```

- **Quick eval + save (CLI)**
```bash
vf-eval <env_name> -b http://localhost:8000/v1 -m .models/Qwen2.5-Coder-3B \
  -n 50 -r 2 -s
```

- **Programmatic eval (uses library saver, vf-eval -s equivalent)**
```bash
uv run python runs/programmatic_eval.py \
  --base-url http://localhost:8000/v1 \
  --model .models/Qwen2.5-Coder-3B \
  --dataset valid --num-examples 50 --rollouts 2 -c 64
```

- **Inspect results**
```bash
vf-tui  # browses environments/<env_name>/outputs/evals/.../results.jsonl
```

- **Best practices**
  - Single-file env + minimal `pyproject.toml` (like Wordle env).
  - Relative paths only; no hardcoded absolutes.
  - Cap `num_examples` to dataset size to avoid index errors.
  - Pass an explicit `eval_dataset`/`eval_split` if you want no fallback.
