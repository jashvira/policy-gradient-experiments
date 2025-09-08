# MBPP Baseline (Verifiers environment)
Baseline MBPP environment for verifiers with compile and tests scoring via SandboxFusion.

- Task: MBPP (Mostly Basic Python Programs).
- Entry point: `load_environment(...) -> verifiers.Environment`.
- Key parameters: `dataset_split`, `eval_split`, `num_examples`, `data_dir`, `prompt_file`, `include_tests_in_prompt`.

## Rubric and SandboxFusion
The rubric executes code via SandboxFusion:
- `compile_reward`: extracts the model’s Python code block, removes any `if __name__ == "__main__":` section, and executes it.
- `tests_reward`: builds a test harness from MBPP `test_list` and optional `test_setup_code`, then executes it.
- Both call `mbpp_baseline.util.sandbox.run_code`, which POSTs to `SANDBOX_URL` (default `http://localhost:8080`).

Run SandboxFusion locally:
```bash
export SANDBOX_URL=http://localhost:8080
sudo docker run -d -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```
If unreachable, a warning is logged.

## Data
Provide MBPP JSONL under `data_dir` or place files at a known location. Expected filenames:
- `mbpp_train.jsonl`, `mbpp_valid.jsonl`, `mbpp_test.jsonl` (or `mbpp.jsonl` for `full`)

Dataset available at: **https://huggingface.co/datasets/jash404/mbpp**

You can pass `data_dir` via code or CLI flags supported by verifiers.

## Quick start (local)
```bash
uv pip install -e . && uv run vf-eval mbpp_baseline -n 10
```

## Minimal usage (programmatic)
```python
from verifiers import load_environment
env = load_environment("mbpp_baseline", dataset_split="valid", num_examples=50)
```

## Notes
- Single-turn env; weights: compile 0.2, tests 0.8.
- Ensure `SANDBOX_URL` points to a running SandboxFusion server.
