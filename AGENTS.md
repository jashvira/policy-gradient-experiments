# Repository Guidelines

## Project Structure & Module Organization
- `experiments/`: Training entry points and configs
  - `sft/` (supervised fine-tuning), `grpo/` (policy-gradient), `baseline/`, `ei/`
  - Example config: `experiments/grpo/configs/grpo_clip.yaml`
- `utils/`: Shared helpers (data, grading, model/setup, training, memory)
- `tests/`: Pytest suite with snapshots in `tests/_snapshots`
- `scripts/`: Env/setup scripts (see `scripts/setup_env.sh`)
- `assets/`: Plots and figures; `logs/`, `eval_runs/` contain outputs
- `.models/`: Local model cache (e.g., `Qwen2.5-Math-1.5B`)

## Build, Test, and Development Commands
- Environment: `uv sync` (installs base deps from `pyproject.toml`)
- Optional dev tools: `uv pip install -e .[dev]` (Black, pre-commit, Jupyter)
- Format: `uv run black .` (line length 100)
- Test all: `uv run pytest -q`
- Test subset: `uv run pytest -q -k grpo`
- SFT train: `uv run python experiments/sft/sft_train.py --config experiments/sft/configs/<cfg>.yaml`
- GRPO train: `uv run python experiments/grpo/grpo_train.py --config experiments/grpo/configs/grpo_clip.yaml`
- Turnkey setup (Torch + Flash‑Attn + model): `bash setup.sh`

## Coding Style & Naming Conventions
- Language: Python 3.11; formatter: Black (line length 100)
- Indentation: 4 spaces; no tabs
- Names: modules `lower_snake_case.py`, functions `snake_case`, classes `PascalCase`
- Configs: YAML under `experiments/*/configs/`; prefer explicit, descriptive keys

## Testing Guidelines
- Framework: Pytest; tests live in `tests/test_*.py`
- Snapshots: `.npz`/`.pkl` in `tests/_snapshots`; verify with `uv run pytest`
- Exact matching (floats): add `--snapshot-exact`
- Update snapshots only when behavior changes intentionally (regenerate expected files)

## Commit & Pull Request Guidelines
- Messages: imperative mood, short scope prefix when relevant (e.g., `grpo: fix logging`)
- Body (when needed): bullets with rationale and impact
- PRs: clear description, link issues, include commands to reproduce, and sample logs/plots (`assets/` or `logs/`)
- Include config path(s) used for runs and any env vars (e.g., `WANDB_PROJECT`, `WANDB_ENTITY`)

## Security & Configuration Tips
- Prefer local models in `.models/` for reproducibility
- Respect GPU limits; set `CUDA_VISIBLE_DEVICES` as needed
- Keep secrets (API keys, W&B tokens) out of commits; use env vars
