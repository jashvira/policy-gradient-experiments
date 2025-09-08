#!/usr/bin/env python3
"""
Minimal programmatic MBPP evaluation that delegates to Verifiers' built-in evaluator
(equivalent to `vf-eval -s`). Always saves dataset-formatted outputs for vf-tui.

Usage:
  # Local vLLM
  uv run python runs/programmatic_eval.py \
    --base-url http://localhost:8000/v1 \
    --model .models/Qwen2.5-Coder-1.5B \
    --dataset valid \
    --num-examples 5 \
    --rollouts 1

  # OpenAI (API key read from .env -> OPENAI_API_KEY)
  OPENAI_API_KEY=sk-... uv run python runs/programmatic_eval.py \
    --base-url https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --dataset valid \
    --num-examples 5 \
    --rollouts 1
"""

import argparse
import os
import sys
from pathlib import Path

import verifiers as vf
from verifiers.scripts.eval import eval_environment as vf_eval_environment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Programmatic MBPP eval (uses Verifiers saver)")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (e.g., http://localhost:8000/v1)")
    p.add_argument("--model", required=True, help="Model name or local identifier")
    p.add_argument("--dataset", default="valid", help="Dataset split: train|valid|test|full")
    p.add_argument("--num-examples", type=int, default=1000)
    p.add_argument("--rollouts", type=int, default=1)
    p.add_argument("--max-concurrent", "-c", type=int, default=64, help="Maximum number of concurrent requests")
    # API key is read from environment by default: OPENAI_API_KEY or VLLM_API_KEY (unused)
    p.add_argument("--api-key-var", default=None, help="Env var name for API key; default reads OPENAI_API_KEY")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).parents[1]
    envs_dir = project_root / "environments"
    sys.path.insert(0, str(envs_dir))

    # Cap num_examples to dataset size to avoid IndexError
    try:
        env_preview = vf.load_environment("mbpp_baseline", dataset_split=args.dataset)
        dataset_size = len(getattr(env_preview, "dataset", []))
    except Exception:
        dataset_size = None

    adjusted_num = args.num_examples
    if dataset_size is not None and adjusted_num > dataset_size:
        adjusted_num = dataset_size
        print(f"Capping num-examples to dataset size: {dataset_size}")

    # Resolve API key var from .env-loaded environment (uv respects .env automatically)
    api_key_var = args.api_key_var or ("OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY") else "EMPTY")

    vf_eval_environment(
        env="mbpp_baseline",
        env_args={"dataset_split": args.dataset, "eval_split": args.dataset},
        env_dir_path=str(envs_dir),
        endpoints_path=str(project_root / "configs" / "endpoints.py"),
        model=args.model,
        api_key_var=api_key_var,
        api_base_url=args.base_url,
        num_examples=adjusted_num,
        rollouts_per_example=args.rollouts,
        max_concurrent=args.max_concurrent,
        max_tokens=None,
        temperature=None,
        sampling_args=None,
        verbose=True,
        save_dataset=True,
        save_to_hf_hub=False,
        hf_hub_dataset_name="",
    )


if __name__ == "__main__":
    main()


