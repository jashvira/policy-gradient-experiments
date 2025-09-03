#!/usr/bin/env python3
"""Minimal MBPP evaluation scaffold using Verifiers (no bloat)."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MBPP eval with Verifiers")
    parser.add_argument("--dataset", type=str, default="valid", help="Split (train/valid/test/full) or JSONL path")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/jash404/RL_experiments/datasets/mbpp"), help="MBPP data dir")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to run")
    parser.add_argument("--model", type=str, default="llama-3.1-8b", help="Model name")
    parser.add_argument("--evaluate", action="store_true", help="Run env.evaluate()")
    parser.add_argument("--dry-run", action="store_true", help="Show config and exit")
    return parser.parse_args()


def resolve_dataset_path(dataset: str, data_dir: Path) -> Path:
    if Path(dataset).exists():
        return Path(dataset)
    mapping = {
        "train": data_dir / "mbpp_train.jsonl",
        "valid": data_dir / "mbpp_valid.jsonl",
        "test": data_dir / "mbpp_test.jsonl",
        "full": data_dir / "mbpp.jsonl",
    }
    if dataset not in mapping:
        raise ValueError(f"Invalid dataset '{dataset}'. Use one of {list(mapping.keys())} or a JSONL path.")
    path = mapping[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset split '{dataset}' not found at {path}")
    return path


def load_mbpp(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            text = rec.get("text") or rec.get("question") or "Solve the task."
            dataset.append({"prompt": text})
    return dataset


def main() -> None:
    args = parse_args()
    try:
        dataset_path = resolve_dataset_path(args.dataset, args.data_dir)
    except Exception as e:
        print(f"Error resolving dataset: {e}")
        return

    data = load_mbpp(dataset_path, args.num_examples)

    print("Config:")
    print({
        "dataset_path": str(dataset_path),
        "num_examples": len(data),
        "model": args.model,
    })

    if args.dry_run and not args.evaluate:
        if data:
            print("example_prompt:", data[0]["prompt"])
        return

    try:
        import verifiers as vf
        from openai import OpenAI
    except Exception as e:
        print(f"Missing deps: {e}")
        return

    async def non_empty_completion(prompt, completion, answer, state):
        content = (completion[-1].get("content", "") if completion else "").strip()
        return 1.0 if content else 0.0

    rubric = vf.Rubric(funcs=[non_empty_completion], weights=[1.0])
    env = vf.SingleTurnEnv(dataset=data, rubric=rubric)

    if not args.evaluate:
        print("Scaffold ready. Use --evaluate to run evaluation.")
        print("Docs:", "https://verifiers.readthedocs.io/en/latest/overview.html")
        return

    client = OpenAI()

    results = env.evaluate(
        client,
        model=args.model,
        num_examples=args.num_examples,
        rollouts_per_example=1,
        max_concurrent=8,
    )

    rewards = getattr(results, "reward", None)
    if rewards:
        avg = sum(rewards) / len(rewards)
        print({"examples": len(rewards), "avg_reward": avg})
    else:
        print("No rewards returned.")


if __name__ == "__main__":
    main()
