#!/usr/bin/env python3
"""
Evaluate model performance on MBPP coding tasks using Verifiers.

Usage:
    python eval_performance.py --model Qwen2.5-Coder-1.5B --num-examples 50
    python eval_performance.py --dataset test --num-examples 100 --rollouts 2
"""

import argparse
import sys
from pathlib import Path

from openai import OpenAI

from eval.my_coder_env import load_environment


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model performance on MBPP")
    parser.add_argument("--model", default="Qwen2.5-Coder-1.5B", help="Model name")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--dataset", default="valid", help="Dataset split: train|valid|test|full")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mbpp"), help="Dataset directory")
    parser.add_argument("--num-examples", type=int, default=20, help="Number of examples to evaluate")
    parser.add_argument("--rollouts", type=int, default=1, help="Rollouts per example")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent requests")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== MBPP Performance Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({args.num_examples} examples)")
    print(f"Server: {args.base_url}")
    print()

    # Load environment and client
    try:
        env = load_environment(dataset_split=args.dataset, data_dir=args.data_dir)
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    except Exception as e:
        print(f"Error setting up environment: {e}")
        sys.exit(1)

    # Run evaluation per Verifiers docs pattern
    try:
        results = env.evaluate(
            client, args.model,
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts,
            max_concurrent=args.max_concurrent,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

    # Calculate and display performance metrics
    n = len(results.prompt)
    if n == 0:
        print("No examples evaluated.")
        return

    avg_reward = sum(results.reward) / n

    print(f"=== RESULTS ===")
    print(f"Examples evaluated: {n}")
    print(f"Average reward: {avg_reward:.3f}")

if __name__ == "__main__":
    main()
