#!/usr/bin/env python3
"""
Zero-shot MATH evaluation for Qwen 2.5 Math 1.5B

This script evaluates a language model's zero-shot performance on the MATH dataset
using the r1_zero prompt format for mathematical reasoning.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for absolute imports like `utils.*` and `experiments.*`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.drgrpo_grader import r1_zero_reward_fn
from utils.math_data import load_math_validation, format_with_r1_zero_prompt
from utils.vllm_utils import load_vllm_model, create_sampling_params, evaluate_vllm
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import os


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen 2.5 Math 1.5B on MATH dataset")
    parser.add_argument("--model_path", required=True,
                        help="Path to Qwen2.5-Math-1.5B model")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--output_dir", default="evaluation_results",
                        help="Output directory for results")

    args = parser.parse_args()

    print("Loading MATH validation dataset...")
    problems, ground_truths, metadata = load_math_validation()

    print("Formatting prompts...")
    prompts = [format_with_r1_zero_prompt(problem) for problem in problems]

    print(f"Loading model from {args.model_path}...")
    model = load_vllm_model(args.model_path)

    print("Creating sampling parameters...")
    sampling_params = create_sampling_params(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop_tokens=["</answer>"]
    )

    print("Running evaluation...")
    results = evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_dir=args.output_dir,
        model_name="Qwen2.5-Math-1.5B",
        problem_metadata=metadata,
        save_full_responses=True
    )

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {results['metadata']['total_samples']}")
    print(
        f"Answer accuracy: {results['metadata']['overall_metrics']['answer_accuracy']:.4f}")
    print(
        f"Format accuracy: {results['metadata']['overall_metrics']['format_accuracy']:.4f}")
    print(
        f"Overall accuracy: {results['metadata']['overall_metrics']['overall_accuracy']:.4f}")
    print("\nSUBJECT BREAKDOWN:")
    print("-" * 30)
    for subject, stats in results['subject_breakdown'].items():
        print(
            f"{subject:20} {stats['correct']:3d}/{stats['total']:3d} ({stats['accuracy']:.3f})")
    print("="*50)


if __name__ == "__main__":
    main()
