#!/usr/bin/env python3
"""
Zero-shot MATH evaluation for Qwen 2.5 Math 1.5B

This script evaluates a language model's zero-shot performance on the MATH dataset
using the r1_zero prompt format for mathematical reasoning.
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.vllm_utils import load_vllm_model, create_sampling_params, evaluate_vllm
from utils.drgrpo_grader import r1_zero_reward_fn


def load_math_validation() -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load MATH validation dataset from MATH/validation.jsonl

    Returns:
        Tuple of (problems, ground_truths, metadata) lists
    """
    validation_path = Path(__file__).parent.parent / "MATH" / "validation.jsonl"

    problems = []
    answers = []
    metadata = []

    with open(validation_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            problems.append(data['problem'])
            answers.append(data['answer'])
            metadata.append({
                'problem': data['problem'],
                'subject': data['subject'],
                'level': data['level'],
                'unique_id': data['unique_id']
            })

    print(f"Loaded {len(problems)} validation problems")
    return problems, answers, metadata


def format_with_r1_zero_prompt(problem: str) -> str:
    """
    Format problem with r1_zero prompt template

    Args:
        problem: The mathematical problem to solve

    Returns:
        Formatted prompt string
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / "r1_zero.prompt"

    with open(prompt_path, 'r') as f:
        template = f.read()

    return template.format(question=problem)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5 Math 1.5B on MATH dataset")
    parser.add_argument("--model_path", required=True, help="Path to Qwen2.5-Math-1.5B model")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory for results")

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
    print(f"Answer accuracy: {results['metadata']['overall_metrics']['answer_accuracy']:.4f}")
    print(f"Format accuracy: {results['metadata']['overall_metrics']['format_accuracy']:.4f}")
    print(f"Overall accuracy: {results['metadata']['overall_metrics']['overall_accuracy']:.4f}")
    print("\nSUBJECT BREAKDOWN:")
    print("-" * 30)
    for subject, stats in results['subject_breakdown'].items():
        print(f"{subject:20} {stats['correct']:3d}/{stats['total']:3d} ({stats['accuracy']:.3f})")
    print("="*50)


if __name__ == "__main__":
    main()