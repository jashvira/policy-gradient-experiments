#!/usr/bin/env python3
"""MBPP evaluation scaffold using Verifiers."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


# ============================================================================
# Configuration & Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MBPP evaluation with Verifiers")
    parser.add_argument(
        "--dataset",
        type=str,
        default="valid",
        help="Split (train/valid/test/full) or JSONL path"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="MBPP data directory (default: datasets/mbpp relative to project root)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder-1.5b",
        help="Model name or path"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation (default: dry run)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show config and example, then exit"
    )
    return parser.parse_args()


# ============================================================================
# Dataset Loading
# ============================================================================

def resolve_dataset_path(dataset: str, data_dir: Path) -> Path:
    """Resolve dataset string to actual file path."""
    if Path(dataset).exists():
        return Path(dataset)

    mapping = {
        "train": data_dir / "mbpp_train.jsonl",
        "valid": data_dir / "mbpp_valid.jsonl",
        "test": data_dir / "mbpp_test.jsonl",
        "full": data_dir / "mbpp.jsonl",
    }

    if dataset not in mapping:
        raise ValueError(f"Invalid dataset '{dataset}'. Use: {list(mapping.keys())} or JSONL path")

    path = mapping[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset}' not found at {path}")

    return path


def load_mbpp(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load MBPP dataset from JSONL file."""
    dataset = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            rec = json.loads(line.strip())
            text = rec.get("text") or rec.get("question") or "Solve the task."
            dataset.append({"prompt": text})

    return dataset


# ============================================================================
# Evaluation Setup
# ============================================================================

def create_verifiers_env():
    """Create Verifiers environment using the modular environment setup."""
    try:
        from eval.my_coder_env import load_environment
    except ImportError:
        raise ImportError("Environment module not found. Check eval/my_coder_env.py exists")

    return load_environment()


# ============================================================================
# Main Logic
# ============================================================================

def main() -> None:
    """Main evaluation logic."""
    args = parse_args()

    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = _get_project_root() / "datasets" / "mbpp"

    # Load dataset (for display purposes only - env handles its own dataset)
    try:
        dataset_path = resolve_dataset_path(args.dataset, args.data_dir)
        data = load_mbpp(dataset_path, args.num_examples)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Show configuration
    config = {
        "dataset_path": str(dataset_path),
        "num_examples": len(data),
        "model": args.model,
        "evaluate": args.evaluate,
    }
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Dry run mode
    if args.dry_run:
        if data:
            print(f"\nExample prompt:\n{data[0]['prompt'][:200]}...")
        return

    # Create evaluation environment
    try:
        env = create_verifiers_env()
    except Exception as e:
        print(f"Error setting up environment: {e}")
        return

    if not args.evaluate:
        print("\nScaffold ready. Use --evaluate to run evaluation.")
        print("Docs: https://verifiers.readthedocs.io/en/latest/overview.html")
        return

    # TODO: Replace with local model inference
    print("\nEvaluation not implemented yet (OpenAI dependency removed)")
    print("Next steps:")
    print("1. Implement local model loading")
    print("2. Add proper evaluation logic")
    print("3. Integrate with Qwen2.5-Coder model")


if __name__ == "__main__":
    main()
