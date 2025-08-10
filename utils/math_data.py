from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict


def load_math_validation() -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load MATH validation dataset from MATH/validation.jsonl.

    Returns:
        Tuple of (problems, ground_truths, metadata) lists
    """
    repo_root = Path(__file__).resolve().parents[1]
    validation_path = repo_root / "MATH" / "validation.jsonl"

    problems: List[str] = []
    answers: List[str] = []
    metadata: List[Dict] = []

    with open(validation_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            problems.append(data["problem"])
            answers.append(data["answer"])
            metadata.append(
                {
                    "problem": data["problem"],
                    "subject": data.get("subject"),
                    "level": data.get("level"),
                    "unique_id": data.get("unique_id"),
                }
            )

    return problems, answers, metadata


def format_with_r1_zero_prompt(problem: str) -> str:
    """
    Format problem with r1_zero prompt template.

    Args:
        problem: The mathematical problem to solve

    Returns:
        Formatted prompt string
    """
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "r1_zero.prompt"
    template = prompt_path.read_text()
    return template.format(question=problem)


