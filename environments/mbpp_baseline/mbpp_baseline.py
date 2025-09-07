"""MBPP baseline environment for Verifiers.

A clean evaluation environment for the MBPP (Mostly Basic Python Programs) dataset
that uses SandboxFusion for remote code execution and reward calculation.
"""

import json
from pathlib import Path
from datasets import Dataset
import verifiers as vf
from mbpp_baseline.util.mbpp_utils import calculate_compile_reward, calculate_tests_reward


def _get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _load_dataset(
    dataset_split: str = "valid",
    data_dir: Path | None = None,
    include_tests_in_prompt: bool = True,
) -> Dataset:
    if data_dir is None:
        data_dir = _get_project_root() / "datasets" / "mbpp"

    mapping = {
        "train": data_dir / "mbpp_train.jsonl",
        "valid": data_dir / "mbpp_valid.jsonl",
        "test": data_dir / "mbpp_test.jsonl",
        "full": data_dir / "mbpp.jsonl",
    }
    path = mapping.get(dataset_split)
    if path is None:
        raise ValueError(f"Invalid dataset '{dataset_split}'. Use: {list(mapping.keys())}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_split}' not found at {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text") or rec.get("prompt") or "Solve the task."
            if include_tests_in_prompt:
                tests = rec.get("test_list") or []
                # Keep concise and deterministic; avoid challenge tests by default
                if tests:
                    formatted_tests = "\n".join(str(t).strip() for t in tests if str(t).strip())
                    text = (
                        f"{text}\n\nYour code should pass these tests:\n\n{formatted_tests}\n"
                    )
            rows.append({
                "question": text,
                "info": {
                    "test_list": rec.get("test_list", []),
                    "test_setup_code": rec.get("test_setup_code", ""),
                }
            })
    return Dataset.from_list(rows)




def _build_rubric(parser: vf.Parser) -> vf.Rubric:
    """Rubric with compile and test rewards."""

    def compile_reward(parser, completion, answer, **kwargs):
        return calculate_compile_reward(completion)

    def tests_reward(parser, completion, answer, **kwargs):
        info = kwargs.get("info", {})
        return calculate_tests_reward(completion, info)

    return vf.Rubric(
        funcs=[compile_reward, tests_reward],
        weights=[0.2, 0.8],
        parser=parser,
    )


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Python programmer.\n"
    "Think step-by-step inside <think>...</think>.\n"
    "Then output ONLY a single fenced Python code block with the final solution.\n"
    "Format strictly as:\n"
    "```python\n"
    "# your function implementation\n"
    "```\n"
    "No prose or extra text outside the code block.\n"
)


def load_environment(
    dataset_split: str = "train",
    eval_split: str | None = "valid",
    num_examples: int | None = None,
    data_dir: Path | None = None,
    prompt_file: Path | None = None,
    include_tests_in_prompt: bool = True,
    **kwargs,
) -> vf.Environment:
    # Prefer inline default prompt; allow optional override via file for flexibility
    if prompt_file is not None:
        system_prompt = Path(prompt_file).read_text().strip()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    dataset = _load_dataset(dataset_split, data_dir, include_tests_in_prompt)
    if num_examples is not None and num_examples > 0:
        cap = min(num_examples, len(dataset))
        dataset = dataset.select(range(cap))

    eval_dataset = None
    if eval_split is not None:
        try:
            eval_dataset = _load_dataset(eval_split, data_dir, include_tests_in_prompt)
        except Exception:
            eval_dataset = None

    parser = None
    rubric = _build_rubric(parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )


