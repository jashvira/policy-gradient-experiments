"""MBPP baseline environment (single-file module) for Verifiers.

This file exposes `load_environment(...)` and follows the minimal pattern
recommended by the Verifiers docs. It can be installed via
`vf-install environments/mbpp_baseline`.
"""

import json
from pathlib import Path
from datasets import Dataset
import verifiers as vf


def _get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def _load_dataset(dataset_split: str = "valid", data_dir: Path | None = None) -> Dataset:
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
            rows.append({"question": text, "info": {}})
    return Dataset.from_list(rows)


def _build_parser() -> vf.ThinkParser:
    return vf.ThinkParser(extract_fn=lambda x: x)


def _build_rubric(parser: vf.Parser) -> vf.Rubric:
    def code_correctness(parser, completion, answer, **kwargs):
        code = parser.parse_answer(completion) or ""
        if not code.strip():
            return 0.0
        try:
            compile(code, "<string>", "exec")
            return 1.0
        except SyntaxError:
            return 0.0

    def has_code_block(parser, completion, **kwargs):
        code = parser.parse_answer(completion) or ""
        return 1.0 if code and code.strip() else 0.0

    return vf.Rubric(
        funcs=[code_correctness, has_code_block, parser.get_format_reward_func()],
        weights=[1.0, 0.3, 0.2],
        parser=parser,
    )


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Python programmer.\n\n"
    "Here is your task:\n{question}\n\n"
    "The reasoning process must be enclosed within <think> and </think>, and must appear BEFORE the code section. "
    "After </think>, output ONLY the final Python code block. Do not include any other text outside the code block.\n\n"
    "<think>"
)


def load_environment(
    dataset_split: str = "train",
    eval_split: str | None = "valid",
    num_examples: int | None = None,
    data_dir: Path | None = None,
    prompt_file: Path | None = None,
    **kwargs,
) -> vf.Environment:
    # Prefer inline default prompt; allow optional override via file for flexibility
    if prompt_file is not None:
        system_prompt = Path(prompt_file).read_text().strip()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    dataset = _load_dataset(dataset_split, data_dir)
    if num_examples is not None and num_examples > 0:
        cap = min(num_examples, len(dataset))
        dataset = dataset.select(range(cap))

    eval_dataset = None
    if eval_split is not None:
        try:
            eval_dataset = _load_dataset(eval_split, data_dir)
        except Exception:
            eval_dataset = None

    parser = _build_parser()
    rubric = _build_rubric(parser)

    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )


