#!/usr/bin/env python3
"""
Split MBPP into prompt/test/valid/train according to upstream README.

Sources expected:
- datasets/mbpp/sanitized-mbpp.json (list[dict])  [preferred]
- datasets/mbpp/mbpp.jsonl (fallback; json per line)

Outputs (JSONL):
- datasets/mbpp/mbpp_prompts.jsonl     task_id in [1, 10]
- datasets/mbpp/mbpp_test.jsonl        task_id in [11, 510]
- datasets/mbpp/mbpp_valid.jsonl       task_id in [511, 600]
- datasets/mbpp/mbpp_train.jsonl       task_id in [601, 974]

Only tasks present in the input are emitted (sanitized set may exclude some).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


DATA_DIR = Path("datasets/mbpp")
SANITIZED_JSON = DATA_DIR / "sanitized-mbpp.json"
RAW_JSONL = DATA_DIR / "mbpp.jsonl"


def read_tasks() -> List[Dict[str, Any]]:
    if SANITIZED_JSON.exists():
        with SANITIZED_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("sanitized-mbpp.json is not a list of task dicts")
            return data
    if RAW_JSONL.exists():
        tasks: List[Dict[str, Any]] = []
        with RAW_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tasks.append(json.loads(line))
        return tasks
    raise FileNotFoundError(
        f"No input found. Expected {SANITIZED_JSON} or {RAW_JSONL}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    tasks = read_tasks()
    by_id = {int(t["task_id"]): t for t in tasks if "task_id" in t}

    ranges = {
        "prompts": range(1, 10 + 1),
        "test": range(11, 510 + 1),
        "valid": range(511, 600 + 1),
        "train": range(601, 974 + 1),
    }

    outputs = {
        "prompts": DATA_DIR / "mbpp_prompts.jsonl",
        "test": DATA_DIR / "mbpp_test.jsonl",
        "valid": DATA_DIR / "mbpp_valid.jsonl",
        "train": DATA_DIR / "mbpp_train.jsonl",
    }

    for split, r in ranges.items():
        rows = [by_id[i] for i in r if i in by_id]
        n = write_jsonl(outputs[split], rows)
        print(f"Wrote {n:4d} records -> {outputs[split]}")


if __name__ == "__main__":
    main()

