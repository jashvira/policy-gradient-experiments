#!/usr/bin/env python3
"""
One-off: Filter reasoning SFT examples to only those that produce the correct answer.

Usage:
  python experiments/sft/filter_correct_sft.py INPUT.jsonl OUTPUT.jsonl [--no-fast]

Expected JSONL fields per example:
  - response (preferred) or completion
  - ground_truth (preferred) or answer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from utils.drgrpo_grader import r1_zero_reward_fn

JsonDict = Dict[str, Any]


def get_response(ex: JsonDict) -> Optional[str]:
    val = ex.get("response")
    if isinstance(val, str):
        return val
    val = ex.get("completion")
    if isinstance(val, str):
        return val
    return None


def get_ground_truth(ex: JsonDict) -> Optional[Union[str, float, int]]:
    val = ex.get("ground_truth")
    if isinstance(val, (str, float, int)):
        return val
    val = ex.get("answer")
    if isinstance(val, (str, float, int)):
        return val
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SFT JSONL to only correct examples")
    parser.add_argument("input", type=str, help="Path to input SFT JSONL")
    parser.add_argument("output", type=str, help="Path to output filtered JSONL")
    parser.add_argument("--no-fast", dest="fast", action="store_false", help="Disable fast grading mode")
    parser.set_defaults(fast=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    evaluable = 0
    scanned = 0

    with in_path.open("r") as fin, out_path.open("w") as fout:
        for line in fin:
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                ex: JsonDict = json.loads(line)
            except Exception:
                continue

            response = get_response(ex)
            gt = get_ground_truth(ex)
            if response is None or gt is None:
                continue

            evaluable += 1
            rewards = r1_zero_reward_fn(response, gt, fast=args.fast)
            if rewards.get("reward"):
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Kept {kept} correct out of {evaluable} evaluable (scanned {scanned}).")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


