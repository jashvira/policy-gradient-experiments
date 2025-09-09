"""Core utilities for MBPP evaluation."""

from __future__ import annotations

import re
from typing import Any, Iterable

import requests

from mbpp_baseline.util.sandbox import run_code


def extract_text_from_completion(completion: Any) -> str:
    """Extract assistant text content from various completion formats."""
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "")

    if isinstance(completion, list):
        for message in completion:
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content") or message.get("text")
                if isinstance(content, list):
                    parts = [
                        part.get("text") or part.get("content") or ""
                        if isinstance(part, dict)
                        else str(part)
                        for part in content
                    ]
                    content = "".join(parts)
                return str(content or "")

    return ""


def extract_code_from_text(text: str) -> str:
    """Extract Python code from fenced code blocks."""
    if not text:
        return ""

    match = re.search(
        r"```(?:python)?\s*\n?(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else ""


def extract_code_from_completion(completion: Any) -> str:
    """Extract code directly from completion object."""
    text = extract_text_from_completion(completion)
    return extract_code_from_text(text)


def clean_code_main_block(code: str) -> str:
    """Remove `if __name__ == "__main__":` blocks from code."""
    lines = code.split("\n")
    cleaned_lines: list[str] = []
    skipping = False
    base_indent: str | None = None

    pattern = re.compile(r'^(?P<indent>[ \t]*)if __name__ == [\'\"]__main__[\'\"]\s*:\s*$')

    for line in lines:
        if not skipping:
            match = pattern.match(line)
            if match:
                skipping = True
                base_indent = match.group("indent")
                continue
            cleaned_lines.append(line)
            continue

        # Skip lines in main block
        if line.strip() == "":
            continue
        if base_indent is not None and (line.startswith(base_indent + " ") or line.startswith(base_indent + "\t")):
            continue

        # End of main block
        skipping = False
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def build_test_harness(
    solution_code: str,
    setup_code: str | None = None,
    tests: Iterable[str] | None = None,
) -> str:
    """Build a Python script that runs solution code against test assertions."""
    parts: list[str] = []

    setup = (setup_code or "").strip()
    if setup:
        parts.append(setup)

    solution = (solution_code or "").strip()
    if solution:
        parts.append(solution)

    test_lines = [line.strip() for line in (tests or []) if line.strip()]
    if test_lines:
        parts.append("\n".join(test_lines))

    return "\n\n".join(parts) + "\n" if parts else "\n"


def calculate_compile_reward(completion: Any) -> float:
    """Return 1.0 if code compiles and runs, else 0.0."""
    code = extract_code_from_completion(completion)
    if not code.strip():
        return 0.0

    code = clean_code_main_block(code)
    result = run_code(code)

    if result.get("status") == "Success":
        run_result = result.get("run_result", {})
        return 1.0 if run_result.get("return_code") == 0 else 0.0
    return 0.0


def calculate_tests_reward(completion: Any, info: dict[str, Any]) -> float:
    """Return 1.0 if code passes all tests, else 0.0."""
    tests = info.get("test_list", [])
    setup_code = info.get("test_setup_code", "")

    if not tests:
        return 0.0

    code = extract_code_from_completion(completion)
    if not code.strip():
        return 0.0

    code = clean_code_main_block(code)
    script = build_test_harness(code, setup_code, tests)
    result = run_code(script)

    if result.get("status") == "Success":
        run_result = result.get("run_result", {})
        return 1.0 if run_result.get("return_code") == 0 else 0.0
    return 0.0


def calculate_format_reward(completion: Any) -> float:
    """Return 1.0 if completion contains a fenced Python code block, else 0.0.

    Format-only: no execution and no syntax parsing; purely checks for
    ```python\n...\n``` anywhere in the text.
    """
    text = extract_text_from_completion(completion).strip()

    # Reward if any python fenced block exists (not necessarily the only content)
    pattern = re.compile(r"```python\s*\n[\s\S]+?\n```", re.IGNORECASE)
    return 1.0 if pattern.search(text) else 0.0
