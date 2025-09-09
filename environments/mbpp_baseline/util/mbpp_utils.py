"""Core utilities for MBPP evaluation."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

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


_RESULT_JSON_PREFIX = "RESULT_JSON:"


def _build_test_runner_script(test_lines: list[str]) -> str:
    """Generate the test execution script with JSON output."""
    test_repr = ",\n".join(repr(str(t)) for t in test_lines)
    return f"""import json
_passed = 0
_total = 0
_tests = [
{test_repr}
]
for _i, _t in enumerate(_tests):
    try:
        exec(_t, globals(), locals())
        _passed += 1
    except Exception:
        pass
    finally:
        _total += 1
print("{_RESULT_JSON_PREFIX}", json.dumps({{"passed": _passed, "total": _total}}))"""


def build_fractional_test_harness(
    solution_code: str,
    setup_code: str | None = None,
    tests: Iterable[str] | None = None,
) -> str:
    """Build a Python script that executes each test separately and reports JSON.

    Prints a single line to stdout prefixed by "RESULT_JSON: " followed by
    a JSON object {"passed": X, "total": Y} so callers can parse fractional credit.
    """
    parts: list[str] = []

    setup = (setup_code or "").strip()
    if setup:
        parts.append(setup)

    solution = (solution_code or "").strip()
    if solution:
        parts.append(solution)

    test_lines = [str(line).strip() for line in (tests or []) if str(line).strip()]
    if test_lines:
        parts.append(_build_test_runner_script(test_lines))

    return "\n\n".join(parts) + "\n"


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
    """Return fractional reward = passed_tests / total_tests.

    Falls back to binary (1.0/0.0) if JSON parsing fails, preserving prior behaviour.
    """
    tests = info.get("test_list", [])
    setup_code = info.get("test_setup_code", "")

    if not tests:
        return 0.0

    code = extract_code_from_completion(completion)
    if not code.strip():
        return 0.0

    code = clean_code_main_block(code)
    script = build_fractional_test_harness(code, setup_code, tests)
    result = run_code(script)

    if result.get("status") == "Success":
        run_result = result.get("run_result", {})
        stdout = run_result.get("stdout") or run_result.get("output") or ""

        # Try to parse fractional result; fallback to binary
        frac_result = _parse_fractional_result(stdout)
        if frac_result is not None:
            return frac_result

        # Binary fallback: previous behaviour
        return 1.0 if run_result.get("return_code") == 0 else 0.0
    return 0.0


def _parse_fractional_result(stdout: str) -> float | None:
    """Parse fractional test result from stdout. Returns None if parsing fails."""
    try:
        pattern = rf"{re.escape(_RESULT_JSON_PREFIX)}\s*({{.*}})"
        matches = re.findall(pattern, stdout)
        if not matches:
            return None

        payload = json.loads(matches[-1])
        passed = float(payload.get("passed", 0))
        total = float(payload.get("total", 0))

        if total <= 0:
            return 0.0

        return max(0.0, min(1.0, passed / total))
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


def calculate_format_reward(completion: Any) -> float:
    """Return 1.0 if completion contains a fenced Python code block, else 0.0.

    Format-only: no execution and no syntax parsing.
    Rule: allow any preamble, but there must be a single Python fenced
    code block at the end of the text (no content after the closing fence).
    """
    text = extract_text_from_completion(completion)

    # Reward if the LAST thing in the text is a python fenced block.
    # We permit any preamble (e.g., thinking), but forbid any content after the fence.
    end_anchored = re.compile(r"```python\s*\n[\s\S]*?\n?```$", re.IGNORECASE)
    return 1.0 if end_anchored.search((text or "").rstrip()) else 0.0
