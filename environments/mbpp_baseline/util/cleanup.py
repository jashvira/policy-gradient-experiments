"""Code cleaning utilities for MBPP evaluation."""

import re


def clean_code_main_block(code: str) -> str:
    """Remove `if __name__ == "__main__":` blocks from the provided code.

    Keeps user-defined functions while dropping ad-hoc test harnesses that can
    interfere with judging.

    Args:
        code: The input Python code.

    Returns:
        Cleaned code without the main execution block.
    """
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

        # We are skipping lines belonging to the main block. Continue skipping
        # while the line is indented deeper than the detected base indentation.
        # Blank lines inside the block are skipped as well.
        if line.strip() == "":
            continue
        if base_indent is not None and (line.startswith(base_indent + " ") or line.startswith(base_indent + "\t")):
            continue

        # Dedented: end of main block. Resume copying lines.
        skipping = False
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
