import json
import re
import sys
from collections import Counter
from pathlib import Path

# Usage: uv run python scripts/analyse_gpt35_failures.py gpt35_failed_tests.jsonl

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyse_gpt35_failures.py <jsonl_path>")
        sys.exit(2)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)

    pat_func = re.compile(r"assert\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    pat_def = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

    cats: Counter[str] = Counter()
    ids_by_cat: dict[str, list[int]] = {}

    def add(cat: str, i: int) -> None:
        cats[cat] += 1
        ids_by_cat.setdefault(cat, []).append(i)

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            i = d.get("id")
            code: str = d.get("model_response", "")
            tests: list[str] = d.get("test_cases", [])
            tr = d.get("tests_reward")
            cr = d.get("compile_reward")

            # Compile failures
            if cr != 1.0:
                add("compile_error", i)
                continue

            # Expected function names vs defined
            exp: set[str] = set()
            for t in tests:
                m = pat_func.search(t)
                if m:
                    exp.add(m.group(1))
            defs = set(pat_def.findall(code))
            if exp and defs and not (defs & exp):
                add("name_mismatch", i)
                continue

            # print vs return
            if "print(" in code and "return" not in code:
                add("print_vs_return", i)
                continue

            # Missing imports (simple heuristics)
            needs = []
            if "math." in code and "import math" not in code:
                needs.append("math")
            if "heapq." in code and "import heapq" not in code:
                needs.append("heapq")
            if needs:
                add("missing_import", i)
                continue

            # Type/format mismatch heuristics
            tests_expect_numeric = any(re.search(r"==\s*-?\d", t) for t in tests)
            returns_string_literal = bool(re.search(r"return\s*(['\"])", code))
            if tests_expect_numeric and returns_string_literal:
                add("type_format_mismatch", i)
                continue

            # tuple/list vs expected tuple literals in tests
            tests_expect_tuple = any("(" in t and ")" in t and "[" not in t for t in tests)
            returns_list_literal = "[" in code and "]" in code and "tuple(" not in code
            if tests_expect_tuple and returns_list_literal:
                add("type_format_mismatch", i)
                continue

            # Ordering vs set-like
            if "set(" in code:
                add("ordering_mismatch", i)
                continue

            # Otherwise: algorithm/spec gap
            add("algo_spec", i)

    print("COUNTS")
    for k, v in cats.most_common():
        print(f"{k}: {v}")

    print("\nEXAMPLES")
    for k in [
        "name_mismatch",
        "type_format_mismatch",
        "print_vs_return",
        "missing_import",
        "ordering_mismatch",
        "compile_error",
        "algo_spec",
    ]:
        if k in ids_by_cat:
            sample = sorted(ids_by_cat[k])[:8]
            print(f"{k}: {sample}")

if __name__ == "__main__":
    main()
