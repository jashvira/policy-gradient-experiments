from environments.mbpp_baseline.util.mbpp_utils import extract_code_from_completion, build_test_harness


def test_extract_code_from_gnarly_completion():
    # Real problematic completion format from OpenAI API
    completion = [{'role': 'assistant', 'content': '''```python
import math

def min_sum_of_factors(n: int) -> int:
    """Return the minimum possible sum a + b where a and b are positive integers"""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    root = math.isqrt(n)
    for a in range(root, 0, -1):
        if n % a == 0:
            return a + (n // a)
    return n + 1
```'''}]

    code = extract_code_from_completion(completion)
    assert len(code) > 200
    assert "def min_sum_of_factors" in code
    assert "import math" in code


def test_build_test_harness_with_real_mbpp():
    sol = "def find_Min_Sum(num):\n    sum = 0\n    return sum"
    tests = ["assert find_Min_Sum(12) == 7", "assert find_Min_Sum(2) == 2"]
    script = build_test_harness(sol, tests=tests)
    assert "def find_Min_Sum" in script
    assert "assert find_Min_Sum(12) == 7" in script


