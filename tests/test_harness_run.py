import sys
import subprocess
import tempfile
import re
import json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environments'))
from mbpp_baseline.util.mbpp_utils import build_fractional_test_harness


def _run_script_returncode(script: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
    return proc.returncode


def _parse_json_output(script: str) -> dict | None:
    """Run script and parse RESULT_JSON output."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
    matches = re.findall(r"RESULT_JSON:\s*(\{.*\})", proc.stdout)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass
    return None


def test_harness_failure_returns_zero_with_json():
    sol = "def f(x):\n    return x+1"
    script = build_fractional_test_harness(sol, tests=["assert f(1)==3"])
    # Fractional harness always returns 0 (success) because it catches exceptions
    assert _run_script_returncode(script) == 0

    # Test JSON output shows 0 passed
    result = _parse_json_output(script)
    assert result == {"passed": 0, "total": 1}


def test_fractional_reward_calculation():
    """Test that we can calculate fractional rewards from partial test success."""
    sol = "def f(x):\n    return x+1"
    tests = ["assert f(1)==2", "assert f(5)==6", "assert f(10)==12", "assert f(0)==1"]  # 3/4 pass
    script = build_fractional_test_harness(sol, tests=tests)

    result = _parse_json_output(script)
    assert result == {"passed": 3, "total": 4}

    # Fractional reward should be 0.75
    frac = result["passed"] / result["total"]
    assert frac == 0.75

