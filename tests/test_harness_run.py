import sys
import subprocess
import tempfile
from environments.mbpp_baseline.util.mbpp_utils import build_test_harness


def _run_script_returncode(script: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
    return proc.returncode


def test_harness_success_returns_zero():
    sol = "def f(x):\n    return x+1"
    script = build_test_harness(sol, tests=["assert f(1)==2"])
    assert _run_script_returncode(script) == 0


def test_harness_failure_returns_nonzero():
    sol = "def f(x):\n    return x+1"
    script = build_test_harness(sol, tests=["assert f(1)==3"])
    assert _run_script_returncode(script) != 0


def test_harness_multiple_tests_all_pass():
    sol = "def f(x):\n    return x+1"
    tests = ["assert f(1)==2", "assert f(5)==6", "assert f(10)==11"]
    script = build_test_harness(sol, tests=tests)
    assert _run_script_returncode(script) == 0


def test_harness_multiple_tests_one_fails():
    sol = "def f(x):\n    return x+1"
    tests = ["assert f(1)==2", "assert f(5)==6", "assert f(10)==12"]  # last fails
    script = build_test_harness(sol, tests=tests)
    assert _run_script_returncode(script) != 0

