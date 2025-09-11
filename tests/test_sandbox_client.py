import os
import json
import sys
import importlib
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("SANDBOX_URLS", raising=False)
    monkeypatch.delenv("SANDBOX_URL", raising=False)
    monkeypatch.delenv("MBPP_SANDBOX_TIMEOUT", raising=False)
    monkeypatch.delenv("SANDBOX_CACHE_SIZE", raising=False)


def _make_response(status_code=200, payload=None):
    class Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload if payload is not None else {}

        def raise_for_status(self):
            if not (200 <= self.status_code < 300):
                raise Exception("HTTP error")

        def json(self):
            return self._payload

    return Resp(status_code, payload)


def _fresh_sandbox_module():
    sys.modules.pop("environments.mbpp_baseline.util.sandbox", None)
    from environments.mbpp_baseline.util import sandbox as sb  # type: ignore
    importlib.reload(sb)
    return sb


def test_single_url_success(monkeypatch):
    monkeypatch.setenv("SANDBOX_URL", "http://sandbox:8080")

    sb = _fresh_sandbox_module()

    with mock.patch.object(sb, "_session") as sess:
        sess.post.return_value = _make_response(
            200, {"status": "Success", "run_result": {"return_code": 0}}
        )
        out = sb.run_code("print(1)")
        assert out["status"] == "Success"
        assert out["run_result"]["return_code"] == 0


def test_shard_fallback(monkeypatch):
    monkeypatch.setenv(
        "SANDBOX_URLS",
        "http://bad:8080,http://good:8081",
    )
    sb = _fresh_sandbox_module()

    def side_effect(url, json=None, timeout=None):  # noqa: A002
        if url.startswith("http://bad:8080"):
            raise sb.requests.exceptions.ConnectionError("down")
        return _make_response(200, {"status": "Success", "run_result": {"return_code": 0}})

    with mock.patch.object(sb, "_session") as sess:
        sess.post.side_effect = side_effect
        out = sb.run_code("print(1)")
        assert out["status"] == "Success"


def test_timeout_override(monkeypatch):
    monkeypatch.setenv("MBPP_SANDBOX_TIMEOUT", "1.5")
    sb = _fresh_sandbox_module()

    with mock.patch.object(sb, "_session") as sess:
        def check_timeout(url, json=None, timeout=None):  # noqa: A002
            assert abs(timeout - 1.5) < 1e-6
            return _make_response(200, {"status": "Success", "run_result": {"return_code": 0}})

        sess.post.side_effect = check_timeout
        out = sb.run_code("print(1)")
        assert out["status"] == "Success"


def test_caches_only_success(monkeypatch):
    monkeypatch.setenv("SANDBOX_URL", "http://sandbox:8080")
    sb = _fresh_sandbox_module()

    calls = {"count": 0}

    def seq(url, json=None, timeout=None):  # noqa: A002
        calls["count"] += 1
        # First call returns error (should not cache); second returns success
        if calls["count"] == 1:
            return _make_response(200, {"status": "Error", "error": "boom"})
        return _make_response(200, {"status": "Success", "run_result": {"return_code": 0}})

    with mock.patch.object(sb, "_session") as sess:
        sess.post.side_effect = seq
        out1 = sb.run_code("print(1)")
        assert out1["status"] == "Error"
        out2 = sb.run_code("print(1)")
        assert out2["status"] == "Success"
        # Third call should be cached now
        out3 = sb.run_code("print(1)")
        assert out3["status"] == "Success"
        assert calls["count"] == 2


def test_all_shards_fail(monkeypatch):
    monkeypatch.setenv("SANDBOX_URLS", "http://a:1,http://b:2")
    sb = _fresh_sandbox_module()

    with mock.patch.object(sb, "_session") as sess:
        sess.post.side_effect = sb.requests.exceptions.Timeout("t")
        out = sb.run_code("print(1)")
        assert out["status"] == "Error"
        assert "failed" in out["error"].lower()


