"""SandboxFusion integration for code execution rewards.

Adds sharding (SANDBOX_URLS), connection pooling, and a small LRU cache.
"""

import json
import logging
import os
import threading
from collections import OrderedDict
from hashlib import sha256
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

# Parse shard URLs from env. Fallback to single SANDBOX_URL
_env_urls = os.getenv("SANDBOX_URLS")
if _env_urls:
    SANDBOX_URLS: List[str] = [u.strip() for u in _env_urls.split(",") if u.strip()]
else:
    SANDBOX_URLS = [os.getenv("SANDBOX_URL", "http://localhost:8080").strip()]

# Connection pooling: one session shared across hosts
_session = requests.Session()
_session.mount("http://", HTTPAdapter(pool_connections=128, pool_maxsize=512, max_retries=0))
_session.mount("https://", HTTPAdapter(pool_connections=128, pool_maxsize=512, max_retries=0))

# Round-robin index for shards
_rr_index = 0
_rr_lock = threading.Lock()


def _pick_start_index() -> int:
    global _rr_index
    with _rr_lock:
        idx = _rr_index
        _rr_index = (_rr_index + 1) % len(SANDBOX_URLS)
        return idx


def _effective_timeout(user_timeout: int) -> float:
    try:
        return float(os.getenv("MBPP_SANDBOX_TIMEOUT", user_timeout))
    except Exception:
        return float(user_timeout)


# Simple LRU cache (status-aware):
# - Cache only results with status == "Success" (deterministic for same code)
_CACHE_MAXSIZE = int(os.getenv("SANDBOX_CACHE_SIZE", "2048"))
_cache_lock = threading.Lock()
_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _cache_key(code: str, language: str) -> str:
    return sha256(f"{language}:{code}".encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Dict[str, Any] | None:
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        return None


def _cache_put(key: str, value: Dict[str, Any]) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAXSIZE:
            _cache.popitem(last=False)


def run_code(code: str, language: str = "python", timeout: int = 8) -> dict:
    """Execute code via SandboxFusion API with sharding and caching.

    Environment overrides:
      - SANDBOX_URLS: comma-separated list of base URLs. Example: "http://localhost:8080,http://localhost:8081"
      - SANDBOX_URL: single base URL (fallback)
      - MBPP_SANDBOX_TIMEOUT: float seconds to override per-request timeout
      - SANDBOX_CACHE_SIZE: LRU cache size (default 2048)
    """
    if not code:
        return {"status": "Error", "error": "Empty code"}

    key = _cache_key(code, language)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    effective_timeout = _effective_timeout(timeout)

    # Try each shard starting from round-robin index
    start = _pick_start_index()
    shard_order = [
        SANDBOX_URLS[(start + i) % len(SANDBOX_URLS)] for i in range(len(SANDBOX_URLS))
    ]

    last_error: str | None = None
    for base in shard_order:
        url = f"{base}/run_code"
        try:
            response = _session.post(
                url,
                json={"code": code, "language": language},
                timeout=effective_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            # Cache only successful executions (deterministic for identical inputs)
            if isinstance(payload, dict) and payload.get("status") == "Success":
                _cache_put(key, payload)
            return payload
        except requests.exceptions.Timeout as e:
            last_error = f"Timeout contacting {url}: {e}"
            logger.warning(f"SandboxFusion timeout: {url} (timeout={effective_timeout})")
            continue
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error contacting {url}: {e}"
            logger.warning(f"SandboxFusion not reachable at {base}")
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"Request error contacting {url}: {e}"
            logger.error(f"SandboxFusion request error: {e}")
            continue
        except json.JSONDecodeError:
            last_error = f"Invalid JSON from {url}"
            logger.error("SandboxFusion invalid JSON response")
            continue

    # If all shards failed, return a unified error
    return {"status": "Error", "error": last_error or "All SandboxFusion shards failed"}

