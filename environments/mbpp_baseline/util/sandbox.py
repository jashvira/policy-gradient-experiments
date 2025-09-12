"""SandboxFusion integration for code execution rewards.

Provides sharded code execution with connection pooling.
"""

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter

# Constants
DEFAULT_SANDBOX_URL = "http://localhost:8080"
DEFAULT_TIMEOUT = 8
POOL_CONNECTIONS = 128
POOL_MAX_SIZE = 512
ERROR_STATUS = "Error"

# Response keys
STATUS_KEY = "status"
ERROR_KEY = "error"

logger = logging.getLogger(__name__)


class SandboxConfig:
    """Configuration for sandbox connections."""

    def __init__(self):
        self.urls = self._parse_sandbox_urls()

    def _parse_sandbox_urls(self) -> List[str]:
        """Parse sandbox URLs from environment variables."""
        env_urls = os.getenv("SANDBOX_URLS")
        if env_urls:
            return [url.strip() for url in env_urls.split(",") if url.strip()]
        return [os.getenv("SANDBOX_URL", DEFAULT_SANDBOX_URL).strip()]


class ConnectionManager:
    """Manages HTTP connections with pooling."""

    def __init__(self):
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=POOL_CONNECTIONS,
            pool_maxsize=POOL_MAX_SIZE,
            max_retries=0
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)


class LoadBalancer:
    """Thread-safe round-robin load balancer."""

    def __init__(self, urls: List[str]):
        self.urls = urls
        self._index = 0
        self._lock = threading.Lock()

    def get_shard_order(self) -> List[str]:
        """Get shards in round-robin order starting from next index."""
        with self._lock:
            start_idx = self._index
            self._index = (self._index + 1) % len(self.urls)

        return [self.urls[(start_idx + i) % len(self.urls)] for i in range(len(self.urls))]


class SandboxException(Exception):
    """Exception raised when sandbox execution fails."""
    pass


# Global instances
_config = SandboxConfig()
_connection_manager = ConnectionManager()
_load_balancer = LoadBalancer(_config.urls)


def _get_effective_timeout(user_timeout: int) -> float:
    """Get effective timeout from environment or user preference."""
    try:
        return float(os.getenv("MBPP_SANDBOX_TIMEOUT", user_timeout))
    except (ValueError, TypeError):
        return float(user_timeout)


def run_code(code: str, language: str = "python", timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Execute code via SandboxFusion API with sharding.

    Args:
        code: Source code to execute
        language: Programming language (default: python)
        timeout: Request timeout in seconds

    Returns:
        Dictionary with 'status' and either 'output' (success) or 'error' (failure)

    Environment overrides:
        SANDBOX_URLS: Comma-separated shard URLs
        SANDBOX_URL: Single sandbox URL (fallback)
        MBPP_SANDBOX_TIMEOUT: Override request timeout
    """
    if not code.strip():
        return {STATUS_KEY: ERROR_STATUS, ERROR_KEY: "Empty code"}

    effective_timeout = _get_effective_timeout(timeout)

    # Try all shards in round-robin order
    last_error: Optional[str] = None
    for shard_url in _load_balancer.get_shard_order():
        try:
            result = _execute_on_shard(shard_url, code, language, effective_timeout)
            return result
        except SandboxException as e:
            last_error = str(e)
            continue

    return {STATUS_KEY: ERROR_STATUS, ERROR_KEY: last_error or "All shards failed"}


def _execute_on_shard(shard_url: str, code: str, language: str, timeout: float) -> Dict[str, Any]:
    """Execute code on a specific shard."""
    url = f"{shard_url}/run_code"

    try:
        response = _connection_manager.session.post(
            url,
            json={"code": code, "language": language},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout as e:
        logger.warning(f"Sandbox timeout: {url} (timeout={timeout}s)")
        raise SandboxException(f"Timeout: {url}")

    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Sandbox unreachable: {shard_url}")
        raise SandboxException(f"Connection failed: {shard_url}")

    except requests.exceptions.HTTPError as e:
        logger.error(f"Sandbox HTTP error: {e}")
        raise SandboxException(f"HTTP error: {e}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from sandbox: {url}")
        raise SandboxException(f"Invalid JSON response: {url}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Sandbox request error: {e}")
        raise SandboxException(f"Request failed: {e}")

