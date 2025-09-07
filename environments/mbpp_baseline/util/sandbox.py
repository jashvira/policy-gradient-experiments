"""SandboxFusion integration for code execution rewards."""

import json
import logging
import os
import requests

logger = logging.getLogger(__name__)

SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080")


def run_code(code: str, language: str = "python", timeout: int = 8) -> dict:
    """Execute code via SandboxFusion API.

    Args:
        code: Code to execute
        language: Programming language (default: "python")
        timeout: Request timeout in seconds

    Returns:
        Response dict with status, run_result, etc.
    """
    try:
        response = requests.post(
            f"{SANDBOX_URL}/run_code",
            json={"code": code, "language": language},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"SandboxFusion server not reachable at {SANDBOX_URL}")
        logger.warning("   Start it with: sudo docker run -d -p 8080:8080 volcengine/sandbox-fusion:server-20250609")
        return {"status": "Error", "error": f"Sandbox server not running: {e}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"SandboxFusion connection error: {e}")
        return {"status": "Error", "error": str(e)}
    except json.JSONDecodeError as e:
        logger.error(f"SandboxFusion invalid JSON response")
        return {"status": "Error", "error": "Invalid JSON response"}

