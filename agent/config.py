"""Agent configuration loaded from environment variables.

Values are read from the process environment first. If they are not present,
the repository-root `.env` file is loaded as a local fallback.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def _load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _get_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


_load_dotenv()

API_KEY = os.environ.get("MRI_AGENT_API_KEY", "")
BASE_URL = os.environ.get("MRI_AGENT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("MRI_AGENT_MODEL", "qwen3.5-plus")
TEMPERATURE = _get_float("MRI_AGENT_TEMPERATURE", 0.7)
MAX_TOKENS = _get_int("MRI_AGENT_MAX_TOKENS", 2048)


def require_api_key() -> str:
    """Return the configured API key or raise a clear setup error."""
    if not API_KEY:
        raise RuntimeError("MRI_AGENT_API_KEY is not configured. Set it in the root .env file or environment.")
    return API_KEY
