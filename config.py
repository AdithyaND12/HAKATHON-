"""Centralized environment configuration for the HAKATHON chatbot.

All env parsing lives here so `app.py` and `ragtool.py` share one source of truth.
Blank env values are treated as unset. Numeric envs fall back to safe defaults
when they cannot be parsed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Defaults -----------------------------------------------------------------

DEFAULT_LM_STUDIO_MODEL = "qwen2.5-coder-7b-instruct"
DEFAULT_LM_STUDIO_EMBEDDING_MODEL = "nomic-embed-text-v1.5"
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_WAIT_MAX_SECONDS = 3600.0
DEFAULT_MAX_AUTO_RUNS = 20
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / ".hakathon"
DEFAULT_PDF_PATH = Path(__file__).resolve().parent / "pdfs" / "c9fe9c9b6840524844316f74bb1c556c.pdf"

_PROJECT_ROOT = Path(__file__).resolve().parent


# ---- Helpers ------------------------------------------------------------------


def env_str(name: str, default: str = "") -> str:
    """Read a trimmed env value, treating blank/whitespace as unset."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value <= 0:
        return default
    return value


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return value


def env_path(name: str, default: Path) -> Path:
    raw = env_str(name, "")
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


# ---- Public config ------------------------------------------------------------

LM_STUDIO_MODEL = env_str("LM_STUDIO_MODEL", DEFAULT_LM_STUDIO_MODEL)
LM_STUDIO_EMBEDDING_MODEL = env_str(
    "LM_STUDIO_EMBEDDING_MODEL", DEFAULT_LM_STUDIO_EMBEDDING_MODEL
)
LM_STUDIO_BASE_URL = env_str("LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL)
LM_STUDIO_API_KEY = env_str("LM_STUDIO_API_KEY", DEFAULT_LM_STUDIO_API_KEY)
ALPHAVANTAGE_API_KEY = env_str("ALPHAVANTAGE_API_KEY")
HTTP_TIMEOUT_SECONDS = env_float("HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT_SECONDS)
WAIT_MAX_SECONDS = env_float("WAIT_MAX_SECONDS", DEFAULT_WAIT_MAX_SECONDS)
MAX_AUTO_RUNS = env_int("MAX_AUTO_RUNS", DEFAULT_MAX_AUTO_RUNS)
DATA_DIR = env_path("HAKATHON_DATA_DIR", DEFAULT_DATA_DIR)
CONSTITUTION_PDF_PATH = env_path("CONSTITUTION_PDF_PATH", DEFAULT_PDF_PATH)
DUCKDUCKGO_REGION = env_str("DUCKDUCKGO_REGION", "us-en")
SEARCH_MAX_RETRIES = env_int("SEARCH_MAX_RETRIES", 3)
SEARCH_RETRY_BACKOFF_SECONDS = env_float("SEARCH_RETRY_BACKOFF_SECONDS", 2.0)


def ensure_data_dir() -> Path:
    """Create the data directory if it does not exist and return it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
