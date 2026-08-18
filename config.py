"""Centralized environment configuration for the HAKATHON chatbot."""

from __future__ import annotations

import math
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Defaults -----------------------------------------------------------------

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_JINA_EMBEDDING_MODEL = "jina-embeddings-v4"
_RETIRED_GEMINI_MODELS = {
    "gemini-1.5-flash": "gemini-3.5-flash-lite",
    "models/gemini-1.5-flash": "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
    "models/gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-2.5-flash": "gemini-3.5-flash-lite",
    "models/gemini-2.5-flash": "gemini-3.5-flash-lite",
}
# User-selectable chat models (label -> model id), in display order.
GEMINI_MODEL_OPTIONS = {
    "Gemini 3.5 Flash-Lite": "gemini-3.5-flash-lite",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite",
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemma 4 31B IT": "gemma-4-31b-it",
    "Gemma 4 26B MoE IT": "gemma-4-26b-a4b-it",
}
DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_WAIT_MAX_SECONDS = 3600.0
DEFAULT_MAX_AUTO_RUNS = 20
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / ".hakathon"

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

_legacy_lm_studio_model = env_str("LM_STUDIO_MODEL", "")
_legacy_lm_studio_api_key = env_str("LM_STUDIO_API_KEY", "")

_configured_model = env_str("GEMINI_MODEL", _legacy_lm_studio_model or DEFAULT_GEMINI_MODEL)
GEMINI_MODEL = _RETIRED_GEMINI_MODELS.get(_configured_model, _configured_model)

# Embeddings come from Jina's free API; the chat LLM stays Gemini.
JINA_API_KEY = env_str("JINA_API_KEY")
JINA_EMBEDDING_MODEL = env_str("JINA_EMBEDDING_MODEL", DEFAULT_JINA_EMBEDDING_MODEL)
GEMINI_API_KEY = env_str(
    "GEMINI_API_KEY",
    env_str("GOOGLE_API_KEY", _legacy_lm_studio_api_key),
)

# Backward-compatible aliases for older imports.
LM_STUDIO_MODEL = GEMINI_MODEL
LM_STUDIO_API_KEY = GEMINI_API_KEY
LM_STUDIO_BASE_URL = "Gemini API"
ALPHAVANTAGE_API_KEY = env_str("ALPHAVANTAGE_API_KEY")
HTTP_TIMEOUT_SECONDS = env_float("HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT_SECONDS)
WAIT_MAX_SECONDS = env_float("WAIT_MAX_SECONDS", DEFAULT_WAIT_MAX_SECONDS)
MAX_AUTO_RUNS = env_int("MAX_AUTO_RUNS", DEFAULT_MAX_AUTO_RUNS)
DATA_DIR = env_path("HAKATHON_DATA_DIR", DEFAULT_DATA_DIR)
# Chunks per embedding request to Jina (one HTTP POST per batch). Batches of
# ~100 keep request counts low on the free tier (1M tokens/day, 100 RPM).
JINA_EMBEDDING_BATCH_SIZE = env_int("JINA_EMBEDDING_BATCH_SIZE", 100)
DUCKDUCKGO_REGION = env_str("DUCKDUCKGO_REGION", "us-en")
SEARCH_MAX_RETRIES = env_int("SEARCH_MAX_RETRIES", 3)
SEARCH_RETRY_BACKOFF_SECONDS = env_float("SEARCH_RETRY_BACKOFF_SECONDS", 2.0)


def ensure_data_dir() -> Path:
    """Create the data directory if it does not exist and return it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
