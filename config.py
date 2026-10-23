"""Centralized environment configuration for the HAKATHON chatbot."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Defaults -----------------------------------------------------------------

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
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
# LLM calls need a much longer timeout than the stock API: Gemini routinely
# takes longer than 10s under load, and a short timeout triggers the SDK's
# retry loop (default 6 attempts), which multiplies every stall.
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_CLIENT_MAX_RETRIES = 2
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


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_path(name: str, default: Path) -> Path:
    raw = env_str(name, "")
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def streamlit_secret(name: str) -> str:
    """Read a root-level Streamlit secret when running under Streamlit.

    Community Cloud usually exposes root-level secrets as environment
    variables, but this also covers secrets pasted under a TOML section and
    local ``.streamlit/secrets.toml`` files.
    """
    if "streamlit" not in sys.modules:
        return ""
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return value.strip() if isinstance(value, str) and value.strip() else ""


# ---- Public config ------------------------------------------------------------

_legacy_lm_studio_model = env_str("LM_STUDIO_MODEL", "")
_legacy_lm_studio_api_key = env_str("LM_STUDIO_API_KEY", "")

_configured_model = env_str("GEMINI_MODEL", _legacy_lm_studio_model or DEFAULT_GEMINI_MODEL)
GEMINI_MODEL = _RETIRED_GEMINI_MODELS.get(_configured_model, _configured_model)

# A single Gemini key powers both chat and PDF embeddings. In the Streamlit
# UI the user can enter their own key at runtime via `set_gemini_api_key`.
GEMINI_API_KEY = (
    env_str("GEMINI_API_KEY", env_str("GOOGLE_API_KEY", _legacy_lm_studio_api_key))
    or streamlit_secret("GEMINI_API_KEY")
    or streamlit_secret("GOOGLE_API_KEY")
)


def set_gemini_api_key(api_key: str) -> None:
    """Set the Gemini API key for the current process at runtime.

    Used by the Streamlit UI when a user pastes their own key. Environment
    values are still the fallback for CLI/API deployments.
    """
    global GEMINI_API_KEY, LM_STUDIO_API_KEY
    GEMINI_API_KEY = (api_key or "").strip()
    LM_STUDIO_API_KEY = GEMINI_API_KEY

# Backward-compatible aliases for older imports.
LM_STUDIO_MODEL = GEMINI_MODEL
LM_STUDIO_API_KEY = GEMINI_API_KEY
LM_STUDIO_BASE_URL = "Gemini API"
ALPHAVANTAGE_API_KEY = env_str("ALPHAVANTAGE_API_KEY") or streamlit_secret(
    "ALPHAVANTAGE_API_KEY"
)
HTTP_TIMEOUT_SECONDS = env_float("HTTP_TIMEOUT_SECONDS", DEFAULT_HTTP_TIMEOUT_SECONDS)
LLM_TIMEOUT_SECONDS = env_float("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS)
LLM_CLIENT_MAX_RETRIES = env_int(
    "LLM_CLIENT_MAX_RETRIES", DEFAULT_LLM_CLIENT_MAX_RETRIES
)
WAIT_MAX_SECONDS = env_float("WAIT_MAX_SECONDS", DEFAULT_WAIT_MAX_SECONDS)
# Recurrence intervals are not bounded by the wait tool's limit: "every 6
# hours" or "daily" are valid schedules. Cap at one week to stay bounded.
MAX_SCHEDULE_INTERVAL_MINUTES = env_float(
    "MAX_SCHEDULE_INTERVAL_MINUTES", 7 * 24 * 60
)
MAX_AUTO_RUNS = env_int("MAX_AUTO_RUNS", DEFAULT_MAX_AUTO_RUNS)
DATA_DIR = env_path("HAKATHON_DATA_DIR", DEFAULT_DATA_DIR)
# Chunks per embedding request to Gemini (one HTTP POST per batch).
GOOGLE_EMBEDDING_BATCH_SIZE = env_int("GOOGLE_EMBEDDING_BATCH_SIZE", 32)

GOOGLE_EMBEDDING_MODEL = env_str("GOOGLE_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL)

DUCKDUCKGO_REGION = env_str("DUCKDUCKGO_REGION", "us-en")
SEARCH_MAX_RETRIES = env_int("SEARCH_MAX_RETRIES", 3)
SEARCH_RETRY_BACKOFF_SECONDS = env_float("SEARCH_RETRY_BACKOFF_SECONDS", 2.0)


def ensure_data_dir() -> Path:
    """Create the data directory if it does not exist and return it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
