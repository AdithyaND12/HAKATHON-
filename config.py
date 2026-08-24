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
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_PROVIDER = "jina"
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
JINA_EMBEDDING_BATCH_SIZE = env_int("JINA_EMBEDDING_BATCH_SIZE", 32)

# Embedding provider: "jina" (default, free tier) or "google" (Gemini embeddings).
EMBEDDING_PROVIDER = env_str("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)
GOOGLE_EMBEDDING_MODEL = env_str("GOOGLE_EMBEDDING_MODEL", DEFAULT_GEMINI_EMBEDDING_MODEL)
GOOGLE_EMBEDDING_BATCH_SIZE = env_int("GOOGLE_EMBEDDING_BATCH_SIZE", 32)

DUCKDUCKGO_REGION = env_str("DUCKDUCKGO_REGION", "us-en")
SEARCH_MAX_RETRIES = env_int("SEARCH_MAX_RETRIES", 3)
SEARCH_RETRY_BACKOFF_SECONDS = env_float("SEARCH_RETRY_BACKOFF_SECONDS", 2.0)

# Gmail MCP integration (inbox read + mailbox write tools). Disabled by
# default so the test suite never spawns a Node subprocess; set
# GMAIL_MCP_ENABLED=true to opt in.
GMAIL_MCP_ENABLED = env_str("GMAIL_MCP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
# Locally the cloned gmail-mcp build is used; elsewhere (e.g. Streamlit Cloud,
# where that path does not exist) fall back to the published npm package via
# npx. Explicit GMAIL_MCP_COMMAND / GMAIL_MCP_ARGS always win.
_GMAIL_MCP_LOCAL_DIST = Path("/Users/adithya/agent2/gmail-mcp/dist/index.js")
if _GMAIL_MCP_LOCAL_DIST.exists():
    GMAIL_MCP_COMMAND = env_str("GMAIL_MCP_COMMAND", "node")
    GMAIL_MCP_ARGS = env_str("GMAIL_MCP_ARGS", str(_GMAIL_MCP_LOCAL_DIST))
else:
    GMAIL_MCP_COMMAND = env_str("GMAIL_MCP_COMMAND", "npx")
    GMAIL_MCP_ARGS = env_str("GMAIL_MCP_ARGS", "-y @shinzolabs/gmail-mcp")
# The gmail-mcp server always binds its HTTP listener; port "0" lets the OS
# assign a free one so instances never clash (a fixed value would collide
# with orphaned servers). Override only when you know no other instance runs.
GMAIL_MCP_PORT = env_str("GMAIL_MCP_PORT", "0")

# Per-user Gmail OAuth 2.0 (direct Google API, not gmail-mcp).
# Each visitor authenticates with their own Google account.
GOOGLE_OAUTH_CLIENT_ID = env_str("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET = env_str("GOOGLE_OAUTH_CLIENT_SECRET")
GOOGLE_OAUTH_REDIRECT_URI = env_str(
    "GOOGLE_OAUTH_REDIRECT_URI",
    "http://localhost:8501",
)
GOOGLE_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]
# Directory for per-user token files (.hakathon/gmail_tokens/{session_id}.json)
GMAIL_TOKENS_DIR = DATA_DIR / "gmail_tokens"

# Waggle MCP memory integration: persistent graph memory (decisions,
# preferences, project facts) shared across conversations and sessions.
# Disabled by default so the test suite never spawns a subprocess; set
# WAGGLE_MCP_ENABLED=true to opt in. The memory DB lives under the data dir.
WAGGLE_MCP_ENABLED = env_str("WAGGLE_MCP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
WAGGLE_MCP_COMMAND = env_str("WAGGLE_MCP_COMMAND", "waggle-mcp")
WAGGLE_MCP_ARGS = env_str("WAGGLE_MCP_ARGS", "")
# Embeddings ride on the same free Jina API as the PDF RAG (JINA_API_KEY
# above), so no local model download happens. "pytorch" / "onnx" / "jina"
# are supported; "deterministic" needs no key at all (weaker retrieval).
WAGGLE_EMBEDDING_BACKEND = env_str("WAGGLE_EMBEDDING_BACKEND", "jina")
WAGGLE_MODEL = env_str("WAGGLE_MODEL", "jina-embeddings-v4")
WAGGLE_EMBEDDING_DIMENSIONS = env_int("WAGGLE_EMBEDDING_DIMENSIONS", 768)
WAGGLE_DB_PATH = env_path("WAGGLE_DB_PATH", DATA_DIR / "waggle" / "memory.db")
# Memory scope used for every recall/store; keeps HAKATHON memory separate
# from any other Waggle tenants on this machine.
WAGGLE_PROJECT = env_str("WAGGLE_PROJECT", "hakathon")
WAGGLE_AGENT_ID = env_str("WAGGLE_AGENT_ID", "hakathon-chat")
# Cap per-call timeouts across the bridge thread. observe_conversation runs a
# local LLM extraction step and can be slow on first use (model warm-up).
WAGGLE_CALL_TIMEOUT_SECONDS = env_float("WAGGLE_CALL_TIMEOUT_SECONDS", 300.0)
WAGGLE_OUTPUT_MAX_CHARS = env_int("WAGGLE_OUTPUT_MAX_CHARS", 4000)


def ensure_data_dir() -> Path:
    """Create the data directory if it does not exist and return it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
