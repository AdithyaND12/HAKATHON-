import os
import sys
from pathlib import Path

# The app builds its Gemini client at import time, so the suite needs a
# syntactically valid key even though no network call is made. Tests that
# exercise real requests monkeypatch the client, so this only unblocks import.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("JINA_API_KEY", "test-key")

# Keep the suite hermetic: never spawn the gmail-mcp server during tests,
# even when the developer's .env has GMAIL_MCP_ENABLED=true. This must run
# before any test module imports `app` (dotenv won't override it).
os.environ["GMAIL_MCP_ENABLED"] = "false"
# Same for waggle-mcp memory: tests never spawn the waggle-mcp subprocess.
os.environ["WAGGLE_MCP_ENABLED"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))