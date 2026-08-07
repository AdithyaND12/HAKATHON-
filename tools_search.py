"""DuckDuckGo web-search tool with retry + backoff.

Wraps `DuckDuckGoSearchRun` so intermittent rate-limits or network hiccups do not
kill a scheduled job. Exposed as a LangChain tool so the LangGraph agent can call it.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import config

log = logging.getLogger(__name__)

_search_backend = DuckDuckGoSearchRun(region=config.DUCKDUCKGO_REGION)


def _run_with_retries(query: str, max_retries: int, backoff: float) -> str:
    last_error: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            return _search_backend.invoke(query)
        except Exception as exc:  # noqa: BLE001 - DDG raises many exception types
            last_error = exc
            if attempt == max_retries:
                break
            sleep_for = backoff * (2 ** (attempt - 1))
            log.warning(
                "web_search attempt %d/%d failed: %s. Retrying in %.1fs.",
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    return f"[web_search failed after {max_retries} attempts: {last_error}]"


@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo. Retries with exponential backoff on failure.

    Use this when the user's request needs current or web-based information.
    """
    return _run_with_retries(
        query,
        max_retries=config.SEARCH_MAX_RETRIES,
        backoff=config.SEARCH_RETRY_BACKOFF_SECONDS,
    )


# Backwards-compatible export used by existing tool binding.
search_tool = web_search
