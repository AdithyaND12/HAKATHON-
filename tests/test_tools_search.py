"""Tests for the DuckDuckGo tool wrapper configuration."""

from __future__ import annotations

import config
import tools_search


def test_configured_region_reaches_the_api_wrapper():
    assert tools_search._search_backend.api_wrapper.region == config.DUCKDUCKGO_REGION


def test_search_tool_is_exposed():
    assert tools_search.search_tool.name == "web_search"
