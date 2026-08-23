"""Tests for the optional Gmail MCP integration (gmail_tools.py).

All MCP/network interactions are mocked — nothing here spawns Node or touches
the real server, so the suite stays hermetic.
"""

from types import SimpleNamespace

import pytest

import config
import gmail_tools
from gmail_tools import GMAIL_TOOL_NAMES


def _fake_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description="", args_schema=None)


@pytest.fixture
def enable_gmail(monkeypatch):
    monkeypatch.setattr(config, "GMAIL_MCP_ENABLED", True)
    yield


def test_init_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(config, "GMAIL_MCP_ENABLED", False)
    assert gmail_tools.init_gmail_tools() == []


def test_select_tools_keeps_read_and_mailbox_write_subset():
    raw = [
        _fake_tool("gmail.list_messages"),
        _fake_tool("gmail.get_message"),
        _fake_tool("gmail.get_profile"),
        _fake_tool("gmail.send_message"),
        _fake_tool("gmail.create_draft"),
        _fake_tool("gmail.modify_message"),
        _fake_tool("gmail.delete_message"),
        _fake_tool("gmail.update_imap"),
        _fake_tool("gmail.delete_filter"),
        _fake_tool("gmail.update_vacation"),
    ]
    selected = gmail_tools._select_tools(raw)
    assert {t.name for t in selected} == {
        "gmail.list_messages",
        "gmail.get_message",
        "gmail.get_profile",
        "gmail.send_message",
        "gmail.create_draft",
        "gmail.modify_message",
        "gmail.delete_message",
    }
    # Settings endpoints must never be exposed.
    assert all("update_imap" not in t.name and "delete_filter" not in t.name for t in selected)


def test_init_returns_tools_when_host_ready(enable_gmail, monkeypatch):
    gmail_tools._tools_cache = None

    async def fake_host(ready, teardown):
        ready.set_result([_fake_tool("list_messages")])

    monkeypatch.setattr(gmail_tools, "_host", fake_host)
    assert [t.name for t in gmail_tools.init_gmail_tools()] == ["list_messages"]
    assert gmail_tools.init_gmail_tools()[0].name == "list_messages"  # cached


def test_init_fails_open_on_error(enable_gmail, monkeypatch):
    gmail_tools._tools_cache = None

    async def boom(ready, teardown):
        ready.set_exception(RuntimeError("node missing"))

    monkeypatch.setattr(gmail_tools, "_host", boom)
    assert gmail_tools.init_gmail_tools() == []


def test_matches_tolerates_server_prefix():
    assert gmail_tools._matches("list_messages")
    assert gmail_tools._matches("gmail.list_messages")
    assert gmail_tools._matches("gmail.send_message")
    assert gmail_tools._matches("gmail.batch_delete_messages")
    assert not gmail_tools._matches("gmail.update_imap")
    assert not gmail_tools._matches("gmail.update_vacation")


def test_planner_search_instruction_mentions_gmail_tools():
    from planner import SearchExecutionInstruction

    text = SearchExecutionInstruction("my inbox").render()
    assert "list_messages" in text
    assert "web search tool" in text
    assert "never display them" in text
    assert "JSON code blocks" in text