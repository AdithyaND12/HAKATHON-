"""Tests for the optional Waggle MCP memory integration (waggle_tools.py).

All MCP interactions are mocked — nothing here spawns the waggle-mcp
subprocess or touches a real graph DB, so the suite stays hermetic.
"""

from types import SimpleNamespace

import pytest

import config
import waggle_tools
from waggle_tools import WAGGLE_MEMORY_POLICY, WAGGLE_READ_TOOL_NAMES


def _fake_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description="", args_schema=None)


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    waggle_tools._tools_cache = None
    waggle_tools._raw_tools_cache = None
    yield


@pytest.fixture
def enable_waggle(monkeypatch):
    monkeypatch.setattr(config, "WAGGLE_MCP_ENABLED", True)
    yield


def test_init_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(config, "WAGGLE_MCP_ENABLED", False)
    assert waggle_tools.init_waggle_tools() == []


def test_select_tools_keeps_only_read_surface():
    raw = [
        _fake_tool("query_graph"),
        _fake_tool("prime_context"),
        _fake_tool("get_stats"),
        _fake_tool("observe_conversation"),
        _fake_tool("store_node"),
        _fake_tool("decompose_and_store"),
        _fake_tool("clear_session"),
    ]
    selected = waggle_tools._select_tools(raw)
    assert {t.name for t in selected} == {"query_graph", "prime_context", "get_stats"}
    # Write tools must never be exposed to the agent.
    assert all(name not in WAGGLE_READ_TOOL_NAMES for name in ("observe_conversation", "store_node"))


def test_init_returns_tools_when_host_ready(enable_waggle, monkeypatch):
    async def fake_host(ready, teardown):
        ready.set_result([_fake_tool("query_graph")])

    monkeypatch.setattr(waggle_tools, "_host", fake_host)
    assert [t.name for t in waggle_tools.init_waggle_tools()] == ["query_graph"]
    assert waggle_tools.init_waggle_tools()[0].name == "query_graph"  # cached


def test_init_fails_open_on_error(enable_waggle, monkeypatch):
    async def boom(ready, teardown):
        ready.set_exception(RuntimeError("waggle-mcp missing"))

    monkeypatch.setattr(waggle_tools, "_host", boom)
    assert waggle_tools.init_waggle_tools() == []


def test_matches_tolerates_server_prefix():
    assert waggle_tools._matches("query_graph")
    assert waggle_tools._matches("waggle.query_graph")
    assert waggle_tools._matches("prime_context")
    assert waggle_tools._matches("get_stats")
    assert not waggle_tools._matches("observe_conversation")
    assert not waggle_tools._matches("store_node")


def test_hooks_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "WAGGLE_MCP_ENABLED", False)
    assert waggle_tools.prime_session("conv-1") is None
    assert waggle_tools.memorize_turn("hi", "hello", "conv-1", block=True) is None


def test_hooks_noop_without_live_session(enable_waggle):
    assert waggle_tools.prime_session("conv-1") is None
    assert waggle_tools.memorize_turn("hi", "hello", "conv-1", block=True) is None


def test_memorize_turn_skips_empty_messages(enable_waggle, monkeypatch):
    calls = []

    async def fake_tool_ainvoke(arguments):
        calls.append(arguments)

    fake_tool = SimpleNamespace(ainvoke=fake_tool_ainvoke)
    waggle_tools._raw_tools_cache = {"observe_conversation": fake_tool}
    waggle_tools._ensure_loop_thread()

    waggle_tools.memorize_turn("", "reply", "conv-1", block=True)
    waggle_tools.memorize_turn("msg", "", "conv-1", block=True)
    assert calls == []


@pytest.mark.parametrize("tool_name,arguments", [
    ("observe_conversation", {"user_message": "msg", "assistant_response": "reply"}),
    ("prime_context", {}),
])
def test_hooks_route_calls_with_scope(
    enable_waggle, monkeypatch, tool_name: str, arguments: dict
):
    calls = []

    async def fake_tool_ainvoke(call_arguments):
        calls.append(call_arguments)

    fake_tool = SimpleNamespace(ainvoke=fake_tool_ainvoke)
    waggle_tools._raw_tools_cache = {tool_name: fake_tool}
    waggle_tools._ensure_loop_thread()

    if tool_name == "observe_conversation":
        waggle_tools.memorize_turn("msg", "reply", "conv-42", block=True)
    else:
        waggle_tools.prime_session("conv-42", block=True)

    assert len(calls) == 1
    call = calls[0]
    assert call.get("project") == config.WAGGLE_PROJECT
    assert call.get("agent_id") == config.WAGGLE_AGENT_ID
    assert call.get("session_id") == "conv-42"
    if tool_name == "observe_conversation":
        assert call.get("user_message") == "msg"
        assert call.get("assistant_response") == "reply"


def test_memorize_turn_missing_tool_logs_without_raising(enable_waggle, monkeypatch):
    waggle_tools._raw_tools_cache = {}
    waggle_tools._ensure_loop_thread()
    assert waggle_tools.memorize_turn("msg", "reply", "conv-1", block=True) is None


def test_memory_policy_mentions_query_graph_and_no_writes():
    assert "query_graph" in WAGGLE_MEMORY_POLICY
    assert "store_node" in WAGGLE_MEMORY_POLICY
    assert "NO write access" in WAGGLE_MEMORY_POLICY