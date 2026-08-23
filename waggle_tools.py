"""Optional Waggle MCP memory integration for the HAKATHON chatbot.

The waggle-mcp server (PyPI package, console script `waggle-mcp`) is started
as a stdio subprocess and its tools are adapted to LangChain tools via
mcp_tool_bridge (a minimal bridge over the raw mcp client; the
langchain-mcp-adapters package pins mcp<2.0.0 so it cannot be used).
The agent gets the *read* surface (query_graph,
prime_context, get_stats) so it can recall decisions, preferences, and
project facts from earlier conversations; *writes* are handled
deterministically by the runtime hooks `prime_session()` (start of a
conversation) and `memorize_turn()` (after every completed turn), which call
`prime_context` / `observe_conversation` on the MCP session.

The Waggle graph DB lives at config.WAGGLE_DB_PATH (default
.hakathon/waggle/memory.db); all memory is scoped to
project=config.WAGGLE_PROJECT, agent_id=config.WAGGLE_AGENT_ID so HAKATHON
memory never bleeds into other Waggle tenants on this machine.

Like gmail_tools.py, exactly ONE long-lived stdio session is kept on a
dedicated daemon event-loop thread; every call is bridged onto that loop as a
plain sync call, so the existing LangGraph ToolNode and the runtime hooks can
use it unchanged. Everything is fail-open: a missing server, failed
initialization, or a timed-out call logs a warning and never blocks the chat.

Memory-automation policy: if memory looks empty, the likely cause is that the
server is disabled (WAGGLE_MCP_ENABLED=false) or the DB is empty — the hooks
below are always invoked on the covered paths when the feature is on.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import os
import shlex
import threading
from typing import Optional

import config

logger = logging.getLogger(__name__)

# System-message policy telling the agent when to read memory. Memory writes
# happen automatically via the runtime hooks; the agent only reads.
WAGGLE_MEMORY_POLICY = """Waggle automatic memory policy

Waggle remembers durable facts across conversations automatically. The runtime
writes memory after every completed turn (via observe_conversation), and a
fresh briefing (prime_context) is issued at the start of each conversation, so
you usually do NOT need to trigger memory yourself.

Before answering:
- If the user's question may depend on prior decisions, preferences,
  constraints, project facts, or earlier conversation context, call
  query_graph with a natural-language query about that context (plus your
  conversation's session_id when known).
- Keep retrieval narrow: start with max_nodes 8-12 and max_depth 1-2.

You have NO write access to memory (store_node / observe_conversation are not
available to you); memory writes are handled automatically by the runtime.
Never claim you cannot remember anything — query_graph returns an
empty/advisory result for genuinely new topics, which is normal.
"""

# The agent-facing (read) subset of the Waggle endpoints. Write tools
# (observe_conversation, store_node, decompose_and_store, ...) are NOT exposed
# to the model — memory writes go through the deterministic runtime hooks.
WAGGLE_READ_TOOL_NAMES = frozenset(
    {
        "query_graph",
        "prime_context",
        "get_stats",
    }
)

# Per-call timeout across the bridge thread (observe_conversation runs a local
# LLM extraction step and can be slow, especially on model warm-up).
WAGGLE_CALL_TIMEOUT_SECONDS = float(config.WAGGLE_CALL_TIMEOUT_SECONDS)

# Tool results are shown to the model; cap each result so huge graph dumps
# cannot balloon the agent's context budget.
WAGGLE_OUTPUT_MAX_CHARS = int(config.WAGGLE_OUTPUT_MAX_CHARS)

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_ready = threading.Event()
_host_stop: Optional[asyncio.Event] = None
_ready_future: Optional[concurrent.futures.Future] = None  # tools channel
_teardown_future: Optional[concurrent.futures.Future] = None  # host finished
_tools_cache: Optional[list] = None  # sync-bridged tools (agent-facing)
_raw_tools_cache: Optional[dict] = None  # raw async MCP tools (runtime hooks)


def init_waggle_tools() -> list:
    """Return the adapted Waggle LangChain read tools, or [] if unavailable.

    Only called when WAGGLE_MCP_ENABLED=true. Failures are logged and swallowed
    so the rest of the agent keeps working. The session and tools are loaded
    once per process and reused on any subsequent call.
    """
    global _tools_cache
    if not config.WAGGLE_MCP_ENABLED:
        return []
    if _tools_cache is not None:
        return list(_tools_cache)
    _ensure_loop_thread()
    try:
        ready, teardown = concurrent.futures.Future(), concurrent.futures.Future()
        global _ready_future, _teardown_future
        _ready_future, _teardown_future = ready, teardown
        _loop.call_soon_threadsafe(_loop.create_task, _host(ready, teardown))
        _tools_cache = ready.result(timeout=WAGGLE_CALL_TIMEOUT_SECONDS)
        return list(_tools_cache)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("Waggle MCP unavailable; memory tools disabled: %s", exc)
        return []


def prime_session(session_id: str, *, block: bool = False) -> None:
    """Hydrate memory for a new conversation (call at session start).

    Best-effort: posts `prime_context` to the MCP loop thread without blocking
    the chat; failures are logged and ignored. No-op when the feature is off.
    `block=True` waits for the call (used by tests).
    """
    if not _session_ready():
        return
    _call_on_loop(
        "prime_context",
        {"project": config.WAGGLE_PROJECT, "agent_id": config.WAGGLE_AGENT_ID, "session_id": session_id},
        timeout=WAGGLE_CALL_TIMEOUT_SECONDS,
        block=block,
    )


def memorize_turn(
    user_message: str,
    assistant_response: str,
    session_id: str,
    *,
    block: bool = False,
) -> None:
    """Store durable memory for a completed turn (call after the reply).

    Runs `observe_conversation` on the MCP loop thread. Fire-and-forget by
    default so chat latency is never affected; `block=True` waits for the
    result (used by tests). Never raises.
    """
    if not _session_ready():
        return
    if not user_message or not assistant_response:
        return
    _call_on_loop(
        "observe_conversation",
        {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "project": config.WAGGLE_PROJECT,
            "agent_id": config.WAGGLE_AGENT_ID,
            "session_id": session_id,
        },
        timeout=WAGGLE_CALL_TIMEOUT_SECONDS,
        block=block,
    )


def _session_ready() -> bool:
    """True when the MCP session is live (host populated the raw tools)."""
    return config.WAGGLE_MCP_ENABLED and _loop is not None and _raw_tools_cache is not None


def _call_on_loop(tool_name: str, arguments: dict, *, timeout: float, block: bool = False) -> None:
    """Bridge a single runtime-hook tool call onto the session loop thread.

    Looks up the raw async MCP tool (not the sync-bridged agent tools).
    Fire-and-forget unless `block=True`: schedules the coroutine on the loop
    and optionally waits for it. Exceptions are logged, never raised to the
    callers of the hooks.
    """
    if _loop is None or not _raw_tools_cache:
        return
    tool = _raw_tools_cache.get(tool_name)
    if tool is None:
        logger.warning("Waggle tool %r not found in the session", tool_name)
        return

    async def _invoke_raw() -> None:
        await tool.ainvoke(arguments)

    try:
        future = asyncio.run_coroutine_threadsafe(_invoke_raw(), _loop)
        if block:
            future.result(timeout=timeout)
        else:
            future.add_done_callback(_swallow_future_error)
    except Exception as exc:  # noqa: BLE001 - memory hooks must never crash the chat
        logger.warning("Waggle %s call failed: %s", tool_name, exc)


def _swallow_future_error(future: "concurrent.futures.Future") -> None:
    """Consume a fire-and-forget future's exception (log, never re-raise)."""
    try:
        exc = future.exception(timeout=0)
    except Exception:  # noqa: BLE001 - canceled/failed futures are expected
        return
    if exc is not None:
        logger.warning("Waggle background memory call failed: %s", exc)


def _ensure_loop_thread() -> None:
    """Start the daemon thread that owns the MCP event loop (once)."""
    global _loop
    if _loop is not None:
        return

    def runner() -> None:
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _loop_ready.set()
        _loop.run_forever()

    threading.Thread(target=runner, name="waggle-mcp-loop", daemon=True).start()
    _loop_ready.wait(timeout=5.0)


async def _host(ready, teardown) -> None:
    """Enter the long-lived MCP session and keep it alive for the process.

    `ready` carries the selected tools once loaded; the host then blocks on
    the stop event so the stdio/session contexts (entered in THIS task) can be
    exited in this same task at shutdown (anyio requirement), which terminates
    the child process. `teardown` completes after the contexts close.
    """
    global _host_stop
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp_tool_bridge import load_mcp_tools

    server_params = StdioServerParameters(
        command=config.WAGGLE_MCP_COMMAND,
        args=shlex.split(config.WAGGLE_MCP_ARGS) if config.WAGGLE_MCP_ARGS else [],
        env={**os.environ, **_server_environment()},
    )

    _host_stop = asyncio.Event()
    stdio = stdio_client(server_params)
    session = None
    try:
        read_stream, write_stream = await stdio.__aenter__()
        session = await ClientSession(read_stream, write_stream).__aenter__()
        await session.initialize()

        raw_tools = await load_mcp_tools(session)
        global _raw_tools_cache
        _raw_tools_cache = {tool.name: tool for tool in raw_tools}
        selected = _select_tools(raw_tools)
        if not selected:
            logger.warning(
                "No Waggle MCP tools matched the allow-list %s; got %r",
                sorted(WAGGLE_READ_TOOL_NAMES),
                [getattr(t, "name", None) for t in raw_tools],
            )
        else:
            logger.info(
                "Waggle MCP ready; exposed tools: %s",
                ", ".join(sorted(tool.name for tool in selected)),
            )
        global _tools_cache
        _tools_cache = selected
        ready.set_result(selected)
        await _host_stop.wait()
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        if not ready.done():
            ready.set_exception(exc)
        logger.warning("Waggle MCP host failed: %s", exc)
    finally:
        try:
            if session is not None:
                await session.__aexit__(None, None, None)
        finally:
            await stdio.__aexit__(None, None, None)
            teardown.set_result(True)


def _server_environment() -> dict:
    """Environment for the waggle-mcp subprocess (DB, embedding, scope).

    The Jina API key is forwarded so the server can embed through the same
    jina-embeddings-v4 service the PDF RAG uses — no local model download.
    """
    env = {
        "WAGGLE_DB_PATH": str(config.WAGGLE_DB_PATH),
        "WAGGLE_MODEL": config.WAGGLE_MODEL,
        "WAGGLE_EMBEDDING_BACKEND": config.WAGGLE_EMBEDDING_BACKEND,
        "WAGGLE_EMBEDDING_DIMENSIONS": str(config.WAGGLE_EMBEDDING_DIMENSIONS),
        "WAGGLE_DEFAULT_TENANT_ID": "hakathon",
        "WAGGLE_TRANSPORT": "stdio",
        "WAGGLE_LOG_LEVEL": "WARNING",
    }
    if config.JINA_API_KEY:
        env["JINA_API_KEY"] = config.JINA_API_KEY
    return env


def _to_sync_tool(tool) -> "object":
    """Bridge the async MCP tool onto the dedicated event-loop thread."""
    from mcp_tool_bridge import build_langchain_tool

    def invoke_impl(ainvoke, **kwargs) -> str:
        future = asyncio.run_coroutine_threadsafe(ainvoke(kwargs), _loop)
        raw = future.result(timeout=WAGGLE_CALL_TIMEOUT_SECONDS)
        return _flatten_content(raw, max_chars=WAGGLE_OUTPUT_MAX_CHARS)

    return build_langchain_tool(tool, invoke_impl)


def _flatten_content(raw, max_chars: Optional[int] = None) -> str:
    """Collapse MCP tool output (list of content blocks or text) to a string.

    Optionally caps the length so huge payloads don't balloon the conversation
    context; a marker notes the truncation.
    """
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        text = "".join(parts)
    else:
        text = str(raw)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def _select_tools(raw_tools) -> list:
    """Keep only the allow-listed tools, bridged to sync tools."""
    return [_to_sync_tool(tool) for tool in raw_tools if _matches(tool.name)]


def _matches(name: str) -> bool:
    """Match tool names, tolerating MCP-server prefixes (e.g. 'waggle.query_graph')."""
    return name in WAGGLE_READ_TOOL_NAMES or name.rsplit(".", 1)[-1] in WAGGLE_READ_TOOL_NAMES


@atexit.register
def _shutdown() -> None:
    """Signal the host task to close the session, then stop the loop."""
    global _loop, _host_stop, _ready_future, _teardown_future, _tools_cache, _raw_tools_cache
    loop, stop = _loop, _host_stop
    teardown = _teardown_future
    _loop = None
    _host_stop = None
    _ready_future = None
    _teardown_future = None
    _tools_cache = None
    _raw_tools_cache = None
    if loop is None or stop is None or teardown is None:
        return

    try:
        loop.call_soon_threadsafe(stop.set)
        teardown.result(timeout=15.0)
    except Exception:  # noqa: BLE001 - shutdown must never raise
        pass
    finally:
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:  # noqa: BLE001
            pass