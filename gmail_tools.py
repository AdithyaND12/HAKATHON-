"""Optional Gmail MCP integration for the HAKATHON chatbot.

The gmail-mcp server (@shinzolabs/gmail-mcp, cloned + built under
/Users/adithya/agent2/gmail-mcp) is started as a stdio subprocess and its
tools are adapted to LangChain tools via mcp_tool_bridge (a minimal bridge
over the raw mcp client). Both inbox
reads and mailbox writes (send, drafts, labels, trash/delete, threads, watch)
are exposed; account-level settings are excluded. OAuth credentials live in
~/.gmail-mcp/ (the server's default config dir), so no extra setup is needed
once GMAIL_MCP_ENABLED=true is set in .env.

The gmail-mcp server unconditionally binds an HTTP listener on PORT and stream
servers only support a single stdio client per subprocess, so exactly ONE
long-lived session is kept open for the app's lifetime. A daemon thread runs
the MCP event loop, and every tool call is bridged onto that loop as a plain
sync LangChain tool, so the existing LangGraph ToolNode works unchanged.

Shutdown uses a long-lived "host task" on that loop: it enters the stdio and
session contexts (which anyio requires to be exited in the same task that
entered them), blocks on a stop event, then exits them itself — killing the
Node subprocess. An atexit hook signals the event and waits for the host task.

Everything here is fail-open: if the server is missing, authentication is
unavailable, or the load fails, the app logs a warning and continues with the
rest of the tools. When GMAIL_MCP_ENABLED is falsy this module never imports
the MCP libraries and never spawns a subprocess (keeps the test suite
hermetic).
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

# Read + mailbox-write subset of the gmail-mcp endpoints the agent may use.
# Account-level settings (IMAP/POP, vacation responder, filters, forwarding,
# send-as aliases, delegates, S/MIME) are deliberately excluded.
GMAIL_READ_TOOL_NAMES = frozenset(
    {
        "get_profile",
        "list_messages",
        "get_message",
        "get_attachment",
        "list_threads",
        "get_thread",
        "list_drafts",
        "get_draft",
        "list_labels",
        "get_label",
    }
)
GMAIL_WRITE_TOOL_NAMES = frozenset(
    {
        "send_message",
        "modify_message",
        "trash_message",
        "untrash_message",
        "delete_message",
        "batch_modify_messages",
        "batch_delete_messages",
        "create_draft",
        "update_draft",
        "delete_draft",
        "send_draft",
        "create_label",
        "update_label",
        "patch_label",
        "delete_label",
        "modify_thread",
        "trash_thread",
        "untrash_thread",
        "delete_thread",
        "watch_mailbox",
        "stop_mail_watch",
    }
)
GMAIL_TOOL_NAMES = GMAIL_READ_TOOL_NAMES | GMAIL_WRITE_TOOL_NAMES

# Per-call timeout across the bridge thread (the Gmail API itself can be slow,
# but an infinite hang is worse than a failed tool call).
GMAIL_CALL_TIMEOUT_SECONDS = 60.0

# Tool results are shown to the model; multi-KB JSON payloads can exceed the
# agent's per-invocation context budget and trigger pathological history
# trimming. Cap each result so the model gets enough to summarize while the
# conversation stays small.
GMAIL_OUTPUT_MAX_CHARS = 3000

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_ready = threading.Event()
_host_stop: Optional[asyncio.Event] = None
_ready_future: Optional[concurrent.futures.Future] = None  # tools channel
_teardown_future: Optional[concurrent.futures.Future] = None  # host finished
_tools_cache: Optional[list] = None  # one-time load per process


def init_gmail_tools() -> list:
    """Return the adapted Gmail LangChain tools, or [] if unavailable.

    Only called when GMAIL_MCP_ENABLED=true. Failures are logged and swallowed
    so the rest of the agent keeps working. The session and tools are loaded
    once per process and reused on any subsequent call.
    """
    global _tools_cache
    if not config.GMAIL_MCP_ENABLED:
        return []
    if _tools_cache is not None:
        return list(_tools_cache)
    _ensure_loop_thread()
    try:
        ready, teardown = concurrent.futures.Future(), concurrent.futures.Future()
        global _ready_future, _teardown_future
        _ready_future, _teardown_future = ready, teardown
        _loop.call_soon_threadsafe(_loop.create_task, _host(ready, teardown))
        _tools_cache = ready.result(timeout=GMAIL_CALL_TIMEOUT_SECONDS)
        return list(_tools_cache)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        logger.warning("Gmail MCP unavailable; email tools disabled: %s", exc)
        return []


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

    threading.Thread(target=runner, name="gmail-mcp-loop", daemon=True).start()
    _loop_ready.wait(timeout=5.0)


async def _host(ready, teardown) -> None:
    """Enter the long-lived MCP session and keep it alive for the process.

    `ready` carries the selected tools once loaded; the host then blocks on
    the stop event so the stdio/session contexts (entered in THIS task) can be
    exited in this same task at shutdown (anyio requirement), which terminates
    the Node subprocess. `teardown` completes after the contexts close.
    """
    global _host_stop
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp_tool_bridge import load_mcp_tools

    server_params = StdioServerParameters(
        command=config.GMAIL_MCP_COMMAND,
        args=shlex.split(config.GMAIL_MCP_ARGS),
        env={
            **os.environ,
            # The gmail-mcp always binds an HTTP listener; port "0" lets the
            # OS assign a free one so instances never clash.
            "PORT": config.GMAIL_MCP_PORT,
            "AUTH_SERVER_PORT": config.GMAIL_MCP_PORT,
            "TELEMETRY_ENABLED": "false",
        },
    )

    _host_stop = asyncio.Event()
    stdio = stdio_client(server_params)
    session = None
    try:
        read_stream, write_stream = await stdio.__aenter__()
        session = await ClientSession(read_stream, write_stream).__aenter__()
        await session.initialize()

        raw_tools = await load_mcp_tools(session)
        selected = _select_tools(raw_tools)
        if not selected:
            logger.warning(
                "No Gmail MCP tools matched the allow-list %s; got %r",
                sorted(GMAIL_TOOL_NAMES),
                [getattr(t, "name", None) for t in raw_tools],
            )
        else:
            logger.info(
                "Gmail MCP ready; exposed tools: %s",
                ", ".join(sorted(tool.name for tool in selected)),
            )
        global _tools_cache
        _tools_cache = selected
        ready.set_result(selected)
        await _host_stop.wait()
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller
        if not ready.done():
            ready.set_exception(exc)
        logger.warning("Gmail MCP host failed: %s", exc)
    finally:
        try:
            if session is not None:
                await session.__aexit__(None, None, None)
        finally:
            await stdio.__aexit__(None, None, None)
            teardown.set_result(True)


def _to_sync_tool(tool) -> "object":
    """Bridge the async MCP tool onto the dedicated event-loop thread.

    MCP tools are async-only and bound to the session living on the gmail-mcp
    loop thread, so each sync call is scheduled there and awaited via
    run_coroutine_threadsafe. This gives the existing LangGraph ToolNode a
    plain sync tool that reuses the one persistent session.
    """
    from mcp_tool_bridge import build_langchain_tool

    def invoke_impl(ainvoke, **kwargs) -> str:
        future = asyncio.run_coroutine_threadsafe(ainvoke(kwargs), _loop)
        raw = future.result(timeout=GMAIL_CALL_TIMEOUT_SECONDS)
        return _flatten_content(raw, max_chars=GMAIL_OUTPUT_MAX_CHARS)

    return build_langchain_tool(tool, invoke_impl)


def _flatten_content(raw, max_chars: Optional[int] = None) -> str:
    """Collapse MCP tool output (list of content blocks or text) to a string.

    Optionally caps the length so huge payloads (full email bodies) don't
    balloon the conversation context; a marker notes the truncation.
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

    # Try to parse as Gmail message list and extract clean summary
    text = _summarize_gmail_output(text)

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def _summarize_gmail_output(text: str) -> str:
    """Extract key fields from Gmail JSON responses to reduce noise.

    Handles list_messages, get_message, list_threads, etc.
    """
    import json

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text

    # Single message (get_message) with payload
    if isinstance(data, dict) and "payload" in data:
        return _format_single_message(data)

    # list_messages response: {"messages": [...], "nextPageToken": ...}
    if isinstance(data, dict) and "messages" in data:
        messages = data.get("messages", [])
        lines = []
        for msg in messages:
            lines.append(_format_single_message(msg))
        if data.get("nextPageToken"):
            lines.append(f"[More messages available — next page token present]")
        return "\n".join(lines) if lines else text

    # list_threads response: {"threads": [...], "nextPageToken": ...}
    if isinstance(data, dict) and "threads" in data:
        threads = data.get("threads", [])
        lines = []
        for t in threads:
            snippet = t.get("snippet", "")
            if len(snippet) > 120:
                snippet = snippet[:120].rstrip() + "..."
            lines.append(f"[Thread] {snippet}")
        if data.get("nextPageToken"):
            lines.append(f"[More threads available — next page token present]")
        return "\n".join(lines) if lines else text

    return text


def _format_single_message(msg: dict) -> str:
    """Format a single Gmail message dict into a readable one-liner.

    Handles both full messages (with payload.headers) and lightweight
    list_messages responses (only id/threadId/snippet).
    """
    import json

    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            return msg

    payload = msg.get("payload", {})
    headers_list = payload.get("headers", []) if isinstance(payload, dict) else []
    headers = {h["name"].lower(): h["value"] for h in headers_list if isinstance(h, dict)}

    subject = headers.get("subject")
    sender = headers.get("from")
    date = headers.get("date")
    snippet = msg.get("snippet", "")

    if len(snippet) > 120:
        snippet = snippet[:120].rstrip() + "..."

    # If no headers (lightweight list_messages response), show snippet only
    if not subject and not sender:
        return f"[Email] {snippet}" if snippet else f"[Message {msg.get('id', '?')}]"

    parts = []
    if sender:
        parts.append(f"From: {sender}")
    if subject:
        parts.append(f"Subject: {subject}")
    if date:
        parts.append(f"Date: {date}")
    if snippet:
        parts.append(snippet)

    return " | ".join(parts)


def _select_tools(raw_tools) -> list:
    """Keep only the allow-listed tools, bridged to sync tools."""
    return [_to_sync_tool(tool) for tool in raw_tools if _matches(tool.name)]


def _matches(name: str) -> bool:
    """Match tool names, tolerating MCP-server prefixes (e.g. 'gmail.list_messages')."""
    return name in GMAIL_TOOL_NAMES or name.rsplit(".", 1)[-1] in GMAIL_TOOL_NAMES


@atexit.register
def _shutdown() -> None:
    """Signal the host task to close the session, then stop the loop."""
    global _loop, _host_stop, _ready_future, _teardown_future, _tools_cache
    loop, stop = _loop, _host_stop
    teardown = _teardown_future
    _loop = None
    _host_stop = None
    _ready_future = None
    _teardown_future = None
    _tools_cache = None
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