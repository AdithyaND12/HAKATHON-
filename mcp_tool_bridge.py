"""Minimal mcp → LangChain tool bridge.

Replaces `langchain-mcp-adapters`, which pins `mcp<2.0.0` while waggle-mcp
requires `mcp>=2.0.0`. We drive the mcp 2.0 client directly and wrap the raw
tools as plain sync LangChain tools, so the existing LangGraph ToolNode and
both MCP bridges (gmail_tools, waggle_tools) work unchanged.

The bridge exposes:

    load_mcp_tools(session) -> list[McpTool]
        Lists the server's tools; each McpTool has `name`, `description`,
        `args_schema` (a pydantic model built from the JSON schema) and an
        async `ainvoke(arguments: dict)` that calls `session.call_tool` and
        flattens the response to plain text.

    build_langchain_tool(tool, invoke_impl) -> LangChain tool
        Wraps an McpTool as a sync LangChain tool whose calls go through
        the provided `invoke_impl(ainvoke_fn, **kwargs)` (the impl bridges
        onto the module's dedicated event-loop thread).

Everything here is dependency-light: only langchain-core and pydantic.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from langchain_core.tools import tool as langchain_tool
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _args_model_from_schema(schema: Optional[dict]) -> Optional[type[BaseModel]]:
    """Build a pydantic args model from an mcp tool's JSON input schema.

    Returns None when the schema is unusable (langchain then infers a generic
    passthrough). Simple scalars map directly; nested structures become
    loose-typed fields — good enough for tool dispatch.
    """
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        json_type = prop.get("type") if isinstance(prop, dict) else None
        field_type = _JSON_TYPE_MAP.get(json_type, Any)
        if name in required:
            fields[name] = (field_type, ...)
        else:
            fields[name] = (Optional[field_type], None)
    if not fields:
        return None
    try:
        return create_model("McpToolArgs", **fields)
    except Exception:  # noqa: BLE001 - schema quirks must not break loading
        logger.warning("Could not build args model for schema %r", schema)
        return None


class McpTool:
    """An mcp server tool callable as `await tool.ainvoke(arguments)`."""

    def __init__(self, name: str, description: str, args_schema: Optional[type[BaseModel]]) -> None:
        self.name = name
        self.description = description or ""
        self.args_schema = args_schema
        self._invoke: Optional[Callable[[dict], Any]] = None

    def bind(self, invoke: Callable[[dict], Any]) -> None:
        """Attach the session-bound callable (`session.call_tool`)."""
        self._invoke = invoke

    async def ainvoke(self, arguments: dict) -> str:
        """Invoke this tool on the session; returns flattened text."""
        if self._invoke is None:
            raise RuntimeError(f"Tool {self.name!r} is not session-bound")
        filtered = {k: v for k, v in arguments.items() if v is not None}
        result = await self._invoke(filtered)
        return flatten_mcp_result(result)

    def with_session(self, invoke: Callable[[dict], Any]) -> "McpTool":
        clone = McpTool(self.name, self.description, self.args_schema)
        clone.bind(invoke)
        return clone


async def load_mcp_tools(session) -> list[McpTool]:
    """List the server's tools as McpTool instances bound to *session*."""
    result = await session.list_tools()
    items = result.tools if hasattr(result, "tools") else result

    tools: list[McpTool] = []
    for item in items:
        schema = getattr(item, "inputSchema", None) or getattr(item, "input_schema", None)
        description = getattr(item, "description", "") or ""
        tool = McpTool(item.name, description, _args_model_from_schema(schema))
        tools.append(tool.with_session(lambda args, _name=item.name: session.call_tool(_name, args)))
    return tools


def build_langchain_tool(mcp_tool: McpTool, invoke_impl: Callable[..., str]) -> Any:
    """Wrap *mcp_tool* as a sync LangChain tool routed via *invoke_impl*.

    `invoke_impl(ainvoke_fn, **kwargs) -> str` receives the async ainvoke
    callable and the tool arguments; the caller bridges it onto its own
    event-loop thread (with timeouts + output caps).
    """

    def invoke_sync(**kwargs: Any) -> str:
        return invoke_impl(mcp_tool.ainvoke, **kwargs)

    return langchain_tool(
        mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
    )(invoke_sync)


def flatten_mcp_result(result: Any) -> str:
    """Collapse an mcp CallToolResult (content blocks / text) to a string."""
    is_error = bool(getattr(result, "isError", False))
    content = getattr(result, "content", result)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
            else:
                parts.append(str(block))
        text = "".join(parts)
    else:
        text = str(content)
    text = text.strip()
    if is_error:
        text = f"Error: {text}" if text else "Error: tool call failed"
    return text or "(no output)"