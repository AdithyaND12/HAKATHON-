"""Interactive CLI entrypoint for the HAKATHON LangGraph chatbot.

Dedicated logic lives in modules:

    config.py         — env parsing
    planner.py        — search + schedule planner (LLM + regex fallback)
    scheduler.py      — background job registry, persistence, retries
    tools_search.py   — DuckDuckGo tool wrapped with retry/backoff
    ragtool.py        — user-uploaded PDF RAG

This module owns the LangGraph agent (tools + graph), the scheduler wiring,
and the interactive CLI (including scheduled-run message preparation).

Public names (`get_stock_price`, `run_scheduled_search`, `chatbot`,
`ALPHAVANTAGE_API_KEY`, `HTTP_TIMEOUT_SECONDS`, `requests`, `llm`, `_registry`,
`scheduler`, `set_chatbot_model`) are re-exported here so the existing tests
and `streamlit_app.py` continue to work unchanged.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

import config
from planner import (
    CalculationExecutionInstruction,
    ChatExecutionInstruction,
    EmailExecutionInstruction,
    RagExecutionInstruction,
    ReminderExecutionInstruction,
    SearchExecutionInstruction,
    SearchPlan,
    _invoke_with_transient_retry,
    build_chat_llm,
    create_search_plan,
    fallback_search_plan,
    set_planner_model,
    strip_schedule_phrases,
)
from scheduler import (
    JobRegistry,
    ScheduledSearchJob,
    ScheduleValidationError,
    Scheduler,
    _extract_content,
    console_print,
    format_jobs_table,
    prompt_state,
)
from tools_search import search_tool
from ragtool import retrieve_active_chunks
from gmail_tools import init_gmail_tools
from waggle_tools import WAGGLE_MEMORY_POLICY, init_waggle_tools, memorize_turn, prime_session

load_dotenv()

# Guard rails for the agent loop and CLI robustness.
CHAT_RECURSION_LIMIT = 10
MAX_CONTEXT_TOKENS = 10000
MAX_LOG_CHARS = 100_000


# ---- Backward-compatible constants -------------------------------------------

LM_STUDIO_MODEL = config.LM_STUDIO_MODEL
LM_STUDIO_BASE_URL = config.LM_STUDIO_BASE_URL
LM_STUDIO_API_KEY = config.LM_STUDIO_API_KEY
ALPHAVANTAGE_API_KEY = config.ALPHAVANTAGE_API_KEY
HTTP_TIMEOUT_SECONDS = config.HTTP_TIMEOUT_SECONDS
MAX_WAIT_SECONDS = config.WAIT_MAX_SECONDS
MAX_AUTO_RUNS = config.MAX_AUTO_RUNS


# ---- LLM ---------------------------------------------------------------------

llm = build_chat_llm()


# ---- Tools -------------------------------------------------------------------


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.

    Supported operations: add, sub, mul, div.
    """
    try:
        operation = (operation or "").strip().lower()
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@tool
def get_time() -> str:
    """Return the current local time in YYYY-MM-DD HH:mm:ss format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a given symbol (e.g. 'AAPL', 'TSLA').

    External API failures are returned as structured errors so the agent can
    explain them without crashing the graph.
    """
    if not ALPHAVANTAGE_API_KEY:
        return _stock_error(
            "missing_api_key",
            "ALPHAVANTAGE_API_KEY is not configured; stock prices are unavailable.",
        )

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": ALPHAVANTAGE_API_KEY,
            },
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return _stock_error("timeout", "The stock price request timed out.")
    except requests.exceptions.ConnectionError:
        return _stock_error("connection_error", "Could not connect to Alpha Vantage.")
    except requests.exceptions.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f" (HTTP {status})" if status else ""
        return _stock_error("http_error", f"Alpha Vantage returned an HTTP error{suffix}.")
    except requests.exceptions.RequestException as exc:
        return _stock_error("request_error", f"Stock price request failed: {exc}")

    try:
        payload = response.json()
    except ValueError:
        return _stock_error("invalid_json", "Alpha Vantage returned invalid JSON.")

    if not isinstance(payload, dict):
        return _stock_error("invalid_response", "Alpha Vantage returned an invalid response.")
    if "Error Message" in payload:
        return _stock_error("api_error", str(payload["Error Message"]))
    rate_limit_message = payload.get("Note") or payload.get("Information")
    if rate_limit_message:
        return _stock_error("rate_limit", str(rate_limit_message))

    return payload


def _stock_error(code: str, message: str) -> dict:
    return {"error": message, "error_type": code}


@tool
def get_rag_chunks(query: str) -> str:
    """Retrieve relevant passages from the user's active RAG document.

    The active document is a PDF the user has uploaded and selected. Returns
    an explicit "no matches" message when the retriever finds nothing, and a
    prompt to upload a document when none is active.
    """
    result = retrieve_active_chunks(query)
    if not result.strip():
        return "No matching passages found in the active document."
    return result


# ---- Graph -------------------------------------------------------------------

SCHEDULED_RUN_INSTRUCTIONS = (
    "The user's original request was time-based (recurring/scheduled). "
    "The application's scheduler has ALREADY handled all timing, waiting, and "
    "repetition for you. You are now inside one such run. Your ONLY job is to "
    "perform the underlying task once — call the appropriate tool (usually web "
    "search) and answer with the results. "
    "DO NOT refuse the task, DO NOT mention that the request looks recurring, "
    "and DO NOT say the scheduling is impossible. Just do the task now."
)

agent_tools = [get_stock_price, search_tool, calculator, get_rag_chunks, get_time]

# Optional Gmail tools (gmail_mcp): inbox reads + mailbox writes. Fail-open:
# when the MCP server or auth is unavailable, init_gmail_tools() logs a
# warning and returns [].
agent_tools += init_gmail_tools()

# Optional Waggle memory tools (waggle_mcp): persistent cross-session recall.
# Read-only surface (query_graph / prime_context / get_stats); writes happen
# through the runtime hooks. Fail-open: a missing server logs a warning and
# returns [].
agent_tools += init_waggle_tools()

llm_with_tools = llm.bind_tools(agent_tools)
scheduled_llm_with_tools = llm.bind_tools(agent_tools)


def set_chatbot_model(model: str) -> None:
    """Switch the active chat model at runtime.

    Rebuilds the tool-bound LLM and the planner LLM, then reassigns the module
    globals. The compiled graph reads these globals on every invocation, so
    both interactive chat and scheduled runs pick up the new model immediately.
    """
    global llm, llm_with_tools, scheduled_llm_with_tools
    llm = build_chat_llm(model)
    llm_with_tools = llm.bind_tools(agent_tools)
    scheduled_llm_with_tools = llm.bind_tools(agent_tools)
    set_planner_model(model)


def _message_text(message: BaseMessage) -> str:
    """Flatten a message's content (string or block list) to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _rough_token_count(messages: list[BaseMessage]) -> int:
    """Approximate token count without a tokenizer (~4 chars per token)."""
    total = 0
    for message in messages:
        total += max(1, len(_message_text(message)) // 4) + 2
    return total


def _trim_for_invoke(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Bound the invocation context to MAX_CONTEXT_TOKENS.

    History is trimmed per-invocation only; the LangGraph state keeps the full
    conversation, so trimming never loses information from the user's session.

    Guarantee: the result always contains the system message(s) plus at least
    one conversational message. Very large tool outputs can make the token
    trimmer drop every non-system message, which some model backends reject.
    """
    if _rough_token_count(messages) <= MAX_CONTEXT_TOKENS:
        return list(messages)
    try:
        trimmed = trim_messages(
            messages,
            token_counter=_rough_token_count,
            max_tokens=MAX_CONTEXT_TOKENS,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
    except Exception:  # noqa: BLE001 - trimming must never block the chat
        system = [m for m in messages if isinstance(m, SystemMessage)]
        rest = [m for m in messages if not isinstance(m, SystemMessage)]
        return _safe_fallback(system, rest)
    result = list(trimmed)
    if _has_user_voice(result):
        return result
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    return _safe_fallback(system, rest)


def _has_user_voice(messages: list[BaseMessage]) -> bool:
    """True when the trimmed history contains something other than pure system."""
    return any(not isinstance(m, SystemMessage) for m in messages)


def _safe_fallback(system: list[BaseMessage], rest: list[BaseMessage]) -> list[BaseMessage]:
    """Keep system context plus the newest messages when trimming misbehaves.

    Each message is hard-capped so oversized tool output cannot break the
    invocation; the selection favours the latest turns.
    """
    max_pre_system = MAX_CONTEXT_TOKENS // 4
    capped = []
    for message in (system + rest[-8:]) if rest else system:
        text = _message_text(message)
        if len(text) > max_pre_system:
            capped_message = message.model_copy(
                update={"content": text[:max_pre_system].rstrip() + "\n...[truncated]"}
            )
            capped.append(capped_message)
        else:
            capped.append(message)
    guard = list(capped)
    if not _has_user_voice(guard) and rest:
        guard = capped + rest[-1:]
    return guard[-12:]


def _is_scheduled_run(messages: list[BaseMessage]) -> bool:
    """True when this invocation came from the scheduler (not interactive chat)."""
    for message in messages:
        if isinstance(message, SystemMessage) and _message_text(message).strip() == SCHEDULED_RUN_INSTRUCTIONS:
            return True
    return False


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = _trim_for_invoke(state["messages"])
    selected = scheduled_llm_with_tools if _is_scheduled_run(messages) else llm_with_tools
    return {
        "messages": [
            _invoke_with_transient_retry(lambda: selected.invoke(messages))
        ]
    }


tool_node = ToolNode(agent_tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()


# ---- Scheduler wiring --------------------------------------------------------

_registry = JobRegistry(config.ensure_data_dir())


def _sanitize_prompt_for_scheduled_run(prompt: str, search_query: str) -> str:
    """Rewrite the user's original prompt so the LLM doesn't latch on to the
    scheduling verbs (which cause some local models to refuse the whole task).

    Strategy: drop everything that looks like a schedule directive, keep the
    task words. If the resulting prompt looks empty or trivial, fall back to
    the planner's cleaned `search_query`.
    """
    text = strip_schedule_phrases(prompt)
    # If we stripped too much, fall back to the planner's search_query.
    if len(text.split()) < 2:
        return search_query
    return text


def _build_messages(prompt: str, search_query: str, task_type: str = "search",
                    reminder_text: Optional[str] = None) -> list[BaseMessage]:
    """Build the message list for a single scheduled-run invocation.

    Dispatches on `task_type` so each kind of scheduled task uses the right
    tool-orchestration instruction. Reminders never nudge toward web_search;
    calculations route to the calculator tool; RAG routes to get_rag_chunks.
    """
    clean_prompt = _sanitize_prompt_for_scheduled_run(prompt, search_query)

    if task_type == "reminder":
        text = reminder_text or clean_prompt or search_query
        return [
            SystemMessage(content=SCHEDULED_RUN_INSTRUCTIONS),
            SystemMessage(content=ReminderExecutionInstruction(text).render()),
            HumanMessage(content="Please deliver the reminder now."),
        ]

    if task_type == "calculation":
        return [
            SystemMessage(content=SCHEDULED_RUN_INSTRUCTIONS),
            SystemMessage(content=CalculationExecutionInstruction(clean_prompt).render()),
            HumanMessage(content=clean_prompt),
        ]

    if task_type == "rag":
        return [
            SystemMessage(content=SCHEDULED_RUN_INSTRUCTIONS),
            SystemMessage(content=RagExecutionInstruction(clean_prompt).render()),
            HumanMessage(content=clean_prompt),
        ]

    if task_type == "chat":
        return [
            SystemMessage(content=SCHEDULED_RUN_INSTRUCTIONS),
            SystemMessage(content=ChatExecutionInstruction(clean_prompt).render()),
            HumanMessage(content=clean_prompt),
        ]

    if task_type == "email":
        return [
            SystemMessage(content=SCHEDULED_RUN_INSTRUCTIONS),
            SystemMessage(content=EmailExecutionInstruction(clean_prompt).render()),
            HumanMessage(content=clean_prompt),
        ]

    # Default: search
    return [
        SystemMessage(content=SCHEDULED_RUN_INSTRUCTIONS),
        SystemMessage(content=SearchExecutionInstruction(search_query).render()),
        HumanMessage(content=clean_prompt),
    ]


def _invoke_chatbot(state: dict) -> dict:
    """Invoke the compiled graph with an explicit recursion limit.

    A `TypeError` fallback keeps strict test doubles (which only accept
    `state`) working unchanged. `GraphRecursionError` propagates so callers
    can surface a polite "too many tool calls" message.
    """
    try:
        out = chatbot.invoke(
            state, config={"recursion_limit": CHAT_RECURSION_LIMIT}
        )
    except TypeError:
        out = chatbot.invoke(state)
    except GraphRecursionError:
        raise
    # Completed runs produce durable digests (prices, news summaries) worth
    # remembering. Best-effort and never blocking: failures are logged inside.
    session_id = "scheduled" if _is_scheduled_run(state.get("messages", [])) else "cli"
    _memorize_last_turn(state.get("messages", []), out, session_id=session_id)
    return out


def _memorize_last_turn(
    messages: list[BaseMessage], out: dict, *, session_id: str
) -> None:
    """Store the last user/assistant turn into Waggle memory (best-effort).

    Extracts the final human message from the input state and the final
    assistant reply from the invocation result, then fires
    observe_conversation on the memory loop thread. Always a no-op when the
    feature is disabled; never raises.
    """
    try:
        user_message = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_message = _message_text(message).strip()
                break
        if not user_message:
            return
        assistant_response = ""
        for message in reversed(out.get("messages", [])):
            if not isinstance(message, (HumanMessage, SystemMessage)):
                assistant_response = _message_text(message).strip()
                break
        if not assistant_response:
            return
        memorize_turn(
            user_message, assistant_response, session_id=session_id, block=False
        )
    except Exception:  # noqa: BLE001 - memory must never break the chat
        logger.exception("Waggle memorize_turn failed")


scheduler = Scheduler(
    registry=_registry,
    invoker=_invoke_chatbot,
    message_builder=_build_messages,
    planner=create_search_plan,
    max_invoke_retries=3,
    invoke_retry_backoff=2.0,
)


def run_scheduled_search(
    prompt: str,
    interval_minutes: Optional[float] = None,
    run_count: Optional[int] = None,
    search_query: Optional[str] = None,
    absolute_start_iso: Optional[str] = None,
    task_type: str = "search",
    reminder_text: Optional[str] = None,
) -> ScheduledSearchJob:
    """Start a search schedule in a background thread.

    Preserved for backward compatibility with existing tests.
    """
    return scheduler.start(
        prompt=prompt,
        interval_minutes=interval_minutes,
        run_count=run_count,
        search_query=search_query,
        absolute_start_iso=absolute_start_iso,
        task_type=task_type,
        reminder_text=reminder_text,
    )


# ---- CLI ---------------------------------------------------------------------


COMMAND_HELP = """
Available slash commands:
  /help                 Show this message.
  /jobs                 List active and finished jobs in this session.
  /cancel <id>          Cancel a running job.
  /pause <id>           Pause a job (finishes the current run, then waits).
  /resume <id>          Resume a paused job.
  /logs <id>            Show the last run's output for a job.
  /clear                Remove finished/cancelled/failed jobs from the list.
  exit | quit | q       Leave the CLI (running schedules are stopped).
"""


def _handle_command(command: str) -> bool:
    """Return True when the input was a slash command (and was handled)."""
    if not command.startswith("/"):
        return False

    parts = command.split()
    verb = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    if verb == "/help":
        console_print(COMMAND_HELP.strip())
    elif verb == "/jobs":
        console_print(format_jobs_table(_registry.all()))
    elif verb in ("/cancel", "/pause", "/resume", "/logs"):
        if not arg:
            console_print(f"Usage: {verb} <job-id>. Run /jobs to list ids.")
        else:
            job = _registry.get(arg)
            if not job:
                console_print(f"No job with id {arg!r}.")
            elif verb == "/cancel":
                job.stop()
                console_print(f"Cancellation requested for job {arg}.")
            elif verb == "/pause":
                job.pause()
                _registry.update(job)
                console_print(f"Paused job {arg}.")
            elif verb == "/resume":
                job.resume()
                _registry.update(job)
                console_print(f"Resumed job {arg}.")
            else:  # /logs
                history_dir = _registry.history_path(arg)
                files = sorted(history_dir.glob("run-*.json"))
                if not files:
                    console_print(f"No runs recorded yet for job {arg}.")
                else:
                    try:
                        content = files[-1].read_text(encoding="utf-8")
                    except OSError as exc:
                        console_print(f"Could not read the run log for job {arg}: {exc}")
                    else:
                        if len(content) > MAX_LOG_CHARS:
                            content = content[:MAX_LOG_CHARS] + "\n... (truncated)"
                        console_print(f"--- Last run for job {arg} ({files[-1].name}) ---")
                        console_print(content)
    elif verb == "/clear":
        removed = 0
        for job in _registry.all():
            if job.status in ("completed", "cancelled", "failed"):
                _registry.remove(job.id)
                removed += 1
        console_print(f"Removed {removed} finished job(s).")
    else:
        console_print(f"Unknown command: {command}. Type /help for options.")
    return True


def _run_one_off(user_input: str, plan: SearchPlan) -> None:
    """Invoke the chatbot once with the planner-chosen query and task routing."""
    if plan.task_type == "reminder":
        instruction = ReminderExecutionInstruction(plan.reminder_text or plan.search_query)
    elif plan.task_type == "calculation":
        instruction = CalculationExecutionInstruction(plan.search_query)
    elif plan.task_type == "rag":
        instruction = RagExecutionInstruction(plan.search_query)
    elif plan.task_type == "chat":
        instruction = ChatExecutionInstruction(plan.search_query)
    else:
        instruction = SearchExecutionInstruction(plan.search_query)
    try:
        out = _invoke_chatbot(
            {
                "messages": [
                    SystemMessage(content=WAGGLE_MEMORY_POLICY),
                    SystemMessage(content=instruction.render()),
                    HumanMessage(content=user_input),
                ]
            }
        )
    except GraphRecursionError:
        console_print(
            "Assistant: stopped after "
            f"{CHAT_RECURSION_LIMIT} tool calls; try rephrasing."
        )
        return
    console_print(f"Assistant: {_extract_content(out)}")


def run_cli() -> None:
    """Run the interactive client with LLM-selected search scheduling."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not config.GEMINI_API_KEY:
        console_print(
            "GEMINI_API_KEY is not configured. Add it to .env (or export it) and restart."
        )
        return

    console_print("Chatbot is ready. Type /help for commands, or 'exit' to quit.")

    # Hydrate Waggle memory for this CLI session (best-effort, no-op when off).
    prime_session("cli")

    # Resume any schedules persisted from previous sessions.
    restored = scheduler.resume_persisted()
    if restored:
        console_print(f"Resumed {len(restored)} persisted schedule(s):")
        console_print(format_jobs_table(restored))

    try:
        prompt_state["enabled"] = True
        prompt_state["text"] = "You: "
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                console_print("Goodbye!")
                break

            if user_input.lower() in {"exit", "quit", "q"}:
                console_print("Goodbye!")
                break
            if not user_input:
                continue
            if _handle_command(user_input):
                continue

            try:
                plan = create_search_plan(user_input)
            except Exception as exc:  # noqa: BLE001
                console_print(f"Planner error: {exc}. Using local fallback.")
                plan = fallback_search_plan(user_input)

            start_msg = (
                f"Planned search: {plan.search_query!r}; "
                f"wait={plan.wait_minutes:g} minute(s); runs={plan.run_count}"
            )
            if plan.absolute_start_iso:
                start_msg += f"; first run at {plan.absolute_start_iso}"
            console_print(start_msg)

            if plan.should_schedule:
                try:
                    job = scheduler.start(
                        prompt=user_input,
                        interval_minutes=plan.wait_minutes if plan.wait_minutes else None,
                        run_count=plan.run_count,
                        search_query=plan.search_query,
                        absolute_start_iso=plan.absolute_start_iso,
                        task_type=plan.task_type,
                        reminder_text=plan.reminder_text,
                    )
                except ScheduleValidationError as exc:
                    console_print(f"Invalid schedule: {exc}")
                    continue
                console_print(
                    f"Schedule started with id {job.id}. "
                    "Use /jobs to view, /cancel {id} to stop."
                )
                continue

            _run_one_off(user_input, plan)
    except KeyboardInterrupt:
        console_print("\nInterrupt received; cancelling active schedules.")
    finally:
        prompt_state["enabled"] = False
        for job in _registry.active():
            job.stop()
        exit_deadline = time.monotonic() + 5.0
        for job in _registry.active():
            job.join(timeout=max(0.0, exit_deadline - time.monotonic()))
        for job in _registry.active():
            if not job.done:
                job.status = "cancelled"
                job.error_message = job.error_message or "Interrupted at shutdown."
                _registry.update(job)


if __name__ == "__main__":
    run_cli()
