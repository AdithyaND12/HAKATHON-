"""Interactive CLI entrypoint for the HAKATHON LangGraph chatbot.

This file is intentionally thin. All heavy logic lives in dedicated modules:

    config.py         — env parsing
    planner.py        — search + schedule planner (LLM + regex fallback)
    scheduler.py      — background job registry, persistence, retries
    tools_search.py   — DuckDuckGo tool wrapped with retry/backoff
    ragtool.py        — Constitution PDF RAG

Public names (`get_stock_price`, `run_scheduled_search`, `chatbot`,
`ALPHAVANTAGE_API_KEY`, `HTTP_TIMEOUT_SECONDS`, `requests`) are re-exported here
so the existing `tests/test_app.py` continues to work unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import arrow  # kept for backward compatibility with the original tests
import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

import config
from planner import (
    CalculationExecutionInstruction,
    ChatExecutionInstruction,
    RagExecutionInstruction,
    ReminderExecutionInstruction,
    SearchExecutionInstruction,
    SearchPlan,
    create_search_plan,
)
from scheduler import (
    JobRegistry,
    ScheduledSearchJob,
    ScheduleValidationError,
    Scheduler,
    console_print,
    format_jobs_table,
    _extract_content,
)
from tools_search import search_tool
from ragtool import retrieve_constitution_chunks

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---- Backward-compatible constants -------------------------------------------

LM_STUDIO_MODEL = config.LM_STUDIO_MODEL
LM_STUDIO_BASE_URL = config.LM_STUDIO_BASE_URL
LM_STUDIO_API_KEY = config.LM_STUDIO_API_KEY
ALPHAVANTAGE_API_KEY = config.ALPHAVANTAGE_API_KEY
HTTP_TIMEOUT_SECONDS = config.HTTP_TIMEOUT_SECONDS
MAX_WAIT_SECONDS = config.WAIT_MAX_SECONDS
MAX_AUTO_RUNS = config.MAX_AUTO_RUNS


# ---- LLM ---------------------------------------------------------------------

llm = ChatOpenAI(
    model=config.LM_STUDIO_MODEL,
    base_url=config.LM_STUDIO_BASE_URL,
    api_key=config.LM_STUDIO_API_KEY,
)


# ---- Tools -------------------------------------------------------------------


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.

    Supported operations: add, sub, mul, div.
    """
    try:
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
    return arrow.now().format("YYYY-MM-DD HH:mm:ss")


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
    """Retrieve chunks from the Constitution PDF that match the query.

    Returns an explicit "no matches" message when the retriever finds nothing.
    """
    result = retrieve_constitution_chunks(query)
    if not result.strip():
        return "No matching passages found in the Constitution PDF."
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
tools = list(agent_tools)

llm_with_tools = llm.bind_tools(tools)
scheduled_llm_with_tools = llm.bind_tools(agent_tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    is_scheduled_run = any(
        isinstance(m, SystemMessage) and m.content == SCHEDULED_RUN_INSTRUCTIONS
        for m in messages
    )
    selected = scheduled_llm_with_tools if is_scheduled_run else llm_with_tools
    return {"messages": [selected.invoke(messages)]}


tool_node = ToolNode(tools)

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
    import re
    text = prompt
    # Remove "every N minutes/hours/days" phrases.
    text = re.sub(
        r"\b(?:for\s+)?(?:every|each)\s+\S+\s*"
        r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
        " ", text, flags=re.IGNORECASE,
    )
    # Remove "for N times/runs/checks".
    text = re.sub(
        r"\b(?:for\s+)?\S+\s*(?:times?|runs?|checks?|iterations?)\b",
        " ", text, flags=re.IGNORECASE,
    )
    # Remove standalone scheduling words.
    text = re.sub(
        r"\b(?:hourly|daily|weekly|monthly|repeatedly|periodically|recurring|"
        r"repeat|monitor|refresh|schedule|at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|"
        r"tomorrow|tonight|in\s+\d+\s+(?:minutes?|mins?|hours?|hrs?))\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,.!?")
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

    # Default: search
    return [
        SystemMessage(content=SCHEDULED_RUN_INSTRUCTIONS),
        SystemMessage(content=SearchExecutionInstruction(search_query).render()),
        HumanMessage(content=clean_prompt),
    ]


def _invoke_chatbot(state: dict) -> dict:
    return chatbot.invoke(state)


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
    elif verb == "/cancel" and arg:
        job = _registry.get(arg)
        if not job:
            console_print(f"No job with id {arg!r}.")
        else:
            job.stop()
            console_print(f"Cancellation requested for job {arg}.")
    elif verb == "/pause" and arg:
        job = _registry.get(arg)
        if not job:
            console_print(f"No job with id {arg!r}.")
        else:
            job.pause()
            _registry.update(job)
            console_print(f"Paused job {arg}.")
    elif verb == "/resume" and arg:
        job = _registry.get(arg)
        if not job:
            console_print(f"No job with id {arg!r}.")
        else:
            job.resume()
            _registry.update(job)
            console_print(f"Resumed job {arg}.")
    elif verb == "/logs" and arg:
        job = _registry.get(arg)
        if not job:
            console_print(f"No job with id {arg!r}.")
        else:
            history_dir = _registry.history_path(arg)
            files = sorted(history_dir.glob("run-*.json"))
            if not files:
                console_print(f"No runs recorded yet for job {arg}.")
            else:
                console_print(f"--- Last run for job {arg} ({files[-1].name}) ---")
                console_print(files[-1].read_text(encoding="utf-8"))
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
    """Invoke the chatbot once with the planner-chosen query."""
    out = chatbot.invoke(
        {
            "messages": [
                SystemMessage(content=SearchExecutionInstruction(plan.search_query).render()),
                HumanMessage(content=user_input),
            ]
        }
    )
    console_print(f"Assistant: {_extract_content(out)}")


def run_cli() -> None:
    """Run the interactive client with LLM-selected search scheduling."""
    console_print("Chatbot is ready. Type /help for commands, or 'exit' to quit.")

    # Resume any schedules persisted from previous sessions.
    restored = scheduler.resume_persisted()
    if restored:
        console_print(f"Resumed {len(restored)} persisted schedule(s):")
        console_print(format_jobs_table(restored))

    try:
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
                console_print(f"Planner error: {exc}. Falling back to a single run.")
                plan = SearchPlan(
                    search_query=user_input,
                    should_schedule=False,
                    wait_minutes=0.0,
                    run_count=1,
                )

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
                        interval_minutes=plan.wait_minutes or None,
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
        for job in _registry.active():
            job.stop()
        for job in _registry.active():
            job.join(timeout=2)


if __name__ == "__main__":
    run_cli()
