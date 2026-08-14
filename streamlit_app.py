"""Streamlit UI for the HAKATHON LangGraph chatbot.

A single-page, chat-only interface with multiple conversations (sidebar
"chats" section), persisted to `.hakathon/conversations.json`. Replaces the
CLI in `app.py` but reuses every module (config, planner, scheduler, tools,
RAG, LangGraph wiring).

Run with:

    streamlit run streamlit_app.py

Design: dark terminal aesthetic, monospace headers, minimal chrome.

Scheduled runs still work — start one from chat ("search AI news every 10 min
for 3 times") and completed runs are surfaced back into the chat. A
background `@st.fragment` poller (instead of full-app auto-refresh) picks up
completed runs every few seconds while jobs are active, without redrawing the
whole page on every tick.

Improvements over the original UI:

* Multi-turn memory — prior turns are fed to the LLM (budget-trimmed).
* Streaming responses — tokens render as they arrive, with a token counter.
* Thread- and process-safe conversation persistence (file locking).
* Friendly errors — details go to the log, not the chat.
* LLM-generated chat titles and markdown/JSON export.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - only hit when the optional dep is missing
    def st_autorefresh(**kwargs):  # type: ignore[no-redef]
        """Inert compat stub.

        The UI no longer uses `streamlit-autorefresh` — background polling is
        handled by a `@st.fragment(run_every=...)` declared only while jobs are
        active. The stub is kept so older imports (and tests) keep working.
        """
        return 0

# Reuse everything the CLI used.
import config
import ragtool
from app import _registry, chatbot, llm, scheduler, set_chatbot_model
from planner import SearchExecutionInstruction, SearchPlan, create_search_plan
from scheduler import (
    ScheduleValidationError,
    _content_to_text,
    _extract_content,
)

log = logging.getLogger(__name__)

# Budget for feeding prior turns back to the LLM: newest-first, capped by both
# message count and total characters so long chats can't blow the context window.
HISTORY_MAX_MESSAGES = 20
HISTORY_MAX_CHARS = 24_000


# ---- Page config -------------------------------------------------------------

st.set_page_config(
    page_title="HAKATHON// chat",
    page_icon="◼",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---- Terminal-inspired dark theme --------------------------------------------

CUSTOM_CSS = """
<style>
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg-0: #0a0e12;
    --bg-1: #12181f;
    --bg-2: #1a2029;
    --bg-3: #232a35;
    --fg-0: #d7dde4;
    --fg-1: #8b95a5;
    --fg-dim: #5a6270;
    --accent: #7ee787;
    --accent-2: #ffb454;
    --danger: #ff7b72;
    --border: #232a35;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
    --sans: 'IBM Plex Sans', system-ui, sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-0) !important;
    color: var(--fg-0);
    font-family: var(--sans);
}
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border);
}

/* Hide the default Streamlit menu + footer for a cleaner look */
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

/* Header block */
.term-header {
    font-family: var(--mono);
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--fg-0);
    font-size: 1.75rem;
    padding: 0.5rem 0 0 0;
    margin: 0;
}
.term-header .accent { color: var(--accent); }
.term-header .dim { color: var(--fg-dim); }
.term-sub {
    font-family: var(--mono);
    color: var(--fg-dim);
    font-size: 0.8rem;
    padding: 0.25rem 0 1.5rem 0;
    letter-spacing: 0.02em;
}
.term-sub .dot {
    display: inline-block;
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent);
    margin-right: 6px;
    vertical-align: middle;
    box-shadow: 0 0 12px var(--accent);
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.5rem 0 !important;
    border: none !important;
}
[data-testid="stChatMessage"] > div:first-child {
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.65rem;
}
[data-testid="stChatMessageContent"] {
    background: transparent !important;
    color: var(--fg-0);
    font-family: var(--sans);
    line-height: 1.55;
}
[data-testid="stChatMessageContent"] code {
    font-family: var(--mono);
    background: var(--bg-2);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.85em;
    color: var(--accent-2);
}
[data-testid="stChatMessageContent"] pre {
    background: var(--bg-2) !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem !important;
}

/* User bubble accent (nth-child user first, then assistant) */
[data-testid="stChatMessage"][data-testid*="user"] > div:first-child,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:first-child {
    background: var(--bg-2);
    border-color: var(--bg-3);
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--bg-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--fg-0) !important;
    font-family: var(--mono) !important;
    font-size: 0.95rem !important;
    caret-color: var(--accent);
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--fg-dim) !important;
    font-family: var(--mono) !important;
}
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    color: var(--bg-0) !important;
    border: none !important;
    border-radius: 8px !important;
}

/* Buttons in sidebar */
.stButton > button {
    background: var(--bg-2);
    color: var(--fg-0);
    border: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.8rem;
    border-radius: 6px;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* Scheduled-run cards — prominent, distinct from regular chat */
.sched-card {
    font-family: var(--sans);
    background: linear-gradient(180deg, rgba(255, 180, 84, 0.06), rgba(255, 180, 84, 0.02));
    border: 1px solid rgba(255, 180, 84, 0.3);
    border-left: 3px solid var(--accent-2);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
}
.sched-card .sched-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--fg-dim);
    margin-bottom: 0.65rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px dashed var(--border);
}
.sched-card .sched-badge {
    background: var(--accent-2);
    color: var(--bg-0);
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.sched-card .sched-jobid {
    background: var(--bg-2);
    color: var(--accent);
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    font-family: var(--mono);
}
.sched-card .sched-runno {
    color: var(--fg-1);
    font-weight: 600;
}
.sched-card .sched-time {
    color: var(--fg-dim);
    margin-left: auto;
    font-size: 0.72rem;
}
.sched-card .sched-body {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--fg-0);
}
.sched-card .sched-body pre {
    background: var(--bg-2) !important;
    padding: 0.6rem !important;
    border-radius: 6px;
}

/* Active-schedules status strip above the chat */
.sched-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding: 0.5rem 0 1rem 0;
    margin-bottom: 0.75rem;
    border-bottom: 1px dashed var(--border);
}
.sched-chip {
    font-family: var(--mono);
    font-size: 0.72rem;
    background: var(--bg-1);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    padding: 0.35rem 0.6rem 0.35rem 0.55rem;
    border-radius: 6px;
    color: var(--fg-0);
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}
.sched-chip .sc-id { color: var(--accent-2); }
.sched-chip .sc-status {
    padding: 0.05rem 0.4rem;
    border-radius: 3px;
    font-size: 0.65rem;
    text-transform: uppercase;
    background: var(--bg-2);
    color: var(--fg-1);
}
.sched-chip .sc-status.running { color: var(--accent); }
.sched-chip .sc-status.paused { color: var(--accent-2); }
.sched-chip .sc-progress {
    color: var(--fg-1);
    font-variant-numeric: tabular-nums;
}
.sched-chip .sc-next { color: var(--fg-dim); }

/* Info banner for scheduled runs (kept as compact fallback) */
.scheduled-banner {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--accent-2);
    background: rgba(255, 180, 84, 0.07);
    border-left: 2px solid var(--accent-2);
    padding: 0.5rem 0.75rem;
    margin: 0.25rem 0 0.75rem 0;
    border-radius: 4px;
}
.plan-line {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--fg-dim);
    padding: 0.15rem 0 0 0;
}
.plan-line .kw { color: var(--accent); }
.plan-line .val { color: var(--fg-1); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-1);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }

/* Selected text */
::selection { background: var(--accent); color: var(--bg-0); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---- Sessions (multiple chats) ------------------------------------------------
# Each conversation keeps its own message history. Conversations, the active
# selection, and the job->chat routing map are persisted to a JSON sidecar next
# to the scheduler's registry so chats survive page refreshes. The RAG source,
# model picker, and raw mode stay app-wide (shared by every chat).
#
#   conversations: {conv_id: {"title": str, "messages": [msg, ...]}}
#   active_conv:   conv_id currently being viewed
#   job_conv:      job_id -> conv_id that started the schedule (run cards route
#                  back to the chat that created them)


def _conversations_path() -> Path:
    return config.DATA_DIR / "conversations.json"


# Serializes access to the conversations sidecar: an in-process re-entrant
# thread lock plus an OS advisory lock (fcntl), so concurrent tabs *and*
# concurrent Streamlit processes can't clobber each other's read-modify-write
# cycles. RLock (not Lock) so a read under a held store lock can't deadlock.
_CONVERSATIONS_LOCK = threading.RLock()
_STORE_LOCK_PATH = None
# Set while this process holds the POSIX advisory lock. flock locks conflict
# even between two file descriptors in the SAME process, so the fcntl lock is
# taken only once per process (outermost reader/writer); the RLock above covers
# re-entrant nested reads/writes within the process.
_STORE_LOCK_HELD = False


def _store_lock_path() -> Path:
    global _STORE_LOCK_PATH
    if _STORE_LOCK_PATH is None:
        _STORE_LOCK_PATH = _conversations_path().with_suffix(".json.lock")
    return _STORE_LOCK_PATH


@contextlib.contextmanager
def _store_lock() -> "contextlib.AbstractContextManager[None]":
    """Acquire the store lock (in-process re-entrant lock + a single POSIX
    advisory lock per process, where available). Falls back to the in-process
    lock on non-POSIX platforms."""
    global _STORE_LOCK_HELD
    _CONVERSATIONS_LOCK.acquire()
    lock_file = None
    try:
        if not _STORE_LOCK_HELD:
            try:
                import fcntl  # POSIX only.

                lock_file = open(_store_lock_path(), "w", encoding="utf-8")
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                _STORE_LOCK_HELD = True
            except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
                if lock_file is not None:
                    lock_file.close()
                lock_file = None
        yield
    finally:
        if lock_file is not None:
            _STORE_LOCK_HELD = False
            try:
                import fcntl

                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except (ImportError, OSError):  # pragma: no cover
                pass
            lock_file.close()
        _CONVERSATIONS_LOCK.release()


def _load_conversations() -> dict:
    """Read persisted conversations (conversations + active id + job routing)."""
    empty = {"conversations": {}, "active_conv": None, "job_conv": {}}
    path = _conversations_path()
    if not path.is_file():
        return empty
    try:
        with _store_lock():
            payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # noqa: S110 - corrupt sidecar, start fresh
        return empty
    if not isinstance(payload, dict):
        return empty
    conversations = payload.get("conversations") if isinstance(payload.get("conversations"), dict) else {}
    job_conv = payload.get("job_conv") if isinstance(payload.get("job_conv"), dict) else {}
    active_conv = payload.get("active_conv")
    if active_conv not in conversations:
        active_conv = None
    return {"conversations": conversations, "active_conv": active_conv, "job_conv": job_conv}


def _persist_conversations(conversations: dict, active_conv: "str | None", job_conv: dict) -> None:
    """Atomically write the conversation store; never raises (best effort)."""
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = _conversations_path()
        tmp = path.with_suffix(".json.tmp")
        with _store_lock():
            tmp.write_text(
                json.dumps(
                    {
                        "conversations": conversations,
                        "active_conv": active_conv,
                        "job_conv": job_conv,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, path)
    except OSError:  # noqa: S110 - persistence is best-effort
        log.exception("Could not persist conversations to %s", path if "path" in locals() else "?")


def _create_conversation(conversations: dict) -> str:
    """Add a fresh empty conversation and return its id."""
    conv_id = uuid.uuid4().hex[:12]
    conversations[conv_id] = {"title": "new chat", "messages": []}
    return conv_id


def _conversation_title(messages: list[dict]) -> str:
    """Name a chat after its first user message (truncated); fallback otherwise."""
    for message in messages:
        if message.get("role") == "user":
            title = " ".join(str(message.get("content", "")).split())
            return (title[:30] + "…") if len(title) > 30 else (title or "new chat")
    return "new chat"


def _conversation_to_markdown(messages: list[dict]) -> str:
    """Render a conversation as plain markdown for export/download."""
    lines: list[str] = [f"# {_conversation_title(messages)}", ""]
    for message in messages:
        role = message.get("role", "?")
        content = str(message.get("content", ""))
        if message.get("kind") == "scheduled_run":
            content = re.sub(r"<[^>]+>", "", content)  # strip card HTML
        lines.append(f"## {role}\n\n{content}\n")
    return "\n".join(lines)


def _route_scheduled_run(job_id: str, conversations: dict, job_conv: dict) -> "str | None":
    """Which chat receives a completed run: the chat that started the job, else
    the oldest surviving chat (covers jobs resumed after a restart)."""
    target = job_conv.get(job_id)
    if target in conversations:
        return target
    if conversations:
        return next(iter(conversations))
    return None


# ---- Session state -----------------------------------------------------------

if "conversations" not in st.session_state:
    _loaded = _load_conversations()
    conversations = _loaded["conversations"]
    if not conversations:
        # Migrate the pre-multi-chat single history into a default conversation.
        legacy = st.session_state.get("messages")
        if legacy:
            conversations["default"] = {
                "title": _conversation_title(legacy),
                "messages": list(legacy),
            }
            st.session_state.pop("messages", None)
        else:
            _create_conversation(conversations)
    st.session_state.conversations = conversations
    st.session_state.job_conv = _loaded["job_conv"]
    st.session_state.active_conv = _loaded["active_conv"] or next(iter(conversations))

if "seen_runs" not in st.session_state:
    # Track scheduled-run history files we have already surfaced in this session.
    # Pre-seed with everything that existed *before* this session began so we
    # only surface runs completed after the user opens the page.
    st.session_state.seen_runs = set()
    _history_root = _registry.history_dir
    if _history_root.is_dir():
        for _job_dir in _history_root.iterdir():
            if _job_dir.is_dir():
                for _existing in _job_dir.glob("run-*.json"):
                    st.session_state.seen_runs.add(str(_existing))

if "known_jobs" not in st.session_state:
    st.session_state.known_jobs = set()

if "resumed_once" not in st.session_state:
    # Resume persisted schedules exactly once per Streamlit process.
    st.session_state.resumed_once = True
    try:
        scheduler.resume_persisted()
    except Exception:  # noqa: BLE001
        pass


# ---- Live scheduled-run polling ------------------------------------------------
# Streamlit only reruns on interaction, so background scheduled runs would never
# surface on their own. A fragment with `run_every` polls in the background and
# appends newly-completed runs to a container created *outside* the fragment —
# Streamlit accumulates those elements across fragment reruns, so the rest of
# the page (chat history) is NOT redrawn on every poll tick. The fragment is
# declared lazily at the bottom of the script, so it exists only while a job is
# active (or within the short post-job grace window).

_active_now = _registry.active()
if _active_now:
    st.session_state["_last_active_at"] = datetime.now().timestamp()

_grace_active = False
_last_seen = st.session_state.get("_last_active_at")
if _last_seen and (datetime.now().timestamp() - _last_seen) < 15:
    _grace_active = True


def _render_live_runs_in(runs: list[dict], render_fn) -> bool:
    """Append scheduled-run cards for `runs` via `render_fn` (a callable like
    `container.markdown` or `st.markdown`). Returns True when at least one card
    was appended (i.e. user-visible work happened on this tick)."""
    appended = False
    for run in runs:
        card_html = _render_scheduled_run_card(run)
        target = _route_scheduled_run(
            run["job_id"], st.session_state.conversations, st.session_state.job_conv
        )
        if target is None:
            continue
        st.session_state.conversations[target]["messages"].append(
            {
                "role": "assistant",
                "kind": "scheduled_run",
                "content": card_html,
            }
        )
        if target == st.session_state.active_conv:
            render_fn(card_html, unsafe_allow_html=True)
            appended = True
    if runs:
        _persist_conversations(
            st.session_state.conversations,
            st.session_state.active_conv,
            st.session_state.job_conv,
        )
    return appended


@st.fragment(run_every="3s")
def _poll_scheduled_runs(live_container) -> None:
    """Background poller: surfaces completed scheduled runs while any job is
    active (or within the post-job grace window).

    Runs as its own fragment so the full app is NOT rerun every interval. When
    polling is no longer needed it triggers an app rerun so the main script can
    stop declaring it (a fragment can't un-declare itself).
    """
    job_count = len(_registry.active())
    if job_count:
        st.session_state["_last_active_at"] = datetime.now().timestamp()

    new_runs = _fetch_new_scheduled_runs()
    if new_runs:
        _render_live_runs_in(new_runs, live_container.markdown)

    # No active jobs and past the grace window: tear the poller down via an app
    # rerun. The main script will see stale `_last_active_at` and not re-declare
    # the fragment.
    if not job_count and (
        datetime.now().timestamp()
        - st.session_state.get("_last_active_at", 0.0)
        > 15
    ):
        st.rerun()


# ---- Helpers -----------------------------------------------------------------


# ---- LLM message construction --------------------------------------------------


def _history_to_langchain(messages: list[dict]) -> list[BaseMessage]:
    """Convert the persisted chat history to LangChain messages for the LLM.

    Scheduled-run cards (raw HTML) and empty entries are skipped. The newest
    messages are kept first, subject to `HISTORY_MAX_MESSAGES` / `HISTORY_MAX_CHARS`
    budgets, so long chats are trimmed instead of overflowing the context window.
    """
    kept: list[BaseMessage] = []
    used = 0
    for message in reversed(messages):
        if message.get("kind") == "scheduled_run":
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        if len(kept) >= HISTORY_MAX_MESSAGES:
            break
        content = str(content)
        if used + len(content) > HISTORY_MAX_CHARS:
            remaining = HISTORY_MAX_CHARS - used
            if remaining <= 0:
                break
            content = content[:remaining]
        used += len(content)
        kept.append(
            (HumanMessage if role == "user" else AIMessage)(content=content)
        )
    kept.reverse()
    return kept


def _build_llm_messages(prompt: str, plan: SearchPlan, history: list[dict]) -> list[SystemMessage]:
    """Assemble the full message list for one chatbot invocation: system context
    (RAG source label or raw document, plus the planner's search instruction),
    the trimmed prior conversation, then the user's current prompt."""
    system_messages: list[SystemMessage] = []
    if st.session_state.get("rag_mode", True):
        # RAG on: tell the LLM which uploaded document the RAG tool can see so
        # it reaches for get_rag_chunks when the question targets that PDF.
        source_label = ragtool.active_source_label()
        if source_label:
            system_messages.append(SystemMessage(content=source_label))
    else:
        # RAG off: hand the WHOLE PDF to the model — no chunking, no embedding.
        full_text = ragtool.retrieve_full_text()
        if full_text:
            system_messages.append(
                SystemMessage(
                    content=(
                        "The user's active document is provided below IN FULL "
                        "(RAG is disabled — nothing was chunked or embedded). "
                        "Answer questions about it strictly from this text and "
                        "do not call any retrieval tools.\n\n" + full_text
                    )
                )
            )
    system_messages.append(
        SystemMessage(content=SearchExecutionInstruction(plan.search_query).render())
    )
    return system_messages + _history_to_langchain(history) + [HumanMessage(content=prompt)]


def _run_chatbot(prompt: str, plan: SearchPlan, history: list[dict]) -> str:
    """One-off invocation of the LangGraph chatbot (non-streaming fallback)."""
    out = chatbot.invoke({"messages": _build_llm_messages(prompt, plan, history)})
    return _extract_content(out)


def _stream_chatbot(prompt: str, plan: SearchPlan, history: list[dict]):
    """Stream the chatbot's reply token by token from the LangGraph graph.

    Yields `(text_chunk, usage_metadata | None)` pairs. If the graph or model
    cannot stream (network hiccup, unsupported mode), falls back to a single
    non-streamed invocation and yields the whole reply as one chunk.
    """
    inputs = {"messages": _build_llm_messages(prompt, plan, history)}
    try:
        saw_chunk = False
        for chunk, _metadata in chatbot.stream(inputs, stream_mode="messages"):
            saw_chunk = True
            text = _content_to_text(getattr(chunk, "content", ""))
            usage = getattr(chunk, "usage_metadata", None) or None
            if text or usage:
                yield text, usage
        if not saw_chunk:
            log.warning("Streaming produced no chunks for %r; using invoke()", prompt)
            yield _run_chatbot(prompt, plan, history), None
    except Exception:  # noqa: BLE001 - degraded, not fatal
        log.exception("Streaming failed for prompt %r; falling back to invoke()", prompt)
        yield _run_chatbot(prompt, plan, history), None


def _auto_title_conversation(history: list[dict], current_title: str) -> str:
    """Ask the LLM for a short conversation title; best-effort and never fatal.

    Falls back to `current_title` when the call fails or the model refuses to
    produce a usable title.
    """
    user_text = " / ".join(
        " ".join(str(m.get("content", "")).split())[:300]
        for m in history
        if m.get("role") == "user" and m.get("content")
    )[:2000]
    if not user_text:
        return current_title
    try:
        out = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You invent short chat titles. Reply with ONLY the "
                        "title: 2-5 words, no punctuation, no quotes, no newlines."
                    )
                ),
                HumanMessage(content=f"Conversation so far:\n{user_text}\n\nTitle:"),
            ]
        )
        # llm.invoke returns a BaseMessage; normalize to plain text.
        raw = getattr(out, "content", None)
        if raw is None:
            raw = _extract_content(out)
        title = " ".join(_content_to_text(raw).split())
        title = title.strip('"\'“”.,').strip()
        if title and len(title) <= 40:
            return title
    except Exception:  # noqa: BLE001 - auto-titling is optional
        log.exception("Auto-title generation failed; keeping %r", current_title)
    return current_title


def _user_friendly_error(exc: Exception) -> str:
    """Render a user-safe error line; the full traceback goes to the log."""
    return (
        f"⚠️ The model call failed (`{type(exc).__name__}`). Check your Gemini "
        "API key and quota, then try again. Details were logged."
    )


def _fetch_new_scheduled_runs() -> list[dict]:
    """Read any new run files from `.hakathon/history/` since last check."""
    updates: list[dict] = []
    history_root = _registry.history_dir
    if not history_root.is_dir():
        return updates
    for job_dir in history_root.iterdir():
        if not job_dir.is_dir():
            continue
        for run_file in sorted(job_dir.glob("run-*.json")):
            key = str(run_file)
            if key in st.session_state.seen_runs:
                continue
            st.session_state.seen_runs.add(key)
            try:
                payload = json.loads(run_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            updates.append(
                {
                    "job_id": payload.get("job_id", job_dir.name),
                    "run_number": payload.get("run_number", 0),
                    "completed_at": payload.get("completed_at", ""),
                    "content": payload.get("content", ""),
                }
            )
    updates.sort(key=lambda u: u["completed_at"])
    return updates


def _plan_summary_html(plan: SearchPlan) -> str:
    parts = [
        f'<span class="kw">query</span> <span class="val">{plan.search_query!r}</span>',
        f'<span class="kw">wait</span> <span class="val">{plan.wait_minutes:g}m</span>',
        f'<span class="kw">runs</span> <span class="val">{plan.run_count}</span>',
    ]
    if plan.absolute_start_iso:
        parts.append(f'<span class="kw">start</span> <span class="val">{plan.absolute_start_iso}</span>')
    return " · ".join(parts)


def _handle_pdf_upload(uploaded, *, index: bool) -> None:
    """Save an uploaded PDF; index it for RAG or store it raw (no embeddings)."""
    data = uploaded.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    if st.session_state.get("_pdf_upload_sha") == sha:
        return  # same file re-offered on a rerun — already handled
    st.session_state["_pdf_upload_sha"] = sha

    uploads_dir = config.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", uploaded.name or "document.pdf")
    target = uploads_dir / f"{sha[:16]}_{safe_name}"
    try:
        target.write_bytes(data)
    except OSError as exc:
        st.error(f"Could not save upload: {exc}")
        return

    if index:
        with st.status(f"indexing `{uploaded.name}`…", expanded=True) as status:
            bar = st.progress(0.0, text="embedding chunks…")

            def _report(done: int, total: int) -> None:
                bar.progress(min(done / max(total, 1), 1.0))

            try:
                count = ragtool.index_pdf(target, progress=_report, name=uploaded.name)
            except Exception:  # noqa: BLE001
                log.exception("PDF indexing failed for %s", target)
                status.update(label="indexing failed", state="error")
                st.error(
                    "Indexing failed — check the Jina API key / quota. Details were logged."
                )
                return
            ragtool.set_active_collection(ragtool.collection_name_for_sha(sha))
            status.update(label=f"indexed {count} chunks", state="complete")
    else:
        with st.status(f"reading `{uploaded.name}`…", expanded=True) as status:
            try:
                meta = ragtool.store_raw_pdf(target, name=uploaded.name)
            except Exception:  # noqa: BLE001
                log.exception("Raw PDF read failed for %s", target)
                status.update(label="reading failed", state="error")
                st.error("Could not read PDF — is it a valid file? Details were logged.")
                return
            status.update(
                label=f"stored {meta['pages']} pages · no chunking, no embeddings",
                state="complete",
            )
    st.rerun()


# ---- Header ------------------------------------------------------------------

st.markdown(
    """
<div class="term-header">
  <span class="dim">$</span> HAKATHON <span class="accent">//</span> chat
</div>
<div class="term-sub"><span class="dot"></span> LangGraph + Gemini · dark terminal build</div>
""",
    unsafe_allow_html=True,
)


# ---- Sidebar (health, RAG documents, reset) ---------------------------------

with st.sidebar:
    st.markdown("### chats")

    if st.button("＋ new chat", use_container_width=True):
        conv_id = _create_conversation(st.session_state.conversations)
        st.session_state.active_conv = conv_id
        st.session_state.pop("conv_picker", None)
        _persist_conversations(
            st.session_state.conversations,
            st.session_state.active_conv,
            st.session_state.job_conv,
        )
        st.rerun()

    chat_id_order = list(st.session_state.conversations)
    chat_labels = {
        cid: conv.get("title") or "new chat"
        for cid, conv in st.session_state.conversations.items()
    }
    try:
        picker_index = chat_id_order.index(st.session_state.active_conv)
    except ValueError:
        picker_index = 0
    picked_chat = st.radio(
        "chat",
        options=chat_id_order,
        index=picker_index,
        format_func=lambda cid: chat_labels.get(cid, cid),
        key="conv_picker",
        label_visibility="collapsed",
    )
    if picked_chat != st.session_state.active_conv:
        st.session_state.active_conv = picked_chat
        _persist_conversations(
            st.session_state.conversations,
            st.session_state.active_conv,
            st.session_state.job_conv,
        )
        st.rerun()

    if st.button("delete chat", use_container_width=True, disabled=len(chat_id_order) <= 1):
        st.session_state.conversations.pop(st.session_state.active_conv, None)
        if not st.session_state.conversations:
            _create_conversation(st.session_state.conversations)
        st.session_state.active_conv = next(iter(st.session_state.conversations))
        st.session_state.pop("conv_picker", None)
        _persist_conversations(
            st.session_state.conversations,
            st.session_state.active_conv,
            st.session_state.job_conv,
        )
        st.rerun()

    st.divider()
    st.markdown("### session")

    model_labels = list(config.GEMINI_MODEL_OPTIONS)
    default_label = next(
        (
            label
            for label, model_id in config.GEMINI_MODEL_OPTIONS.items()
            if model_id == config.GEMINI_MODEL
        ),
        model_labels[0],
    )
    picked_model = st.selectbox(
        "model",
        options=model_labels,
        index=model_labels.index(default_label),
        key="model_picker",
    )
    # `app.llm` already uses config.GEMINI_MODEL at import, so the first run
    # must NOT rebuild it — only rebuild when the user actively switches models.
    if "_active_model_label" not in st.session_state:
        st.session_state["_active_model_label"] = picked_model
    elif picked_model != st.session_state["_active_model_label"]:
        set_chatbot_model(config.GEMINI_MODEL_OPTIONS[picked_model])
        st.session_state["_active_model_label"] = picked_model
    st.caption(f"active: **{st.session_state['_active_model_label']}**")
    if st.session_state.get("rag_mode", True):
        st.caption(f"embedding: `{config.JINA_EMBEDDING_MODEL}`")
    else:
        st.caption("embedding: off (raw mode)")
    active = _registry.active()
    st.caption(f"active schedules: **{len(active)}**")
    if active:
        for job in active[:6]:
            st.caption(
                f"• `{job.id}` — {job.status} — {job.completed_runs}/{job.run_count or '?'}"
            )
    st.divider()

    # ---- Documents: RAG (chunk+embed) or raw (whole PDF to the model) ------
    st.markdown("### documents")
    rag_mode = st.toggle(
        "RAG mode",
        value=st.session_state.get("rag_mode", True),
        key="rag_mode",
        help=(
            "On: PDFs are chunked, embedded, and searched via get_rag_chunks. "
            "Off: no chunking or embedding — the whole PDF goes to the model."
        ),
    )

    if rag_mode:
        docs = ragtool.list_indexed_documents()
        doc_options = [None] + [d["collection"] for d in docs]
        doc_labels = {None: "no document"}
        for doc in docs:
            doc_labels[doc["collection"]] = f'{doc["name"]} · {doc["chunks"]} chunks'
        current = ragtool.active_collection()
        if current is None and docs:
            # Nothing selected (e.g. fresh process after a restart): fall back to
            # the most recently indexed document so RAG works immediately.
            current = docs[0]["collection"]
            ragtool.set_active_collection(current)
        try:
            picker_index = doc_options.index(current)
        except ValueError:
            picker_index = 0
        picked = st.selectbox(
            "RAG source",
            options=doc_options,
            index=picker_index,
            format_func=lambda c: doc_labels.get(c, str(c)),
            key="rag_source_picker",
            help="The document the LLM's get_rag_chunks tool queries.",
        )
        if picked != current:
            ragtool.set_active_collection(picked)
            st.rerun()

        uploaded = st.file_uploader("Upload a PDF to index", type=["pdf"], key="pdf_uploader")
        if uploaded is not None:
            _handle_pdf_upload(uploaded, index=True)

        if st.button(
            "remove active document",
            use_container_width=True,
            disabled=picked is None,
        ):
            ragtool.remove_indexed_document(picked)
            st.rerun()
    else:
        raw_id = ragtool.active_raw()
        if raw_id:
            meta = ragtool.raw_meta(raw_id)
            label = f'{meta["name"]} · {meta["pages"]} pages' if meta else raw_id
            st.caption(f"active: **{label}**")
            st.caption("passed to the model in full · no chunking, no embeddings")
        else:
            st.caption("no document — upload one below")

        uploaded = st.file_uploader(
            "Upload a PDF (raw — no chunking/embedding)",
            type=["pdf"],
            key="raw_pdf_uploader",
        )
        if uploaded is not None:
            _handle_pdf_upload(uploaded, index=False)

        if st.button(
            "remove active document",
            use_container_width=True,
            disabled=raw_id is None,
        ):
            ragtool.remove_raw_document(raw_id)
            st.rerun()

    st.divider()
    _active_chat_for_export = st.session_state.conversations[st.session_state.active_conv]
    st.download_button(
        "export chat (.md)",
        data=_conversation_to_markdown(_active_chat_for_export["messages"]),
        file_name=f'chat-{st.session_state.active_conv}.md',
        mime="text/markdown",
        use_container_width=True,
        key=f"export_md_{st.session_state.active_conv}",
    )
    st.download_button(
        "export chat (.json)",
        data=json.dumps(
            {
                "title": _active_chat_for_export.get("title", "new chat"),
                "messages": _active_chat_for_export["messages"],
            },
            indent=2,
        ),
        file_name=f'chat-{st.session_state.active_conv}.json',
        mime="application/json",
        use_container_width=True,
        key=f"export_json_{st.session_state.active_conv}",
    )
    st.divider()
    if st.button("clear chat", use_container_width=True):
        chat = st.session_state.conversations[st.session_state.active_conv]
        chat["messages"] = []
        chat["title"] = "new chat"
        _persist_conversations(
            st.session_state.conversations,
            st.session_state.active_conv,
            st.session_state.job_conv,
        )
        st.rerun()
    if st.button("cancel all schedules", use_container_width=True):
        for job in _registry.active():
            job.stop()
        st.toast(f"Cancelled {len(_registry.active())} job(s)")


# ---- Render chat history -----------------------------------------------------

active_chat = st.session_state.conversations[st.session_state.active_conv]
for message in active_chat["messages"]:
    role = message["role"]
    # Render pre-formatted scheduled-run cards as raw HTML+markdown; keep
    # regular chat bubbles as plain st.markdown so they get the bubble style.
    if role == "assistant" and message.get("kind") == "scheduled_run":
        st.markdown(message["content"], unsafe_allow_html=True)
        continue
    with st.chat_message(role):
        st.markdown(message["content"])
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else None
        if usage:
            # Gemini reports usage_metadata on the final chunk; show a subtle
            # token count under the reply as a session-visibility aid.
            tokens_in = usage.get("input_tokens")
            tokens_out = usage.get("output_tokens")
            if tokens_in is not None or tokens_out is not None:
                parts = []
                if tokens_in is not None:
                    parts.append(f"in {tokens_in:,}")
                if tokens_out is not None:
                    parts.append(f"out {tokens_out:,}")
                st.caption(" · ".join(parts) + " tokens")


# ---- Active schedules status strip ------------------------------------------
# Show a compact chip row for every active job so users see progress even
# when the current run is still in flight (i.e. no history file yet).

if _active_now:
    def _status_class(status: str) -> str:
        return status if status in ("running", "paused", "pending") else ""

    chips = []
    for job in _active_now:
        progress = f"{job.completed_runs}/{job.run_count or '?'}"
        next_run = job.next_run_at or "—"
        # Trim ISO string to HH:MM for display
        try:
            next_run_short = (
                datetime.fromisoformat(next_run.replace("Z", "+00:00")).strftime("%H:%M")
                if next_run != "—" else "—"
            )
        except (ValueError, TypeError, AttributeError):
            next_run_short = next_run
        chips.append(
            f'<span class="sched-chip">'
            f'<span class="sc-id">◉ {job.id}</span>'
            f'<span class="sc-status {_status_class(job.status)}">{job.status}</span>'
            f'<span class="sc-progress">{progress}</span>'
            f'<span class="sc-next">next {next_run_short}</span>'
            f'</span>'
        )
    st.markdown(
        f'<div class="sched-strip">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


# ---- Handle any newly-completed scheduled runs ------------------------------


def _render_scheduled_run_card(run: dict) -> str:
    """Return the HTML string for a single scheduled-run card."""
    # Human-friendly time (HH:MM:SS).
    ts_short = run["completed_at"]
    try:
        ts_short = datetime.fromisoformat(
            run["completed_at"].replace("Z", "+00:00")
        ).strftime("%H:%M:%S")
    except (ValueError, AttributeError):
        pass
    return (
        f'<div class="sched-card">'
        f'<div class="sched-head">'
        f'<span class="sched-badge">scheduled run</span>'
        f'<span class="sched-jobid">{run["job_id"]}</span>'
        f'<span class="sched-runno">run #{run["run_number"]}</span>'
        f'<span class="sched-time">{ts_short}</span>'
        f'</div>'
        f'<div class="sched-body">{_markdown_to_html(run["content"])}</div>'
        f'</div>'
    )


def _markdown_to_html(text: str) -> str:
    """Very lightweight markdown → HTML for the sched-card body.

    We deliberately do not pull in a full markdown lib — the LLM output is
    usually simple lists and bold spans. Falls back to `<br>`-preserved text.
    """
    import html
    import re
    escaped = html.escape(text)
    # **bold**
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    # `code`
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    # newlines
    escaped = escaped.replace("\n", "<br>")
    return escaped


new_runs = _fetch_new_scheduled_runs()
if new_runs:
    # Render completed runs as inline cards in the chat flow. Runs completed
    # *while the page is idle* are handled by the background poller instead.
    _render_live_runs_in(new_runs, st.markdown)


# ---- Chat input --------------------------------------------------------------

# Declare the background poller only while a job is active (or within the grace
# window). The container is created here — below the history — so live cards
# accumulate at the bottom of the page, above the input.
if _active_now or _grace_active:
    _live_container = st.container()
    _poll_scheduled_runs(_live_container)

prompt = st.chat_input("ask anything · try 'search AI news every 5 min for 3 times'")

if prompt:
    # Show the user's message immediately in the active conversation.
    active_chat = st.session_state.conversations[st.session_state.active_conv]
    history_before = list(active_chat["messages"])
    active_chat["messages"].append({"role": "user", "content": prompt})
    auto_title_pending = active_chat["title"] == "new chat"
    if auto_title_pending:
        active_chat["title"] = _conversation_title(active_chat["messages"])
    with st.chat_message("user"):
        st.markdown(prompt)

    # Plan the request (chooses schedule vs. one-off).
    try:
        plan = create_search_plan(prompt)
    except Exception:  # noqa: BLE001
        log.exception("Planner failed for prompt %r", prompt)
        plan = SearchPlan(
            search_query=prompt,
            should_schedule=False,
            wait_minutes=0.0,
            run_count=1,
        )
        st.warning("Planner error — falling back to a single run. Details logged.")

    with st.chat_message("assistant"):
        # Show planner summary as a subtle line above the answer.
        st.markdown(
            f'<div class="plan-line">▸ {_plan_summary_html(plan)}</div>',
            unsafe_allow_html=True,
        )

        if plan.should_schedule:
            try:
                job = scheduler.start(
                    prompt=prompt,
                    interval_minutes=plan.wait_minutes or None,
                    run_count=plan.run_count,
                    search_query=plan.search_query,
                    absolute_start_iso=plan.absolute_start_iso,
                    task_type=plan.task_type,
                    reminder_text=plan.reminder_text,
                )
                start_text = (
                    f"**Schedule started** · id `{job.id}` · "
                    f"{plan.run_count} run(s) every {plan.wait_minutes:g} min"
                )
                if plan.absolute_start_iso:
                    start_text += f" · first run at `{plan.absolute_start_iso}`"
                start_text += (
                    "\n\nEach completed run will appear here automatically. "
                    "Use the sidebar to cancel."
                )
                st.markdown(start_text)
                active_chat["messages"].append(
                    {"role": "assistant", "content": start_text}
                )
                # Route completed runs of this job back to this chat, even if
                # the user switches conversations before the job finishes.
                st.session_state.job_conv[job.id] = st.session_state.active_conv
                _persist_conversations(
                    st.session_state.conversations,
                    st.session_state.active_conv,
                    st.session_state.job_conv,
                )
                # Force a rerun so the polling block at the top of the script
                # picks up the new active schedule and starts the poller.
                st.rerun()
            except ScheduleValidationError as exc:
                log.warning("Invalid schedule from planner: %s", exc)
                err = "Invalid schedule — check the phrasing (e.g. `every 5 min for 3 times`)."
                st.error(err)
                active_chat["messages"].append({"role": "assistant", "content": err})
                _persist_conversations(
                    st.session_state.conversations,
                    st.session_state.active_conv,
                    st.session_state.job_conv,
                )
        else:
            # Stream the reply token by token; capture usage metadata from the
            # final chunk so we can show a per-message token count.
            placeholder = st.empty()
            chunks: list[str] = []
            usage: dict | None = None
            try:
                for text_chunk, chunk_usage in _stream_chatbot(prompt, plan, history_before):
                    if text_chunk:
                        chunks.append(text_chunk)
                        placeholder.markdown("".join(chunks))
                    if chunk_usage:
                        usage = chunk_usage
            except Exception as exc:  # noqa: BLE001
                log.exception("Chatbot invocation failed for prompt %r", prompt)
                chunks = [_user_friendly_error(exc)]
                usage = None
            reply = "".join(chunks).strip()
            if not reply:
                reply = "_(no response — the model returned nothing; details logged)_"
            placeholder.markdown(reply)
            if usage:
                st.caption(
                    " · ".join(
                        p
                        for p in (
                            f"in {usage.get('input_tokens'):,}" if usage.get("input_tokens") is not None else "",
                            f"out {usage.get('output_tokens'):,}" if usage.get("output_tokens") is not None else "",
                        )
                        if p
                    )
                    + " tokens"
                )
            message_entry: dict = {"role": "assistant", "content": reply}
            if usage:
                message_entry["usage"] = usage
            active_chat["messages"].append(message_entry)
            _persist_conversations(
                st.session_state.conversations,
                st.session_state.active_conv,
                st.session_state.job_conv,
            )
            # Best-effort LLM title for a brand-new chat (replaces the
            # truncated first-message fallback).
            if auto_title_pending and reply and not reply.startswith("⚠️"):
                better_title = _auto_title_conversation(
                    active_chat["messages"], active_chat["title"]
                )
                if better_title and better_title != active_chat["title"]:
                    active_chat["title"] = better_title
                    _persist_conversations(
                        st.session_state.conversations,
                        st.session_state.active_conv,
                        st.session_state.job_conv,
                    )
