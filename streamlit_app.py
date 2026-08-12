"""Streamlit UI for the HAKATHON LangGraph chatbot.

A single-page, chat-only interface. Replaces the CLI in `app.py` but reuses
every module (config, planner, scheduler, tools, RAG, LangGraph wiring).

Run with:

    streamlit run streamlit_app.py

Design: dark terminal aesthetic, monospace headers, minimal chrome.
Scheduled runs still work — start one from chat ("search AI news every 10 min
for 3 times") and completed runs are surfaced back into the chat the next
time you type.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover - only hit when the optional dep is missing
    def st_autorefresh(**kwargs):  # type: ignore[no-redef]
        """Fallback stub used when `streamlit-autorefresh` is not installed.

        The app still works — scheduled runs won't appear until the user types
        or clicks something. Fix by running: `pip install streamlit-autorefresh`.
        """
        st.session_state.setdefault("_autorefresh_missing_warned", False)
        if not st.session_state["_autorefresh_missing_warned"]:
            st.warning(
                "`streamlit-autorefresh` is not installed. Scheduled runs will "
                "only appear after your next interaction. Install it with "
                "`pip install streamlit-autorefresh` for automatic updates.",
                icon="⚠️",
            )
            st.session_state["_autorefresh_missing_warned"] = True
        return 0

# Reuse everything the CLI used.
import config
from app import _build_messages, _registry, chatbot, scheduler
from planner import SearchExecutionInstruction, SearchPlan, create_search_plan
from scheduler import ScheduleValidationError, _extract_content


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


# ---- Session state -----------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}

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


# ---- Auto-refresh ------------------------------------------------------------
# Streamlit only reruns on user interaction, so background scheduled runs would
# never surface unless the user typed something. When there are active schedules
# in the registry, poll every few seconds so completed runs appear on their own.
# We also poll for a short grace window after the *last* active job finishes
# so the final run(s) can flush into the chat.

_active_now = _registry.active()
if _active_now:
    st.session_state["_last_active_at"] = datetime.now().timestamp()

_grace_active = False
_last_seen = st.session_state.get("_last_active_at")
if _last_seen and (datetime.now().timestamp() - _last_seen) < 15:
    _grace_active = True

if _active_now or _grace_active:
    # Interval in milliseconds. Every tick triggers a full script rerun, which
    # re-executes `_fetch_new_scheduled_runs()` below and picks up new files
    # written by the scheduler's daemon threads.
    st_autorefresh(interval=3000, key="poll_history")


# ---- Helpers -----------------------------------------------------------------


def _run_chatbot(prompt: str, plan: SearchPlan) -> str:
    """One-off invocation of the LangGraph chatbot (non-scheduled path)."""
    out = chatbot.invoke(
        {
            "messages": [
                SystemMessage(content=SearchExecutionInstruction(plan.search_query).render()),
                HumanMessage(content=prompt),
            ]
        }
    )
    return _extract_content(out)


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


# ---- Sidebar (minimal — just health + reset) ---------------------------------

with st.sidebar:
    st.markdown("### session")
    st.caption(f"model: `{config.GEMINI_MODEL}`")
    st.caption(f"embedding: `{config.GEMINI_EMBEDDING_MODEL}`")
    active = _registry.active()
    st.caption(f"active schedules: **{len(active)}**")
    if active:
        for job in active[:6]:
            st.caption(
                f"• `{job.id}` — {job.status} — {job.completed_runs}/{job.run_count or '?'}"
            )
    st.divider()
    if st.button("clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("cancel all schedules", use_container_width=True):
        for job in _registry.active():
            job.stop()
        st.toast(f"Cancelled {len(_registry.active())} job(s)")


# ---- Render chat history -----------------------------------------------------

for message in st.session_state.messages:
    role = message["role"]
    # Render pre-formatted scheduled-run cards as raw HTML+markdown; keep
    # regular chat bubbles as plain st.markdown so they get the bubble style.
    if role == "assistant" and message.get("kind") == "scheduled_run":
        st.markdown(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(role):
            st.markdown(message["content"])


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
for run in new_runs:
    card_html = _render_scheduled_run_card(run)
    st.markdown(card_html, unsafe_allow_html=True)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "kind": "scheduled_run",
            "content": card_html,
        }
    )


# ---- Chat input --------------------------------------------------------------

prompt = st.chat_input("ask anything · try 'search AI news every 5 min for 3 times'")

if prompt:
    # Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Plan the request (chooses schedule vs. one-off).
    try:
        plan = create_search_plan(prompt)
    except Exception as exc:  # noqa: BLE001
        plan = SearchPlan(
            search_query=prompt,
            should_schedule=False,
            wait_minutes=0.0,
            run_count=1,
        )
        st.warning(f"Planner error: {exc}. Falling back to a single run.")

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
                st.session_state.messages.append(
                    {"role": "assistant", "content": start_text}
                )
                # Force a rerun so the auto-refresh block at the top of the
                # script picks up the new active schedule and starts polling.
                st.rerun()
            except ScheduleValidationError as exc:
                err = f"Invalid schedule: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            with st.spinner("thinking..."):
                try:
                    reply = _run_chatbot(prompt, plan)
                except Exception as exc:  # noqa: BLE001
                    reply = f"⚠️ Error talking to Gemini: `{exc}`"
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
