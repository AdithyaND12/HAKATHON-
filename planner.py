"""Local (regex-only) search + scheduling planner.

The planner turns a natural-language user prompt into a strict `SearchPlan`
without an LLM call. Regex parsing recognizes schedules (interval, count,
absolute start), and a lightweight classifier assigns the task type so runs
can be routed to the right execution strategy — search / reminder /
calculation / rag / chat.

`SearchPlan` carries:
    * `task_type` — routes each run to the appropriate handler.
    * `reminder_text` — populated for `task_type='reminder'`.
    * `absolute_start` — ISO timestamp for one-shot times ("at 3pm",
      "tomorrow 9am", "in 5 minutes").
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

import config

log = logging.getLogger(__name__)


# ---- Data ---------------------------------------------------------------------


TaskType = Literal["search", "reminder", "calculation", "rag", "chat"]


class SearchPlan(BaseModel):
    """The task and scheduling decisions extracted from one user prompt.

    The name is kept as `SearchPlan` for backward compatibility with existing
    imports; the underlying model now carries a `task_type` so the runtime can
    route scheduled runs to different execution strategies.
    """

    task_type: TaskType = Field(
        default="search",
        description=(
            "The kind of task this run performs. 'search' calls web_search "
            "(default). 'reminder' has no tool call — the LLM just writes a "
            "short reminder line. 'calculation' calls the calculator tool. "
            "'rag' calls the user's active-document retrieval tool. 'chat' asks the "
            "LLM directly with no tool nudging."
        ),
    )
    search_query: str = Field(
        description="A concise web-search query that directly answers the user's request."
    )
    reminder_text: Optional[str] = Field(
        default=None,
        description=(
            "Only used when task_type='reminder'. The specific thing to remind "
            "the user about, in plain natural language (e.g. 'drink water')."
        ),
    )
    should_schedule: bool = Field(
        description=(
            "True only when the user asks to repeat, monitor, or check the task later."
        )
    )
    wait_minutes: float = Field(
        description=(
            "Minutes between repeated runs. Use the user's explicit interval; "
            "use 60 when repetition is requested without an interval; use 0 otherwise."
        )
    )
    run_count: int = Field(
        description=(
            "Total number of runs. Use the user's requested count; use 2 for an "
            "open-ended repeat request; use 1 otherwise."
        )
    )
    absolute_start_iso: Optional[str] = Field(
        default=None,
        description=(
            "Optional ISO-8601 timestamp for when the first run should fire. "
            "Set for requests like 'at 3pm' or 'tomorrow 9am'. None otherwise."
        ),
    )


DEFAULT_AUTO_WAIT_MINUTES = 60.0



# ---- Regex fallback -----------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "sixty": 60,
    "once": 1, "twice": 2, "thrice": 3,
}
_NUMBER_TOKEN = (
    r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|twenty|thirty|sixty|once|twice|thrice)"
)
_DURATION_RE = re.compile(
    rf"\b(?P<value>{_NUMBER_TOKEN})\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    rf"\b(?P<count>{_NUMBER_TOKEN})\s*(?:times?|runs?|checks?|iterations?)\b",
    re.IGNORECASE,
)
_EVERY_INTERVAL_RE = re.compile(
    rf"\b(?:every|each)\s+(?P<value>{_NUMBER_TOKEN})\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE,
)
_HOURLY_RE = re.compile(r"\b(?:hourly|every\s+hour)\b", re.IGNORECASE)
_DAILY_RE = re.compile(r"\b(?:daily|every\s+day)\b", re.IGNORECASE)
_ABSOLUTE_TIME_RE = re.compile(
    r"""
    \b(?P<at>at\s+)?
    (?P<hour>\d{1,2})           # 3 or 15
    (?::(?P<minute>\d{2}))?     # optional :30
    \s*(?P<ampm>am|pm)?         # optional am/pm
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
_TONIGHT_RE = re.compile(r"\btonight\b", re.IGNORECASE)
_IN_MINUTES_RE = re.compile(
    rf"\b(?:in|after)\s+(?P<value>{_NUMBER_TOKEN})\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)
_SCHEDULE_HINT_RE = re.compile(
    r"\b(?:every|each|repeat|monitor|refresh|again|later|hourly|daily|"
    r"at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|tomorrow|tonight)\b",
    re.IGNORECASE,
)

def _parse_number_token(value: str) -> float:
    normalized = value.lower()
    if normalized in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[normalized])
    return float(normalized)


def _unit_to_minutes(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit.startswith("second") or unit.startswith("sec"):
        return value / 60
    if unit.startswith("hour") or unit.startswith("hr"):
        return value * 60
    if unit.startswith("day"):
        return value * 60 * 24
    return value  # minutes


def _local_wall_time_to_utc(naive: datetime) -> datetime:
    """Resolve a naive local wall-clock time to UTC, honoring DST.

    `datetime.now().astimezone()` carries a fixed offset, so adding a day
    across a DST boundary lands an hour off. `time.mktime` asks the OS to
    resolve the local wall time with the correct offset for that date.
    """
    return datetime.fromtimestamp(time.mktime(naive.timetuple())).astimezone(
        timezone.utc
    )


def _parse_absolute_start(prompt: str) -> Optional[datetime]:
    """Best-effort absolute-time parser. Returns UTC datetime or None."""
    now_local = datetime.now()

    match = _IN_MINUTES_RE.search(prompt)
    if match:
        value = _parse_number_token(match.group("value"))
        minutes = _unit_to_minutes(value, match.group("unit"))
        return datetime.now(timezone.utc) + timedelta(minutes=minutes)

    # Absolute clock time. Require an explicit "at" prefix or am/pm suffix;
    # otherwise bare digits in prompts like "2 + 2" or "tomorrow to buy 2
    # apples" get misread as clock times.
    for time_match in _ABSOLUTE_TIME_RE.finditer(prompt):
        has_at_prefix = bool(time_match.group("at"))
        has_ampm = bool(time_match.group("ampm"))
        if not (has_at_prefix or has_ampm):
            continue
        # Skip if the number is part of "X minutes"/"X times" phrasing
        following = prompt[time_match.end(): time_match.end() + 12].lower()
        if any(
            following.lstrip().startswith(kw)
            for kw in ("minute", "min", "second", "sec", "hour", "hr", "time", "run", "check", "day")
        ):
            continue
        try:
            hour = int(time_match.group("hour"))
        except (TypeError, ValueError):
            continue
        minute = int(time_match.group("minute") or 0)
        ampm = (time_match.group("ampm") or "").lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            continue

        base = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if _TOMORROW_RE.search(prompt):
            base += timedelta(days=1)
        elif _TONIGHT_RE.search(prompt) and hour < 18:
            base += timedelta(days=1)
        elif base <= now_local:
            base += timedelta(days=1)
        return _local_wall_time_to_utc(base)

    if _TOMORROW_RE.search(prompt):
        # No explicit time — default to 9am tomorrow
        target = (now_local + timedelta(days=1)).replace(
            hour=9, minute=0, second=0, microsecond=0
        )
        return _local_wall_time_to_utc(target)

    return None


# Removes "every N minutes", "for N times", "hourly", "at 3pm", "tomorrow",
# "in 5 minutes", and other scheduling verbs from a prompt. Shared between the
# planner and app.py's scheduled-run sanitizer so the two never drift apart.
_SCHEDULING_WORDS = (
    r"hourly|daily|weekly|monthly|repeatedly|periodically|recurring|"
    r"repeat|monitor|refresh|schedule"
)


def strip_schedule_phrases(prompt: str) -> str:
    """Return `prompt` with scheduling directive phrases removed.

    Keeps the task words ("search bike info") and drops everything that looks
    like a schedule ("every 1 minute for 3 times", "tomorrow at 9am").
    """
    text = re.sub(
        rf"\b(?:for\s+)?(?:every|each)\s+{_NUMBER_TOKEN}\s*"
        r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
        " ", prompt, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:every|each)\s+(?:\S+\s+)?(?:hour|day|week|month)\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\b(?:for\s+)?{_NUMBER_TOKEN}\s*(?:times?|runs?|checks?|iterations?)\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(rf"\b(?:{_SCHEDULING_WORDS})\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:in|after)\s+\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?)\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:tomorrow|tonight)\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ,.!?")


def _fallback_search_plan(prompt: str) -> SearchPlan:
    """Parse the task type and schedule from the prompt with regexes."""
    duration_match = _DURATION_RE.search(prompt)
    every_hint = bool(re.search(r"\b(?:every|each)\b", prompt, re.IGNORECASE))
    hourly_hint = bool(_HOURLY_RE.search(prompt))
    daily_hint = bool(_DAILY_RE.search(prompt))
    in_duration_hint = bool(_IN_MINUTES_RE.search(prompt))
    repeat_verb_hint = bool(
        re.search(r"\b(?:repeat|monitor|refresh)\b", prompt, re.IGNORECASE)
    )
    absolute_start = _parse_absolute_start(prompt)

    # A duration alone is not a schedule ("the movie is 90 minutes long").
    # Require an explicit recurrence, a delay ("in 5 minutes"), or an
    # absolute start time.
    recurrence_hint = bool(
        (every_hint and (duration_match or re.search(
            r"\b(?:hour|day|week|month)\b", prompt, re.IGNORECASE
        )))
        or hourly_hint
        or daily_hint
        or in_duration_hint
        or repeat_verb_hint
    )
    schedule_requested = bool(
        recurrence_hint
        or absolute_start
        or re.search(r"\b(?:again|later)\b", prompt, re.IGNORECASE)
    )

    wait_minutes = 0.0
    if duration_match and every_hint:
        value = _parse_number_token(duration_match.group("value"))
        wait_minutes = _unit_to_minutes(value, duration_match.group("unit"))
    elif hourly_hint:
        wait_minutes = 60.0
    elif daily_hint:
        wait_minutes = 60.0 * 24
    elif schedule_requested and not absolute_start:
        wait_minutes = DEFAULT_AUTO_WAIT_MINUTES

    count_match = _COUNT_RE.search(prompt)
    run_count = int(_parse_number_token(count_match.group("count"))) if count_match else 1
    # Hourly/daily/monitor requests are open-ended; keep them bounded at two
    # runs instead of silently doing a single run.
    if recurrence_hint and not count_match and not absolute_start:
        run_count = 2

    # An absolute start counts as scheduling too, even without "every".
    if absolute_start and not schedule_requested:
        schedule_requested = True
    if absolute_start and not every_hint and not count_match:
        # "remind me at 3pm" / "in 2 minutes" - one-shot at that instant.
        run_count = 1
        wait_minutes = 0.0

    # Clean the schedule/count phrases out of the query.
    query = strip_schedule_phrases(prompt) or prompt.strip()

    # ---- Task-type detection --------------------------------------------------
    task_type, reminder_text = _detect_task_type(prompt, query)

    return SearchPlan(
        task_type=task_type,
        search_query=query,
        reminder_text=reminder_text,
        should_schedule=schedule_requested,
        wait_minutes=wait_minutes,
        run_count=run_count,
        absolute_start_iso=absolute_start.isoformat() if absolute_start else None,
    )


# ---- Task-type detection -----------------------------------------------------

_REMINDER_RE = re.compile(
    r"\b(?:remind|reminder|remember|notify|ping|alert|nudge|wake\s+me)\b",
    re.IGNORECASE,
)
_CALCULATION_RE = re.compile(
    r"\b(?:calculate|compute|what(?:'s|\s+is)\s+\d)|"
    r"^\s*[-+*/\d\s().]+\s*[?]?\s*$",
    re.IGNORECASE,
)
_RAG_RE = re.compile(
    r"\b(?:constitution|article\s+\d+|fundamental\s+rights?|"
    r"directive\s+principles?|preamble)\b",
    re.IGNORECASE,
)
_CHAT_RE = re.compile(
    r"\b(?:tell\s+me\s+a\s+joke|write\s+(?:a|me)\s+(?:poem|haiku|story|song)|"
    r"give\s+me\s+a\s+(?:joke|poem)|make\s+up|imagine|pretend)\b",
    re.IGNORECASE,
)


def _detect_task_type(prompt: str, cleaned_query: str) -> tuple[TaskType, Optional[str]]:
    """Best-effort intent detection used by the local planner.

    Returns (task_type, reminder_text). reminder_text is only non-None for
    reminders.
    """
    if _REMINDER_RE.search(prompt):
        # Extract the thing to remind about — strip "remind me to/about/that"
        text = re.sub(
            r"\b(?:please\s+)?(?:remind|notify|ping|alert|nudge|wake)"
            r"(?:\s+me)?\s*(?:to|about|that|of)?\s*",
            "", cleaned_query, flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+using\s+the\s+tools?\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" ,.!?")
        return "reminder", (text or cleaned_query or "your reminder")

    if _CALCULATION_RE.search(prompt):
        return "calculation", None

    if _RAG_RE.search(prompt):
        return "rag", None

    if _CHAT_RE.search(prompt):
        return "chat", None

    return "search", None


# ---- LLM client + retries ----------------------------------------------------


def build_chat_llm(model: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Build the chat LLM client with the configured API key and timeout.

    This is the single construction point for the chat model so the app and
    the CLI share the same settings.
    """
    return ChatGoogleGenerativeAI(
        model=model or config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        request_timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_CLIENT_MAX_RETRIES,
    )


LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_SECONDS = 1.0

_TRANSIENT_API_ERROR_RE = re.compile(
    r"\b(?:429|50[0-9])\b|UNAVAILABLE|RESOURCE_EXHAUSTED|high demand|rate limit",
    re.IGNORECASE,
)


def is_transient_api_error(exc: BaseException) -> bool:
    """True for server-side / rate-limit errors worth retrying."""
    return bool(_TRANSIENT_API_ERROR_RE.search(str(exc)))


def _invoke_with_transient_retry(call, attempts: int = LLM_MAX_RETRIES,
                                 backoff: float = LLM_RETRY_BACKOFF_SECONDS):
    """Retry a callable only on transient API errors (429/5xx); re-raise others."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts or not is_transient_api_error(exc):
                raise
            sleep_for = backoff * (2 ** (attempt - 1))
            log.warning(
                "Transient API error on attempt %d/%d (%s). Retrying in %.1fs.",
                attempt, attempts, exc, sleep_for,
            )
            time.sleep(sleep_for)


def _normalize_absolute_start(value: object) -> Optional[str]:
    """Validate an ISO timestamp and normalize it to UTC.

    LLM output occasionally omits the timezone or returns a non-ISO string.
    Naive timestamps are interpreted as local wall time; unparseable values
    are discarded so a malformed plan cannot fire at an arbitrary moment.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        when = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = _local_wall_time_to_utc(when)
    return when.astimezone(timezone.utc).isoformat()


def _clamp_plan(plan: SearchPlan) -> SearchPlan:
    query = plan.search_query.strip() or "current information"
    try:
        wait_minutes = float(plan.wait_minutes)
    except (TypeError, ValueError):
        wait_minutes = 0.0
    if not math.isfinite(wait_minutes) or wait_minutes < 0:
        wait_minutes = 0.0

    try:
        run_count = int(plan.run_count)
    except (TypeError, ValueError):
        run_count = 1
    run_count = max(1, min(run_count, config.MAX_AUTO_RUNS))

    absolute_start_iso = _normalize_absolute_start(plan.absolute_start_iso)
    should_schedule = bool(plan.should_schedule) or bool(absolute_start_iso)
    if should_schedule and wait_minutes == 0 and not absolute_start_iso:
        wait_minutes = DEFAULT_AUTO_WAIT_MINUTES
    if not should_schedule:
        wait_minutes = 0.0
        run_count = 1

    # Scheduling intervals are independent of the single `wait` tool's bound;
    # "every 6 hours" and "daily" must not silently become hourly.
    wait_minutes = min(wait_minutes, config.MAX_SCHEDULE_INTERVAL_MINUTES)

    # Validate task_type; unknown values collapse to 'search'.
    task_type: TaskType = plan.task_type if plan.task_type in (
        "search", "reminder", "calculation", "rag", "chat"
    ) else "search"
    reminder_text = plan.reminder_text if task_type == "reminder" else None

    return SearchPlan(
        task_type=task_type,
        search_query=query,
        reminder_text=reminder_text,
        should_schedule=should_schedule,
        wait_minutes=wait_minutes,
        run_count=run_count,
        absolute_start_iso=absolute_start_iso,
    )


def create_search_plan(prompt: str) -> SearchPlan:
    """Derive the task type and schedule from the user's prompt (regex-only)."""
    return _clamp_plan(_fallback_search_plan(prompt))


@dataclass
class SearchExecutionInstruction:
    """Renders the instruction the answering LLM must follow for a single run.

    Kept as a lightweight helper for the 'search' path — other task types use
    dedicated helpers below so the LLM is not nudged toward web_search when it
    shouldn't be.
    """

    search_query: str

    def render(self) -> str:
        return (
            "The task planner selected this exact web-search query: "
            f"{self.search_query!r}. Use the web search tool with this query before "
            "answering when the user's request needs current or web-based information. "
            "Do not invent a different query. "
            "After getting search results, synthesize a CONCISE answer — 2-4 sentences "
            "maximum. Do NOT dump raw search results or copy long paragraphs. "
            "Remove citation markers like [1], [2], [j], [k], etc. "
            "If the user asks a simple factual question (e.g. capital, population, "
            "president), give a direct answer in 1 sentence, not an essay. "
            "Do not emit JSON code blocks unless the user explicitly asks for "
            "technical details."
        )


@dataclass
class ReminderExecutionInstruction:
    """Instructs the LLM to fire a plain reminder — no tool call needed."""

    reminder_text: str

    def render(self) -> str:
        return (
            "This scheduled run is a REMINDER. The application has already handled "
            "the timing; your only job is to produce a short, friendly, one-line "
            "reminder message. Do NOT call any tool. Do NOT explain that you cannot "
            "set reminders — you ARE the reminder. "
            f"Remind the user about: {self.reminder_text!r}."
        )


@dataclass
class CalculationExecutionInstruction:
    """Instructs the LLM to route via the calculator tool."""

    expression: str

    def render(self) -> str:
        return (
            "This run is a CALCULATION. Use the calculator tool for arithmetic. "
            f"The expression the user wants evaluated: {self.expression!r}. "
            "Do not use web search."
        )


@dataclass
class RagExecutionInstruction:
    """Instructs the LLM to route via the active document retrieval tool."""

    query: str

    def render(self) -> str:
        return (
            "This run asks about the user's active document. Use the get_rag_chunks "
            f"tool with a query relevant to: {self.query!r}. Do not use web search."
        )


@dataclass
class ChatExecutionInstruction:
    """Instructs the LLM to answer directly with no tool nudging."""

    prompt_summary: str

    def render(self) -> str:
        return (
            "This run is a plain conversational answer — no tools needed. "
            f"Fulfil the user's request: {self.prompt_summary!r}. Be concise."
        )

