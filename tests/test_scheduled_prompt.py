"""Tests for the scheduled-run message preparation.

These target the bug where local LLMs (e.g. Qwen 2.5) refuse to perform a
scheduled task because the user's original prompt contains scheduling verbs
("every 1 minute for 3 times") — combined with our "do not implement
scheduling" system message, the model treats the entire request as forbidden.

The fix strips scheduling directives before forwarding to the LLM.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

import app


def test_sanitize_removes_every_n_minutes():
    out = app._sanitize_prompt_for_scheduled_run(
        "search bike info every 1 minute for 3 times",
        "bike info",
    )
    assert "every" not in out.lower()
    assert "minute" not in out.lower()
    assert "times" not in out.lower()
    assert "bike" in out.lower()


def test_sanitize_removes_hourly_shorthand():
    out = app._sanitize_prompt_for_scheduled_run(
        "monitor tesla stock price hourly for 5 times",
        "tesla stock price",
    )
    assert "hourly" not in out.lower()
    assert "for 5 times" not in out.lower()
    assert "monitor" not in out.lower()
    assert "tesla" in out.lower()


def test_sanitize_removes_absolute_time_verbs():
    out = app._sanitize_prompt_for_scheduled_run(
        "search python news tomorrow at 9am",
        "python news",
    )
    assert "tomorrow" not in out.lower()
    assert "9am" not in out.lower()
    assert "python" in out.lower()


def test_sanitize_falls_back_to_search_query_when_stripped_bare():
    """If stripping the scheduling verbs leaves nothing useful, we must
    fall back to the planner-selected `search_query` rather than an empty
    prompt (which would confuse the LLM even more)."""
    out = app._sanitize_prompt_for_scheduled_run(
        "every 1 minute for 3 times",  # ONLY scheduling verbs
        "current AI news",
    )
    assert out == "current AI news"


def test_build_messages_uses_sanitized_prompt():
    """The scheduled-run message list must contain the sanitized task, not
    the raw user prompt with scheduling verbs still in it."""
    messages = app._build_messages(
        "search bike info every 1 minute for 3 times",
        "bike info",
    )
    # SCHEDULED_RUN_INSTRUCTIONS + SearchExecutionInstruction + Human
    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], SystemMessage)
    assert isinstance(messages[2], HumanMessage)

    human_text = messages[2].content
    assert "every" not in human_text.lower()
    assert "minute" not in human_text.lower()
    assert "times" not in human_text.lower()


def test_scheduled_run_instructions_do_not_forbid_the_task():
    """Regression guard: the scheduled-run system message must NOT contain
    negative words that could be misread by a cautious LLM ('cannot',
    'forbidden', 'not allowed'). It must explicitly tell the model to
    proceed with the task."""
    text = app.SCHEDULED_RUN_INSTRUCTIONS.lower()
    assert "do not refuse" in text or "do the task" in text or "just do" in text
    assert "cannot" not in text
    assert "forbidden" not in text
    assert "not allowed" not in text
