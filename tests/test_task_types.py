"""Tests for the task-type dispatch introduced in the TaskPlan refactor.

The planner now assigns a `task_type` to every prompt (search / reminder /
calculation / rag / chat). The scheduled-run message builder dispatches on
that type so the LLM is not nudged toward web_search when it should be doing
something else. These tests exercise every branch.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

import app
import planner


# ---- Planner intent detection ------------------------------------------------


def test_fallback_detects_reminder_intent():
    plan = planner._fallback_search_plan("remind me to drink water in 5 minutes")
    assert plan.task_type == "reminder"
    assert plan.reminder_text is not None
    assert "drink water" in plan.reminder_text.lower()
    assert plan.should_schedule
    assert plan.absolute_start_iso is not None


def test_fallback_detects_reminder_without_target():
    plan = planner._fallback_search_plan("remind me after 2 minutes using the tools")
    assert plan.task_type == "reminder"
    # reminder_text should be non-empty even when target is vague
    assert plan.reminder_text
    assert plan.absolute_start_iso is not None


def test_fallback_detects_calculation_intent():
    plan = planner._fallback_search_plan("calculate 27 times 14")
    assert plan.task_type == "calculation"


def test_fallback_detects_rag_intent():
    plan = planner._fallback_search_plan(
        "what does the constitution say about freedom of speech"
    )
    assert plan.task_type == "rag"


def test_fallback_detects_chat_intent():
    plan = planner._fallback_search_plan("tell me a joke every 5 minutes for 3 times")
    assert plan.task_type == "chat"
    assert plan.should_schedule
    assert plan.wait_minutes == 5


def test_fallback_defaults_to_search():
    plan = planner._fallback_search_plan("check the latest python news every hour")
    assert plan.task_type == "search"
    assert plan.reminder_text is None


def test_clamp_preserves_task_type():
    raw = planner.SearchPlan(
        task_type="reminder",
        search_query="drink water",
        reminder_text="drink water",
        should_schedule=True,
        wait_minutes=10,
        run_count=3,
    )
    clamped = planner._clamp_plan(raw)
    assert clamped.task_type == "reminder"
    assert clamped.reminder_text == "drink water"


def test_clamp_collapses_unknown_task_type_to_search():
    raw = planner.SearchPlan.model_construct(
        task_type="invalid-type",
        search_query="x",
        reminder_text=None,
        should_schedule=False,
        wait_minutes=0,
        run_count=1,
    )
    clamped = planner._clamp_plan(raw)
    assert clamped.task_type == "search"


def test_clamp_clears_reminder_text_for_non_reminder():
    raw = planner.SearchPlan(
        task_type="search",
        search_query="python news",
        reminder_text="leftover from an earlier state",
        should_schedule=False,
        wait_minutes=0,
        run_count=1,
    )
    clamped = planner._clamp_plan(raw)
    assert clamped.reminder_text is None


def test_llm_search_gets_regex_reminder_upgrade(monkeypatch):
    """If the LLM misses the reminder intent and returns task_type='search',
    the safety-net regex detection must upgrade the plan to 'reminder'."""
    def fake_invoke(messages):
        return planner.SearchPlan(
            task_type="search",   # LLM missed it
            search_query="drink water",
            reminder_text=None,
            should_schedule=True,
            wait_minutes=5,
            run_count=1,
        )

    monkeypatch.setattr(planner, "_invoke_search_planner", fake_invoke)
    plan = planner.create_search_plan("remind me to drink water in 5 minutes")
    assert plan.task_type == "reminder"
    assert plan.reminder_text is not None


# ---- Message builder dispatch ------------------------------------------------


def test_build_messages_reminder_does_not_mention_web_search():
    messages = app._build_messages(
        prompt="remind me to drink water in 5 minutes",
        search_query="drink water",
        task_type="reminder",
        reminder_text="drink water",
    )
    assert len(messages) == 3
    # System messages must NOT nudge the LLM toward web_search
    system_text = " ".join(m.content for m in messages if isinstance(m, SystemMessage))
    assert "web search" not in system_text.lower() or "do not" in system_text.lower()
    # Reminder text must be present
    assert "drink water" in system_text.lower()
    # LLM must be told this IS the reminder
    assert "reminder" in system_text.lower()


def test_build_messages_calculation_routes_to_calculator():
    messages = app._build_messages(
        prompt="calculate 27 times 14",
        search_query="27 times 14",
        task_type="calculation",
    )
    system_text = " ".join(m.content for m in messages if isinstance(m, SystemMessage))
    assert "calculator" in system_text.lower()
    assert "do not use web search" in system_text.lower()


def test_build_messages_rag_routes_to_constitution():
    messages = app._build_messages(
        prompt="what does the constitution say about freedom of speech",
        search_query="constitution freedom of speech",
        task_type="rag",
    )
    system_text = " ".join(m.content for m in messages if isinstance(m, SystemMessage))
    assert "get_rag_chunks" in system_text.lower() or "constitution" in system_text.lower()
    assert "do not use web search" in system_text.lower()


def test_build_messages_chat_gives_no_tool_nudge():
    messages = app._build_messages(
        prompt="tell me a joke",
        search_query="joke",
        task_type="chat",
    )
    system_text = " ".join(m.content for m in messages if isinstance(m, SystemMessage))
    # Chat mode must NOT push any specific tool
    assert "web search" not in system_text.lower() or "no tool" in system_text.lower()
    assert "no tools needed" in system_text.lower() or "no tool" in system_text.lower()


def test_build_messages_default_search_unchanged():
    """Regression guard: passing no task_type must produce the classic search flow."""
    messages = app._build_messages(
        prompt="search python news every hour",
        search_query="python news",
    )
    system_text = " ".join(m.content for m in messages if isinstance(m, SystemMessage))
    assert "web-search query" in system_text.lower()
    assert "python news" in system_text.lower()


# ---- Scheduler carries task_type through -------------------------------------


def test_scheduler_job_persists_task_type(tmp_path):
    from scheduler import JobRegistry, ScheduledSearchJob

    registry = JobRegistry(tmp_path)
    job = ScheduledSearchJob(
        prompt="remind me",
        search_query="drink water",
        interval_minutes=5,
        run_count=1,
        task_type="reminder",
        reminder_text="drink water",
    )
    registry.register(job)

    # Round-trip via disk
    import json
    payload = json.loads((tmp_path / "jobs.json").read_text())
    assert payload[0]["task_type"] == "reminder"
    assert payload[0]["reminder_text"] == "drink water"

    reloaded = ScheduledSearchJob.from_dict(payload[0])
    assert reloaded.task_type == "reminder"
    assert reloaded.reminder_text == "drink water"


def test_scheduler_worker_forwards_task_type_to_builder(tmp_path):
    """The worker must pass task_type/reminder_text into the message_builder."""
    from scheduler import JobRegistry, Scheduler

    seen = {}

    def builder(prompt, query, task_type="search", reminder_text=None):
        seen["prompt"] = prompt
        seen["query"] = query
        seen["task_type"] = task_type
        seen["reminder_text"] = reminder_text
        return [{"role": "user", "content": prompt}]

    def invoker(state):
        class _Msg:
            content = "ok"
        return {"messages": [_Msg()]}

    registry = JobRegistry(tmp_path)
    sched = Scheduler(registry, invoker, builder)
    job = sched.start(
        prompt="remind me to breathe",
        interval_minutes=0.001,
        run_count=1,
        search_query="breathe",
        task_type="reminder",
        reminder_text="breathe deeply",
    )
    job.join(timeout=3)

    assert seen["task_type"] == "reminder"
    assert seen["reminder_text"] == "breathe deeply"
