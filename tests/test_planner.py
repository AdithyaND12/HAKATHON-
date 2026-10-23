"""Tests for the extracted planner + absolute-time parsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import planner


def test_fallback_recognizes_interval_and_count():
    plan = planner._fallback_search_plan("check tesla news every 10 minutes for 5 times")
    assert plan.should_schedule
    assert plan.wait_minutes == 10
    assert plan.run_count == 5
    assert "tesla news" in plan.search_query.lower()


def test_fallback_hourly_shorthand():
    plan = planner._fallback_search_plan("check hackernews hourly for 3 times")
    assert plan.should_schedule
    assert plan.wait_minutes == 60
    assert plan.run_count == 3


def test_fallback_daily_shorthand():
    plan = planner._fallback_search_plan("check python news daily")
    assert plan.should_schedule
    assert plan.wait_minutes == 60 * 24


def test_fallback_no_schedule_for_plain_question():
    plan = planner._fallback_search_plan("what is the capital of France")
    assert not plan.should_schedule
    assert plan.run_count == 1
    assert plan.wait_minutes == 0


def test_fallback_absolute_time_tomorrow():
    plan = planner._fallback_search_plan("search python news tomorrow at 9am")
    assert plan.should_schedule
    assert plan.absolute_start_iso is not None
    when = datetime.fromisoformat(plan.absolute_start_iso)
    now_utc = datetime.now(timezone.utc)
    assert when > now_utc
    assert (when - now_utc) < timedelta(days=2)


def test_fallback_in_minutes():
    plan = planner._fallback_search_plan("search AI news in 5 minutes")
    assert plan.should_schedule
    assert plan.absolute_start_iso is not None


def test_fallback_twice_thrice():
    plan = planner._fallback_search_plan("check python news every 2 minutes twice")
    assert plan.should_schedule
    assert plan.run_count == 2
    assert plan.wait_minutes == 2


def test_clamp_bounds_run_count():
    raw = planner.SearchPlan(
        search_query="x", should_schedule=True, wait_minutes=5, run_count=9999
    )
    clamped = planner._clamp_plan(raw)
    assert clamped.run_count <= 1000  # sanity


def test_clamp_zero_out_wait_when_no_schedule():
    raw = planner.SearchPlan(
        search_query="x", should_schedule=False, wait_minutes=10, run_count=5
    )
    clamped = planner._clamp_plan(raw)
    assert clamped.wait_minutes == 0
    assert clamped.run_count == 1


def test_create_search_plan_parses_interval_and_count():
    plan = planner.create_search_plan(
        "can you search about car in internet for 2 times every 1 min"
    )
    assert plan.wait_minutes == 1.0
    assert plan.run_count == 2
    assert plan.should_schedule is True


def test_create_search_plan_hourly_twice():
    plan = planner.create_search_plan("check news every hour twice")
    assert plan.wait_minutes == 60.0
    assert plan.run_count == 2
    assert plan.should_schedule is True


# ---- Regression tests for interval/time parsing bugs -------------------------


def test_long_intervals_are_not_capped_to_the_wait_tool_limit():
    """A 6-hour recurrence must stay 6 hours; the wait tool's 1h bound is
    unrelated to scheduling."""
    plan = planner._clamp_plan(
        planner.SearchPlan(
            search_query="news",
            should_schedule=True,
            wait_minutes=6 * 60,
            run_count=2,
        )
    )
    assert plan.wait_minutes == 6 * 60


def test_daily_fallback_keeps_24_hour_interval():
    plan = planner.create_search_plan("check python news daily")
    assert plan.should_schedule
    assert plan.wait_minutes == 24 * 60


def test_hourly_monitor_defaults_to_two_runs():
    plan = planner.create_search_plan("monitor python news hourly")
    assert plan.should_schedule
    assert plan.wait_minutes == 60
    assert plan.run_count == 2


def test_at_prefix_without_ampm_is_parsed():
    when = planner._parse_absolute_start("remind me at 15:30")
    assert when is not None
    local = when.astimezone()
    assert (local.hour, local.minute) == (15, 30)


def test_bare_number_after_tomorrow_is_not_a_clock_time():
    """'tomorrow to buy 2 apples' must use the 9am default, not 02:00."""
    when = planner._parse_absolute_start("remind me tomorrow to buy 2 apples")
    assert when is not None
    assert when.astimezone().hour == 9


def test_duration_without_recurrence_is_not_scheduling():
    plan = planner.create_search_plan("the movie is 90 minutes long")
    assert not plan.should_schedule
    assert plan.run_count == 1
    assert plan.wait_minutes == 0


def test_absolute_start_is_normalized_to_utc():
    assert planner._normalize_absolute_start("2030-01-01T09:00:00+02:00") == (
        "2030-01-01T07:00:00+00:00"
    )
    naive = planner._normalize_absolute_start("2030-01-01T09:00:00")
    assert naive is not None and naive.endswith("+00:00")
    assert planner._normalize_absolute_start("tomorrow at 9") is None


def test_chat_llm_uses_dedicated_timeout_and_retries():
    import config

    llm = planner.build_chat_llm()
    assert llm.timeout == config.LLM_TIMEOUT_SECONDS
    assert llm.max_retries == config.LLM_CLIENT_MAX_RETRIES
