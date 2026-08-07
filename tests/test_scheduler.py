"""Tests for the improved scheduler.

These tests exercise the registry, persistence, retries, and pause/resume
without depending on any LLM, LM Studio, or network calls.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import scheduler as scheduler_module
from scheduler import (
    JobRegistry,
    ScheduleValidationError,
    Scheduler,
    ScheduledSearchJob,
    validate_schedule_inputs,
)


def _fake_result(text: str = "ok"):
    class _Msg:
        content = text
    return {"messages": [_Msg()]}


def _dummy_builder(prompt: str, query: str):
    return [{"role": "user", "content": prompt}]


def _fresh_registry(tmp_path: Path) -> JobRegistry:
    return JobRegistry(tmp_path)


# ---- Validation ---------------------------------------------------------------


def test_validate_schedule_inputs_rejects_bad_values():
    with pytest.raises(ScheduleValidationError):
        validate_schedule_inputs(0, 1)
    with pytest.raises(ScheduleValidationError):
        validate_schedule_inputs(1, 0)
    with pytest.raises(ScheduleValidationError):
        validate_schedule_inputs("nope", 1)
    with pytest.raises(ScheduleValidationError):
        validate_schedule_inputs(True, 1)  # bools rejected


def test_validate_schedule_inputs_accepts_positive_values():
    assert validate_schedule_inputs(2.5, 3) == (2.5, 3)


# ---- Basic scheduling --------------------------------------------------------


def test_scheduler_runs_requested_count(tmp_path):
    registry = _fresh_registry(tmp_path)
    calls: list[dict] = []

    def invoker(state):
        calls.append(state)
        return _fake_result("done")

    sched = Scheduler(registry, invoker, _dummy_builder)
    job = sched.start("check", interval_minutes=0.001, run_count=2, search_query="check")
    job.join(timeout=3)

    assert job.done
    assert job.status == "completed"
    assert job.completed_runs == 2
    assert len(calls) == 2
    assert len(job.results) == 2


def test_scheduler_stops_on_cancel(tmp_path):
    registry = _fresh_registry(tmp_path)
    first_run = threading.Event()

    def invoker(state):
        first_run.set()
        return _fake_result()

    sched = Scheduler(registry, invoker, _dummy_builder)
    job = sched.start("monitor", interval_minutes=1, run_count=3, search_query="monitor")
    assert first_run.wait(timeout=2)
    job.stop()
    job.join(timeout=2)

    assert job.done
    assert job.status == "cancelled"
    assert job.completed_runs == 1


def test_scheduler_does_not_wait_after_final_run(tmp_path):
    registry = _fresh_registry(tmp_path)

    def invoker(state):
        return _fake_result()

    sched = Scheduler(registry, invoker, _dummy_builder)
    started = time.monotonic()
    job = sched.start("once", interval_minutes=1, run_count=1, search_query="once")
    job.join(timeout=2)

    assert time.monotonic() - started < 1.0
    assert job.status == "completed"


# ---- Retries -----------------------------------------------------------------


def test_scheduler_retries_transient_failures(tmp_path):
    registry = _fresh_registry(tmp_path)
    attempts = {"n": 0}

    def flaky(state):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return _fake_result("recovered")

    sched = Scheduler(
        registry, flaky, _dummy_builder,
        max_invoke_retries=3, invoke_retry_backoff=0.01,
    )
    job = sched.start("retry me", interval_minutes=0.001, run_count=1, search_query="retry")
    job.join(timeout=3)

    assert job.status == "completed"
    assert attempts["n"] == 3


def test_scheduler_marks_failed_after_max_retries(tmp_path):
    registry = _fresh_registry(tmp_path)

    def always_fail(state):
        raise RuntimeError("no dice")

    sched = Scheduler(
        registry, always_fail, _dummy_builder,
        max_invoke_retries=2, invoke_retry_backoff=0.01,
    )
    job = sched.start("break", interval_minutes=0.001, run_count=1, search_query="break")
    job.join(timeout=3)

    assert job.status == "failed"
    assert "no dice" in (job.error_message or "")


# ---- Registry / persistence --------------------------------------------------


def test_registry_persists_jobs_to_disk(tmp_path):
    registry = _fresh_registry(tmp_path)

    def invoker(state):
        return _fake_result()

    sched = Scheduler(registry, invoker, _dummy_builder)
    job = sched.start("persist me", interval_minutes=0.001, run_count=1, search_query="persist me")
    job.join(timeout=3)

    jobs_file = tmp_path / "jobs.json"
    assert jobs_file.is_file()
    payload = json.loads(jobs_file.read_text())
    assert any(item["id"] == job.id for item in payload)


def test_registry_writes_per_run_history(tmp_path):
    registry = _fresh_registry(tmp_path)

    def invoker(state):
        return _fake_result("hello world")

    sched = Scheduler(registry, invoker, _dummy_builder)
    job = sched.start("history", interval_minutes=0.001, run_count=2, search_query="history")
    job.join(timeout=3)

    history_dir = tmp_path / "history" / job.id
    files = sorted(history_dir.glob("run-*.json"))
    assert len(files) == 2
    payload = json.loads(files[0].read_text())
    assert payload["content"] == "hello world"


def test_resume_persisted_relaunches_incomplete_jobs(tmp_path):
    registry = _fresh_registry(tmp_path)
    # Simulate a saved-but-incomplete job on disk.
    saved = ScheduledSearchJob(
        prompt="resume me",
        search_query="resume me",
        interval_minutes=0.001,
        run_count=1,
        status="running",
    )
    (tmp_path / "jobs.json").write_text(json.dumps([saved.to_dict()]))

    calls = []

    def invoker(state):
        calls.append(state)
        return _fake_result()

    sched = Scheduler(registry, invoker, _dummy_builder)
    resumed = sched.resume_persisted()
    assert len(resumed) == 1
    resumed[0].join(timeout=3)
    assert resumed[0].status == "completed"
    assert len(calls) == 1


def test_resume_persisted_skips_finished_jobs(tmp_path):
    registry = _fresh_registry(tmp_path)
    finished = ScheduledSearchJob(
        prompt="already done",
        search_query="already done",
        interval_minutes=0.001,
        run_count=1,
        status="completed",
        completed_runs=1,
    )
    (tmp_path / "jobs.json").write_text(json.dumps([finished.to_dict()]))

    def invoker(state):
        raise AssertionError("Should not run for a completed job")

    sched = Scheduler(registry, invoker, _dummy_builder)
    resumed = sched.resume_persisted()
    assert resumed == []
    assert registry.get(finished.id) is not None


# ---- Pause / resume ----------------------------------------------------------


def test_pause_and_resume(tmp_path):
    registry = _fresh_registry(tmp_path)
    invocations = []
    first_run_done = threading.Event()

    def invoker(state):
        invocations.append(1)
        first_run_done.set()
        return _fake_result()

    sched = Scheduler(registry, invoker, _dummy_builder)
    # interval = 0.005 min = 0.3 s so 3 runs finish in well under our join timeout.
    job = sched.start("pauseable", interval_minutes=0.005, run_count=3, search_query="pauseable")

    # Wait for run 1, then pause before the second run's interval elapses.
    assert first_run_done.wait(timeout=3)
    job.pause()

    # Wait long enough for at least one pause check to observe the pause state.
    time.sleep(0.6)
    before_resume = len(invocations)
    assert job.status == "paused"

    job.resume()
    job.join(timeout=5)
    assert job.status == "completed"
    assert len(invocations) == 3
    assert len(invocations) > before_resume
