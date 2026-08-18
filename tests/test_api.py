from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api
from planner import SearchPlan
from scheduler import ScheduleValidationError, ScheduledSearchJob


client = TestClient(api.app)


def _plan(**overrides):
    values = {
        "task_type": "search",
        "search_query": "python news",
        "reminder_text": None,
        "should_schedule": False,
        "wait_minutes": 0.0,
        "run_count": 1,
        "absolute_start_iso": None,
    }
    values.update(overrides)
    return SearchPlan(**values)


def test_chat_invokes_chatbot_with_plan_and_active_document(monkeypatch):
    seen = {}

    class FakeChatbot:
        def invoke(self, state):
            seen["state"] = state
            return {"messages": [SimpleNamespace(content="API answer")]}

    monkeypatch.setattr(api, "create_search_plan", lambda prompt: _plan(search_query="latest python"))
    monkeypatch.setattr(api.ragtool, "active_source_label", lambda: "Active document: doc.pdf")
    monkeypatch.setattr(api, "chatbot", FakeChatbot())

    response = client.post("/chat", json={"prompt": "what is new in python?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "API answer"
    assert payload["plan"]["search_query"] == "latest python"
    messages = seen["state"]["messages"]
    assert messages[0].content == "Active document: doc.pdf"
    assert "latest python" in messages[1].content
    assert messages[-1].content == "what is new in python?"


def test_chat_rejects_scheduling_prompts(monkeypatch):
    monkeypatch.setattr(
        api,
        "create_search_plan",
        lambda prompt: _plan(should_schedule=True, wait_minutes=60.0, run_count=2),
    )

    response = client.post("/chat", json={"prompt": "check python news hourly"})

    assert response.status_code == 400
    assert "Use /jobs/start" in response.json()["detail"]


def test_chat_returns_500_when_chatbot_fails(monkeypatch):
    class FailingChatbot:
        def invoke(self, state):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(api, "create_search_plan", lambda prompt: _plan())
    monkeypatch.setattr(api.ragtool, "active_source_label", lambda: "")
    monkeypatch.setattr(api, "chatbot", FailingChatbot())

    response = client.post("/chat", json={"prompt": "hello"})

    assert response.status_code == 500
    assert "Chat invocation failed" in response.json()["detail"]


def test_start_job_uses_scheduler_with_parsed_plan(monkeypatch):
    calls = []
    job = ScheduledSearchJob(prompt="check python hourly", id="job123")

    class FakeScheduler:
        def start(self, **kwargs):
            calls.append(kwargs)
            return job

    plan = _plan(
        should_schedule=True,
        wait_minutes=60.0,
        run_count=3,
        absolute_start_iso="2026-08-15T10:00:00+00:00",
    )
    monkeypatch.setattr(api, "create_search_plan", lambda prompt: plan)
    monkeypatch.setattr(api, "scheduler", FakeScheduler())

    response = client.post("/jobs/start", json={"prompt": "check python hourly"})

    assert response.status_code == 200
    assert response.json()["job_id"] == "job123"
    assert calls == [
        {
            "prompt": "check python hourly",
            "interval_minutes": 60.0,
            "run_count": 3,
            "search_query": "python news",
            "absolute_start_iso": "2026-08-15T10:00:00+00:00",
            "task_type": "search",
            "reminder_text": None,
        }
    ]


def test_start_job_rejects_non_schedule_prompt(monkeypatch):
    monkeypatch.setattr(api, "create_search_plan", lambda prompt: _plan(should_schedule=False))

    response = client.post("/jobs/start", json={"prompt": "what is python?"})

    assert response.status_code == 400
    assert "does not contain a scheduling command" in response.json()["detail"]


def test_start_job_reports_schedule_validation_errors(monkeypatch):
    class FailingScheduler:
        def start(self, **kwargs):
            raise ScheduleValidationError("Interval minutes must be a positive number.")

    monkeypatch.setattr(
        api,
        "create_search_plan",
        lambda prompt: _plan(should_schedule=True, wait_minutes=-1.0),
    )
    monkeypatch.setattr(api, "scheduler", FailingScheduler())

    response = client.post("/jobs/start", json={"prompt": "check repeatedly"})

    assert response.status_code == 422
    assert "Schedule validation failed" in response.json()["detail"]


def test_jobs_endpoints_read_and_control_registry(monkeypatch, tmp_path):
    job = ScheduledSearchJob(
        prompt="monitor python",
        id="abc123",
        search_query="python news",
        interval_minutes=1.0,
        run_count=2,
    )
    history_dir = tmp_path / "history" / job.id
    history_dir.mkdir(parents=True)
    (history_dir / "run-001.json").write_text(
        json.dumps({"content": "first result"}), encoding="utf-8"
    )

    class FakeRegistry:
        def all(self):
            return [job]

        def get(self, job_id):
            return job if job_id == job.id else None

        def update(self, updated):
            self.updated = updated

        def history_path(self, job_id):
            return history_dir

    registry = FakeRegistry()
    monkeypatch.setattr(api, "_registry", registry)

    list_response = client.get("/jobs")
    assert list_response.status_code == 200
    assert list_response.json()[0]["id"] == job.id

    pause_response = client.post(f"/jobs/{job.id}/pause")
    assert pause_response.status_code == 200
    assert job.status == "paused"

    resume_response = client.post(f"/jobs/{job.id}/resume")
    assert resume_response.status_code == 200
    assert job.status == "running"

    cancel_response = client.post(f"/jobs/{job.id}/cancel")
    assert cancel_response.status_code == 200
    assert job.stop_event.is_set()

    logs_response = client.get(f"/jobs/{job.id}/logs")
    assert logs_response.status_code == 200
    assert logs_response.json() == [{"content": "first result"}]


def test_job_control_returns_404_for_unknown_job(monkeypatch):
    class EmptyRegistry:
        def get(self, job_id):
            return None

    monkeypatch.setattr(api, "_registry", EmptyRegistry())

    response = client.post("/jobs/missing/cancel")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_document_upload_indexes_and_activates_pdf(monkeypatch, tmp_path):
    calls = {}

    def fake_index_pdf(path, name=None):
        calls["index"] = (path, name)
        return 7

    monkeypatch.setattr(api.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(api.ragtool, "index_pdf", fake_index_pdf)
    monkeypatch.setattr(api.ragtool, "collection_name_for_sha", lambda sha: f"rag_{sha[:8]}")
    monkeypatch.setattr(api.ragtool, "set_active_collection", lambda name: calls.setdefault("active", name))

    response = client.post(
        "/documents/upload",
        files={"file": ("my doc.pdf", b"%PDF-1.7\nfake\n", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks"] == 7
    assert payload["collection_name"].startswith("rag_")
    indexed_path, indexed_name = calls["index"]
    assert indexed_path.parent == tmp_path / "uploads"
    assert indexed_path.name.endswith("_my_doc.pdf")
    assert indexed_name == "my doc.pdf"
    assert calls["active"] == payload["collection_name"]


def test_document_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"


def test_document_endpoints_list_and_set_active(monkeypatch):
    active = {}
    monkeypatch.setattr(api.ragtool, "list_indexed_documents", lambda: [{"name": "doc.pdf"}])
    monkeypatch.setattr(api.ragtool, "set_active_collection", lambda name: active.setdefault("name", name))

    list_response = client.get("/documents")
    active_response = client.post("/documents/active", json={"collection_name": "rag_123"})

    assert list_response.status_code == 200
    assert list_response.json() == [{"name": "doc.pdf"}]
    assert active_response.status_code == 200
    assert active["name"] == "rag_123"
