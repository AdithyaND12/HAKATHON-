"""End-to-end interaction tests using Streamlit's AppTest harness.

These exercise the real UI script top-to-bottom with a fake LangGraph bot and
planner, so no API keys or network are needed. They verify the user-facing
flow: streaming replies, token-usage captions, and LLM chat titles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def fake_ai(monkeypatch, tmp_path):
    """Patch app's LLM + graph, planner, and all persistent stores with fakes
    so no API keys, network, or real user state are touched.

    Config may already be imported by earlier test files (with the repo's real
    data dir), so patching the env var alone is not enough — the module-level
    paths are re-pointed explicitly here.

    Returns (app_module, search_plan) so tests can tweak plan fields.
    """
    monkeypatch.setenv("HAKATHON_DATA_DIR", str(tmp_path))

    import config
    import ragtool

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ragtool, "STORE_ROOT", tmp_path / "stores")
    monkeypatch.setattr(
        ragtool,
        "PERSIST_DIRECTORY",
        ragtool.STORE_ROOT / ragtool._model_slug(config.JINA_EMBEDDING_MODEL),
    )

    import app
    import planner
    from planner import SearchPlan, create_search_plan
    from scheduler import JobRegistry, Scheduler

    monkeypatch.setattr(
        app, "_registry", JobRegistry(tmp_path)
    )  # fresh, empty registry in tmp
    # Re-bind the scheduler to the fresh registry so `resume_persisted()` and
    # `start()` in the script can't touch (or relaunch) real user jobs.
    monkeypatch.setattr(
        app,
        "scheduler",
        Scheduler(
            registry=app._registry,
            invoker=app._invoke_chatbot,
            message_builder=app._build_messages,
            planner=create_search_plan,
        ),
    )

    from langchain_core.messages import AIMessage

    class FakeChunk:
        def __init__(self, content="", usage=None):
            self.content = content
            self.usage_metadata = usage

    class FakeBot:
        """Duck-typed compiled graph: streams token chunks, or fails."""

        def __init__(self):
            self.fail_stream = False

        def stream(self, inputs, stream_mode="messages"):
            if self.fail_stream:
                raise RuntimeError("stream backend broken")
            yield FakeChunk("Hel"), {}
            yield FakeChunk("lo world", {"input_tokens": 3, "output_tokens": 2}), {}

        def invoke(self, inputs):
            return {"messages": [AIMessage(content="fallback reply")]}

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(content="Tesla market update")

    class FakeSearchPlanner:
        def __init__(self, plan):
            self.plan = plan

        def invoke(self, messages):
            return self.plan

    plan = SearchPlan(
        search_query="tesla news",
        should_schedule=False,
        wait_minutes=0.0,
        run_count=1,
    )
    monkeypatch.setattr(app, "chatbot", FakeBot())
    monkeypatch.setattr(app, "llm", FakeLLM())
    monkeypatch.setattr(planner, "search_planner", FakeSearchPlanner(plan))
    return app, plan


def _launch(fixtures):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=90)
    at.run()
    assert not at.exception, f"script raised: {at.exception}"
    return at


def test_send_prompt_streams_reply_with_usage_caption(fake_ai):
    at = _launch(fake_ai)
    at.chat_input[0].set_value("what about tesla?")
    at.run()
    assert not at.exception, f"script raised: {at.exception}"

    # The streamed reply was rendered (placeholder markdown).
    texts = " ".join(m.value for m in at.markdown)
    assert "Hel" in texts.replace("Hel", "Hel")  # sanity: markdown present
    # Usage caption under the reply.
    captions = " ".join(c.value for c in at.caption)
    assert "in 3 · out 2 tokens" in captions.replace("&nbsp;", " ")

    # Persisted message carries the usage metadata.
    convs = at.session_state["conversations"]
    active = next(iter(convs.values()))
    assert active["messages"][-1]["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_stream_failure_falls_back_and_shows_friendly_error(fake_ai):
    fake_ai[0].chatbot.fail_stream = True
    at = _launch(fake_ai)
    at.chat_input[0].set_value("hello")
    at.run()
    texts = " ".join(m.value for m in at.markdown)
    # Fallback replied with the non-streamed text.
    assert "fallback reply" in texts
    assert not at.exception


def test_first_chat_gets_llm_title(fake_ai):
    at = _launch(fake_ai)
    at.chat_input[0].set_value("tesla stock news")
    at.run()
    convs = at.session_state["conversations"]
    active = next(iter(convs.values()))
    assert active["title"] == "Tesla market update"


def test_active_job_declares_poll_fragment_without_errors(fake_ai):
    """With a live job in the registry, the page must render and the poller
    fragment must be declared without raising."""
    app, _ = fake_ai
    from scheduler import ScheduledSearchJob

    job = ScheduledSearchJob(prompt="track tesla", interval_minutes=1, run_count=5)
    app._registry.register(job)
    assert app._registry.active()  # sanity: the job is active

    at = _launch(fake_ai)
    # The session state set by the active-job path is visible.
    assert at.session_state["_last_active_at"] is not None
    # A dialog/alert-free run with the fragment declared.
    assert not at.exception