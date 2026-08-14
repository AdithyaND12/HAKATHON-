"""Tests for the Streamlit UI polling logic.

We don't spin up a real Streamlit runtime — instead we exercise the pure
functions responsible for reading run-history files and building payloads
that the UI renders. This keeps the test fast and hermetic.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


def _stub_streamlit_and_load(tmp_path, monkeypatch):
    """Stub the streamlit + streamlit_autorefresh modules so `streamlit_app`
    can be imported in a plain test process, and point the app's data
    directory at a tmp folder so we don't touch real user state.
    """
    # Route all app state to tmp_path so tests are hermetic.
    monkeypatch.setenv("HAKATHON_DATA_DIR", str(tmp_path))

    # Minimal `streamlit` stub — only the surface `streamlit_app` touches at
    # import time. session_state supports both dict-style and attribute access.
    class _AttrDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc
        def __setattr__(self, key, value):
            self[key] = value

    st = types.ModuleType("streamlit")
    st.session_state = _AttrDict()
    st.set_page_config = lambda **kwargs: None
    st.markdown = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.divider = lambda *a, **k: None
    st.button = lambda *a, **k: False
    st.toast = lambda *a, **k: None
    st.chat_message = lambda *a, **k: _NullContext()
    st.chat_input = lambda *a, **k: None
    st.warning = st.error = st.rerun = lambda *a, **k: None
    st.spinner = lambda *a, **k: _NullContext()
    st.file_uploader = lambda *a, **k: None
    st.selectbox = lambda *a, **k: None
    st.download_button = lambda *a, **k: None
    st.container = lambda *a, **k: _NullContext()
    st.fragment = lambda *a, **k: (lambda f: f)  # re-declared lazily per call

    def _radio(*a, **k):
        """Mimic Streamlit: a keyed radio returns its stored value, otherwise
        the option at the requested index (so it never returns None)."""
        key = k.get("key")
        if key and key in st.session_state:
            return st.session_state[key]
        options = k.get("options") or [None]
        return options[k.get("index", 0)]

    st.radio = _radio
    st.toggle = lambda *a, **k: True
    st.progress = lambda *a, **k: _NullContext()
    st.status = lambda *a, **k: _NullContext()
    st.success = lambda *a, **k: None
    st.sidebar = _NullContext()
    sys.modules["streamlit"] = st

    autorefresh = types.ModuleType("streamlit_autorefresh")
    autorefresh.st_autorefresh = lambda **kwargs: 0
    sys.modules["streamlit_autorefresh"] = autorefresh

    # Reload streamlit_app so it re-reads HAKATHON_DATA_DIR + our stubs.
    for mod in ("streamlit_app", "app", "config"):
        if mod in sys.modules:
            del sys.modules[mod]
    import streamlit_app  # noqa: F401
    # Point the registry at tmp_path so tests are hermetic even if the
    # config default has already been resolved elsewhere in the process.
    streamlit_app._registry.history_dir = tmp_path / "history"
    streamlit_app._registry.history_dir.mkdir(parents=True, exist_ok=True)
    return streamlit_app, st


class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def __call__(self, *a, **k):
        return _NullContext()


def _write_run_file(root: Path, job_id: str, run_number: int, content: str, timestamp: str):
    job_dir = root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "run_number": run_number,
        "completed_at": timestamp,
        "content": content,
    }
    path = job_dir / f"run-{run_number:03d}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fetch_new_runs_returns_empty_when_no_history(tmp_path, monkeypatch):
    app_mod, st = _stub_streamlit_and_load(tmp_path, monkeypatch)
    st.session_state["seen_runs"] = set()
    assert app_mod._fetch_new_scheduled_runs() == []


def test_fetch_new_runs_surfaces_newly_written_files(tmp_path, monkeypatch):
    app_mod, st = _stub_streamlit_and_load(tmp_path, monkeypatch)
    st.session_state["seen_runs"] = set()

    _write_run_file(tmp_path / "history", "job-a", 1, "hello from run 1", "2026-01-01T00:00:00+00:00")
    _write_run_file(tmp_path / "history", "job-a", 2, "hello from run 2", "2026-01-01T00:05:00+00:00")

    updates = app_mod._fetch_new_scheduled_runs()
    assert len(updates) == 2
    assert updates[0]["content"] == "hello from run 1"
    assert updates[1]["content"] == "hello from run 2"
    # Second call must return nothing — files have been marked seen.
    assert app_mod._fetch_new_scheduled_runs() == []


def test_fetch_new_runs_skips_preexisting_files_when_seeded(tmp_path, monkeypatch):
    """Verifies the "only surface runs completed after session start" behavior:
    files that existed BEFORE seen_runs was seeded must never appear."""
    # Write a file BEFORE the app loads.
    _write_run_file(tmp_path / "history", "job-old", 1, "old run", "2026-01-01T00:00:00+00:00")

    app_mod, st = _stub_streamlit_and_load(tmp_path, monkeypatch)
    # Seed seen_runs with everything that exists on disk.
    st.session_state["seen_runs"] = {str(p) for p in (tmp_path / "history").rglob("run-*.json")}

    # Fetching should return nothing.
    assert app_mod._fetch_new_scheduled_runs() == []

    # Write a NEW file after seeding.
    _write_run_file(tmp_path / "history", "job-new", 1, "new run", "2026-01-01T00:10:00+00:00")
    updates = app_mod._fetch_new_scheduled_runs()
    assert len(updates) == 1
    assert updates[0]["content"] == "new run"


def test_fetch_new_runs_returns_updates_sorted_by_completion_time(tmp_path, monkeypatch):
    app_mod, st = _stub_streamlit_and_load(tmp_path, monkeypatch)
    st.session_state["seen_runs"] = set()

    _write_run_file(tmp_path / "history", "job-b", 2, "later", "2026-01-01T02:00:00+00:00")
    _write_run_file(tmp_path / "history", "job-a", 1, "earlier", "2026-01-01T00:00:00+00:00")

    updates = app_mod._fetch_new_scheduled_runs()
    assert [u["content"] for u in updates] == ["earlier", "later"]


def test_plan_summary_html_includes_absolute_start(tmp_path, monkeypatch):
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    from planner import SearchPlan

    plan = SearchPlan(
        search_query="ai news",
        should_schedule=True,
        wait_minutes=10,
        run_count=3,
        absolute_start_iso="2026-08-10T09:00:00+00:00",
    )
    html = app_mod._plan_summary_html(plan)
    assert "ai news" in html
    assert "10m" in html
    assert "runs" in html and "3" in html
    assert "2026-08-10T09:00:00+00:00" in html


def test_streamlit_app_imports_when_autorefresh_missing(tmp_path, monkeypatch):
    """If `streamlit-autorefresh` is not installed, importing streamlit_app
    must still succeed and expose a callable `st_autorefresh` stub — the app
    just runs without auto-polling and shows a warning.
    """
    # Ensure streamlit_autorefresh cannot be imported.
    import sys
    monkeypatch.setitem(sys.modules, "streamlit_autorefresh", None)

    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)

    # The stub must be callable and return an integer (mimicking real signature).
    result = app_mod.st_autorefresh(interval=5000, key="test")
    assert result == 0


def test_render_scheduled_run_card_produces_expected_html(tmp_path, monkeypatch):
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    run = {
        "job_id": "abc12345",
        "run_number": 2,
        "completed_at": "2026-08-09T17:37:56.786280+00:00",
        "content": "**Bold text** and a bullet\n- point one\n- point two",
    }
    html = app_mod._render_scheduled_run_card(run)
    assert 'class="sched-card"' in html
    assert 'class="sched-badge"' in html
    assert "scheduled run" in html
    assert "abc12345" in html
    assert "run #2" in html
    # Time was reformatted to HH:MM:SS
    assert "17:37:56" in html
    # Bold markdown got converted
    assert "<strong>Bold text</strong>" in html


def test_markdown_to_html_handles_bold_code_and_newlines(tmp_path, monkeypatch):
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    out = app_mod._markdown_to_html("Hello **world** with `code` and\nnewline")
    assert "<strong>world</strong>" in out
    assert "<code>code</code>" in out
    assert "<br>" in out


def test_markdown_to_html_escapes_html_tags_from_llm_output(tmp_path, monkeypatch):
    """LLM output is untrusted — must not allow raw HTML injection into the card."""
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    out = app_mod._markdown_to_html('<script>alert("xss")</script>')
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_conversation_title_uses_first_user_message_and_truncates(tmp_path, monkeypatch):
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    assert app_mod._conversation_title([]) == "new chat"
    assert app_mod._conversation_title([{"role": "assistant", "content": "hi"}]) == "new chat"
    assert app_mod._conversation_title([{"role": "user", "content": "  hello   world "}]) == "hello world"
    long_prompt = "a very long question about the constitution " * 3
    title = app_mod._conversation_title([{"role": "user", "content": long_prompt}])
    assert title.endswith("…")
    assert len(title) <= 31


def test_route_scheduled_run_prefers_owner_chat_then_fallback(tmp_path, monkeypatch):
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    conversations = {
        "old": {"title": "old", "messages": []},
        "newer": {"title": "newer", "messages": []},
    }
    # A job started in "newer" routes there even while another chat is older.
    assert app_mod._route_scheduled_run("job-1", conversations, {"job-1": "newer"}) == "newer"
    # Unknown job (e.g. resumed after restart) falls back to the oldest chat.
    assert app_mod._route_scheduled_run("job-2", conversations, {}) == "old"
    # Jobs routed to a deleted chat fall back too.
    assert app_mod._route_scheduled_run("job-1", {"old": conversations["old"]}, {"job-1": "gone"}) == "old"
    # No chats at all: nothing to route to.
    assert app_mod._route_scheduled_run("job-2", {}, {}) is None


def test_conversations_persist_roundtrip(tmp_path, monkeypatch):
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    conversations = {
        "c1": {"title": "t", "messages": [{"role": "user", "content": "question"}]},
    }
    conv_id = app_mod._create_conversation(conversations)
    assert conv_id in conversations and conversations[conv_id]["messages"] == []
    app_mod._persist_conversations(conversations, "c1", {"job-x": "c1"})

    loaded = app_mod._load_conversations()
    assert loaded["conversations"]["c1"]["messages"][0]["content"] == "question"
    assert loaded["conversations"][conv_id]["title"] == "new chat"
    assert loaded["active_conv"] == "c1"
    assert loaded["job_conv"] == {"job-x": "c1"}

    # A corrupt file must not crash loading — start fresh.
    app_mod._conversations_path().write_text("{not json", encoding="utf-8")
    assert app_mod._load_conversations()["conversations"] == {}


def test_conversation_lock_is_shared_locked_file(tmp_path, monkeypatch):
    """Lock and persist use the same lock file beside the conversations store."""
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    assert app_mod._store_lock_path() == app_mod._conversations_path().with_suffix(".json.lock")
    assert not app_mod._store_lock_path().exists()
    # Persist under the lock writes the store fine and leaves a lock file.
    app_mod._persist_conversations({"c1": {"title": "t", "messages": []}}, "c1", {})
    assert app_mod._conversations_path().is_file()
    assert app_mod._store_lock_path().exists()
    # Re-entrant use (load view) must not deadlock.
    with app_mod._store_lock():
        loaded = app_mod._load_conversations()
    assert "c1" in loaded["conversations"]


def test_history_to_langchain_skips_sched_cards_and_trims(tmp_path, monkeypatch):
    from langchain_core.messages import HumanMessage

    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "<div class=\"sched-card\">run card</div>", "kind": "scheduled_run"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    msgs = app_mod._history_to_langchain(history)
    assert len(msgs) == 3  # the scheduled-run card is skipped
    assert isinstance(msgs[0], HumanMessage) and msgs[0].content == "first question"
    assert msgs[-1].content == "second answer"

    # Budget trimming: an oversized earlier message is truncated to fit.
    long_msg = {"role": "assistant", "content": "x" * 1000}
    trimmed = app_mod._history_to_langchain([long_msg, {"role": "user", "content": "y"}])
    assert len(trimmed) == 2
    total = sum(len(str(m.content)) for m in trimmed)
    assert total <= app_mod.HISTORY_MAX_CHARS


def test_build_llm_messages_places_history_between_system_and_prompt(tmp_path, monkeypatch):
    from langchain_core.messages import HumanMessage, SystemMessage
    from planner import SearchPlan

    app_mod, st = _stub_streamlit_and_load(tmp_path, monkeypatch)
    st.session_state["rag_mode"] = True
    app_mod.ragtool.active_source_label = lambda: "ACTIVE DOC LABEL"

    plan = SearchPlan(search_query="python news", should_schedule=False, wait_minutes=0.0, run_count=1)
    history = [{"role": "user", "content": "tell me about python"}, {"role": "assistant", "content": "python rocks"}]
    msgs = app_mod._build_llm_messages("more details", plan, history)

    system_count = sum(isinstance(m, SystemMessage) for m in msgs)
    assert system_count == 2  # RAG label + search instruction
    assert "ACTIVE DOC LABEL" in msgs[0].content
    # History sits between system context and the live prompt.
    assert msgs[system_count].content == "tell me about python"
    assert msgs[-1].content == "more details"
    assert isinstance(msgs[-1], HumanMessage)


def test_stream_chatbot_yields_chunks_and_usage(tmp_path, monkeypatch):
    from planner import SearchPlan

    app_mod, st = _stub_streamlit_and_load(tmp_path, monkeypatch)
    st.session_state["rag_mode"] = True
    app_mod.ragtool.active_source_label = lambda: None

    class FakeChunk:
        def __init__(self, content="", usage=None):
            self.content = content
            self.usage_metadata = usage

    class FakeBot:
        def __init__(self, fail_stream=False):
            self.fail_stream = fail_stream
        def stream(self, inputs, stream_mode="messages"):
            if self.fail_stream:
                raise RuntimeError("stream broke")
            yield FakeChunk("Hel"), {}
            yield FakeChunk("lo", {"input_tokens": 3, "output_tokens": 2}), {}
        def invoke(self, inputs):
            return {"messages": [type("Msg", (), {"content": "fallback reply"})()]}

    plan = SearchPlan(search_query="q", should_schedule=False, wait_minutes=0.0, run_count=1)

    app_mod.chatbot = FakeBot()
    out = list(app_mod._stream_chatbot("hi", plan, []))
    assert out == [("Hel", None), ("lo", {"input_tokens": 3, "output_tokens": 2})]

    # Streaming failure falls back to a single non-streamed reply.
    app_mod.chatbot = FakeBot(fail_stream=True)
    out = list(app_mod._stream_chatbot("hi", plan, []))
    assert out == [("fallback reply", None)]


def test_user_friendly_error_hides_internals(tmp_path, monkeypatch):
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    msg = app_mod._user_friendly_error(RuntimeError("SECRET_KEY=abc123"))
    assert "SECRET_KEY" not in msg
    assert "RuntimeError" in msg


def test_conversation_to_markdown_strips_card_html(tmp_path, monkeypatch):
    app_mod, _ = _stub_streamlit_and_load(tmp_path, monkeypatch)
    md = app_mod._conversation_to_markdown(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "plain answer"},
            {"role": "assistant", "content": '<div class="sched-card">**bold run**</div>', "kind": "scheduled_run"},
        ]
    )
    assert "plain answer" in md
    assert '<div class="sched-card">' not in md
    assert "bold run" in md


def test_render_live_runs_in_appends_cards_to_container(tmp_path, monkeypatch):
    app_mod, st = _stub_streamlit_and_load(tmp_path, monkeypatch)
    st.session_state["conversations"] = {"c1": {"title": "t", "messages": []}}
    st.session_state["active_conv"] = "c1"
    st.session_state["job_conv"] = {}
    st.session_state["seen_runs"] = set()

    rendered = []
    run = {
        "job_id": "job-x",
        "run_number": 1,
        "completed_at": "2026-08-09T17:37:56.786280+00:00",
        "content": "run produced this",
    }
    ok = app_mod._render_live_runs_in([run], lambda html, unsafe_allow_html: rendered.append(html))
    assert ok is True
    assert len(rendered) == 1 and "run produced this" in rendered[0]
    stored = st.session_state["conversations"]["c1"]["messages"]
    assert stored[-1]["kind"] == "scheduled_run"
    # Runs routed to another chat render nothing for the active view.
    st.session_state["conversations"]["other"] = {"title": "other chat", "messages": []}
    st.session_state["job_conv"]["job-x"] = "other"
    rendered.clear()
    app_mod._render_live_runs_in([run], lambda html, unsafe_allow_html: rendered.append(html))
    assert rendered == []
    assert st.session_state["conversations"]["other"]["messages"][-1]["kind"] == "scheduled_run"
