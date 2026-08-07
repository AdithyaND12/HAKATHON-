import threading
import time
from types import SimpleNamespace

import pytest
import requests

import app


class FakeResponse:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


def invoke_stock(symbol="AAPL"):
    return app.get_stock_price.invoke({"symbol": symbol})


def test_stock_request_uses_configuration_and_returns_payload(monkeypatch):
    calls = []
    payload = {"Global Quote": {"05. price": "123.45"}}

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(payload)

    monkeypatch.setattr(app, "ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setattr(app, "HTTP_TIMEOUT_SECONDS", 2.5)
    monkeypatch.setattr(app.requests, "get", fake_get)

    assert invoke_stock() == payload
    assert calls == [
        (
            "https://www.alphavantage.co/query",
            {
                "params": {
                    "function": "GLOBAL_QUOTE",
                    "symbol": "AAPL",
                    "apikey": "test-key",
                },
                "timeout": 2.5,
            },
        )
    ]


def test_stock_request_reports_missing_key_without_request(monkeypatch):
    request_called = False

    def fake_get(*args, **kwargs):
        nonlocal request_called
        request_called = True

    monkeypatch.setattr(app, "ALPHAVANTAGE_API_KEY", "")
    monkeypatch.setattr(app.requests, "get", fake_get)

    result = invoke_stock()

    assert result["error_type"] == "missing_api_key"
    assert "ALPHAVANTAGE_API_KEY" in result["error"]
    assert not request_called


@pytest.mark.parametrize(
    ("exception", "error_type"),
    [
        (requests.exceptions.Timeout(), "timeout"),
        (requests.exceptions.ConnectionError(), "connection_error"),
        (requests.exceptions.HTTPError(), "http_error"),
    ],
)
def test_stock_request_reports_request_failures(monkeypatch, exception, error_type):
    monkeypatch.setattr(app, "ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setattr(app.requests, "get", lambda *args, **kwargs: FakeResponse(error=exception))

    result = invoke_stock()

    assert result["error_type"] == error_type
    assert result["error"]


def test_stock_request_reports_invalid_json(monkeypatch):
    monkeypatch.setattr(app, "ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setattr(app.requests, "get", lambda *args, **kwargs: FakeResponse(ValueError()))

    result = invoke_stock()

    assert result["error_type"] == "invalid_json"


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"Error Message": "Invalid API call."}, "api_error"),
        ({"Note": "Thank you for using Alpha Vantage!"}, "rate_limit"),
        ({"Information": "Our standard API call frequency is 5 calls per minute."}, "rate_limit"),
    ],
)
def test_stock_request_reports_api_errors(monkeypatch, payload, error_type):
    monkeypatch.setattr(app, "ALPHAVANTAGE_API_KEY", "test-key")
    monkeypatch.setattr(app.requests, "get", lambda *args, **kwargs: FakeResponse(payload))

    result = invoke_stock()

    assert result["error_type"] == error_type


def fake_chatbot_result():
    return {"messages": [SimpleNamespace(content="answer")]}


def test_schedule_returns_without_blocking_and_runs_requested_count(monkeypatch):
    calls = []

    class FakeChatbot:
        def invoke(self, state):
            calls.append(state)
            return fake_chatbot_result()

    monkeypatch.setattr(app, "chatbot", FakeChatbot())
    started = time.monotonic()
    job = app.run_scheduled_search(
        "check the price",
        interval_minutes=0.001,
        run_count=2,
        search_query="price",
    )
    returned_in = time.monotonic() - started
    job.join(timeout=2)

    assert returned_in < 0.2
    assert job.done
    assert job.error is None
    assert len(job.results) == 2
    assert len(calls) == 2


def test_schedule_can_be_stopped_during_interval(monkeypatch):
    first_run = threading.Event()

    class FakeChatbot:
        def invoke(self, state):
            first_run.set()
            return fake_chatbot_result()

    monkeypatch.setattr(app, "chatbot", FakeChatbot())
    job = app.run_scheduled_search(
        "monitor this",
        interval_minutes=1,
        run_count=3,
        search_query="this",
    )
    assert first_run.wait(timeout=1)
    job.stop()
    job.join(timeout=1)

    assert job.done
    assert len(job.results) == 1


def test_schedule_does_not_wait_after_final_run(monkeypatch):
    class FakeChatbot:
        def invoke(self, state):
            return fake_chatbot_result()

    monkeypatch.setattr(app, "chatbot", FakeChatbot())
    started = time.monotonic()
    job = app.run_scheduled_search(
        "one check",
        interval_minutes=1,
        run_count=1,
        search_query="one check",
    )
    job.join(timeout=1)

    assert time.monotonic() - started < 0.5
    assert len(job.results) == 1
