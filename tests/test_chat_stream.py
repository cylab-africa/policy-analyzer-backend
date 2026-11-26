"""The chat stream ends with an error event, not silence, when the answer fails part-way, and
error details never reach the visitor. The query handler and history store are fakes."""
import json

import pytest
from fastapi.testclient import TestClient


class FakeHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, **kwargs):
        self.messages.append(kwargs)


@pytest.fixture()
def chat(monkeypatch):
    import main
    from app.services import query_handler

    state = {"fail_after": None}

    class Handler:
        async def process_query(self, **_):
            yield "First part. "
            if state["fail_after"] is not None:
                raise RuntimeError("qdrant at 10.0.0.5 refused the connection")
            yield "Second part."

    monkeypatch.setattr(query_handler, "get_query_handler", lambda *a, **k: Handler())
    history = FakeHistory()
    monkeypatch.setattr(main, "chat_history_db", history)
    client = TestClient(main.app)

    def ask():
        response = client.post(
            "/api/query_v2",
            json={"query": "What is the law?", "session_id": "s1"},
            headers={"X-API-Key": main.settings.api_key},
        )
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        return response, events, history

    return state, ask


def test_a_complete_answer_streams_its_parts(chat):
    state, ask = chat
    response, events, history = ask()
    assert response.status_code == 200
    assert "".join(e["content"] for e in events if e["type"] == "content") == "First part. Second part."
    assert not any(e["type"] == "error" for e in events)
    assert history.messages[-1]["role"] == "assistant"


def test_a_failure_mid_answer_ends_with_an_error_event(chat):
    state, ask = chat
    state["fail_after"] = 1
    response, events, _ = ask()
    assert events[-1] == {"type": "error", "message": "The answer could not be completed. Please try again."}
    assert "10.0.0.5" not in response.text


def test_a_question_over_the_length_limit_is_refused(chat):
    import main

    response = TestClient(main.app).post(
        "/api/query_v2",
        json={"query": "x" * 4001},
        headers={"X-API-Key": main.settings.api_key},
    )
    assert response.status_code == 422
