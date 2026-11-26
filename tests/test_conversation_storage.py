"""Conversation storage: the history that follow-up questions are answered from, and the session
store behind save_conversation()."""
import asyncio
import sqlite3
from datetime import datetime, timedelta

import pytest

from app.core.config import settings
from app.db.chat_history import SQLiteHistoryStore
from app.services import conversation_service as service_module
from app.services.query_handler import DynamicQueryHandler


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_data_dir", str(tmp_path))
    monkeypatch.setattr(service_module, "_conversation_service", None)
    return service_module.get_conversation_service()


def conversation(history, session_id, exchanges):
    for number in range(1, exchanges + 1):
        history.add_message(session_id, "human", f"question {number}")
        history.add_message(session_id, "assistant", f"answer {number}")


def contents(messages):
    return [message["content"] for message in messages]


# ---------------------------------------------------------------- chat history

def test_a_limited_history_holds_the_latest_messages_oldest_first(tmp_path):
    history = SQLiteHistoryStore(str(tmp_path / "chat_history.db"))
    conversation(history, "s1", 5)
    assert contents(history.get_session_history("s1", limit=4)) == [
        "question 4", "answer 4", "question 5", "answer 5"]


def test_the_full_history_is_returned_in_order(tmp_path):
    history = SQLiteHistoryStore(str(tmp_path / "chat_history.db"))
    conversation(history, "s1", 3)
    conversation(history, "another", 1)
    assert contents(history.get_session_history("s1")) == [
        "question 1", "answer 1", "question 2", "answer 2", "question 3", "answer 3"]


def test_the_question_being_asked_is_not_part_of_its_own_context(service, tmp_path, monkeypatch):
    """The chat route stores the question before it is answered."""
    from app.db import chat_history

    history = SQLiteHistoryStore(str(tmp_path / "chat_history.db"))
    monkeypatch.setattr(chat_history, "database", history)
    conversation(history, "s1", 3)
    history.add_message("s1", "human", "is it mandatory?")

    handler = DynamicQueryHandler(db_client=None, conversation_service=service)
    context = asyncio.run(handler.get_context_history("s1", "is it mandatory?"))
    assert contents(context) == ["question 2", "answer 2", "question 3", "answer 3"]


def test_a_first_question_has_no_context(service, tmp_path, monkeypatch):
    from app.db import chat_history

    history = SQLiteHistoryStore(str(tmp_path / "chat_history.db"))
    monkeypatch.setattr(chat_history, "database", history)
    history.add_message("new", "human", "what is this law?")

    handler = DynamicQueryHandler(db_client=None, conversation_service=service)
    assert asyncio.run(handler.get_context_history("new", "what is this law?")) == []


# ---------------------------------------------------------------- session store

class NoMongo:
    """Fails the test if saving falls back to MongoDB."""

    def get_collection(self, name):
        raise AssertionError("the session store failed and fell back to MongoDB")


def save(service, session_id, question, client_id=None):
    handler = DynamicQueryHandler(db_client=NoMongo(), conversation_service=service)
    asyncio.run(handler.save_conversation(session_id, question, "an answer", client_id=client_id))


def stored(service, session_id, client_id=None):
    return contents(asyncio.run(service.get_conversation_history(session_id, client_id=client_id)))


def test_the_database_is_kept_with_the_other_data(service, tmp_path):
    """Not in the working directory, which the app cannot write to in Docker."""
    save(service, "s1", "q")
    assert (tmp_path / "conversations.db").exists()


def test_a_conversation_is_saved_under_the_callers_session_id(service):
    save(service, "s1", "first")
    save(service, "s1", "second")
    assert stored(service, "s1") == ["first", "an answer", "second", "an answer"]


def expire(tmp_path, session_id):
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    with sqlite3.connect(tmp_path / "conversations.db") as database:
        database.execute("UPDATE sessions SET expires_at = ? WHERE session_id = ?", (past, session_id))


def test_an_expired_session_is_renewed_and_keeps_its_id(service, tmp_path):
    save(service, "s1", "before")
    expire(tmp_path, "s1")
    save(service, "s1", "after")
    assert stored(service, "s1") == ["before", "an answer", "after", "an answer"]
    assert asyncio.run(service.sessions.get("s1")).is_active()


@pytest.mark.parametrize("expired", [False, True])
def test_another_client_cannot_use_a_session(service, tmp_path, expired):
    asyncio.run(service.get_or_create_session("s1", client_id="owner"))
    if expired:
        expire(tmp_path, "s1")
    with pytest.raises(service_module.UnauthorizedAccessError):
        asyncio.run(service.get_or_create_session("s1", client_id="intruder"))
