"""A batch job is only "completed" when its PDF report really exists.

The model, PDF generator, storage and email are replaced by fakes, so nothing external is called.
"""
import asyncio

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.services import batch_processor
from app.services.batch_processor import (
    BATCH_FAILED_MESSAGE,
    QUERY_FAILED_MESSAGE,
    REPORT_FAILED_MESSAGE,
)

# The kind of detail AWS puts in its error: it must never reach the user
AWS_ERROR = (
    'Could not upload file: User: arn:aws:iam::123456789012:user/pdp_user is not authorized to '
    'perform: s3:PutObject on resource: "arn:aws:s3:::secret-bucket/batch_reports/x.pdf"'
)


class FakeCollection:
    def __init__(self):
        self.doc = {}
        self.updates = []

    def update_one(self, _query, update):
        self.updates.append(update)
        self.doc.update(update.get("$set", {}))
        for field, value in update.get("$push", {}).items():
            self.doc.setdefault(field, []).append(value)


class FakeQueryHandler:
    async def process_query_non_streaming(self, **_):
        return {"response": "An answer.", "tab": "data-protection"}


class FakePdf:
    def __init__(self, error=None):
        self.error = error

    def generate_batch_report(self, *_):
        if self.error:
            raise self.error
        return b"%PDF-1.4"


class FakeEmail:
    def __init__(self):
        self.complete = []
        self.failed = []

    def send_batch_complete_email(self, **kwargs):
        self.complete.append(kwargs)

    def send_batch_failed_email(self, **kwargs):
        self.failed.append(kwargs)


class FakeWebsocket:
    def __init__(self):
        self.messages = []

    async def send_update(self, _job_id, message):
        self.messages.append(message)


class FakeStorage:
    upload_error = None

    def upload_file_bytes(self, **_):
        if FakeStorage.upload_error:
            raise FakeStorage.upload_error

    def generate_presigned_url(self, path, expiration=None):
        return f"https://storage.example/{path}?signed"


@pytest.fixture()
def run(monkeypatch):
    email = FakeEmail()
    websocket = FakeWebsocket()
    collection = FakeCollection()
    state = {"pdf": FakePdf()}

    monkeypatch.setattr(batch_processor, "get_query_handler", lambda: FakeQueryHandler())
    monkeypatch.setattr(batch_processor, "get_pdf_generator", lambda: state["pdf"])
    monkeypatch.setattr(batch_processor, "get_email_service", lambda: email)
    monkeypatch.setattr(batch_processor, "FileStorageService", FakeStorage)
    monkeypatch.setattr(FakeStorage, "upload_error", None)
    monkeypatch.setattr(settings, "use_s3", True)

    def go(user_email="user@example.com", pdf_error=None, storage_error=None):
        state["pdf"] = FakePdf(pdf_error)
        FakeStorage.upload_error = storage_error
        asyncio.run(batch_processor.process_batch_queries_background(
            job_id="job-1", queries=["What is the law?"], user_email=user_email,
            collection=collection, websocket_manager=websocket,
        ))
        return collection.doc, email, websocket

    return go


def test_a_saved_report_completes_the_job(run):
    doc, email, websocket = run()
    assert doc["status"] == "completed"
    assert doc["pdf_file_path"] == "batch_reports/job-1.pdf"
    assert len(email.complete) == 1
    assert email.failed == []
    assert websocket.messages[-1]["type"] == "batch_completed"


def test_a_report_that_cannot_be_stored_fails_the_job(run):
    doc, email, websocket = run(storage_error=HTTPException(500, AWS_ERROR))
    assert doc["status"] == "failed"
    assert doc["pdf_file_path"] is None
    assert doc["error_message"] == REPORT_FAILED_MESSAGE
    # The user is told, instead of being sent a "report ready" message
    assert len(email.failed) == 1
    assert email.complete == []
    assert websocket.messages[-1]["type"] == "batch_failed"
    assert all(m["type"] != "batch_completed" for m in websocket.messages)


def test_the_answers_are_kept_when_the_report_fails(run):
    doc, _, _ = run(storage_error=HTTPException(500, AWS_ERROR))
    assert doc["completed_queries"] == 1
    assert doc["failed_queries"] == 0
    assert doc["results"][0]["response"] == "An answer."
    assert doc["processing_time"] is not None


def test_storage_details_never_reach_the_user(run):
    doc, email, websocket = run(storage_error=HTTPException(500, AWS_ERROR))
    everything_the_user_can_see = [doc["error_message"], str(email.failed), str(websocket.messages)]
    for text in everything_the_user_can_see:
        for secret in ("arn:aws", "secret-bucket", "pdp_user", "123456789012"):
            assert secret not in text


def test_a_pdf_that_cannot_be_built_fails_the_job_too(run):
    doc, email, _ = run(pdf_error=ValueError("bad markdown"))
    assert doc["status"] == "failed"
    assert doc["error_message"] == REPORT_FAILED_MESSAGE
    assert len(email.failed) == 1


def test_no_email_is_attempted_without_an_address(run):
    doc, email, websocket = run(user_email=None, storage_error=HTTPException(500, AWS_ERROR))
    assert doc["status"] == "failed"
    assert email.failed == []
    assert websocket.messages[-1]["type"] == "batch_failed"


def test_local_storage_still_completes_the_job(run, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "use_s3", False)
    monkeypatch.setattr(settings, "upload_directory", str(tmp_path))
    doc, _, _ = run()
    assert doc["status"] == "completed"
    assert (tmp_path / "batch_reports" / "job-1.pdf").read_bytes() == b"%PDF-1.4"


def test_each_answer_is_appended_not_the_whole_list_rewritten(run, monkeypatch):
    collection_updates = []
    original = FakeCollection.update_one

    def spy(self, query, update):
        collection_updates.append(update)
        return original(self, query, update)

    monkeypatch.setattr(FakeCollection, "update_one", spy)
    run()
    progress = [u for u in collection_updates if "$push" in u]
    assert len(progress) == 1  # one question
    assert "results" not in progress[0]["$set"]
    assert progress[0]["$push"]["results"]["response"] == "An answer."


def test_an_unexpected_failure_shows_a_generic_message(run, monkeypatch):
    def broken():
        raise RuntimeError("connection to mongodb://admin:hunter2@db failed")

    monkeypatch.setattr(batch_processor, "get_query_handler", broken)
    doc, email, websocket = run()
    assert doc["status"] == "failed"
    assert doc["error_message"] == BATCH_FAILED_MESSAGE
    assert email.failed[0]["error_message"] == BATCH_FAILED_MESSAGE
    assert "hunter2" not in str(websocket.messages)


def test_a_failed_question_shows_a_generic_message_in_the_report(run, monkeypatch):
    class Failing:
        async def process_query_non_streaming(self, **_):
            raise RuntimeError("provider said: invalid key sk-123")

    monkeypatch.setattr(batch_processor, "get_query_handler", lambda: Failing())
    doc, _, _ = run()
    assert doc["failed_queries"] == 1
    assert doc["results"][0]["error"] == QUERY_FAILED_MESSAGE


def test_an_empty_email_address_sends_nothing(run):
    _, email, _ = run(user_email="")
    assert email.complete == [] and email.failed == []
