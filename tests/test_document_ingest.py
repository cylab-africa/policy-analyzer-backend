"""Document ingestion reports the files that failed, and removes private uploads once indexed.

Before, a failing file was skipped silently and the task was marked "completed", and private
uploads were never deleted: the route did not pass their visibility, and the S3 delete needed an
event loop that the background worker thread does not have. Nothing external is contacted.
"""
import asyncio
import threading

import pytest
from langchain_core.documents import Document

from app.routes import document_processing as route
from app.services.document_processing import DocumentProcessingV2


class FakeLangchainStore:
    def __init__(self):
        self.added = []

    def add_documents(self, docs):
        self.added.extend(docs)


class FakeVectorStore:
    def __init__(self):
        self.store = FakeLangchainStore()

    def get_vector_store(self):
        return self.store


class FakeStorage:
    def __init__(self):
        self.deleted = []

    def delete_file_sync(self, key):
        self.deleted.append(key)


def make_processor(storage=None):
    processor = DocumentProcessingV2(llm=None, vector_store=FakeVectorStore(), file_storage=storage)

    def read_text(doc, file_local=False):
        if "bad" in doc["link"]:
            raise RuntimeError("cannot read https://bucket/presigned?X-Amz-Signature=secret")
        return [Document(page_content="Some policy text.", metadata={})]

    processor._process_txt = read_text
    return processor


def txt(link, **extra):
    return {"link": link, "document_type": "text/plain", **extra}


def run_in_worker_thread(fn):
    """Run as Starlette runs a background task: in a thread with no event loop."""
    out = {}
    thread = threading.Thread(target=lambda: out.update(result=fn()))
    thread.start()
    thread.join()
    return out["result"]


def test_failed_files_are_reported_and_the_others_still_indexed():
    processor = make_processor()
    summary = processor.process_and_store(
        [txt("https://s3/good", file_name="good.txt"), txt("https://s3/bad", file_name="bad.txt")]
    )
    assert summary["processed"] == 1
    assert summary["failed"] == ["bad.txt"]
    assert len(processor.vector_store.store.added) >= 1


def test_web_links_count_as_failed_rather_than_vanishing():
    summary = make_processor().process_and_store(
        [{"link": "https://example.org", "document_type": "web", "file_name": "page"}]
    )
    assert summary == {"total": 1, "processed": 0, "failed": ["page"], "private_not_deleted": []}


def test_private_s3_uploads_are_deleted_from_a_worker_thread():
    storage = FakeStorage()
    processor = make_processor(storage)
    docs = [
        txt("https://s3/one", visibility="private", stored_file="uploads/one.txt"),
        txt("https://s3/public", visibility="public", stored_file="uploads/public.txt"),
        txt("https://s3/bad", visibility="private", stored_file="uploads/bad.txt"),
    ]
    summary = run_in_worker_thread(lambda: processor.process_and_store(docs, local_files=False))
    # Only the private file that was indexed is removed; a failed one is kept, public ones too
    assert storage.deleted == ["uploads/one.txt"]
    assert summary["private_not_deleted"] == []


def test_private_s3_upload_without_storage_is_reported_as_not_deleted():
    summary = make_processor(storage=None).process_and_store(
        [txt("https://s3/one", visibility="private", stored_file="uploads/one.txt")]
    )
    assert summary["private_not_deleted"] == ["uploads/one.txt"]


def test_private_local_upload_is_deleted_but_ingest_folder_files_are_not(tmp_path):
    upload = tmp_path / "upload.txt"
    folder_file = tmp_path / "operator_folder.txt"
    upload.write_text("x")
    folder_file.write_text("x")
    make_processor().process_and_store(
        [
            txt(str(upload), visibility="private", stored_file=str(upload)),
            txt(str(folder_file), visibility="private"),  # from folder processing: no stored_file
        ],
        local_files=True,
    )
    assert not upload.exists()
    assert folder_file.exists()


# --- the background task in the route --------------------------------------------------------

class FakeTasks:
    def __init__(self):
        self.updates = []

    def update_one(self, _query, update):
        self.updates.append(update["$set"])


class FakeDownload:
    def __init__(self, path):
        self.download_url = f"https://s3.example/{path}?signed"


class FakeRouteStorage:
    def generate_download_url(self, file_path):
        return FakeDownload(file_path)


def run_background(monkeypatch, summary, files, allowed_root=None, local_files=False):
    seen = {}

    class FakeProcessor:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def process_and_store(self, documents, local_files):
            seen["documents"] = documents
            return summary

    monkeypatch.setattr(route, "DocumentProcessingV2", FakeProcessor)
    tasks = FakeTasks()
    storage = FakeRouteStorage()
    asyncio.run(route.process_documents_background_v2(
        files, "pid", storage, tasks, None, None, "general", local_files, allowed_root
    ))
    return tasks.updates[-1], seen, storage


UPLOADS = [
    {"file_path": "uploads/a.pdf", "filename": "a.pdf", "content_type": "application/pdf",
     "client_id": "c1", "visibility": "private"},
    {"file_path": "uploads/b.pdf", "filename": "b.pdf", "content_type": "application/pdf",
     "client_id": "c1", "visibility": "private"},
]


def test_uploads_pass_visibility_storage_and_the_copy_to_delete(monkeypatch):
    _, seen, storage = run_background(
        monkeypatch, {"processed": 2, "failed": []}, UPLOADS
    )
    assert seen["init"]["file_storage"] is storage
    first = seen["documents"][0]
    assert first["visibility"] == "private"
    assert first["stored_file"] == "uploads/a.pdf"
    assert first["file_name"] == "a.pdf"


def test_folder_processing_never_marks_files_for_deletion(monkeypatch, tmp_path):
    real = tmp_path / "doc.pdf"
    real.write_text("x")
    files = [{"file_path": str(real), "filename": "doc.pdf", "content_type": "application/pdf",
              "client_id": "c1", "visibility": "private"}]
    _, seen, _ = run_background(
        monkeypatch, {"processed": 1, "failed": []}, files, allowed_root=str(tmp_path), local_files=True
    )
    assert "stored_file" not in seen["documents"][0]


def test_all_files_indexed_is_completed(monkeypatch):
    update, _, _ = run_background(monkeypatch, {"processed": 2, "failed": []}, UPLOADS)
    assert update["status"] == "completed"
    assert update["processed_count"] == 2
    assert update["failed_count"] == 0
    assert update["error_message"] is None


def test_some_files_failing_is_completed_with_their_names(monkeypatch):
    update, _, _ = run_background(monkeypatch, {"processed": 1, "failed": ["b.pdf"]}, UPLOADS)
    assert update["status"] == "completed"
    assert update["failed_count"] == 1
    assert update["failed_files"] == ["b.pdf"]
    assert update["error_message"] == "1 of 2 files could not be processed: b.pdf"


def test_no_file_indexed_is_an_error(monkeypatch):
    update, _, _ = run_background(monkeypatch, {"processed": 0, "failed": ["a.pdf", "b.pdf"]}, UPLOADS)
    assert update["status"] == "error"
    assert update["processed_count"] == 0


def test_an_unexpected_failure_hides_its_details(monkeypatch):
    class Exploding:
        def __init__(self, **_):
            pass

        def process_and_store(self, documents, local_files):
            raise RuntimeError("s3://bucket/secret-path")

    monkeypatch.setattr(route, "DocumentProcessingV2", Exploding)
    tasks = FakeTasks()
    asyncio.run(route.process_documents_background_v2(
        UPLOADS, "pid", FakeRouteStorage(), tasks, None, None, "general", False, None
    ))
    assert tasks.updates[-1]["status"] == "error"
    assert "secret" not in tasks.updates[-1]["error_message"]


# --- the status the website polls ------------------------------------------------------------

@pytest.fixture()
def status_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    task = {
        "processing_id": "pid-1", "status": "completed", "file_count": 2, "processed_count": 1,
        "failed_count": 1, "failed_files": ["b.pdf"], "error_message": "1 of 2 files could not be processed: b.pdf",
        "client_id": "client-secret", "visibility": "private",
        "files_data": [{"file_path": "uploads/secret-key-a.pdf"}],
    }

    class Collection:
        def find_one(self, query):
            return dict(task) if query.get("processing_id") == "pid-1" else None

    monkeypatch.setattr(route, "get_mongo_client", lambda: type("M", (), {"get_collection": lambda self, n: Collection()})())
    app = FastAPI()
    app.include_router(route.router, prefix="/api/documents")
    return TestClient(app)


def test_status_reports_counts_and_failed_names_only(status_client):
    response = status_client.get("/api/documents/status/pid-1")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["processed_count"] == 1 and body["failed_files"] == ["b.pdf"]
    text = response.text
    assert "client-secret" not in text and "secret-key-a" not in text


def test_status_of_an_unknown_upload_is_404(status_client):
    assert status_client.get("/api/documents/status/nope").status_code == 404


def test_the_task_is_marked_processing_when_it_starts(monkeypatch):
    first, _, _ = None, None, None
    tasks = FakeTasks()

    class Processor:
        def __init__(self, **_):
            pass

        def process_and_store(self, documents, local_files):
            assert tasks.updates[-1]["status"] == "processing"
            return {"processed": 1, "failed": []}

    monkeypatch.setattr(route, "DocumentProcessingV2", Processor)
    asyncio.run(route.process_documents_background_v2(
        UPLOADS[:1], "pid", FakeRouteStorage(), tasks, None, None, "general", False, None
    ))
    assert tasks.updates[-1]["status"] == "completed"
