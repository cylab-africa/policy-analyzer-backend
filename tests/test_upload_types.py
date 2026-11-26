"""Uploads are judged by their file extension, and the folder listing no longer breaks the allowed list."""
import pytest
from fastapi import HTTPException

from app.services.file_management import FileStorageService

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture()
def storage(monkeypatch):
    def fake_init(self):
        self.s3_client = object()
        self.bucket_name = "test-bucket"
        self.allowed_mime_types = FileStorageService._parse_allowed_types(None)
        self.max_file_size = FileStorageService._DEFAULT_MAX_FILE_SIZE

    monkeypatch.setattr(FileStorageService, "_init_s3_client", fake_init)
    return FileStorageService()


def test_the_default_allowed_types_are_the_documents_the_app_can_read():
    types = FileStorageService._parse_allowed_types(None)
    assert set(types) == {"application/pdf", "text/plain", "text/markdown", DOCX}


def test_the_old_word_format_is_refused_even_if_configured(storage):
    storage.allowed_mime_types = FileStorageService._parse_allowed_types("application/msword,application/pdf")
    assert storage.allowed_mime_types == ["application/pdf"]
    with pytest.raises(HTTPException):
        storage.validate_mime_type(storage.resolve_mime_type("law.doc", "application/msword"))


def test_a_configured_list_replaces_the_default():
    assert FileStorageService._parse_allowed_types(" application/pdf , text/plain,") == ["application/pdf", "text/plain"]


@pytest.mark.parametrize("declared", ["text/markdown", "application/octet-stream", "", None, "text/x-markdown"])
def test_a_markdown_file_is_accepted_whatever_the_browser_declares(storage, declared):
    mime = storage.resolve_mime_type("notes.md", declared)
    assert mime == "text/markdown"
    storage.validate_mime_type(mime)


@pytest.mark.parametrize("name, expected", [
    ("law.PDF", "application/pdf"),
    ("law.txt", "text/plain"),
    ("law.docx", DOCX),
    ("archive.tar.md", "text/markdown"),
])
def test_the_extension_decides_the_type(storage, name, expected):
    assert storage.resolve_mime_type(name, "application/octet-stream") == expected


def test_an_unknown_extension_keeps_the_declared_type_and_is_refused(storage):
    mime = storage.resolve_mime_type("photo.png", "image/png")
    assert mime == "image/png"
    with pytest.raises(HTTPException) as refused:
        storage.validate_mime_type(mime)
    assert refused.value.status_code == 400


def test_a_file_without_a_name_or_type_is_refused(storage):
    mime = storage.resolve_mime_type(None, None)
    assert mime == "application/octet-stream"
    with pytest.raises(HTTPException):
        storage.validate_mime_type(mime)


def test_a_type_missing_from_the_configured_list_is_still_refused(storage):
    storage.allowed_mime_types = ["application/pdf"]
    with pytest.raises(HTTPException):
        storage.validate_mime_type(storage.resolve_mime_type("notes.md", "text/markdown"))


def test_listing_a_folder_works_without_the_spreadsheet_types_in_the_allowed_list(storage, tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.md").write_text("# hi")
    (tmp_path / "sheet.xlsx").write_bytes(b"x")
    names = sorted(f["file_name"] for f in storage.load_file_paths(str(tmp_path)))
    assert names == ["a.pdf", "notes.md"]


def test_listing_a_folder_leaves_the_allowed_list_untouched(storage, tmp_path):
    storage.allowed_mime_types = storage._parse_allowed_types(None) + [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]
    before = list(storage.allowed_mime_types)
    (tmp_path / "sheet.xls").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("hello")
    assert [f["file_name"] for f in storage.load_file_paths(str(tmp_path))] == ["a.txt"]
    storage.load_file_paths(str(tmp_path))  # a second call used to crash after the first one emptied the list
    assert storage.allowed_mime_types == before


# --- through the upload route -------------------------------------------------------------

@pytest.fixture()
def upload_client(monkeypatch, tmp_path, storage):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.routes import document_processing

    queued = []

    class FakeCollection:
        def insert_one(self, record):
            queued.append(record)

    monkeypatch.setattr(settings, "use_s3", False)
    monkeypatch.setattr(settings, "upload_rate_limit_per_hour", 0)  # tested on its own below
    monkeypatch.setattr(settings, "upload_directory", str(tmp_path))
    monkeypatch.setattr(document_processing, "get_mongo_client", lambda: type("M", (), {"get_collection": lambda self, name: FakeCollection()})())
    monkeypatch.setattr(document_processing, "get_vector_stores_v2", lambda: {"public": None, "private": None})
    monkeypatch.setattr(document_processing.LLMFactory, "create_llm", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(document_processing, "process_documents_background_v2", lambda *a, **k: None)

    app = FastAPI()
    app.include_router(document_processing.router, prefix="/api/documents")
    app.dependency_overrides[document_processing.get_file_storage_service] = lambda: storage
    return TestClient(app), queued, tmp_path


def test_the_upload_route_accepts_markdown_whatever_content_type_the_browser_sends(upload_client):
    client, queued, folder = upload_client
    response = client.post(
        "/api/documents/upload-documents",
        files=[("files", ("policy.md", b"# Policy", "application/octet-stream"))],
    )
    assert response.status_code == 200, response.text
    assert queued[0]["files_data"][0]["content_type"] == "text/markdown"
    assert any(p.suffix == ".md" for p in folder.iterdir())


def test_the_upload_route_still_refuses_other_file_types(upload_client):
    client, queued, _ = upload_client
    response = client.post(
        "/api/documents/upload-documents",
        files=[("files", ("script.exe", b"MZ", "application/octet-stream"))],
    )
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]
    assert queued == []


def test_an_empty_file_is_refused(upload_client):
    client, queued, _ = upload_client
    response = client.post("/api/documents/upload-documents", files=[("files", ("empty.txt", b"", "text/plain"))])
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    assert queued == []


def test_a_file_over_the_size_limit_is_refused(upload_client, storage, monkeypatch):
    client, queued, _ = upload_client
    monkeypatch.setattr(storage, "max_file_size", 10)
    response = client.post(
        "/api/documents/upload-documents", files=[("files", ("big.txt", b"x" * 11, "text/plain"))]
    )
    assert response.status_code == 413
    assert "big.txt" in response.json()["detail"]
    assert queued == []


# --- limits: each uploaded file costs model calls, and any website visitor can upload --------

@pytest.fixture()
def limited_client(upload_client, monkeypatch):
    from app.core.config import settings
    from app.routes.document_processing import upload_rate_limit

    monkeypatch.setattr(settings, "upload_rate_limit_per_hour", 2)
    upload_rate_limit.limiter.reset()
    yield upload_client
    upload_rate_limit.limiter.reset()


def send(client, *names_and_bodies):
    return client.post(
        "/api/documents/upload-documents",
        files=[("files", (name, body, "text/plain")) for name, body in names_and_bodies],
    )


def test_uploads_over_the_hourly_limit_are_refused_and_not_stored(limited_client):
    client, queued, folder = limited_client
    assert [send(client, ("a.txt", b"x")).status_code for _ in range(2)] == [200, 200]
    refused = send(client, ("b.txt", b"x"))
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0
    assert len(queued) == 2
    assert len(list(folder.iterdir())) == 2


def test_a_refused_upload_does_not_use_up_the_limit(limited_client):
    client, _, _ = limited_client
    for _ in range(3):
        assert send(client, ("empty.txt", b"")).status_code == 400
    assert send(client, ("a.txt", b"x")).status_code == 200


def test_one_upload_holds_a_limited_number_of_files(upload_client):
    client, queued, folder = upload_client
    refused = send(client, *[(f"{i}.txt", b"x") for i in range(11)])
    assert refused.status_code == 400
    assert "10 files" in refused.json()["detail"]
    assert queued == [] and list(folder.iterdir()) == []
    assert send(client, *[(f"{i}.txt", b"x") for i in range(10)]).status_code == 200


def test_one_bad_file_means_nothing_is_stored(upload_client):
    client, queued, folder = upload_client
    assert send(client, ("good.txt", b"x"), ("empty.txt", b"")).status_code == 400
    assert queued == [] and list(folder.iterdir()) == []
