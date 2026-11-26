"""Route-level checks for batch job ownership and the folder-process allow-list.

The database is replaced by a small in-memory fake, so nothing external is contacted.
"""
import io

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import settings
from app.routes import batch_query, document_processing

ADMIN_KEY = "operator-only-admin-key"


class FakeCollection:
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(dict(doc))

    def find_one(self, query):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in query.items()):
                return dict(doc)
        return None

    def delete_one(self, query):
        self.docs = [d for d in self.docs if not all(d.get(k) == v for k, v in query.items())]

    def find(self, query):
        return FakeCursor([dict(d) for d in self.docs if all(d.get(k) == v for k, v in query.items())])

    def count_documents(self, query):
        return len(self.find(query).items)


class FakeCursor:
    def __init__(self, items):
        self.items = items

    def sort(self, *_):
        return self

    def skip(self, n):
        self.items = self.items[n:]
        return self

    def limit(self, n):
        self.items = self.items[:n]
        return self

    def __iter__(self):
        return iter(self.items)


@pytest.fixture()
def client(monkeypatch):
    collection = FakeCollection()
    monkeypatch.setattr(batch_query, "get_batch_collection", lambda: collection)
    # Do not run the real processor (it would call the language model)
    monkeypatch.setattr(batch_query, "run_batch_processor_sync", lambda **_: None)
    monkeypatch.setattr(settings, "batch_upload_rate_limit_per_hour", 0)
    monkeypatch.setattr(settings, "use_s3", False)

    app = FastAPI()
    app.include_router(batch_query.router, prefix="/api/batch")
    test_client = TestClient(app)
    test_client.collection = collection
    return test_client


def _upload(client, email="user@example.com"):
    buffer = io.BytesIO()
    pd.DataFrame({"q": ["What is the law?"]}).to_excel(buffer, index=False)
    return client.post(
        "/api/batch/upload",
        files={"file": ("q.xlsx", buffer.getvalue(), "application/vnd.ms-excel")},
        data={"user_email": email} if email else {},  # the website sends none when the box is blank
    )


def test_a_job_without_an_email_can_be_read_back(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)
    body = _upload(client, email=None).json()
    assert "user_email" not in client.collection.docs[0] or client.collection.docs[0]["user_email"] is None

    status = client.get(f"/api/batch/status/{body['job_id']}")
    assert status.status_code == 200
    assert status.json()["user_email"] is None

    listed = client.get("/api/batch/jobs", headers={"X-Admin-Key": ADMIN_KEY})
    assert listed.status_code == 200
    assert listed.json()["total"] == 1


def test_upload_returns_a_delete_token_and_stores_only_its_hash(client):
    response = _upload(client)
    assert response.status_code == 200
    body = response.json()
    token = body["delete_token"]
    stored = client.collection.docs[0]
    assert token and token not in str(stored)
    assert stored["delete_token_hash"]


def test_delete_requires_the_creators_token(client):
    body = _upload(client).json()
    job_id, token = body["job_id"], body["delete_token"]

    assert client.delete(f"/api/batch/{job_id}").status_code == 403
    assert client.delete(f"/api/batch/{job_id}", headers={"X-Job-Token": "wrong"}).status_code == 403
    assert client.collection.docs  # still there

    assert client.delete(f"/api/batch/{job_id}", headers={"X-Job-Token": token}).status_code == 200
    assert client.collection.docs == []


def test_one_users_token_cannot_delete_another_users_job(client):
    first = _upload(client, "a@example.com").json()
    second = _upload(client, "b@example.com").json()
    response = client.delete(f"/api/batch/{second['job_id']}", headers={"X-Job-Token": first["delete_token"]})
    assert response.status_code == 403


def test_legacy_job_without_a_token_needs_the_admin_key(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)
    client.collection.insert_one({"job_id": "legacy", "user_email": "x@example.com"})
    assert client.delete("/api/batch/legacy", headers={"X-Job-Token": "anything"}).status_code == 403
    assert client.delete("/api/batch/legacy", headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert client.delete("/api/batch/legacy", headers={"X-Admin-Key": settings.api_key}).status_code == 403
    assert client.delete("/api/batch/legacy", headers={"X-Admin-Key": ADMIN_KEY}).status_code == 200
    assert client.collection.docs == []


def test_admin_key_can_delete_a_job_whose_token_was_lost(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)
    job_id = _upload(client).json()["job_id"]
    assert client.delete(f"/api/batch/{job_id}", headers={"X-Admin-Key": ADMIN_KEY}).status_code == 200


@pytest.mark.parametrize("configured", [None, "", "same-as-api-key"])
def test_no_admin_access_without_a_usable_admin_key(client, monkeypatch, configured):
    if configured == "same-as-api-key":
        configured = settings.api_key
    monkeypatch.setattr(settings, "admin_api_key", configured)
    client.collection.insert_one({"job_id": "legacy"})
    for supplied in ("", settings.api_key):
        assert client.delete("/api/batch/legacy", headers={"X-Admin-Key": supplied}).status_code == 403
        assert client.get("/api/batch/jobs", headers={"X-Admin-Key": supplied}).status_code == 403


def test_status_never_exposes_the_token_hash(client):
    job_id = _upload(client).json()["job_id"]
    response = client.get(f"/api/batch/status/{job_id}")
    assert response.status_code == 200
    assert "delete_token" not in response.text


def test_listing_is_for_the_operator_only_and_hides_the_token_hash(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)
    _upload(client, "a@example.com")
    _upload(client, "b@example.com")

    # A self-declared filter is not proof of ownership, and the shared API key is not enough
    assert client.get("/api/batch/jobs?user_email=a@example.com").status_code == 403
    assert client.get("/api/batch/jobs", headers={"X-Admin-Key": settings.api_key}).status_code == 403

    admin = {"X-Admin-Key": ADMIN_KEY}
    assert client.get("/api/batch/jobs?limit=1000", headers=admin).status_code == 422
    listed = client.get("/api/batch/jobs", headers=admin)
    assert listed.status_code == 200
    assert listed.json()["total"] == 2
    assert "delete_token" not in listed.text
    assert client.get("/api/batch/jobs?user_email=a@example.com", headers=admin).json()["total"] == 1


@pytest.mark.parametrize("questions", [[], [None, None]])
def test_upload_without_questions_is_a_400_not_a_500(client, questions):
    buffer = io.BytesIO()
    pd.DataFrame({"q": questions}, dtype=object).to_excel(buffer, index=False)
    response = client.post(
        "/api/batch/upload",
        files={"file": ("q.xlsx", buffer.getvalue(), "application/vnd.ms-excel")},
    )
    assert response.status_code == 400, response.text


def test_upload_with_too_many_questions_is_a_400(client):
    buffer = io.BytesIO()
    pd.DataFrame({"q": [f"Question {i}" for i in range(101)]}).to_excel(buffer, index=False)
    response = client.post(
        "/api/batch/upload",
        files={"file": ("q.xlsx", buffer.getvalue(), "application/vnd.ms-excel")},
    )
    assert response.status_code == 400
    assert "Too many queries" in response.json()["detail"]


def test_invalid_uploads_do_not_use_up_the_hourly_quota(client, monkeypatch):
    monkeypatch.setattr(settings, "batch_upload_rate_limit_per_hour", 1)
    batch_query.upload_rate_limit.limiter.reset()
    try:
        for _ in range(3):
            wrong_type = client.post(
                "/api/batch/upload",
                files={"file": ("q.csv", b"What is the law?", "text/csv")},
            )
            assert wrong_type.status_code == 400
        assert _upload(client).status_code == 200  # the quota is still there
        assert _upload(client).status_code == 429
    finally:
        batch_query.upload_rate_limit.limiter.reset()


def test_batch_upload_is_rate_limited(client, monkeypatch):
    from fastapi import Depends

    from app.middleware.rate_limit import rate_limit

    guard = rate_limit("batch upload", lambda: 1, window=3600)
    app = FastAPI()

    @app.post("/upload", dependencies=[Depends(guard)])
    def upload():
        return {}

    limited = TestClient(app)
    assert limited.post("/upload").status_code == 200
    assert limited.post("/upload").status_code == 429


# --- folder-process ------------------------------------------------------------------------

@pytest.fixture()
def folder_client():
    app = FastAPI()
    app.include_router(document_processing.router, prefix="/api/documents")
    # The real storage service connects to S3 when USE_S3 is true (the default, and the case in
    # CI, which has no .env). These tests are about the path checks, so storage is not needed.
    app.dependency_overrides[document_processing.get_file_storage_service] = lambda: None
    return TestClient(app)


def test_folder_process_is_disabled_without_a_configured_root(folder_client, monkeypatch):
    monkeypatch.setattr(settings, "ingest_folder_root", None)
    response = folder_client.post("/api/documents/folder-process", json={"folder_path": "/etc"})
    assert response.status_code == 403


def test_background_ingest_skips_a_file_swapped_for_a_symlink(monkeypatch, tmp_path):
    import asyncio

    root = tmp_path / "root"
    root.mkdir()
    good = root / "good.pdf"
    good.write_text("x")
    swapped = root / "swapped.pdf"
    secret = tmp_path / "secret.env"
    secret.write_text("API_KEY=...")
    try:
        swapped.symlink_to(secret)  # replaced after the request was checked
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")

    stored = []

    class FakeProcessor:
        def __init__(self, **_):
            pass

        def process_and_store(self, documents, local_files):
            stored.extend(documents)

    class FakeCollection:
        def update_one(self, *_):
            pass

    monkeypatch.setattr(document_processing, "DocumentProcessingV2", FakeProcessor)
    files = [
        {"file_path": str(p), "content_type": "application/pdf", "client_id": None}
        for p in (good, swapped)
    ]
    asyncio.run(document_processing.process_documents_background_v2(
        files, "pid", None, FakeCollection(), None, None, "general", True, str(root)
    ))
    assert [d["link"] for d in stored] == [str(good.resolve())]


def test_downloads_go_to_a_unique_temp_file_not_the_working_directory(monkeypatch, tmp_path):
    import os
    import tempfile

    from app.services import document_processing as dp

    class FakeResponse:
        content = b"%PDF-1.4"

        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.Session.get", lambda self, *a, **k: FakeResponse())
    monkeypatch.chdir(tmp_path)
    download = dp.DocumentProcessingV2._download_file_remote
    first, cleanup = download(None, "https://example.org/a.pdf", "temp.pdf")
    second, _ = download(None, "https://example.org/b.pdf", "temp.pdf")
    try:
        assert cleanup
        assert first != second  # two documents at once no longer overwrite each other
        assert os.path.dirname(first) == tempfile.gettempdir()
        assert first.endswith(".pdf")
        assert not list(tmp_path.iterdir())  # nothing written to the working directory
    finally:
        for path in (first, second):
            os.remove(path)


def test_folder_process_rejects_paths_outside_the_root(folder_client, monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(settings, "ingest_folder_root", str(root))
    for bad in ("/", "..", str(tmp_path), "../" + tmp_path.name):
        response = folder_client.post("/api/documents/folder-process", json={"folder_path": bad})
        assert response.status_code == 400, bad


# --- upload inputs -----------------------------------------------------------------------------

def _upload_with(client, **data):
    buffer = io.BytesIO()
    pd.DataFrame({"q": ["What is the law?"]}).to_excel(buffer, index=False)
    return client.post(
        "/api/batch/upload",
        files={"file": ("q.xlsx", buffer.getvalue(), "application/vnd.ms-excel")},
        data=data,
    )


@pytest.mark.parametrize("client_id", [None, "", "   "])
def test_a_private_batch_needs_a_client_id(client, client_id):
    data = {"visibility": "private"}
    if client_id is not None:
        data["client_id"] = client_id
    response = _upload_with(client, **data)
    assert response.status_code == 400
    assert client.collection.docs == []


def test_a_private_batch_with_a_client_id_is_accepted(client):
    assert _upload_with(client, visibility="private", client_id="client-1").status_code == 200
    assert client.collection.docs[0]["client_id"] == "client-1"


@pytest.mark.parametrize("email", ["not-an-email", "a@b.com\nBcc: everyone@example.com", "@example.com"])
def test_an_invalid_email_is_refused(client, email):
    assert _upload_with(client, user_email=email).status_code == 400
    assert client.collection.docs == []


def test_a_valid_email_is_stored_normalised(client):
    assert _upload_with(client, user_email="  User@Example.COM ").status_code == 200
    assert client.collection.docs[0]["user_email"] == "User@example.com"
