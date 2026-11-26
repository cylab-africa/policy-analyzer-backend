"""With USE_S3=false, which .env.example uses, files are kept on disk and no S3 settings are needed."""
import pytest

from app.core.config import settings
from app.services.file_management import FileStorageService


@pytest.fixture(autouse=True)
def no_s3_settings(monkeypatch):
    for name in ("S3_BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setattr(settings, name, None)


def test_local_storage_needs_no_s3_bucket(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "use_s3", False)
    monkeypatch.setattr(settings, "upload_directory", str(tmp_path))
    path = FileStorageService().save_file_locally("law.txt", b"text", "text/plain")
    with open(path, "rb") as saved:
        assert saved.read() == b"text"


def test_s3_storage_still_needs_a_bucket(monkeypatch):
    monkeypatch.setattr(settings, "use_s3", True)
    with pytest.raises(ValueError, match="S3_BUCKET_NAME"):
        FileStorageService()
