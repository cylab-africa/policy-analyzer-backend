"""Every accepted file type must really be readable. These tests run the real loaders on small
files; only the language model and the vector store are faked. (Markdown and Word uploads were
once accepted and then always failed, because other tests replaced the loaders.)"""
import io
import zipfile

import fitz
import pytest

from app.services import document_processing
from app.services.document_processing import DOCX_MIME_TYPE, DocumentProcessingV2
from app.services.file_management import FileStorageService

TEXT = "Data subjects may request erasure."


def make_docx(text: str) -> bytes:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", body)
    return buffer.getvalue()


def make_pdf(text: str) -> bytes:
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), text)
    return pdf.tobytes()


FILES = {
    "application/pdf": ("law.pdf", make_pdf(TEXT)),
    "text/plain": ("law.txt", TEXT.encode()),
    "text/markdown": ("law.md", f"# Rights\n\n{TEXT}".encode()),
    DOCX_MIME_TYPE: ("law.docx", make_docx(TEXT)),
}


class FakeStore:
    def __init__(self):
        self.added = []

    def get_vector_store(self):
        return self

    def add_documents(self, chunks):
        self.added.extend(chunks)


@pytest.fixture()
def store():
    return FakeStore()


@pytest.fixture()
def processor(store, monkeypatch):
    monkeypatch.setattr(DocumentProcessingV2, "_detect_metadata", lambda self, text: {"title": "Law"})
    return DocumentProcessingV2(llm=None, vector_store=store)


def ingest(processor, tmp_path, mime_type, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    return processor.process_and_store(
        [{"link": str(path), "document_type": mime_type, "file_name": name}], local_files=True
    )


def test_every_accepted_upload_type_has_a_loader_test():
    assert set(FileStorageService._parse_allowed_types(None)) == set(FILES)


@pytest.mark.parametrize("mime_type", list(FILES))
def test_each_accepted_type_is_read_and_indexed(processor, store, tmp_path, mime_type):
    name, content = FILES[mime_type]
    summary = ingest(processor, tmp_path, mime_type, name, content)
    assert summary["failed"] == [] and summary["processed"] == 1
    assert TEXT in " ".join(chunk.page_content for chunk in store.added)
    assert store.added[0].metadata["title"] == "Law"


def test_a_docx_that_unpacks_too_large_is_not_read(processor, store, tmp_path, monkeypatch):
    monkeypatch.setattr(document_processing, "MAX_DOCX_TEXT_BYTES", 10_000)
    bomb = make_docx("a" * 20_000)
    assert len(bomb) < 1_000  # small to upload, large once unpacked
    summary = ingest(processor, tmp_path, DOCX_MIME_TYPE, "bomb.docx", bomb)
    assert summary["failed"] == ["bomb.docx"]
    assert store.added == []


def test_a_file_that_is_not_really_a_docx_is_reported_as_failed(processor, store, tmp_path):
    summary = ingest(processor, tmp_path, DOCX_MIME_TYPE, "old.docx", b"\xd0\xcf\x11\xe0" + b"\0" * 600)
    assert summary["failed"] == ["old.docx"]
    assert store.added == []
