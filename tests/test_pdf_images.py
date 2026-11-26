"""Pictures inside a PDF are described by the model while the PDF is read. Claude refuses an image
whose declared type does not match its bytes, so the fake model here checks that the same way."""
import base64
from typing import Any, List

import fitz
import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.services.document_processing import DocumentProcessingV2

TEXT = "Data subjects may request erasure."
DESCRIPTION = "A roadmap chart with four milestones."
SIGNATURES = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff"}


def api_error(error_class, status, message):
    response = httpx.Response(status, request=httpx.Request("POST", "http://gateway/v1/chat/completions"))
    return error_class(message, response=response, body=None)


class StrictVisionModel(BaseChatModel):
    """Describes images, after checking them as strictly as Claude does."""
    error: Any = None  # raised for every image when set
    seen: List[str] = []

    @property
    def _llm_type(self) -> str:
        return "strict-vision-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        url = next(part["image_url"]["url"] for part in messages[0].content if part["type"] == "image_url")
        declared, data = url[len("data:"):].split(";base64,", 1)
        if not base64.b64decode(data).startswith(SIGNATURES[declared]):
            raise api_error(openai.BadRequestError, 400,
                            f"The image was specified using the {declared} media type, but it is not")
        self.seen.append(declared)
        if self.error:
            raise self.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=DESCRIPTION))])


def pdf_with_a_picture() -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), TEXT)
    picture = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 200), False)
    picture.set_rect(picture.irect, (200, 30, 30))
    page.insert_image(fitz.Rect(72, 150, 372, 350), pixmap=picture)
    return pdf.tobytes()


class FakeStore:
    def __init__(self):
        self.added = []

    def get_vector_store(self):
        return self

    def add_documents(self, chunks):
        self.added.extend(chunks)


def ingest(model, tmp_path, monkeypatch):
    monkeypatch.setattr(DocumentProcessingV2, "_detect_metadata", lambda self, text: {"title": "Roadmap"})
    store = FakeStore()
    path = tmp_path / "roadmap.pdf"
    path.write_bytes(pdf_with_a_picture())
    summary = DocumentProcessingV2(llm=model, vector_store=store).process_and_store(
        [{"link": str(path), "document_type": "application/pdf", "file_name": "roadmap.pdf"}],
        local_files=True,
    )
    return summary, " ".join(chunk.page_content for chunk in store.added)


def test_a_pdf_with_a_picture_is_indexed_with_its_description(tmp_path, monkeypatch):
    model = StrictVisionModel(seen=[])
    summary, indexed = ingest(model, tmp_path, monkeypatch)
    assert summary["failed"] == []
    assert model.seen == ["image/png"]
    assert TEXT in indexed and DESCRIPTION in indexed


def test_a_picture_the_model_refuses_does_not_lose_the_document(tmp_path, monkeypatch):
    model = StrictVisionModel(seen=[], error=api_error(openai.BadRequestError, 400, "image exceeds 5 MB"))
    summary, indexed = ingest(model, tmp_path, monkeypatch)
    assert summary["failed"] == []
    assert TEXT in indexed and DESCRIPTION not in indexed


def test_an_outage_still_fails_the_document(tmp_path, monkeypatch):
    """So that it is uploaded again, rather than silently indexed without its pictures."""
    model = StrictVisionModel(seen=[], error=api_error(openai.InternalServerError, 503, "unavailable"))
    summary, indexed = ingest(model, tmp_path, monkeypatch)
    assert summary["failed"] == ["roadmap.pdf"]
    assert indexed == ""


def test_refused_requests_do_not_open_the_circuit_breaker(monkeypatch):
    from uuid import uuid4

    from app.services import llm_factory

    breaker = llm_factory.CircuitBreaker(failure_threshold=5)
    monkeypatch.setattr(llm_factory, "get_circuit_breaker", lambda: breaker)
    callback = llm_factory.ProviderHealthCallback("openai", "default")

    def fail(error):
        run_id = uuid4()
        callback.on_chat_model_start({}, [], run_id=run_id)
        callback.on_llm_error(error, run_id=run_id)

    for _ in range(10):
        fail(api_error(openai.BadRequestError, 400, "image exceeds 5 MB"))
    assert breaker.get_state("openai") == breaker.CLOSED
    for _ in range(5):
        fail(api_error(openai.InternalServerError, 503, "unavailable"))
    assert breaker.get_state("openai") == breaker.OPEN


SECRET_LINK = ("https://bucket.s3.amazonaws.com/uploads/law.pdf"
               "?X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=topsecretsignature")


@pytest.mark.parametrize("failure", ["refused", "unreachable"])
def test_download_links_are_not_written_to_the_log(failure, monkeypatch, capfd):
    import requests

    def get(self, url, **kwargs):
        if failure == "unreachable":
            raise requests.ConnectionError(f"Max retries exceeded with url: {url.split('.com', 1)[1]}")
        response = requests.Response()
        response.status_code, response.url, response.reason = 403, url, "Forbidden"
        return response

    monkeypatch.setattr(requests.Session, "get", get)
    summary = DocumentProcessingV2(llm=None, vector_store=FakeStore()).process_and_store(
        [{"link": SECRET_LINK, "document_type": "application/pdf", "file_name": "law.pdf"}]
    )
    logged = "".join(capfd.readouterr())
    assert summary["failed"] == ["law.pdf"]
    assert "Could not download the file" in logged and "uploads/law.pdf" in logged
    assert "topsecretsignature" not in logged and "AKIAEXAMPLE" not in logged
