"""Checks on the batch report PDF.

These tests need no database, API keys or network. They build real PDFs from sample results
and read the text back, so they catch broken Markdown handling and unsafe content.
"""
import io
from datetime import datetime, timezone

import pytest

from app.services.pdf_generator import PDFGenerator, _normalize_list_indent, _pdf_safe

pypdf = pytest.importorskip("pypdf", reason="pypdf is only used to read the PDFs back")


def _metadata(total=1, completed=1, failed=0):
    return {
        "job_id": "550e8400-e29b-41d4-a716-446655440000",
        "total_queries": total,
        "completed_queries": completed,
        "failed_queries": failed,
        "success_rate": (completed / total * 100) if total else 0,
        "processing_time": 12.3,
        "completed_at": datetime(2026, 1, 15, 9, 30, tzinfo=timezone.utc),
    }


def _result(query="What is the law?", response="An answer.", status="success", **extra):
    return {"query": query, "response": response, "status": status, **extra}


def _build(results, metadata=None):
    queries = [r["query"] for r in results]
    meta = metadata or _metadata(len(results), sum(r["status"] == "success" for r in results))
    return PDFGenerator().generate_batch_report(queries, results, meta)


def _text(pdf_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() for page in reader.pages)


def test_builds_a_valid_pdf_with_summary_and_answers():
    pdf = _build([_result(response="Kenya has the **Data Protection Act, 2019**.", tab="data-protection")])
    assert pdf.startswith(b"%PDF-")
    text = _text(pdf)
    assert "Batch Analysis Report" in text
    assert "What is the law?" in text
    assert "Data Protection Act, 2019" in text
    assert "Page 1 of" in text


def test_empty_job_still_produces_a_pdf():
    pdf = PDFGenerator().generate_batch_report([], [], _metadata(total=0, completed=0))
    assert pdf.startswith(b"%PDF-")
    assert "no results" in _text(pdf)


def test_failed_question_shows_the_error_as_plain_text():
    pdf = _build([_result(response="", status="failed", error="Timeout <b>after</b> 60s")])
    text = _text(pdf)
    assert "Could not be answered" in text
    assert "<b>after</b>" in text  # shown literally, not interpreted as markup


def test_tables_lists_quotes_and_code_render():
    markdown_answer = (
        "## Heading\n\n"
        "| Country | Law |\n|---|---|\n| Kenya | Data Protection Act |\n\n"
        "- one\n- two\n\n1. first\n2. second\n\n> quoted text\n\n```\ncode line\n```\n"
    )
    text = _text(_build([_result(response=markdown_answer)]))
    for expected in ("Heading", "Kenya", "Data Protection Act", "one", "second", "quoted text", "code line"):
        assert expected in text


def test_raw_html_in_answers_cannot_inject_markup_or_load_resources():
    hostile = (
        'x <img src="file:///C:/Windows/win.ini"> <img src="http://169.254.169.254/latest"> '
        "<script>alert(1)</script> [bad](javascript:alert(1)) [ok](https://example.org)"
    )
    pdf = _build([_result(response=hostile)])  # must not raise or try to fetch anything
    assert pdf.startswith(b"%PDF-")
    text = _text(pdf)
    assert "alert(1)" not in text
    assert "win.ini" not in text
    assert "ok" in text


def test_links_are_only_kept_for_safe_schemes():
    pdf = _build([_result(response="[bad](javascript:alert(1)) [ok](https://example.org)")])
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    targets = [
        str(annot.get_object().get("/A", {}).get("/URI"))
        for page in reader.pages
        for annot in (page.get("/Annots") or [])
    ]
    assert targets == ["https://example.org"]


def test_characters_the_fonts_cannot_draw_are_mapped_or_dropped():
    assert _pdf_safe("a \u2192 b \u2265 c \u2713") == "a -> b >= c (yes)"
    assert _pdf_safe("C\u00f4te d'Ivoire") == "C\u00f4te d'Ivoire"
    assert "\U0001f680" not in _pdf_safe("launch \U0001f680")
    assert _pdf_safe(None) == ""


def test_long_reports_get_page_numbers_and_an_index():
    results = [_result(query=f"Question {i}", response="Some answer text. " * 80) for i in range(1, 13)]
    text = _text(_build(results))
    assert "Questions at a glance" in text
    assert "Question 12" in text
    assert "Page 1 of 1" not in text  # the report spans several pages


def test_two_space_nested_lists_keep_their_nesting():
    md = "- parent\n  - child\n- next"
    assert _normalize_list_indent(md) == "- parent\n    - child\n- next"
    already_four = "- parent\n    - child"
    assert _normalize_list_indent(already_four) == already_four


def test_generate_from_job_data_uses_stored_totals():
    job = {
        "job_id": "abc",
        "queries": ["Q1"],
        "results": [_result(query="Q1", response="Done.")],
        "total_queries": 1,
        "completed_queries": 1,
        "failed_queries": 0,
        "processing_time": 1.5,
    }
    pdf = PDFGenerator().generate_from_job_data(job)
    assert "100%" in _text(pdf)
