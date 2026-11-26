"""
PDF report generation for batch query jobs.

The report is built with ReportLab: a branded cover band, summary cards, an index of the
questions, then every question with its answer. Answers are Markdown written by the language
model, so they are parsed into a small tree and rendered element by element (headings, lists,
tables, quotes, code). Only a fixed set of tags is ever turned into ReportLab markup, so text
that comes from users or documents can never inject tags such as ``<img>`` into the PDF.
"""

import re
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from markdown import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
)
from reportlab.platypus import Table as RLTable
from reportlab.platypus import TableStyle

# --------------------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------------------

BRAND = colors.HexColor("#009647")
BRAND_DARK = colors.HexColor("#0B3D2E")
BRAND_TINT = colors.HexColor("#E8F5EE")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
SURFACE = colors.HexColor("#F5F7F6")
BORDER = colors.HexColor("#D9E2DD")
DANGER = colors.HexColor("#B42318")
DANGER_TINT = colors.HexColor("#FEF3F2")

PRODUCT_NAME = "PDP Africa Policy Analyzer"
REPORT_TITLE = "Batch Analysis Report"

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN_X = 0.75 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X
BAND_HEIGHT = 2.0 * inch

# The index of questions is only useful (and only short enough to read) up to this many rows.
MAX_INDEX_ROWS = 50
INDEX_QUESTION_CHARS = 110


# --------------------------------------------------------------------------------------
# Text safety
# --------------------------------------------------------------------------------------

_CHAR_MAP = {
    " ": " ",
    "‑": "-",
    "‒": "-",
    "−": "-",
    "→": "->",
    "←": "<-",
    "⇒": "=>",
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    "≈": "~",
    "×": "x",
    "✓": "(yes)",
    "✔": "(yes)",
    "✅": "(yes)",
    "✗": "(no)",
    "✘": "(no)",
    "❌": "(no)",
    "●": "•",
    "▪": "•",
    "○": "•",
    "‣": "•",
    "​": "",
    "‌": "",
    "‍": "",
    "﻿": "",
}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _pdf_safe(text: Optional[str]) -> str:
    """
    Make text printable with the built-in PDF fonts.

    The standard Helvetica fonts cover Western European characters only. Anything else would
    be drawn as a black box, so common symbols are mapped to plain equivalents, accents are
    kept where possible, and the rest is dropped.
    """
    if not text:
        return ""
    text = _CONTROL_CHARS.sub("", str(text))
    out = []
    for ch in text:
        ch = _CHAR_MAP.get(ch, ch)
        for part in ch:
            try:
                part.encode("cp1252")
                out.append(part)
            except UnicodeEncodeError:
                folded = unicodedata.normalize("NFKD", part).encode("ascii", "ignore").decode("ascii")
                out.append(folded)
    return "".join(out)


def _esc(text: Optional[str]) -> str:
    """Escape text for use inside ReportLab Paragraph markup."""
    return escape(_pdf_safe(text))


# --------------------------------------------------------------------------------------
# Markdown -> small DOM -> flowables
# --------------------------------------------------------------------------------------

_VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "wbr"}
_SKIPPED_TAGS = {"script", "style", "iframe", "object", "embed", "svg", "math", "head", "title"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "table", "thead", "tbody", "tr", "pre", "blockquote", "hr",
}
_SAFE_LINK = re.compile(r"^(https?://|mailto:)", re.IGNORECASE)


class _Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag: str, attrs: Optional[dict] = None):
        self.tag = tag
        self.attrs = attrs or {}
        self.children: List[Any] = []


class _TreeBuilder(HTMLParser):
    """Parse the HTML produced by the Markdown library into a tolerant tree."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root")
        self._stack = [self.root]
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        node = _Node(tag, dict(attrs))
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag, attrs):
        if tag in _SKIPPED_TAGS or self._skip_depth:
            return
        self._stack[-1].children.append(_Node(tag, dict(attrs)))

    def handle_endtag(self, tag):
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth or tag in _VOID_TAGS:
            return
        # Close the nearest matching element; ignore stray closing tags.
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return

    def handle_data(self, data):
        if not self._skip_depth:
            self._stack[-1].children.append(data)


def _text_of(node: Any) -> str:
    """Plain text content of a node."""
    if isinstance(node, str):
        return node
    return "".join(_text_of(child) for child in node.children)


class _Styles:
    """All paragraph styles used by the report, built once."""

    def __init__(self):
        base = dict(fontName="Helvetica", textColor=INK, alignment=TA_LEFT)

        self.body = ParagraphStyle("Body", fontSize=10, leading=14.8, spaceAfter=6, **base)
        self.cell = ParagraphStyle("Cell", fontSize=8.8, leading=12, **base)
        self.cell_head = ParagraphStyle(
            "CellHead", fontSize=8.8, leading=12, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_LEFT,
        )
        self.h1 = ParagraphStyle(
            "AnsH1", parent=self.body, fontName="Helvetica-Bold", fontSize=13, leading=17,
            textColor=BRAND_DARK, spaceBefore=8, spaceAfter=4, keepWithNext=1,
        )
        self.h2 = ParagraphStyle(
            "AnsH2", parent=self.h1, fontSize=11.5, leading=15, spaceBefore=6,
        )
        self.h3 = ParagraphStyle(
            "AnsH3", parent=self.h1, fontSize=10.5, leading=14, spaceBefore=4, textColor=INK,
        )
        self.quote = ParagraphStyle(
            "Quote", parent=self.body, fontName="Helvetica-Oblique", textColor=MUTED, spaceAfter=3,
        )
        self.code = ParagraphStyle(
            "CodeBlock", parent=self.body, fontName="Courier", fontSize=8.6, leading=11.5,
            backColor=SURFACE, borderPadding=(5, 6, 5, 6), borderColor=BORDER, borderWidth=0.5,
            spaceBefore=4, spaceAfter=10,
        )
        self.section = ParagraphStyle(
            "Section", parent=self.body, fontName="Helvetica-Bold", fontSize=15, leading=19,
            textColor=BRAND_DARK, spaceBefore=18, spaceAfter=2, keepWithNext=1,
        )
        self.question = ParagraphStyle(
            "Question", parent=self.body, fontName="Helvetica-Bold", fontSize=11.5, leading=15.5,
            textColor=BRAND_DARK, spaceAfter=0,
        )
        self.badge = ParagraphStyle(
            "Badge", parent=self.body, fontName="Helvetica-Bold", fontSize=12.5, leading=15,
            textColor=colors.white, alignment=1, spaceAfter=0,
        )
        self.meta = ParagraphStyle(
            "Meta", parent=self.body, fontSize=8, leading=11, textColor=MUTED, spaceBefore=4, spaceAfter=6,
        )
        self.label = ParagraphStyle(
            "Label", parent=self.body, fontName="Helvetica-Bold", fontSize=7.5, leading=10,
            textColor=BRAND, spaceAfter=3, spaceBefore=2, keepWithNext=1,
        )
        self.kpi_value = ParagraphStyle(
            "KpiValue", parent=self.body, fontName="Helvetica-Bold", fontSize=22, leading=26,
            textColor=BRAND, spaceAfter=0,
        )
        self.kpi_label = ParagraphStyle(
            "KpiLabel", parent=self.body, fontSize=8, leading=10, textColor=MUTED, spaceAfter=0,
        )
        self.error = ParagraphStyle(
            "ErrorText", parent=self.body, fontSize=9.5, leading=13.5, textColor=DANGER, spaceAfter=0,
        )
        self.note = ParagraphStyle(
            "Note", parent=self.body, fontSize=8.4, leading=12, textColor=MUTED,
        )


_LIST_LINE = re.compile(r"^( *)([-*+]|\d+[.)])\s+")


def _normalize_list_indent(text: str) -> str:
    """
    Language models often indent nested list items by 2 spaces, but Python-Markdown only
    nests items indented by 4. When a text clearly uses 2-space nesting, double the indent
    of its list lines so the nesting survives. Fenced code blocks are left alone.
    """
    lines = text.split("\n")
    in_fence = False
    indents: List[int] = []
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            m = _LIST_LINE.match(line)
            if m:
                indents.append(len(m.group(1)))
    if not any(n % 4 == 2 for n in indents):
        return text
    in_fence = False
    out = []
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _LIST_LINE.match(line)
        out.append(" " * len(m.group(1)) + line if m else line)
    return "\n".join(out)


class _MarkdownRenderer:
    """Turn Markdown text into a list of ReportLab flowables."""

    def __init__(self, styles: _Styles):
        self.s = styles

    # -- public ------------------------------------------------------------------------

    def render(self, text: str, width: float) -> List[Flowable]:
        if not text or not text.strip():
            return []
        html = markdown(_normalize_list_indent(text), extensions=["tables", "nl2br", "fenced_code", "sane_lists"])
        builder = _TreeBuilder()
        builder.feed(html)
        builder.close()
        return self._blocks(builder.root.children, width, self.s.body)

    # -- inline ------------------------------------------------------------------------

    def _inline(self, children: List[Any]) -> str:
        parts: List[str] = []
        for child in children:
            if isinstance(child, str):
                parts.append(re.sub(r"\s+", " ", _esc(child)))
                continue
            tag = child.tag
            inner = self._inline(child.children)
            if tag in ("strong", "b"):
                parts.append(f"<b>{inner}</b>")
            elif tag in ("em", "i"):
                parts.append(f"<i>{inner}</i>")
            elif tag == "u":
                parts.append(f"<u>{inner}</u>")
            elif tag in ("del", "s", "strike"):
                parts.append(f"<strike>{inner}</strike>")
            elif tag == "sup":
                parts.append(f"<super>{inner}</super>")
            elif tag == "sub":
                parts.append(f"<sub>{inner}</sub>")
            elif tag == "code":
                code = re.sub(r"\s+", " ", _esc(_text_of(child)))
                parts.append(f'<font face="Courier" size="9" color="#0B3D2E">{code}</font>')
            elif tag == "br":
                parts.append("<br/>")
            elif tag == "a":
                href = (child.attrs.get("href") or "").strip()
                if _SAFE_LINK.match(href):
                    safe_href = escape(_pdf_safe(href), {'"': "&quot;"})
                    parts.append(f'<a href="{safe_href}" color="#007A39"><u>{inner}</u></a>')
                else:
                    parts.append(inner)
            elif tag in _BLOCK_TAGS:
                # A block element inside inline content (e.g. a table cell): keep its text.
                parts.append(inner + " ")
            else:
                # Unknown or unsafe tag (img, font, span, ...): keep only its text.
                parts.append(inner)
        return "".join(parts)

    # -- blocks ------------------------------------------------------------------------

    def _blocks(self, children: List[Any], width: float, style: ParagraphStyle) -> List[Flowable]:
        flowables: List[Flowable] = []
        run: List[Any] = []

        def flush():
            if not run:
                return
            markup = self._inline(run).strip()
            run.clear()
            if markup:
                flowables.append(Paragraph(markup, style))

        for child in children:
            if isinstance(child, str) or child.tag not in _BLOCK_TAGS:
                run.append(child)
                continue
            flush()
            flowables.extend(self._block(child, width, style))
        flush()
        return flowables

    def _block(self, node: _Node, width: float, style: ParagraphStyle) -> List[Flowable]:
        tag = node.tag
        if tag in ("p", "li"):
            return self._blocks(node.children, width, style)
        if tag in ("div", "section", "article", "thead", "tbody", "tr"):
            return self._blocks(node.children, width, style)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading_style = {"h1": self.s.h1, "h2": self.s.h1, "h3": self.s.h2}.get(tag, self.s.h3)
            markup = self._inline(node.children).strip()
            return [Paragraph(markup, heading_style)] if markup else []
        if tag in ("ul", "ol"):
            return self._list(node, width, style)
        if tag == "table":
            table = self._table(node, width)
            return [table, Spacer(1, 8)] if table is not None else []
        if tag == "pre":
            return self._code_block(node)
        if tag == "blockquote":
            return self._quote(node, width)
        if tag == "hr":
            return [HRFlowable(width="100%", thickness=0.6, color=BORDER, spaceBefore=6, spaceAfter=8)]
        return self._blocks(node.children, width, style)

    def _list(self, node: _Node, width: float, style: ParagraphStyle, depth: int = 0) -> List[Flowable]:
        items = []
        indent = 16
        for child in node.children:
            if isinstance(child, str) or child.tag != "li":
                continue
            content = self._blocks(child.children, width - indent, self._tight(style))
            if content:
                items.append(ListItem(content, leftIndent=indent))
        if not items:
            return []
        ordered = node.tag == "ol"
        kwargs = dict(
            bulletFontName="Helvetica-Bold" if ordered else "Helvetica",
            bulletFontSize=11,
            bulletColor=BRAND,
            leftIndent=indent,
            spaceAfter=4,
        )
        if ordered:
            try:
                kwargs["start"] = int(node.attrs.get("start", 1))
            except (TypeError, ValueError):
                kwargs["start"] = 1
            return [ListFlowable(items, bulletType="1", bulletFormat="%s.", **kwargs)]
        return [ListFlowable(items, bulletType="bullet", start="•" if depth == 0 else "-", **kwargs)]

    @staticmethod
    def _tight(style: ParagraphStyle) -> ParagraphStyle:
        return ParagraphStyle(f"{style.name}Tight", parent=style, spaceAfter=2)

    def _code_block(self, node: _Node) -> List[Flowable]:
        text = _pdf_safe(_text_of(node)).strip("\n")
        if not text.strip():
            return []
        lines = []
        for line in text.split("\n"):
            stripped = line.lstrip(" ")
            lead = "&nbsp;" * (len(line) - len(stripped))
            lines.append(lead + escape(stripped))
        return [Paragraph("<br/>".join(lines), self.s.code)]

    def _quote(self, node: _Node, width: float) -> List[Flowable]:
        inner = self._blocks(node.children, width - 16, self.s.quote)
        if not inner:
            return []
        table = RLTable([[inner]], colWidths=[width])
        table.setStyle(TableStyle([
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, BRAND),
            ("BACKGROUND", (0, 0), (-1, -1), BRAND_TINT),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return [table, Spacer(1, 8)]

    def _table(self, node: _Node, width: float) -> Optional[RLTable]:
        header: List[List[Any]] = []
        body: List[List[Any]] = []

        def collect(parent: _Node, in_head: bool):
            for child in parent.children:
                if isinstance(child, str):
                    continue
                if child.tag in ("thead", "tbody", "tfoot"):
                    collect(child, child.tag == "thead")
                elif child.tag == "tr":
                    cells = [c for c in child.children if not isinstance(c, str) and c.tag in ("th", "td")]
                    if not cells:
                        continue
                    is_header = in_head or all(c.tag == "th" for c in cells)
                    (header if is_header and not header and not body else body).append(cells)

        collect(node, False)
        rows = header + body
        if not rows:
            return None

        columns = max(len(r) for r in rows)
        # Every column is at least wide enough for its longest word (so words are not split),
        # and the remaining width is shared in proportion to how much text each column holds.
        longest_word = [0] * columns
        text_weight = [1.0] * columns
        for row in rows:
            for i, cell in enumerate(row):
                words = _text_of(cell).split()
                longest_word[i] = max(longest_word[i], max((len(w) for w in words), default=0))
                text_weight[i] = max(text_weight[i], min(len(" ".join(words)), 80))
        floors = [min(max(n, 3) * 5.2 + 16, width * 0.4) for n in longest_word]
        spare = max(width - sum(floors), 0)
        weight_total = sum(text_weight)
        col_widths = [floors[i] + spare * text_weight[i] / weight_total for i in range(columns)]
        scale = width / sum(col_widths)
        col_widths = [w * scale for w in col_widths]

        data = []
        for r, row in enumerate(rows):
            is_head = bool(header) and r < len(header)
            style = self.s.cell_head if is_head else self.s.cell
            cells = [Paragraph(self._inline(c.children).strip() or "&nbsp;", style) for c in row]
            cells += [Paragraph("&nbsp;", style)] * (columns - len(cells))
            data.append(cells)

        table = RLTable(data, colWidths=col_widths, repeatRows=len(header), hAlign="LEFT")
        commands = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ]
        first_body = len(header)
        if header:
            commands.append(("BACKGROUND", (0, 0), (-1, len(header) - 1), BRAND_DARK))
        if len(rows) > first_body:
            commands.append(("ROWBACKGROUNDS", (0, first_body), (-1, -1), [colors.white, SURFACE]))
        table.setStyle(TableStyle(commands))
        return table


# --------------------------------------------------------------------------------------
# Custom flowables and page chrome
# --------------------------------------------------------------------------------------

class _CompletionBar(Flowable):
    """A thin bar showing the share of questions that were answered."""

    def __init__(self, width: float, answered: int, failed: int, total: int, height: float = 7):
        super().__init__()
        self.width = width
        self.height = height
        self.answered = answered
        self.failed = failed
        self.total = max(total, 1)

    def wrap(self, avail_width, avail_height):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(BORDER)
        c.roundRect(0, 0, self.width, self.height, self.height / 2, stroke=0, fill=1)
        answered_w = self.width * self.answered / self.total
        failed_w = self.width * self.failed / self.total
        c.saveState()
        path = c.beginPath()
        path.roundRect(0, 0, self.width, self.height, self.height / 2)
        c.clipPath(path, stroke=0, fill=0)
        if answered_w > 0:
            c.setFillColor(BRAND)
            c.rect(0, 0, answered_w, self.height, stroke=0, fill=1)
        if failed_w > 0:
            c.setFillColor(DANGER)
            c.rect(answered_w, 0, failed_w, self.height, stroke=0, fill=1)
        c.restoreState()


class _ReportCanvas(canvas.Canvas):
    """Canvas that draws the cover band, running header and footer with 'Page x of y'."""

    def __init__(self, *args, **kwargs):
        self.report_meta = kwargs.pop("report_meta", {})
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[dict] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(total_pages)
            super().showPage()
        super().save()

    def _draw_chrome(self, total_pages: int):
        meta = self.report_meta
        if self._pageNumber == 1:
            self._draw_cover_band(meta)
        else:
            self._draw_running_header(meta)
        self._draw_footer(total_pages)

    def _draw_cover_band(self, meta: dict):
        top = PAGE_HEIGHT
        self.saveState()
        self.setFillColor(BRAND_DARK)
        self.rect(0, top - BAND_HEIGHT, PAGE_WIDTH, BAND_HEIGHT, stroke=0, fill=1)
        self.setFillColor(BRAND)
        self.rect(0, top - BAND_HEIGHT, PAGE_WIDTH, 5, stroke=0, fill=1)

        self.setFillColor(colors.HexColor("#8FD3AE"))
        self.setFont("Helvetica-Bold", 8.5)
        self.drawString(MARGIN_X, top - 0.62 * inch, PRODUCT_NAME.upper())

        self.setFillColor(colors.white)
        self.setFont("Helvetica-Bold", 28)
        self.drawString(MARGIN_X, top - 1.17 * inch, REPORT_TITLE)

        self.setFillColor(colors.HexColor("#CFE9DA"))
        self.setFont("Helvetica", 10.5)
        self.drawString(MARGIN_X, top - 1.5 * inch, meta.get("subtitle", ""))
        self.setFont("Helvetica", 9)
        self.drawRightString(PAGE_WIDTH - MARGIN_X, top - 0.62 * inch, meta.get("date", ""))
        self.restoreState()

    def _draw_running_header(self, meta: dict):
        y = PAGE_HEIGHT - 0.55 * inch
        self.saveState()
        self.setFillColor(BRAND_DARK)
        self.setFont("Helvetica-Bold", 8.5)
        self.drawString(MARGIN_X, y, REPORT_TITLE.upper())
        self.setFillColor(MUTED)
        self.setFont("Helvetica", 8.5)
        self.drawRightString(PAGE_WIDTH - MARGIN_X, y, meta.get("job_label", ""))
        self.setStrokeColor(BRAND)
        self.setLineWidth(1.2)
        self.line(MARGIN_X, y - 7, PAGE_WIDTH - MARGIN_X, y - 7)
        self.restoreState()

    def _draw_footer(self, total_pages: int):
        y = 0.5 * inch
        self.saveState()
        self.setStrokeColor(BORDER)
        self.setLineWidth(0.6)
        self.line(MARGIN_X, y + 12, PAGE_WIDTH - MARGIN_X, y + 12)
        self.setFillColor(MUTED)
        self.setFont("Helvetica", 8)
        self.drawString(MARGIN_X, y, PRODUCT_NAME)
        self.drawRightString(PAGE_WIDTH - MARGIN_X, y, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


# --------------------------------------------------------------------------------------
# Report generator
# --------------------------------------------------------------------------------------

def _format_date(value: Any) -> str:
    """Format a datetime for display, marking UTC when the value is timezone aware."""
    if not isinstance(value, datetime):
        value = datetime.now(timezone.utc)
    text = value.strftime("%d %B %Y, %H:%M")
    if value.tzinfo is not None and value.utcoffset() == timezone.utc.utcoffset(None):
        text += " UTC"
    return text


def _format_duration(seconds: float) -> str:
    seconds = float(seconds or 0)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest:02d} s"


def _topic_label(tab: Optional[str]) -> str:
    return tab.replace("-", " ").replace("_", " ").title() if tab else ""


class PDFGenerator:
    """Generate PDF reports from batch query results."""

    def __init__(self):
        self.styles = _Styles()
        self.renderer = _MarkdownRenderer(self.styles)

    # -- building blocks ----------------------------------------------------------------

    def _section(self, title: str) -> List[Flowable]:
        rule = HRFlowable(width="100%", thickness=0.8, color=BORDER, spaceBefore=2, spaceAfter=10)
        rule.keepWithNext = 1
        return [Paragraph(_esc(title), self.styles.section), rule]

    def _kpi_cards(self, metadata: Dict[str, Any]) -> RLTable:
        total = int(metadata.get("total_queries", 0) or 0)
        answered = int(metadata.get("completed_queries", 0) or 0)
        failed = int(metadata.get("failed_queries", 0) or 0)
        rate = float(metadata.get("success_rate", 0) or 0)

        def card(value: str, label: str, accent=BRAND):
            value_style = ParagraphStyle("kv", parent=self.styles.kpi_value, textColor=accent)
            return [Paragraph(value, value_style), Paragraph(label.upper(), self.styles.kpi_label)]

        gap = 8
        card_w = (CONTENT_WIDTH - 3 * gap) / 4
        cards = [
            card(str(total), "Questions", BRAND_DARK),
            card(str(answered), "Answered"),
            card(str(failed), "Failed", DANGER if failed else MUTED),
            card(f"{rate:.0f}%", "Success rate"),
        ]
        row: List[Any] = []
        widths: List[float] = []
        for i, content in enumerate(cards):
            row.append(content)
            widths.append(card_w)
            if i < len(cards) - 1:
                row.append("")
                widths.append(gap)

        table = RLTable([row], colWidths=widths)
        commands = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]
        for col in range(0, len(row), 2):
            commands.append(("BACKGROUND", (col, 0), (col, 0), SURFACE))
            commands.append(("LINEABOVE", (col, 0), (col, 0), 2.5, BRAND))
        table.setStyle(TableStyle(commands))
        return table

    def _details_table(self, metadata: Dict[str, Any], date_text: str) -> RLTable:
        label = ParagraphStyle("dl", parent=self.styles.cell, textColor=MUTED)
        value = ParagraphStyle("dv", parent=self.styles.cell, fontName="Helvetica-Bold")
        rows = [
            [Paragraph("Job ID", label), Paragraph(_esc(str(metadata.get("job_id", "N/A"))), value)],
            [Paragraph("Completed", label), Paragraph(_esc(date_text), value)],
            [Paragraph("Processing time", label), Paragraph(_format_duration(metadata.get("processing_time", 0)), value)],
        ]
        table = RLTable(rows, colWidths=[1.3 * inch, CONTENT_WIDTH - 1.3 * inch], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return table

    def _index_table(self, queries: List[str], results: List[Dict[str, Any]]) -> RLTable:
        head = ParagraphStyle("ih", parent=self.styles.cell_head)
        rows: List[List[Any]] = [[Paragraph("#", head), Paragraph("QUESTION", head), Paragraph("STATUS", head)]]
        for i, result in enumerate(results, 1):
            question = result.get("query") or (queries[i - 1] if i <= len(queries) else f"Query {i}")
            question = _pdf_safe(question).strip().replace("\n", " ")
            if len(question) > INDEX_QUESTION_CHARS:
                question = question[: INDEX_QUESTION_CHARS - 1].rstrip() + "…"
            ok = result.get("status") == "success" and bool(result.get("response"))
            status = (
                '<font color="#007A39"><b>Answered</b></font>' if ok
                else '<font color="#B42318"><b>Failed</b></font>'
            )
            rows.append([
                Paragraph(str(i), self.styles.cell),
                Paragraph(escape(question), self.styles.cell),
                Paragraph(status, self.styles.cell),
            ])
        table = RLTable(rows, colWidths=[0.45 * inch, CONTENT_WIDTH - 1.45 * inch, 1.0 * inch], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SURFACE]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        return table

    def _question_header(self, number: int, question: str) -> RLTable:
        badge_w = 0.5 * inch
        table = RLTable(
            [[Paragraph(str(number), self.styles.badge), Paragraph(escape(_pdf_safe(question)), self.styles.question)]],
            colWidths=[badge_w, CONTENT_WIDTH - badge_w],
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), BRAND),
            ("BACKGROUND", (1, 0), (1, 0), BRAND_TINT),
            ("VALIGN", (0, 0), (0, 0), "MIDDLE"),
            ("VALIGN", (1, 0), (1, 0), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (0, 0), 0),
            ("RIGHTPADDING", (0, 0), (0, 0), 0),
            ("LEFTPADDING", (1, 0), (1, 0), 12),
            ("RIGHTPADDING", (1, 0), (1, 0), 12),
        ]))
        return table

    def _question_block(self, number: int, queries: List[str], result: Dict[str, Any]) -> List[Flowable]:
        question = result.get("query") or (queries[number - 1] if number <= len(queries) else f"Query {number}")
        response = result.get("response") or ""
        error = result.get("error")
        ok = result.get("status") == "success" and bool(response.strip())

        meta_bits = []
        topic = _topic_label(result.get("tab"))
        if topic:
            meta_bits.append(f"<b>{_esc(topic.upper())}</b>")
        elapsed = result.get("processing_time")
        if ok and isinstance(elapsed, (int, float)):
            meta_bits.append(f"Answered in {_format_duration(elapsed)}")
        if not ok:
            meta_bits.append('<font color="#B42318"><b>NO ANSWER</b></font>')

        header: List[Flowable] = [self._question_header(number, question)]
        if meta_bits:
            header.append(Paragraph(" &nbsp;·&nbsp; ".join(meta_bits), self.styles.meta))
        else:
            header.append(Spacer(1, 6))

        if ok:
            answer = self.renderer.render(response, CONTENT_WIDTH)
            if not answer:
                answer = [Paragraph("No answer text was returned.", self.styles.note)]
            header.append(Paragraph("ANSWER", self.styles.label))
            # Keep the question with the start of its answer so a card never ends a page alone.
            first, rest = answer[0], answer[1:]
            return [CondPageBreak(1.6 * inch), KeepTogether(header + [first]), *rest, Spacer(1, 14)]

        message = _esc(str(error)) if error else "No response was available for this question."
        box = RLTable([[Paragraph(f"<b>Could not be answered.</b> {message}", self.styles.error)]], colWidths=[CONTENT_WIDTH])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), DANGER_TINT),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, DANGER),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return [CondPageBreak(1.6 * inch), KeepTogether(header + [box]), Spacer(1, 14)]

    # -- public API ---------------------------------------------------------------------

    def generate_batch_report(
        self,
        queries: List[str],
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> bytes:
        """
        Generate the PDF report for a batch job.

        Args:
            queries: Original queries, in order.
            results: One result per query (query, response, status, error, tab, processing_time).
            metadata: job_id, total_queries, completed_queries, failed_queries, success_rate,
                processing_time and completed_at.

        Returns:
            The PDF file as bytes.
        """
        buffer = BytesIO()
        completed_at = metadata.get("completed_at") or datetime.now(timezone.utc)
        date_text = _format_date(completed_at)
        job_id = str(metadata.get("job_id", "N/A"))
        total = int(metadata.get("total_queries", len(queries)) or 0)

        report_meta = {
            "date": date_text,
            "subtitle": f"{total} question{'s' if total != 1 else ''} analysed against the policy knowledge base",
            "job_label": f"Job {job_id[:8]}",
        }

        bottom = 0.85 * inch
        first_frame = Frame(
            MARGIN_X, bottom, CONTENT_WIDTH, PAGE_HEIGHT - BAND_HEIGHT - 0.3 * inch - bottom,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="first",
        )
        later_frame = Frame(
            MARGIN_X, bottom, CONTENT_WIDTH, PAGE_HEIGHT - 0.95 * inch - bottom,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="later",
        )
        doc = BaseDocTemplate(
            buffer,
            pagesize=letter,
            title=f"{REPORT_TITLE} - {job_id}",
            author=PRODUCT_NAME,
            subject="Answers to a batch of policy questions",
            creator=PRODUCT_NAME,
            pageTemplates=[
                PageTemplate(id="first", frames=[first_frame]),
                PageTemplate(id="later", frames=[later_frame]),
            ],
        )

        story: List[Flowable] = [NextPageTemplate("later")]

        # Summary
        story += self._section("Summary")
        story.append(self._kpi_cards(metadata))
        story.append(Spacer(1, 12))
        story.append(_CompletionBar(
            CONTENT_WIDTH,
            int(metadata.get("completed_queries", 0) or 0),
            int(metadata.get("failed_queries", 0) or 0),
            total,
        ))
        story.append(Spacer(1, 10))
        story.append(self._details_table(metadata, date_text))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "Answers are generated by an AI model from the documents in the knowledge base and, where enabled, "
            "web search. Check important points against the cited legislation or policy before relying on them.",
            self.styles.note,
        ))

        # Index
        if results and len(results) <= MAX_INDEX_ROWS:
            story += self._section("Questions at a glance")
            story.append(self._index_table(queries, results))

        # Answers
        story.append(CondPageBreak(2.5 * inch))
        story += self._section("Questions and answers")
        if not results:
            story.append(Paragraph("This job produced no results.", self.styles.body))
        for number, result in enumerate(results, 1):
            story += self._question_block(number, queries, result)

        doc.build(
            story,
            canvasmaker=lambda *args, **kwargs: _ReportCanvas(*args, report_meta=report_meta, **kwargs),
        )

        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    def generate_from_job_data(self, job_data: Dict[str, Any]) -> bytes:
        """
        Generate a PDF from a stored job document.

        Args:
            job_data: Complete job data from MongoDB.

        Returns:
            The PDF file as bytes.
        """
        queries = job_data.get("queries", [])
        results = job_data.get("results", [])

        total_queries = job_data.get("total_queries", len(queries))
        completed = job_data.get("completed_queries", 0)
        failed = job_data.get("failed_queries", 0)
        success_rate = (completed / total_queries * 100) if total_queries > 0 else 0

        metadata = {
            "job_id": job_data.get("job_id", "N/A"),
            "total_queries": total_queries,
            "completed_queries": completed,
            "failed_queries": failed,
            "success_rate": success_rate,
            "processing_time": job_data.get("processing_time", 0),
            "completed_at": job_data.get("completed_at") or datetime.now(timezone.utc),
        }
        return self.generate_batch_report(queries, results, metadata)


# Singleton instance
_pdf_generator: Optional[PDFGenerator] = None


def get_pdf_generator() -> PDFGenerator:
    """Get or create the singleton PDF generator instance."""
    global _pdf_generator
    if _pdf_generator is None:
        _pdf_generator = PDFGenerator()
    return _pdf_generator
