import pdfplumber
from pypdf import PdfReader

from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


def _table_to_text(table: list[list[str | None]]) -> str:
    """Render an extracted table as pipe-delimited rows so columns stay
    distinguishable instead of being smashed into one run-on line."""
    rows = [" | ".join(cell or "" for cell in row) for row in table]
    return "\n".join(rows)


def _extract_with_pdfplumber(path: str) -> str | None:
    """Try pdfplumber (table-aware). Returns None if it can't parse the file at all."""
    with pdfplumber.open(path) as pdf:
        if not pdf.pages:
            return None
        pages_text = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_blocks = [_table_to_text(table) for table in tables if table]
            pages_text.append("\n\n".join([text, *table_blocks]).strip())
        return "\n".join(pages_text)


def _extract_with_pypdf(path: str) -> str:
    """Plain text fallback, no table awareness."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


class PdfLoader(Loader):
    """Extracts text and tables from PDFs.

    Tries pdfplumber first because pypdf's ``extract_text`` reads
    left-to-right with no column awareness: a financial table's columns get
    smashed into one run-on line, making numbers unattributable to their
    row/column headers. pdfplumber's ``extract_tables`` detects table grid
    lines/whitespace and returns actual rows/cells, rendered here as
    pipe-delimited text appended after the page's prose so both survive into
    chunking.

    Falls back to pypdf (no table awareness) if pdfplumber can't parse the
    file at all (via pdfminer.six, it's stricter about malformed/minimal PDF
    structure than pypdf and can silently yield zero pages) -- so a
    non-table document that used to extract fine doesn't regress to empty
    content. This is a text-layer improvement either way, not OCR: pages
    that are scanned images with no text layer still yield no content.
    """

    format = "pdf"

    def load(self, path: str) -> Document:
        content = _extract_with_pdfplumber(path)
        if content is None:
            content = _extract_with_pypdf(path)
        return self._build_document(path, content)
