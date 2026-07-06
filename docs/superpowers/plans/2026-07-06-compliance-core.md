# Compliance Core (v2.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compliance-aware layer to rag-hybrid-search that detects legal document structure (Article/Section/Clause), chunks along clause boundaries, routes queries by structured/metadata/semantic/mixed intent, and returns citations mapped to exact clauses — all additive, with zero change to existing general-purpose RAG behavior.

**Architecture:** New `rag_hybrid_search/compliance/` package: a regex-based `clause_parser`, a `Chunker` subclass (`ClauseChunker`) that consumes its output, a `query_router` that classifies questions and picks a retrieval path, and a `citation_mapper` that builds structured `Citation` objects. `Chunk` gains an optional `legal_metadata` field; `SqliteChunkStore` gains indexed columns/query method for it. Everything is optional — undecorated documents and queries behave exactly as today.

**Tech Stack:** Python 3.11+, pydantic v2, pytest, sqlite3 (stdlib), chromadb (already a dependency), regex (stdlib `re`).

## Global Constraints

- Spec file: `docs/superpowers/specs/2026-07-06-compliance-core-design.md` — this plan implements it in full.
- `LegalMetadata` fields are all `Optional` — never make a compliance field required on `Chunk` or in storage schemas.
- Existing 142 tests (per last cleanup) must stay green after every task — no modification to existing test files except where a task explicitly says so.
- No LLM calls anywhere in this plan — clause parsing is regex-only in v1 (spec section "Future enhancement" is out of scope).
- `citation_id` uses `uuid7()` from `rag_hybrid_search/uuid7.py` (already used for `chunk_id`) — don't introduce a second UUID scheme.
- Date fields use Python `datetime.date`, not `str`.

---

### Task 1: Compliance domain models

**Files:**
- Create: `rag_hybrid_search/compliance/__init__.py`
- Create: `rag_hybrid_search/compliance/regulation_models.py`
- Modify: `rag_hybrid_search/models.py` (add `legal_metadata` field to `Chunk`)
- Test: `tests/compliance/__init__.py`
- Test: `tests/compliance/test_regulation_models.py`

**Interfaces:**
- Consumes: `rag_hybrid_search.uuid7.uuid7() -> str` (existing)
- Produces: `LegalMetadata`, `ClauseSpan`, `ClauseParseResult`, `Citation` pydantic models, importable as `from rag_hybrid_search.compliance.regulation_models import LegalMetadata, ClauseSpan, ClauseParseResult, Citation`. `Chunk.legal_metadata: Optional[LegalMetadata] = None`.

- [ ] **Step 1: Write the failing test for `LegalMetadata` defaults and `Citation` construction**

```python
# tests/compliance/test_regulation_models.py
from datetime import date

from rag_hybrid_search.compliance.regulation_models import (
    Citation,
    ClauseParseResult,
    ClauseSpan,
    LegalMetadata,
)


def test_legal_metadata_all_fields_optional():
    meta = LegalMetadata(document_id="doc-1", document_title="GDPR")
    assert meta.regulation is None
    assert meta.version is None
    assert meta.jurisdiction is None
    assert meta.article is None
    assert meta.section is None
    assert meta.clause is None
    assert meta.effective_date is None
    assert meta.document_type is None
    assert meta.page is None
    assert meta.document_id == "doc-1"
    assert meta.document_title == "GDPR"


def test_legal_metadata_typed_effective_date():
    meta = LegalMetadata(
        document_id="doc-1",
        document_title="GDPR",
        effective_date=date(2018, 5, 25),
    )
    assert meta.effective_date == date(2018, 5, 25)


def test_clause_parse_result_defaults_no_fallback():
    result = ClauseParseResult(
        clauses=[
            ClauseSpan(
                text="Personal data shall be processed lawfully.",
                metadata=LegalMetadata(document_id="doc-1", document_title="GDPR", article="5"),
            )
        ],
        confidence=0.92,
        parser="regex",
    )
    assert result.fallback_used is False
    assert result.parser == "regex"
    assert len(result.clauses) == 1


def test_citation_construction_and_display():
    citation = Citation(
        citation_id="0198c1a2-0000-7000-8000-000000000001",
        regulation="GDPR",
        version="2018",
        jurisdiction="EU",
        article="17",
        section="3",
        clause="17.3(b)",
        page=42,
        document_id="doc-1",
        document_title="GDPR Consolidated Text",
        document_type="regulation",
        effective_date=date(2018, 5, 25),
        chunk_id="chunk-1",
        confidence=0.9,
        display="GDPR Art. 17(3)(b), p.42",
    )
    assert citation.citation_id == "0198c1a2-0000-7000-8000-000000000001"
    assert citation.display == "GDPR Art. 17(3)(b), p.42"


def test_citation_missing_fields_are_optional():
    citation = Citation(
        citation_id="0198c1a2-0000-7000-8000-000000000002",
        document_id="doc-2",
        document_title="internal_notes.txt",
        chunk_id="chunk-2",
        confidence=0.5,
        display="internal_notes.txt, chunk 2",
    )
    assert citation.regulation is None
    assert citation.article is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/compliance/test_regulation_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.compliance'`

- [ ] **Step 3: Create the compliance package and models**

```python
# rag_hybrid_search/compliance/__init__.py
```

```python
# tests/compliance/__init__.py
```

```python
# rag_hybrid_search/compliance/regulation_models.py
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel

DocumentType = Literal["regulation", "policy", "contract", "standard", "guideline"]


class LegalMetadata(BaseModel):
    """Legal/structural metadata attached to a compliance document chunk.

    Every field except document_id/document_title is optional: a
    non-legal document ingested through the same pipeline gets an
    all-null LegalMetadata and behaves exactly as it does today.
    """

    document_id: str
    document_title: str
    regulation: Optional[str] = None
    version: Optional[str] = None
    jurisdiction: Optional[str] = None
    article: Optional[str] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    effective_date: Optional[date] = None
    document_type: Optional[DocumentType] = None
    page: Optional[int] = None


class ClauseSpan(BaseModel):
    """A single parsed clause: its text and the legal metadata locating it."""

    text: str
    metadata: LegalMetadata


class ClauseParseResult(BaseModel):
    """Output of clause_parser.parse(): all detected clauses plus a confidence score.

    confidence bands: 0.0-0.4 poor, 0.4-0.7 acceptable, 0.7-1.0 high confidence.
    """

    clauses: list[ClauseSpan]
    confidence: float
    parser: Literal["regex", "gemini"] = "regex"
    fallback_used: bool = False


class Citation(BaseModel):
    """A structured citation pointing at an exact clause (or, for non-legal
    documents, degrading gracefully to filename/chunk_id only)."""

    citation_id: str
    document_id: str
    document_title: str
    chunk_id: str
    confidence: float
    display: str
    regulation: Optional[str] = None
    version: Optional[str] = None
    jurisdiction: Optional[str] = None
    article: Optional[str] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    effective_date: Optional[date] = None
    document_type: Optional[DocumentType] = None
    page: Optional[int] = None
```

Now add the field to `Chunk` in `rag_hybrid_search/models.py`:

```python
# rag_hybrid_search/models.py — add import and field
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from rag_hybrid_search.compliance.regulation_models import LegalMetadata


class Document(BaseModel):
    document_id: str
    source_path: str
    content: str
    format: Literal["markdown", "html", "text", "pdf", "csv", "xlsx", "docx"]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    strategy_version: str
    heading: Optional[str] = None
    page: Optional[int] = None
    char_count: int
    legal_metadata: Optional[LegalMetadata] = None
```

(Leave the rest of `models.py` — `EmbeddingRecord`, `RetrievedChunk`, `IndexStatus`, `RetrievalTrace` — unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/compliance/test_regulation_models.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full existing suite to confirm no regression**

Run: `pytest -q`
Expected: all existing tests still PASS, plus the 4 new ones (142 + 4)

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/compliance/__init__.py rag_hybrid_search/compliance/regulation_models.py rag_hybrid_search/models.py tests/compliance/__init__.py tests/compliance/test_regulation_models.py
git commit -m "feat(compliance): add LegalMetadata, ClauseParseResult, Citation models"
```

---

### Task 2: Clause parser (regex-based)

**Files:**
- Create: `rag_hybrid_search/compliance/clause_parser.py`
- Test: `tests/compliance/test_clause_parser.py`
- Test fixtures: `tests/compliance/fixtures/gdpr_style.txt`, `tests/compliance/fixtures/hipaa_style.txt`, `tests/compliance/fixtures/unstructured.txt`

**Interfaces:**
- Consumes: `LegalMetadata`, `ClauseSpan`, `ClauseParseResult` from Task 1.
- Produces: `parse_clauses(text: str, document_id: str, document_title: str) -> ClauseParseResult`, importable as `from rag_hybrid_search.compliance.clause_parser import parse_clauses`.

- [ ] **Step 1: Write fixture files**

```text
# tests/compliance/fixtures/gdpr_style.txt
Article 5

1. Personal data shall be processed lawfully, fairly and in a transparent manner.

2. The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1.

Article 17

1. The data subject shall have the right to obtain from the controller the erasure of personal data.

3. Paragraph 1 shall not apply to the extent that processing is necessary:

(a) for exercising the right of freedom of expression and information;

(b) for compliance with a legal obligation.
```

```text
# tests/compliance/fixtures/hipaa_style.txt
Section 164.502

(a) Standard. A covered entity may not use or disclose protected health information.

(b) Implementation specification: Minimum necessary applies.

Section 164.508

(a) Standard: authorization for uses and disclosures.
```

```text
# tests/compliance/fixtures/unstructured.txt
This is a plain internal memo with no numbered structure at all.
It just talks about our office policy in prose form, paragraph after
paragraph, with no Article, Section, or Clause markers anywhere in it.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/compliance/test_clause_parser.py
from pathlib import Path

from rag_hybrid_search.compliance.clause_parser import parse_clauses

_FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_gdpr_style_detects_articles():
    result = parse_clauses(_read("gdpr_style.txt"), document_id="doc-1", document_title="GDPR")
    articles = {c.metadata.article for c in result.clauses}
    assert "5" in articles
    assert "17" in articles
    assert result.parser == "regex"
    assert result.fallback_used is False


def test_gdpr_style_high_confidence():
    result = parse_clauses(_read("gdpr_style.txt"), document_id="doc-1", document_title="GDPR")
    assert result.confidence >= 0.7


def test_hipaa_style_detects_sections():
    result = parse_clauses(_read("hipaa_style.txt"), document_id="doc-2", document_title="HIPAA")
    sections = {c.metadata.section for c in result.clauses}
    assert "164.502" in sections
    assert "164.508" in sections


def test_unstructured_document_low_confidence_single_clause():
    result = parse_clauses(_read("unstructured.txt"), document_id="doc-3", document_title="Memo")
    assert result.confidence < 0.4
    assert len(result.clauses) == 1
    assert result.clauses[0].metadata.article is None


def test_empty_text_returns_empty_result():
    result = parse_clauses("", document_id="doc-4", document_title="Empty")
    assert result.clauses == []
    assert result.confidence == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/compliance/test_clause_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.compliance.clause_parser'`

- [ ] **Step 4: Implement the parser**

```python
# rag_hybrid_search/compliance/clause_parser.py
import re

from rag_hybrid_search.compliance.regulation_models import (
    ClauseParseResult,
    ClauseSpan,
    LegalMetadata,
)

_ARTICLE_RE = re.compile(r"^(?:Article|ARTICLE|Art\.)\s+(\d+[A-Za-z]?)\s*$", re.MULTILINE)
_SECTION_RE = re.compile(r"^(?:Section|SECTION|Sec\.)\s+([\d.]+)\s*$", re.MULTILINE)
_CHAPTER_RE = re.compile(r"^(?:Chapter|CHAPTER)\s+([IVXLCDM]+|\d+)\s*$", re.MULTILINE)
_ANNEX_RE = re.compile(r"^(?:Annex|ANNEX|Appendix|APPENDIX)\s+([A-Za-z0-9]+)\s*$", re.MULTILINE)
_CLAUSE_RE = re.compile(r"^\(?(\d+(?:\.\d+)*(?:\([a-z]\))?)\)?[\s.:]", re.MULTILINE)

_HEADING_PATTERNS = [
    ("article", _ARTICLE_RE),
    ("section", _SECTION_RE),
    ("chapter", _CHAPTER_RE),
    ("annex", _ANNEX_RE),
]


def parse_clauses(text: str, document_id: str, document_title: str) -> ClauseParseResult:
    """Split text into clause spans using regex heading detection.

    Splits at top-level Article/Section/Chapter/Annex headings, then
    tags nested numbered sub-clauses (e.g. "1.", "5.2(a)") within each
    top-level span. Falls back to a single whole-document clause with
    confidence 0.0 if no heading is recognized anywhere.
    """
    if not text.strip():
        return ClauseParseResult(clauses=[], confidence=0.0)

    matches: list[tuple[int, str, str]] = []
    for label, pattern in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), label, m.group(1)))
    matches.sort(key=lambda m: m[0])

    if not matches:
        span = ClauseSpan(
            text=text.strip(),
            metadata=LegalMetadata(document_id=document_id, document_title=document_title),
        )
        return ClauseParseResult(clauses=[span], confidence=0.0)

    boundaries = [m[0] for m in matches] + [len(text)]
    clauses: list[ClauseSpan] = []
    current_article: str | None = None
    current_section: str | None = None

    for i, (start, label, value) in enumerate(matches):
        end = boundaries[i + 1]
        block = text[start:end].strip()
        if label == "article":
            current_article = value
        elif label == "section":
            current_section = value

        sub_clauses = list(_CLAUSE_RE.finditer(block))
        if not sub_clauses:
            clauses.append(
                ClauseSpan(
                    text=block,
                    metadata=LegalMetadata(
                        document_id=document_id,
                        document_title=document_title,
                        article=current_article,
                        section=current_section,
                    ),
                )
            )
            continue

        sub_boundaries = [sc.start() for sc in sub_clauses] + [len(block)]
        for j, sc in enumerate(sub_clauses):
            sub_text = block[sc.start() : sub_boundaries[j + 1]].strip()
            clause_number = sc.group(1)
            full_clause = (
                f"{current_article}.{clause_number}" if current_article else clause_number
            )
            clauses.append(
                ClauseSpan(
                    text=sub_text,
                    metadata=LegalMetadata(
                        document_id=document_id,
                        document_title=document_title,
                        article=current_article,
                        section=current_section,
                        clause=full_clause,
                    ),
                )
            )

    coverage = sum(len(c.text) for c in clauses) / max(len(text), 1)
    confidence = min(1.0, 0.5 + 0.5 * min(coverage, 1.0)) if clauses else 0.0

    return ClauseParseResult(clauses=clauses, confidence=confidence)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/compliance/test_clause_parser.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Run full suite**

Run: `pytest -q`
Expected: all still green

- [ ] **Step 7: Commit**

```bash
git add rag_hybrid_search/compliance/clause_parser.py tests/compliance/test_clause_parser.py tests/compliance/fixtures/
git commit -m "feat(compliance): add regex-based clause parser"
```

---

### Task 3: Clause-aware chunker

**Files:**
- Create: `rag_hybrid_search/compliance/clause_chunker.py`
- Test: `tests/compliance/test_clause_chunker.py`

**Interfaces:**
- Consumes: `Chunker` ABC (`rag_hybrid_search/ingestion/chunkers/base.py`), `Document`/`Chunk` (`rag_hybrid_search/models.py`), `uuid7()`, `parse_clauses()` from Task 2.
- Produces: `ClauseChunker(Chunker)` with `.version = "clause-v1"` and `.chunk(document: Document) -> list[Chunk]`, importable as `from rag_hybrid_search.compliance.clause_chunker import ClauseChunker`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/compliance/test_clause_chunker.py
from rag_hybrid_search.compliance.clause_chunker import ClauseChunker
from rag_hybrid_search.models import Document

_GDPR_TEXT = """Article 5

1. Personal data shall be processed lawfully, fairly and in a transparent manner.

2. The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1.

Article 17

1. The data subject shall have the right to obtain from the controller the erasure of personal data.
"""

_UNSTRUCTURED_TEXT = "Just a plain memo with no legal structure of any kind whatsoever."


def _doc(text: str, document_id: str = "doc-1") -> Document:
    return Document(document_id=document_id, source_path="/tmp/x.txt", content=text, format="text")


def test_chunks_have_legal_metadata_per_clause():
    chunker = ClauseChunker(document_title="GDPR")
    chunks = chunker.chunk(_doc(_GDPR_TEXT))
    assert len(chunks) >= 2
    articles = {c.legal_metadata.article for c in chunks}
    assert "5" in articles
    assert "17" in articles
    for c in chunks:
        assert c.strategy_version == "clause-v1"
        assert c.document_id == "doc-1"


def test_chunk_ids_are_unique():
    chunker = ClauseChunker(document_title="GDPR")
    chunks = chunker.chunk(_doc(_GDPR_TEXT))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_falls_back_to_single_chunk_for_unstructured_document():
    chunker = ClauseChunker(document_title="Memo")
    chunks = chunker.chunk(_doc(_UNSTRUCTURED_TEXT, document_id="doc-2"))
    assert len(chunks) == 1
    assert chunks[0].legal_metadata.article is None
    assert chunks[0].text == _UNSTRUCTURED_TEXT


def test_empty_document_returns_no_chunks():
    chunker = ClauseChunker(document_title="Empty")
    chunks = chunker.chunk(_doc("", document_id="doc-3"))
    assert chunks == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/compliance/test_clause_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.compliance.clause_chunker'`

- [ ] **Step 3: Implement the chunker**

```python
# rag_hybrid_search/compliance/clause_chunker.py
from rag_hybrid_search.compliance.clause_parser import parse_clauses
from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.models import Chunk, Document
from rag_hybrid_search.uuid7 import uuid7


class ClauseChunker(Chunker):
    """Chunks a document along Article/Section/Clause boundaries instead
    of fixed-size windows, attaching LegalMetadata to each resulting Chunk.

    Falls back to a single whole-document chunk (still tagged with
    LegalMetadata, all fields null except document_id/document_title)
    when clause_parser finds no recognizable structure — the document is
    never dropped or left unchunked.
    """

    version = "clause-v1"

    def __init__(self, document_title: str):
        self._document_title = document_title

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.content.strip():
            return []

        parse_result = parse_clauses(
            document.content,
            document_id=document.document_id,
            document_title=self._document_title,
        )

        chunks: list[Chunk] = []
        for index, clause_span in enumerate(parse_result.clauses):
            chunks.append(
                Chunk(
                    chunk_id=uuid7(),
                    document_id=document.document_id,
                    chunk_index=index,
                    text=clause_span.text,
                    strategy_version=self.version,
                    heading=clause_span.metadata.article or clause_span.metadata.section,
                    page=clause_span.metadata.page,
                    char_count=len(clause_span.text),
                    legal_metadata=clause_span.metadata,
                )
            )
        return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/compliance/test_clause_chunker.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: all still green

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/compliance/clause_chunker.py tests/compliance/test_clause_chunker.py
git commit -m "feat(compliance): add ClauseChunker producing clause-scoped chunks"
```

---

### Task 4: Persist legal metadata in chunk store

**Files:**
- Modify: `rag_hybrid_search/storage/chunk_store.py` (schema + read/write for legal metadata columns, new `get_by_legal_metadata`)
- Test: `tests/storage/test_chunk_store_legal_metadata.py`

**Interfaces:**
- Consumes: `Chunk.legal_metadata` (Task 1), `LegalMetadata` (Task 1).
- Produces: `SqliteChunkStore.put()`/`get()`/`all()`/`get_by_document()` round-trip `legal_metadata`. New method `SqliteChunkStore.get_by_legal_metadata(filters: dict[str, str]) -> list[Chunk]` for indexed metadata lookup (used by Task 5's query router).

- [ ] **Step 1: Write the failing tests**

```python
# tests/storage/test_chunk_store_legal_metadata.py
import tempfile

from rag_hybrid_search.compliance.regulation_models import LegalMetadata
from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore


def _chunk_with_metadata(chunk_id: str, **legal_kwargs) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        text="Personal data shall be processed lawfully.",
        strategy_version="clause-v1",
        char_count=42,
        legal_metadata=LegalMetadata(document_id="doc-1", document_title="GDPR", **legal_kwargs),
    )


def test_put_and_get_round_trips_legal_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        chunk = _chunk_with_metadata("chunk-1", regulation="GDPR", article="17", jurisdiction="EU")
        store.put(chunk, source_path="/tmp/gdpr.pdf")

        fetched = store.get("chunk-1")
        assert fetched.legal_metadata is not None
        assert fetched.legal_metadata.regulation == "GDPR"
        assert fetched.legal_metadata.article == "17"
        assert fetched.legal_metadata.jurisdiction == "EU"


def test_chunk_without_legal_metadata_round_trips_as_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        chunk = Chunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            chunk_index=0,
            text="plain text",
            strategy_version="fixed-v1",
            char_count=10,
        )
        store.put(chunk, source_path="/tmp/plain.txt")

        fetched = store.get("chunk-2")
        assert fetched.legal_metadata is None


def test_get_by_legal_metadata_filters_on_indexed_fields():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        store.put(_chunk_with_metadata("chunk-3", regulation="GDPR", article="5"), source_path="/tmp/a.pdf")
        store.put(_chunk_with_metadata("chunk-4", regulation="HIPAA", article=None), source_path="/tmp/b.pdf")

        results = store.get_by_legal_metadata({"regulation": "GDPR"})
        assert [c.chunk_id for c in results] == ["chunk-3"]

        results = store.get_by_legal_metadata({"regulation": "GDPR", "article": "5"})
        assert [c.chunk_id for c in results] == ["chunk-3"]

        results = store.get_by_legal_metadata({"regulation": "PCI-DSS"})
        assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/storage/test_chunk_store_legal_metadata.py -v`
Expected: FAIL — legal metadata columns don't exist / `get_by_legal_metadata` not defined

- [ ] **Step 3: Update the schema and store implementation**

```python
# rag_hybrid_search/storage/chunk_store.py
import sqlite3
from typing import Iterator, Optional

from rag_hybrid_search.compliance.regulation_models import LegalMetadata
from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.base import ChunkStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    heading TEXT,
    page INTEGER,
    char_count INTEGER NOT NULL,
    source_path TEXT,
    legal_regulation TEXT,
    legal_version TEXT,
    legal_jurisdiction TEXT,
    legal_article TEXT,
    legal_section TEXT,
    legal_clause TEXT,
    legal_effective_date TEXT,
    legal_document_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_legal_regulation ON chunks(legal_regulation);
CREATE INDEX IF NOT EXISTS idx_chunks_legal_jurisdiction ON chunks(legal_jurisdiction);
CREATE INDEX IF NOT EXISTS idx_chunks_legal_article ON chunks(legal_article);
CREATE INDEX IF NOT EXISTS idx_chunks_legal_document_type ON chunks(legal_document_type);
"""

_LEGAL_FILTER_COLUMNS = {
    "regulation": "legal_regulation",
    "version": "legal_version",
    "jurisdiction": "legal_jurisdiction",
    "article": "legal_article",
    "section": "legal_section",
    "clause": "legal_clause",
    "document_type": "legal_document_type",
}


class SqliteChunkStore(ChunkStore):
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def put(self, chunk: Chunk, source_path: Optional[str] = None) -> None:
        lm = chunk.legal_metadata
        self._conn.execute(
            """
            INSERT INTO chunks
                (chunk_id, document_id, chunk_index, text, strategy_version,
                 heading, page, char_count, source_path,
                 legal_regulation, legal_version, legal_jurisdiction, legal_article,
                 legal_section, legal_clause, legal_effective_date, legal_document_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_index=excluded.chunk_index,
                text=excluded.text,
                strategy_version=excluded.strategy_version,
                heading=excluded.heading,
                page=excluded.page,
                char_count=excluded.char_count,
                source_path=COALESCE(excluded.source_path, chunks.source_path),
                legal_regulation=excluded.legal_regulation,
                legal_version=excluded.legal_version,
                legal_jurisdiction=excluded.legal_jurisdiction,
                legal_article=excluded.legal_article,
                legal_section=excluded.legal_section,
                legal_clause=excluded.legal_clause,
                legal_effective_date=excluded.legal_effective_date,
                legal_document_type=excluded.legal_document_type
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.text,
                chunk.strategy_version,
                chunk.heading,
                chunk.page,
                chunk.char_count,
                source_path,
                lm.regulation if lm else None,
                lm.version if lm else None,
                lm.jurisdiction if lm else None,
                lm.article if lm else None,
                lm.section if lm else None,
                lm.clause if lm else None,
                lm.effective_date.isoformat() if lm and lm.effective_date else None,
                lm.document_type if lm else None,
            ),
        )
        self._conn.commit()

    def get(self, chunk_id: str) -> Optional[Chunk]:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_by_document(self, document_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_document_hash(self, source_path: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT document_id FROM chunks WHERE source_path = ? LIMIT 1",
            (source_path,),
        ).fetchone()
        return row["document_id"] if row else None

    def delete_by_document(self, document_id: str) -> None:
        self._conn.execute(
            "DELETE FROM chunks WHERE document_id = ?", (document_id,)
        )
        self._conn.commit()

    def all(self) -> Iterator[Chunk]:
        rows = self._conn.execute("SELECT * FROM chunks").fetchall()
        for row in rows:
            yield self._row_to_chunk(row)

    def get_by_legal_metadata(self, filters: dict[str, str]) -> list[Chunk]:
        """Indexed lookup by LegalMetadata fields (regulation/version/jurisdiction/
        article/section/clause/document_type). Unknown filter keys raise ValueError."""
        if not filters:
            return []
        clauses = []
        params = []
        for key, value in filters.items():
            column = _LEGAL_FILTER_COLUMNS.get(key)
            if column is None:
                raise ValueError(f"unknown legal metadata filter key: {key!r}")
            clauses.append(f"{column} = ?")
            params.append(value)
        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE {where} ORDER BY document_id, chunk_index", params
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_document_summaries(self) -> list[dict]:
        """Aggregate chunk counts per document, for corpus-wide stats endpoints."""
        rows = self._conn.execute(
            """
            SELECT document_id, source_path, COUNT(*) as chunk_count
            FROM chunks
            GROUP BY document_id
            ORDER BY document_id
            """
        ).fetchall()
        return [
            {
                "document_id": row["document_id"],
                "source_path": row["source_path"],
                "chunk_count": row["chunk_count"],
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        legal_metadata = None
        if row["legal_regulation"] is not None or row["legal_document_type"] is not None or row["legal_article"] is not None:
            legal_metadata = LegalMetadata(
                document_id=row["document_id"],
                document_title=row["document_id"],
                regulation=row["legal_regulation"],
                version=row["legal_version"],
                jurisdiction=row["legal_jurisdiction"],
                article=row["legal_article"],
                section=row["legal_section"],
                clause=row["legal_clause"],
                effective_date=row["legal_effective_date"] or None,
                document_type=row["legal_document_type"],
            )
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            strategy_version=row["strategy_version"],
            heading=row["heading"],
            page=row["page"],
            char_count=row["char_count"],
            legal_metadata=legal_metadata,
        )
```

Note: `document_title` isn't persisted as its own column — v1 reuses `document_id` as a stand-in when reconstructing `LegalMetadata` from a row, since `document_title` is display-only and not used for filtering. This is a known simplification; if `document_title` needs independent persistence later, add a `legal_document_title` column then.

`chroma_store.py` needs no change for v1 — metadata-filtered retrieval goes entirely through `SqliteChunkStore.get_by_legal_metadata` (Task 5 uses the sqlite path, not Chroma's). Confirmed by re-reading the file; recorded here so the reviewer knows it was considered, not missed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/storage/test_chunk_store_legal_metadata.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: all still green (existing `tests/storage/` tests for `SqliteChunkStore` must still pass unmodified — confirms the new columns are backward compatible)

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/storage/chunk_store.py tests/storage/test_chunk_store_legal_metadata.py
git commit -m "feat(storage): persist and query LegalMetadata on SqliteChunkStore"
```

---

### Task 5: Query router (structured / metadata / semantic / mixed)

**Files:**
- Create: `rag_hybrid_search/compliance/query_router.py`
- Test: `tests/compliance/test_query_router.py`
- Test: `tests/compliance/test_query_router_routing.py`

**Interfaces:**
- Consumes: `SqliteChunkStore.get_by_legal_metadata()` (Task 4), `HybridRetriever.retrieve()` (existing, `rag_hybrid_search/retrieval/retriever.py`), `RetrievedChunk`/`RetrievalTrace` (existing, `rag_hybrid_search/models.py`).
- Produces: `classify_query(question: str) -> QueryIntent`, `route_query(question: str, chunk_store: ChunkStore, retriever: HybridRetriever) -> tuple[list[RetrievedChunk], RetrievalTrace]`, importable as `from rag_hybrid_search.compliance.query_router import classify_query, route_query, QueryIntent`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/compliance/test_query_router.py
from rag_hybrid_search.compliance.query_router import QueryIntent, classify_query


def test_classifies_structured_article_reference():
    intent = classify_query("Show Article 17")
    assert intent.kind == "structured"
    assert intent.filters == {"article": "17"}


def test_classifies_structured_clause_reference():
    intent = classify_query("What does clause 5.2(a) say?")
    assert intent.kind == "mixed"
    assert intent.filters == {"clause": "5.2(a)"}


def test_classifies_metadata_only_scope():
    intent = classify_query("Show only HIPAA documents")
    assert intent.kind == "metadata"
    assert intent.filters == {"regulation": "HIPAA"}


def test_classifies_jurisdiction_metadata_scope():
    intent = classify_query("only EU regulations")
    assert intent.kind == "metadata"
    assert intent.filters == {"jurisdiction": "EU"}


def test_classifies_pure_semantic_query():
    intent = classify_query("What is the purpose of data minimization?")
    assert intent.kind == "semantic"
    assert intent.filters == {}


def test_classifies_mixed_query_with_intent_beyond_lookup():
    intent = classify_query("Explain Article 17 in plain terms")
    assert intent.kind == "mixed"
    assert intent.filters == {"article": "17"}
```

```python
# tests/compliance/test_query_router_routing.py
from unittest.mock import MagicMock

from rag_hybrid_search.compliance.query_router import route_query
from rag_hybrid_search.models import Chunk, RetrievalTrace, RetrievedChunk


def _retrieved(chunk_id: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id, document_id="doc-1", chunk_index=0, text="t",
        strategy_version="clause-v1", char_count=1,
    )
    return RetrievedChunk(chunk=chunk, rrf_score=1.0, final_rank=0)


def test_structured_query_uses_metadata_filter_only_no_retriever_call():
    chunk_store = MagicMock()
    chunk_store.get_by_legal_metadata.return_value = [
        Chunk(chunk_id="c1", document_id="doc-1", chunk_index=0, text="Article 17 text",
              strategy_version="clause-v1", char_count=10)
    ]
    retriever = MagicMock()

    results, trace = route_query("Show Article 17", chunk_store, retriever)

    chunk_store.get_by_legal_metadata.assert_called_once_with({"article": "17"})
    retriever.retrieve.assert_not_called()
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"


def test_semantic_query_delegates_to_retriever_unchanged():
    chunk_store = MagicMock()
    retriever = MagicMock()
    retriever.retrieve.return_value = ([_retrieved("c2")], RetrievalTrace())

    results, trace = route_query("What is data minimization?", chunk_store, retriever)

    chunk_store.get_by_legal_metadata.assert_not_called()
    retriever.retrieve.assert_called_once_with("What is data minimization?")
    assert results[0].chunk.chunk_id == "c2"


def test_mixed_query_filters_then_retrieves():
    chunk_store = MagicMock()
    chunk_store.get_by_legal_metadata.return_value = [
        Chunk(chunk_id="c3", document_id="doc-1", chunk_index=0, text="Article 17 text",
              strategy_version="clause-v1", char_count=10)
    ]
    retriever = MagicMock()
    retriever.retrieve.return_value = ([_retrieved("c3")], RetrievalTrace())

    results, trace = route_query("Explain Article 17", chunk_store, retriever)

    chunk_store.get_by_legal_metadata.assert_called_once_with({"article": "17"})
    retriever.retrieve.assert_called_once()
    assert results[0].chunk.chunk_id == "c3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/compliance/test_query_router.py tests/compliance/test_query_router_routing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.compliance.query_router'`

- [ ] **Step 3: Implement the router**

```python
# rag_hybrid_search/compliance/query_router.py
import re
from dataclasses import dataclass
from typing import Literal

from rag_hybrid_search.models import RetrievalTrace, RetrievedChunk
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.retrieval.retriever import HybridRetriever

QueryKind = Literal["structured", "metadata", "semantic", "mixed"]

_CLAUSE_REF_RE = re.compile(r"\bclause\s+([\d.]+(?:\([a-z]\))?)", re.IGNORECASE)
_ARTICLE_REF_RE = re.compile(r"\barticle\s+(\d+[A-Za-z]?)", re.IGNORECASE)
_SECTION_REF_RE = re.compile(r"\bsection\s+([\d.]+)", re.IGNORECASE)

_KNOWN_REGULATIONS = {"GDPR", "HIPAA", "SOC2", "PCI-DSS", "ISO27001", "CCPA"}
_KNOWN_JURISDICTIONS = {"EU", "US", "UK", "INDIA"}

# Words that signal the user wants more than a bare lookup — pushes a
# "structured" match into "mixed" instead.
_ELABORATION_WORDS = {"explain", "what", "why", "how", "summarize", "does", "say", "mean"}


@dataclass
class QueryIntent:
    kind: QueryKind
    filters: dict[str, str]


def classify_query(question: str) -> QueryIntent:
    """Classify a question into structured/metadata/semantic/mixed intent.

    - structured: a bare clause-level reference with no elaboration words.
    - mixed: a clause-level reference plus additional intent/elaboration.
    - metadata: a regulation/jurisdiction scope filter with no clause reference.
    - semantic: none of the above — full existing hybrid pipeline, unchanged.
    """
    filters: dict[str, str] = {}

    clause_match = _CLAUSE_REF_RE.search(question)
    article_match = _ARTICLE_REF_RE.search(question)
    section_match = _SECTION_REF_RE.search(question)

    if clause_match:
        filters["clause"] = clause_match.group(1)
    elif article_match:
        filters["article"] = article_match.group(1)
    elif section_match:
        filters["section"] = section_match.group(1)

    if filters:
        has_elaboration = any(
            re.search(rf"\b{word}\b", question, re.IGNORECASE) for word in _ELABORATION_WORDS
        )
        return QueryIntent(kind="mixed" if has_elaboration else "structured", filters=filters)

    upper_question = question.upper()
    for regulation in _KNOWN_REGULATIONS:
        if regulation in upper_question:
            return QueryIntent(kind="metadata", filters={"regulation": regulation})
    for jurisdiction in _KNOWN_JURISDICTIONS:
        if re.search(rf"\b{jurisdiction}\b", upper_question):
            return QueryIntent(kind="metadata", filters={"jurisdiction": jurisdiction})

    return QueryIntent(kind="semantic", filters={})


def route_query(
    question: str, chunk_store: ChunkStore, retriever: HybridRetriever
) -> tuple[list[RetrievedChunk], RetrievalTrace]:
    """Route a question to the retrieval path matching its classified intent.

    structured -> metadata filter only, no retriever call.
    metadata   -> metadata filter, then existing hybrid retrieval unchanged
                  (v1 simplification: metadata-only intent still runs the
                  full retriever; scoping the retriever's candidate set to
                  the filtered chunks is deferred, see spec open questions).
    semantic   -> existing hybrid retrieval, completely unchanged.
    mixed      -> metadata filter narrows candidates conceptually; v1 runs
                  the existing retriever and filters its output down to
                  matching chunk_ids, since HybridRetriever has no
                  candidate-subset parameter yet.
    """
    intent = classify_query(question)
    trace = RetrievalTrace()

    if intent.kind == "structured":
        matched = chunk_store.get_by_legal_metadata(intent.filters)
        results = [
            RetrievedChunk(chunk=chunk, rrf_score=1.0, final_rank=i)
            for i, chunk in enumerate(matched)
        ]
        return results, trace

    if intent.kind in ("metadata", "mixed"):
        matched_ids = {c.chunk_id for c in chunk_store.get_by_legal_metadata(intent.filters)}
        results, trace = retriever.retrieve(question)
        if matched_ids:
            results = [r for r in results if r.chunk.chunk_id in matched_ids]
        return results, trace

    return retriever.retrieve(question)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/compliance/test_query_router.py tests/compliance/test_query_router_routing.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: all still green

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/compliance/query_router.py tests/compliance/test_query_router.py tests/compliance/test_query_router_routing.py
git commit -m "feat(compliance): add query router for structured/metadata/semantic/mixed intent"
```

---

### Task 6: Citation mapper and RagAnswer wiring

**Files:**
- Create: `rag_hybrid_search/compliance/citation_mapper.py`
- Modify: `rag_pipeline/models.py` (add `structured_citations` field to `RagAnswer`)
- Modify: `rag_pipeline/rag_pipeline.py` (populate `structured_citations`)
- Test: `tests/compliance/test_citation_mapper.py`
- Test: `tests/rag_pipeline/test_rag_pipeline_structured_citations.py`

**Interfaces:**
- Consumes: `RetrievedChunk` (existing), `Citation` (Task 1), `uuid7()` (existing).
- Produces: `build_citations(retrieved_chunks: list[RetrievedChunk]) -> list[Citation]`, importable as `from rag_hybrid_search.compliance.citation_mapper import build_citations`. `RagAnswer.structured_citations: list[Citation]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/compliance/test_citation_mapper.py
from rag_hybrid_search.compliance.citation_mapper import build_citations
from rag_hybrid_search.compliance.regulation_models import LegalMetadata
from rag_hybrid_search.models import Chunk, RetrievedChunk


def _retrieved(chunk_id: str, legal_metadata=None, rerank_score=0.9) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id, document_id="doc-1", chunk_index=0,
        text="Personal data shall be processed lawfully.",
        strategy_version="clause-v1", char_count=42, page=42,
        legal_metadata=legal_metadata,
    )
    return RetrievedChunk(chunk=chunk, rrf_score=1.0, rerank_score=rerank_score, final_rank=0)


def test_build_citation_for_legal_chunk():
    legal_metadata = LegalMetadata(
        document_id="doc-1", document_title="GDPR Consolidated Text",
        regulation="GDPR", article="17", clause="17.3(b)", page=42,
    )
    citations = build_citations([_retrieved("c1", legal_metadata=legal_metadata)])
    assert len(citations) == 1
    citation = citations[0]
    assert citation.regulation == "GDPR"
    assert citation.article == "17"
    assert citation.display == "GDPR Art. 17(3)(b), p.42"
    assert citation.confidence == 0.9
    assert citation.chunk_id == "c1"


def test_build_citation_for_non_legal_chunk_degrades_gracefully():
    citations = build_citations([_retrieved("c2", legal_metadata=None)])
    assert len(citations) == 1
    citation = citations[0]
    assert citation.regulation is None
    assert "doc-1" in citation.display or "c2" in citation.display


def test_citation_ids_are_unique_across_calls():
    legal_metadata = LegalMetadata(document_id="doc-1", document_title="GDPR", article="5")
    citations_1 = build_citations([_retrieved("c3", legal_metadata=legal_metadata)])
    citations_2 = build_citations([_retrieved("c3", legal_metadata=legal_metadata)])
    assert citations_1[0].citation_id != citations_2[0].citation_id


def test_missing_rerank_score_defaults_confidence_to_zero():
    citations = build_citations([_retrieved("c4", rerank_score=None)])
    assert citations[0].confidence == 0.0
```

```python
# tests/rag_pipeline/test_rag_pipeline_structured_citations.py
from rag_pipeline.models import RagAnswer, ConfidenceScores, VerificationReport


def test_rag_answer_structured_citations_defaults_to_empty_list():
    answer = RagAnswer(
        answer="ok",
        citations=[],
        confidence=ConfidenceScores(retrieval=1.0, citations=1.0, coverage=1.0, overall=1.0),
        verification=VerificationReport(
            total_claims=0, verified_claims=0, failed_claims=0,
            hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
        ),
    )
    assert answer.structured_citations == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/compliance/test_citation_mapper.py tests/rag_pipeline/test_rag_pipeline_structured_citations.py -v`
Expected: FAIL with `ModuleNotFoundError` for citation_mapper, and `AttributeError`/validation error for `structured_citations`

- [ ] **Step 3: Implement citation_mapper.py**

```python
# rag_hybrid_search/compliance/citation_mapper.py
from rag_hybrid_search.compliance.regulation_models import Citation
from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.uuid7 import uuid7


def _render_display(retrieved: RetrievedChunk) -> str:
    lm = retrieved.chunk.legal_metadata
    if lm is None or (lm.regulation is None and lm.article is None):
        page_part = f", chunk {retrieved.chunk.chunk_index}"
        return f"{retrieved.chunk.document_id}{page_part}"

    parts = []
    if lm.regulation:
        parts.append(lm.regulation)
    if lm.article:
        clause_suffix = f"({lm.clause.split('.', 1)[-1]})" if lm.clause and "." in lm.clause else ""
        parts.append(f"Art. {lm.article}{clause_suffix}")
    elif lm.clause:
        parts.append(f"Clause {lm.clause}")
    display = " ".join(parts)
    page = lm.page or retrieved.chunk.page
    if page:
        display += f", p.{page}"
    return display


def build_citations(retrieved_chunks: list[RetrievedChunk]) -> list[Citation]:
    """Build structured Citation objects from post-rerank retrieved chunks.

    Non-legal chunks (legal_metadata is None) degrade gracefully to a
    document_id/chunk_index based display string instead of failing.
    """
    citations = []
    for retrieved in retrieved_chunks:
        lm = retrieved.chunk.legal_metadata
        citations.append(
            Citation(
                citation_id=uuid7(),
                document_id=retrieved.chunk.document_id,
                document_title=lm.document_title if lm else retrieved.chunk.document_id,
                chunk_id=retrieved.chunk.chunk_id,
                confidence=retrieved.rerank_score or 0.0,
                display=_render_display(retrieved),
                regulation=lm.regulation if lm else None,
                version=lm.version if lm else None,
                jurisdiction=lm.jurisdiction if lm else None,
                article=lm.article if lm else None,
                section=lm.section if lm else None,
                clause=lm.clause if lm else None,
                effective_date=lm.effective_date if lm else None,
                document_type=lm.document_type if lm else None,
                page=(lm.page if lm else None) or retrieved.chunk.page,
            )
        )
    return citations
```

Now wire `structured_citations` onto `RagAnswer`:

```python
# rag_pipeline/models.py — add import and field, rest of file unchanged
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from rag_hybrid_search.compliance.regulation_models import Citation


class Claim(BaseModel):
    text: str
    citation_ids: list[str]
    supporting_quote: str


class GenerationMetadata(BaseModel):
    provider: str
    model: str
    prompt_version: str
    generated_at: datetime


class RagAnswerDraft(BaseModel):
    answer: str
    claims: list[Claim]
    metadata: GenerationMetadata


class ClaimResult(BaseModel):
    claim: Claim
    doc_ids_valid: bool
    quote_match_score: float
    passed: bool


class VerificationReport(BaseModel):
    total_claims: int
    verified_claims: int
    failed_claims: int
    hallucinated_doc_ids: list[str]
    missing_quotes: list[str]
    claim_results: list[ClaimResult]


class ConfidenceScores(BaseModel):
    retrieval: float
    citations: float
    coverage: float
    overall: float


class RagAnswer(BaseModel):
    answer: Optional[str]
    citations: list[str]
    structured_citations: list[Citation] = []
    confidence: ConfidenceScores
    verification: VerificationReport
    error: Optional[str] = None


class PromptContext(BaseModel):
    text: str
    doc_id_map: dict[str, str]
```

Now populate it in `rag_pipeline.py` (only the `answer` method changes):

```python
# rag_pipeline/rag_pipeline.py — add import and one line in answer()
import json
from datetime import datetime, timezone

from pydantic import ValidationError

from rag_hybrid_search.compliance.citation_mapper import build_citations
from rag_pipeline.confidence_scorer import score_confidence
from rag_pipeline.context_builder import build_context
from rag_pipeline.generation_provider import GenerationProvider
from rag_pipeline.citation_verifier import verify_citations
from rag_pipeline.models import (
    Claim,
    ConfidenceScores,
    GenerationMetadata,
    RagAnswer,
    RagAnswerDraft,
    VerificationReport,
)
from rag_pipeline.prompt_builder import build_prompt

_EMPTY_VERIFICATION = VerificationReport(
    total_claims=0, verified_claims=0, failed_claims=0,
    hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
)
_ZERO_CONFIDENCE = ConfidenceScores(retrieval=0.0, citations=0.0, coverage=0.0, overall=0.0)


class RagPipeline:
    def __init__(self, retriever, generation_provider: GenerationProvider, prompt_version: str = "v1"):
        self._retriever = retriever
        self._generation_provider = generation_provider
        self._prompt_version = prompt_version

    def answer(self, question: str, max_chunks: int = 5, verify: bool = True) -> RagAnswer:
        retrieved_chunks, _trace = self._retriever.retrieve(question)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]

        context = build_context(retrieved_chunks)
        prompt = build_prompt(question, context, prompt_version=self._prompt_version)

        try:
            raw_output = self._generation_provider.generate(prompt)
        except Exception as e:
            return RagAnswer(
                answer=None, citations=[], confidence=_ZERO_CONFIDENCE,
                verification=_EMPTY_VERIFICATION, error=str(e),
            )

        draft, parse_error = self._parse_draft(raw_output)

        if verify:
            verification = verify_citations(draft, context)
            confidence = score_confidence(retrieved_chunks, verification, context)
        else:
            verification = _EMPTY_VERIFICATION
            confidence = ConfidenceScores(
                retrieval=score_confidence(retrieved_chunks, _EMPTY_VERIFICATION, context).retrieval,
                citations=0.0, coverage=0.0, overall=0.0,
            )

        citations = sorted({cid for c in draft.claims for cid in c.citation_ids})
        structured_citations = build_citations(retrieved_chunks)

        return RagAnswer(
            answer=draft.answer, citations=citations, structured_citations=structured_citations,
            confidence=confidence, verification=verification, error=parse_error,
        )

    def _parse_draft(self, raw_output: str) -> tuple[RagAnswerDraft, str | None]:
        metadata = GenerationMetadata(
            provider=type(self._generation_provider).__name__,
            model="unknown",
            prompt_version=self._prompt_version,
            generated_at=datetime.now(timezone.utc),
        )
        try:
            parsed = json.loads(raw_output)
            claims = [Claim(**c) for c in parsed.get("claims", [])]
            draft = RagAnswerDraft(answer=parsed["answer"], claims=claims, metadata=metadata)
            return draft, None
        except (json.JSONDecodeError, KeyError, ValidationError, TypeError) as e:
            degraded = RagAnswerDraft(answer=raw_output, claims=[], metadata=metadata)
            return degraded, f"failed to parse structured generation output: {e}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/compliance/test_citation_mapper.py tests/rag_pipeline/test_rag_pipeline_structured_citations.py -v`
Expected: PASS (4 + 1 tests)

- [ ] **Step 5: Run full suite**

Run: `pytest -q`
Expected: all still green — existing `tests/rag_pipeline/` tests must pass unmodified since `structured_citations` defaults to `[]`

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/compliance/citation_mapper.py rag_pipeline/models.py rag_pipeline/rag_pipeline.py tests/compliance/test_citation_mapper.py tests/rag_pipeline/test_rag_pipeline_structured_citations.py
git commit -m "feat(compliance): add citation mapper and wire structured_citations onto RagAnswer"
```

---

### Task 7: End-to-end integration test

**Files:**
- Test: `tests/compliance/test_end_to_end.py`

**Interfaces:**
- Consumes: `ClauseChunker` (Task 3), `SqliteChunkStore` (Task 4), `classify_query`/`route_query` (Task 5), `build_citations` (Task 6), `HybridRetriever`, `DenseRetriever`, `SparseRetriever`, `BM25Index`, `ChromaVectorStore`, `IndexManager`, `PassthroughReranker`, `FakeEmbeddingProvider`, `EmbeddingRecord` (all existing).
- Produces: nothing new — this is a pure verification task proving Tasks 1-6 compose correctly end-to-end.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/compliance/test_end_to_end.py
import tempfile
from datetime import datetime, timezone

from rag_hybrid_search.compliance.citation_mapper import build_citations
from rag_hybrid_search.compliance.clause_chunker import ClauseChunker
from rag_hybrid_search.compliance.query_router import route_query
from rag_hybrid_search.models import Document, EmbeddingRecord
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.passthrough_rerank import PassthroughReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager
from tests.fakes import FakeEmbeddingProvider

_GDPR_TEXT = """Article 5

1. Personal data shall be processed lawfully, fairly and in a transparent manner.

Article 17

1. The data subject shall have the right to obtain from the controller the erasure of personal data.

3. Paragraph 1 shall not apply to the extent that processing is necessary for compliance with a legal obligation.
"""


def _build_pipeline_components(tmp_dir: str):
    chunk_store = SqliteChunkStore(db_path=f"{tmp_dir}/chunks.db")
    vector_store = ChromaVectorStore(data_dir=f"{tmp_dir}/chroma")
    bm25_index = BM25Index(index_path=f"{tmp_dir}/bm25.pkl")
    index_manager = IndexManager(chunk_store, vector_store, bm25_index)
    embedding_provider = FakeEmbeddingProvider()

    document = Document(document_id="doc-gdpr", source_path="/tmp/gdpr.txt", content=_GDPR_TEXT, format="text")
    chunker = ClauseChunker(document_title="GDPR Consolidated Text")
    chunks = chunker.chunk(document)

    embeddings = embedding_provider.embed([c.text for c in chunks])
    records = [
        EmbeddingRecord(
            chunk_id=c.chunk_id, embedding=e, embedding_model=embedding_provider.model_name,
            embedding_dimension=embedding_provider.dimension, provider="FakeEmbeddingProvider",
            created_at=datetime.now(timezone.utc),
        )
        for c, e in zip(chunks, embeddings)
    ]
    for chunk in chunks:
        chunk_store.put(chunk, source_path=document.source_path)
    index_manager.index(chunks, records)

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(embedding_provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25_index),
        rerank_provider=PassthroughReranker(),
        dense_weight=0.7, sparse_weight=0.3, rrf_k=60,
        dense_k=10, sparse_k=10, rerank_top_n=5,
    )
    return chunk_store, retriever


def test_structured_query_returns_exact_article():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp)
        results, _trace = route_query("Show Article 17", chunk_store, retriever)
        assert len(results) >= 1
        assert all(r.chunk.legal_metadata.article == "17" for r in results)


def test_semantic_query_returns_results_via_hybrid_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp)
        results, _trace = route_query("What does the regulation say about data processing?", chunk_store, retriever)
        assert len(results) >= 1


def test_mixed_query_filters_to_matching_article():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp)
        results, _trace = route_query("Explain Article 5", chunk_store, retriever)
        assert len(results) >= 1
        assert all(r.chunk.legal_metadata.article == "5" for r in results)


def test_citations_built_from_structured_query_results():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp)
        results, _trace = route_query("Show Article 17", chunk_store, retriever)
        citations = build_citations(results)
        assert len(citations) >= 1
        assert citations[0].article == "17"
        assert "Art. 17" in citations[0].display
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/compliance/test_end_to_end.py -v`
Expected: FAIL if any Task 1-6 wiring has a mismatch (e.g. import error, signature mismatch) — this is the integration checkpoint. If Tasks 1-6 were implemented exactly as specified, this may pass on first run; that's fine — treat a first-run PASS as this step's success and continue to Step 4.

- [ ] **Step 3: Fix any integration mismatch found, then re-run to verify it passes**

Run: `pytest tests/compliance/test_end_to_end.py -v`
Expected: PASS (4 tests)

- [ ] **Step 4: Run full suite**

Run: `pytest -q`
Expected: all existing tests plus all new compliance tests green (142 existing + ~26 new)

- [ ] **Step 5: Commit**

```bash
git add tests/compliance/test_end_to_end.py
git commit -m "test(compliance): add end-to-end integration test for Compliance Core v2.0"
```

---

## Explicitly out of scope (per spec, not tasks in this plan)

- API-level wiring (`api/routes.py`, `api/dependencies.py`, `api/schemas.py`) to let callers choose `ClauseChunker` vs the default chunker at ingest time, and to expose `structured_citations`/query routing over HTTP. The spec names this as a future wiring point (`document_type="regulation"` flag) but the core engine (Tasks 1-6) is usable and testable standalone first. Follow-up plan once this core is validated.
- LLM-fallback clause parsing, regulation versioning/diffing, external feeds, business impact analysis, dashboard UI, evaluation harness — all deferred to v2.1+ per the spec.
