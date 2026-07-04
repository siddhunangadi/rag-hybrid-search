# Ingestion & Hybrid Retrieval Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ingestion pipeline (multi-format loaders, configurable chunking, dedup, indexing) and hybrid retrieval core (dense + BM25 + weighted RRF + cross-encoder rerank) for `rag-hybrid-search`, exactly as specified in `docs/superpowers/specs/2026-07-05-ingestion-hybrid-retrieval-design.md`.

**Architecture:** Layered package — `models`/`config` (typed data + settings) at the base; `storage` (ChunkStore, VectorStore, BM25Index, IndexManager) on top of that; `providers` (Embedding/Generation/Rerank interfaces + NVIDIA NIM + Ollama implementations) alongside storage; `ingestion` (loaders → chunkers → dedup → pipeline) and `retrieval` (dense/sparse/fusion/rerank → HybridRetriever) both depend on storage + providers but never on each other's concrete classes — only interfaces.

**Tech Stack:** Python 3.11+, pydantic v2 + pydantic-settings, ChromaDB (embedded/persistent client), `rank_bm25`, `sentence-transformers` (CrossEncoder), `httpx` (NVIDIA NIM / Ollama HTTP calls), `beautifulsoup4` (HTML), `pypdf` (PDF), SQLite (stdlib `sqlite3`), pytest.

## Global Constraints

- Python >= 3.11.
- All cross-module dependencies go through the ABCs in `storage/base.py` and `providers/base.py` — no module outside `storage/` imports `chromadb` or `sqlite3` directly, no module outside `providers/` imports `httpx`/NVIDIA/Ollama specifics, no module outside `retrieval/rerank.py` imports `sentence_transformers`.
- Chunk IDs are UUIDv7 (time-ordered). Document IDs are `sha256(content)` hex digests.
- Embeddings are never stored on the `Chunk` model — only in `EmbeddingRecord`, associated by `chunk_id`.
- All embedding calls are batched (`embed(list[str]) -> list[list[float]]`), never per-text.
- Every new module gets a corresponding test module under `tests/` mirroring its path (e.g. `retrieval/fusion.py` → `tests/retrieval/test_fusion.py`).
- Every task ends green: `pytest` passes for the whole suite before committing.
- Commit after every task with a `feat:`/`test:`/`chore:` prefix matching the change.

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `rag_hybrid_search/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: an installable package `rag_hybrid_search` importable as `import rag_hybrid_search`, and a working `pytest` command.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "rag-hybrid-search"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "chromadb>=0.5",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=3.0",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "pypdf>=4.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.14",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["rag_hybrid_search*"]
```

- [ ] **Step 2: Create empty package markers**

`rag_hybrid_search/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
data/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: Install and verify**

Run:
```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
Expected: `pytest` runs with "no tests ran" (exit code 5 is acceptable here — it means the harness works but nothing exists yet).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml rag_hybrid_search/__init__.py tests/__init__.py .gitignore
git commit -m "chore: scaffold rag_hybrid_search package"
```

---

### Task 2: Core data models

**Files:**
- Create: `rag_hybrid_search/models.py`
- Create: `rag_hybrid_search/uuid7.py`
- Test: `tests/test_models.py`
- Test: `tests/test_uuid7.py`

**Interfaces:**
- Produces: `Document`, `Chunk`, `EmbeddingRecord`, `RetrievedChunk`, `IndexStatus`, `RetrievalTrace` (pydantic `BaseModel`/`enum.Enum`), and `uuid7() -> str` used by later tasks to mint `chunk_id`s.

- [ ] **Step 1: Write failing test for `uuid7`**

`tests/test_uuid7.py`:
```python
import re
import time

from rag_hybrid_search.uuid7 import uuid7


def test_uuid7_format():
    value = uuid7()
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        value,
    )


def test_uuid7_is_time_ordered():
    first = uuid7()
    time.sleep(0.002)
    second = uuid7()
    assert first < second


def test_uuid7_unique():
    values = {uuid7() for _ in range(1000)}
    assert len(values) == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_uuid7.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.uuid7'`

- [ ] **Step 3: Implement `uuid7.py`**

```python
import os
import time


def uuid7() -> str:
    """Generate a UUIDv7 string: 48-bit millisecond timestamp + random bits."""
    unix_ms = int(time.time() * 1000)
    ts_bytes = unix_ms.to_bytes(6, byteorder="big")
    rand = bytearray(os.urandom(10))

    # Version 7 in top nibble of byte 6, variant bits (10xx) in byte 8.
    rand[0] = (0x70 | (rand[0] & 0x0F))
    rand[2] = (0x80 | (rand[2] & 0x3F))

    raw = ts_bytes + bytes(rand)
    hex_str = raw.hex()
    return (
        f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-"
        f"{hex_str[16:20]}-{hex_str[20:32]}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_uuid7.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write failing test for models**

`tests/test_models.py`:
```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag_hybrid_search.models import (
    Chunk,
    Document,
    EmbeddingRecord,
    IndexStatus,
    RetrievalTrace,
    RetrievedChunk,
)


def test_document_roundtrip():
    doc = Document(
        document_id="a" * 64,
        source_path="/docs/readme.md",
        content="hello world",
        format="markdown",
    )
    assert doc.format == "markdown"


def test_document_rejects_bad_format():
    with pytest.raises(ValidationError):
        Document(
            document_id="a" * 64,
            source_path="/docs/readme.docx",
            content="hi",
            format="docx",
        )


def test_chunk_defaults():
    chunk = Chunk(
        chunk_id="018f7b1a-0000-7000-8000-000000000000",
        document_id="a" * 64,
        chunk_index=0,
        text="some chunk text",
        strategy_version="recursive-v1",
        heading=None,
        page=None,
        char_count=15,
    )
    assert chunk.chunk_index == 0
    assert chunk.heading is None


def test_embedding_record_dimension_matches_vector():
    record = EmbeddingRecord(
        chunk_id="018f7b1a-0000-7000-8000-000000000000",
        embedding=[0.1, 0.2, 0.3],
        embedding_model="nvidia/nv-embedqa-e5-v5",
        embedding_dimension=3,
        provider="nvidia",
        created_at=datetime.now(timezone.utc),
    )
    assert len(record.embedding) == record.embedding_dimension


def test_embedding_record_rejects_dimension_mismatch():
    with pytest.raises(ValidationError):
        EmbeddingRecord(
            chunk_id="018f7b1a-0000-7000-8000-000000000000",
            embedding=[0.1, 0.2, 0.3],
            embedding_model="nvidia/nv-embedqa-e5-v5",
            embedding_dimension=4,
            provider="nvidia",
            created_at=datetime.now(timezone.utc),
        )


def test_retrieved_chunk_final_rank():
    chunk = Chunk(
        chunk_id="018f7b1a-0000-7000-8000-000000000000",
        document_id="a" * 64,
        chunk_index=0,
        text="text",
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=4,
    )
    retrieved = RetrievedChunk(
        chunk=chunk,
        dense_score=0.9,
        bm25_score=None,
        rrf_score=0.5,
        rerank_score=0.8,
        final_rank=1,
    )
    assert retrieved.final_rank == 1


def test_index_status_values():
    assert IndexStatus.PENDING == "pending"
    assert IndexStatus.READY == "ready"


def test_retrieval_trace_total_latency():
    trace = RetrievalTrace(
        dense_latency_ms=1.0,
        bm25_latency_ms=2.0,
        fusion_latency_ms=0.5,
        rerank_latency_ms=3.5,
    )
    assert trace.total_latency_ms == 7.0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.models'`

- [ ] **Step 7: Implement `models.py`**

```python
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class Document(BaseModel):
    document_id: str
    source_path: str
    content: str
    format: Literal["markdown", "html", "text", "pdf"]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    strategy_version: str
    heading: Optional[str] = None
    page: Optional[int] = None
    char_count: int


class EmbeddingRecord(BaseModel):
    chunk_id: str
    embedding: list[float]
    embedding_model: str
    embedding_dimension: int
    provider: str
    created_at: datetime

    @model_validator(mode="after")
    def _check_dimension(self) -> "EmbeddingRecord":
        if len(self.embedding) != self.embedding_dimension:
            raise ValueError(
                f"embedding length {len(self.embedding)} != "
                f"embedding_dimension {self.embedding_dimension}"
            )
        return self


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float
    rerank_score: Optional[float] = None
    final_rank: int


class IndexStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class RetrievalTrace(BaseModel):
    dense_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return (
            self.dense_latency_ms
            + self.bm25_latency_ms
            + self.fusion_latency_ms
            + self.rerank_latency_ms
        )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_models.py tests/test_uuid7.py -v`
Expected: PASS (all tests)

- [ ] **Step 9: Commit**

```bash
git add rag_hybrid_search/models.py rag_hybrid_search/uuid7.py tests/test_models.py tests/test_uuid7.py
git commit -m "feat: add core data models and uuid7 generator"
```

---

### Task 3: Settings / configuration

**Files:**
- Create: `rag_hybrid_search/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` (pydantic-settings `BaseSettings`) and `get_settings() -> Settings` used by every later task that needs `chunk_size`, `provider`, `rrf_*`, `data_dir`, etc.

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
import pytest
from pydantic import ValidationError

from rag_hybrid_search.config import Settings


def test_defaults():
    settings = Settings()
    assert settings.provider == "nvidia"
    assert settings.chunking_strategy == "recursive"
    assert settings.rrf_dense_weight == 0.7
    assert settings.rrf_sparse_weight == 0.3


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        Settings(rrf_dense_weight=0.9, rrf_sparse_weight=0.3)


def test_weight_out_of_range():
    with pytest.raises(ValidationError):
        Settings(rrf_dense_weight=1.5, rrf_sparse_weight=-0.5)


def test_rerank_top_n_cannot_exceed_k_sum():
    with pytest.raises(ValidationError):
        Settings(dense_k=2, sparse_k=2, rerank_top_n=10)


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "1000")
    settings = Settings()
    assert settings.chunk_size == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.config'`

- [ ] **Step 3: Implement `config.py`**

```python
from typing import Literal, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_")

    provider: Literal["nvidia", "ollama"] = "nvidia"
    nvidia_api_key: Optional[str] = None

    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50

    dense_k: int = 10
    sparse_k: int = 10
    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    rrf_k: int = 60
    rerank_top_n: int = 5

    dedup_cosine_threshold: float = 0.95
    dedup_text_similarity_threshold: float = 0.9

    data_dir: str = "./data"

    @model_validator(mode="after")
    def _validate_weights_and_k(self) -> "Settings":
        if not (0.0 <= self.rrf_dense_weight <= 1.0):
            raise ValueError("rrf_dense_weight must be in [0, 1]")
        if not (0.0 <= self.rrf_sparse_weight <= 1.0):
            raise ValueError("rrf_sparse_weight must be in [0, 1]")
        if abs(self.rrf_dense_weight + self.rrf_sparse_weight - 1.0) > 1e-6:
            raise ValueError(
                "rrf_dense_weight + rrf_sparse_weight must sum to 1.0"
            )
        if self.rerank_top_n > self.dense_k + self.sparse_k:
            raise ValueError("rerank_top_n cannot exceed dense_k + sparse_k")
        return self


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/config.py tests/test_config.py
git commit -m "feat: add validated Settings configuration"
```

---

### Task 4: Storage interfaces (`VectorStore`, `ChunkStore` ABCs)

**Files:**
- Create: `rag_hybrid_search/storage/__init__.py`
- Create: `rag_hybrid_search/storage/base.py`
- Test: `tests/storage/__init__.py`
- Test: `tests/storage/test_base.py`

**Interfaces:**
- Consumes: `Chunk`, `EmbeddingRecord` from `rag_hybrid_search.models`.
- Produces: `VectorStore` ABC (`upsert(chunk_id, embedding_record)`, `query(embedding, k) -> list[tuple[str, float]]`, `delete(chunk_ids)`), `ChunkStore` ABC (`get`, `get_by_document`, `get_document_hash`, `put`, `delete_by_document`, `all`). Later tasks (`chunk_store.py`, `chroma_store.py`) implement these.

- [ ] **Step 1: Write failing test asserting ABCs cannot be instantiated directly**

`tests/storage/test_base.py`:
```python
import pytest

from rag_hybrid_search.storage.base import ChunkStore, VectorStore


def test_vector_store_is_abstract():
    with pytest.raises(TypeError):
        VectorStore()


def test_chunk_store_is_abstract():
    with pytest.raises(TypeError):
        ChunkStore()


def test_vector_store_subclass_must_implement_all_methods():
    class Incomplete(VectorStore):
        def upsert(self, chunk_id, embedding_record):
            pass

    with pytest.raises(TypeError):
        Incomplete()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.storage'`

- [ ] **Step 3: Create package markers**

`rag_hybrid_search/storage/__init__.py`:
```python
```

`tests/storage/__init__.py`:
```python
```

- [ ] **Step 4: Implement `storage/base.py`**

```python
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from rag_hybrid_search.models import Chunk, EmbeddingRecord


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunk_id: str, embedding_record: EmbeddingRecord) -> None:
        ...

    @abstractmethod
    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        ...


class ChunkStore(ABC):
    @abstractmethod
    def get(self, chunk_id: str) -> Optional[Chunk]:
        ...

    @abstractmethod
    def get_by_document(self, document_id: str) -> list[Chunk]:
        ...

    @abstractmethod
    def get_document_hash(self, source_path: str) -> Optional[str]:
        ...

    @abstractmethod
    def put(self, chunk: Chunk) -> None:
        ...

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        ...

    @abstractmethod
    def all(self) -> Iterator[Chunk]:
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/storage/test_base.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/storage/__init__.py rag_hybrid_search/storage/base.py tests/storage/__init__.py tests/storage/test_base.py
git commit -m "feat: add VectorStore and ChunkStore interfaces"
```

---

### Task 5: SQLite `ChunkStore` implementation

**Files:**
- Create: `rag_hybrid_search/storage/chunk_store.py`
- Test: `tests/storage/test_chunk_store.py`

**Interfaces:**
- Consumes: `ChunkStore` ABC from Task 4, `Chunk` from `models.py`.
- Produces: `SqliteChunkStore(db_path: str)` implementing `ChunkStore`, with `put(chunk, source_path=None)`. Later tasks (`IndexManager`, `IngestionPipeline`, `SparseRetriever`, `DenseRetriever`) construct and depend on this concrete class via the `ChunkStore` type.

- [ ] **Step 1: Write failing tests**

`tests/storage/test_chunk_store.py`:
```python
import pytest

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore


def make_chunk(chunk_id="c1", document_id="d1", index=0, text="hello"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=index,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


@pytest.fixture
def store(tmp_path):
    return SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))


def test_put_and_get(store):
    chunk = make_chunk()
    store.put(chunk)
    fetched = store.get("c1")
    assert fetched is not None
    assert fetched.text == "hello"


def test_get_missing_returns_none(store):
    assert store.get("missing") is None


def test_get_by_document(store):
    store.put(make_chunk(chunk_id="c1", document_id="d1", index=0))
    store.put(make_chunk(chunk_id="c2", document_id="d1", index=1))
    store.put(make_chunk(chunk_id="c3", document_id="d2", index=0))
    chunks = store.get_by_document("d1")
    assert {c.chunk_id for c in chunks} == {"c1", "c2"}


def test_document_hash_tracking(store):
    chunk = make_chunk(chunk_id="c1", document_id="deadbeef")
    store.put(chunk, source_path="/docs/a.md")
    assert store.get_document_hash("/docs/a.md") == "deadbeef"
    assert store.get_document_hash("/docs/missing.md") is None


def test_delete_by_document(store):
    store.put(make_chunk(chunk_id="c1", document_id="d1"), source_path="/docs/a.md")
    store.delete_by_document("d1")
    assert store.get("c1") is None
    assert store.get_document_hash("/docs/a.md") is None


def test_all_returns_every_chunk(store):
    store.put(make_chunk(chunk_id="c1", document_id="d1"))
    store.put(make_chunk(chunk_id="c2", document_id="d2"))
    ids = {c.chunk_id for c in store.all()}
    assert ids == {"c1", "c2"}


def test_reopening_store_persists_data(tmp_path):
    db_path = str(tmp_path / "chunks.db")
    store1 = SqliteChunkStore(db_path=db_path)
    store1.put(make_chunk(chunk_id="c1", document_id="d1"), source_path="/docs/a.md")

    store2 = SqliteChunkStore(db_path=db_path)
    assert store2.get("c1") is not None
    assert store2.get_document_hash("/docs/a.md") == "d1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_chunk_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.storage.chunk_store'`

- [ ] **Step 3: Implement `storage/chunk_store.py`**

```python
import sqlite3
from typing import Iterator, Optional

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
    source_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
"""


class SqliteChunkStore(ChunkStore):
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def put(self, chunk: Chunk, source_path: Optional[str] = None) -> None:
        self._conn.execute(
            """
            INSERT INTO chunks
                (chunk_id, document_id, chunk_index, text, strategy_version,
                 heading, page, char_count, source_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_index=excluded.chunk_index,
                text=excluded.text,
                strategy_version=excluded.strategy_version,
                heading=excluded.heading,
                page=excluded.page,
                char_count=excluded.char_count,
                source_path=COALESCE(excluded.source_path, chunks.source_path)
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

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            strategy_version=row["strategy_version"],
            heading=row["heading"],
            page=row["page"],
            char_count=row["char_count"],
        )
```

Note: `put` takes an optional `source_path` keyword beyond the base ABC's
minimal signature; this is allowed since the parameter has a default,
keeping `SqliteChunkStore` a valid `ChunkStore`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_chunk_store.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/storage/chunk_store.py tests/storage/test_chunk_store.py
git commit -m "feat: add SQLite-backed ChunkStore"
```

---

### Task 6: ChromaDB `VectorStore` implementation

**Files:**
- Create: `rag_hybrid_search/storage/chroma_store.py`
- Test: `tests/storage/test_chroma_store.py`

**Interfaces:**
- Consumes: `VectorStore` ABC from Task 4, `EmbeddingRecord` from `models.py`.
- Produces: `ChromaVectorStore(data_dir: str, collection_name: str = "chunks")` implementing `VectorStore`.

- [ ] **Step 1: Write failing tests**

`tests/storage/test_chroma_store.py`:
```python
from datetime import datetime, timezone

import pytest

from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore


def make_record(chunk_id, embedding):
    return EmbeddingRecord(
        chunk_id=chunk_id,
        embedding=embedding,
        embedding_model="test-model",
        embedding_dimension=len(embedding),
        provider="test",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(data_dir=str(tmp_path / "chroma"))


def test_upsert_and_query_returns_closest_first(store):
    store.upsert("a", make_record("a", [1.0, 0.0, 0.0]))
    store.upsert("b", make_record("b", [0.0, 1.0, 0.0]))
    store.upsert("c", make_record("c", [0.9, 0.1, 0.0]))

    results = store.query([1.0, 0.0, 0.0], k=2)

    assert results[0][0] == "a"
    assert len(results) == 2


def test_upsert_overwrites_existing_id(store):
    store.upsert("a", make_record("a", [1.0, 0.0, 0.0]))
    store.upsert("a", make_record("a", [0.0, 0.0, 1.0]))

    results = store.query([0.0, 0.0, 1.0], k=1)

    assert results[0][0] == "a"
    assert results[0][1] > 0.99


def test_delete_removes_vector(store):
    store.upsert("a", make_record("a", [1.0, 0.0, 0.0]))
    store.delete(["a"])

    results = store.query([1.0, 0.0, 0.0], k=5)

    assert "a" not in {chunk_id for chunk_id, _ in results}


def test_persists_across_reopen(tmp_path):
    data_dir = str(tmp_path / "chroma")
    store1 = ChromaVectorStore(data_dir=data_dir)
    store1.upsert("a", make_record("a", [1.0, 0.0, 0.0]))

    store2 = ChromaVectorStore(data_dir=data_dir)
    results = store2.query([1.0, 0.0, 0.0], k=1)

    assert results[0][0] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_chroma_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.storage.chroma_store'`

- [ ] **Step 3: Implement `storage/chroma_store.py`**

```python
import chromadb

from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, data_dir: str, collection_name: str = "chunks"):
        self._client = chromadb.PersistentClient(path=data_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunk_id: str, embedding_record: EmbeddingRecord) -> None:
        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding_record.embedding],
            metadatas=[
                {
                    "embedding_model": embedding_record.embedding_model,
                    "embedding_dimension": embedding_record.embedding_dimension,
                    "provider": embedding_record.provider,
                    "created_at": embedding_record.created_at.isoformat(),
                }
            ],
        )

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        result = self._collection.query(query_embeddings=[embedding], n_results=k)
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma cosine space returns distance = 1 - cosine_similarity.
        return [(chunk_id, 1.0 - dist) for chunk_id, dist in zip(ids, distances)]

    def delete(self, chunk_ids: list[str]) -> None:
        self._collection.delete(ids=chunk_ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_chroma_store.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/storage/chroma_store.py tests/storage/test_chroma_store.py
git commit -m "feat: add ChromaDB-backed VectorStore"
```

---

### Task 7: BM25 index

**Files:**
- Create: `rag_hybrid_search/storage/bm25_index.py`
- Test: `tests/storage/test_bm25_index.py`

**Interfaces:**
- Consumes: `Chunk` from `models.py`.
- Produces: `BM25Index(index_path: str)` with `build(chunks: list[Chunk]) -> None`, `search(query: str, k: int) -> list[tuple[str, float]]`, `save() -> None`, `load() -> bool` (returns `False` if no persisted index exists yet), and an internal `_chunk_ids: list[str]` attribute used by `IndexManager.verify_sync`.

- [ ] **Step 1: Write failing tests**

`tests/storage/test_bm25_index.py`:
```python
import pytest

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.bm25_index import BM25Index


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


def test_search_finds_exact_keyword_match(tmp_path):
    index = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index.build(
        [
            make_chunk("c1", "how to fix ERROR_CODE_0x834 in production"),
            make_chunk("c2", "general onboarding guide for new engineers"),
            make_chunk("c3", "deploying the service to staging"),
        ]
    )

    results = index.search("ERROR_CODE_0x834", k=2)

    assert results[0][0] == "c1"


def test_search_on_empty_index_returns_empty(tmp_path):
    index = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index.build([])

    assert index.search("anything", k=5) == []


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "bm25.pkl")
    index = BM25Index(index_path=path)
    index.build([make_chunk("c1", "unique keyword banana")])
    index.save()

    reloaded = BM25Index(index_path=path)
    loaded = reloaded.load()

    assert loaded is True
    results = reloaded.search("banana", k=1)
    assert results[0][0] == "c1"


def test_load_returns_false_when_no_file(tmp_path):
    index = BM25Index(index_path=str(tmp_path / "missing.pkl"))
    assert index.load() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_bm25_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.storage.bm25_index'`

- [ ] **Step 3: Implement `storage/bm25_index.py`**

```python
import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from rag_hybrid_search.models import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, index_path: str):
        self._index_path = Path(index_path)
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return ranked[:k]

    def save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)

    def load(self) -> bool:
        if not self._index_path.exists():
            return False
        with open(self._index_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_bm25_index.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/storage/bm25_index.py tests/storage/test_bm25_index.py
git commit -m "feat: add persistent BM25 index"
```

---

### Task 8: `IndexManager`

**Files:**
- Create: `rag_hybrid_search/storage/index_manager.py`
- Test: `tests/storage/test_index_manager.py`

**Interfaces:**
- Consumes: `ChunkStore`, `VectorStore` (Task 4/5/6), `BM25Index` (Task 7), `Chunk`/`EmbeddingRecord`/`IndexStatus` (Task 2).
- Produces: `IndexManager(chunk_store, vector_store, bm25_index)` with public attributes `chunk_store`, `vector_store`, `bm25_index`, and methods `index(chunks, embeddings) -> IndexStatus`, `remove_document(document_id) -> None`, `rebuild_bm25_index() -> None`, `rebuild_all() -> None`, `verify_sync() -> list[str]`. `IngestionPipeline` (Task 16) is the sole caller of `index`/`remove_document`.

- [ ] **Step 1: Write failing tests**

`tests/storage/test_index_manager.py`:
```python
from datetime import datetime, timezone

import pytest

from rag_hybrid_search.models import Chunk, EmbeddingRecord, IndexStatus
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager


def make_chunk(chunk_id, document_id="d1", text="hello world"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


def make_record(chunk_id, embedding=(1.0, 0.0, 0.0)):
    return EmbeddingRecord(
        chunk_id=chunk_id,
        embedding=list(embedding),
        embedding_model="test-model",
        embedding_dimension=len(embedding),
        provider="test",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def manager(tmp_path):
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    return IndexManager(chunk_store, vector_store, bm25)


def test_index_writes_to_all_stores(manager):
    chunk = make_chunk("c1")
    manager.chunk_store.put(chunk, source_path="/docs/a.md")

    status = manager.index([chunk], [make_record("c1")])

    assert status == IndexStatus.READY
    assert manager.vector_store.query([1.0, 0.0, 0.0], k=1)[0][0] == "c1"
    assert manager.bm25_index.search("hello", k=1)[0][0] == "c1"


def test_remove_document_clears_both_indexes(manager):
    chunk = make_chunk("c1", document_id="d1")
    manager.chunk_store.put(chunk, source_path="/docs/a.md")
    manager.index([chunk], [make_record("c1")])

    manager.remove_document("d1")

    assert manager.chunk_store.get("c1") is None
    assert manager.vector_store.query([1.0, 0.0, 0.0], k=1) == []
    assert manager.bm25_index.search("hello", k=1) == []


def test_verify_sync_reports_no_mismatches_when_healthy(manager):
    chunk = make_chunk("c1")
    manager.chunk_store.put(chunk, source_path="/docs/a.md")
    manager.index([chunk], [make_record("c1")])

    assert manager.verify_sync() == []


def test_verify_sync_detects_bm25_drift(manager):
    chunk = make_chunk("c1")
    manager.chunk_store.put(chunk, source_path="/docs/a.md")
    manager.index([chunk], [make_record("c1")])

    # Simulate drift: rebuild BM25 from an empty chunk list directly,
    # bypassing IndexManager, so ChunkStore and BM25Index disagree.
    manager.bm25_index.build([])

    assert manager.verify_sync() == ["c1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_index_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.storage.index_manager'`

- [ ] **Step 3: Implement `storage/index_manager.py`**

```python
from rag_hybrid_search.models import Chunk, EmbeddingRecord, IndexStatus
from rag_hybrid_search.storage.base import ChunkStore, VectorStore
from rag_hybrid_search.storage.bm25_index import BM25Index


class IndexManager:
    def __init__(
        self,
        chunk_store: ChunkStore,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.chunk_store = chunk_store
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def index(
        self, chunks: list[Chunk], embeddings: list[EmbeddingRecord]
    ) -> IndexStatus:
        try:
            for chunk, record in zip(chunks, embeddings):
                self.vector_store.upsert(chunk.chunk_id, record)
            self.rebuild_bm25_index()
        except Exception:
            return IndexStatus.FAILED
        return IndexStatus.READY

    def remove_document(self, document_id: str) -> None:
        chunks = self.chunk_store.get_by_document(document_id)
        chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_store.delete_by_document(document_id)
        if chunk_ids:
            self.vector_store.delete(chunk_ids)
        self.rebuild_bm25_index()

    def rebuild_bm25_index(self) -> None:
        all_chunks = list(self.chunk_store.all())
        self.bm25_index.build(all_chunks)
        self.bm25_index.save()

    def rebuild_all(self) -> None:
        self.rebuild_bm25_index()

    def verify_sync(self) -> list[str]:
        chunk_ids = {c.chunk_id for c in self.chunk_store.all()}
        bm25_ids = set(self.bm25_index._chunk_ids)
        return sorted(chunk_ids.symmetric_difference(bm25_ids))
```

`verify_sync` reaches into `BM25Index._chunk_ids` directly rather than
adding a public accessor solely for this one internal consistency check;
this is acceptable since `IndexManager` and `BM25Index` are both part of
the `storage` package and evolve together.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_index_manager.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/storage/index_manager.py tests/storage/test_index_manager.py
git commit -m "feat: add IndexManager to synchronize vector and BM25 indexes"
```

---

### Task 9: Provider interfaces

**Files:**
- Create: `rag_hybrid_search/providers/__init__.py`
- Create: `rag_hybrid_search/providers/base.py`
- Test: `tests/providers/__init__.py`
- Test: `tests/providers/test_base.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from `models.py` (for `RerankProvider`).
- Produces: `EmbeddingProvider` ABC (`embed(texts) -> list[list[float]]`, properties `model_name`, `dimension`), `GenerationProvider` ABC (`generate(prompt, **kwargs) -> str`), `RerankProvider` ABC (`rerank(query, candidates, top_n) -> list[RetrievedChunk]`).

- [ ] **Step 1: Write failing tests**

`tests/providers/test_base.py`:
```python
import pytest

from rag_hybrid_search.providers.base import (
    EmbeddingProvider,
    GenerationProvider,
    RerankProvider,
)


def test_embedding_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_generation_provider_is_abstract():
    with pytest.raises(TypeError):
        GenerationProvider()


def test_rerank_provider_is_abstract():
    with pytest.raises(TypeError):
        RerankProvider()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.providers'`

- [ ] **Step 3: Create package markers**

`rag_hybrid_search/providers/__init__.py`:
```python
```

`tests/providers/__init__.py`:
```python
```

- [ ] **Step 4: Implement `providers/base.py`**

```python
from abc import ABC, abstractmethod

from rag_hybrid_search.models import RetrievedChunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class GenerationProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...


class RerankProvider(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/providers/test_base.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/providers/__init__.py rag_hybrid_search/providers/base.py tests/providers/__init__.py tests/providers/test_base.py
git commit -m "feat: add EmbeddingProvider, GenerationProvider, RerankProvider interfaces"
```

---

### Task 10: Fake providers for testing (test-only, no network)

**Files:**
- Create: `tests/fakes.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`, `GenerationProvider` from Task 9.
- Produces: `FakeEmbeddingProvider` (deterministic hash-based embeddings, no network), `FakeGenerationProvider`, used by every ingestion/retrieval test from here on so the suite never makes real HTTP or model-download calls (except Task 19's cross-encoder, which is a real small local model, not a network provider).

- [ ] **Step 1: Implement `tests/fakes.py`**

```python
import hashlib

from rag_hybrid_search.providers.base import EmbeddingProvider, GenerationProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding stand-in for tests.

    Produces an 8-dim vector derived from character trigram hashes so that
    textually similar strings land close together in cosine space, which is
    enough to exercise dense retrieval and dedup logic without a real model.
    """

    _DIM = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._DIM
        normalized = text.lower()
        for i in range(len(normalized) - 2):
            trigram = normalized[i : i + 3]
            digest = hashlib.sha256(trigram.encode()).digest()
            bucket = digest[0] % self._DIM
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    @property
    def model_name(self) -> str:
        return "fake-embedding-v1"

    @property
    def dimension(self) -> int:
        return self._DIM


class FakeGenerationProvider(GenerationProvider):
    def __init__(self, fixed_response: str = "fake response"):
        self._fixed_response = fixed_response

    def generate(self, prompt: str, **kwargs) -> str:
        return self._fixed_response
```

- [ ] **Step 2: Verify it imports and behaves as expected**

Run: `python -c "from tests.fakes import FakeEmbeddingProvider; v = FakeEmbeddingProvider().embed(['hello world']); print(len(v[0]), sum(x*x for x in v[0]) ** 0.5)"`
Expected: prints `8 1.0` (an 8-dim unit vector), no errors.

- [ ] **Step 3: Commit**

```bash
git add tests/fakes.py
git commit -m "test: add fake embedding/generation providers for network-free tests"
```

---

### Task 11: NVIDIA NIM provider

**Files:**
- Create: `rag_hybrid_search/providers/nvidia.py`
- Test: `tests/providers/test_nvidia.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`, `GenerationProvider` from Task 9.
- Produces: `NvidiaProvider(api_key: str, embedding_model: str = "nvidia/nv-embedqa-e5-v5", generation_model: str = "meta/llama-3.1-70b-instruct")` implementing both interfaces via `httpx`, HTTP calls mocked in tests via `pytest-mock`.

- [ ] **Step 1: Write failing tests**

`tests/providers/test_nvidia.py`:
```python
import httpx
import pytest

from rag_hybrid_search.providers.nvidia import NvidiaProvider


@pytest.fixture
def provider():
    return NvidiaProvider(api_key="test-key")


def test_embed_calls_expected_endpoint_and_parses_response(provider, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        },
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/embeddings"),
    )
    mock_post = mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    result = provider.embed(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    called_url = mock_post.call_args[0][0]
    assert called_url == "https://integrate.api.nvidia.com/v1/embeddings"
    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["input"] == ["hello", "world"]


def test_generate_calls_chat_completions_and_returns_content(provider, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "an answer"}}]},
        request=httpx.Request(
            "POST", "https://integrate.api.nvidia.com/v1/chat/completions"
        ),
    )
    mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    result = provider.generate("What is RAG?")

    assert result == "an answer"


def test_model_name_and_dimension_reflect_configured_model(provider):
    assert provider.model_name == "nvidia/nv-embedqa-e5-v5"
    assert provider.dimension == 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_nvidia.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.providers.nvidia'`

- [ ] **Step 3: Implement `providers/nvidia.py`**

```python
import httpx

from rag_hybrid_search.providers.base import EmbeddingProvider, GenerationProvider

_BASE_URL = "https://integrate.api.nvidia.com/v1"

_MODEL_DIMENSIONS = {
    "nvidia/nv-embedqa-e5-v5": 1024,
    "nvidia/nv-embed-v2": 4096,
}


class NvidiaProvider(EmbeddingProvider, GenerationProvider):
    def __init__(
        self,
        api_key: str,
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        generation_model: str = "meta/llama-3.1-70b-instruct",
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._embedding_model = embedding_model
        self._generation_model = generation_model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            f"{_BASE_URL}/embeddings",
            json={
                "input": texts,
                "model": self._embedding_model,
                "input_type": "passage",
            },
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.post(
            f"{_BASE_URL}/chat/completions",
            json={
                "model": self._generation_model,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def model_name(self) -> str:
        return self._embedding_model

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS.get(self._embedding_model, 1024)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_nvidia.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/providers/nvidia.py tests/providers/test_nvidia.py
git commit -m "feat: add NVIDIA NIM embedding/generation provider"
```

---

### Task 12: Ollama provider

**Files:**
- Create: `rag_hybrid_search/providers/ollama.py`
- Test: `tests/providers/test_ollama.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`, `GenerationProvider` from Task 9.
- Produces: `OllamaProvider(base_url: str = "http://localhost:11434", embedding_model: str = "nomic-embed-text", generation_model: str = "llama3.1")` implementing both interfaces, HTTP mocked in tests.

- [ ] **Step 1: Write failing tests**

`tests/providers/test_ollama.py`:
```python
import httpx
import pytest

from rag_hybrid_search.providers.ollama import OllamaProvider


@pytest.fixture
def provider():
    return OllamaProvider()


def test_embed_calls_per_text_and_collects_in_order(provider, mocker):
    def fake_post(url, json):
        text = json["prompt"]
        value = 0.1 if text == "hello" else 0.2
        return httpx.Response(
            status_code=200,
            json={"embedding": [value, value, value]},
            request=httpx.Request("POST", url),
        )

    mocker.patch.object(httpx.Client, "post", side_effect=fake_post)

    result = provider.embed(["hello", "world"])

    assert result == [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]


def test_generate_calls_generate_endpoint(provider, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={"response": "an answer"},
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )
    mock_post = mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    result = provider.generate("What is RAG?")

    assert result == "an answer"
    assert mock_post.call_args[0][0] == "http://localhost:11434/api/generate"


def test_model_name_and_dimension(provider):
    assert provider.model_name == "nomic-embed-text"
    assert provider.dimension == 768
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_ollama.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.providers.ollama'`

- [ ] **Step 3: Implement `providers/ollama.py`**

Ollama's embedding endpoint (`/api/embeddings`) takes one prompt per call
(no native batch endpoint as of this writing), so `embed()` loops
internally — this is an implementation detail isolated inside the provider;
every caller still sees a single batched `embed(list[str])` call, matching
the `EmbeddingProvider` interface.

```python
import httpx

from rag_hybrid_search.providers.base import EmbeddingProvider, GenerationProvider

_MODEL_DIMENSIONS = {
    "nomic-embed-text": 768,
}


class OllamaProvider(EmbeddingProvider, GenerationProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        generation_model: str = "llama3.1",
        timeout: float = 30.0,
    ):
        self._base_url = base_url
        self._embedding_model = embedding_model
        self._generation_model = generation_model
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._embedding_model, "prompt": text},
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._generation_model,
                "prompt": prompt,
                "stream": False,
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()["response"]

    @property
    def model_name(self) -> str:
        return self._embedding_model

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS.get(self._embedding_model, 768)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_ollama.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/providers/ollama.py tests/providers/test_ollama.py
git commit -m "feat: add Ollama embedding/generation provider"
```

---

### Task 13: Document loaders

**Files:**
- Create: `rag_hybrid_search/ingestion/__init__.py`
- Create: `rag_hybrid_search/ingestion/loaders/__init__.py`
- Create: `rag_hybrid_search/ingestion/loaders/base.py`
- Create: `rag_hybrid_search/ingestion/loaders/markdown.py`
- Create: `rag_hybrid_search/ingestion/loaders/text.py`
- Create: `rag_hybrid_search/ingestion/loaders/html.py`
- Create: `rag_hybrid_search/ingestion/loaders/pdf.py`
- Test: `tests/ingestion/__init__.py`
- Test: `tests/ingestion/loaders/__init__.py`
- Test: `tests/ingestion/loaders/test_markdown.py`
- Test: `tests/ingestion/loaders/test_text.py`
- Test: `tests/ingestion/loaders/test_html.py`
- Test: `tests/ingestion/loaders/test_pdf.py`

**Interfaces:**
- Consumes: `Document` from `models.py`.
- Produces: `Loader` ABC (`format: Literal[...]` class attr, `load(path: str) -> Document`), plus `MarkdownLoader`, `TextLoader`, `HtmlLoader`, `PdfLoader`. `IngestionPipeline` (Task 16) is constructed with one concrete loader per ingest call site.

- [ ] **Step 1: Write failing test for markdown loader**

`tests/ingestion/loaders/test_markdown.py`:
```python
import hashlib

from rag_hybrid_search.ingestion.loaders.markdown import MarkdownLoader


def test_load_normalizes_content_and_computes_hash(tmp_path):
    path = tmp_path / "readme.md"
    content = "# Title\n\nSome body text.\n"
    path.write_text(content)

    doc = MarkdownLoader().load(str(path))

    assert doc.format == "markdown"
    assert doc.source_path == str(path)
    assert doc.content == content
    assert doc.document_id == hashlib.sha256(content.encode()).hexdigest()


def test_same_content_produces_same_document_id(tmp_path):
    path_a = tmp_path / "a.md"
    path_b = tmp_path / "b.md"
    path_a.write_text("identical content")
    path_b.write_text("identical content")

    doc_a = MarkdownLoader().load(str(path_a))
    doc_b = MarkdownLoader().load(str(path_b))

    assert doc_a.document_id == doc_b.document_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/loaders/test_markdown.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.ingestion'`

- [ ] **Step 3: Create package markers**

`rag_hybrid_search/ingestion/__init__.py`, `rag_hybrid_search/ingestion/loaders/__init__.py`,
`tests/ingestion/__init__.py`, `tests/ingestion/loaders/__init__.py`: all empty.

- [ ] **Step 4: Implement `ingestion/loaders/base.py`**

```python
import hashlib
from abc import ABC, abstractmethod
from typing import Literal

from rag_hybrid_search.models import Document


class Loader(ABC):
    format: Literal["markdown", "html", "text", "pdf"]

    @abstractmethod
    def load(self, path: str) -> Document:
        ...

    def _build_document(self, path: str, content: str) -> Document:
        document_id = hashlib.sha256(content.encode()).hexdigest()
        return Document(
            document_id=document_id,
            source_path=path,
            content=content,
            format=self.format,
        )
```

- [ ] **Step 5: Implement `ingestion/loaders/markdown.py`**

```python
from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class MarkdownLoader(Loader):
    format = "markdown"

    def load(self, path: str) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._build_document(path, content)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/ingestion/loaders/test_markdown.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Write and implement text loader**

`tests/ingestion/loaders/test_text.py`:
```python
from rag_hybrid_search.ingestion.loaders.text import TextLoader


def test_load_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("just plain text\nwith two lines")

    doc = TextLoader().load(str(path))

    assert doc.format == "text"
    assert "just plain text" in doc.content
```

`rag_hybrid_search/ingestion/loaders/text.py`:
```python
from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class TextLoader(Loader):
    format = "text"

    def load(self, path: str) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._build_document(path, content)
```

Run: `pytest tests/ingestion/loaders/test_text.py -v`
Expected: PASS (1 test)

- [ ] **Step 8: Write and implement HTML loader**

`tests/ingestion/loaders/test_html.py`:
```python
from rag_hybrid_search.ingestion.loaders.html import HtmlLoader


def test_load_strips_tags_and_scripts(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><script>evil()</script></head>"
        "<body><h1>Title</h1><p>Body text.</p></body></html>"
    )

    doc = HtmlLoader().load(str(path))

    assert doc.format == "html"
    assert "evil()" not in doc.content
    assert "Title" in doc.content
    assert "Body text." in doc.content
```

`rag_hybrid_search/ingestion/loaders/html.py`:
```python
from bs4 import BeautifulSoup

from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class HtmlLoader(Loader):
    format = "html"

    def load(self, path: str) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            raw_html = f.read()
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        content = soup.get_text(separator="\n", strip=True)
        return self._build_document(path, content)
```

Run: `pytest tests/ingestion/loaders/test_html.py -v`
Expected: PASS (1 test)

- [ ] **Step 9: Generate the PDF fixture used by the PDF loader test**

Create `tests/ingestion/loaders/fixtures/` and generate a real, minimal PDF
containing extractable text, using only `pypdf` (already a dependency —
no `reportlab` needed):

```bash
mkdir -p tests/ingestion/loaders/fixtures
python3 - <<'PY'
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

writer = PdfWriter()
writer.add_blank_page(width=200, height=200)
page = writer.pages[0]

content = b"BT /F1 12 Tf 20 100 Td (Sample PDF content for testing) Tj ET"
stream = DecodedStreamObject()
stream.set_data(content)
page[NameObject("/Contents")] = stream

resources = DictionaryObject()
resources[NameObject("/Font")] = DictionaryObject({
    NameObject("/F1"): DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
})
page[NameObject("/Resources")] = resources

with open("tests/ingestion/loaders/fixtures/sample.pdf", "wb") as f:
    writer.write(f)
print("fixture written")
PY
```

Expected output: `fixture written`, and
`tests/ingestion/loaders/fixtures/sample.pdf` exists.

If this `pypdf` low-level object API errs on the installed version, add
`reportlab` to `pyproject.toml`'s `dev` extras and regenerate with:
```python
from reportlab.pdfgen import canvas
c = canvas.Canvas("tests/ingestion/loaders/fixtures/sample.pdf")
c.drawString(20, 100, "Sample PDF content for testing")
c.save()
```
Either way the fixture is a checked-in binary generated once, not
regenerated by the test suite itself.

- [ ] **Step 10: Write the PDF loader test**

`tests/ingestion/loaders/test_pdf.py`:
```python
from rag_hybrid_search.ingestion.loaders.pdf import PdfLoader


def test_load_extracts_text_from_fixture_pdf():
    fixture_path = "tests/ingestion/loaders/fixtures/sample.pdf"

    doc = PdfLoader().load(fixture_path)

    assert doc.format == "pdf"
    assert "Sample PDF content for testing" in doc.content
```

- [ ] **Step 11: Run test to verify it fails**

Run: `pytest tests/ingestion/loaders/test_pdf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.ingestion.loaders.pdf'`

- [ ] **Step 12: Implement `ingestion/loaders/pdf.py`**

```python
from pypdf import PdfReader

from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class PdfLoader(Loader):
    format = "pdf"

    def load(self, path: str) -> Document:
        reader = PdfReader(path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        content = "\n".join(pages_text)
        return self._build_document(path, content)
```

- [ ] **Step 13: Run all loader tests**

Run: `pytest tests/ingestion/loaders/ -v`
Expected: PASS (all tests across markdown/text/html/pdf)

- [ ] **Step 14: Commit**

```bash
git add rag_hybrid_search/ingestion/__init__.py rag_hybrid_search/ingestion/loaders/ tests/ingestion/__init__.py tests/ingestion/loaders/
git commit -m "feat: add markdown, text, html, and pdf document loaders"
```

---

### Task 14: Chunkers

**Files:**
- Create: `rag_hybrid_search/ingestion/chunkers/__init__.py`
- Create: `rag_hybrid_search/ingestion/chunkers/base.py`
- Create: `rag_hybrid_search/ingestion/chunkers/fixed.py`
- Create: `rag_hybrid_search/ingestion/chunkers/recursive.py`
- Create: `rag_hybrid_search/ingestion/chunkers/semantic.py`
- Test: `tests/ingestion/chunkers/__init__.py`
- Test: `tests/ingestion/chunkers/test_fixed.py`
- Test: `tests/ingestion/chunkers/test_recursive.py`
- Test: `tests/ingestion/chunkers/test_semantic.py`

**Interfaces:**
- Consumes: `Document`, `Chunk` from `models.py`; `uuid7` from `uuid7.py`; `EmbeddingProvider` (semantic chunker only).
- Produces: `Chunker` ABC (`version: str` class attr, `chunk(document: Document) -> list[Chunk]`), `FixedChunker(chunk_size, chunk_overlap)` (version `"fixed-v1"`), `RecursiveChunker(chunk_size, chunk_overlap)` (version `"recursive-v1"`), `SemanticChunker(embedding_provider, similarity_threshold=0.5)` (version `"semantic-v1"`). `IngestionPipeline` (Task 16) is constructed with one concrete chunker.

- [ ] **Step 1: Write failing test for `FixedChunker`**

`tests/ingestion/chunkers/test_fixed.py`:
```python
from rag_hybrid_search.ingestion.chunkers.fixed import FixedChunker
from rag_hybrid_search.models import Document


def make_document(content):
    return Document(
        document_id="a" * 64, source_path="/docs/a.txt", content=content, format="text"
    )


def test_chunk_respects_size_and_overlap():
    content = "x" * 1000
    chunker = FixedChunker(chunk_size=300, chunk_overlap=50)

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) == 4
    assert chunks[0].char_count == 300
    assert chunks[0].text[-50:] == chunks[1].text[:50]
    assert all(c.strategy_version == "fixed-v1" for c in chunks)


def test_chunk_indexes_are_sequential():
    chunker = FixedChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk(make_document("y" * 250))

    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_short_document_produces_one_chunk():
    chunker = FixedChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(make_document("short text"))

    assert len(chunks) == 1
    assert chunks[0].text == "short text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/chunkers/test_fixed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.ingestion.chunkers'`

- [ ] **Step 3: Create package markers**

`rag_hybrid_search/ingestion/chunkers/__init__.py`, `tests/ingestion/chunkers/__init__.py`: empty.

- [ ] **Step 4: Implement `ingestion/chunkers/base.py`**

```python
from abc import ABC, abstractmethod

from rag_hybrid_search.models import Chunk, Document


class Chunker(ABC):
    version: str

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...
```

- [ ] **Step 5: Implement `ingestion/chunkers/fixed.py`**

```python
from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.models import Chunk, Document
from rag_hybrid_search.uuid7 import uuid7


class FixedChunker(Chunker):
    version = "fixed-v1"

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        step = self._chunk_size - self._chunk_overlap
        chunks = []
        index = 0
        position = 0
        while position < len(text):
            piece = text[position : position + self._chunk_size]
            chunks.append(
                Chunk(
                    chunk_id=uuid7(),
                    document_id=document.document_id,
                    chunk_index=index,
                    text=piece,
                    strategy_version=self.version,
                    heading=None,
                    page=None,
                    char_count=len(piece),
                )
            )
            index += 1
            position += step
        return chunks
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/ingestion/chunkers/test_fixed.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Write failing test for `RecursiveChunker`**

`tests/ingestion/chunkers/test_recursive.py`:
```python
from rag_hybrid_search.ingestion.chunkers.recursive import RecursiveChunker
from rag_hybrid_search.models import Document


def make_document(content):
    return Document(
        document_id="b" * 64, source_path="/docs/b.md", content=content, format="markdown"
    )


def test_splits_on_markdown_headers_and_tags_heading():
    content = (
        "# Intro\n\nSome intro text.\n\n"
        "## Setup\n\nSetup instructions here.\n\n"
        "## Usage\n\nUsage instructions here."
    )
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)

    chunks = chunker.chunk(make_document(content))

    headings = [c.heading for c in chunks]
    assert "Intro" in headings
    assert "Setup" in headings
    assert "Usage" in headings
    assert all(c.strategy_version == "recursive-v1" for c in chunks)


def test_splits_large_section_further_by_chunk_size():
    content = "# Section\n\n" + ("word " * 400)
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) > 1
    assert all(c.heading == "Section" for c in chunks)
```

- [ ] **Step 8: Run test to verify it fails**

Run: `pytest tests/ingestion/chunkers/test_recursive.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.ingestion.chunkers.recursive'`

- [ ] **Step 9: Implement `ingestion/chunkers/recursive.py`**

```python
import re

from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.models import Chunk, Document
from rag_hybrid_search.uuid7 import uuid7

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class RecursiveChunker(Chunker):
    version = "recursive-v1"

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        sections = self._split_by_headers(document.content)
        chunks: list[Chunk] = []
        index = 0
        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            for piece in self._split_by_size(body):
                chunks.append(
                    Chunk(
                        chunk_id=uuid7(),
                        document_id=document.document_id,
                        chunk_index=index,
                        text=piece,
                        strategy_version=self.version,
                        heading=heading,
                        page=None,
                        char_count=len(piece),
                    )
                )
                index += 1
        return chunks

    def _split_by_headers(self, text: str) -> list[tuple[str | None, str]]:
        matches = list(_HEADER_RE.finditer(text))
        if not matches:
            return [(None, text)]

        sections = []
        if matches[0].start() > 0:
            sections.append((None, text[: matches[0].start()]))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((heading, text[start:end]))
        return sections

    def _split_by_size(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text]
        step = self._chunk_size - self._chunk_overlap
        pieces = []
        position = 0
        while position < len(text):
            pieces.append(text[position : position + self._chunk_size])
            position += step
        return pieces
```

- [ ] **Step 10: Run test to verify it passes**

Run: `pytest tests/ingestion/chunkers/test_recursive.py -v`
Expected: PASS (2 tests)

- [ ] **Step 11: Write failing test for `SemanticChunker`**

`tests/ingestion/chunkers/test_semantic.py`:
```python
from rag_hybrid_search.ingestion.chunkers.semantic import SemanticChunker
from rag_hybrid_search.models import Document
from tests.fakes import FakeEmbeddingProvider


def make_document(content):
    return Document(
        document_id="c" * 64, source_path="/docs/c.txt", content=content, format="text"
    )


def test_splits_on_topic_boundary():
    content = (
        "The soup needs more salt. Add pepper to taste. "
        "The moon orbits the earth. Stars burn hydrogen for fuel."
    )
    chunker = SemanticChunker(
        embedding_provider=FakeEmbeddingProvider(), similarity_threshold=0.3
    )

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) >= 2
    assert all(c.strategy_version == "semantic-v1" for c in chunks)


def test_single_sentence_document_produces_one_chunk():
    chunker = SemanticChunker(
        embedding_provider=FakeEmbeddingProvider(), similarity_threshold=0.3
    )
    chunks = chunker.chunk(make_document("Just one sentence here."))

    assert len(chunks) == 1
```

- [ ] **Step 12: Run test to verify it fails**

Run: `pytest tests/ingestion/chunkers/test_semantic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.ingestion.chunkers.semantic'`

- [ ] **Step 13: Implement `ingestion/chunkers/semantic.py`**

```python
import re

from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.models import Chunk, Document
from rag_hybrid_search.providers.base import EmbeddingProvider
from rag_hybrid_search.uuid7 import uuid7

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(Chunker):
    version = "semantic-v1"

    def __init__(self, embedding_provider: EmbeddingProvider, similarity_threshold: float = 0.5):
        self._embedding_provider = embedding_provider
        self._similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = [s.strip() for s in _SENTENCE_RE.split(document.content) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(document, 0, sentences[0])]

        embeddings = self._embedding_provider.embed(sentences)

        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            similarity = _cosine(embeddings[i - 1], embeddings[i])
            if similarity >= self._similarity_threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        return [
            self._make_chunk(document, idx, " ".join(group))
            for idx, group in enumerate(groups)
        ]

    def _make_chunk(self, document: Document, index: int, text: str) -> Chunk:
        return Chunk(
            chunk_id=uuid7(),
            document_id=document.document_id,
            chunk_index=index,
            text=text,
            strategy_version=self.version,
            heading=None,
            page=None,
            char_count=len(text),
        )
```

- [ ] **Step 14: Run test to verify it passes**

Run: `pytest tests/ingestion/chunkers/test_semantic.py -v`
Expected: PASS (2 tests)

- [ ] **Step 15: Commit**

```bash
git add rag_hybrid_search/ingestion/chunkers/ tests/ingestion/chunkers/
git commit -m "feat: add fixed, recursive, and semantic chunking strategies"
```

---

### Task 15: Deduplication

**Files:**
- Create: `rag_hybrid_search/ingestion/dedup.py`
- Test: `tests/ingestion/test_dedup.py`

**Interfaces:**
- Consumes: `Chunk` from `models.py`.
- Produces: `is_duplicate(candidate: Chunk, candidate_embedding: list[float], existing: list[tuple[Chunk, list[float]]], cosine_threshold: float, text_threshold: float) -> bool`. `IngestionPipeline` (Task 16) calls this once per new chunk against already-indexed chunks.

- [ ] **Step 1: Write failing tests**

`tests/ingestion/test_dedup.py`:
```python
from rag_hybrid_search.ingestion.dedup import is_duplicate
from rag_hybrid_search.models import Chunk


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


def test_true_duplicate_is_detected():
    existing_chunk = make_chunk("c1", "def foo(): return bar()")
    existing = [(existing_chunk, [1.0, 0.0, 0.0])]
    candidate = make_chunk("c2", "def foo(): return bar()")

    result = is_duplicate(
        candidate,
        candidate_embedding=[1.0, 0.0, 0.0],
        existing=existing,
        cosine_threshold=0.95,
        text_threshold=0.9,
    )

    assert result is True


def test_high_cosine_but_different_text_is_not_duplicate():
    existing_chunk = make_chunk("c1", "x = [i for i in range(10)]")
    existing = [(existing_chunk, [0.99, 0.01, 0.0])]
    candidate = make_chunk("c2", "y = (j for j in range(20))")

    result = is_duplicate(
        candidate,
        candidate_embedding=[1.0, 0.0, 0.0],
        existing=existing,
        cosine_threshold=0.95,
        text_threshold=0.9,
    )

    assert result is False


def test_below_cosine_threshold_short_circuits_without_duplicate():
    existing_chunk = make_chunk("c1", "completely unrelated content")
    existing = [(existing_chunk, [0.0, 1.0, 0.0])]
    candidate = make_chunk("c2", "totally different topic here")

    result = is_duplicate(
        candidate,
        candidate_embedding=[1.0, 0.0, 0.0],
        existing=existing,
        cosine_threshold=0.95,
        text_threshold=0.9,
    )

    assert result is False


def test_no_existing_chunks_is_never_duplicate():
    candidate = make_chunk("c1", "anything")

    result = is_duplicate(
        candidate,
        candidate_embedding=[1.0, 0.0, 0.0],
        existing=[],
        cosine_threshold=0.95,
        text_threshold=0.9,
    )

    assert result is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_dedup.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.ingestion.dedup'`

- [ ] **Step 3: Implement `ingestion/dedup.py`**

```python
from difflib import SequenceMatcher

from rag_hybrid_search.models import Chunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()


def is_duplicate(
    candidate: Chunk,
    candidate_embedding: list[float],
    existing: list[tuple[Chunk, list[float]]],
    cosine_threshold: float,
    text_threshold: float,
) -> bool:
    for existing_chunk, existing_embedding in existing:
        cosine_sim = _cosine(candidate_embedding, existing_embedding)
        if cosine_sim <= cosine_threshold:
            continue
        text_sim = _text_similarity(candidate.text, existing_chunk.text)
        if text_sim > text_threshold:
            return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_dedup.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/ingestion/dedup.py tests/ingestion/test_dedup.py
git commit -m "feat: add two-stage cosine+text-similarity deduplication"
```

---

### Task 16: `IngestionPipeline`

**Files:**
- Create: `rag_hybrid_search/ingestion/pipeline.py`
- Test: `tests/ingestion/test_pipeline.py`

**Interfaces:**
- Consumes: `Loader` subclasses (Task 13), `Chunker` subclasses (Task 14), `is_duplicate` (Task 15), `SqliteChunkStore`/`ChunkStore` (Task 5/4), `IndexManager` (Task 8), `EmbeddingProvider` (Task 9), `EmbeddingRecord`/`IndexStatus` (Task 2).
- Produces: `IngestionPipeline(loader, chunker, embedding_provider, chunk_store, index_manager, dedup_cosine_threshold, dedup_text_threshold)` with public attributes `loader`, `chunker`, `embedding_provider`, `chunk_store`, `index_manager`, and method `ingest(path: str) -> IndexStatus`.

- [ ] **Step 1: Write failing tests**

`tests/ingestion/test_pipeline.py`:
```python
import pytest

from rag_hybrid_search.ingestion.chunkers.fixed import FixedChunker
from rag_hybrid_search.ingestion.loaders.text import TextLoader
from rag_hybrid_search.ingestion.pipeline import IngestionPipeline
from rag_hybrid_search.models import IndexStatus
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager
from tests.fakes import FakeEmbeddingProvider


@pytest.fixture
def pipeline(tmp_path):
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index_manager = IndexManager(chunk_store, vector_store, bm25)
    return IngestionPipeline(
        loader=TextLoader(),
        chunker=FixedChunker(chunk_size=100, chunk_overlap=0),
        embedding_provider=FakeEmbeddingProvider(),
        chunk_store=chunk_store,
        index_manager=index_manager,
        dedup_cosine_threshold=0.95,
        dedup_text_threshold=0.9,
    )


def test_ingest_produces_ready_status_and_chunks(pipeline, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Some content about hybrid retrieval systems.")

    status = pipeline.ingest(str(path))

    assert status == IndexStatus.READY
    chunks = list(pipeline.chunk_store.all())
    assert len(chunks) >= 1


def test_reingesting_unchanged_document_is_noop(pipeline, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Stable content that never changes.")

    pipeline.ingest(str(path))
    first_count = len(list(pipeline.chunk_store.all()))

    pipeline.ingest(str(path))
    second_count = len(list(pipeline.chunk_store.all()))

    assert first_count == second_count


def test_reingesting_edited_document_replaces_old_chunks(pipeline, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Original content version one.")
    pipeline.ingest(str(path))
    original_ids = {c.chunk_id for c in pipeline.chunk_store.all()}

    path.write_text("Completely different content version two, much longer than before.")
    pipeline.ingest(str(path))
    new_ids = {c.chunk_id for c in pipeline.chunk_store.all()}

    assert original_ids.isdisjoint(new_ids)
    assert len(new_ids) >= 1


def test_dedup_skips_near_duplicate_chunk_across_documents(pipeline, tmp_path):
    path_a = tmp_path / "a.txt"
    path_a.write_text("The quick brown fox jumps over the lazy dog repeatedly.")
    path_b = tmp_path / "b.txt"
    path_b.write_text("The quick brown fox jumps over the lazy dog repeatedly.")

    pipeline.ingest(str(path_a))
    count_after_first = len(list(pipeline.chunk_store.all()))

    pipeline.ingest(str(path_b))
    count_after_second = len(list(pipeline.chunk_store.all()))

    assert count_after_second == count_after_first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.ingestion.pipeline'`

- [ ] **Step 3: Implement `ingestion/pipeline.py`**

Deduplication is checked against every chunk already indexed (across all
previously ingested documents), by re-embedding existing chunk text once
per `ingest()` call — acceptable for this sub-spec's scope since embedding
is batched and corpora here are small; revisit if profiling later shows
this dominates ingest time on large corpora.

```python
from datetime import datetime, timezone

from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.ingestion.dedup import is_duplicate
from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Chunk, EmbeddingRecord, IndexStatus
from rag_hybrid_search.providers.base import EmbeddingProvider
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager


class IngestionPipeline:
    def __init__(
        self,
        loader: Loader,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        chunk_store: ChunkStore,
        index_manager: IndexManager,
        dedup_cosine_threshold: float,
        dedup_text_threshold: float,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.chunk_store = chunk_store
        self.index_manager = index_manager
        self._dedup_cosine_threshold = dedup_cosine_threshold
        self._dedup_text_threshold = dedup_text_threshold

    def ingest(self, path: str) -> IndexStatus:
        document = self.loader.load(path)

        existing_hash = self.chunk_store.get_document_hash(path)
        if existing_hash == document.document_id:
            return IndexStatus.READY
        if existing_hash is not None:
            self.index_manager.remove_document(existing_hash)

        new_chunks = self.chunker.chunk(document)
        if not new_chunks:
            return IndexStatus.READY

        embeddings = self.embedding_provider.embed([c.text for c in new_chunks])
        existing_pairs = self._existing_chunk_embeddings()

        surviving_chunks: list[Chunk] = []
        surviving_records: list[EmbeddingRecord] = []
        for chunk, embedding in zip(new_chunks, embeddings):
            if is_duplicate(
                chunk,
                embedding,
                existing_pairs,
                self._dedup_cosine_threshold,
                self._dedup_text_threshold,
            ):
                continue
            record = EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                embedding_model=self.embedding_provider.model_name,
                embedding_dimension=self.embedding_provider.dimension,
                provider=type(self.embedding_provider).__name__,
                created_at=datetime.now(timezone.utc),
            )
            surviving_chunks.append(chunk)
            surviving_records.append(record)
            existing_pairs.append((chunk, embedding))

        if not surviving_chunks:
            return IndexStatus.READY

        for chunk in surviving_chunks:
            self.chunk_store.put(chunk, source_path=path)

        return self.index_manager.index(surviving_chunks, surviving_records)

    def _existing_chunk_embeddings(self) -> list[tuple[Chunk, list[float]]]:
        existing_chunks = list(self.chunk_store.all())
        if not existing_chunks:
            return []
        texts = [c.text for c in existing_chunks]
        embeddings = self.embedding_provider.embed(texts)
        return list(zip(existing_chunks, embeddings))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_pipeline.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full suite to check no regressions**

Run: `pytest -v`
Expected: PASS, all tests across all modules so far green.

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/ingestion/pipeline.py tests/ingestion/test_pipeline.py
git commit -m "feat: add IngestionPipeline orchestrating load->chunk->dedup->index"
```

---

### Task 17: Dense and sparse retrievers

**Files:**
- Create: `rag_hybrid_search/retrieval/__init__.py`
- Create: `rag_hybrid_search/retrieval/dense.py`
- Create: `rag_hybrid_search/retrieval/sparse.py`
- Test: `tests/retrieval/__init__.py`
- Test: `tests/retrieval/test_dense.py`
- Test: `tests/retrieval/test_sparse.py`

**Interfaces:**
- Consumes: `EmbeddingProvider` (Task 9), `VectorStore` (Task 4/6), `ChunkStore`/`BM25Index` (Task 5/7), `RetrievedChunk` (Task 2).
- Produces: `DenseRetriever(embedding_provider, vector_store, chunk_store).search(query, k) -> list[RetrievedChunk]` (only `dense_score`/`chunk` populated, `rrf_score=0.0`, `final_rank=0`), `SparseRetriever(chunk_store, bm25_index).search(query, k) -> list[RetrievedChunk]` (only `bm25_score`/`chunk` populated). `HybridRetriever` (Task 20) is the sole caller of both.

- [ ] **Step 1: Create package markers**

`rag_hybrid_search/retrieval/__init__.py`, `tests/retrieval/__init__.py`: empty.

- [ ] **Step 2: Write failing test for `DenseRetriever`**

`tests/retrieval/test_dense.py`:
```python
from datetime import datetime, timezone

import pytest

from rag_hybrid_search.models import Chunk, EmbeddingRecord
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from tests.fakes import FakeEmbeddingProvider


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


@pytest.fixture
def retriever(tmp_path):
    provider = FakeEmbeddingProvider()
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    chunks = {
        "c1": make_chunk("c1", "hybrid retrieval combines dense and sparse search"),
        "c2": make_chunk("c2", "the weather today is sunny and warm"),
    }
    for chunk_id, chunk in chunks.items():
        chunk_store.put(chunk)
        embedding = provider.embed([chunk.text])[0]
        vector_store.upsert(
            chunk_id,
            EmbeddingRecord(
                chunk_id=chunk_id,
                embedding=embedding,
                embedding_model=provider.model_name,
                embedding_dimension=provider.dimension,
                provider="fake",
                created_at=datetime.now(timezone.utc),
            ),
        )
    return DenseRetriever(provider, vector_store, chunk_store)


def test_search_returns_chunk_with_dense_score(retriever):
    results = retriever.search("dense and sparse retrieval", k=2)

    assert len(results) <= 2
    assert all(r.dense_score is not None for r in results)
    assert all(r.bm25_score is None for r in results)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/retrieval/test_dense.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.retrieval.dense'`

- [ ] **Step 4: Implement `retrieval/dense.py`**

```python
from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.providers.base import EmbeddingProvider
from rag_hybrid_search.storage.base import ChunkStore, VectorStore


class DenseRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_store: ChunkStore,
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunk_store = chunk_store

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        query_embedding = self._embedding_provider.embed([query])[0]
        raw_results = self._vector_store.query(query_embedding, k)

        results = []
        for chunk_id, score in raw_results:
            chunk = self._chunk_store.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=score,
                    bm25_score=None,
                    rrf_score=0.0,
                    rerank_score=None,
                    final_rank=0,
                )
            )
        return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/retrieval/test_dense.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Write failing test for `SparseRetriever`**

`tests/retrieval/test_sparse.py`:
```python
import pytest

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


@pytest.fixture
def retriever(tmp_path):
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    chunks = [
        make_chunk("c1", "how to resolve ERROR_CODE_0x834 during deployment"),
        make_chunk("c2", "onboarding guide for new hires"),
    ]
    for chunk in chunks:
        chunk_store.put(chunk)
    bm25.build(chunks)
    return SparseRetriever(chunk_store, bm25)


def test_search_finds_exact_keyword_with_bm25_score(retriever):
    results = retriever.search("ERROR_CODE_0x834", k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].bm25_score is not None
    assert results[0].dense_score is None
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pytest tests/retrieval/test_sparse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.retrieval.sparse'`

- [ ] **Step 8: Implement `retrieval/sparse.py`**

```python
from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.bm25_index import BM25Index


class SparseRetriever:
    def __init__(self, chunk_store: ChunkStore, bm25_index: BM25Index):
        self._chunk_store = chunk_store
        self._bm25_index = bm25_index

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        raw_results = self._bm25_index.search(query, k)

        results = []
        for chunk_id, score in raw_results:
            chunk = self._chunk_store.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=None,
                    bm25_score=score,
                    rrf_score=0.0,
                    rerank_score=None,
                    final_rank=0,
                )
            )
        return results
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/retrieval/test_sparse.py -v`
Expected: PASS (1 test)

- [ ] **Step 10: Commit**

```bash
git add rag_hybrid_search/retrieval/__init__.py rag_hybrid_search/retrieval/dense.py rag_hybrid_search/retrieval/sparse.py tests/retrieval/
git commit -m "feat: add DenseRetriever and SparseRetriever"
```

---

### Task 18: Weighted RRF fusion

**Files:**
- Create: `rag_hybrid_search/retrieval/fusion.py`
- Test: `tests/retrieval/test_fusion.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from `models.py`.
- Produces: `weighted_rrf(dense_results: list[RetrievedChunk], sparse_results: list[RetrievedChunk], dense_weight: float, sparse_weight: float, k: int) -> list[RetrievedChunk]` — merges by `chunk.chunk_id`, sets `rrf_score`, sorts descending, preserves both `dense_score` and `bm25_score` when a chunk appears in both lists. `HybridRetriever` (Task 20) is the sole caller.

- [ ] **Step 1: Write failing tests**

`tests/retrieval/test_fusion.py`:
```python
from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_hybrid_search.retrieval.fusion import weighted_rrf


def make_result(chunk_id, dense_score=None, bm25_score=None):
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=f"text {chunk_id}",
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=10,
    )
    return RetrievedChunk(
        chunk=chunk,
        dense_score=dense_score,
        bm25_score=bm25_score,
        rrf_score=0.0,
        rerank_score=None,
        final_rank=0,
    )


def test_fuses_and_ranks_by_combined_reciprocal_rank():
    dense = [make_result("a", dense_score=0.9), make_result("b", dense_score=0.8)]
    sparse = [make_result("b", bm25_score=5.0), make_result("a", bm25_score=1.0)]

    fused = weighted_rrf(dense, sparse, dense_weight=0.7, sparse_weight=0.3, k=60)

    assert [r.chunk.chunk_id for r in fused] == ["a", "b"]
    expected_a = 0.7 * (1 / (60 + 1)) + 0.3 * (1 / (60 + 2))
    assert abs(fused[0].rrf_score - expected_a) < 1e-9


def test_chunk_only_in_one_list_still_included():
    dense = [make_result("a", dense_score=0.9)]
    sparse = [make_result("b", bm25_score=3.0)]

    fused = weighted_rrf(dense, sparse, dense_weight=0.7, sparse_weight=0.3, k=60)

    ids = {r.chunk.chunk_id for r in fused}
    assert ids == {"a", "b"}


def test_empty_inputs_return_empty_list():
    assert weighted_rrf([], [], dense_weight=0.7, sparse_weight=0.3, k=60) == []


def test_preserves_original_dense_and_bm25_scores_in_merged_result():
    dense = [make_result("a", dense_score=0.9)]
    sparse = [make_result("a", bm25_score=2.0)]

    fused = weighted_rrf(dense, sparse, dense_weight=0.7, sparse_weight=0.3, k=60)

    assert fused[0].dense_score == 0.9
    assert fused[0].bm25_score == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.retrieval.fusion'`

- [ ] **Step 3: Implement `retrieval/fusion.py`**

```python
from rag_hybrid_search.models import RetrievedChunk


def weighted_rrf(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    dense_weight: float,
    sparse_weight: float,
    k: int,
) -> list[RetrievedChunk]:
    """Weighted Reciprocal Rank Fusion.

    Standard RRF sums 1/(k+rank) contributions from each ranked list
    unweighted. This variant scales each list's contribution by a
    configured weight before summing, so callers can bias fusion toward
    dense or sparse results — hence "weighted", not vanilla RRF.
    """
    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk.chunk_id
        merged[chunk_id] = result
        scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight * (1 / (k + rank))

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result.chunk.chunk_id
        if chunk_id in merged:
            existing = merged[chunk_id]
            merged[chunk_id] = existing.model_copy(update={"bm25_score": result.bm25_score})
        else:
            merged[chunk_id] = result
        scores[chunk_id] = scores.get(chunk_id, 0.0) + sparse_weight * (1 / (k + rank))

    fused = [
        merged[chunk_id].model_copy(update={"rrf_score": score})
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    return fused
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_fusion.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/retrieval/fusion.py tests/retrieval/test_fusion.py
git commit -m "feat: add weighted RRF fusion"
```

---

### Task 19: Cross-encoder reranker

**Files:**
- Create: `rag_hybrid_search/retrieval/rerank.py`
- Test: `tests/retrieval/test_rerank.py`

**Interfaces:**
- Consumes: `RerankProvider` ABC (Task 9), `RetrievedChunk` (Task 2).
- Produces: `CrossEncoderReranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2")` implementing `RerankProvider`, setting `rerank_score` and 1-indexed `final_rank` on the returned top-`n`. `HybridRetriever` (Task 20) is the sole caller.

- [ ] **Step 1: Write failing tests**

`tests/retrieval/test_rerank.py`:
```python
from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_hybrid_search.retrieval.rerank import CrossEncoderReranker


def make_result(chunk_id, text):
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )
    return RetrievedChunk(
        chunk=chunk,
        dense_score=0.5,
        bm25_score=0.5,
        rrf_score=0.01,
        rerank_score=None,
        final_rank=0,
    )


def test_rerank_orders_by_relevance_and_sets_final_rank():
    reranker = CrossEncoderReranker()
    candidates = [
        make_result("a", "The Eiffel Tower is located in Paris, France."),
        make_result("b", "Bananas are a good source of potassium."),
    ]

    results = reranker.rerank("Where is the Eiffel Tower?", candidates, top_n=2)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].rerank_score is not None
    assert [r.final_rank for r in results] == [1, 2]


def test_rerank_respects_top_n():
    reranker = CrossEncoderReranker()
    candidates = [make_result(str(i), f"filler text number {i}") for i in range(5)]

    results = reranker.rerank("filler text", candidates, top_n=2)

    assert len(results) == 2


def test_rerank_empty_candidates_returns_empty():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("anything", [], top_n=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.retrieval.rerank'`

- [ ] **Step 3: Implement `retrieval/rerank.py`**

```python
from sentence_transformers import CrossEncoder

from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.providers.base import RerankProvider


class CrossEncoderReranker(RerankProvider):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        top = scored[:top_n]
        return [
            candidate.model_copy(update={"rerank_score": float(score), "final_rank": rank})
            for rank, (candidate, score) in enumerate(top, start=1)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_rerank.py -v`
Expected: PASS (3 tests). Note: this test downloads the small
`ms-marco-MiniLM-L-6-v2` model (~80MB) on first run — expect it to be slow
once, then cached under `~/.cache/torch/sentence_transformers/`.

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "feat: add cross-encoder reranker"
```

---

### Task 20: `HybridRetriever` orchestrator with telemetry

**Files:**
- Create: `rag_hybrid_search/retrieval/retriever.py`
- Test: `tests/retrieval/test_retriever.py`

**Interfaces:**
- Consumes: `DenseRetriever` (Task 17), `SparseRetriever` (Task 17), `weighted_rrf` (Task 18), `RerankProvider` (Task 9/19), `RetrievalTrace`, `RetrievedChunk` (Task 2).
- Produces: `HybridRetriever(dense_retriever, sparse_retriever, rerank_provider, dense_weight, sparse_weight, rrf_k, dense_k, sparse_k, rerank_top_n).retrieve(query: str) -> tuple[list[RetrievedChunk], RetrievalTrace]`. This is the top-level entry point later phases (generation, API) will call.

- [ ] **Step 1: Write failing test**

`tests/retrieval/test_retriever.py`:
```python
from datetime import datetime, timezone

import pytest

from rag_hybrid_search.models import Chunk, EmbeddingRecord
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.rerank import CrossEncoderReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from tests.fakes import FakeEmbeddingProvider


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


@pytest.fixture
def hybrid_retriever(tmp_path):
    provider = FakeEmbeddingProvider()
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))

    docs = [
        make_chunk("c1", "how to resolve ERROR_CODE_0x834 during deployment"),
        make_chunk("c2", "onboarding guide for new engineering hires"),
        make_chunk("c3", "deploying services safely to production"),
    ]
    for chunk in docs:
        chunk_store.put(chunk)
        embedding = provider.embed([chunk.text])[0]
        vector_store.upsert(
            chunk.chunk_id,
            EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                embedding_model=provider.model_name,
                embedding_dimension=provider.dimension,
                provider="fake",
                created_at=datetime.now(timezone.utc),
            ),
        )
    bm25.build(docs)

    dense = DenseRetriever(provider, vector_store, chunk_store)
    sparse = SparseRetriever(chunk_store, bm25)
    reranker = CrossEncoderReranker()

    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        rerank_provider=reranker,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        dense_k=10,
        sparse_k=10,
        rerank_top_n=2,
    )


def test_retrieve_returns_ranked_results_and_trace(hybrid_retriever):
    results, trace = hybrid_retriever.retrieve("ERROR_CODE_0x834")

    assert len(results) <= 2
    assert results[0].chunk.chunk_id == "c1"
    assert [r.final_rank for r in results] == list(range(1, len(results) + 1))
    assert trace.dense_latency_ms > 0
    assert trace.bm25_latency_ms > 0
    assert trace.total_latency_ms == pytest.approx(
        trace.dense_latency_ms
        + trace.bm25_latency_ms
        + trace.fusion_latency_ms
        + trace.rerank_latency_ms
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_hybrid_search.retrieval.retriever'`

- [ ] **Step 3: Implement `retrieval/retriever.py`**

```python
import time

from rag_hybrid_search.models import RetrievalTrace, RetrievedChunk
from rag_hybrid_search.providers.base import RerankProvider
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.fusion import weighted_rrf
from rag_hybrid_search.retrieval.sparse import SparseRetriever


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rerank_provider: RerankProvider,
        dense_weight: float,
        sparse_weight: float,
        rrf_k: int,
        dense_k: int,
        sparse_k: int,
        rerank_top_n: int,
    ):
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._rerank_provider = rerank_provider
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._rrf_k = rrf_k
        self._dense_k = dense_k
        self._sparse_k = sparse_k
        self._rerank_top_n = rerank_top_n

    def retrieve(self, query: str) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        trace = RetrievalTrace()

        start = time.perf_counter()
        dense_results = self._dense_retriever.search(query, k=self._dense_k)
        trace.dense_latency_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        sparse_results = self._sparse_retriever.search(query, k=self._sparse_k)
        trace.bm25_latency_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        fused = weighted_rrf(
            dense_results,
            sparse_results,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
            k=self._rrf_k,
        )
        trace.fusion_latency_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        reranked = self._rerank_provider.rerank(query, fused, top_n=self._rerank_top_n)
        trace.rerank_latency_ms = (time.perf_counter() - start) * 1000

        return reranked, trace
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_retriever.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the entire test suite**

Run: `pytest -v`
Expected: PASS — every test across `models`, `config`, `storage`,
`providers`, `ingestion`, `retrieval` is green with no regressions.

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/retrieval/retriever.py tests/retrieval/test_retriever.py
git commit -m "feat: add HybridRetriever orchestrator with latency telemetry"
```

---

### Task 21: End-to-end integration test with a small sample corpus

**Files:**
- Create: `tests/fixtures/sample_docs/setup.md`
- Create: `tests/fixtures/sample_docs/deployment.md`
- Create: `tests/fixtures/sample_docs/onboarding.md`
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: `IngestionPipeline` (Task 16), `HybridRetriever` (Task 20), all storage/provider concrete classes.
- Produces: nothing new for later tasks — this is the acceptance test proving the full Phase 1+2 pipeline works together end to end.

- [ ] **Step 1: Create three small sample markdown docs**

`tests/fixtures/sample_docs/setup.md`:
```markdown
# Environment Setup

## Prerequisites

Install Python 3.11 or later and create a virtual environment before
installing dependencies.

## Configuration

Set the RAG_PROVIDER environment variable to choose between nvidia and
ollama. If RAG_PROVIDER is unset, nvidia is used by default.
```

`tests/fixtures/sample_docs/deployment.md`:
```markdown
# Deployment Guide

## Common Errors

If you see ERROR_CODE_0x834 during deployment, it means the persistent
disk was not mounted before the service started. Remount the disk and
restart the container.

## Rollback Procedure

To roll back a bad deployment, redeploy the previous Docker image tag and
verify health checks pass before routing traffic to it.
```

`tests/fixtures/sample_docs/onboarding.md`:
```markdown
# New Engineer Onboarding

## First Week

New engineers should read the architecture overview and set up their local
development environment using the setup guide.

## Access Requests

Request access to the internal documentation repository and the deployment
dashboard through the access request form.
```

- [ ] **Step 2: Write the end-to-end test**

`tests/test_end_to_end.py`:
```python
import pytest

from rag_hybrid_search.ingestion.chunkers.recursive import RecursiveChunker
from rag_hybrid_search.ingestion.loaders.markdown import MarkdownLoader
from rag_hybrid_search.ingestion.pipeline import IngestionPipeline
from rag_hybrid_search.models import IndexStatus
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.rerank import CrossEncoderReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager
from tests.fakes import FakeEmbeddingProvider

SAMPLE_DOCS = [
    "tests/fixtures/sample_docs/setup.md",
    "tests/fixtures/sample_docs/deployment.md",
    "tests/fixtures/sample_docs/onboarding.md",
]


@pytest.fixture
def system(tmp_path):
    provider = FakeEmbeddingProvider()
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index_manager = IndexManager(chunk_store, vector_store, bm25)

    pipeline = IngestionPipeline(
        loader=MarkdownLoader(),
        chunker=RecursiveChunker(chunk_size=300, chunk_overlap=30),
        embedding_provider=provider,
        chunk_store=chunk_store,
        index_manager=index_manager,
        dedup_cosine_threshold=0.95,
        dedup_text_threshold=0.9,
    )

    for path in SAMPLE_DOCS:
        status = pipeline.ingest(path)
        assert status == IndexStatus.READY

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25),
        rerank_provider=CrossEncoderReranker(),
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        dense_k=10,
        sparse_k=10,
        rerank_top_n=3,
    )
    return retriever


def test_keyword_query_surfaces_deployment_error_doc(system):
    results, trace = system.retrieve("ERROR_CODE_0x834")

    assert any("ERROR_CODE_0x834" in r.chunk.text for r in results)
    assert trace.total_latency_ms > 0


def test_conceptual_query_surfaces_onboarding_doc(system):
    results, _trace = system.retrieve("What should new engineers do in their first week?")

    assert any(
        "onboarding" in r.chunk.text.lower() or "first week" in r.chunk.text.lower()
        for r in results
    )


def test_all_results_have_final_rank_set(system):
    results, _trace = system.retrieve("How do I configure the provider?")

    ranks = [r.final_rank for r in results]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1
```

- [ ] **Step 3: Run test and investigate any failures as real bugs**

Run: `pytest tests/test_end_to_end.py -v`
Expected: PASS (3 tests), since every dependency already exists from prior
tasks. If a relevance assertion fails, inspect the actual `results` (e.g.
`print([r.chunk.text for r in results])` temporarily) to see the real
ranking, and fix the underlying retrieval/fusion/rerank logic or adjust the
sample doc wording/query to be unambiguous — do not loosen the assertion
to mask a real ranking bug.

- [ ] **Step 4: Run the full suite one final time**

Run: `pytest -v`
Expected: PASS, full suite green, no regressions in any earlier task's tests.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/sample_docs/ tests/test_end_to_end.py
git commit -m "test: add end-to-end integration test over sample doc corpus"
```

---

## Self-Review Notes

- **Spec coverage:** loaders (markdown/html/text/pdf) → Task 13; chunkers
  (fixed/recursive/semantic) → Task 14; two-stage dedup → Task 15; document
  identity + incremental re-index → Task 16; dense+sparse+weighted RRF+rerank
  → Tasks 17-20; IndexManager → Task 8; provider abstraction incl.
  RerankProvider → Tasks 9, 19; typed models incl. EmbeddingRecord separate
  from Chunk → Task 2; SQLite ChunkStore → Task 5; config validation →
  Task 3; telemetry hooks → Task 20; end-to-end proof → Task 21. No spec
  section lacks a task.
- **Placeholder scan:** no TBD/TODO markers found.
- **Type consistency:** `RetrievedChunk` field names (`dense_score`,
  `bm25_score`, `rrf_score`, `rerank_score`, `final_rank`) match across
  Tasks 2, 17, 18, 19, 20, 21. `IndexStatus` values (`READY`/`FAILED`/
  `PENDING`/`INDEXING`) match between Task 2 and Tasks 8/16.
  `Chunker.version`/`Chunk.strategy_version` naming matches between Task 14
  and Task 2. `ChunkStore.put(chunk, source_path=None)` signature is
  consistent across Tasks 4, 5, 8, 16. `DenseRetriever(embedding_provider,
  vector_store, chunk_store)` constructor order matches every call site in
  Tasks 17, 20, 21.
