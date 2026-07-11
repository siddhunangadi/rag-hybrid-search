# Pinecone Storage Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Note for this project:** the user has said they will implement the code
> changes themselves after this plan is written. This plan is a reference for
> manual implementation, not necessarily for automated subagent execution —
> follow whichever workflow you're actually using it for.

**Goal:** Replace Chroma (dense vectors) + SQLite (chunk metadata) + local BM25
(sparse index) with Pinecone, behind a single `RAG_STORAGE_BACKEND` flag, while
leaving `DenseRetriever`/`SparseRetriever`/`HybridRetriever`/`weighted_rrf`/the
reranker/the entire generation pipeline untouched — fixing Render's
ephemeral-disk redeploy problem without changing retrieval behavior.

**Architecture:** A small `PineconeClient` wrapper holds one Pinecone SDK
index connection. `PineconeVectorStore` and `PineconeChunkStore` are two
separate classes, each implementing exactly one ABC (`VectorStore`,
`ChunkStore` respectively), both constructed from the same shared
`PineconeClient` instance — preserving the existing interface-segregation
design rather than merging responsibilities into one class. `PineconeSparseIndex`
matches `BM25Index`'s `search(query, k)` shape against a separate Pinecone
index using a hosted sparse embedding model — verified as a real
prerequisite, not assumed available, with a documented client-side-encoder
fallback if it isn't. `api/dependencies.py`'s `build_container` branches once
on `settings.storage_backend` to wire either the existing Chroma/SQLite/BM25
trio or the new Pinecone-backed set behind the same ABCs. Dense migration
(vectors + metadata) is deployed and verified on Render on its own, before
any sparse-migration code is written — isolating which layer is responsible
if something breaks.

**Tech Stack:** Python 3.11+, `pinecone` SDK (official Python client), Pydantic
v2 (`Settings`), pytest with `unittest.mock` for Pinecone client mocking.

**Spec:** `docs/superpowers/specs/2026-07-11-pinecone-storage-migration-design.md`

## Global Constraints

- `DenseRetriever`, `SparseRetriever`, `HybridRetriever`, `weighted_rrf`,
  `rerank.py`/`passthrough_rerank.py` are **never modified** by this plan —
  they only depend on the `VectorStore`/`ChunkStore` ABCs already.
- `RetrievedChunk.bm25_score`/`rrf_score` fields are **never removed** by this
  plan (deferred to a future cleanup phase, after production validation).
- No renames: `HybridRetriever` keeps its name and interface exactly.
- Fusion stays application-level: `DenseRetriever` and `SparseRetriever` each
  issue one independent Pinecone query; Pinecone's native alpha-weighted
  hybrid search is never used (would bypass `weighted_rrf`).
- Default behavior unchanged: `RAG_STORAGE_BACKEND` defaults to `local` — no
  existing deployment/test changes behavior without opting in.
- `PineconeChunkStore` implements `ChunkStore`'s full real contract, including
  the three methods beyond the ABC's declared four that have real callers today:
  `get_document_hash` (ingestion dedup, `rag_hybrid_search/ingestion/pipeline.py:42`),
  `get_by_legal_metadata` (compliance query routing,
  `rag_hybrid_search/compliance/query_router.py:87,95`), `get_document_summaries`
  (`rag_pipeline/rag_pipeline.py:485`, `api/routes.py:347`). Missing any of
  these breaks a real code path under `RAG_STORAGE_BACKEND=pinecone`, not just
  a theoretical one.
- Before writing any Task 4/5 (sparse) code: verify Pinecone hosted sparse
  embedding availability against the real account/SDK in use (Task 4, Step 1).
  Do not assume it works.

---

### Task 1: Extend `ChunkStore` ABC with its real contract

**Files:**
- Modify: `rag_hybrid_search/storage/base.py`
- Test: `tests/storage/test_chunk_store_contract.py` (new — a contract test
  parametrized to run against every `ChunkStore` implementation, starting
  with `SqliteChunkStore`; `PineconeChunkStore` gets added to this same
  parametrization in Task 2)

**Interfaces:**
- Produces: `ChunkStore` ABC gains three new abstract methods:
  `get_document_hash(source_path: str) -> Optional[str]` (already present —
  moving from implicit to explicit contract, no signature change),
  `get_by_legal_metadata(filters: dict[str, str]) -> list[Chunk]`,
  `get_document_summaries() -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/storage/test_chunk_store_contract.py
import pytest

from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore


def _sqlite_store(tmp_path):
    return SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))


IMPLEMENTATIONS = [_sqlite_store]


@pytest.mark.parametrize("make_store", IMPLEMENTATIONS)
def test_implements_full_chunk_store_contract(make_store, tmp_path):
    store = make_store(tmp_path)
    assert isinstance(store, ChunkStore)
    # These three are real, load-bearing methods beyond the ABC's original
    # four -- this test exists so a future ChunkStore implementation can't
    # silently skip them and break ingestion dedup / compliance routing /
    # document-summary endpoints.
    assert hasattr(store, "get_document_hash")
    assert hasattr(store, "get_by_legal_metadata")
    assert hasattr(store, "get_document_summaries")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/storage/test_chunk_store_contract.py -v`
Expected: FAIL — `SqliteChunkStore` isn't (yet) required by the ABC to have
these, but since it already does, this specific assertion form
(`hasattr`) would actually pass today. Instead, verify the *abstractness*
change fails correctly:

```python
def test_abc_requires_the_three_extra_methods():
    with pytest.raises(TypeError, match="abstract method"):
        class Incomplete(ChunkStore):
            def get(self, chunk_id): ...
            def get_by_document(self, document_id): ...
            def get_document_hash(self, source_path): ...
            def put(self, chunk): ...
            def delete_by_document(self, document_id): ...
            def all(self): ...
            # missing get_by_legal_metadata and get_document_summaries
        Incomplete()
```

Add this test alongside the one above. Run again — this one should FAIL
(no `TypeError` raised yet) since `base.py` hasn't been changed.

- [ ] **Step 3: Extend the ABC**

In `rag_hybrid_search/storage/base.py`, add to the `ChunkStore` class (after
its existing `all` abstract method):

```python
    @abstractmethod
    def get_by_legal_metadata(self, filters: dict[str, str]) -> list[Chunk]:
        ...

    @abstractmethod
    def get_document_summaries(self) -> list[dict]:
        ...
```

(`get_document_hash` is already declared abstract in the current file per
the codebase read for this plan — if it isn't, add it identically to the
existing abstract methods' style.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/storage/test_chunk_store_contract.py -v`
Expected: both tests pass. Also run the full existing storage test suite to
confirm `SqliteChunkStore` (already implementing all these methods) isn't
broken by the ABC becoming stricter:

Run: `uv run python -m pytest tests/storage/ -v`
Expected: all pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/storage/base.py tests/storage/test_chunk_store_contract.py
git commit -m "feat: make ChunkStore ABC declare its full real contract"
```

---

### Task 2: `PineconeClient`, `PineconeVectorStore`, `PineconeChunkStore`

**Files:**
- Create: `rag_hybrid_search/storage/pinecone_client.py`
- Create: `rag_hybrid_search/storage/pinecone_vector_store.py`
- Create: `rag_hybrid_search/storage/pinecone_chunk_store.py`
- Test: `tests/storage/test_pinecone_vector_store.py`,
  `tests/storage/test_pinecone_chunk_store.py`
- Modify: `pyproject.toml` (add `pinecone` to dependencies)

**Interfaces:**
- Consumes: `VectorStore`, `ChunkStore` ABCs (Task 1) from
  `rag_hybrid_search.storage.base`; `Chunk`, `EmbeddingRecord` from
  `rag_hybrid_search.models`.
- Produces: `PineconeClient(api_key: str, index_name: str, environment:
  str | None = None)` — thin wrapper exposing `.index` (the raw Pinecone SDK
  index handle), shared by both store classes so there's exactly one
  connection per Pinecone index, not one per store. `PineconeVectorStore(client:
  PineconeClient)` implementing only `VectorStore`
  (`upsert(chunk_id, embedding_record)`, `query(embedding, k) ->
  list[tuple[str, float]]`, `delete(chunk_ids: list[str])`).
  `PineconeChunkStore(client: PineconeClient)` implementing only `ChunkStore`
  (`get(chunk_id) -> Optional[Chunk]`, `get_by_document(document_id) ->
  list[Chunk]`, `get_document_hash(source_path) -> Optional[str]`,
  `put(chunk, source_path=None)`, `delete_by_document(document_id)`,
  `all() -> Iterator[Chunk]`, `get_by_legal_metadata(filters) -> list[Chunk]`,
  `get_document_summaries() -> list[dict]`).

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"pinecone>=5.0"` to `[project.dependencies]`
(alongside the existing `chromadb`, `rank_bm25` entries). Run:

```bash
uv sync
```

- [ ] **Step 2: Verify metadata-filter query semantics before writing any
  `PineconeChunkStore` code**

`get_by_document`, `get_document_hash`, and `get_by_legal_metadata` all need
to look up chunks by metadata alone, not by vector similarity. Against the
real Pinecone account/SDK version this project uses, confirm one of:

(a) `index.query(filter=..., top_k=N, include_metadata=True)` works with no
    `vector`/`sparse_vector` argument at all, or
(b) it requires *some* vector argument even when filtering by metadata only
    (in which case confirm whether a zero-vector of the index's real
    dimension is accepted, and what that dimension is), or
(c) a different primitive exists for metadata-only lookup (e.g. `list()`
    combined with a metadata filter, or a dedicated fetch-by-filter call).

Record which of (a)/(b)/(c) is true for the real SDK before writing Step 4's
implementation — do not guess at query-call shape and ship it; the codebase
this plan is based on doesn't have a `PineconeChunkStore` yet, so nothing
depends on this being right on the first attempt except correctness. If (b),
the `_metadata_query` dimension constant in Step 4's code needs to be set to
the index's actual configured dimension (e.g. the NVIDIA embedding model's
output dimension), not an arbitrary placeholder.

- [ ] **Step 3: Write the failing tests**

```python
# tests/storage/test_pinecone_vector_store.py
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.base import VectorStore
from rag_hybrid_search.storage.pinecone_client import PineconeClient
from rag_hybrid_search.storage.pinecone_vector_store import PineconeVectorStore


def _embedding_record(chunk_id="c1"):
    return EmbeddingRecord(
        chunk_id=chunk_id, embedding=[0.1, 0.2, 0.3], embedding_model="nv-embed",
        embedding_dimension=3, provider="nvidia", created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_client():
    with patch("rag_hybrid_search.storage.pinecone_client.Pinecone") as mock_pc_cls:
        mock_index = MagicMock()
        mock_pc_cls.return_value.Index.return_value = mock_index
        client = PineconeClient(api_key="k", index_name="idx")
        yield client, mock_index


def test_implements_vector_store(mock_client):
    client, _ = mock_client
    store = PineconeVectorStore(client)
    assert isinstance(store, VectorStore)


def test_upsert_sends_vector_only(mock_client):
    client, mock_index = mock_client
    store = PineconeVectorStore(client)
    store.upsert("c1", _embedding_record())
    mock_index.upsert.assert_called_once_with(
        vectors=[{"id": "c1", "values": [0.1, 0.2, 0.3]}],
    )


def test_query_returns_chunk_id_score_pairs(mock_client):
    client, mock_index = mock_client
    mock_index.query.return_value = MagicMock(
        matches=[MagicMock(id="c1", score=0.9), MagicMock(id="c2", score=0.7)]
    )
    store = PineconeVectorStore(client)
    results = store.query([0.1, 0.2, 0.3], k=5)
    assert results == [("c1", 0.9), ("c2", 0.7)]
    mock_index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3], top_k=5, include_metadata=False,
    )


def test_delete(mock_client):
    client, mock_index = mock_client
    store = PineconeVectorStore(client)
    store.delete(["c1", "c2"])
    mock_index.delete.assert_called_once_with(ids=["c1", "c2"])


def test_shares_one_client_across_two_stores(mock_client):
    """Both PineconeVectorStore and PineconeChunkStore, constructed from the
    same PineconeClient, must issue calls against the same underlying index
    object -- not open a second connection."""
    client, mock_index = mock_client
    from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
    vector_store = PineconeVectorStore(client)
    chunk_store = PineconeChunkStore(client)
    assert vector_store._index is chunk_store._index is mock_index
```

```python
# tests/storage/test_pinecone_chunk_store.py
from unittest.mock import MagicMock, patch

import pytest

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
from rag_hybrid_search.storage.pinecone_client import PineconeClient


def _chunk(chunk_id="c1", document_id="d1", chunk_index=0, text="hello world",
           heading=None, page=None):
    return Chunk(
        chunk_id=chunk_id, document_id=document_id, chunk_index=chunk_index,
        text=text, strategy_version="fixed-v1", heading=heading, page=page,
        char_count=len(text),
    )


@pytest.fixture
def mock_client():
    with patch("rag_hybrid_search.storage.pinecone_client.Pinecone") as mock_pc_cls:
        mock_index = MagicMock()
        mock_pc_cls.return_value.Index.return_value = mock_index
        client = PineconeClient(api_key="k", index_name="idx")
        yield client, mock_index


def test_implements_chunk_store(mock_client):
    client, _ = mock_client
    store = PineconeChunkStore(client)
    assert isinstance(store, ChunkStore)


def test_put_stores_metadata_without_touching_vector_values(mock_client):
    client, mock_index = mock_client
    mock_index.fetch.return_value = MagicMock(vectors={})
    store = PineconeChunkStore(client)
    store.put(_chunk(), source_path="doc.md")
    mock_index.update.assert_called_once()
    call_kwargs = mock_index.update.call_args.kwargs
    assert call_kwargs["id"] == "c1"
    assert call_kwargs["set_metadata"]["document_id"] == "d1"
    assert call_kwargs["set_metadata"]["text"] == "hello world"
    assert call_kwargs["set_metadata"]["source_path"] == "doc.md"


def test_get_by_chunk_id_reconstructs_chunk(mock_client):
    client, mock_index = mock_client
    mock_index.fetch.return_value = MagicMock(
        vectors={
            "c1": MagicMock(metadata={
                "document_id": "d1", "chunk_index": 0, "text": "hello world",
                "strategy_version": "fixed-v1", "heading": "", "page": -1,
                "char_count": 11, "source_path": "doc.md",
            })
        }
    )
    store = PineconeChunkStore(client)
    chunk = store.get("c1")
    assert chunk is not None
    assert chunk.chunk_id == "c1"
    assert chunk.heading is None  # sentinel "" round-trips back to None
    assert chunk.page is None     # sentinel -1 round-trips back to None


def test_get_missing_chunk_returns_none(mock_client):
    client, mock_index = mock_client
    mock_index.fetch.return_value = MagicMock(vectors={})
    store = PineconeChunkStore(client)
    assert store.get("missing") is None


def test_delete_by_document(mock_client):
    client, mock_index = mock_client
    store = PineconeChunkStore(client)
    store.delete_by_document("d1")
    mock_index.delete.assert_called_once_with(
        filter={"document_id": {"$eq": "d1"}},
    )

# get_by_document / get_document_hash / get_by_legal_metadata tests follow
# the same shape as test_delete_by_document's filter assertion, once Step 2's
# verification confirms the real query-without-a-real-vector call shape --
# write them to match that finding, not to the placeholder shape below.
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run python -m pytest tests/storage/test_pinecone_vector_store.py tests/storage/test_pinecone_chunk_store.py -v`
Expected: FAIL — `ModuleNotFoundError` (none of the three new modules exist yet).

- [ ] **Step 5: Write `PineconeClient`**

```python
# rag_hybrid_search/storage/pinecone_client.py
"""Shared Pinecone index connection, used by both PineconeVectorStore and
PineconeChunkStore so they operate against one connection per index, not one
each -- vector and chunk-metadata storage are two ABCs/two classes here, but
one real Pinecone index underneath (see the migration spec for why they
aren't merged into a single class despite sharing storage).
"""
from typing import Optional

from pinecone import Pinecone


class PineconeClient:
    def __init__(self, api_key: str, index_name: str, environment: Optional[str] = None):
        self._client = Pinecone(api_key=api_key)
        self.index = self._client.Index(index_name)
```

- [ ] **Step 6: Write `PineconeVectorStore`**

```python
# rag_hybrid_search/storage/pinecone_vector_store.py
from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.base import VectorStore
from rag_hybrid_search.storage.pinecone_client import PineconeClient


class PineconeVectorStore(VectorStore):
    def __init__(self, client: PineconeClient):
        self._index = client.index

    def upsert(self, chunk_id: str, embedding_record: EmbeddingRecord) -> None:
        self._index.upsert(vectors=[{"id": chunk_id, "values": embedding_record.embedding}])

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        result = self._index.query(vector=embedding, top_k=k, include_metadata=False)
        return [(m.id, m.score) for m in result.matches]

    def delete(self, chunk_ids: list[str]) -> None:
        self._index.delete(ids=chunk_ids)
```

Note: `upsert` here only sets vector values, using Pinecone's partial
`update`/`upsert` semantics so it doesn't clobber metadata written separately
by `PineconeChunkStore.put()` on the same record — confirm against the real
SDK whether `index.upsert(vectors=[{"id":..., "values":...}])` (no
`metadata` key) leaves existing metadata untouched or wipes it; if it wipes
it, switch this to `index.update(id=chunk_id, values=embedding_record.embedding)`
instead, matching the `update`-based approach `PineconeChunkStore.put` uses
below for the same reason.

- [ ] **Step 7: Write `PineconeChunkStore`**

```python
# rag_hybrid_search/storage/pinecone_chunk_store.py
"""PineconeChunkStore: metadata-only operations against the same Pinecone
index PineconeVectorStore writes vectors to (shared via PineconeClient).

Metadata sentinels: Optional[str] fields (heading, source_path) store "" for
None, Optional[int] (page) stores -1 for None -- both reconstructed back to
None on read, since Pinecone metadata doesn't reliably support None values
across all SDK/client versions.
"""
from typing import Iterator, Optional

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.pinecone_client import PineconeClient

_LEGAL_FILTER_KEYS = {
    "regulation", "version", "jurisdiction", "article", "section",
    "clause", "document_type",
}


def _chunk_to_metadata(chunk: Chunk, source_path: Optional[str] = None) -> dict:
    lm = chunk.legal_metadata
    metadata = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "strategy_version": chunk.strategy_version,
        "heading": chunk.heading or "",
        "page": chunk.page if chunk.page is not None else -1,
        "char_count": chunk.char_count,
        "source_path": source_path or "",
    }
    if lm:
        metadata.update({
            "legal_regulation": lm.regulation or "",
            "legal_version": lm.version or "",
            "legal_jurisdiction": lm.jurisdiction or "",
            "legal_article": lm.article or "",
            "legal_section": lm.section or "",
            "legal_clause": lm.clause or "",
            "legal_document_type": lm.document_type or "",
        })
    return metadata


def _metadata_to_chunk(chunk_id: str, metadata: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=metadata["document_id"],
        chunk_index=metadata["chunk_index"],
        text=metadata["text"],
        strategy_version=metadata["strategy_version"],
        heading=metadata["heading"] or None,
        page=metadata["page"] if metadata["page"] != -1 else None,
        char_count=metadata["char_count"],
    )


class PineconeChunkStore(ChunkStore):
    def __init__(self, client: PineconeClient):
        self._index = client.index

    def put(self, chunk: Chunk, source_path: Optional[str] = None) -> None:
        # update() (not upsert()) so this never touches vector values --
        # PineconeVectorStore.upsert() writes those independently, same
        # record, same index, via the shared client.
        self._index.update(id=chunk.chunk_id, set_metadata=_chunk_to_metadata(chunk, source_path))

    def get(self, chunk_id: str) -> Optional[Chunk]:
        result = self._index.fetch(ids=[chunk_id])
        if chunk_id not in result.vectors:
            return None
        return _metadata_to_chunk(chunk_id, result.vectors[chunk_id].metadata)

    def get_by_document(self, document_id: str) -> list[Chunk]:
        # Query-call shape (vector argument, if any) depends on Step 2's
        # verification finding -- write this once that's confirmed.
        raise NotImplementedError("implement per Step 2's verified query shape")

    def get_document_hash(self, source_path: str) -> Optional[str]:
        raise NotImplementedError("implement per Step 2's verified query shape")

    def delete_by_document(self, document_id: str) -> None:
        self._index.delete(filter={"document_id": {"$eq": document_id}})

    def all(self) -> Iterator[Chunk]:
        for id_batch in self._index.list():
            fetched = self._index.fetch(ids=id_batch)
            for chunk_id, vector in fetched.vectors.items():
                yield _metadata_to_chunk(chunk_id, vector.metadata)

    def get_by_legal_metadata(self, filters: dict[str, str]) -> list[Chunk]:
        raise NotImplementedError("implement per Step 2's verified query shape")

    def get_document_summaries(self) -> list[dict]:
        counts: dict[str, dict] = {}
        for chunk in self.all():
            key = chunk.document_id
            if key not in counts:
                counts[key] = {"document_id": chunk.document_id, "source_path": None, "chunk_count": 0}
            counts[key]["chunk_count"] += 1
        return sorted(counts.values(), key=lambda d: d["document_id"])
```

`get_by_document`, `get_document_hash`, `get_by_legal_metadata` are
deliberately left as `NotImplementedError` here rather than guessed-at code —
Step 2 of this task is what determines their real query-call shape ((a)/(b)/(c)
above). Filling these in with your Step 2 finding is the last part of this
task, not a follow-up task. `delete_by_document` and `all()` don't need a
similarity vector at all (`delete` takes a pure filter; `list()` is
filter/pagination-based), so those two are safe to implement without
Step 2's answer, which is why they're filled in above.

- [ ] **Step 8: Fill in the three deferred methods per Step 2's finding, then
  run tests to verify they pass**

Run: `uv run python -m pytest tests/storage/test_pinecone_vector_store.py tests/storage/test_pinecone_chunk_store.py -v`
Expected: all pass once the three methods are implemented and their tests
(added per the comment at the bottom of the chunk-store test file above) match.

- [ ] **Step 9: Run the `ChunkStore` contract test against `PineconeChunkStore`
  too**

Add to `tests/storage/test_chunk_store_contract.py`'s `IMPLEMENTATIONS` list:

```python
def _pinecone_chunk_store(tmp_path):
    from unittest.mock import MagicMock, patch
    patcher = patch("rag_hybrid_search.storage.pinecone_client.Pinecone")
    mock_pc_cls = patcher.start()
    mock_pc_cls.return_value.Index.return_value = MagicMock()
    from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
    from rag_hybrid_search.storage.pinecone_client import PineconeClient
    client = PineconeClient(api_key="k", index_name="idx")
    return PineconeChunkStore(client)


IMPLEMENTATIONS = [_sqlite_store, _pinecone_chunk_store]
```

Run: `uv run python -m pytest tests/storage/test_chunk_store_contract.py -v`
Expected: all pass for both implementations.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock rag_hybrid_search/storage/pinecone_client.py rag_hybrid_search/storage/pinecone_vector_store.py rag_hybrid_search/storage/pinecone_chunk_store.py tests/storage/test_pinecone_vector_store.py tests/storage/test_pinecone_chunk_store.py tests/storage/test_chunk_store_contract.py
git commit -m "feat: add PineconeVectorStore and PineconeChunkStore over a shared PineconeClient"
```

---

### Task 3: `RAG_STORAGE_BACKEND` config flag + `build_container` wiring

**Files:**
- Modify: `rag_hybrid_search/config.py`
- Modify: `api/dependencies.py:165-215` (`build_container`)
- Test: `tests/api/test_dependencies.py` (check if it exists first; if not,
  create it)

**Interfaces:**
- Consumes: `PineconeClient`, `PineconeVectorStore`, `PineconeChunkStore` (Task 2).
- Produces: `Settings.storage_backend: Literal["local", "pinecone"] = "local"`,
  `Settings.pinecone_api_key: Optional[str] = None`,
  `Settings.pinecone_index_name: Optional[str] = None`,
  `Settings.pinecone_environment: Optional[str] = None`. `build_container`
  branches on `settings.storage_backend`.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_dependencies.py (add to existing file, or create it)
from unittest.mock import MagicMock, patch

from api.dependencies import build_container
from rag_hybrid_search.config import Settings


def test_build_container_defaults_to_local_backend(tmp_path):
    settings = Settings(data_dir=str(tmp_path))
    container = build_container(settings)
    from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
    assert isinstance(container.index_manager.vector_store, ChromaVectorStore)


def test_build_container_wires_pinecone_backend(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path), storage_backend="pinecone",
        pinecone_api_key="k", pinecone_index_name="idx",
    )
    with patch("api.dependencies.PineconeClient") as mock_client_cls, \
         patch("api.dependencies.PineconeVectorStore") as mock_vs_cls, \
         patch("api.dependencies.PineconeChunkStore") as mock_cs_cls:
        mock_client_cls.return_value = MagicMock()
        mock_vs_cls.return_value = MagicMock()
        mock_cs_cls.return_value = MagicMock()
        container = build_container(settings)
        mock_client_cls.assert_called_once_with(
            api_key="k", index_name="idx", environment=None,
        )
        mock_vs_cls.assert_called_once_with(mock_client_cls.return_value)
        mock_cs_cls.assert_called_once_with(mock_client_cls.return_value)
        assert container.index_manager.vector_store is mock_vs_cls.return_value
        assert container.chunk_store is mock_cs_cls.return_value
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/api/test_dependencies.py -v`
Expected: first test passes (matches current default behavior), second FAILS
— `storage_backend` isn't a recognized `Settings` field yet (Pydantic extra-field
error) and `PineconeClient`/`PineconeVectorStore`/`PineconeChunkStore` aren't
imported in `api.dependencies`.

- [ ] **Step 3: Add config fields**

In `rag_hybrid_search/config.py`, add to the `Settings` class (near the
existing `provider`/`nvidia_api_key` fields):

```python
    storage_backend: Literal["local", "pinecone"] = "local"
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_sparse_index_name: Optional[str] = None
```

- [ ] **Step 4: Wire `build_container`**

In `api/dependencies.py`, add the imports near the existing storage imports:

```python
from rag_hybrid_search.storage.pinecone_client import PineconeClient
from rag_hybrid_search.storage.pinecone_vector_store import PineconeVectorStore
from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
```

Replace the block from `chunk_store = SqliteChunkStore(...)` through
`index_manager = IndexManager(chunk_store, vector_store, bm25_index)`
(currently `api/dependencies.py:176-184`):

```python
    if settings.storage_backend == "pinecone":
        pinecone_client = PineconeClient(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            environment=settings.pinecone_environment,
        )
        vector_store = PineconeVectorStore(pinecone_client)
        chunk_store = PineconeChunkStore(pinecone_client)
        bm25_index = BM25Index(index_path=str(data_dir / _BM25_INDEX_FILENAME))
        bm25_index.load()  # Phase 1: sparse still local even on pinecone backend; Task 6 replaces this line
    else:
        chunk_store = SqliteChunkStore(db_path=str(data_dir / _CHUNK_DB_FILENAME))
        vector_store = ChromaVectorStore(data_dir=str(data_dir / _CHROMA_DIRNAME))
        bm25_index = BM25Index(index_path=str(data_dir / _BM25_INDEX_FILENAME))
        bm25_index.load()
    index_manager = IndexManager(chunk_store, vector_store, bm25_index)
```

Note `chunk_store` and `vector_store` are now two distinct objects sharing
`pinecone_client` under the hood — not the same object twice, unlike an
earlier draft of this plan. `IndexManager` and every retriever above it
still only see the `ChunkStore`/`VectorStore` ABCs, unaware two Pinecone
wrapper classes exist.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/api/test_dependencies.py -v`
Expected: both pass.

Run the full existing API test suite to confirm the default path is
unaffected:

Run: `uv run python -m pytest tests/api/ -v`
Expected: all pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/config.py api/dependencies.py tests/api/test_dependencies.py
git commit -m "feat: add RAG_STORAGE_BACKEND flag, wire Pinecone vector/chunk stores into build_container"
```

**Checkpoint reached: indexing works.** At this point,
`RAG_STORAGE_BACKEND=pinecone` with real Pinecone credentials should let
`/index` and `/upload` write real documents into Pinecone (dense vectors +
metadata), verifiable by calling `GET /documents` against a real Pinecone
index. Sparse retrieval still uses local BM25 at this checkpoint (Phase 1 per
spec) — full retrieval isn't validated until Task 6.

---

### Task 3.5: Deploy dense-only Pinecone backend to Render, verify redeploy preserves data

**Files:** none — this is a deployment/verification task, not a code task.
Deliberately its own checkpoint, before any sparse-migration code is
written, so a problem found here is known to be about dense
vectors/metadata/indexing, not sparse — isolating the two migrations makes
debugging either one far easier than doing both before the first real
deploy.

- [ ] **Step 1: Set Render environment variables**

On the Render service, set `RAG_STORAGE_BACKEND=pinecone`,
`RAG_PINECONE_API_KEY`, `RAG_PINECONE_INDEX_NAME` (and
`RAG_PINECONE_ENVIRONMENT` if your account requires it). Leave generation
provider keys as they already are.

- [ ] **Step 2: Deploy and index a real document**

Deploy, then `POST /upload` a real test document, then `GET /documents` to
confirm it's listed with a nonzero `chunk_count`, then `POST /answer` with a
question the document actually answers — confirm citations resolve to real
chunk text (not empty/broken metadata).

- [ ] **Step 3: Trigger a manual redeploy, without re-indexing**

Redeploy the same Render service (no code change needed — a manual redeploy
trigger is enough). Once it's back up, repeat the same `/answer` question
from Step 2 with **no re-upload step in between**.

- [ ] **Step 4: Confirm**

The answer and citations from Step 3 must match Step 2's — proving the
Pinecone-backed index survived the redeploy with zero manual re-indexing.
This is a partial instance of the spec's overall success criterion (dense
only, sparse still pending) — full success is Task 7's job once sparse
migrates too.

No commit for this task.

---

### Task 4: Verify Pinecone hosted sparse embedding availability

**Files:** none — this is a research/spike task, not a code task. Its output
gates whether Task 5 uses the hosted-embedding path or the fallback path.

- [ ] **Step 1: Check SDK and account capability**

Run, against the real Pinecone account/API key intended for this project:

```bash
uv run python -c "import pinecone; print(pinecone.__version__)"
```

Then, following the current `pinecone` SDK's documentation for integrated
inference / hosted sparse embedding models (check the SDK's own docs at the
installed version — API surface for this feature has changed across SDK
major versions), attempt to create a small test index configured with a
hosted sparse embedding model, upsert one record with raw text, and query it
with raw text. Confirm:
1. The account/plan/region supports integrated inference for sparse models.
2. The exact method names for text-in upsert/query (e.g. `upsert_records`/
   `search`, or whatever the installed SDK version actually calls them).

- [ ] **Step 2: Record the outcome**

Write the result (available: yes/no, exact API method names and call shape
if yes) into a short note at the top of Task 5 before starting it, or
directly into `rag_hybrid_search/storage/pinecone_sparse_index.py`'s
module docstring once written. If unavailable, proceed with Task 5's
"Fallback" variant instead of its primary variant.

---

### Task 5: `PineconeSparseIndex` (sparse retrieval backend)

**Files:**
- Create: `rag_hybrid_search/storage/pinecone_sparse_index.py`
- Test: `tests/storage/test_pinecone_sparse_index.py`
- Modify: `api/dependencies.py` (replace the Task 3 placeholder BM25 line
  under the `pinecone` branch)

**Interfaces:**
- Produces: `PineconeSparseIndex` with the same shape `BM25Index` exposes to
  `SparseRetriever`: `search(query: str, k: int) -> list[tuple[str, float]]`,
  plus `upsert(chunk_id: str, text: str) -> None` and
  `delete(chunk_ids: list[str]) -> None` (called from `IndexManager`'s
  Pinecone path instead of `BM25Index.build()`/`.save()`).

**Primary variant (if Task 4 confirms hosted sparse embedding availability):**

- [ ] **Step 1: Write the failing tests**

```python
# tests/storage/test_pinecone_sparse_index.py
from unittest.mock import MagicMock, patch

import pytest

from rag_hybrid_search.storage.pinecone_sparse_index import PineconeSparseIndex


@pytest.fixture
def mock_sparse_index():
    with patch("rag_hybrid_search.storage.pinecone_sparse_index.Pinecone") as mock_pc_cls:
        mock_index = MagicMock()
        mock_pc_cls.return_value.Index.return_value = mock_index
        yield mock_index


def test_search_returns_chunk_id_score_pairs(mock_sparse_index):
    mock_sparse_index.search.return_value = MagicMock(
        result=MagicMock(hits=[
            MagicMock(_id="c1", _score=3.2),
            MagicMock(_id="c2", _score=1.1),
        ])
    )
    index = PineconeSparseIndex(api_key="k", index_name="sparse-idx")
    results = index.search("query text", k=5)
    assert results == [("c1", 3.2), ("c2", 1.1)]


def test_upsert_sends_raw_text(mock_sparse_index):
    index = PineconeSparseIndex(api_key="k", index_name="sparse-idx")
    index.upsert("c1", "hello world")
    mock_sparse_index.upsert_records.assert_called_once()


def test_delete(mock_sparse_index):
    index = PineconeSparseIndex(api_key="k", index_name="sparse-idx")
    index.delete(["c1", "c2"])
    mock_sparse_index.delete.assert_called_once_with(ids=["c1", "c2"])
```

**IMPORTANT — the mock shapes above (`search().result.hits`, `_id`, `_score`,
`upsert_records`) are this plan's best-current-knowledge guess at the
integrated-inference API surface, not a verified fact.** Task 4's Step 1 is
what confirms or corrects these exact method/attribute names — update this
test file to match what Task 4 actually found before treating these tests as
final.

- [ ] **Step 2-4: implement following the same TDD loop as Task 2** — write
  `PineconeSparseIndex` using the exact API confirmed in Task 4, run tests to
  fail then pass, following this plan's established pattern (see Task 2
  Steps 3-5 for the loop shape). No further placeholder code is provided
  here deliberately — Task 4's findings are the actual specification for
  this task's implementation, and writing speculative implementation code
  before that verification would risk shipping code against a guessed API.

**Fallback variant (if Task 4 finds hosted sparse embedding unavailable):**

- [ ] **Step 1: Add `pinecone-text` dependency**

```bash
uv add pinecone-text
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/storage/test_pinecone_sparse_index.py (fallback variant)
from unittest.mock import MagicMock, patch

import pytest

from rag_hybrid_search.storage.pinecone_sparse_index import PineconeSparseIndex


@pytest.fixture
def mock_sparse_backend():
    with patch("rag_hybrid_search.storage.pinecone_sparse_index.Pinecone") as mock_pc_cls, \
         patch("rag_hybrid_search.storage.pinecone_sparse_index.BM25Encoder") as mock_encoder_cls:
        mock_index = MagicMock()
        mock_pc_cls.return_value.Index.return_value = mock_index
        mock_encoder = MagicMock()
        mock_encoder_cls.default.return_value = mock_encoder
        mock_encoder.encode_queries.return_value = {"indices": [1, 2], "values": [0.5, 0.3]}
        yield mock_index, mock_encoder


def test_search_encodes_query_and_returns_pairs(mock_sparse_backend):
    mock_index, mock_encoder = mock_sparse_backend
    mock_index.query.return_value = MagicMock(
        matches=[MagicMock(id="c1", score=3.2), MagicMock(id="c2", score=1.1)]
    )
    index = PineconeSparseIndex(api_key="k", index_name="sparse-idx")
    results = index.search("query text", k=5)
    mock_encoder.encode_queries.assert_called_once_with("query text")
    assert results == [("c1", 3.2), ("c2", 1.1)]


def test_upsert_encodes_document_text(mock_sparse_backend):
    mock_index, mock_encoder = mock_sparse_backend
    mock_encoder.encode_documents.return_value = {"indices": [1, 2], "values": [0.5, 0.3]}
    index = PineconeSparseIndex(api_key="k", index_name="sparse-idx")
    index.upsert("c1", "hello world")
    mock_encoder.encode_documents.assert_called_once_with("hello world")
    mock_index.upsert.assert_called_once()
```

- [ ] **Step 3: Write the implementation**

```python
# rag_hybrid_search/storage/pinecone_sparse_index.py (fallback variant)
"""PineconeSparseIndex, fallback variant: client-side BM25Encoder fit on this
project's corpus (Task 4 found hosted sparse embedding unavailable for this
account/SDK). Matches BM25Index's search(query, k) -> list[tuple] shape.
"""
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder


class PineconeSparseIndex:
    def __init__(self, api_key: str, index_name: str):
        self._client = Pinecone(api_key=api_key)
        self._index = self._client.Index(index_name)
        self._encoder = BM25Encoder.default()

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        sparse_vector = self._encoder.encode_queries(query)
        result = self._index.query(sparse_vector=sparse_vector, top_k=k)
        return [(m.id, m.score) for m in result.matches]

    def upsert(self, chunk_id: str, text: str) -> None:
        sparse_vector = self._encoder.encode_documents(text)
        self._index.upsert(vectors=[{"id": chunk_id, "sparse_values": sparse_vector}])

    def delete(self, chunk_ids: list[str]) -> None:
        self._index.delete(ids=chunk_ids)
```

Note: `BM25Encoder.default()` uses pretrained generic params as a starting
point — refitting on this project's own corpus (`BM25Encoder().fit(corpus)`)
for accurate term weighting, as originally scoped in the spec before the
hosted-model revision, is the fallback's own follow-up if `.default()`'s
relevance proves too generic during Task 6's evaluation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/storage/test_pinecone_sparse_index.py -v`

**Both variants — Step 5: wire into `IndexManager`'s Pinecone path**

In `api/dependencies.py`, replace the Task 3 placeholder line
`bm25_index = BM25Index(...)` / `bm25_index.load()` under the `pinecone`
branch with:

```python
        sparse_index = PineconeSparseIndex(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_sparse_index_name,
        )
```

and pass `sparse_index` where `bm25_index` was passed to `IndexManager` and
`SparseRetriever` for this branch only — the `local` branch keeps using
`BM25Index` unchanged.

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/storage/pinecone_sparse_index.py tests/storage/test_pinecone_sparse_index.py api/dependencies.py
git commit -m "feat: add PineconeSparseIndex, complete pinecone backend wiring"
```

**Checkpoint reached: retrieval works.** `RAG_STORAGE_BACKEND=pinecone` now
exercises the full `DenseRetriever` + `SparseRetriever` + `weighted_rrf` +
reranker path against real Pinecone indexes end-to-end, with zero local
persistence.

---

### Task 6: Evaluation comparison (validation gate)

**Files:** none created — this is a manual validation task using existing
`scripts/run_eval.py` infrastructure (Phase 2 of this project's earlier eval
work), not new code.

- [ ] **Step 1: Establish the local-backend baseline**

```bash
RAG_STORAGE_BACKEND=local uv run python scripts/run_eval.py --update-baseline --baseline-name local --notes "pre-migration baseline"
```

- [ ] **Step 2: Run the pinecone backend against that baseline**

```bash
RAG_STORAGE_BACKEND=pinecone RAG_PINECONE_API_KEY=... RAG_PINECONE_INDEX_NAME=... \
  uv run python scripts/run_eval.py --compare-baseline --baseline-name local
```

- [ ] **Step 3: Check the gate**

Per the spec's success criteria: comparative-category accuracy not
regressed, hallucination rate not increased, verification pass rate not
decreased, latency not regressed beyond an acceptable margin. Exit code 1
means a regression was found — do not flip any default to `pinecone` until
this run is exit code 0.

- [ ] **Step 4: Manual sparse spot-check**

Run 3-5 keyword-heavy queries (exact terms/codes/IDs from the test corpus)
against both backends via `/answer`, compare citation quality by eye — this
catches sparse-relevance drift that aggregate eval metrics might average out.

- [ ] **Step 5: Verify the success criterion from the spec**

Deploy to a Render instance with `RAG_STORAGE_BACKEND=pinecone`, trigger a
manual redeploy, and confirm `/answer` queries succeed against the
previously-indexed corpus with no re-indexing step — this is the spec's
literal success criterion and the actual proof the original deployment
problem is fixed.

No commit for this task — it's a validation gate, not a code change. Once it
passes, `RAG_STORAGE_BACKEND=pinecone` is production-ready to use as your
Render deployment's configuration, though it stays behind the flag (not the
default) until you're confident enough to flip it — that flip, plus removing
Chroma/SQLite/local-BM25 entirely, is Phase 4, explicitly out of scope for
this plan per the spec.
