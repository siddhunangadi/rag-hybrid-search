from datetime import datetime, timezone

import pytest

from rag_hybrid_search.models import Chunk, EmbeddingRecord, IndexStatus
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.index_manager import IndexManager
from tests.fakes import fake_pinecone_stores


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
    chunk_store, vector_store = fake_pinecone_stores(embedding_dimension=3)
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
