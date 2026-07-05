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
