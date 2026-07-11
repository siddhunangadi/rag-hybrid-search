"""Live Pinecone integration tests -- exercise a real Pinecone index over the
network. Skipped by default (matching tests/rag_pipeline/test_live_providers.py's
convention: pytest.mark.skipif on an env var, not a custom marker), since these
need a real RAG_PINECONE_API_KEY/RAG_PINECONE_INDEX_NAME to run.

Scope: Phase 1 (dense + metadata) only, per the migration spec -- sparse
retrieval isn't part of this round trip until Task 5 lands PineconeSparseIndex.
Run manually before/during Task 3.5's Render verification:

    RAG_PINECONE_API_KEY=... RAG_PINECONE_INDEX_NAME=... \
      uv run python -m pytest tests/storage/test_pinecone_live.py -v

The configured index must be a dense index whose dimension matches
_EMBEDDING_DIMENSION below (adjust to match the real index's configured
dimension before running).
"""
import os
from datetime import datetime, timezone

import pytest

from rag_hybrid_search.models import Chunk, EmbeddingRecord
from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
from rag_hybrid_search.storage.pinecone_connection import PineconeConnection
from rag_hybrid_search.storage.pinecone_vector_store import PineconeVectorStore

_EMBEDDING_DIMENSION = 1024  # matches nvidia/nv-embedqa-e5-v5, this project's default embedding model


def _retry_until(action, predicate, attempts=15, delay_seconds=3.0):
    """Retry action() until predicate(result) is True or attempts run out --
    accommodates Pinecone's list() eventual consistency (see call site)."""
    import time

    result = action()
    for _ in range(attempts - 1):
        if predicate(result):
            return result
        time.sleep(delay_seconds)
        result = action()
    return result

pytestmark = pytest.mark.skipif(
    not os.environ.get("RAG_PINECONE_API_KEY") or not os.environ.get("RAG_PINECONE_INDEX_NAME"),
    reason="RAG_PINECONE_API_KEY/RAG_PINECONE_INDEX_NAME not set",
)


@pytest.fixture
def live_stores():
    connection = PineconeConnection(
        api_key=os.environ["RAG_PINECONE_API_KEY"],
        index_name=os.environ["RAG_PINECONE_INDEX_NAME"],
        environment=os.environ.get("RAG_PINECONE_ENVIRONMENT"),
    )
    chunk_store = PineconeChunkStore(connection, embedding_dimension=_EMBEDDING_DIMENSION)
    vector_store = PineconeVectorStore(connection)
    chunk_id = "live-test-chunk-1"
    yield chunk_store, vector_store, chunk_id
    # Best-effort cleanup so repeated runs don't accumulate test data in a
    # real index -- not asserted on, since the test itself already proved
    # delete() works if it reaches this point.
    vector_store.delete([chunk_id])


def test_put_upsert_query_get_delete_round_trip(live_stores):
    chunk_store, vector_store, chunk_id = live_stores
    chunk = Chunk(
        chunk_id=chunk_id, document_id="live-test-doc", chunk_index=0,
        text="Pinecone storage migration live round-trip test.",
        strategy_version="fixed-v1", char_count=48,
    )

    # PineconeChunkStore.put() first (real ingestion order): creates the
    # record with a placeholder vector.
    chunk_store.put(chunk, source_path="live-test.md")

    # PineconeVectorStore.upsert() second: sets the real vector values on
    # the record put() already created.
    embedding = [0.1] * _EMBEDDING_DIMENSION
    vector_store.upsert(chunk_id, EmbeddingRecord(
        chunk_id=chunk_id, embedding=embedding, embedding_model="test-model",
        embedding_dimension=_EMBEDDING_DIMENSION, provider="test",
        created_at=datetime.now(timezone.utc),
    ))

    # Dense retrieval: querying the same vector should surface this chunk.
    # query() freshness after upsert() also isn't instant against a real
    # serverless index (observed directly running this test), so it gets
    # the same retry treatment as list() below.
    results = _retry_until(
        lambda: vector_store.query(embedding, k=5),
        predicate=lambda matches: chunk_id in [c for c, _ in matches],
    )
    assert chunk_id in [chunk_id_ for chunk_id_, _ in results]

    # Metadata retrieval: get() reconstructs the Chunk from Pinecone metadata.
    fetched = _retry_until(
        lambda: chunk_store.get(chunk_id),
        predicate=lambda c: c is not None,
    )
    assert fetched is not None
    assert fetched.document_id == "live-test-doc"
    assert fetched.text == chunk.text

    # get_by_document scan: Pinecone's list() endpoint is eventually
    # consistent (unlike fetch()/query() above, which returned the freshly
    # written record immediately) -- a real index can take a few seconds
    # before list() surfaces a just-upserted id. Retrying with backoff here
    # is a test-timing accommodation for that lag, not a code path this
    # project's production ingestion depends on synchronously.
    by_document = _retry_until(
        lambda: chunk_store.get_by_document("live-test-doc"),
        predicate=lambda chunks: any(c.chunk_id == chunk_id for c in chunks),
    )
    assert any(c.chunk_id == chunk_id for c in by_document)

    # Delete removes it from both retrieval and metadata lookup. Delete is
    # also eventually consistent -- fetch() can briefly still return the
    # just-deleted record.
    chunk_store.delete_by_document("live-test-doc")
    remaining = _retry_until(
        lambda: chunk_store.get(chunk_id),
        predicate=lambda c: c is None,
    )
    assert remaining is None
