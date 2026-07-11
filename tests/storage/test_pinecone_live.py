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

_EMBEDDING_DIMENSION = 3

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
    embedding = [0.1, 0.2, 0.3]
    vector_store.upsert(chunk_id, EmbeddingRecord(
        chunk_id=chunk_id, embedding=embedding, embedding_model="test-model",
        embedding_dimension=_EMBEDDING_DIMENSION, provider="test",
        created_at=datetime.now(timezone.utc),
    ))

    # Dense retrieval: querying the same vector should surface this chunk.
    results = vector_store.query(embedding, k=5)
    assert chunk_id in [chunk_id_ for chunk_id_, _ in results]

    # Metadata retrieval: get() reconstructs the Chunk from Pinecone metadata.
    fetched = chunk_store.get(chunk_id)
    assert fetched is not None
    assert fetched.document_id == "live-test-doc"
    assert fetched.text == chunk.text

    # get_by_document scan.
    by_document = chunk_store.get_by_document("live-test-doc")
    assert any(c.chunk_id == chunk_id for c in by_document)

    # Delete removes it from both retrieval and metadata lookup.
    chunk_store.delete_by_document("live-test-doc")
    assert chunk_store.get(chunk_id) is None
