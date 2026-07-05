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
