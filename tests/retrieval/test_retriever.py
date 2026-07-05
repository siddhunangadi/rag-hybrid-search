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
