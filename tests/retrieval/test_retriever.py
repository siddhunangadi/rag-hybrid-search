from datetime import datetime, timezone

import pytest

from rag_hybrid_search.models import Chunk, EmbeddingRecord
from rag_hybrid_search.providers.base import RerankProvider
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.rerank import CrossEncoderReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from tests.fakes import FakeEmbeddingProvider


class RecordingRerankProvider(RerankProvider):
    """Test double for RerankProvider that records what it was called with.

    Unlike CrossEncoderReranker, it does not re-score candidates by semantic
    relevance -- it preserves the incoming (fused) order when passing
    candidates through, so tests can observe fusion/plumbing behavior that
    would otherwise be masked by the cross-encoder's own scoring. It can
    also be configured to raise, to test error propagation.
    """

    def __init__(self, exception: "Exception | None" = None):
        self.received_candidates = None
        self._exception = exception

    def rerank(self, query, candidates, top_n):
        self.received_candidates = candidates
        if self._exception is not None:
            raise self._exception
        top = candidates[:top_n]
        return [
            candidate.model_copy(update={"final_rank": rank})
            for rank, candidate in enumerate(top, start=1)
        ]


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


def build_retriever(tmp_path, docs, rerank_provider, **kwargs):
    """Build a HybridRetriever against a small fixture of chunks.

    Mirrors the setup used by the `hybrid_retriever` fixture below, but
    parameterized so individual tests can control the chunk set,
    rerank_provider, and HybridRetriever constructor kwargs.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    provider = FakeEmbeddingProvider()
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))

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

    defaults = dict(
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60,
        dense_k=10,
        sparse_k=10,
        rerank_top_n=10,
        rerank_fused_top_n=20,
    )
    defaults.update(kwargs)
    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        rerank_provider=rerank_provider,
        **defaults,
    )


@pytest.fixture
def hybrid_retriever(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        rerank_fused_top_n=10,
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


def test_retrieve_records_fusion_and_rerank_latency(hybrid_retriever):
    """Directly assert fusion/rerank timings, not just their sum.

    `total_latency_ms` is a computed property that always sums the four
    fields correctly by construction, so asserting only the sum (as the
    original test did) does not prove fusion_latency_ms/rerank_latency_ms
    were actually measured -- they could be 0 and the sum would still
    check out against dense+bm25. Assert them individually here.
    """
    _, trace = hybrid_retriever.retrieve("ERROR_CODE_0x834")

    assert trace.fusion_latency_ms > 0
    assert trace.rerank_latency_ms > 0


def test_dense_and_sparse_weights_affect_final_ranking(tmp_path):
    """dense_weight/sparse_weight must actually be wired through fusion.

    Fixture built so dense and sparse retrievers disagree on the top
    result for the query "quick fox":
      - c0 "quikk quikk quikk foxx foxx foxx" has no exact word overlap
        with the query (BM25 score 0) but shares many character trigrams
        with "quick"/"fox", so it wins on dense cosine similarity.
      - c1 "quick fox jumps over lazy river bank today morning" has an
        exact word match for both query terms, so it wins on BM25.

    A RecordingRerankProvider is used instead of CrossEncoderReranker so
    that the real semantic reranker's own scoring (which is orthogonal to
    fusion weights and would otherwise decide the final top-1 regardless
    of how dense_weight/sparse_weight are set) doesn't mask the fusion
    behavior under test.
    """
    docs = [
        make_chunk("c0", "quikk quikk quikk foxx foxx foxx"),
        make_chunk("c1", "quick fox jumps over lazy river bank today morning"),
        make_chunk("c2", "zzz zzz zzz zzz zzz"),
    ]

    dense_heavy = build_retriever(
        tmp_path / "dense_heavy",
        docs,
        RecordingRerankProvider(),
        dense_weight=0.95,
        sparse_weight=0.05,
    )
    sparse_heavy = build_retriever(
        tmp_path / "sparse_heavy",
        docs,
        RecordingRerankProvider(),
        dense_weight=0.05,
        sparse_weight=0.95,
    )

    dense_heavy_results, _ = dense_heavy.retrieve("quick fox")
    sparse_heavy_results, _ = sparse_heavy.retrieve("quick fox")

    assert dense_heavy_results[0].chunk.chunk_id == "c0"
    assert sparse_heavy_results[0].chunk.chunk_id == "c1"
    assert dense_heavy_results[0].chunk.chunk_id != sparse_heavy_results[0].chunk.chunk_id


def test_dense_k_sparse_k_and_rerank_top_n_are_plumbed_through(tmp_path):
    """dense_k/sparse_k/rerank_top_n must reach the right components.

    Uses a RecordingRerankProvider to inspect exactly what candidate list
    the reranker receives (proving dense_k/sparse_k reached the retrievers
    and their fused output reached rerank()), and asserts the returned
    result list is capped at rerank_top_n even though fusion produced more
    candidates than that.
    """
    docs = [
        make_chunk("c0", "alpha document about search"),
        make_chunk("c1", "beta document about search"),
        make_chunk("c2", "gamma document about search"),
        make_chunk("c3", "delta document about search"),
    ]
    reranker = RecordingRerankProvider()
    retriever = build_retriever(
        tmp_path,
        docs,
        reranker,
        dense_weight=0.5,
        sparse_weight=0.5,
        dense_k=2,
        sparse_k=2,
        rerank_top_n=1,
    )

    results, _ = retriever.retrieve("document about search")

    # dense_k=2 and sparse_k=2 over 4 docs, deduped by fusion, should hand
    # the reranker somewhere between 2 and 4 candidates -- never all 4
    # docs unfiltered, and never fewer than dense_k.
    assert reranker.received_candidates is not None
    assert 2 <= len(reranker.received_candidates) <= 4

    # rerank_top_n=1 must cap the final output even though there were
    # multiple fused candidates available.
    assert len(results) == 1


def test_retrieve_propagates_rerank_provider_exception(tmp_path):
    """A rerank() failure must propagate, not be swallowed into a partial result."""
    docs = [
        make_chunk("c0", "alpha document about search"),
        make_chunk("c1", "beta document about search"),
    ]
    reranker = RecordingRerankProvider(exception=RuntimeError("rerank boom"))
    retriever = build_retriever(tmp_path, docs, reranker)

    with pytest.raises(RuntimeError, match="rerank boom"):
        retriever.retrieve("document about search")


def test_rerank_fused_top_n_caps_candidates_sent_to_reranker(tmp_path):
    """rerank_fused_top_n must truncate the fused candidate list before it
    reaches the reranker, independent of rerank_top_n (which caps the
    reranker's own *output*, not its input)."""
    docs = [make_chunk(f"c{i}", f"document number {i} about search") for i in range(6)]
    reranker = RecordingRerankProvider()
    retriever = build_retriever(
        tmp_path, docs, reranker,
        dense_weight=0.5, sparse_weight=0.5,
        dense_k=6, sparse_k=6, rerank_top_n=1, rerank_fused_top_n=3,
    )

    results, _ = retriever.retrieve("document about search")

    assert reranker.received_candidates is not None
    assert len(reranker.received_candidates) == 3
    assert len(results) == 1
