import logging
import time

from rag_hybrid_search.models import RetrievalTrace, RetrievedChunk
from rag_hybrid_search.providers.base import RerankProvider
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.fusion import weighted_rrf
from rag_hybrid_search.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


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
        rerank_fused_top_n: int,
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
        self._rerank_fused_top_n = rerank_fused_top_n

    @property
    def dense_retriever(self) -> DenseRetriever:
        return self._dense_retriever

    @property
    def sparse_retriever(self) -> SparseRetriever:
        return self._sparse_retriever

    @property
    def dense_k(self) -> int:
        return self._dense_k

    @property
    def sparse_k(self) -> int:
        return self._sparse_k

    @property
    def dense_weight(self) -> float:
        return self._dense_weight

    @property
    def sparse_weight(self) -> float:
        return self._sparse_weight

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    @property
    def rerank_fused_top_n(self) -> int:
        return self._rerank_fused_top_n

    def retrieve(self, query: str, dev_trace=None) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        logger.info("retrieve: start query=%r", query)
        trace = RetrievalTrace()

        start = time.perf_counter()
        dense_results = self._dense_retriever.search(query, k=self._dense_k, trace=dev_trace)
        trace.dense_latency_ms = (time.perf_counter() - start) * 1000
        logger.info("retrieve: dense search k=%d hits=%d latency_ms=%.1f", self._dense_k, len(dense_results), trace.dense_latency_ms)
        logger.debug("retrieve: dense hits %s", [(r.chunk.chunk_id, r.dense_score) for r in dense_results])
        if dev_trace is not None:
            dev_trace.log_dense(dense_results, trace.dense_latency_ms)

        start = time.perf_counter()
        sparse_results = self._sparse_retriever.search(query, k=self._sparse_k)
        trace.bm25_latency_ms = (time.perf_counter() - start) * 1000
        logger.info("retrieve: bm25 search k=%d hits=%d latency_ms=%.1f", self._sparse_k, len(sparse_results), trace.bm25_latency_ms)
        logger.debug("retrieve: bm25 hits %s", [(r.chunk.chunk_id, r.bm25_score) for r in sparse_results])
        if dev_trace is not None:
            bm25_pairs = [(r.chunk.chunk_id, r.bm25_score) for r in sparse_results]
            dev_trace.log_bm25(bm25_pairs, trace.bm25_latency_ms)

        start = time.perf_counter()
        fused = weighted_rrf(
            dense_results,
            sparse_results,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
            k=self._rrf_k,
        )
        trace.fusion_latency_ms = (time.perf_counter() - start) * 1000
        trace.fusion_candidates = len(fused)
        trace.budget_applied = self._rerank_fused_top_n
        logger.info(
            "retrieve: fusion (rrf_k=%d, dense_weight=%.2f, sparse_weight=%.2f) produced %d candidates latency_ms=%.1f",
            self._rrf_k, self._dense_weight, self._sparse_weight, len(fused), trace.fusion_latency_ms,
        )
        logger.debug("retrieve: fused candidates %s", [(r.chunk.chunk_id, r.rrf_score) for r in fused])
        if dev_trace is not None:
            dev_trace.log_fusion(
                fused,
                dense_ids_ranked=[r.chunk.chunk_id for r in dense_results],
                bm25_ids_ranked=[r.chunk.chunk_id for r in sparse_results],
                rrf_k=self._rrf_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
                latency_ms=trace.fusion_latency_ms,
            )

        start = time.perf_counter()
        budgeted = fused[: self._rerank_fused_top_n]
        reranked = self._rerank_provider.rerank(query, budgeted, top_n=self._rerank_top_n)
        trace.rerank_latency_ms = (time.perf_counter() - start) * 1000
        trace.sent_to_reranker = len(budgeted)
        trace.returned = len(reranked)
        logger.info(
            "retrieve: rerank (top_n=%d, fused_budget=%d) via provider=%s returned %d results latency_ms=%.1f",
            self._rerank_top_n, self._rerank_fused_top_n, type(self._rerank_provider).__name__,
            len(reranked), trace.rerank_latency_ms,
        )
        logger.debug("retrieve: reranked results %s", [(r.chunk.chunk_id, r.rerank_score, r.final_rank) for r in reranked])
        logger.info("retrieve: done total_latency_ms=%.1f", trace.total_latency_ms)
        if dev_trace is not None:
            dev_trace.log_rerank(
                type(self._rerank_provider).__name__, budgeted, reranked,
                trace.rerank_latency_ms, self._rerank_fused_top_n,
            )

        return reranked, trace
