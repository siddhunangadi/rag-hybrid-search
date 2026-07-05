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
