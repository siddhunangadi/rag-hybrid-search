from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.providers.base import RerankProvider


class PassthroughReranker(RerankProvider):
    """No-model reranker: truncates to ``top_n`` by existing RRF fusion order.

    Deliberately avoids importing ``sentence_transformers``/``torch`` (unlike
    ``CrossEncoderReranker`` in ``rerank.py``), so it can run in memory-constrained
    deployments (e.g. free-tier PaaS instances) without loading a cross-encoder
    model into memory. Retrieval quality is lower than a real reranker — it is
    a fallback, not a replacement.
    """

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
        top = ordered[:top_n]
        return [
            candidate.model_copy(update={"final_rank": rank})
            for rank, candidate in enumerate(top, start=1)
        ]
