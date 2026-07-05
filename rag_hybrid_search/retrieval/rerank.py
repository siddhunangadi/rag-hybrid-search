from sentence_transformers import CrossEncoder

from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.providers.base import RerankProvider


class CrossEncoderReranker(RerankProvider):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        top = scored[:top_n]
        return [
            candidate.model_copy(update={"rerank_score": float(score), "final_rank": rank})
            for rank, (candidate, score) in enumerate(top, start=1)
        ]
