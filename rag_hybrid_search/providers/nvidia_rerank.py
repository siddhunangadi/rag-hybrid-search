import httpx

from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.providers.base import RerankProvider

# NVIDIA's hosted NeMo Retriever reranking API. NOTE: this contract is based on
# published NVIDIA docs, not a verified live call (no API key was available at
# implementation time) — smoke-test against a real key before relying on it in
# production, same as any other unverified external integration.
_RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"


class NvidiaRerankProvider(RerankProvider):
    """Reranks candidates via NVIDIA's hosted reranking API (no local model).

    Opt-in alternative to ``CrossEncoderReranker`` (local torch model) and
    ``PassthroughReranker`` (no reranking) — select via
    ``RAG_RERANK_BACKEND=nvidia``. Requires ``RAG_NVIDIA_API_KEY``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/rerank-qa-mistral-4b",
        timeout: float = 60.0,
    ):
        self._model = model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        response = self._client.post(
            _RERANK_URL,
            json={
                "model": self._model,
                "query": {"text": query},
                "passages": [{"text": c.chunk.text} for c in candidates],
                "truncate": "END",
            },
        )
        response.raise_for_status()
        rankings = response.json()["rankings"]

        ranked = sorted(rankings, key=lambda r: r["logit"], reverse=True)[:top_n]
        return [
            candidates[entry["index"]].model_copy(
                update={"rerank_score": float(entry["logit"]), "final_rank": rank}
            )
            for rank, entry in enumerate(ranked, start=1)
        ]
