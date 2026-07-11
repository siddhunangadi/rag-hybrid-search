from typing import Literal, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_")

    provider: Literal["nvidia", "gemini"] = "gemini"
    nvidia_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    # Overrides NvidiaProvider's own default generation model (currently the
    # 70B model, which dominates request latency). Unset -> NvidiaProvider's
    # built-in default, unchanged behavior.
    generation_model: Optional[str] = None

    # Gates GET /debug/retrieval, which exposes raw indexed chunk text and
    # full prompts. Unset (default) -> endpoint returns 404. Set this to a
    # random secret and pass it as the X-Debug-Token header to enable it.
    debug_token: Optional[str] = None

    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 500
    # 150 (not the original 50): a sentence longer than the overlap window
    # that straddles a chunk boundary gets truncated in both neighboring
    # chunks, so its supporting_quote can never verify against either chunk
    # alone even when the model's synthesis is accurate. A larger overlap
    # doesn't eliminate this (a sentence longer than chunk_overlap can still
    # split), but it substantially reduces how often it happens in practice.
    chunk_overlap: int = 150

    dense_k: int = 10
    sparse_k: int = 10
    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    rrf_k: int = 60
    rerank_top_n: int = 5
    # Fused RRF output is truncated to this many top-scored candidates
    # before being sent to the reranker -- the reranker is the dominant
    # latency cost, and most fused candidates never survive rerank_top_n
    # anyway. dense_k/sparse_k stay wide for RRF diversity; only what
    # reaches the expensive reranker call is trimmed.
    rerank_fused_top_n: int = 8
    rerank_backend: Literal["passthrough", "cross_encoder", "nvidia"] = "passthrough"
    # After reranking, chunks scoring more than this fraction of the
    # top-to-bottom score *range* below the top chunk are dropped before
    # being sent to the LLM -- fewer chunks in the prompt when the reranker
    # is confident one chunk clearly answers the question, without lowering
    # rerank_top_n itself (which would also cap genuinely multi-hop
    # questions that need several chunks). Only applies when rerank_score is
    # populated (nvidia/cross_encoder backends); a no-op under
    # PassthroughReranker, which never scores candidates.
    context_prune_margin: float = 0.3

    dedup_cosine_threshold: float = 0.95
    dedup_text_similarity_threshold: float = 0.9

    data_dir: str = "./data"

    @model_validator(mode="after")
    def _validate_weights_and_k(self) -> "Settings":
        if not (0.0 <= self.rrf_dense_weight <= 1.0):
            raise ValueError("rrf_dense_weight must be in [0, 1]")
        if not (0.0 <= self.rrf_sparse_weight <= 1.0):
            raise ValueError("rrf_sparse_weight must be in [0, 1]")
        if abs(self.rrf_dense_weight + self.rrf_sparse_weight - 1.0) > 1e-6:
            raise ValueError(
                "rrf_dense_weight + rrf_sparse_weight must sum to 1.0"
            )
        if self.rerank_top_n > self.dense_k + self.sparse_k:
            raise ValueError("rerank_top_n cannot exceed dense_k + sparse_k")
        if self.rerank_fused_top_n > self.dense_k + self.sparse_k:
            raise ValueError("rerank_fused_top_n cannot exceed dense_k + sparse_k")
        if self.rerank_top_n > self.rerank_fused_top_n:
            raise ValueError("rerank_top_n cannot exceed rerank_fused_top_n")
        return self


def get_settings() -> Settings:
    return Settings()
