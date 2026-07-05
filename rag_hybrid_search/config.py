from typing import Literal, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_")

    provider: Literal["nvidia", "gemini"] = "gemini"
    nvidia_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50

    dense_k: int = 10
    sparse_k: int = 10
    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    rrf_k: int = 60
    rerank_top_n: int = 5
    rerank_backend: Literal["passthrough", "cross_encoder", "nvidia"] = "passthrough"

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
        return self


def get_settings() -> Settings:
    return Settings()
