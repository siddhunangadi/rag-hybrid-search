from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from rag_hybrid_search.compliance.regulation_models import LegalMetadata


class Document(BaseModel):
    document_id: str
    source_path: str
    content: str
    format: Literal["markdown", "html", "text", "pdf", "csv", "xlsx", "docx"]


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    strategy_version: str
    heading: Optional[str] = None
    page: Optional[int] = None
    char_count: int
    legal_metadata: Optional[LegalMetadata] = None


class EmbeddingRecord(BaseModel):
    chunk_id: str
    embedding: list[float]
    embedding_model: str
    embedding_dimension: int
    provider: str
    created_at: datetime

    @model_validator(mode="after")
    def _check_dimension(self) -> "EmbeddingRecord":
        if len(self.embedding) != self.embedding_dimension:
            raise ValueError(
                f"embedding length {len(self.embedding)} != "
                f"embedding_dimension {self.embedding_dimension}"
            )
        return self


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float
    rerank_score: Optional[float] = None
    final_rank: int


class IndexStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class RetrievalTrace(BaseModel):
    dense_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    fusion_candidates: int = 0
    budget_applied: int = 0
    sent_to_reranker: int = 0
    returned: int = 0

    @property
    def total_latency_ms(self) -> float:
        return (
            self.dense_latency_ms
            + self.bm25_latency_ms
            + self.fusion_latency_ms
            + self.rerank_latency_ms
        )
