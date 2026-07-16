"""Pydantic request/response models for the API layer."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from rag_hybrid_search.audit import AuditEvent

RiskCategoryParam = Literal["low", "medium", "high", "critical"]

_DEFAULT_MAX_CHUNKS = 5

DocumentTypeParam = Literal["general", "regulation", "policy", "contract", "standard", "guideline"]


class AnswerRequest(BaseModel):
    """Request body for POST /answer."""

    question: str = Field(..., min_length=1, description="Natural-language question to answer.")
    max_chunks: int = Field(
        default=_DEFAULT_MAX_CHUNKS, gt=0, description="Maximum number of retrieved chunks to use."
    )
    verify: bool = Field(default=True, description="Whether to run citation verification.")


class IndexDocument(BaseModel):
    """A single document to be ingested via POST /index."""

    filename: str = Field(..., min_length=1, description="Filename, including extension, used to pick a loader.")
    content: str = Field(..., description="Raw text content of the document.")
    document_type: DocumentTypeParam = Field(
        default="general",
        description="Document category. 'regulation' routes ingestion through the "
        "clause-aware compliance chunker instead of the default chunker.",
    )
    regulation: str | None = Field(
        default=None, description="Regulation name, e.g. 'RBI Master Circular'. Compliance documents only."
    )
    authority: str | None = Field(
        default=None, description="Issuing authority, e.g. 'RBI', 'SEBI'. Compliance documents only."
    )
    jurisdiction: str | None = Field(
        default=None, description="Jurisdiction, e.g. 'INDIA'. Compliance documents only."
    )
    effective_date: date | None = Field(
        default=None, description="Date the regulation takes/took effect. Compliance documents only."
    )
    risk_category: RiskCategoryParam | None = Field(
        default=None, description="Compliance risk category. Compliance documents only."
    )


class IndexRequest(BaseModel):
    """Request body for POST /index."""

    documents: list[IndexDocument] = Field(..., min_length=1, description="Documents to ingest.")


class DeleteDocumentResponse(BaseModel):
    """Response body for DELETE /documents/{document_id}."""

    document_id: str
    chunks_deleted: int


class UploadAcceptedResponse(BaseModel):
    """Response body for POST /upload/async: the upload was accepted, not yet processed."""

    job_id: str
    status: Literal["processing"] = "processing"


class IndexResult(BaseModel):
    """Per-document ingestion outcome."""

    filename: str
    status: str
    error: str | None = None


class IndexResponse(BaseModel):
    """Response body for POST /index."""

    results: list[IndexResult]


class JobStatusResponse(BaseModel):
    """Response body for GET /jobs/{job_id}."""

    job_id: str
    status: Literal["processing", "ready", "failed"]
    result: IndexResponse | None = None
    error: str | None = None


class DocumentSummary(BaseModel):
    """Chunk-count summary for a single indexed document."""

    document_id: str
    filename: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """Response body for GET /documents."""

    total_documents: int
    total_chunks: int
    documents: list[DocumentSummary]


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    generation_provider: str
    embedding_provider: str
    data_dir: str


class VersionResponse(BaseModel):
    """Response body for GET /version."""

    name: str
    version: str


class DebugRetrievedChunk(BaseModel):
    """One retrieved chunk with its stage-specific score, for GET /debug/retrieval."""

    chunk_id: str
    chunk_index: int
    score: float | None = None
    text: str


class DebugRetrievalResponse(BaseModel):
    """Response body for GET /debug/retrieval.

    Traces every pipeline stage for one query against already-indexed data,
    so retrieval-quality bugs can be localized to a specific stage without
    re-indexing or guessing from the final answer alone.
    """

    dense_results: list[DebugRetrievedChunk]
    bm25_results: list[DebugRetrievedChunk]
    rrf_results: list[DebugRetrievedChunk]
    rerank_results: list[DebugRetrievedChunk]
    prompt: str
    raw_generation: str


class AuditEventsResponse(BaseModel):
    """Response body for GET /audit/events."""

    events: list[AuditEvent]
    total: int
    offset: int
    limit: int


class LivenessResponse(BaseModel):
    """Response body for GET /health/live: the process is up. No dependency checks."""

    status: Literal["alive"] = "alive"


class ComponentStatus(BaseModel):
    """One dependency's readiness result, for GET /health/ready."""

    name: str
    ok: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Response body for GET /health/ready."""

    status: Literal["ready", "not_ready"]
    checks: list[ComponentStatus]


class MetricsResponse(BaseModel):
    """In-process operational counters, for GET /diagnostics."""

    counts: dict[str, int]
    avg_latency_ms: float
    request_count: int


class DiagnosticsResponse(BaseModel):
    """Response body for GET /diagnostics. Admin-only -- aggregates
    operational state useful for on-call debugging, no secrets."""

    build: dict
    providers: dict
    readiness: list[ComponentStatus]
    config: dict
    ingestion_stats: dict
    audit_stats: dict
    metrics: MetricsResponse
