"""Pydantic request/response models for the API layer."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

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
    error: Optional[str] = None


class IndexResponse(BaseModel):
    """Response body for POST /index."""

    results: list[IndexResult]


class JobStatusResponse(BaseModel):
    """Response body for GET /jobs/{job_id}."""

    job_id: str
    status: Literal["processing", "ready", "failed"]
    result: Optional[IndexResponse] = None
    error: Optional[str] = None


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
    score: Optional[float] = None
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
    prompt: str
    raw_generation: str
