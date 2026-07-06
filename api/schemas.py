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


class IndexResult(BaseModel):
    """Per-document ingestion outcome."""

    filename: str
    status: str
    error: Optional[str] = None


class IndexResponse(BaseModel):
    """Response body for POST /index."""

    results: list[IndexResult]


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
