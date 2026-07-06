from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel

DocumentType = Literal["regulation", "policy", "contract", "standard", "guideline"]


class LegalMetadata(BaseModel):
    """Legal/structural metadata attached to a compliance document chunk.

    Every field except document_id/document_title is optional: a
    non-legal document ingested through the same pipeline gets an
    all-null LegalMetadata and behaves exactly as it does today.
    """

    document_id: str
    document_title: str
    regulation: Optional[str] = None
    version: Optional[str] = None
    jurisdiction: Optional[str] = None
    article: Optional[str] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    effective_date: Optional[date] = None
    document_type: Optional[DocumentType] = None
    page: Optional[int] = None


class ClauseSpan(BaseModel):
    """A single parsed clause: its text and the legal metadata locating it."""

    text: str
    metadata: LegalMetadata


class ClauseParseResult(BaseModel):
    """Output of clause_parser.parse(): all detected clauses plus a confidence score.

    confidence bands: 0.0-0.4 poor, 0.4-0.7 acceptable, 0.7-1.0 high confidence.
    """

    clauses: list[ClauseSpan]
    confidence: float
    parser: Literal["regex", "gemini"] = "regex"
    fallback_used: bool = False


class Citation(BaseModel):
    """A structured citation pointing at an exact clause (or, for non-legal
    documents, degrading gracefully to filename/chunk_id only)."""

    citation_id: str
    document_id: str
    document_title: str
    chunk_id: str
    confidence: float
    display: str
    regulation: Optional[str] = None
    version: Optional[str] = None
    jurisdiction: Optional[str] = None
    article: Optional[str] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    effective_date: Optional[date] = None
    document_type: Optional[DocumentType] = None
    page: Optional[int] = None
