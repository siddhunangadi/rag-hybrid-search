from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from rag_hybrid_search.compliance.regulation_models import Citation


class Claim(BaseModel):
    text: str
    citation_ids: list[str]
    supporting_quote: str


class GenerationMetadata(BaseModel):
    provider: str
    model: str
    prompt_version: str
    generated_at: datetime


class RagAnswerDraft(BaseModel):
    answer: str
    claims: list[Claim]
    metadata: GenerationMetadata


class ClaimResult(BaseModel):
    claim: Claim
    doc_ids_valid: bool
    quote_match_score: float
    passed: bool


class VerificationReport(BaseModel):
    total_claims: int
    verified_claims: int
    failed_claims: int
    hallucinated_doc_ids: list[str]
    missing_quotes: list[str]
    claim_results: list[ClaimResult]


class ConfidenceScores(BaseModel):
    retrieval: float
    citations: float
    coverage: float
    overall: float


class RagAnswer(BaseModel):
    answer: Optional[str]
    citations: list[str]
    structured_citations: list[Citation] = []
    confidence: ConfidenceScores
    verification: VerificationReport
    error: Optional[str] = None


class PromptContext(BaseModel):
    text: str
    doc_id_map: dict[str, str]
