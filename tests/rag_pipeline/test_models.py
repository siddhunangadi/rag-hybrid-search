from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag_pipeline.models import (
    Claim,
    ClaimResult,
    ConfidenceScores,
    GenerationMetadata,
    PromptContext,
    RagAnswer,
    RagAnswerDraft,
    VerificationReport,
)


def test_claim_roundtrip():
    claim = Claim(
        text="Employees get 20 days of paid leave.",
        citation_ids=["d1"],
        supporting_quote="20 days of paid annual leave",
    )
    assert claim.citation_ids == ["d1"]


def test_generation_metadata_roundtrip():
    metadata = GenerationMetadata(
        provider="mock",
        model="mock-v1",
        prompt_version="v1",
        generated_at=datetime.now(timezone.utc),
    )
    assert metadata.prompt_version == "v1"


def test_rag_answer_draft_roundtrip():
    metadata = GenerationMetadata(
        provider="mock",
        model="mock-v1",
        prompt_version="v1",
        generated_at=datetime.now(timezone.utc),
    )
    claim = Claim(text="x", citation_ids=["d1"], supporting_quote="x")
    draft = RagAnswerDraft(answer="Answer [d1].", claims=[claim], metadata=metadata)
    assert draft.claims[0].citation_ids == ["d1"]


def test_claim_result_and_verification_report():
    claim = Claim(text="x", citation_ids=["d1"], supporting_quote="x")
    result = ClaimResult(
        claim=claim, doc_ids_valid=True, quote_match_score=0.95, passed=True
    )
    report = VerificationReport(
        total_claims=1,
        verified_claims=1,
        failed_claims=0,
        hallucinated_doc_ids=[],
        missing_quotes=[],
        claim_results=[result],
    )
    assert report.verified_claims == 1
    assert report.claim_results[0].passed is True


def test_confidence_scores_roundtrip():
    scores = ConfidenceScores(retrieval=0.9, citations=1.0, coverage=0.8, overall=0.92)
    assert scores.overall == 0.92


def test_rag_answer_roundtrip():
    report = VerificationReport(
        total_claims=0,
        verified_claims=0,
        failed_claims=0,
        hallucinated_doc_ids=[],
        missing_quotes=[],
        claim_results=[],
    )
    scores = ConfidenceScores(retrieval=0.0, citations=0.0, coverage=0.0, overall=0.0)
    answer = RagAnswer(
        answer=None,
        citations=[],
        confidence=scores,
        verification=report,
        error="provider unavailable",
    )
    assert answer.error == "provider unavailable"


def test_prompt_context_roundtrip():
    context = PromptContext(text="[d1] some text", doc_id_map={"d1": "chunk-uuid-1"})
    assert context.doc_id_map["d1"] == "chunk-uuid-1"


def test_claim_requires_citation_ids_list():
    with pytest.raises(ValidationError):
        Claim(text="x", citation_ids="not-a-list", supporting_quote="x")
