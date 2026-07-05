from datetime import datetime, timezone

from rag_pipeline.citation_verifier import QUOTE_MATCH_THRESHOLD, verify_citations
from rag_pipeline.models import Claim, GenerationMetadata, PromptContext, RagAnswerDraft

_METADATA = GenerationMetadata(
    provider="mock", model="mock-v1", prompt_version="v1",
    generated_at=datetime.now(timezone.utc),
)

_CONTEXT = PromptContext(
    text="[d1]\nEmployees get 20 days of paid annual leave per year.",
    doc_id_map={"d1": "chunk-1"},
)


def make_draft(claims):
    return RagAnswerDraft(answer="answer", claims=claims, metadata=_METADATA)


def test_valid_citation_and_matching_quote_passes():
    claim = Claim(
        text="Employees get 20 days leave",
        citation_ids=["d1"],
        supporting_quote="20 days of paid annual leave",
    )
    report = verify_citations(make_draft([claim]), _CONTEXT)
    assert report.total_claims == 1
    assert report.verified_claims == 1
    assert report.failed_claims == 0
    assert report.claim_results[0].passed is True
    assert report.claim_results[0].doc_ids_valid is True


def test_hallucinated_doc_id_fails():
    claim = Claim(
        text="Employees get unlimited leave",
        citation_ids=["d99"],
        supporting_quote="unlimited leave",
    )
    report = verify_citations(make_draft([claim]), _CONTEXT)
    assert report.verified_claims == 0
    assert report.failed_claims == 1
    assert "d99" in report.hallucinated_doc_ids
    assert report.claim_results[0].doc_ids_valid is False


def test_missing_quote_fails_even_with_valid_doc_id():
    claim = Claim(
        text="Employees get free lunch",
        citation_ids=["d1"],
        supporting_quote="completely unrelated text about lunch",
    )
    report = verify_citations(make_draft([claim]), _CONTEXT)
    assert report.verified_claims == 0
    assert report.failed_claims == 1
    assert len(report.missing_quotes) == 1
    assert report.claim_results[0].quote_match_score < QUOTE_MATCH_THRESHOLD


def test_multiple_claims_mixed_pass_fail():
    valid_claim = Claim(
        text="20 days leave", citation_ids=["d1"],
        supporting_quote="20 days of paid annual leave",
    )
    invalid_claim = Claim(
        text="unlimited leave", citation_ids=["d99"], supporting_quote="unlimited",
    )
    report = verify_citations(make_draft([valid_claim, invalid_claim]), _CONTEXT)
    assert report.total_claims == 2
    assert report.verified_claims == 1
    assert report.failed_claims == 1


def test_zero_claims_produces_empty_report():
    report = verify_citations(make_draft([]), _CONTEXT)
    assert report.total_claims == 0
    assert report.verified_claims == 0
    assert report.failed_claims == 0
    assert report.claim_results == []
