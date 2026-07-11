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


def test_misattributed_citation_id_is_flagged_not_silently_corrected():
    """The model sometimes copies a verbatim quote correctly but tags it
    with the wrong doc id. The verifier detects a better-matching doc but
    must never rewrite citation_ids -- it validates model output, it
    doesn't repair it. The claim fails with a distinct reason so the
    caller can decide what to do, and citation_ids stay exactly what the
    model wrote."""
    context = PromptContext(
        text=(
            "[d1]\nThe cafeteria serves lunch from 12pm to 2pm daily.\n\n"
            "[d2]\nEmployees get 20 days of paid annual leave per year."
        ),
        doc_id_map={"d1": "chunk-1", "d2": "chunk-2"},
    )
    claim = Claim(
        text="Employees get 20 days leave",
        citation_ids=["d1"],
        supporting_quote="20 days of paid annual leave",
    )
    report = verify_citations(make_draft([claim]), context)
    assert report.verified_claims == 0
    assert report.failed_claims == 1
    assert report.claim_results[0].passed is False
    assert report.claim_results[0].failure_reason == "citation_reattribution_candidate"
    assert report.claim_results[0].claim.citation_ids == ["d1"]


def test_quote_matches_despite_mid_sentence_pdf_linewrap_in_chunk():
    """Chunk text retains the source PDF's line-wrap newlines, so a
    sentence that happened to wrap mid-line has a literal '\\n' where the
    model's clean, single-space supporting_quote has a plain space. That
    must not be scored as a mismatch -- the quote is genuinely verbatim,
    just modulo whitespace layout noise."""
    context = PromptContext(
        text="[d1]\nOur objective in this work is to bridge these gaps through a systematic\ncomparative study of LLM-generated code detection.",
        doc_id_map={"d1": "chunk-1"},
    )
    claim = Claim(
        text="The paper's objective is a comparative study",
        citation_ids=["d1"],
        supporting_quote="Our objective in this work is to bridge these gaps through a systematic comparative study of LLM-generated code detection.",
    )
    report = verify_citations(make_draft([claim]), context)
    assert report.claim_results[0].passed is True
    assert report.claim_results[0].quote_match_score >= QUOTE_MATCH_THRESHOLD


def test_quote_matches_when_chunk_text_contains_an_internal_blank_line():
    """context_builder joins chunks with '\\n\\n', so _chunk_text_for_doc_id
    finds a chunk's end by looking for the next '\\n\\n'. But table chunks
    (rendered as 'prose\\n\\ntable rows') legitimately contain an internal
    blank line too -- naively splitting on the first '\\n\\n' truncates the
    chunk before the table rows the quote actually needs to match against."""
    context = PromptContext(
        text=(
            "[d1]\nPublication date: December 2025.\n\n"
            "Stage | Function-Level\n"
            "2. Initial Sample | 20,000 standalone functions\n\n"
            "[d2]\nEmployees get 20 days of paid annual leave per year."
        ),
        doc_id_map={"d1": "chunk-1", "d2": "chunk-2"},
    )
    claim = Claim(
        text="Initial sample was 20,000 functions",
        citation_ids=["d1"],
        supporting_quote="2. Initial Sample | 20,000 standalone functions",
    )
    report = verify_citations(make_draft([claim]), context)
    assert report.claim_results[0].passed is True
    assert report.claim_results[0].quote_match_score >= QUOTE_MATCH_THRESHOLD


def test_quote_crossing_d1_to_d2_boundary_fails_with_explicit_reason():
    """A quote concatenated from the end of d1's text and the start of d2's
    text scores low against either chunk alone (it's not fully present in
    either), but high against the whole concatenated context (it does
    exist there, split across the boundary). That combination is the
    fingerprint of a cross-chunk quote -- verify_citations must fail it
    with a distinct reason from a genuine hallucination."""
    context = PromptContext(
        text=(
            "[d1]\nThe most critical insight is that granularity dominates.\n\n"
            "[d2]\nWhile RQ2 establishes that structural detection is effective."
        ),
        doc_id_map={"d1": "chunk-1", "d2": "chunk-2"},
    )
    claim = Claim(
        text="Granularity dominates and RQ2 establishes detection is effective",
        citation_ids=["d1"],
        supporting_quote="The most critical insight is that granularity dominates. While RQ2 establishes that structural detection is effective.",
    )
    report = verify_citations(make_draft([claim]), context)
    assert report.claim_results[0].passed is False
    assert report.claim_results[0].failure_reason == "quote_spans_multiple_chunks"


def test_hallucinated_quote_not_present_anywhere_fails_with_distinct_reason():
    """A quote that doesn't exist in the cited chunk OR anywhere else in
    context is a plain hallucination, not a cross-chunk copy -- these two
    failure modes must be distinguishable."""
    claim = Claim(
        text="Employees get free lunch",
        citation_ids=["d1"],
        supporting_quote="completely invented text that appears nowhere",
    )
    report = verify_citations(make_draft([claim]), _CONTEXT)
    assert report.claim_results[0].passed is False
    assert report.claim_results[0].failure_reason == "quote_not_found"


def test_wrong_citation_id_fails_with_hallucinated_reason():
    claim = Claim(
        text="Employees get unlimited leave",
        citation_ids=["d99"],
        supporting_quote="unlimited leave",
    )
    report = verify_citations(make_draft([claim]), _CONTEXT)
    assert report.claim_results[0].failure_reason == "hallucinated_citation_id"


def test_quote_containing_other_citation_marker_fails_immediately():
    """If the quote itself still contains a literal '[d2]' marker while the
    claim cites d1, that's proof of a cross-boundary copy that didn't even
    strip the marker -- fail without needing to score it."""
    context = PromptContext(
        text="[d1]\nFirst chunk text.\n\n[d2]\nSecond chunk text.",
        doc_id_map={"d1": "chunk-1", "d2": "chunk-2"},
    )
    claim = Claim(
        text="Some claim",
        citation_ids=["d1"],
        supporting_quote="First chunk text. [d2]\nSecond chunk text.",
    )
    report = verify_citations(make_draft([claim]), context)
    assert report.claim_results[0].passed is False
    assert report.claim_results[0].failure_reason == "quote_contains_citation_marker"


def test_zero_claims_produces_empty_report():
    report = verify_citations(make_draft([]), _CONTEXT)
    assert report.total_claims == 0
    assert report.verified_claims == 0
    assert report.failed_claims == 0
    assert report.claim_results == []
