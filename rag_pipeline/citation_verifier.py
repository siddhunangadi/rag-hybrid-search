import re
from difflib import SequenceMatcher

from rag_pipeline.citation_utils import chunk_text_for_doc_id
from rag_pipeline.models import (
    ClaimResult,
    PromptContext,
    RagAnswerDraft,
    VerificationReport,
)

QUOTE_MATCH_THRESHOLD = 0.80

_CITATION_MARKER_RE = re.compile(r"\[d\d+\]")


def _verify_claim(claim, context: PromptContext) -> ClaimResult:
    """Validate one claim's citations and supporting quote.

    Pure per-claim logic; report-level aggregation (hallucinated ids,
    missing quotes) is derived from the returned ClaimResult by the caller.
    """
    doc_ids_valid = all(
        citation_id in context.doc_id_map for citation_id in claim.citation_ids
    )

    # A quote that still contains a literal "[dN]" marker is proof the
    # model (or an old prompt version) copied text spanning a citation
    # boundary without even stripping the marker -- fail immediately,
    # no need to score it.
    if doc_ids_valid and claim.supporting_quote and _CITATION_MARKER_RE.search(claim.supporting_quote):
        return ClaimResult(
            claim=claim, doc_ids_valid=doc_ids_valid, quote_match_score=0.0,
            passed=False, failure_reason="quote_contains_citation_marker",
        )

    best_quote_score = 0.0
    if doc_ids_valid:
        for citation_id in claim.citation_ids:
            chunk_text = chunk_text_for_doc_id(context, citation_id)
            score = _quote_containment_score(claim.supporting_quote, chunk_text)
            best_quote_score = max(best_quote_score, score)

    # The model occasionally copies a supporting_quote verbatim but
    # tags it with the wrong doc id. A better-matching doc might exist
    # in context -- but the verifier never repairs evidence, only
    # validates it. Detecting this case and still failing the claim
    # (with a distinct, actionable reason) is intentional: it favors
    # evidence integrity over answer recovery. Rewriting citation_ids
    # here would mean the caller sees a citation the model never
    # actually wrote.
    reattribution_doc_id = None
    if doc_ids_valid and best_quote_score < QUOTE_MATCH_THRESHOLD:
        best_doc_id, best_doc_score = None, best_quote_score
        for doc_id in context.doc_id_map:
            if doc_id in claim.citation_ids:
                continue
            chunk_text = chunk_text_for_doc_id(context, doc_id)
            score = _quote_containment_score(claim.supporting_quote, chunk_text)
            if score > best_doc_score:
                best_doc_id, best_doc_score = doc_id, score
        if best_doc_id is not None and best_doc_score >= QUOTE_MATCH_THRESHOLD:
            reattribution_doc_id = best_doc_id

    passed = (
        doc_ids_valid
        and best_quote_score >= QUOTE_MATCH_THRESHOLD
        and reattribution_doc_id is None
    )

    failure_reason = None
    if not doc_ids_valid:
        failure_reason = "hallucinated_citation_id"
    elif reattribution_doc_id is not None:
        failure_reason = "citation_reattribution_candidate"
    elif not passed:
        # A quote that scores low against its own cited chunk but scores
        # high against all cited-context chunks concatenated (markers
        # stripped, chunks joined with a plain space) is the signature
        # of a quote that spans multiple chunks: it exists in the
        # document as a whole, just split across a chunk boundary.
        # Using context.text directly would miss this -- the literal
        # "[dN]" marker between chunks breaks contiguous matching right
        # at the boundary, making a genuinely cross-chunk quote score
        # just as low against the whole raw context as a true
        # hallucination would.
        all_chunks_concat = " ".join(
            chunk_text_for_doc_id(context, doc_id) for doc_id in context.doc_id_map
        )
        whole_context_score = _quote_containment_score(claim.supporting_quote, all_chunks_concat)
        failure_reason = (
            "quote_spans_multiple_chunks"
            if whole_context_score >= QUOTE_MATCH_THRESHOLD
            else "quote_not_found"
        )

    return ClaimResult(
        claim=claim,
        doc_ids_valid=doc_ids_valid,
        quote_match_score=best_quote_score,
        passed=passed,
        failure_reason=failure_reason,
    )


def verify_citations(
    draft: RagAnswerDraft, context: PromptContext
) -> VerificationReport:
    claim_results: list[ClaimResult] = []
    hallucinated_doc_ids: list[str] = []
    missing_quotes: list[str] = []

    for claim in draft.claims:
        result = _verify_claim(claim, context)
        claim_results.append(result)
        if not result.doc_ids_valid:
            hallucinated_doc_ids.extend(
                citation_id for citation_id in claim.citation_ids
                if citation_id not in context.doc_id_map
            )
        # Every failure except a hallucinated id means the quote itself
        # couldn't be validated against the cited evidence.
        if result.failure_reason is not None and result.failure_reason != "hallucinated_citation_id":
            missing_quotes.append(claim.supporting_quote)

    verified = sum(1 for r in claim_results if r.passed)
    return VerificationReport(
        total_claims=len(claim_results),
        verified_claims=verified,
        failed_claims=len(claim_results) - verified,
        hallucinated_doc_ids=hallucinated_doc_ids,
        missing_quotes=missing_quotes,
        claim_results=claim_results,
    )


def _quote_containment_score(quote: str, chunk_text: str) -> float:
    """Fuzzy "is this quote present in the chunk" score.

    A plain SequenceMatcher(None, quote, chunk_text).ratio() penalizes exact
    substrings of a much longer chunk (2*M/T shrinks as the chunk grows),
    which misclassifies genuine verbatim quotes as failures. Instead, score
    on the longest matching block relative to the quote's own length: an
    exact substring match scores close to 1.0 regardless of surrounding
    chunk length, while unrelated text scores low.
    """
    if not quote:
        return 0.0
    normalized_quote = _normalize_whitespace(quote)
    normalized_chunk = _normalize_whitespace(chunk_text)
    matcher = SequenceMatcher(None, normalized_quote, normalized_chunk)
    match = matcher.find_longest_match(0, len(normalized_quote), 0, len(normalized_chunk))
    return match.size / len(normalized_quote)


def _normalize_whitespace(text: str) -> str:
    """Collapse PDF line-wrap newlines/runs of whitespace to a single space
    so verbatim quotes match chunk text regardless of source line breaks."""
    return re.sub(r"\s+", " ", text).strip()
