from difflib import SequenceMatcher

from rag_pipeline.models import (
    ClaimResult,
    PromptContext,
    RagAnswerDraft,
    VerificationReport,
)

QUOTE_MATCH_THRESHOLD = 0.80


def verify_citations(
    draft: RagAnswerDraft, context: PromptContext
) -> VerificationReport:
    claim_results: list[ClaimResult] = []
    hallucinated_doc_ids: list[str] = []
    missing_quotes: list[str] = []

    for claim in draft.claims:
        doc_ids_valid = all(
            citation_id in context.doc_id_map for citation_id in claim.citation_ids
        )
        if not doc_ids_valid:
            for citation_id in claim.citation_ids:
                if citation_id not in context.doc_id_map:
                    hallucinated_doc_ids.append(citation_id)

        best_quote_score = 0.0
        if doc_ids_valid:
            for citation_id in claim.citation_ids:
                chunk_text = _chunk_text_for_doc_id(context, citation_id)
                score = _quote_containment_score(claim.supporting_quote, chunk_text)
                best_quote_score = max(best_quote_score, score)

        passed = doc_ids_valid and best_quote_score >= QUOTE_MATCH_THRESHOLD
        if doc_ids_valid and best_quote_score < QUOTE_MATCH_THRESHOLD:
            missing_quotes.append(claim.supporting_quote)

        claim_results.append(
            ClaimResult(
                claim=claim,
                doc_ids_valid=doc_ids_valid,
                quote_match_score=best_quote_score,
                passed=passed,
            )
        )

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
    matcher = SequenceMatcher(None, quote, chunk_text)
    match = matcher.find_longest_match(0, len(quote), 0, len(chunk_text))
    return match.size / len(quote)


def _chunk_text_for_doc_id(context: PromptContext, doc_id: str) -> str:
    marker = f"[{doc_id}]"
    if marker not in context.text:
        return ""
    after_marker = context.text.split(marker, 1)[1]
    return after_marker.split("\n\n", 1)[0].strip()
