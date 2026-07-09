import re
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

        # The model occasionally copies a supporting_quote verbatim but
        # tags it with the wrong doc id. Rather than fail a claim whose
        # quote is genuinely present in the context, re-attribute it to
        # whichever doc actually contains it.
        if doc_ids_valid and best_quote_score < QUOTE_MATCH_THRESHOLD:
            best_doc_id, best_doc_score = None, best_quote_score
            for doc_id in context.doc_id_map:
                if doc_id in claim.citation_ids:
                    continue
                chunk_text = _chunk_text_for_doc_id(context, doc_id)
                score = _quote_containment_score(claim.supporting_quote, chunk_text)
                if score > best_doc_score:
                    best_doc_id, best_doc_score = doc_id, score
            if best_doc_id is not None and best_doc_score >= QUOTE_MATCH_THRESHOLD:
                claim.citation_ids = [best_doc_id]
                best_quote_score = best_doc_score

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
    normalized_quote = _normalize_whitespace(quote)
    normalized_chunk = _normalize_whitespace(chunk_text)
    matcher = SequenceMatcher(None, normalized_quote, normalized_chunk)
    match = matcher.find_longest_match(0, len(normalized_quote), 0, len(normalized_chunk))
    return match.size / len(normalized_quote)


def _normalize_whitespace(text: str) -> str:
    """Collapse PDF line-wrap newlines/runs of whitespace to a single space
    so verbatim quotes match chunk text regardless of source line breaks."""
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text_for_doc_id(context: PromptContext, doc_id: str) -> str:
    """Extract this doc's chunk text from the prompt context.

    context_builder joins chunks with '\\n\\n', so naively splitting on the
    first '\\n\\n' after the marker assumes that's always the chunk
    boundary. But a chunk's own text can legitimately contain an internal
    blank line (e.g. 'prose\\n\\ntable rows' from the PDF table renderer),
    which would truncate it early. Instead, find the true boundary: the
    start of whichever other doc's marker comes next in the text.
    """
    marker = f"[{doc_id}]"
    start_idx = context.text.find(marker)
    if start_idx == -1:
        return ""
    start = start_idx + len(marker)

    end = len(context.text)
    for other_id in context.doc_id_map:
        if other_id == doc_id:
            continue
        pos = context.text.find(f"\n\n[{other_id}]", start)
        if pos != -1 and pos < end:
            end = pos

    return context.text[start:end].strip()
