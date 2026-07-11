"""Backend-authoritative supporting_quote extraction.

The model is no longer trusted to produce supporting_quote (prompt v2 asks
only for {text, citation_ids}). Instead, for each claim the backend slices
the quote directly out of the one chunk of text the claim cites. Because
the quote is a substring of exactly one chunk's own string, it cannot span
multiple chunks by construction -- there is no code path that concatenates
text from two different chunks into one quote.
"""
import re
from difflib import SequenceMatcher

from rag_pipeline.citation_utils import chunk_text_for_doc_id
from rag_pipeline.models import Claim, PromptContext

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _best_sentence_span(claim_text: str, chunk_text: str) -> str:
    """Pick the single sentence (or, if it scores strictly better, that
    sentence plus its neighbor) from chunk_text that best matches claim_text.

    Scored with difflib.SequenceMatcher.ratio() -- a lexical-overlap
    heuristic, not semantic understanding. It can occasionally pick a
    verbatim-but-weakly-relevant sentence; it can never pick text that
    doesn't exist verbatim in this chunk, which is the property that
    matters for citation integrity.
    """
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return ""

    scored = [
        (SequenceMatcher(None, claim_text.lower(), s.lower()).ratio(), i, s)
        for i, s in enumerate(sentences)
    ]
    best_score, best_i, best_sentence = max(scored, key=lambda t: t[0])

    for neighbor_i in (best_i - 1, best_i + 1):
        if 0 <= neighbor_i < len(sentences):
            candidate = " ".join(
                sentences[min(best_i, neighbor_i):max(best_i, neighbor_i) + 1]
            )
            candidate_score = SequenceMatcher(None, claim_text.lower(), candidate.lower()).ratio()
            if candidate_score > best_score:
                best_score, best_sentence = candidate_score, candidate

    return best_sentence


def extract_supporting_quotes(claims: list[Claim], context: PromptContext) -> list[Claim]:
    """Return a new list of claims with citation_ids narrowed to one id each
    and supporting_quote filled in mechanically from that cited chunk.

    Claims citing multiple ids are narrowed to the first id -- a single
    supporting_quote can only unambiguously belong to one chunk, so keeping
    multiple ids is what allows an ambiguous/cross-chunk quote to happen in
    the first place. Claims citing an unknown id are left with an empty
    quote; verify_citations already handles hallucinated ids downstream.
    """
    fixed: list[Claim] = []
    for claim in claims:
        citation_id = claim.citation_ids[0] if claim.citation_ids else None
        if citation_id is None or citation_id not in context.doc_id_map:
            fixed.append(claim.model_copy(update={
                "citation_ids": claim.citation_ids[:1],
                "supporting_quote": "",
            }))
            continue

        chunk_text = chunk_text_for_doc_id(context, citation_id)
        quote = _best_sentence_span(claim.text, chunk_text)
        fixed.append(claim.model_copy(update={
            "citation_ids": [citation_id],
            "supporting_quote": quote,
        }))
    return fixed
