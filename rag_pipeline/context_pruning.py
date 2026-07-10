"""Trims post-rerank chunks before they're sent to the LLM, when the
reranker's score spread signals one chunk clearly answers the question.

Deliberately range-based, not a multiplicative ratio of the raw score
value: rerank_score can be an NVIDIA reranker logit (frequently negative --
seen as low as -3.4 in production), a CrossEncoder logit, or a cosine
similarity, depending on backend. A naive "keep if score >= 0.7 * top"
check inverts for negative top scores (0.7 * -3.4 is a *higher*, i.e.
stricter, bar than -3.4 itself). Measuring the margin against the
observed top-to-bottom score *range* for this specific batch of candidates
is scale- and sign-invariant across all three backends.
"""
from rag_hybrid_search.models import RetrievedChunk


def prune_by_score_margin(chunks: list[RetrievedChunk], margin: float) -> list[RetrievedChunk]:
    """Drop chunks whose rerank_score falls more than `margin` of the
    top-to-bottom score range below the top chunk.

    No-op (returns chunks unchanged) when: fewer than 2 chunks, any chunk is
    missing rerank_score (PassthroughReranker never scores candidates -- no
    ground truth to prune by), or all chunks score identically (no basis to
    discriminate). Chunks are assumed already sorted best-first.
    """
    if len(chunks) < 2:
        return chunks
    scores = [c.rerank_score for c in chunks]
    if any(s is None for s in scores):
        return chunks

    top_score = scores[0]
    score_range = top_score - min(scores)
    if score_range <= 0:
        return chunks

    threshold = top_score - margin * score_range
    return [c for c in chunks if c.rerank_score >= threshold]
