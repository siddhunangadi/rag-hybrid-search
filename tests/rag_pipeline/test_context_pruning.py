from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_pipeline.context_pruning import prune_by_score_margin


def make_chunk(chunk_id, rerank_score, final_rank):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text="text",
        strategy_version="v1", char_count=4,
    )
    return RetrievedChunk(
        chunk=chunk, rrf_score=0.5, rerank_score=rerank_score, final_rank=final_rank,
    )


def test_single_chunk_is_unaffected():
    chunks = [make_chunk("c1", 3.6, 1)]
    assert prune_by_score_margin(chunks, margin=0.3) == chunks


def test_clear_winner_prunes_weakly_scored_chunks():
    """One chunk clearly answers the question (large score gap to the
    rest) -- the weak trailing chunks should be dropped."""
    chunks = [
        make_chunk("c1", 3.6, 1),
        make_chunk("c2", -1.2, 2),
        make_chunk("c3", -2.7, 3),
        make_chunk("c4", -2.9, 4),
        make_chunk("c5", -3.4, 5),
    ]
    kept = prune_by_score_margin(chunks, margin=0.3)
    assert [c.chunk.chunk_id for c in kept] == ["c1"]


def test_negative_scores_do_not_invert_the_threshold():
    """Regression test: a naive `score >= 0.7 * top_score` check inverts
    when top_score is negative (0.7 * -3.4 = -2.38 is a *stricter* bar than
    -3.4, the opposite of what "keep 70%" should mean). With range-based
    margin, a tight cluster of negative scores (-1.0, -1.1, -1.2, margin
    0.5 of a 0.2 range) keeps the two within margin of the top and drops
    the one that's fully a margin-width away -- not a nonsensical result
    like keeping nothing or everything regardless of spread."""
    chunks = [
        make_chunk("c1", -1.0, 1),
        make_chunk("c2", -1.1, 2),
        make_chunk("c3", -1.2, 3),
    ]
    kept = prune_by_score_margin(chunks, margin=0.5)
    assert [c.chunk.chunk_id for c in kept] == ["c1", "c2"]


def test_two_close_scores_kept_one_clear_outlier_dropped():
    chunks = [
        make_chunk("c1", 0.91, 1),
        make_chunk("c2", 0.905, 2),
        make_chunk("c3", 0.10, 3),
    ]
    kept = prune_by_score_margin(chunks, margin=0.3)
    assert [c.chunk.chunk_id for c in kept] == ["c1", "c2"]


def test_identical_scores_are_a_noop():
    chunks = [make_chunk("c1", 0.5, 1), make_chunk("c2", 0.5, 2)]
    assert prune_by_score_margin(chunks, margin=0.3) == chunks


def test_missing_rerank_score_is_a_noop():
    """PassthroughReranker never sets rerank_score -- nothing to prune by,
    so all chunks pass through unchanged rather than being (incorrectly)
    treated as tied or dropped."""
    chunk_a = make_chunk("c1", None, 1)
    chunk_b = make_chunk("c2", None, 2)
    chunks = [chunk_a, chunk_b]
    assert prune_by_score_margin(chunks, margin=0.3) == chunks


def test_margin_zero_keeps_only_exact_ties_with_top():
    chunks = [
        make_chunk("c1", 1.0, 1),
        make_chunk("c2", 0.99, 2),
    ]
    kept = prune_by_score_margin(chunks, margin=0.0)
    assert [c.chunk.chunk_id for c in kept] == ["c1"]


def test_margin_one_keeps_everything_within_full_range():
    chunks = [
        make_chunk("c1", 1.0, 1),
        make_chunk("c2", 0.0, 2),
        make_chunk("c3", -5.0, 3),
    ]
    kept = prune_by_score_margin(chunks, margin=1.0)
    assert len(kept) == 3
