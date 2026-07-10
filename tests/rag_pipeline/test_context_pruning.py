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


def make_chunk_with_score(chunk_id: str, rerank_score: float):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text="text",
        strategy_version="fixed-v1", heading=None, page=None, char_count=4,
    )
    return RetrievedChunk(
        chunk=chunk, dense_score=0.5, bm25_score=0.5, rrf_score=0.5,
        rerank_score=rerank_score, final_rank=1,
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


def test_min_keep_prevents_over_pruning_below_the_floor():
    """A tight margin would normally prune down to 1 chunk (the top
    scorer dominates the range) -- min_keep=3 must override that and keep
    the top 3 regardless of how wide the score gap is."""
    chunks = [
        make_chunk_with_score("c1", 5.0),
        make_chunk_with_score("c2", 0.1),
        make_chunk_with_score("c3", 0.05),
        make_chunk_with_score("c4", 0.01),
    ]

    result = prune_by_score_margin(chunks, margin=0.1, min_keep=3)

    assert len(result) == 3
    assert [c.chunk.chunk_id for c in result] == ["c1", "c2", "c3"]


def test_min_keep_is_noop_when_margin_already_keeps_more():
    chunks = [
        make_chunk_with_score("c1", 1.0),
        make_chunk_with_score("c2", 0.95),
        make_chunk_with_score("c3", 0.9),
    ]

    result = prune_by_score_margin(chunks, margin=0.5, min_keep=1)

    assert len(result) == 2


def test_min_keep_default_preserves_existing_behavior():
    """Existing call sites that don't pass min_keep must behave exactly as
    before -- default min_keep=1 means "no floor beyond the margin rule"."""
    chunks = [
        make_chunk_with_score("c1", 5.0),
        make_chunk_with_score("c2", 0.1),
    ]

    result = prune_by_score_margin(chunks, margin=0.1)

    assert len(result) == 1
    assert result[0].chunk.chunk_id == "c1"
