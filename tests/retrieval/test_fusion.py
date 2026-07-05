from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_hybrid_search.retrieval.fusion import weighted_rrf


def make_result(chunk_id, dense_score=None, bm25_score=None):
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=f"text {chunk_id}",
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=10,
    )
    return RetrievedChunk(
        chunk=chunk,
        dense_score=dense_score,
        bm25_score=bm25_score,
        rrf_score=0.0,
        rerank_score=None,
        final_rank=0,
    )


def test_fuses_and_ranks_by_combined_reciprocal_rank():
    dense = [make_result("a", dense_score=0.9), make_result("b", dense_score=0.8)]
    sparse = [make_result("b", bm25_score=5.0), make_result("a", bm25_score=1.0)]

    fused = weighted_rrf(dense, sparse, dense_weight=0.7, sparse_weight=0.3, k=60)

    assert [r.chunk.chunk_id for r in fused] == ["a", "b"]
    expected_a = 0.7 * (1 / (60 + 1)) + 0.3 * (1 / (60 + 2))
    assert abs(fused[0].rrf_score - expected_a) < 1e-9


def test_chunk_only_in_one_list_still_included():
    dense = [make_result("a", dense_score=0.9)]
    sparse = [make_result("b", bm25_score=3.0)]

    fused = weighted_rrf(dense, sparse, dense_weight=0.7, sparse_weight=0.3, k=60)

    ids = {r.chunk.chunk_id for r in fused}
    assert ids == {"a", "b"}


def test_empty_inputs_return_empty_list():
    assert weighted_rrf([], [], dense_weight=0.7, sparse_weight=0.3, k=60) == []


def test_preserves_original_dense_and_bm25_scores_in_merged_result():
    dense = [make_result("a", dense_score=0.9)]
    sparse = [make_result("a", bm25_score=2.0)]

    fused = weighted_rrf(dense, sparse, dense_weight=0.7, sparse_weight=0.3, k=60)

    assert fused[0].dense_score == 0.9
    assert fused[0].bm25_score == 2.0
