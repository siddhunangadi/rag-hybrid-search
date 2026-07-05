from rag_hybrid_search.models import RetrievedChunk


def weighted_rrf(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    dense_weight: float,
    sparse_weight: float,
    k: int,
) -> list[RetrievedChunk]:
    """Weighted Reciprocal Rank Fusion.

    Standard RRF sums 1/(k+rank) contributions from each ranked list
    unweighted. This variant scales each list's contribution by a
    configured weight before summing, so callers can bias fusion toward
    dense or sparse results -- hence "weighted", not vanilla RRF.
    """
    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.chunk.chunk_id
        merged[chunk_id] = result
        scores[chunk_id] = scores.get(chunk_id, 0.0) + dense_weight * (1 / (k + rank))

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result.chunk.chunk_id
        if chunk_id in merged:
            existing = merged[chunk_id]
            merged[chunk_id] = existing.model_copy(update={"bm25_score": result.bm25_score})
        else:
            merged[chunk_id] = result
        scores[chunk_id] = scores.get(chunk_id, 0.0) + sparse_weight * (1 / (k + rank))

    fused = [
        merged[chunk_id].model_copy(update={"rrf_score": score})
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    return fused
