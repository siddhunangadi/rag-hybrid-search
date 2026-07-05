from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_hybrid_search.retrieval.passthrough_rerank import PassthroughReranker


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text="x",
        strategy_version="v1", heading=None, page=None, char_count=1,
    )


def test_passthrough_reranker_orders_by_rrf_score_and_truncates():
    candidates = [
        RetrievedChunk(chunk=_chunk("low"), rrf_score=0.1, final_rank=0),
        RetrievedChunk(chunk=_chunk("high"), rrf_score=0.9, final_rank=0),
        RetrievedChunk(chunk=_chunk("mid"), rrf_score=0.5, final_rank=0),
    ]

    result = PassthroughReranker().rerank("irrelevant query", candidates, top_n=2)

    assert [r.chunk.chunk_id for r in result] == ["high", "mid"]
    assert [r.final_rank for r in result] == [1, 2]
    assert all(r.rerank_score is None for r in result)


def test_passthrough_reranker_handles_empty_candidates():
    assert PassthroughReranker().rerank("q", [], top_n=3) == []
