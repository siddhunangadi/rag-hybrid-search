from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_hybrid_search.retrieval.rerank import CrossEncoderReranker


def make_result(chunk_id, text):
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )
    return RetrievedChunk(
        chunk=chunk,
        dense_score=0.5,
        bm25_score=0.5,
        rrf_score=0.01,
        rerank_score=None,
        final_rank=0,
    )


def test_rerank_orders_by_relevance_and_sets_final_rank():
    reranker = CrossEncoderReranker()
    candidates = [
        make_result("a", "The Eiffel Tower is located in Paris, France."),
        make_result("b", "Bananas are a good source of potassium."),
    ]

    results = reranker.rerank("Where is the Eiffel Tower?", candidates, top_n=2)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].rerank_score is not None
    assert [r.final_rank for r in results] == [1, 2]


def test_rerank_respects_top_n():
    reranker = CrossEncoderReranker()
    candidates = [make_result(str(i), f"filler text number {i}") for i in range(5)]

    results = reranker.rerank("filler text", candidates, top_n=2)

    assert len(results) == 2


def test_rerank_empty_candidates_returns_empty():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("anything", [], top_n=5) == []
