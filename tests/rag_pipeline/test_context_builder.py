from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_pipeline.context_builder import build_context


def make_retrieved_chunk(chunk_id, text, final_rank):
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
        dense_score=0.9,
        bm25_score=0.9,
        rrf_score=0.5,
        rerank_score=0.8,
        final_rank=final_rank,
    )


def test_empty_context():
    context = build_context([])
    assert context.text == ""
    assert context.doc_id_map == {}


def test_numbers_chunks_in_rank_order():
    chunks = [
        make_retrieved_chunk("c1", "first chunk text", final_rank=1),
        make_retrieved_chunk("c2", "second chunk text", final_rank=2),
    ]
    context = build_context(chunks)
    assert "[d1]" in context.text
    assert "[d2]" in context.text
    assert context.text.index("[d1]") < context.text.index("[d2]")
    assert context.doc_id_map == {"d1": "c1", "d2": "c2"}


def test_deduplicates_by_chunk_id():
    chunk = make_retrieved_chunk("c1", "same chunk", final_rank=1)
    context = build_context([chunk, chunk])
    assert len(context.doc_id_map) == 1


def test_truncates_lowest_ranked_chunks_first_without_splitting_text():
    big_text = "word " * 400  # ~2000 chars, ~500 approx tokens
    chunks = [
        make_retrieved_chunk("c1", big_text, final_rank=1),
        make_retrieved_chunk("c2", big_text, final_rank=2),
        make_retrieved_chunk("c3", big_text, final_rank=3),
    ]
    # Budget only large enough for ~1 chunk (500 tokens * 4 chars/token = 2000 chars)
    context = build_context(chunks, approx_token_budget=500)
    assert "[d1]" in context.text
    assert "[d3]" not in context.text
    assert big_text.strip() in context.text
