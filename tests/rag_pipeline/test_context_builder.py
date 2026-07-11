from rag_hybrid_search.models import Chunk, ChunkProvenance, ContextChunk, RetrievedChunk
from rag_pipeline.context_builder import ContextLayout, build_context


def make_context_chunk(chunk_id, text, final_rank, primary_subquery=0, all_subqueries=None):
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
    retrieved = RetrievedChunk(
        chunk=chunk,
        dense_score=0.9,
        bm25_score=0.9,
        rrf_score=0.5,
        rerank_score=0.8,
        final_rank=final_rank,
    )
    provenance = ChunkProvenance(
        primary_subquery=primary_subquery,
        all_subqueries=all_subqueries or [primary_subquery],
    )
    return ContextChunk(chunk=retrieved, provenance=provenance)


def test_empty_context():
    context = build_context([], subqueries=[])
    assert context.text == ""
    assert context.doc_id_map == {}


def test_flat_numbers_chunks_in_rank_order():
    chunks = [
        make_context_chunk("c1", "first chunk text", final_rank=1),
        make_context_chunk("c2", "second chunk text", final_rank=2),
    ]
    context = build_context(chunks, subqueries=["q"])
    assert "[d1]" in context.text
    assert "[d2]" in context.text
    assert context.text.index("[d1]") < context.text.index("[d2]")
    assert context.doc_id_map == {"d1": "c1", "d2": "c2"}
    assert "Evidence for subquery" not in context.text  # flat has no group headers


def test_deduplicates_by_chunk_id():
    chunk = make_context_chunk("c1", "same chunk", final_rank=1)
    context = build_context([chunk, chunk], subqueries=["q"])
    assert len(context.doc_id_map) == 1


def test_truncates_lowest_ranked_chunks_first_without_splitting_text():
    big_text = "word " * 400  # ~2000 chars, ~500 approx tokens
    chunks = [
        make_context_chunk("c1", big_text, final_rank=1),
        make_context_chunk("c2", big_text, final_rank=2),
        make_context_chunk("c3", big_text, final_rank=3),
    ]
    context = build_context(chunks, subqueries=["q"], approx_token_budget=500)
    assert "[d1]" in context.text
    assert "[d3]" not in context.text


def test_grouped_sections_by_primary_subquery_in_decomposition_order():
    chunks = [
        make_context_chunk("c1", "about rq1", final_rank=1, primary_subquery=0),
        make_context_chunk("c2", "also about rq1", final_rank=2, primary_subquery=0),
        make_context_chunk("c3", "about rq3", final_rank=3, primary_subquery=1),
    ]
    subqueries = ["What does RQ1 conclude?", "What does RQ3 conclude?"]
    context = build_context(chunks, subqueries, layout=ContextLayout.GROUPED)

    assert "Evidence for subquery 1" in context.text
    assert "Evidence for subquery 2" in context.text
    assert '"What does RQ1 conclude?"' in context.text
    assert '"What does RQ3 conclude?"' in context.text
    # subquery 1's section (index 0) must appear before subquery 2's (index 1)
    assert context.text.index("Evidence for subquery 1") < context.text.index("Evidence for subquery 2")
    # chunks within a group keep final_rank order
    assert context.text.index("[d1]") < context.text.index("[d2]")


def test_grouped_citation_numbering_is_global_and_monotonic():
    chunks = [
        make_context_chunk("c1", "text 1", final_rank=1, primary_subquery=0),
        make_context_chunk("c2", "text 2", final_rank=2, primary_subquery=1),
    ]
    subqueries = ["sub a", "sub b"]
    context = build_context(chunks, subqueries, layout=ContextLayout.GROUPED)
    assert context.doc_id_map == {"d1": "c1", "d2": "c2"}
    assert context.text.index("[d1]") < context.text.index("[d2]")


def test_grouped_renders_multi_subquery_chunk_exactly_once():
    chunk = make_context_chunk("c1", "shared text", final_rank=1, primary_subquery=0, all_subqueries=[0, 1])
    subqueries = ["sub a", "sub b"]
    context = build_context([chunk], subqueries, layout=ContextLayout.GROUPED)
    assert context.text.count("[d1]") == 1
    assert "Evidence for subquery 2" not in context.text  # empty group not rendered


def test_flat_layout_byte_identical_to_grouped_chunk_set_but_no_headers():
    chunks = [
        make_context_chunk("c1", "first", final_rank=1, primary_subquery=0),
        make_context_chunk("c2", "second", final_rank=2, primary_subquery=1),
    ]
    subqueries = ["sub a", "sub b"]
    flat = build_context(chunks, subqueries, layout=ContextLayout.FLAT)
    assert flat.text == "[d1]\nfirst\n\n[d2]\nsecond"
