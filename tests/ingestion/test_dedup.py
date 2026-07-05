from rag_hybrid_search.ingestion.dedup import is_duplicate
from rag_hybrid_search.models import Chunk


def make_chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


def test_true_duplicate_is_detected():
    existing_chunk = make_chunk("c1", "def foo(): return bar()")
    existing = [(existing_chunk, [1.0, 0.0, 0.0])]
    candidate = make_chunk("c2", "def foo(): return bar()")

    result = is_duplicate(
        candidate,
        candidate_embedding=[1.0, 0.0, 0.0],
        existing=existing,
        cosine_threshold=0.95,
        text_threshold=0.9,
    )

    assert result is True


def test_high_cosine_but_different_text_is_not_duplicate():
    existing_chunk = make_chunk("c1", "x = [i for i in range(10)]")
    existing = [(existing_chunk, [0.99, 0.01, 0.0])]
    candidate = make_chunk("c2", "y = (j for j in range(20))")

    result = is_duplicate(
        candidate,
        candidate_embedding=[1.0, 0.0, 0.0],
        existing=existing,
        cosine_threshold=0.95,
        text_threshold=0.9,
    )

    assert result is False


def test_below_cosine_threshold_short_circuits_without_duplicate():
    existing_chunk = make_chunk("c1", "completely unrelated content")
    existing = [(existing_chunk, [0.0, 1.0, 0.0])]
    candidate = make_chunk("c2", "totally different topic here")

    result = is_duplicate(
        candidate,
        candidate_embedding=[1.0, 0.0, 0.0],
        existing=existing,
        cosine_threshold=0.95,
        text_threshold=0.9,
    )

    assert result is False


def test_no_existing_chunks_is_never_duplicate():
    candidate = make_chunk("c1", "anything")

    result = is_duplicate(
        candidate,
        candidate_embedding=[1.0, 0.0, 0.0],
        existing=[],
        cosine_threshold=0.95,
        text_threshold=0.9,
    )

    assert result is False
