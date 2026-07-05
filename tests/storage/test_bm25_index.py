from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.bm25_index import BM25Index


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


def test_search_finds_exact_keyword_match(tmp_path):
    index = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index.build(
        [
            make_chunk("c1", "how to fix ERROR_CODE_0x834 in production"),
            make_chunk("c2", "general onboarding guide for new engineers"),
            make_chunk("c3", "deploying the service to staging"),
        ]
    )

    results = index.search("ERROR_CODE_0x834", k=2)

    assert results[0][0] == "c1"


def test_search_on_empty_index_returns_empty(tmp_path):
    index = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index.build([])

    assert index.search("anything", k=5) == []


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "bm25.pkl")
    index = BM25Index(index_path=path)
    index.build([make_chunk("c1", "unique keyword banana")])
    index.save()

    reloaded = BM25Index(index_path=path)
    loaded = reloaded.load()

    assert loaded is True
    results = reloaded.search("banana", k=1)
    assert results[0][0] == "c1"


def test_load_returns_false_when_no_file(tmp_path):
    index = BM25Index(index_path=str(tmp_path / "missing.pkl"))
    assert index.load() is False
