import pytest

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from tests.fakes import fake_pinecone_stores


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


@pytest.fixture
def retriever(tmp_path):
    chunk_store, _vector_store = fake_pinecone_stores()
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    chunks = [
        make_chunk("c1", "how to resolve ERROR_CODE_0x834 during deployment"),
        make_chunk("c2", "onboarding guide for new hires"),
    ]
    for chunk in chunks:
        chunk_store.put(chunk)
    bm25.build(chunks)
    return SparseRetriever(chunk_store, bm25)


def test_search_finds_exact_keyword_with_bm25_score(retriever):
    results = retriever.search("ERROR_CODE_0x834", k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].bm25_score is not None
    assert results[0].dense_score is None
