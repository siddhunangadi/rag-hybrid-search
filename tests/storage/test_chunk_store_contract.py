import pytest

from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore


def _sqlite_store(tmp_path):
    return SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))


IMPLEMENTATIONS = [_sqlite_store]


@pytest.mark.parametrize("make_store", IMPLEMENTATIONS)
def test_implements_full_chunk_store_contract(make_store, tmp_path):
    store = make_store(tmp_path)
    assert isinstance(store, ChunkStore)
    # These three are real, load-bearing methods beyond the ABC's original
    # four -- this test exists so a future ChunkStore implementation can't
    # silently skip them and break ingestion dedup / compliance routing /
    # document-summary endpoints.
    assert hasattr(store, "get_document_hash")
    assert hasattr(store, "get_by_legal_metadata")
    assert hasattr(store, "get_document_summaries")


def test_abc_requires_the_three_extra_methods():
    with pytest.raises(TypeError, match="abstract method"):
        class Incomplete(ChunkStore):
            def get(self, chunk_id): ...
            def get_by_document(self, document_id): ...
            def get_document_hash(self, source_path): ...
            def put(self, chunk): ...
            def delete_by_document(self, document_id): ...
            def all(self): ...
            # missing get_by_legal_metadata and get_document_summaries

        Incomplete()
