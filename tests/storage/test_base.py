import pytest

from rag_hybrid_search.storage.base import ChunkStore, VectorStore


def test_vector_store_is_abstract():
    with pytest.raises(TypeError):
        VectorStore()


def test_chunk_store_is_abstract():
    with pytest.raises(TypeError):
        ChunkStore()


def test_vector_store_subclass_must_implement_all_methods():
    class Incomplete(VectorStore):
        def upsert(self, chunk_id, embedding_record):
            pass

    with pytest.raises(TypeError):
        Incomplete()
