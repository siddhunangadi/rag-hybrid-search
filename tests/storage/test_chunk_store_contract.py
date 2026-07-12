import pytest

from rag_hybrid_search.storage.base import ChunkStore


def _pinecone_chunk_store(tmp_path):
    from unittest.mock import MagicMock, patch

    patcher = patch("rag_hybrid_search.storage.pinecone_connection.Pinecone")
    mock_pc_cls = patcher.start()
    mock_pc_cls.return_value.Index.return_value = MagicMock()
    from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
    from rag_hybrid_search.storage.pinecone_connection import PineconeConnection

    client = PineconeConnection(api_key="k", index_name="idx")
    return PineconeChunkStore(client, embedding_dimension=3)


IMPLEMENTATIONS = [_pinecone_chunk_store]


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
    assert hasattr(store, "all_with_embeddings")


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
