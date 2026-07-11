from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.base import VectorStore
from rag_hybrid_search.storage.pinecone_connection import PineconeConnection
from rag_hybrid_search.storage.pinecone_vector_store import PineconeVectorStore


def _embedding_record(chunk_id="c1"):
    return EmbeddingRecord(
        chunk_id=chunk_id, embedding=[0.1, 0.2, 0.3], embedding_model="nv-embed",
        embedding_dimension=3, provider="nvidia", created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_client():
    with patch("rag_hybrid_search.storage.pinecone_connection.Pinecone") as mock_pc_cls:
        mock_index = MagicMock()
        mock_pc_cls.return_value.Index.return_value = mock_index
        client = PineconeConnection(api_key="k", index_name="idx")
        yield client, mock_index


def test_implements_vector_store(mock_client):
    client, _ = mock_client
    store = PineconeVectorStore(client)
    assert isinstance(store, VectorStore)


def test_upsert_updates_vector_values_on_existing_record(mock_client):
    """upsert() uses index.update(), not index.upsert() -- the record
    already exists by the time this runs (PineconeChunkStore.put() always
    runs first per the real ingestion order), and update() leaves that
    record's metadata untouched."""
    client, mock_index = mock_client
    store = PineconeVectorStore(client)
    store.upsert("c1", _embedding_record())
    mock_index.update.assert_called_once_with(id="c1", values=[0.1, 0.2, 0.3])


def test_query_returns_chunk_id_score_pairs(mock_client):
    client, mock_index = mock_client
    mock_index.query.return_value = MagicMock(
        matches=[MagicMock(id="c1", score=0.9), MagicMock(id="c2", score=0.7)]
    )
    store = PineconeVectorStore(client)
    results = store.query([0.1, 0.2, 0.3], k=5)
    assert results == [("c1", 0.9), ("c2", 0.7)]
    mock_index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3], top_k=5, include_metadata=False,
    )


def test_delete(mock_client):
    client, mock_index = mock_client
    store = PineconeVectorStore(client)
    store.delete(["c1", "c2"])
    mock_index.delete.assert_called_once_with(ids=["c1", "c2"])


def test_shares_one_client_across_two_stores(mock_client):
    """Both PineconeVectorStore and PineconeChunkStore, constructed from the
    same PineconeConnection, must issue calls against the same underlying index
    object -- not open a second connection."""
    client, mock_index = mock_client
    from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
    vector_store = PineconeVectorStore(client)
    chunk_store = PineconeChunkStore(client, embedding_dimension=3)
    assert vector_store._index is chunk_store._index is mock_index
