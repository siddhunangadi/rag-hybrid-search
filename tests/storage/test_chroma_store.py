from datetime import datetime, timezone

import pytest

from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore


def make_record(chunk_id, embedding):
    return EmbeddingRecord(
        chunk_id=chunk_id,
        embedding=embedding,
        embedding_model="test-model",
        embedding_dimension=len(embedding),
        provider="test",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(data_dir=str(tmp_path / "chroma"))


def test_upsert_and_query_returns_closest_first(store):
    store.upsert("a", make_record("a", [1.0, 0.0, 0.0]))
    store.upsert("b", make_record("b", [0.0, 1.0, 0.0]))
    store.upsert("c", make_record("c", [0.9, 0.1, 0.0]))

    results = store.query([1.0, 0.0, 0.0], k=2)

    assert results[0][0] == "a"
    assert len(results) == 2


def test_upsert_overwrites_existing_id(store):
    store.upsert("a", make_record("a", [1.0, 0.0, 0.0]))
    store.upsert("a", make_record("a", [0.0, 0.0, 1.0]))

    results = store.query([0.0, 0.0, 1.0], k=1)

    assert results[0][0] == "a"
    assert results[0][1] > 0.99


def test_delete_removes_vector(store):
    store.upsert("a", make_record("a", [1.0, 0.0, 0.0]))
    store.delete(["a"])

    results = store.query([1.0, 0.0, 0.0], k=5)

    assert "a" not in {chunk_id for chunk_id, _ in results}


def test_persists_across_reopen(tmp_path):
    data_dir = str(tmp_path / "chroma")
    store1 = ChromaVectorStore(data_dir=data_dir)
    store1.upsert("a", make_record("a", [1.0, 0.0, 0.0]))

    store2 = ChromaVectorStore(data_dir=data_dir)
    results = store2.query([1.0, 0.0, 0.0], k=1)

    assert results[0][0] == "a"
