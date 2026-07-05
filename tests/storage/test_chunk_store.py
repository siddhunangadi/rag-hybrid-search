import pytest

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore


def make_chunk(chunk_id="c1", document_id="d1", index=0, text="hello"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=index,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )


@pytest.fixture
def store(tmp_path):
    return SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))


def test_put_and_get(store):
    chunk = make_chunk()
    store.put(chunk)
    fetched = store.get("c1")
    assert fetched is not None
    assert fetched.text == "hello"


def test_get_missing_returns_none(store):
    assert store.get("missing") is None


def test_get_by_document(store):
    store.put(make_chunk(chunk_id="c1", document_id="d1", index=0))
    store.put(make_chunk(chunk_id="c2", document_id="d1", index=1))
    store.put(make_chunk(chunk_id="c3", document_id="d2", index=0))
    chunks = store.get_by_document("d1")
    assert {c.chunk_id for c in chunks} == {"c1", "c2"}


def test_document_hash_tracking(store):
    chunk = make_chunk(chunk_id="c1", document_id="deadbeef")
    store.put(chunk, source_path="/docs/a.md")
    assert store.get_document_hash("/docs/a.md") == "deadbeef"
    assert store.get_document_hash("/docs/missing.md") is None


def test_delete_by_document(store):
    store.put(make_chunk(chunk_id="c1", document_id="d1"), source_path="/docs/a.md")
    store.delete_by_document("d1")
    assert store.get("c1") is None
    assert store.get_document_hash("/docs/a.md") is None


def test_all_returns_every_chunk(store):
    store.put(make_chunk(chunk_id="c1", document_id="d1"))
    store.put(make_chunk(chunk_id="c2", document_id="d2"))
    ids = {c.chunk_id for c in store.all()}
    assert ids == {"c1", "c2"}


def test_reopening_store_persists_data(tmp_path):
    db_path = str(tmp_path / "chunks.db")
    store1 = SqliteChunkStore(db_path=db_path)
    store1.put(make_chunk(chunk_id="c1", document_id="d1"), source_path="/docs/a.md")

    store2 = SqliteChunkStore(db_path=db_path)
    assert store2.get("c1") is not None
    assert store2.get_document_hash("/docs/a.md") == "d1"


def test_get_document_summaries_groups_by_document(store):
    store.put(make_chunk(chunk_id="c1", document_id="d1", index=0), source_path="/docs/a.md")
    store.put(make_chunk(chunk_id="c2", document_id="d1", index=1), source_path="/docs/a.md")
    store.put(make_chunk(chunk_id="c3", document_id="d2", index=0), source_path="/docs/b.md")

    summaries = {s["document_id"]: s for s in store.get_document_summaries()}

    assert summaries["d1"]["source_path"] == "/docs/a.md"
    assert summaries["d1"]["chunk_count"] == 2
    assert summaries["d2"]["source_path"] == "/docs/b.md"
    assert summaries["d2"]["chunk_count"] == 1


def test_get_document_summaries_empty_store(store):
    assert store.get_document_summaries() == []
