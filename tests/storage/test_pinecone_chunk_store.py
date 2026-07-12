from unittest.mock import MagicMock, patch

import pytest

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
from rag_hybrid_search.storage.pinecone_connection import PineconeConnection


def _chunk(chunk_id="c1", document_id="d1", chunk_index=0, text="hello world",
           heading=None, page=None):
    return Chunk(
        chunk_id=chunk_id, document_id=document_id, chunk_index=chunk_index,
        text=text, strategy_version="fixed-v1", heading=heading, page=page,
        char_count=len(text),
    )


@pytest.fixture
def mock_client():
    with patch("rag_hybrid_search.storage.pinecone_connection.Pinecone") as mock_pc_cls:
        mock_index = MagicMock()
        mock_pc_cls.return_value.Index.return_value = mock_index
        client = PineconeConnection(api_key="k", index_name="idx")
        yield client, mock_index


def test_implements_chunk_store(mock_client):
    client, _ = mock_client
    store = PineconeChunkStore(client, embedding_dimension=3)
    assert isinstance(store, ChunkStore)


def test_put_creates_record_with_placeholder_vector(mock_client):
    """put() runs first in the real ingestion order (before any vector
    exists), so it must upsert() a full record -- a placeholder zero-vector
    of the configured dimension plus the real metadata -- not update() a
    record that doesn't exist yet."""
    client, mock_index = mock_client
    store = PineconeChunkStore(client, embedding_dimension=3)
    store.put(_chunk(), source_path="doc.md")
    mock_index.upsert.assert_called_once()
    call_kwargs = mock_index.upsert.call_args.kwargs
    vectors = call_kwargs["vectors"]
    assert vectors[0]["id"] == "c1"
    # Pinecone rejects an all-zero dense vector -- the placeholder must have
    # exactly one non-zero component (found against a real index; see
    # tests/storage/test_pinecone_live.py and the module docstring).
    assert vectors[0]["values"][0] != 0.0
    assert vectors[0]["values"][1:] == [0.0, 0.0]
    assert vectors[0]["metadata"]["document_id"] == "d1"
    assert vectors[0]["metadata"]["text"] == "hello world"
    assert vectors[0]["metadata"]["source_path"] == "doc.md"


def test_put_many_batches_into_one_upsert_call(mock_client):
    """put_many() must batch all chunks into as few index.upsert() calls as
    possible instead of one call per chunk -- this is the ingestion-latency
    fix: writing N chunks used to mean N sequential round-trips."""
    client, mock_index = mock_client
    store = PineconeChunkStore(client, embedding_dimension=3)
    chunks = [_chunk(chunk_id=f"c{i}") for i in range(5)]

    store.put_many(chunks, source_path="doc.md")

    mock_index.upsert.assert_called_once()
    vectors = mock_index.upsert.call_args.kwargs["vectors"]
    assert len(vectors) == 5
    assert {v["id"] for v in vectors} == {f"c{i}" for i in range(5)}
    assert all(v["metadata"]["source_path"] == "doc.md" for v in vectors)


def test_put_many_splits_into_multiple_batches_over_batch_size(mock_client):
    from rag_hybrid_search.storage.pinecone_chunk_store import _UPSERT_BATCH_SIZE

    client, mock_index = mock_client
    store = PineconeChunkStore(client, embedding_dimension=3)
    chunks = [_chunk(chunk_id=f"c{i}") for i in range(_UPSERT_BATCH_SIZE + 10)]

    store.put_many(chunks, source_path="doc.md")

    # Batches are dispatched concurrently, so their arrival order isn't
    # guaranteed -- assert on the set of batch sizes, not call order.
    assert mock_index.upsert.call_count == 2
    batch_sizes = sorted(
        len(call.kwargs["vectors"]) for call in mock_index.upsert.call_args_list
    )
    assert batch_sizes == sorted([_UPSERT_BATCH_SIZE, 10])


def test_get_by_chunk_id_reconstructs_chunk(mock_client):
    client, mock_index = mock_client
    mock_index.fetch.return_value = MagicMock(
        vectors={
            "c1": MagicMock(metadata={
                "document_id": "d1", "chunk_index": 0, "text": "hello world",
                "strategy_version": "fixed-v1", "heading": "", "page": -1,
                "char_count": 11, "source_path": "doc.md",
            })
        }
    )
    store = PineconeChunkStore(client, embedding_dimension=3)
    chunk = store.get("c1")
    assert chunk is not None
    assert chunk.chunk_id == "c1"
    assert chunk.heading is None  # sentinel "" round-trips back to None
    assert chunk.page is None     # sentinel -1 round-trips back to None


def test_get_round_trips_legal_metadata(mock_client):
    """A chunk matched by get_by_legal_metadata must come back with that
    same legal_metadata populated, not dropped -- _metadata_to_chunk must
    reconstruct LegalMetadata from the legal_* fields it wrote, not just
    read the plain fields."""
    client, mock_index = mock_client
    mock_index.fetch.return_value = MagicMock(
        vectors={
            "c1": MagicMock(metadata={
                "document_id": "d1", "chunk_index": 0, "text": "hello world",
                "strategy_version": "fixed-v1", "heading": "", "page": -1,
                "char_count": 11, "source_path": "doc.md",
                "legal_regulation": "GDPR", "legal_version": "",
                "legal_jurisdiction": "", "legal_article": "", "legal_section": "",
                "legal_clause": "", "legal_effective_date": "2024-01-01",
                "legal_document_type": "regulation",
            })
        }
    )
    store = PineconeChunkStore(client, embedding_dimension=3)
    chunk = store.get("c1")
    assert chunk.legal_metadata is not None
    assert chunk.legal_metadata.regulation == "GDPR"
    assert chunk.legal_metadata.document_type == "regulation"
    assert str(chunk.legal_metadata.effective_date) == "2024-01-01"


def test_put_writes_legal_effective_date(mock_client):
    from datetime import date
    from rag_hybrid_search.compliance.regulation_models import LegalMetadata

    client, mock_index = mock_client
    chunk = _chunk().model_copy(update={
        "legal_metadata": LegalMetadata(
            document_id="d1", document_title="d1", effective_date=date(2024, 1, 1),
        )
    })
    store = PineconeChunkStore(client, embedding_dimension=3)
    store.put(chunk, source_path="doc.md")
    metadata = mock_index.upsert.call_args.kwargs["vectors"][0]["metadata"]
    assert metadata["legal_effective_date"] == "2024-01-01"


def test_get_missing_chunk_returns_none(mock_client):
    client, mock_index = mock_client
    mock_index.fetch.return_value = MagicMock(vectors={})
    store = PineconeChunkStore(client, embedding_dimension=3)
    assert store.get("missing") is None


def test_delete_by_document(mock_client):
    client, mock_index = mock_client
    store = PineconeChunkStore(client, embedding_dimension=3)
    store.delete_by_document("d1")
    mock_index.delete.assert_called_once_with(
        filter={"document_id": {"$eq": "d1"}},
    )


# -- get_by_document / get_document_hash / get_by_legal_metadata --------------
#
# Step 2 finding: the installed pinecone SDK's Index.query() is a similarity-
# search primitive (vector/id/sparse_vector all optional in its type hints,
# but Pinecone's query endpoint has no supported "none of the above" mode),
# and Index.list() only supports prefix/limit, no metadata filter. So these
# three methods use the same list()-then-fetch()-then-filter-client-side scan
# this file's all()/get_document_summaries/delete_by_document already use --
# not a server-side filtered query.

def _metadata(document_id, chunk_index, text, source_path=""):
    return {
        "document_id": document_id, "chunk_index": chunk_index, "text": text,
        "strategy_version": "fixed-v1", "heading": "", "page": -1,
        "char_count": len(text), "source_path": source_path,
    }


def _list_page(*chunk_ids):
    """index.list() yields ListResponse pages (page.vectors is a list of
    ListItem objects with .id), not plain id strings -- this mock must match
    that shape, or a bug in code that assumes list() yields raw id lists
    would pass its mocked test while breaking against a real index (as
    happened here; see the _scan_all fix in pinecone_chunk_store.py)."""
    return MagicMock(vectors=[MagicMock(id=cid) for cid in chunk_ids])


def test_get_by_document_scans_and_filters_client_side(mock_client):
    client, mock_index = mock_client
    mock_index.list.return_value = iter([_list_page("c1", "c2")])
    mock_index.fetch.return_value = MagicMock(
        vectors={
            "c1": MagicMock(metadata=_metadata("d1", 0, "hello")),
            "c2": MagicMock(metadata=_metadata("d2", 0, "world")),
        }
    )
    store = PineconeChunkStore(client, embedding_dimension=3)
    chunks = store.get_by_document("d1")
    assert [c.chunk_id for c in chunks] == ["c1"]


def test_all_with_embeddings_returns_chunk_and_stored_vector(mock_client):
    """all_with_embeddings() must reuse the embedding already present in
    fetch()'s response instead of requiring a caller to re-embed -- this is
    the fix for ingestion re-embedding the whole corpus on every ingest()."""
    client, mock_index = mock_client
    mock_index.list.return_value = iter([_list_page("c1")])
    mock_index.fetch.return_value = MagicMock(
        vectors={
            "c1": MagicMock(metadata=_metadata("d1", 0, "hello"), values=[0.1, 0.2, 0.3]),
        }
    )
    store = PineconeChunkStore(client, embedding_dimension=3)
    items = list(store.all_with_embeddings())
    assert len(items) == 1
    assert items[0].chunk.chunk_id == "c1"
    assert items[0].embedding == [0.1, 0.2, 0.3]


def test_get_document_hash_scans_and_filters_client_side(mock_client):
    client, mock_index = mock_client
    mock_index.list.return_value = iter([_list_page("c1")])
    mock_index.fetch.return_value = MagicMock(
        vectors={"c1": MagicMock(metadata=_metadata("d1", 0, "hello", source_path="doc.md"))}
    )
    store = PineconeChunkStore(client, embedding_dimension=3)
    assert store.get_document_hash("doc.md") == "d1"
    assert store.get_document_hash("missing.md") is None


def test_get_by_legal_metadata_scans_and_filters_client_side(mock_client):
    client, mock_index = mock_client
    metadata = _metadata("d1", 0, "hello")
    metadata["legal_regulation"] = "GDPR"
    mock_index.list.return_value = iter([_list_page("c1")])
    mock_index.fetch.return_value = MagicMock(vectors={"c1": MagicMock(metadata=metadata)})
    store = PineconeChunkStore(client, embedding_dimension=3)
    chunks = store.get_by_legal_metadata({"regulation": "GDPR"})
    assert [c.chunk_id for c in chunks] == ["c1"]
    assert store.get_by_legal_metadata({"regulation": "CCPA"}) == []


def test_get_by_legal_metadata_rejects_unknown_filter_key(mock_client):
    client, _ = mock_client
    store = PineconeChunkStore(client, embedding_dimension=3)
    with pytest.raises(ValueError, match="unknown legal metadata filter key"):
        store.get_by_legal_metadata({"not_a_real_field": "x"})
