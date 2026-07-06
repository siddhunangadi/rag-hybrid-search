import sqlite3
import tempfile
from datetime import date

from rag_hybrid_search.compliance.regulation_models import LegalMetadata
from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore


def _chunk_with_metadata(chunk_id: str, **legal_kwargs) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        chunk_index=0,
        text="Personal data shall be processed lawfully.",
        strategy_version="clause-v1",
        char_count=42,
        legal_metadata=LegalMetadata(document_id="doc-1", document_title="GDPR", **legal_kwargs),
    )


def test_put_and_get_round_trips_legal_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        chunk = _chunk_with_metadata("chunk-1", regulation="GDPR", article="17", jurisdiction="EU")
        store.put(chunk, source_path="/tmp/gdpr.pdf")

        fetched = store.get("chunk-1")
        assert fetched.legal_metadata is not None
        assert fetched.legal_metadata.regulation == "GDPR"
        assert fetched.legal_metadata.article == "17"
        assert fetched.legal_metadata.jurisdiction == "EU"


def test_chunk_without_legal_metadata_round_trips_as_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        chunk = Chunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            chunk_index=0,
            text="plain text",
            strategy_version="fixed-v1",
            char_count=10,
        )
        store.put(chunk, source_path="/tmp/plain.txt")

        fetched = store.get("chunk-2")
        assert fetched.legal_metadata is None


def test_get_by_legal_metadata_filters_on_indexed_fields():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        store.put(_chunk_with_metadata("chunk-3", regulation="GDPR", article="5"), source_path="/tmp/a.pdf")
        store.put(_chunk_with_metadata("chunk-4", regulation="HIPAA", article=None), source_path="/tmp/b.pdf")

        results = store.get_by_legal_metadata({"regulation": "GDPR"})
        assert [c.chunk_id for c in results] == ["chunk-3"]

        results = store.get_by_legal_metadata({"regulation": "GDPR", "article": "5"})
        assert [c.chunk_id for c in results] == ["chunk-3"]

        results = store.get_by_legal_metadata({"regulation": "PCI-DSS"})
        assert results == []


def test_chunk_with_only_jurisdiction_set_round_trips_as_non_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        chunk = _chunk_with_metadata("chunk-5", jurisdiction="EU")
        store.put(chunk, source_path="/tmp/jurisdiction-only.pdf")

        fetched = store.get("chunk-5")
        assert fetched.legal_metadata is not None
        assert fetched.legal_metadata.jurisdiction == "EU"
        assert fetched.legal_metadata.regulation is None
        assert fetched.legal_metadata.version is None
        assert fetched.legal_metadata.article is None
        assert fetched.legal_metadata.section is None
        assert fetched.legal_metadata.clause is None
        assert fetched.legal_metadata.effective_date is None
        assert fetched.legal_metadata.document_type is None


def test_chunk_with_only_version_set_round_trips_as_non_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        chunk = _chunk_with_metadata("chunk-6", version="2018")
        store.put(chunk, source_path="/tmp/version-only.pdf")

        fetched = store.get("chunk-6")
        assert fetched.legal_metadata is not None
        assert fetched.legal_metadata.version == "2018"
        assert fetched.legal_metadata.regulation is None
        assert fetched.legal_metadata.jurisdiction is None


def test_chunk_with_only_effective_date_set_round_trips_as_non_none():
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteChunkStore(db_path=f"{tmp}/chunks.db")
        chunk = _chunk_with_metadata("chunk-7", effective_date=date(2020, 1, 1))
        store.put(chunk, source_path="/tmp/effective-date-only.pdf")

        fetched = store.get("chunk-7")
        assert fetched.legal_metadata is not None
        assert fetched.legal_metadata.effective_date == date(2020, 1, 1)
        assert fetched.legal_metadata.regulation is None


def test_old_schema_database_is_migrated_on_open():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/old_chunks.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                heading TEXT,
                page INTEGER,
                char_count INTEGER NOT NULL,
                source_path TEXT
            );
            """
        )
        conn.commit()
        conn.close()

        store = SqliteChunkStore(db_path=db_path)
        chunk = _chunk_with_metadata("chunk-8", regulation="GDPR", jurisdiction="EU")
        store.put(chunk, source_path="/tmp/migrated.pdf")

        fetched = store.get("chunk-8")
        assert fetched is not None
        assert fetched.legal_metadata is not None
        assert fetched.legal_metadata.regulation == "GDPR"
        assert fetched.legal_metadata.jurisdiction == "EU"
