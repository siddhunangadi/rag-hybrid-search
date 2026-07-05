import sqlite3
from typing import Iterator, Optional

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.base import ChunkStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
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
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
"""


class SqliteChunkStore(ChunkStore):
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def put(self, chunk: Chunk, source_path: Optional[str] = None) -> None:
        self._conn.execute(
            """
            INSERT INTO chunks
                (chunk_id, document_id, chunk_index, text, strategy_version,
                 heading, page, char_count, source_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_index=excluded.chunk_index,
                text=excluded.text,
                strategy_version=excluded.strategy_version,
                heading=excluded.heading,
                page=excluded.page,
                char_count=excluded.char_count,
                source_path=COALESCE(excluded.source_path, chunks.source_path)
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.chunk_index,
                chunk.text,
                chunk.strategy_version,
                chunk.heading,
                chunk.page,
                chunk.char_count,
                source_path,
            ),
        )
        self._conn.commit()

    def get(self, chunk_id: str) -> Optional[Chunk]:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_by_document(self, document_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_document_hash(self, source_path: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT document_id FROM chunks WHERE source_path = ? LIMIT 1",
            (source_path,),
        ).fetchone()
        return row["document_id"] if row else None

    def delete_by_document(self, document_id: str) -> None:
        self._conn.execute(
            "DELETE FROM chunks WHERE document_id = ?", (document_id,)
        )
        self._conn.commit()

    def all(self) -> Iterator[Chunk]:
        rows = self._conn.execute("SELECT * FROM chunks").fetchall()
        for row in rows:
            yield self._row_to_chunk(row)

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            strategy_version=row["strategy_version"],
            heading=row["heading"],
            page=row["page"],
            char_count=row["char_count"],
        )
