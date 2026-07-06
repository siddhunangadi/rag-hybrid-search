import sqlite3
from typing import Iterator, Optional

from rag_hybrid_search.compliance.regulation_models import LegalMetadata
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
    source_path TEXT,
    legal_regulation TEXT,
    legal_version TEXT,
    legal_jurisdiction TEXT,
    legal_article TEXT,
    legal_section TEXT,
    legal_clause TEXT,
    legal_effective_date TEXT,
    legal_document_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
"""

_LEGAL_FILTER_COLUMNS = {
    "regulation": "legal_regulation",
    "version": "legal_version",
    "jurisdiction": "legal_jurisdiction",
    "article": "legal_article",
    "section": "legal_section",
    "clause": "legal_clause",
    "document_type": "legal_document_type",
}


_LEGAL_COLUMNS = [
    "legal_regulation",
    "legal_version",
    "legal_jurisdiction",
    "legal_article",
    "legal_section",
    "legal_clause",
    "legal_effective_date",
    "legal_document_type",
]


class SqliteChunkStore(ChunkStore):
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_legal_columns()
        self._conn.commit()

    def _migrate_legal_columns(self) -> None:
        """Defensive migration: adds legal_* columns to a chunks table created
        by the pre-legal-metadata schema (CREATE TABLE IF NOT EXISTS is a no-op
        against an existing table, so old databases never gain these columns)."""
        for column in _LEGAL_COLUMNS:
            try:
                self._conn.execute(f"ALTER TABLE chunks ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc):
                    raise
        self._conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_legal_regulation ON chunks(legal_regulation);
            CREATE INDEX IF NOT EXISTS idx_chunks_legal_jurisdiction ON chunks(legal_jurisdiction);
            CREATE INDEX IF NOT EXISTS idx_chunks_legal_article ON chunks(legal_article);
            CREATE INDEX IF NOT EXISTS idx_chunks_legal_document_type ON chunks(legal_document_type);
            """
        )

    def put(self, chunk: Chunk, source_path: Optional[str] = None) -> None:
        lm = chunk.legal_metadata
        self._conn.execute(
            """
            INSERT INTO chunks
                (chunk_id, document_id, chunk_index, text, strategy_version,
                 heading, page, char_count, source_path,
                 legal_regulation, legal_version, legal_jurisdiction, legal_article,
                 legal_section, legal_clause, legal_effective_date, legal_document_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_index=excluded.chunk_index,
                text=excluded.text,
                strategy_version=excluded.strategy_version,
                heading=excluded.heading,
                page=excluded.page,
                char_count=excluded.char_count,
                source_path=COALESCE(excluded.source_path, chunks.source_path),
                legal_regulation=excluded.legal_regulation,
                legal_version=excluded.legal_version,
                legal_jurisdiction=excluded.legal_jurisdiction,
                legal_article=excluded.legal_article,
                legal_section=excluded.legal_section,
                legal_clause=excluded.legal_clause,
                legal_effective_date=excluded.legal_effective_date,
                legal_document_type=excluded.legal_document_type
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
                lm.regulation if lm else None,
                lm.version if lm else None,
                lm.jurisdiction if lm else None,
                lm.article if lm else None,
                lm.section if lm else None,
                lm.clause if lm else None,
                lm.effective_date.isoformat() if lm and lm.effective_date else None,
                lm.document_type if lm else None,
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

    def get_by_legal_metadata(self, filters: dict[str, str]) -> list[Chunk]:
        """Indexed lookup by LegalMetadata fields (regulation/version/jurisdiction/
        article/section/clause/document_type). Unknown filter keys raise ValueError."""
        if not filters:
            return []
        clauses = []
        params = []
        for key, value in filters.items():
            column = _LEGAL_FILTER_COLUMNS.get(key)
            if column is None:
                raise ValueError(f"unknown legal metadata filter key: {key!r}")
            clauses.append(f"{column} = ?")
            params.append(value)
        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE {where} ORDER BY document_id, chunk_index", params
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_document_summaries(self) -> list[dict]:
        """Aggregate chunk counts per document, for corpus-wide stats endpoints."""
        rows = self._conn.execute(
            """
            SELECT document_id, source_path, COUNT(*) as chunk_count
            FROM chunks
            GROUP BY document_id
            ORDER BY document_id
            """
        ).fetchall()
        return [
            {
                "document_id": row["document_id"],
                "source_path": row["source_path"],
                "chunk_count": row["chunk_count"],
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        legal_metadata = None
        if any(row[col] is not None for col in _LEGAL_COLUMNS):
            legal_metadata = LegalMetadata(
                document_id=row["document_id"],
                document_title=row["document_id"],
                regulation=row["legal_regulation"],
                version=row["legal_version"],
                jurisdiction=row["legal_jurisdiction"],
                article=row["legal_article"],
                section=row["legal_section"],
                clause=row["legal_clause"],
                effective_date=row["legal_effective_date"] or None,
                document_type=row["legal_document_type"],
            )
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            strategy_version=row["strategy_version"],
            heading=row["heading"],
            page=row["page"],
            char_count=row["char_count"],
            legal_metadata=legal_metadata,
        )
