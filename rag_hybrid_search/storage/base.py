from abc import ABC, abstractmethod
from typing import Iterator, Optional

from rag_hybrid_search.models import Chunk, ChunkEmbedding, EmbeddingRecord


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunk_id: str, embedding_record: EmbeddingRecord) -> None:
        ...

    @abstractmethod
    def upsert_many(self, chunk_ids: list[str], embedding_records: list[EmbeddingRecord]) -> None:
        """Write many vectors faster than calling upsert() once per id in a
        loop -- implementations may parallelize, batch, or both, as long as
        the end result is the same as calling upsert() for every pair."""
        ...

    @abstractmethod
    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        ...


class ChunkStore(ABC):
    @abstractmethod
    def get(self, chunk_id: str) -> Optional[Chunk]:
        ...

    @abstractmethod
    def get_by_document(self, document_id: str) -> list[Chunk]:
        ...

    @abstractmethod
    def get_document_hash(self, source_path: str) -> Optional[str]:
        ...

    @abstractmethod
    def put(self, chunk: Chunk) -> None:
        ...

    @abstractmethod
    def put_many(self, chunks: list[Chunk], source_path: Optional[str] = None) -> None:
        """Write many chunks faster than calling put() once per chunk in a
        loop -- implementations may batch the underlying writes, as long as
        the end result is the same as calling put() for every chunk."""
        ...

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        ...

    @abstractmethod
    def all(self) -> Iterator[Chunk]:
        ...

    @abstractmethod
    def all_with_embeddings(self) -> Iterator[ChunkEmbedding]:
        """Like all(), but also returns each chunk's already-computed
        embedding when the backing store already has it available (e.g.
        alongside the vector it stores), so callers like ingestion dedup
        don't need to recompute embeddings for existing chunks."""
        ...

    @abstractmethod
    def get_by_legal_metadata(self, filters: dict[str, str]) -> list[Chunk]:
        ...

    @abstractmethod
    def get_document_summaries(self) -> list[dict]:
        ...
