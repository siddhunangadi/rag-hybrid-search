from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.base import VectorStore
from rag_hybrid_search.storage.pinecone_connection import PineconeConnection


class PineconeVectorStore(VectorStore):
    def __init__(self, client: PineconeConnection):
        self._index = client.index

    def upsert(self, chunk_id: str, embedding_record: EmbeddingRecord) -> None:
        # index.update(), not index.upsert(): the real ingestion order
        # (rag_hybrid_search/ingestion/pipeline.py:101,104) always calls
        # chunk_store.put() before vector_store.upsert() for a given chunk,
        # so by the time this runs, PineconeChunkStore.put() has already
        # created the record (with a placeholder vector -- see
        # PineconeChunkStore.put()). update() sets the real vector values on
        # that existing record without touching its metadata.
        self._index.update(id=chunk_id, values=embedding_record.embedding)

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        result = self._index.query(vector=embedding, top_k=k, include_metadata=False)
        return [(m.id, m.score) for m in result.matches]

    def delete(self, chunk_ids: list[str]) -> None:
        self._index.delete(ids=chunk_ids)
