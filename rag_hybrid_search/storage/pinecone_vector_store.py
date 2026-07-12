from concurrent.futures import ThreadPoolExecutor

from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.base import VectorStore
from rag_hybrid_search.storage.pinecone_connection import PineconeConnection

# Pinecone's update() endpoint takes exactly one vector id per call -- there
# is no batch form. Running these concurrently (rather than one at a time in
# a loop) is what actually cuts ingestion wall-clock time for this step,
# since each call is independent (touches only its own vector id).
_MAX_CONCURRENT_UPDATES = 20


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

    def upsert_many(self, chunk_ids: list[str], embedding_records: list[EmbeddingRecord]) -> None:
        # Same per-id update() call as upsert(), issued concurrently instead
        # of sequentially -- there's no ordering dependency between chunks
        # at this point (chunk_store.put_many() already created every
        # record's placeholder before this runs), so this is safe to
        # parallelize. Any exception from a worker is re-raised here so
        # IndexManager.index()'s existing try/except still sees a failure.
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_UPDATES) as executor:
            futures = [
                executor.submit(self.upsert, chunk_id, record)
                for chunk_id, record in zip(chunk_ids, embedding_records)
            ]
            for future in futures:
                future.result()

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        result = self._index.query(vector=embedding, top_k=k, include_metadata=False)
        return [(m.id, m.score) for m in result.matches]

    def delete(self, chunk_ids: list[str]) -> None:
        self._index.delete(ids=chunk_ids)
