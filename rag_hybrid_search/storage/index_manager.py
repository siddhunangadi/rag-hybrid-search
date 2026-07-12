import logging

from rag_hybrid_search.models import Chunk, EmbeddingRecord, IndexStatus
from rag_hybrid_search.storage.base import ChunkStore, VectorStore
from rag_hybrid_search.storage.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class IndexManager:
    def __init__(
        self,
        chunk_store: ChunkStore,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ):
        self.chunk_store = chunk_store
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def index(
        self, chunks: list[Chunk], embeddings: list[EmbeddingRecord]
    ) -> IndexStatus:
        try:
            self.vector_store.upsert_many([c.chunk_id for c in chunks], embeddings)
            self.rebuild_bm25_index()
        except Exception:
            # Previously swallowed silently with no log at all -- meant a
            # real vector_store/chunk_store write failure (Pinecone or
            # otherwise) produced zero trace of what went wrong, only the
            # generic FAILED status the caller sees.
            logger.exception("IndexManager.index() failed")
            return IndexStatus.FAILED
        return IndexStatus.READY

    def remove_document(self, document_id: str) -> None:
        chunks = self.chunk_store.get_by_document(document_id)
        chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_store.delete_by_document(document_id)
        if chunk_ids:
            self.vector_store.delete(chunk_ids)
        self.rebuild_bm25_index()

    def rebuild_bm25_index(self) -> None:
        all_chunks = list(self.chunk_store.all())
        self.bm25_index.build(all_chunks)
        self.bm25_index.save()

    def rebuild_all(self) -> None:
        self.rebuild_bm25_index()

    def verify_sync(self) -> list[str]:
        chunk_ids = {c.chunk_id for c in self.chunk_store.all()}
        bm25_ids = set(self.bm25_index._chunk_ids)
        return sorted(chunk_ids.symmetric_difference(bm25_ids))
