from datetime import datetime, timezone

from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.ingestion.dedup import is_duplicate
from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Chunk, EmbeddingRecord, IndexStatus
from rag_hybrid_search.providers.base import EmbeddingProvider
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager


class IngestionPipeline:
    def __init__(
        self,
        loader: Loader,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        chunk_store: ChunkStore,
        index_manager: IndexManager,
        dedup_cosine_threshold: float,
        dedup_text_threshold: float,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.chunk_store = chunk_store
        self.index_manager = index_manager
        self._dedup_cosine_threshold = dedup_cosine_threshold
        self._dedup_text_threshold = dedup_text_threshold

    def ingest(self, path: str) -> IndexStatus:
        document = self.loader.load(path)

        existing_hash = self.chunk_store.get_document_hash(path)
        if existing_hash == document.document_id:
            return IndexStatus.READY
        if existing_hash is not None:
            self.index_manager.remove_document(existing_hash)

        new_chunks = self.chunker.chunk(document)
        if not new_chunks:
            return IndexStatus.READY

        embeddings = self.embedding_provider.embed([c.text for c in new_chunks])
        existing_pairs = self._existing_chunk_embeddings()

        surviving_chunks: list[Chunk] = []
        surviving_records: list[EmbeddingRecord] = []
        for chunk, embedding in zip(new_chunks, embeddings):
            if is_duplicate(
                chunk,
                embedding,
                existing_pairs,
                self._dedup_cosine_threshold,
                self._dedup_text_threshold,
            ):
                continue
            record = EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                embedding_model=self.embedding_provider.model_name,
                embedding_dimension=self.embedding_provider.dimension,
                provider=type(self.embedding_provider).__name__,
                created_at=datetime.now(timezone.utc),
            )
            surviving_chunks.append(chunk)
            surviving_records.append(record)
            existing_pairs.append((chunk, embedding))

        if not surviving_chunks:
            return IndexStatus.READY

        for chunk in surviving_chunks:
            self.chunk_store.put(chunk, source_path=path)

        return self.index_manager.index(surviving_chunks, surviving_records)

    def _existing_chunk_embeddings(self) -> list[tuple[Chunk, list[float]]]:
        existing_chunks = list(self.chunk_store.all())
        if not existing_chunks:
            return []
        texts = [c.text for c in existing_chunks]
        embeddings = self.embedding_provider.embed(texts)
        return list(zip(existing_chunks, embeddings))
