import logging
from datetime import datetime, timezone

from rag_hybrid_search.diagnostics import rss_mb
from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.ingestion.dedup import is_duplicate
from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Chunk, EmbeddingRecord, IndexStatus
from rag_hybrid_search.providers.base import EmbeddingProvider
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager

logger = logging.getLogger(__name__)


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
        logger.info("ingest: start path=%s rss_mb=%.1f", path, rss_mb())
        document = self.loader.load(path)
        logger.info(
            "ingest: loaded document_id=%s format=%s chars=%d rss_mb=%.1f",
            document.document_id, document.format, len(document.content), rss_mb(),
        )

        existing_hash = self.chunk_store.get_document_hash(path)
        if existing_hash == document.document_id:
            logger.info("ingest: unchanged, skipping path=%s", path)
            return IndexStatus.READY
        if existing_hash is not None:
            logger.info("ingest: content changed, removing old document_id=%s", existing_hash)
            self.index_manager.remove_document(existing_hash)

        new_chunks = self.chunker.chunk(document)
        logger.info(
            "ingest: chunked into %d chunks (strategy=%s) rss_mb=%.1f",
            len(new_chunks), self.chunker.__class__.__name__, rss_mb(),
        )
        if not new_chunks:
            logger.info("ingest: no chunks produced, done path=%s", path)
            return IndexStatus.READY
        logger.debug(
            "ingest: chunk previews %s",
            [(c.chunk_id, c.chunk_index, c.char_count, c.text[:80]) for c in new_chunks],
        )

        embeddings = self.embedding_provider.embed([c.text for c in new_chunks])
        logger.info(
            "ingest: embedded %d chunks with provider=%s model=%s dim=%d rss_mb=%.1f",
            len(embeddings), type(self.embedding_provider).__name__,
            self.embedding_provider.model_name, self.embedding_provider.dimension, rss_mb(),
        )
        existing_pairs = self._existing_chunk_embeddings()
        logger.debug("ingest: comparing against %d existing chunks for dedup", len(existing_pairs))

        surviving_chunks: list[Chunk] = []
        surviving_records: list[EmbeddingRecord] = []
        dropped = 0
        for chunk, embedding in zip(new_chunks, embeddings):
            if is_duplicate(
                chunk,
                embedding,
                existing_pairs,
                self._dedup_cosine_threshold,
                self._dedup_text_threshold,
            ):
                dropped += 1
                logger.debug("ingest: dropped duplicate chunk_id=%s index=%d", chunk.chunk_id, chunk.chunk_index)
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

        logger.info("ingest: dedup dropped %d/%d chunks, %d survive", dropped, len(new_chunks), len(surviving_chunks))
        if not surviving_chunks:
            logger.info("ingest: nothing new to store, done path=%s", path)
            return IndexStatus.READY

        for chunk in surviving_chunks:
            self.chunk_store.put(chunk, source_path=path)
        logger.info(
            "ingest: stored %d chunks in chunk_store rss_mb=%.1f",
            len(surviving_chunks), rss_mb(),
        )

        status = self.index_manager.index(surviving_chunks, surviving_records)
        logger.info("ingest: index_manager.index() returned rss_mb=%.1f", rss_mb())
        logger.info("ingest: indexed %d chunks, status=%s path=%s", len(surviving_chunks), status, path)
        return status

    def _existing_chunk_embeddings(self) -> list[tuple[Chunk, list[float]]]:
        # chunk_store already has the embedding for every existing chunk
        # (it was computed once, when that chunk was first ingested) --
        # re-embedding them here on every ingest() call would be pure waste
        # scaling with corpus size, so reuse what's already stored instead.
        return [
            (item.chunk, item.embedding)
            for item in self.chunk_store.all_with_embeddings()
        ]
