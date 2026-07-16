import logging
import uuid

from rag_hybrid_search.audit import AuditEvent, AuditLog, now_utc
from rag_hybrid_search.models import Chunk, EmbeddingRecord, IndexStatus
from rag_hybrid_search.storage.base import ChunkStore, VectorStore
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.pinecone_chunk_store import PineconeChunkStore
from rag_hybrid_search.storage.pinecone_vector_store import PineconeVectorStore

logger = logging.getLogger(__name__)


class IndexManager:
    def __init__(
        self,
        chunk_store: ChunkStore,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        audit_log: AuditLog | None = None,
    ):
        self.chunk_store = chunk_store
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.audit_log = audit_log

    def index(
        self, chunks: list[Chunk], embeddings: list[EmbeddingRecord], rebuild_bm25: bool = True,
    ) -> IndexStatus:
        try:
            self.vector_store.upsert_many([c.chunk_id for c in chunks], embeddings)
            if rebuild_bm25:
                self.rebuild_bm25_index()
        except Exception:
            # Previously swallowed silently with no log at all -- meant a
            # real vector_store/chunk_store write failure (Pinecone or
            # otherwise) produced zero trace of what went wrong, only the
            # generic FAILED status the caller sees.
            logger.exception("IndexManager.index() failed")
            return IndexStatus.FAILED
        self._detect_and_mark_superseded(chunks)
        return IndexStatus.READY

    def supports_combined_write(self) -> bool:
        # True only when chunk_store and vector_store are the Pinecone
        # classes AND both point at the same underlying Pinecone index --
        # the same in-memory fake index in tests (see tests/fakes.py's
        # fake_pinecone_stores()), a real shared index in production. Any
        # other pairing (a non-Pinecone backend, or two separate indexes)
        # falls back to the existing two-stage index()/put_many() path.
        return (
            isinstance(self.chunk_store, PineconeChunkStore)
            and isinstance(self.vector_store, PineconeVectorStore)
            and self.chunk_store._index is self.vector_store._index
        )

    def index_combined(
        self, chunks: list[Chunk], embeddings: list[EmbeddingRecord],
        source_path: str | None = None, rebuild_bm25: bool = True,
    ) -> IndexStatus:
        """Pinecone fast path: writes metadata + real vectors in one upsert
        per batch instead of chunk_store.put_many() (placeholder upsert)
        followed by vector_store.upsert_many() (per-id update(), no batch
        form). Caller (IngestionPipeline) is responsible for checking
        supports_combined_write() first and skipping its own
        chunk_store.put_many() call when using this method instead."""
        try:
            self.chunk_store.put_many_with_embeddings(chunks, embeddings, source_path=source_path)
            if rebuild_bm25:
                self.rebuild_bm25_index()
        except Exception:
            logger.exception("IndexManager.index_combined() failed")
            return IndexStatus.FAILED
        self._detect_and_mark_superseded(chunks)
        return IndexStatus.READY

    def remove_document(self, document_id: str, rebuild_bm25: bool = True) -> None:
        chunks = self.chunk_store.get_by_document(document_id)
        chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_store.delete_by_document(document_id)
        if chunk_ids:
            self.vector_store.delete(chunk_ids)
        if rebuild_bm25:
            self.rebuild_bm25_index()

    def rebuild_bm25_index(self) -> None:
        all_chunks = list(self.chunk_store.all())
        self.bm25_index.build(all_chunks)
        self.bm25_index.save()

    def rebuild_all(self) -> None:
        self.rebuild_bm25_index()

    def _detect_and_mark_superseded(self, chunks: list[Chunk]) -> None:
        """Compares each newly-indexed compliance chunk against any other
        indexed chunk sharing the same regulation/authority/jurisdiction/
        article/section/clause identity, and flips is_current/superseded_by
        so only the chunk with the latest effective_date stays current.

        Skipped entirely for chunks with no legal_metadata, no regulation,
        no effective_date, or no article/section -- i.e. every non-compliance
        document behaves exactly as it did before this method existed.
        """
        seen_keys: set[tuple] = set()
        for chunk in chunks:
            lm = chunk.legal_metadata
            if not lm or not lm.regulation or not lm.effective_date or not (lm.article or lm.section):
                continue
            key = (lm.regulation, lm.authority, lm.jurisdiction, lm.article, lm.section, lm.clause)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            field_names = ("regulation", "authority", "jurisdiction", "article", "section", "clause")
            filters = {name: value for name, value in zip(field_names, key) if value is not None}
            candidates = self.chunk_store.get_by_legal_metadata(filters)
            dated = [c for c in candidates if c.legal_metadata and c.legal_metadata.effective_date]
            if len(dated) < 2:
                continue

            latest_date = max(c.legal_metadata.effective_date for c in dated)
            latest_doc_ids = {c.document_id for c in dated if c.legal_metadata.effective_date == latest_date}
            # Ambiguous (two docs share the latest date): leave is_current as-is
            # rather than guessing which one wins.
            winner_doc_id = next(iter(latest_doc_ids)) if len(latest_doc_ids) == 1 else None
            if winner_doc_id is None:
                continue

            for candidate in dated:
                should_be_current = candidate.document_id == winner_doc_id
                if candidate.legal_metadata.is_current != should_be_current:
                    self.chunk_store.update_legal_metadata(
                        candidate.chunk_id,
                        is_current=should_be_current,
                        superseded_by=None if should_be_current else winner_doc_id,
                    )
                    if self.audit_log is not None and not should_be_current:
                        lm = candidate.legal_metadata
                        self.audit_log.record(
                            AuditEvent(
                                event_id=str(uuid.uuid4()),
                                event_type="supersession",
                                timestamp=now_utc(),
                                request_id="internal",
                                key_id="system",
                                endpoint="internal:index_manager",
                                action="mark_superseded",
                                status="success",
                                document_id=candidate.document_id,
                                regulation_metadata={
                                    "regulation": lm.regulation,
                                    "authority": lm.authority,
                                    "jurisdiction": lm.jurisdiction,
                                    "article": lm.article,
                                    "section": lm.section,
                                    "clause": lm.clause,
                                    "superseded_by": winner_doc_id,
                                },
                            )
                        )

    def verify_sync(self) -> list[str]:
        chunk_ids = {c.chunk_id for c in self.chunk_store.all()}
        bm25_ids = set(self.bm25_index._chunk_ids)
        return sorted(chunk_ids.symmetric_difference(bm25_ids))
