"""PineconeChunkStore: metadata-only operations against the same Pinecone
index PineconeVectorStore writes vectors to (shared via PineconeConnection).

Metadata sentinels: str | None fields (heading, source_path) store "" for
None, int | None (page) stores -1 for None -- both reconstructed back to
None on read, since Pinecone metadata doesn't reliably support None values
across all SDK/client versions.

Query-shape finding (Task 2 Step 2, against pinecone==9.1.0): Index.query()
is a similarity-search primitive -- vector/id/sparse_vector are all optional
in its type signature, but the Pinecone query endpoint has no supported mode
with none of the three provided. Index.list() only supports prefix/limit, no
metadata filter. So get_by_document/get_document_hash/get_by_legal_metadata
below use the same list()-then-fetch()-then-filter-client-side scan as
all()/get_document_summaries/delete_by_document (delete_by_document is the
one exception: Pinecone's delete endpoint does accept a metadata filter
directly, unlike query/list).
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from rag_hybrid_search.compliance.regulation_models import LegalMetadata
from rag_hybrid_search.models import Chunk, ChunkEmbedding, EmbeddingRecord
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.pinecone_connection import PineconeConnection

_LEGAL_FILTER_KEYS = {
    "regulation", "authority", "version", "jurisdiction", "article", "section",
    "clause", "document_type", "risk_category",
}

# Not equality-matched against metadata directly -- get_by_legal_metadata()
# interprets these specially: is_current ("true"/"false") filters on the
# is_current flag, as_of_date (ISO date string) keeps only the
# latest-effective-dated match at or before that date.
_RESERVED_FILTER_KEYS = {"is_current", "as_of_date"}


# Pinecone's per-vector metadata limit is 40KB (serialized). Chunk text is
# almost all of this payload's size -- everything else here is short fields.
_MAX_METADATA_BYTES = 40 * 1024

# Pinecone rejects an all-zero dense vector as invalid -- put()'s placeholder
# vector needs one non-zero component, negligible enough not to skew cosine
# similarity before PineconeVectorStore.upsert() overwrites it for real.
_PLACEHOLDER_EPSILON = 1e-6

# Conservative batch size for a single index.upsert() call -- keeps well
# under Pinecone's per-request size limits even for chunks with large text
# payloads (see _MAX_METADATA_BYTES above), while still turning what used
# to be one network round-trip per chunk into one every N chunks.
_UPSERT_BATCH_SIZE = 100


class MetadataTooLargeError(ValueError):
    pass


def _chunk_to_metadata(chunk: Chunk, source_path: str | None = None) -> dict:
    lm = chunk.legal_metadata
    metadata = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "strategy_version": chunk.strategy_version,
        "heading": chunk.heading or "",
        "page": chunk.page if chunk.page is not None else -1,
        "char_count": chunk.char_count,
        "source_path": source_path or "",
    }
    if lm:
        metadata.update({
            "legal_regulation": lm.regulation or "",
            "legal_authority": lm.authority or "",
            "legal_version": lm.version or "",
            "legal_jurisdiction": lm.jurisdiction or "",
            "legal_article": lm.article or "",
            "legal_section": lm.section or "",
            "legal_clause": lm.clause or "",
            "legal_effective_date": lm.effective_date.isoformat() if lm.effective_date else "",
            "legal_document_type": lm.document_type or "",
            "legal_risk_category": lm.risk_category or "",
            "legal_is_current": lm.is_current,
            "legal_superseded_by": lm.superseded_by or "",
        })
    import json
    serialized_size = len(json.dumps(metadata).encode("utf-8"))
    if serialized_size > _MAX_METADATA_BYTES:
        raise MetadataTooLargeError(
            f"chunk {chunk.chunk_id!r} metadata is {serialized_size} bytes, "
            f"over Pinecone's {_MAX_METADATA_BYTES}-byte per-vector limit -- "
            f"reduce chunk_size in Settings, since this project's chunker "
            f"controls chunk text length directly."
        )
    return metadata


_LEGAL_ROUND_TRIP_FIELDS = _LEGAL_FILTER_KEYS | {"effective_date"}


def _metadata_to_chunk(chunk_id: str, metadata: dict) -> Chunk:
    legal_metadata = None
    if any(metadata.get(f"legal_{field}") for field in _LEGAL_ROUND_TRIP_FIELDS):
        legal_metadata = LegalMetadata(
            document_id=metadata["document_id"],
            document_title=metadata["document_id"],
            regulation=metadata.get("legal_regulation") or None,
            authority=metadata.get("legal_authority") or None,
            version=metadata.get("legal_version") or None,
            jurisdiction=metadata.get("legal_jurisdiction") or None,
            article=metadata.get("legal_article") or None,
            section=metadata.get("legal_section") or None,
            clause=metadata.get("legal_clause") or None,
            effective_date=metadata.get("legal_effective_date") or None,
            document_type=metadata.get("legal_document_type") or None,
            risk_category=metadata.get("legal_risk_category") or None,
            is_current=metadata.get("legal_is_current", True),
            superseded_by=metadata.get("legal_superseded_by") or None,
        )
    return Chunk(
        chunk_id=chunk_id,
        document_id=metadata["document_id"],
        chunk_index=metadata["chunk_index"],
        text=metadata["text"],
        strategy_version=metadata["strategy_version"],
        heading=metadata["heading"] or None,
        page=metadata["page"] if metadata["page"] != -1 else None,
        char_count=metadata["char_count"],
        legal_metadata=legal_metadata,
    )


class PineconeChunkStore(ChunkStore):
    def __init__(self, client: PineconeConnection, embedding_dimension: int):
        self._index = client.index
        # Needed only for put()'s placeholder-vector creation path below --
        # confirm this matches the real embedding provider's output
        # dimension (e.g. NVIDIA embedding model's dimension) when wiring
        # this up in Task 3, not an arbitrary guess.
        self._embedding_dimension = embedding_dimension

    def ping(self) -> None:
        """Cheap connectivity check for readiness probes.

        ``describe_index_stats`` is a lightweight metadata call (no vector
        search), so this is safe to run on every /health/ready request.
        Raises on failure; callers decide how to report that.
        """
        self._index.describe_index_stats()

    def put(self, chunk: Chunk, source_path: str | None = None) -> None:
        # upsert(), not update(): the real ingestion order
        # (rag_hybrid_search/ingestion/pipeline.py:101,104) always calls
        # chunk_store.put() BEFORE vector_store.upsert() for a given chunk,
        # so this is usually the first write for a new chunk_id -- the
        # record may not exist yet, and index.update() would fail/no-op on
        # a nonexistent id. upsert() with a placeholder vector creates
        # the record (or overwrites metadata on an existing one, e.g. a
        # re-ingested document), and PineconeVectorStore.upsert() -- called
        # second -- then update()s in the real vector values without
        # touching this metadata.
        metadata = _chunk_to_metadata(chunk, source_path)
        self._index.upsert(vectors=[{
            "id": chunk.chunk_id,
            # Pinecone rejects an all-zero dense vector ("Dense vectors must
            # contain at least one non-zero value") -- discovered against a
            # real index, not covered by mocked unit tests. A single tiny
            # epsilon in the first component satisfies that check while
            # staying negligible for cosine similarity; PineconeVectorStore
            # .upsert() (called second, per the ordering above) immediately
            # overwrites this with the real vector via index.update().
            "values": [_PLACEHOLDER_EPSILON] + [0.0] * (self._embedding_dimension - 1),
            "metadata": metadata,
        }])

    def put_many(self, chunks: list[Chunk], source_path: str | None = None) -> None:
        # Same placeholder-vector write as put(), batched into as few
        # index.upsert() calls as possible instead of one call per chunk --
        # ingesting a document with a few hundred chunks used to mean a few
        # hundred sequential network round-trips just for this step.
        #
        # The batches themselves are dispatched concurrently, not one after
        # another: each batch writes disjoint chunk ids, so there's no
        # ordering dependency between them, and observed against a real
        # index, a single ~100-vector batch's write-processing time (not
        # round-trip latency) was the dominant cost -- running batches
        # sequentially just adds them up instead of overlapping them.
        placeholder_values = [_PLACEHOLDER_EPSILON] + [0.0] * (self._embedding_dimension - 1)
        vectors = [
            {
                "id": chunk.chunk_id,
                "values": placeholder_values,
                "metadata": _chunk_to_metadata(chunk, source_path),
            }
            for chunk in chunks
        ]
        batches = [
            vectors[i : i + _UPSERT_BATCH_SIZE]
            for i in range(0, len(vectors), _UPSERT_BATCH_SIZE)
        ]
        with ThreadPoolExecutor(max_workers=len(batches) or 1) as executor:
            futures = [executor.submit(self._index.upsert, vectors=batch) for batch in batches]
            for future in futures:
                future.result()

    def put_many_with_embeddings(
        self, chunks: list[Chunk], embeddings: list[EmbeddingRecord], source_path: str | None = None,
    ) -> None:
        # Pinecone-only fast path: chunk_store and vector_store normally
        # write in two phases (placeholder upsert here, then a real-vector
        # update() per id in PineconeVectorStore) because they're separate
        # ChunkStore/VectorStore backends in the general case. When both
        # happen to share the same Pinecone index (checked by the caller --
        # see IndexManager.supports_combined_write()), the real embedding is
        # already known at metadata-write time, so this writes both in one
        # upsert -- same batching as put_many(), no placeholder, no
        # per-id update() call at all.
        vectors = [
            {
                "id": chunk.chunk_id,
                "values": record.embedding,
                "metadata": _chunk_to_metadata(chunk, source_path),
            }
            for chunk, record in zip(chunks, embeddings)
        ]
        batches = [
            vectors[i : i + _UPSERT_BATCH_SIZE]
            for i in range(0, len(vectors), _UPSERT_BATCH_SIZE)
        ]
        with ThreadPoolExecutor(max_workers=len(batches) or 1) as executor:
            futures = [executor.submit(self._index.upsert, vectors=batch) for batch in batches]
            for future in futures:
                future.result()

    def get(self, chunk_id: str) -> Chunk | None:
        result = self._index.fetch(ids=[chunk_id])
        if chunk_id not in result.vectors:
            return None
        return _metadata_to_chunk(chunk_id, result.vectors[chunk_id].metadata)

    def _scan_all(self) -> Iterator[tuple[str, dict, list[float]]]:
        # index.list() yields ListResponse pages (page.vectors is a list of
        # ListItem objects with .id), not plain id strings -- fetch() needs
        # the extracted ids, not the page object itself. Found running the
        # live test (tests/storage/test_pinecone_live.py) against a real
        # index: the original form silently iterated zero pages/vectors
        # instead of raising, since an empty page-shaped-wrong request
        # returns no matches rather than a type error.
        #
        # fetch() already returns each vector's .values (the embedding)
        # alongside .metadata in the same response -- yielding it here lets
        # all_with_embeddings() reuse it instead of a caller re-embedding.
        for page in self._index.list():
            ids = [item.id for item in page.vectors]
            if not ids:
                continue
            fetched = self._index.fetch(ids=ids)
            for chunk_id, vector in fetched.vectors.items():
                yield chunk_id, vector.metadata, vector.values

    def get_by_document(self, document_id: str) -> list[Chunk]:
        chunks = [
            _metadata_to_chunk(chunk_id, metadata)
            for chunk_id, metadata, _values in self._scan_all()
            if metadata.get("document_id") == document_id
        ]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def get_document_hash(self, source_path: str) -> str | None:
        for chunk_id, metadata, _values in self._scan_all():
            if metadata.get("source_path") == source_path:
                return metadata.get("document_id")
        return None

    def delete_by_document(self, document_id: str) -> None:
        self._index.delete(filter={"document_id": {"$eq": document_id}})

    def all(self) -> Iterator[Chunk]:
        for chunk_id, metadata, _values in self._scan_all():
            yield _metadata_to_chunk(chunk_id, metadata)

    def all_with_embeddings(self) -> Iterator[ChunkEmbedding]:
        for chunk_id, metadata, values in self._scan_all():
            yield ChunkEmbedding(
                chunk=_metadata_to_chunk(chunk_id, metadata),
                embedding=list(values),
            )

    def get_by_legal_metadata(self, filters: dict[str, str]) -> list[Chunk]:
        if not filters:
            return []
        equality_filters = {k: v for k, v in filters.items() if k not in _RESERVED_FILTER_KEYS}
        unknown = set(equality_filters) - _LEGAL_FILTER_KEYS
        if unknown:
            raise ValueError(f"unknown legal metadata filter key: {next(iter(unknown))!r}")
        matched = []
        for chunk_id, metadata, _values in self._scan_all():
            if all(metadata.get(f"legal_{key}") == value for key, value in equality_filters.items()):
                matched.append(_metadata_to_chunk(chunk_id, metadata))

        is_current = filters.get("is_current")
        if is_current is not None:
            want_current = is_current.lower() == "true"
            matched = [c for c in matched if c.legal_metadata and c.legal_metadata.is_current == want_current]

        as_of_date = filters.get("as_of_date")
        if as_of_date is not None:
            eligible = [
                c for c in matched
                if c.legal_metadata and c.legal_metadata.effective_date is not None
                and c.legal_metadata.effective_date.isoformat() <= as_of_date
            ]
            if eligible:
                latest = max(c.legal_metadata.effective_date for c in eligible)
                matched = [c for c in eligible if c.legal_metadata.effective_date == latest]
            else:
                matched = []

        return sorted(matched, key=lambda c: (c.document_id, c.chunk_index))

    def update_legal_metadata(self, chunk_id: str, is_current: bool, superseded_by: str | None) -> None:
        """Metadata-only update -- does not touch the stored vector, so
        marking an existing chunk superseded doesn't require re-embedding
        or re-upserting its full payload."""
        self._index.update(
            id=chunk_id,
            set_metadata={
                "legal_is_current": is_current,
                "legal_superseded_by": superseded_by or "",
            },
        )

    def get_document_summaries(self) -> list[dict]:
        counts: dict[str, dict] = {}
        for chunk_id, metadata, _values in self._scan_all():
            document_id = metadata.get("document_id")
            entry = counts.setdefault(
                document_id,
                {"document_id": document_id, "source_path": metadata.get("source_path"), "chunk_count": 0},
            )
            entry["chunk_count"] += 1
        return sorted(counts.values(), key=lambda d: d["document_id"])
