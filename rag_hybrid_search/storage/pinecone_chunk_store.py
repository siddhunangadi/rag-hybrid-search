"""PineconeChunkStore: metadata-only operations against the same Pinecone
index PineconeVectorStore writes vectors to (shared via PineconeConnection).

Metadata sentinels: Optional[str] fields (heading, source_path) store "" for
None, Optional[int] (page) stores -1 for None -- both reconstructed back to
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
from typing import Iterator, Optional

from rag_hybrid_search.models import Chunk
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.pinecone_connection import PineconeConnection

_LEGAL_FILTER_KEYS = {
    "regulation", "version", "jurisdiction", "article", "section",
    "clause", "document_type",
}


# Pinecone's per-vector metadata limit is 40KB (serialized). Chunk text is
# almost all of this payload's size -- everything else here is short fields.
_MAX_METADATA_BYTES = 40 * 1024


class MetadataTooLargeError(ValueError):
    pass


def _chunk_to_metadata(chunk: Chunk, source_path: Optional[str] = None) -> dict:
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
            "legal_version": lm.version or "",
            "legal_jurisdiction": lm.jurisdiction or "",
            "legal_article": lm.article or "",
            "legal_section": lm.section or "",
            "legal_clause": lm.clause or "",
            "legal_document_type": lm.document_type or "",
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


def _metadata_to_chunk(chunk_id: str, metadata: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=metadata["document_id"],
        chunk_index=metadata["chunk_index"],
        text=metadata["text"],
        strategy_version=metadata["strategy_version"],
        heading=metadata["heading"] or None,
        page=metadata["page"] if metadata["page"] != -1 else None,
        char_count=metadata["char_count"],
    )


class PineconeChunkStore(ChunkStore):
    def __init__(self, client: PineconeConnection, embedding_dimension: int):
        self._index = client.index
        # Needed only for put()'s placeholder-vector creation path below --
        # confirm this matches the real embedding provider's output
        # dimension (e.g. NVIDIA embedding model's dimension) when wiring
        # this up in Task 3, not an arbitrary guess.
        self._embedding_dimension = embedding_dimension

    def put(self, chunk: Chunk, source_path: Optional[str] = None) -> None:
        # upsert(), not update(): the real ingestion order
        # (rag_hybrid_search/ingestion/pipeline.py:101,104) always calls
        # chunk_store.put() BEFORE vector_store.upsert() for a given chunk,
        # so this is usually the first write for a new chunk_id -- the
        # record may not exist yet, and index.update() would fail/no-op on
        # a nonexistent id. upsert() with a placeholder zero-vector creates
        # the record (or overwrites metadata on an existing one, e.g. a
        # re-ingested document), and PineconeVectorStore.upsert() -- called
        # second -- then update()s in the real vector values without
        # touching this metadata.
        metadata = _chunk_to_metadata(chunk, source_path)
        self._index.upsert(vectors=[{
            "id": chunk.chunk_id,
            "values": [0.0] * self._embedding_dimension,
            "metadata": metadata,
        }])

    def get(self, chunk_id: str) -> Optional[Chunk]:
        result = self._index.fetch(ids=[chunk_id])
        if chunk_id not in result.vectors:
            return None
        return _metadata_to_chunk(chunk_id, result.vectors[chunk_id].metadata)

    def _scan_all(self) -> Iterator[tuple[str, dict]]:
        for id_batch in self._index.list():
            fetched = self._index.fetch(ids=id_batch)
            for chunk_id, vector in fetched.vectors.items():
                yield chunk_id, vector.metadata

    def get_by_document(self, document_id: str) -> list[Chunk]:
        chunks = [
            _metadata_to_chunk(chunk_id, metadata)
            for chunk_id, metadata in self._scan_all()
            if metadata.get("document_id") == document_id
        ]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def get_document_hash(self, source_path: str) -> Optional[str]:
        for chunk_id, metadata in self._scan_all():
            if metadata.get("source_path") == source_path:
                return metadata.get("document_id")
        return None

    def delete_by_document(self, document_id: str) -> None:
        self._index.delete(filter={"document_id": {"$eq": document_id}})

    def all(self) -> Iterator[Chunk]:
        for chunk_id, metadata in self._scan_all():
            yield _metadata_to_chunk(chunk_id, metadata)

    def get_by_legal_metadata(self, filters: dict[str, str]) -> list[Chunk]:
        if not filters:
            return []
        unknown = set(filters) - _LEGAL_FILTER_KEYS
        if unknown:
            raise ValueError(f"unknown legal metadata filter key: {next(iter(unknown))!r}")
        matched = []
        for chunk_id, metadata in self._scan_all():
            if all(metadata.get(f"legal_{key}") == value for key, value in filters.items()):
                matched.append(_metadata_to_chunk(chunk_id, metadata))
        return sorted(matched, key=lambda c: (c.document_id, c.chunk_index))

    def get_document_summaries(self) -> list[dict]:
        counts: dict[str, dict] = {}
        for chunk_id, metadata in self._scan_all():
            document_id = metadata.get("document_id")
            entry = counts.setdefault(
                document_id,
                {"document_id": document_id, "source_path": metadata.get("source_path"), "chunk_count": 0},
            )
            entry["chunk_count"] += 1
        return sorted(counts.values(), key=lambda d: d["document_id"])
