from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.ingestion.chunkers.windowing import sliding_window
from rag_hybrid_search.models import Chunk, Document
from rag_hybrid_search.uuid7 import uuid7


class FixedChunker(Chunker):
    version = "fixed-v1"

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = sliding_window(document.content, self._chunk_size, self._chunk_overlap)
        return [
            Chunk(
                chunk_id=uuid7(),
                document_id=document.document_id,
                chunk_index=index,
                text=piece,
                strategy_version=self.version,
                heading=None,
                page=None,
                char_count=len(piece),
            )
            for index, piece in enumerate(pieces)
        ]
