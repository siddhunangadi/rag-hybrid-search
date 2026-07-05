import re

from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.models import Chunk, Document
from rag_hybrid_search.uuid7 import uuid7

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class RecursiveChunker(Chunker):
    version = "recursive-v1"

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        sections = self._split_by_headers(document.content)
        chunks: list[Chunk] = []
        index = 0
        for heading, body in sections:
            body = body.strip()
            if not body:
                continue
            for piece in self._split_by_size(body):
                chunks.append(
                    Chunk(
                        chunk_id=uuid7(),
                        document_id=document.document_id,
                        chunk_index=index,
                        text=piece,
                        strategy_version=self.version,
                        heading=heading,
                        page=None,
                        char_count=len(piece),
                    )
                )
                index += 1
        return chunks

    def _split_by_headers(self, text: str) -> list[tuple[str | None, str]]:
        matches = list(_HEADER_RE.finditer(text))
        if not matches:
            return [(None, text)]

        sections = []
        if matches[0].start() > 0:
            sections.append((None, text[: matches[0].start()]))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((heading, text[start:end]))
        return sections

    def _split_by_size(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text]
        step = self._chunk_size - self._chunk_overlap
        pieces = []
        position = 0
        while position < len(text):
            pieces.append(text[position : position + self._chunk_size])
            position += step
        return pieces
