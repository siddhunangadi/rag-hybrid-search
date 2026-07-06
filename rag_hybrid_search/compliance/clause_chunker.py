from rag_hybrid_search.compliance.clause_parser import parse_clauses
from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.models import Chunk, Document
from rag_hybrid_search.uuid7 import uuid7


class ClauseChunker(Chunker):
    """Chunks a document along Article/Section/Clause boundaries instead
    of fixed-size windows, attaching LegalMetadata to each resulting Chunk.

    Falls back to a single whole-document chunk (still tagged with
    LegalMetadata, all fields null except document_id/document_title)
    when clause_parser finds no recognizable structure — the document is
    never dropped or left unchunked.
    """

    version = "clause-v1"

    def __init__(self, document_title: str):
        self._document_title = document_title

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.content.strip():
            return []

        parse_result = parse_clauses(
            document.content,
            document_id=document.document_id,
            document_title=self._document_title,
        )

        chunks: list[Chunk] = []
        for index, clause_span in enumerate(parse_result.clauses):
            chunks.append(
                Chunk(
                    chunk_id=uuid7(),
                    document_id=document.document_id,
                    chunk_index=index,
                    text=clause_span.text,
                    strategy_version=self.version,
                    heading=clause_span.metadata.article or clause_span.metadata.section,
                    page=clause_span.metadata.page,
                    char_count=len(clause_span.text),
                    legal_metadata=clause_span.metadata,
                )
            )
        return chunks
