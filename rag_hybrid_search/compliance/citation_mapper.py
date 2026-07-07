from rag_hybrid_search.compliance.regulation_models import Citation
from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.uuid7 import uuid7


def _render_display(retrieved: RetrievedChunk, title: str) -> str:
    lm = retrieved.chunk.legal_metadata
    if lm is None or (lm.regulation is None and lm.article is None):
        page = retrieved.chunk.page
        page_part = f", p.{page}" if page else ""
        return f"{title}{page_part}, chunk {retrieved.chunk.chunk_index}"

    parts = []
    if lm.regulation:
        parts.append(lm.regulation)
    if lm.article:
        clause_suffix = ""
        if lm.clause and "." in lm.clause:
            remainder = lm.clause.split(".", 1)[-1]
            paren_idx = remainder.find("(")
            if paren_idx != -1:
                num_part, paren_part = remainder[:paren_idx], remainder[paren_idx:]
                clause_suffix = f"({num_part}){paren_part}"
            else:
                clause_suffix = f"({remainder})"
        parts.append(f"Art. {lm.article}{clause_suffix}")
    elif lm.clause:
        parts.append(f"Clause {lm.clause}")
    display = " ".join(parts)
    page = lm.page or retrieved.chunk.page
    if page:
        display += f", p.{page}"
    return display


def build_citations(
    retrieved_chunks: list[RetrievedChunk],
    filename_by_doc_id: dict[str, str] | None = None,
) -> list[Citation]:
    """Build structured Citation objects from post-rerank retrieved chunks.

    Non-legal chunks (legal_metadata is None) degrade gracefully to a
    filename/chunk_index based display string (falling back to document_id
    if the filename isn't known) instead of failing.
    """
    filename_by_doc_id = filename_by_doc_id or {}
    citations = []
    for retrieved in retrieved_chunks:
        lm = retrieved.chunk.legal_metadata
        title = filename_by_doc_id.get(retrieved.chunk.document_id, retrieved.chunk.document_id)
        citations.append(
            Citation(
                citation_id=uuid7(),
                document_id=retrieved.chunk.document_id,
                document_title=lm.document_title if lm else title,
                chunk_id=retrieved.chunk.chunk_id,
                confidence=retrieved.rerank_score or 0.0,
                display=_render_display(retrieved, title),
                regulation=lm.regulation if lm else None,
                version=lm.version if lm else None,
                jurisdiction=lm.jurisdiction if lm else None,
                article=lm.article if lm else None,
                section=lm.section if lm else None,
                clause=lm.clause if lm else None,
                effective_date=lm.effective_date if lm else None,
                document_type=lm.document_type if lm else None,
                page=(lm.page if lm else None) or retrieved.chunk.page,
            )
        )
    return citations
