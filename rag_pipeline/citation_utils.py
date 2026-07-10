"""Shared chunk-text lookup used by both quote extraction and citation
verification, so both always agree on exactly where one chunk's text ends
and the next begins -- the single source of truth that makes cross-chunk
supporting_quote structurally impossible to construct.
"""
from rag_pipeline.models import PromptContext


def chunk_text_for_doc_id(context: PromptContext, doc_id: str) -> str:
    """Extract this doc's chunk text from the prompt context.

    context_builder joins chunks with '\\n\\n', so naively splitting on the
    first '\\n\\n' after the marker assumes that's always the chunk
    boundary. But a chunk's own text can legitimately contain an internal
    blank line (e.g. 'prose\\n\\ntable rows' from the PDF table renderer),
    which would truncate it early. Instead, find the true boundary: the
    start of whichever other doc's marker comes next in the text.
    """
    marker = f"[{doc_id}]"
    start_idx = context.text.find(marker)
    if start_idx == -1:
        return ""
    start = start_idx + len(marker)

    end = len(context.text)
    for other_id in context.doc_id_map:
        if other_id == doc_id:
            continue
        pos = context.text.find(f"\n\n[{other_id}]", start)
        if pos != -1 and pos < end:
            end = pos

    return context.text[start:end].strip()
