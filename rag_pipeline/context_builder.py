from rag_hybrid_search.models import RetrievedChunk
from rag_pipeline.models import PromptContext

_CHARS_PER_TOKEN_ESTIMATE = 4


def build_context(
    chunks: list[RetrievedChunk], approx_token_budget: int = 2000
) -> PromptContext:
    """Builds a numbered prompt context from ranked, deduplicated chunks.

    approx_token_budget is estimated from character count
    (len(text) // CHARS_PER_TOKEN_ESTIMATE) -- an approximation, not an
    exact tokenizer count. If the budget would be exceeded, the
    lowest-ranked chunks are dropped whole (never truncated mid-text) so
    every included chunk stays intact and citable.
    """
    char_budget = approx_token_budget * _CHARS_PER_TOKEN_ESTIMATE

    seen_chunk_ids: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for retrieved in sorted(chunks, key=lambda r: r.final_rank):
        if retrieved.chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(retrieved.chunk.chunk_id)
        deduped.append(retrieved)

    included: list[RetrievedChunk] = []
    used_chars = 0
    for retrieved in deduped:
        chunk_chars = len(retrieved.chunk.text)
        if included and used_chars + chunk_chars > char_budget:
            break
        included.append(retrieved)
        used_chars += chunk_chars

    doc_id_map: dict[str, str] = {}
    lines: list[str] = []
    for i, retrieved in enumerate(included, start=1):
        doc_id = f"d{i}"
        doc_id_map[doc_id] = retrieved.chunk.chunk_id
        lines.append(f"[{doc_id}]\n{retrieved.chunk.text.strip()}")

    return PromptContext(text="\n\n".join(lines), doc_id_map=doc_id_map)
