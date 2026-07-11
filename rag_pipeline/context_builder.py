from enum import Enum

from rag_hybrid_search.models import ContextChunk
from rag_pipeline.models import PromptContext

_CHARS_PER_TOKEN_ESTIMATE = 4


class ContextLayout(str, Enum):
    """Current layouts:
      FLAT     -- numbered chunks, no grouping (today's behavior).
      GROUPED  -- sectioned by the subquery that retrieved each chunk.

    Expected to grow. Plausible future values: HIERARCHICAL (group by
    document, then chunk, within each subquery), DOCUMENT_FIRST (group by
    source document instead of subquery), COMPRESSED (merge adjacent
    chunks from the same section before rendering). Not implemented --
    documented so new layouts extend this enum rather than growing a
    parallel ad-hoc flag.
    """

    FLAT = "flat"
    GROUPED = "grouped"


def _dedup_and_budget(
    context_chunks: list[ContextChunk], approx_token_budget: int
) -> list[ContextChunk]:
    char_budget = approx_token_budget * _CHARS_PER_TOKEN_ESTIMATE

    seen_chunk_ids: set[str] = set()
    deduped: list[ContextChunk] = []
    for cc in sorted(context_chunks, key=lambda cc: cc.chunk.final_rank):
        if cc.chunk.chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cc.chunk.chunk.chunk_id)
        deduped.append(cc)

    included: list[ContextChunk] = []
    used_chars = 0
    for cc in deduped:
        chunk_chars = len(cc.chunk.chunk.text)
        if included and used_chars + chunk_chars > char_budget:
            break
        included.append(cc)
        used_chars += chunk_chars
    return included


def build_context(
    context_chunks: list[ContextChunk],
    subqueries: list[str],
    layout: ContextLayout = ContextLayout.FLAT,
    approx_token_budget: int = 2000,
) -> PromptContext:
    """Builds a numbered prompt context from ranked, deduplicated chunks.

    Presentation-only: never prunes, dedups across calls, reranks, or
    otherwise changes which chunks are included beyond the one dedup/budget
    pass below -- every chunk in context_chunks is assumed already final
    from retrieval and pruning. approx_token_budget is estimated from
    character count (len(text) // CHARS_PER_TOKEN_ESTIMATE) -- an
    approximation, not an exact tokenizer count. If the budget would be
    exceeded, the lowest-ranked chunks are dropped whole (never truncated
    mid-text) so every included chunk stays intact and citable.

    Citation ids ([d1], [d2], ...) are assigned once, globally, in
    final_rank order, before layout decides how to arrange them --
    GROUPED never restarts numbering per section.

    layout=FLAT renders a flat numbered list (byte-identical to the
    original single-layout build_context). layout=GROUPED sections chunks
    by provenance.primary_subquery, in subqueries' decomposition order;
    within each section, chunks keep final_rank order. Each chunk renders
    exactly once, under its primary_subquery, even if provenance.all_subqueries
    lists more than one match (duplicating evidence across sections is
    deferred -- see spec).
    """
    included = _dedup_and_budget(context_chunks, approx_token_budget)

    doc_id_map: dict[str, str] = {}
    doc_id_by_chunk_id: dict[str, str] = {}
    for i, cc in enumerate(included, start=1):
        doc_id = f"d{i}"
        doc_id_map[doc_id] = cc.chunk.chunk.chunk_id
        doc_id_by_chunk_id[cc.chunk.chunk.chunk_id] = doc_id

    if layout == ContextLayout.FLAT:
        lines = [
            f"[{doc_id_by_chunk_id[cc.chunk.chunk.chunk_id]}]\n{cc.chunk.chunk.text.strip()}"
            for cc in included
        ]
        return PromptContext(text="\n\n".join(lines), doc_id_map=doc_id_map)

    groups: dict[int, list[ContextChunk]] = {}
    for cc in included:
        groups.setdefault(cc.provenance.primary_subquery, []).append(cc)

    sections: list[str] = []
    for idx, subquery_text in enumerate(subqueries):
        group = groups.get(idx)
        if not group:
            continue
        chunk_lines = "\n\n".join(
            f"[{doc_id_by_chunk_id[cc.chunk.chunk.chunk_id]}]\n{cc.chunk.chunk.text.strip()}"
            for cc in group
        )
        sections.append(
            f"Evidence for subquery {idx + 1}\n\n"
            f"This evidence was retrieved to answer:\n\n"
            f'"{subquery_text}"\n\n'
            f"{chunk_lines}"
        )
    return PromptContext(text="\n\n".join(sections), doc_id_map=doc_id_map)
