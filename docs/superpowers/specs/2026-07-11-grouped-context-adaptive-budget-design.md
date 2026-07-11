# Grouped-by-Subquery Context Assembly + Adaptive Evidence Budget

**Date:** 2026-07-11
**Status:** Frozen — approved design
**Depends on:** comparative query retrieval (`is_comparative_query`, `decompose_query`),
dynamic pruning (`prune_by_score_margin`, `min_keep=3` fix), evaluation Phase 2
(regression comparison, used as the eval gate below).

## Problem

Traced example: "How do the function-level and class-level detection patterns differ
across the three research questions?" — a comparative question. Retrieval, fusion,
reranking, and pruning all worked; `min_keep=3` (already shipped) guarantees at least
3 chunks survive pruning for comparative questions. But `build_context` still flattens
everything into `[d1]...[d2]...[d3]` with no indication of which chunk answers which
part of the question. The retriever *knows* the question decomposes into subqueries;
`build_context` throws that structure away. The LLM has to re-discover the mapping
from evidence to sub-question with no help from the prompt.

This spec fixes two things: (1) render context grouped by the subquery that surfaced
each chunk, and (2) make the evidence-count floor account for how many subqueries were
asked, not just a fixed 3.

## Data flow

```
Query
  │
  ▼
Subqueries (decompose_query, ordered by importance)
  │
  ▼
Per-subquery retrieval (existing, unchanged)
  │
  ▼
_merge_multi_query_results → list[RetrievedChunk] (unchanged ranking/dedup)
                            + provenance side-map {chunk_id: ChunkProvenance}
  │
  ▼
prune_by_score_margin(list[RetrievedChunk], min_keep=required_chunks)  ← UNCHANGED function
  │
  ▼
wrap pruned chunks + provenance map → list[ContextChunk]
  │
  ▼
build_context(context_chunks, subqueries, layout) → PromptContext   ← presentation-only
  │
  ▼
Prompt (template unchanged — question/context interpolation already separate)
```

`prune_by_score_margin` is never touched — it keeps operating on plain
`list[RetrievedChunk]` exactly as today, at zero regression risk. Provenance rides
alongside as a `{chunk_id: ChunkProvenance}` dict built during merge, and gets attached
to the surviving chunks only after pruning decides which chunks exist. `build_context`
receives fully-decided `ContextChunk`s and only decides ordering/formatting/citation
numbering — never which chunks exist. This is an explicit architectural rule for this
feature: **`build_context` never changes retrieval decisions** (no regrouping,
rescoring, pruning, or dedup inside it).

## Data model

```python
# rag_hybrid_search/models.py

class ChunkProvenance(BaseModel):
    primary_subquery: int       # index into decomposition order (first match, merge-loop order)
    all_subqueries: list[int]   # every subquery index that surfaced this chunk (not yet
                                 # used for rendering — see Deferred)

class ContextChunk(BaseModel):
    chunk: RetrievedChunk
    provenance: ChunkProvenance
```

`RetrievedChunk` itself is unchanged — no provenance field added to it, keeping it a
pure retrieval-result object. Single-query (non-comparative) path: every chunk gets
`ChunkProvenance(primary_subquery=0, all_subqueries=[0])` — trivial, not `None`,
so `build_context` never branches on presence/absence of provenance.

## `build_context`

One function, one signature, replacing the current `build_context`:

```python
class ContextLayout(str, Enum):
    FLAT = "flat"
    GROUPED = "grouped"

def build_context(
    context_chunks: list[ContextChunk],
    subqueries: list[str],
    layout: ContextLayout = ContextLayout.FLAT,
    approx_token_budget: int = 2000,
) -> PromptContext:
```

Dedup-by-chunk-id, char-budget truncation (drop whole chunks, never mid-text), and
`[dN]` citation-id assignment are computed once, identically, regardless of layout —
citation numbers are assigned globally across the whole context before layout decides
how to arrange them into groups. **Citation numbering is stable and never restarts per
group**: `[d1]`, `[d2]`, `[d3]`... increment monotonically across the entire prompt,
grouped layout only changes where line breaks and headers go, never renumbers.

`layout=FLAT`: renders exactly what today's `build_context` produces (byte-identical),
ignoring `subqueries`/`provenance` entirely.

`layout=GROUPED`: buckets `context_chunks` by `provenance.primary_subquery`, in
subquery-decomposition order; within each group, chunks stay in their existing
`final_rank` order. Renders:

```
Evidence relevant to subquery 1

Question:
"What does RQ1 conclude?"

Relevant excerpts:

[d1]
...

[d2]
...

Evidence relevant to subquery 2

Question:
"What does RQ3 conclude?"

Relevant excerpts:

[d3]
...
```

Each chunk renders exactly once, under its `primary_subquery` group, even if
`all_subqueries` lists more than one match — avoids duplicating chunk text and
inflating prompt tokens beyond the eval gate's cap. (See Deferred.)

## Prompt template

No new template. `PromptTemplate` stays `V1`/`V2` — the question/context interpolation
in `build_prompt` is already separate from context rendering, so grouped layout is a
pure `context.text` change, not a template change. No `prompt_version="v3"`.

## Pipeline wiring (`rag_pipeline.py`, `_prepare_context`)

```python
required_chunks = max(3 if comparative else 1, len(subqueries))
pruned_chunks = prune_by_score_margin(retrieved_chunks, margin, min_keep=required_chunks)
dev_trace.log_pruning(retrieved_chunks, pruned_chunks)

context_chunks = _attach_provenance(pruned_chunks, provenance_map)  # new small helper
dev_trace.log_provenance(context_chunks)

layout = (
    ContextLayout.GROUPED
    if comparative and self._context_layout == ContextLayout.GROUPED
    else ContextLayout.FLAT
)
context = build_context(context_chunks, subqueries, layout=layout)
```

`RagPipeline` gains constructor param `context_layout: ContextLayout = ContextLayout.FLAT`
— independent of `prompt_version`. Default is `FLAT` (today's behavior, unchanged for
all existing callers/tests). Single-query path always resolves to `FLAT` regardless of
`context_layout`, since `comparative` is `False` — guarantees byte-identical output for
factual questions with zero config changes required anywhere.

## Trace

`RequestTrace` gains `log_provenance(context_chunks: list[ContextChunk]) -> None`,
following the existing `log_*` convention (see `log_pruning`, `log_query_decomposition`).
Records, per chunk id: primary subquery index/text, and all matched subquery indices —
printed under a new trace section when `self.enabled` and matches occur across more
than one subquery (mirrors `log_pruning`'s "only print when there's something to show"
pattern).

## Testing

Unit:
- provenance side-map construction in merge: single-subquery path trivial provenance,
  multi-subquery path records first-match `primary_subquery` and full `all_subqueries`
- `build_context` FLAT layout byte-identical to pre-change output (regression guard)
- `build_context` GROUPED layout: correct bucketing, decomposition-order groups,
  `final_rank` order within groups, monotonic global citation numbering across groups,
  each chunk rendered exactly once even when `all_subqueries` has multiple entries
- adaptive `min_keep` floor: `max(3, len(subqueries))` for comparative,
  `max(1, len(subqueries))` for non-comparative (always 1, since non-comparative is
  always a single subquery)
- `log_provenance` trace output shape

Integration:
- comparative question, `context_layout=FLAT` vs `GROUPED` — same chunk set, different
  rendering
- factual (non-comparative) question — `context_layout=GROUPED` pipeline produces
  identical prompt to `context_layout=FLAT` pipeline (proves the "always FLAT for
  single-query" guarantee)

## Evaluation gate

Before defaulting any pipeline construction to `context_layout=GROUPED`, run
`scripts/run_eval.py --compare-baseline` comparing a `FLAT`-configured baseline against
a `GROUPED` run, with per-category breakdown. Required to pass, all measured via the
Phase 2 regression infrastructure already built (no new evaluation mechanism):

- Comparative category accuracy ≥ baseline
- Hallucination rate not increased
- Verification pass rate not decreased
- Factual category unchanged (±1%)
- Prompt token count increase ≤20% (comparative category average)

This is a manual gate run once before flipping any default — not wired into CI, since
it requires a real corpus and generation provider (same reasoning that keeps CI
objective-only per the Phase 2 design).

## Deferred (explicitly out of scope for this spec)

- Rendering a chunk under every subquery in `all_subqueries` (duplicating evidence
  across groups) instead of only its `primary_subquery`. `all_subqueries` is preserved
  now specifically so this is possible later without a data-model change — deferred to
  avoid prompt bloat until there's evidence duplication actually improves answers.
- Hierarchical document/chunk grouping within a subquery section (Tier 2 in the
  earlier prioritization).
- Diversity-aware pruning, adjacent-chunk compression (Tier 2/3).
- New multi-hop/definition/summarization budget tiers — only the existing
  comparative/non-comparative split is used, floored by `len(subqueries)`.
