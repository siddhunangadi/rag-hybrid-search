# Grouped-by-Subquery Context Assembly + Adaptive Evidence Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render RAG prompt context grouped by the subquery that retrieved each chunk (instead of a flat list), and make the pruning evidence floor scale with the number of subqueries, so comparative/multi-subquery questions never lose their sub-question structure between retrieval and the LLM.

**Architecture:** `_merge_multi_query_results` builds a provenance side-map `{chunk_id: ChunkProvenance}` alongside its existing ranked list (unchanged). `prune_by_score_margin` is untouched, still operates on plain `RetrievedChunk`. After pruning, chunks are wrapped into `ContextChunk` (chunk + provenance) and handed to a rewritten `build_context`, which gains a `layout: ContextLayout` parameter (`FLAT` — byte-identical to today; `GROUPED` — sectioned by subquery) but never changes which chunks exist, only how they're rendered.

**Tech Stack:** Python 3.12, Pydantic v2, pytest (existing repo conventions — no new dependencies).

**Spec:** `docs/superpowers/specs/2026-07-11-grouped-context-adaptive-budget-design.md`

## Global Constraints

- `prune_by_score_margin` (`rag_pipeline/context_pruning.py`) is never modified — it keeps its exact current signature and behavior, operating on `list[RetrievedChunk]`.
- **Design Invariant:** `build_context` is presentation-only. It may order, format, and assign citation ids. It must never prune, deduplicate across calls, rerank, or regroup in a way that changes which chunks are included.
- Citation numbering (`[d1]`, `[d2]`, ...) is assigned once, globally, monotonically increasing across the whole context — never restarts per group, regardless of layout.
- `layout=ContextLayout.FLAT` must produce output byte-identical to the current (pre-change) `build_context` for the same chunk set.
- Non-comparative (single-subquery) questions always resolve to `ContextLayout.FLAT`, regardless of a pipeline's configured `context_layout` — `comparative` is `False` on that path, so grouping never applies.
- `RetrievedChunk` (`rag_hybrid_search/models.py`) gets no new field — provenance lives only on the new `ContextChunk` wrapper.
- No new prompt template version. `PromptTemplate` stays `v1`/`v2`; `build_prompt`'s signature and template strings are untouched.
- Each chunk renders exactly once in `GROUPED` layout, under its `primary_subquery` group, even if it matched multiple subqueries (`all_subqueries` is stored but not yet used for duplication — deferred).
- `required_chunks = max(3 if comparative else 1, len(subqueries))` (min_keep floor).

---

### Task 1: `ChunkProvenance` and `ContextChunk` models

**Files:**
- Modify: `rag_hybrid_search/models.py` (add after `RetrievedChunk`, currently ending at line 54)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `ChunkProvenance(BaseModel)` with fields `primary_subquery: int`, `all_subqueries: list[int]`; `ContextChunk(BaseModel)` with fields `chunk: RetrievedChunk`, `provenance: ChunkProvenance`. Both importable from `rag_hybrid_search.models`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models.py` (check the file's existing imports first; it already imports from `rag_hybrid_search.models` — add `ChunkProvenance, ContextChunk` to that import line and append):

```python
def test_chunk_provenance_and_context_chunk():
    chunk = Chunk(
        chunk_id="c1", document_id="d1", chunk_index=0, text="hello",
        strategy_version="fixed-v1", heading=None, page=None, char_count=5,
    )
    retrieved = RetrievedChunk(
        chunk=chunk, dense_score=0.9, bm25_score=0.9, rrf_score=0.5,
        rerank_score=0.8, final_rank=1,
    )
    provenance = ChunkProvenance(primary_subquery=0, all_subqueries=[0, 2])
    context_chunk = ContextChunk(chunk=retrieved, provenance=provenance)

    assert context_chunk.chunk is retrieved
    assert context_chunk.provenance.primary_subquery == 0
    assert context_chunk.provenance.all_subqueries == [0, 2]
```

(If `tests/test_models.py` doesn't already import `Chunk`/`RetrievedChunk`, add them to the import line too.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_models.py::test_chunk_provenance_and_context_chunk -v`
Expected: FAIL with `ImportError: cannot import name 'ChunkProvenance'`

- [ ] **Step 3: Add the models**

In `rag_hybrid_search/models.py`, immediately after the `RetrievedChunk` class (after its `final_rank: int` line):

```python
class ChunkProvenance(BaseModel):
    primary_subquery: int
    all_subqueries: list[int]


class ContextChunk(BaseModel):
    chunk: RetrievedChunk
    provenance: ChunkProvenance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_models.py::test_chunk_provenance_and_context_chunk -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rag_hybrid_search/models.py tests/test_models.py
git commit -m "feat: add ChunkProvenance and ContextChunk models"
```

---

### Task 2: Provenance side-map in `_merge_multi_query_results`

**Files:**
- Modify: `rag_pipeline/rag_pipeline.py:131-177` (the `_merge_multi_query_results` function)
- Test: `tests/rag_pipeline/test_rag_pipeline.py` (or a new `tests/rag_pipeline/test_merge_provenance.py` if the existing file doesn't already import this helper — check first)

**Interfaces:**
- Consumes: `ChunkProvenance` from Task 1 (`rag_hybrid_search.models`).
- Produces: `_merge_multi_query_results(results_per_query: list[list]) -> tuple[list, dict[str, ChunkProvenance]]` — same ranked-list behavior as today (unchanged), plus a second return value: `{chunk_id: ChunkProvenance}` built during the same merge loop. `primary_subquery` = index of the first `results_per_query[i]` list the chunk appeared in (loop order, i.e. decomposition-priority order). `all_subqueries` = sorted list of every index `i` where the chunk appeared.

- [ ] **Step 1: Write the failing test**

Check whether `tests/rag_pipeline/test_rag_pipeline.py` already imports `_merge_multi_query_results` (it's a private helper — grep first: `grep -n "_merge_multi_query_results" tests/rag_pipeline/test_rag_pipeline.py`). Add this test to that file (or create `tests/rag_pipeline/test_merge_provenance.py` importing `from rag_pipeline.rag_pipeline import _merge_multi_query_results` and the chunk-building helper already used in that test file — reuse the existing `make_retrieved_chunk`/equivalent fixture builder in the target file rather than redefining it):

```python
def test_merge_returns_provenance_side_map():
    chunk_a = make_retrieved_chunk("a", "text a", final_rank=1)
    chunk_b = make_retrieved_chunk("b", "text b", final_rank=1)
    chunk_c = make_retrieved_chunk("c", "text c", final_rank=1)

    # subquery 0 retrieves a, b; subquery 1 retrieves b, c
    results_per_query = [[chunk_a, chunk_b], [chunk_b, chunk_c]]

    merged, provenance = _merge_multi_query_results(results_per_query)

    assert provenance["a"].primary_subquery == 0
    assert provenance["a"].all_subqueries == [0]
    assert provenance["b"].primary_subquery == 0  # first seen under subquery 0
    assert provenance["b"].all_subqueries == [0, 1]
    assert provenance["c"].primary_subquery == 1
    assert provenance["c"].all_subqueries == [1]
    assert {r.chunk.chunk_id for r in merged} == {"a", "b", "c"}


def test_merge_provenance_single_subquery():
    chunk_a = make_retrieved_chunk("a", "text a", final_rank=1)
    merged, provenance = _merge_multi_query_results([[chunk_a]])
    assert provenance["a"].primary_subquery == 0
    assert provenance["a"].all_subqueries == [0]
```

(`make_retrieved_chunk` is whatever helper the target test file already uses to build a `RetrievedChunk` fixture — if none exists, add one following the pattern already shown in Task 1's test: build a `Chunk` then wrap in `RetrievedChunk`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/rag_pipeline/test_rag_pipeline.py -k provenance -v`
Expected: FAIL — either `ImportError` (new test file) or a tuple-unpacking error (`_merge_multi_query_results` currently returns a plain list, not a tuple)

- [ ] **Step 3: Modify `_merge_multi_query_results`**

Replace the function body in `rag_pipeline/rag_pipeline.py` (currently lines 131-177) — keep the docstring's existing content, append a note about the new return value, and change the loop and return statement:

```python
def _merge_multi_query_results(results_per_query: list[list]) -> tuple[list, dict]:
    """Merge reranked result lists from multiple sub-query retrievals into
    one ranked list, deduping by chunk_id.

    Each sub-query already went through the full retrieve() pipeline
    (dense+sparse+fuse+rerank) independently. For ranking the merged list,
    a chunk that surfaced in N independent sub-query retrievals is a
    stronger relevance signal than its single best rerank_score alone
    would suggest -- appearing under multiple distinct queries means
    multiple lines of evidence point to it, not just one lucky embedding
    match. combined_score = best_rerank_score + 0.15 * log2(appearances),
    so a single appearance is unaffected (log2(1) == 0, matches
    pre-frequency-weighting behavior) and each additional appearance adds
    a *diminishing* bonus rather than a linear one -- a chunk with a
    clearly weaker score shouldn't out-rank a much stronger one just by
    showing up many times (log2(8) is only 2x log2(4), not 8x). The chunk
    object itself keeps its original (best) rerank_score unchanged --
    combined_score is sort-only, not stored on the model. Chunks with
    rerank_score=None (e.g. PassthroughReranker) sort last but are never
    dropped.

    Also returns a provenance side-map ({chunk_id: ChunkProvenance}) built
    from the same loop: primary_subquery is the index of the first
    results_per_query[i] list the chunk appeared in (decomposition-priority
    order), all_subqueries lists every index it appeared under. Kept as a
    side-map rather than attached to RetrievedChunk directly so
    prune_by_score_margin's existing behavior and tests are untouched --
    provenance is only attached (see ContextChunk) after pruning decides
    which chunks survive.
    """
    from rag_hybrid_search.models import ChunkProvenance

    best_by_id: dict[str, object] = {}
    appearances: dict[str, int] = {}
    subqueries_by_id: dict[str, list[int]] = {}
    for i, results in enumerate(results_per_query):
        for r in results:
            chunk_id = r.chunk.chunk_id
            appearances[chunk_id] = appearances.get(chunk_id, 0) + 1
            subqueries_by_id.setdefault(chunk_id, [])
            if i not in subqueries_by_id[chunk_id]:
                subqueries_by_id[chunk_id].append(i)
            existing = best_by_id.get(chunk_id)
            if existing is None:
                best_by_id[chunk_id] = r
                continue
            existing_score = existing.rerank_score
            new_score = r.rerank_score
            if new_score is not None and (existing_score is None or new_score > existing_score):
                best_by_id[chunk_id] = r

    def combined_score(r) -> float:
        base = r.rerank_score if r.rerank_score is not None else 0.0
        bonus = _FREQUENCY_BONUS_SCALE * math.log2(appearances[r.chunk.chunk_id])
        return base + bonus

    merged = sorted(
        best_by_id.values(),
        key=lambda r: (r.rerank_score is None, -combined_score(r)),
    )
    provenance = {
        chunk_id: ChunkProvenance(primary_subquery=indices[0], all_subqueries=sorted(indices))
        for chunk_id, indices in subqueries_by_id.items()
    }
    return [r.model_copy(update={"final_rank": i}) for i, r in enumerate(merged, start=1)], provenance
```

- [ ] **Step 4: Update the only call site**

`_merge_multi_query_results` is called at `rag_pipeline.py:367` inside `_prepare_context` (`retrieved_chunks = _merge_multi_query_results(results_per_query)`). This will be updated in Task 4 alongside the rest of `_prepare_context` — for this task, run only the merge-focused tests, not the full pipeline suite (which will fail until Task 4 lands).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/rag_pipeline/test_rag_pipeline.py -k provenance -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add rag_pipeline/rag_pipeline.py tests/rag_pipeline/test_rag_pipeline.py
git commit -m "feat: build provenance side-map in _merge_multi_query_results"
```

Note: this commit temporarily breaks `_prepare_context` (still calls the old single-return-value form) — Task 4 fixes the call site. If your CI runs the full suite between tasks, expect `_prepare_context`-dependent tests to fail until Task 4 completes; this is expected and documented here so it isn't mistaken for a Task 2 regression.

---

### Task 3: `ContextLayout` enum + rewritten `build_context`

**Files:**
- Modify: `rag_pipeline/context_builder.py` (full rewrite of the module)
- Modify: `tests/rag_pipeline/test_context_builder.py` (full rewrite — current tests call `build_context(chunks)` with plain `RetrievedChunk`; new signature takes `ContextChunk` + `subqueries` + `layout`)

**Interfaces:**
- Consumes: `ContextChunk`, `ChunkProvenance` from Task 1.
- Produces: `ContextLayout(str, Enum)` with values `FLAT = "flat"`, `GROUPED = "grouped"`; `build_context(context_chunks: list[ContextChunk], subqueries: list[str], layout: ContextLayout = ContextLayout.FLAT, approx_token_budget: int = 2000) -> PromptContext`.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/rag_pipeline/test_context_builder.py`:

```python
from rag_hybrid_search.models import Chunk, ChunkProvenance, ContextChunk, RetrievedChunk
from rag_pipeline.context_builder import ContextLayout, build_context


def make_context_chunk(chunk_id, text, final_rank, primary_subquery=0, all_subqueries=None):
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )
    retrieved = RetrievedChunk(
        chunk=chunk,
        dense_score=0.9,
        bm25_score=0.9,
        rrf_score=0.5,
        rerank_score=0.8,
        final_rank=final_rank,
    )
    provenance = ChunkProvenance(
        primary_subquery=primary_subquery,
        all_subqueries=all_subqueries or [primary_subquery],
    )
    return ContextChunk(chunk=retrieved, provenance=provenance)


def test_empty_context():
    context = build_context([], subqueries=[])
    assert context.text == ""
    assert context.doc_id_map == {}


def test_flat_numbers_chunks_in_rank_order():
    chunks = [
        make_context_chunk("c1", "first chunk text", final_rank=1),
        make_context_chunk("c2", "second chunk text", final_rank=2),
    ]
    context = build_context(chunks, subqueries=["q"])
    assert "[d1]" in context.text
    assert "[d2]" in context.text
    assert context.text.index("[d1]") < context.text.index("[d2]")
    assert context.doc_id_map == {"d1": "c1", "d2": "c2"}
    assert "Evidence for subquery" not in context.text  # flat has no group headers


def test_deduplicates_by_chunk_id():
    chunk = make_context_chunk("c1", "same chunk", final_rank=1)
    context = build_context([chunk, chunk], subqueries=["q"])
    assert len(context.doc_id_map) == 1


def test_truncates_lowest_ranked_chunks_first_without_splitting_text():
    big_text = "word " * 400  # ~2000 chars, ~500 approx tokens
    chunks = [
        make_context_chunk("c1", big_text, final_rank=1),
        make_context_chunk("c2", big_text, final_rank=2),
        make_context_chunk("c3", big_text, final_rank=3),
    ]
    context = build_context(chunks, subqueries=["q"], approx_token_budget=500)
    assert "[d1]" in context.text
    assert "[d3]" not in context.text


def test_grouped_sections_by_primary_subquery_in_decomposition_order():
    chunks = [
        make_context_chunk("c1", "about rq1", final_rank=1, primary_subquery=0),
        make_context_chunk("c2", "also about rq1", final_rank=2, primary_subquery=0),
        make_context_chunk("c3", "about rq3", final_rank=3, primary_subquery=1),
    ]
    subqueries = ["What does RQ1 conclude?", "What does RQ3 conclude?"]
    context = build_context(chunks, subqueries, layout=ContextLayout.GROUPED)

    assert "Evidence for subquery 1" in context.text
    assert "Evidence for subquery 2" in context.text
    assert '"What does RQ1 conclude?"' in context.text
    assert '"What does RQ3 conclude?"' in context.text
    # subquery 1's section (index 0) must appear before subquery 2's (index 1)
    assert context.text.index("Evidence for subquery 1") < context.text.index("Evidence for subquery 2")
    # chunks within a group keep final_rank order
    assert context.text.index("[d1]") < context.text.index("[d2]")


def test_grouped_citation_numbering_is_global_and_monotonic():
    chunks = [
        make_context_chunk("c1", "text 1", final_rank=1, primary_subquery=0),
        make_context_chunk("c2", "text 2", final_rank=2, primary_subquery=1),
    ]
    subqueries = ["sub a", "sub b"]
    context = build_context(chunks, subqueries, layout=ContextLayout.GROUPED)
    assert context.doc_id_map == {"d1": "c1", "d2": "c2"}
    assert context.text.index("[d1]") < context.text.index("[d2]")


def test_grouped_renders_multi_subquery_chunk_exactly_once():
    chunk = make_context_chunk("c1", "shared text", final_rank=1, primary_subquery=0, all_subqueries=[0, 1])
    subqueries = ["sub a", "sub b"]
    context = build_context([chunk], subqueries, layout=ContextLayout.GROUPED)
    assert context.text.count("[d1]") == 1
    assert "Evidence for subquery 2" not in context.text  # empty group not rendered


def test_flat_layout_byte_identical_to_grouped_chunk_set_but_no_headers():
    chunks = [
        make_context_chunk("c1", "first", final_rank=1, primary_subquery=0),
        make_context_chunk("c2", "second", final_rank=2, primary_subquery=1),
    ]
    subqueries = ["sub a", "sub b"]
    flat = build_context(chunks, subqueries, layout=ContextLayout.FLAT)
    assert flat.text == "[d1]\nfirst\n\n[d2]\nsecond"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/rag_pipeline/test_context_builder.py -v`
Expected: FAIL — `ImportError: cannot import name 'ContextLayout'`

- [ ] **Step 3: Rewrite `context_builder.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rag_pipeline/test_context_builder.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline/context_builder.py tests/rag_pipeline/test_context_builder.py
git commit -m "feat: add ContextLayout and grouped-by-subquery build_context"
```

---

### Task 4: Wire provenance + adaptive min_keep + layout into `RagPipeline`

**Files:**
- Modify: `rag_pipeline/rag_pipeline.py` — constructor (lines 251-259), `_prepare_context` (lines ~339-378)
- Test: `tests/rag_pipeline/test_rag_pipeline.py`

**Interfaces:**
- Consumes: `ContextLayout`, `build_context` (Task 3); `ContextChunk`, `ChunkProvenance` (Task 1); `_merge_multi_query_results` returning `(merged, provenance)` tuple (Task 2).
- Produces: `RagPipeline.__init__` gains `context_layout: ContextLayout = ContextLayout.FLAT` parameter; `RagPipeline.context_layout` property; `_prepare_context` now computes `required_chunks = max(3 if comparative else 1, len(subqueries))`, wraps pruned chunks into `ContextChunk`, and resolves layout per the rule: `ContextLayout.GROUPED if (comparative and self._context_layout == ContextLayout.GROUPED) else ContextLayout.FLAT`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/rag_pipeline/test_rag_pipeline.py`, reusing the file's existing
`make_retrieved_chunk`, `MultiQueryFakeRetriever`, and `MockProvider`
decompose-then-answer canned-response pattern (see
`test_comparative_question_keeps_multiple_chunks_after_pruning`, already in this
file, for the exact shape being followed here):

```python
from rag_pipeline.context_builder import ContextLayout


def test_context_layout_defaults_to_flat_and_is_exposed():
    chunks = [make_retrieved_chunk("c1", "Some evidence text.")]
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider())
    assert pipeline.context_layout == ContextLayout.FLAT


def test_factual_question_stays_flat_even_with_grouped_configured():
    from rag_hybrid_search.trace import RequestTrace

    chunk = make_retrieved_chunk("c1", "Employees get 20 days of paid leave.")
    retriever = MultiQueryFakeRetriever({"How many days of paid leave?": [chunk]})
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{"text": "Employees get 20 days of paid leave.", "citation_ids": ["d1"]}],
    })
    pipeline = RagPipeline(
        retriever, MockProvider(canned_json=canned), context_layout=ContextLayout.GROUPED,
    )

    trace = RequestTrace("How many days of paid leave?", {"Generation": "MockProvider"})
    result = pipeline.answer("How many days of paid leave?", dev_trace=trace)

    assert result.error is None
    assert "Evidence for subquery" not in trace.data["prompt"]["text"]


def test_comparative_question_groups_when_configured_grouped():
    from rag_hybrid_search.trace import RequestTrace

    chunks_by_query = {
        "RQ1 findings": [make_retrieved_chunk("rq1", "RQ1: granularity matters.", rerank_score=5.0)],
        "RQ3 findings": [make_retrieved_chunk("rq3", "RQ3: features overlap.", rerank_score=0.1, final_rank=2)],
    }
    retriever = MultiQueryFakeRetriever(chunks_by_query)
    decompose_canned = json.dumps(["RQ1 findings", "RQ3 findings"])
    answer_canned = json.dumps({
        "answer": "RQ1 shows granularity matters [d1] and RQ3 shows features overlap [d2].",
        "claims": [
            {"text": "Granularity matters.", "citation_ids": ["d1"]},
            {"text": "Features overlap.", "citation_ids": ["d2"]},
        ],
    })
    provider = MockProvider(canned_json=answer_canned)
    calls = {"n": 0}
    original_generate = provider.generate

    def generate(prompt, **kwargs):
        calls["n"] += 1
        return decompose_canned if calls["n"] == 1 else original_generate(prompt, **kwargs)

    provider.generate = generate

    pipeline = RagPipeline(retriever, provider, context_layout=ContextLayout.GROUPED)
    trace = RequestTrace("How do RQ1 and RQ3 findings differ?", {"Generation": "MockProvider"})
    result = pipeline.answer("How do RQ1 and RQ3 findings differ?", dev_trace=trace)

    assert result.error is None
    assert "Evidence for subquery 1" in trace.data["prompt"]["text"]
    assert "Evidence for subquery 2" in trace.data["prompt"]["text"]


def test_comparative_question_min_keep_floors_at_subquery_count():
    """4 subqueries, one dominant score -- min_keep must be 4, not the
    fixed 3 the pre-existing comparative min_keep used."""
    chunks_by_query = {
        "RQ1 findings": [make_retrieved_chunk("rq1", "RQ1 finding.", rerank_score=5.0)],
        "RQ2 findings": [make_retrieved_chunk("rq2", "RQ2 finding.", rerank_score=0.2, final_rank=2)],
        "RQ3 findings": [make_retrieved_chunk("rq3", "RQ3 finding.", rerank_score=0.1, final_rank=3)],
        "RQ4 findings": [make_retrieved_chunk("rq4", "RQ4 finding.", rerank_score=0.05, final_rank=4)],
    }
    retriever = MultiQueryFakeRetriever(chunks_by_query)
    decompose_canned = json.dumps(["RQ1 findings", "RQ2 findings", "RQ3 findings", "RQ4 findings"])
    answer_canned = json.dumps({
        "answer": "RQ1 [d1], RQ2 [d2], RQ3 [d3], RQ4 [d4].",
        "claims": [
            {"text": "RQ1 finding.", "citation_ids": ["d1"]},
            {"text": "RQ2 finding.", "citation_ids": ["d2"]},
            {"text": "RQ3 finding.", "citation_ids": ["d3"]},
            {"text": "RQ4 finding.", "citation_ids": ["d4"]},
        ],
    })
    provider = MockProvider(canned_json=answer_canned)
    calls = {"n": 0}
    original_generate = provider.generate

    def generate(prompt, **kwargs):
        calls["n"] += 1
        return decompose_canned if calls["n"] == 1 else original_generate(prompt, **kwargs)

    provider.generate = generate

    pipeline = RagPipeline(retriever, provider)
    result = pipeline.answer("How do RQ1, RQ2, RQ3, and RQ4 findings differ?")

    assert result.error is None
    assert {c for c in result.citations} == {"d1", "d2", "d3", "d4"}
```

These four tests use `RequestTrace.data["prompt"]["text"]` to inspect the built
prompt — verified against `RequestTrace.log_prompt` (`rag_hybrid_search/trace.py:244-246`,
`self._data["prompt"] = {"text": prompt, ...}`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/rag_pipeline/test_rag_pipeline.py -k "context_layout or grouped or min_keep_floors" -v`
Expected: FAIL — `ImportError` (no `context_layout` param yet) or `AttributeError`

- [ ] **Step 3: Update the constructor**

In `rag_pipeline/rag_pipeline.py`, replace lines 251-259:

```python
    def __init__(
        self, retriever, generation_provider: GenerationProvider, chunk_store=None,
        prompt_version: str = "v2", context_prune_margin: float = 0.3,
        context_layout: ContextLayout = ContextLayout.FLAT,
    ):
        self._retriever = retriever
        self._generation_provider = generation_provider
        self._chunk_store = chunk_store
        self._prompt_version = prompt_version
        self._context_prune_margin = context_prune_margin
        self._context_layout = context_layout

    @property
    def context_layout(self) -> ContextLayout:
        return self._context_layout
```

(Insert the new `context_layout` property right after the existing `prompt_version` property, or anywhere among the other `@property` blocks in that section — match the file's existing property ordering.)

Add the import near the top of the file, alongside the existing `from rag_pipeline.context_builder import build_context` line:

```python
from rag_pipeline.context_builder import ContextLayout, build_context
```

And add to the models import (near `ContextChunk`'s home):

```python
from rag_hybrid_search.models import ChunkProvenance, ContextChunk
```

(Place this with whatever existing `rag_hybrid_search` imports the file already has — check the top of `rag_pipeline.py` for the exact existing import block to extend rather than adding a new duplicate import line.)

- [ ] **Step 4: Update `_prepare_context`**

Replace the body from `retrieved_chunks = _merge_multi_query_results(results_per_query)` through `return retrieved_chunks, context, prompt` (currently `rag_pipeline.py:367-378`):

```python
        retrieved_chunks, provenance_map = _merge_multi_query_results(results_per_query)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]
        required_chunks = max(3 if comparative else 1, len(subqueries))
        pruned_chunks = prune_by_score_margin(
            retrieved_chunks, self._context_prune_margin, min_keep=required_chunks,
        )
        dev_trace.log_pruning(retrieved_chunks, pruned_chunks)
        retrieved_chunks = pruned_chunks

        context_chunks = [
            ContextChunk(
                chunk=r,
                provenance=provenance_map.get(
                    r.chunk.chunk_id, ChunkProvenance(primary_subquery=0, all_subqueries=[0])
                ),
            )
            for r in retrieved_chunks
        ]
        dev_trace.log_provenance(context_chunks)

        layout = (
            ContextLayout.GROUPED
            if comparative and self._context_layout == ContextLayout.GROUPED
            else ContextLayout.FLAT
        )
        context = build_context(context_chunks, subqueries, layout=layout)
        prompt = build_prompt(question, context, prompt_version=self._prompt_version)
        dev_trace.log_prompt(prompt)
        return retrieved_chunks, context, prompt
```

The `provenance_map.get(..., ChunkProvenance(primary_subquery=0, all_subqueries=[0]))` fallback covers the single-query path, where `results_per_query` has exactly one list and every chunk's provenance is trivially `primary_subquery=0`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/rag_pipeline/test_rag_pipeline.py -v`
Expected: all pass, including the pre-existing tests in this file (confirms `_prepare_context`'s call-site update didn't regress anything else)

- [ ] **Step 6: Commit**

```bash
git add rag_pipeline/rag_pipeline.py tests/rag_pipeline/test_rag_pipeline.py
git commit -m "feat: wire ContextChunk provenance, adaptive min_keep, and layout selection into RagPipeline"
```

---

### Task 5: `RequestTrace.log_provenance`

**Files:**
- Modify: `rag_hybrid_search/trace.py` (add method after `log_pruning`, currently ending at `trace.py:242`)
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `ContextChunk` from Task 1.
- Produces: `RequestTrace.log_provenance(context_chunks: list[ContextChunk]) -> None`.

- [ ] **Step 1: Write the failing test**

Check `tests/test_trace.py` for its existing `RequestTrace` construction pattern (grep `RequestTrace(` in that file) and add, matching that pattern:

```python
def test_log_provenance_records_primary_and_all_subqueries():
    trace = RequestTrace("question", {})  # match this file's existing constructor call
    chunk = Chunk(
        chunk_id="c1", document_id="d1", chunk_index=0, text="hello",
        strategy_version="fixed-v1", heading=None, page=None, char_count=5,
    )
    retrieved = RetrievedChunk(
        chunk=chunk, dense_score=0.9, bm25_score=0.9, rrf_score=0.5,
        rerank_score=0.8, final_rank=1,
    )
    context_chunk = ContextChunk(
        chunk=retrieved,
        provenance=ChunkProvenance(primary_subquery=0, all_subqueries=[0, 1]),
    )

    trace.log_provenance([context_chunk])

    assert trace._data["provenance"] == {
        "c1": {"primary_subquery": 0, "all_subqueries": [0, 1]},
    }
```

(Import `Chunk, ChunkProvenance, ContextChunk, RetrievedChunk` from `rag_hybrid_search.models` and `RequestTrace` from wherever this test file already imports it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_trace.py::test_log_provenance_records_primary_and_all_subqueries -v`
Expected: FAIL — `AttributeError: 'RequestTrace' object has no attribute 'log_provenance'`

- [ ] **Step 3: Add the method**

In `rag_hybrid_search/trace.py`, immediately after `log_pruning` (after its closing line, currently `trace.py:242`):

```python
    def log_provenance(self, context_chunks: list) -> None:
        self._data["provenance"] = {
            cc.chunk.chunk.chunk_id: {
                "primary_subquery": cc.provenance.primary_subquery,
                "all_subqueries": cc.provenance.all_subqueries,
            }
            for cc in context_chunks
        }
        multi_subquery = [
            cc for cc in context_chunks if len(cc.provenance.all_subqueries) > 1
        ]
        if not self.enabled or not multi_subquery:
            return
        _section("STEP 5c -- CHUNK PROVENANCE")
        for cc in multi_subquery:
            others = [i for i in cc.provenance.all_subqueries if i != cc.provenance.primary_subquery]
            print(
                f"  chunk={cc.chunk.chunk.chunk_id[:12]}  "
                f"primary=subquery {cc.provenance.primary_subquery + 1}  "
                f"also matched=subqueries {[i + 1 for i in others]}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_trace.py::test_log_provenance_records_primary_and_all_subqueries -v`
Expected: PASS

- [ ] **Step 5: Run the full trace test file and the full suite**

Run: `uv run python -m pytest tests/test_trace.py -v`
Expected: all pass

Run: `uv run python -m pytest -q`
Expected: all pass — this confirms Tasks 1-5 together haven't broken anything elsewhere (e.g. `api/` routes that construct `RagPipeline` or call `build_context`/`_merge_multi_query_results` indirectly)

- [ ] **Step 6: Commit**

```bash
git add rag_hybrid_search/trace.py tests/test_trace.py
git commit -m "feat: add RequestTrace.log_provenance for chunk-to-subquery debugging"
```

---

### Task 6: Full-suite regression check + lint

**Files:**
- None created/modified — verification-only task.

- [ ] **Step 1: Run the full test suite**

Run: `uv run python -m pytest -q`
Expected: all pass, 0 failures. Pay particular attention to any test under `tests/api/` or `tests/rag_pipeline/test_end_to_end.py` that constructs a `RagPipeline` or calls `build_context`/`_merge_multi_query_results` directly — these are the most likely places an unrewritten call site could have been missed (grep first: `grep -rn "build_context(\|_merge_multi_query_results(" --include="*.py" . | grep -v tests/rag_pipeline/test_context_builder.py | grep -v tests/rag_pipeline/test_rag_pipeline.py | grep -v rag_pipeline/context_builder.py | grep -v rag_pipeline/rag_pipeline.py`; every hit must be updated to the new signatures).

- [ ] **Step 2: Run lint**

Run: `uv run ruff check .`
Expected: clean

- [ ] **Step 3: Manual smoke — confirm FLAT is still the default everywhere**

Run: `uv run python -c "from rag_pipeline.rag_pipeline import RagPipeline; import inspect; print(inspect.signature(RagPipeline.__init__))"`
Expected: signature includes `context_layout: ContextLayout = ContextLayout.FLAT`, confirming no existing caller's behavior changes without explicit opt-in.

- [ ] **Step 4: Commit (only if Step 1 required fixes)**

If any call site needed updating in Step 1, commit those fixes separately:

```bash
git add -A
git commit -m "fix: update remaining build_context/_merge_multi_query_results call sites for new signatures"
```

If no fixes were needed, skip this step — Task 6 is verification-only.

---

## Deferred (explicitly out of scope for this plan, per spec)

- Rendering a chunk under every `all_subqueries` entry (evidence duplication across groups).
- `ContextLayout.HIERARCHICAL` / `DOCUMENT_FIRST` / `COMPRESSED`.
- New multi-hop/definition/summarization budget tiers.
- Provenance-aware retrieval visualization beyond the plain-text `log_provenance` trace section.
- The evaluation gate itself (comparative accuracy, hallucination rate, verification pass rate, factual stability, token increase, latency) is a **manual run** using the existing Phase 2 `scripts/run_eval.py --compare-baseline` infrastructure, performed once after this plan's tasks land and before any caller is switched to `context_layout=ContextLayout.GROUPED` by default. Not a task in this plan — no production code defaults to `GROUPED` anywhere after Task 6; that switch is a separate, explicit follow-up decision gated on the eval run's results.
