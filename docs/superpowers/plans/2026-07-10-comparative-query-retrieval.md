# Comparative Query Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix comparative/multi-concept questions (e.g. "how do X and Y differ across A, B, C") that currently retrieve one semantic-center chunk and get pruned to a single chunk, starving the generator of evidence for anything but the dominant sub-topic.

**Architecture:** Detect comparative questions by keyword, decompose them into sub-queries via one small LLM call, run the existing `HybridRetriever.retrieve()` once per sub-query concurrently (unchanged internals — dense/sparse/fuse/rerank all stay as-is; a thread pool avoids latency scaling linearly with sub-query count), merge the reranked results by chunk_id with a diminishing-returns frequency bonus (drop duplicates, prefer chunks multiple sub-queries agree on), then let a widened adaptive-pruning floor keep enough chunks for a real comparison instead of collapsing to the single top chunk. Three additive, independently testable changes; no schema migrations.

**Tech Stack:** Python 3.11, Pydantic models, pytest, existing `GenerationProvider` protocol for the decomposition call.

## Global Constraints

- No changes to `HybridRetriever` internals — multi-query retrieval is achieved by calling `retrieve()` multiple times and merging in `rag_pipeline.py`, not by changing fusion/rerank code.
- Decomposition must degrade safely: if the LLM call fails, returns unparseable output, or returns zero sub-queries, fall back to treating the question as non-comparative (single retrieve, current behavior). Never raise out of the pipeline because decomposition failed.
- `prune_by_score_margin`'s existing single-question behavior (margin-based, no minimum) must be unchanged when `min_keep=1` (its current implicit default) — this is purely additive.
- Comparative detection is a plain keyword/regex check (no LLM call for the detection step itself) — cheap, always runs.

---

### Task 1: Comparative query detection + decomposition

**Files:**
- Create: `rag_pipeline/query_decomposer.py`
- Test: `tests/rag_pipeline/test_query_decomposer.py`

**Interfaces:**
- Produces: `is_comparative_query(question: str) -> bool`; `decompose_query(question: str, generation_provider, max_subqueries: int = 4, capture: dict | None = None) -> list[str]`. When `capture` is passed, `decompose_query` sets `capture["raw"]` to the LLM's raw decomposition output (or `None` if the provider call itself raised) before falling back on any validation failure -- this lets Task 2 log the raw LLM output to the trace for debugging even when decomposition was rejected. Task 2 consumes both functions.

- [ ] **Step 1: Write failing tests for comparative detection**

Create `tests/rag_pipeline/test_query_decomposer.py`:

```python
import json

from rag_pipeline.generation_provider import MockProvider
from rag_pipeline.query_decomposer import decompose_query, is_comparative_query


def test_is_comparative_query_detects_keywords():
    assert is_comparative_query("How do function-level and class-level patterns differ across RQ1, RQ2, RQ3?") is True
    assert is_comparative_query("Compare the results of RQ1 and RQ2.") is True
    assert is_comparative_query("What is the relationship between granularity and detectability?") is True


def test_is_comparative_query_detects_extended_phrasings():
    assert is_comparative_query("How is precision related to recall?") is True
    assert is_comparative_query("Explain function-level vs class-level detection.") is True
    assert is_comparative_query("Which model performs better on this benchmark?") is True
    assert is_comparative_query("Why does GPT-4 outperform GPT-3.5 here?") is True
    assert is_comparative_query("What are the pros and cons of each approach?") is True
    assert is_comparative_query("List the advantages of dense retrieval.") is True
    assert is_comparative_query("What are the tradeoffs of a larger chunk size?") is True
    assert is_comparative_query("What is the relative performance of the two rerankers?") is True
    assert is_comparative_query("Is there a correlation between chunk size and accuracy?") is True
    assert is_comparative_query("Contrast the two retrieval strategies.") is True


def test_is_comparative_query_false_for_simple_factual_question():
    assert is_comparative_query("What does the paper say about chunk overlap?") is False
    assert is_comparative_query("How many days of paid leave do employees get?") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_query_decomposer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.query_decomposer'`.

- [ ] **Step 3: Implement `is_comparative_query`**

Create `rag_pipeline/query_decomposer.py`:

```python
import json
import re

# Deliberately broad: false negatives silently fall back to single-query
# retrieval (today's unchanged behavior), so the cost of over-matching a
# borderline question is low, while under-matching a genuine comparison
# reproduces the original bug. Groups: direct comparison words, relational
# phrasing ("related to", "vs"), superiority/ranking ("outperform",
# "better", "worse"), and cost/benefit framing ("pros and cons",
# "advantages", "tradeoffs", "relative performance", "correlation").
_COMPARATIVE_RE = re.compile(
    r"\b("
    r"compare|comparison|comparative|difference|differ|differs|different|"
    r"versus|vs\.?|across|contrast|relationship|related|relative|between|"
    r"outperform|underperform|better|worse|superior|inferior|"
    r"pros and cons|advantage|disadvantage|tradeoff|trade-off|"
    r"correlation|correlate"
    r")\b",
    re.IGNORECASE,
)


def is_comparative_query(question: str) -> bool:
    """Cheap keyword check for questions that need evidence from more than
    one concept/section to answer well (e.g. "compare X and Y across A, B, C",
    "which performs better", "what's the tradeoff", "how is X related to Y").

    False negatives just mean the pipeline falls back to single-query
    retrieval (today's behavior) -- never a hard failure, so a plain
    regex is an acceptable heuristic here rather than an LLM call.
    """
    return bool(_COMPARATIVE_RE.search(question))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_query_decomposer.py -v`
Expected: all detection tests PASS; `decompose_query` tests (added next) still fail on import.

- [ ] **Step 5: Write failing tests for decomposition**

Add to `tests/rag_pipeline/test_query_decomposer.py`:

```python
def test_decompose_query_parses_llm_json_array():
    canned = json.dumps(["function-level detection patterns", "class-level detection patterns", "RQ1 findings", "RQ2 findings"])
    provider = MockProvider(canned_json=canned)

    subqueries = decompose_query(
        "How do function-level and class-level detection patterns differ across RQ1 and RQ2?",
        provider,
    )

    assert subqueries == [
        "function-level detection patterns",
        "class-level detection patterns",
        "RQ1 findings",
        "RQ2 findings",
    ]


def test_decompose_query_caps_at_max_subqueries():
    canned = json.dumps(["a", "b", "c", "d", "e", "f"])
    provider = MockProvider(canned_json=canned)

    subqueries = decompose_query("compare a, b, c, d, e, f", provider, max_subqueries=3)

    assert subqueries == ["a", "b", "c"]


def test_decompose_query_falls_back_to_original_question_on_malformed_json():
    provider = MockProvider(canned_json="not json at all")

    subqueries = decompose_query("compare X and Y", provider)

    assert subqueries == ["compare X and Y"]


def test_decompose_query_falls_back_to_original_question_on_empty_array():
    provider = MockProvider(canned_json="[]")

    subqueries = decompose_query("compare X and Y", provider)

    assert subqueries == ["compare X and Y"]


def test_decompose_query_falls_back_to_original_question_on_provider_exception():
    class RaisingProvider:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("provider down")

    subqueries = decompose_query("compare X and Y", RaisingProvider())

    assert subqueries == ["compare X and Y"]


def test_decompose_query_rejects_single_subquery_that_echoes_the_question():
    """The LLM sometimes 'decomposes' a question into itself, verbatim or
    with only whitespace/case differences -- that's not a decomposition,
    it's a no-op wearing a JSON array. Treat it as a failed decomposition
    and fall back, same as malformed JSON."""
    question = "How do RQ1 and RQ2 differ?"
    provider = MockProvider(canned_json=json.dumps([question]))

    subqueries = decompose_query(question, provider)

    assert subqueries == [question]


def test_decompose_query_rejects_single_subquery_echo_case_and_whitespace_insensitive():
    question = "How do RQ1 and RQ2 differ?"
    provider = MockProvider(canned_json=json.dumps([f"  {question.upper()}  "]))

    subqueries = decompose_query(question, provider)

    assert subqueries == [question]


def test_decompose_query_trusts_short_but_specific_subqueries():
    """Word count is not a specificity signal: "RQ1 findings", "SOC2
    evidence", and "OAuth flow" are all short (2 words) but carry real
    retrieval signal (proper nouns / alphanumeric identifiers). A
    word-count floor would wrongly reject these -- validation must not
    second-guess sub-query quality beyond the echo check."""
    question = "Compare RQ1, SOC2, and OAuth handling."
    provider = MockProvider(canned_json=json.dumps(["RQ1", "SOC2", "OAuth"]))

    subqueries = decompose_query(question, provider)

    assert subqueries == ["RQ1", "SOC2", "OAuth"]


def test_decompose_query_captures_raw_llm_output_when_requested():
    """The `capture` dict is a debug-mode side channel: even when
    validation rejects the LLM's output and falls back, the caller (trace
    logging) can still see exactly what the LLM returned."""
    canned = json.dumps(["RQ1 findings", "RQ2 findings"])
    provider = MockProvider(canned_json=canned)
    capture: dict = {}

    decompose_query("compare RQ1 and RQ2", provider, capture=capture)

    assert capture["raw"] == canned


def test_decompose_query_captures_raw_output_even_on_validation_failure():
    question = "compare X and Y across many concepts"
    canned = json.dumps([question])
    provider = MockProvider(canned_json=canned)
    capture: dict = {}

    subqueries = decompose_query(question, provider, capture=capture)

    assert subqueries == [question]
    assert capture["raw"] == canned


def test_decompose_query_captures_none_on_provider_exception():
    class RaisingProvider:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("provider down")

    capture: dict = {}
    decompose_query("compare X and Y", RaisingProvider(), capture=capture)

    assert capture["raw"] is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_query_decomposer.py -v`
Expected: all eleven new tests FAIL with `ImportError: cannot import name 'decompose_query'`.

- [ ] **Step 7: Implement `decompose_query` with quality validation**

Add to `rag_pipeline/query_decomposer.py`. The prompt explicitly asks for
sub-queries ordered most-important-first, so truncating to
``max_subqueries`` (Step below) keeps the highest-importance items rather
than an arbitrary prefix of an unordered list.

Quality validation deliberately checks only for an exact (whitespace/case-
insensitive) echo of the question, not word count or "genericity": short
sub-queries like "RQ1", "SOC2", "OAuth flow" carry just as much retrieval
signal as longer ones (they're proper nouns / identifiers, not vague
filler), so a length-based heuristic would reject good decompositions
along with bad ones. An echo is the one case that's unambiguously a
non-decomposition -- the LLM handed back the input, not a breakdown of it.

```python
_DECOMPOSITION_PROMPT_TEMPLATE = """The following question asks for a comparison across multiple concepts, sections, or sources. List the distinct concepts that need to be retrieved separately to answer it well, as a JSON array of short search-query strings (no more than {max_subqueries} items), ordered from most important to least important. Respond with ONLY the JSON array, no prose.

Question: {question}

Example:
Question: How do function-level and class-level detection patterns differ across RQ1 and RQ2?
["function-level detection patterns", "class-level detection patterns", "RQ1 findings", "RQ2 findings"]
"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def decompose_query(
    question: str, generation_provider, max_subqueries: int = 4, capture: dict | None = None,
) -> list[str]:
    """Break a comparative question into independent sub-queries so each
    referenced concept gets its own retrieval pass instead of being averaged
    into one dominant embedding.

    Never raises: any failure -- provider exception, malformed JSON, empty
    result, or a single sub-query that just echoes the question back --
    falls back to ``[question]``, identical to today's non-comparative
    single-retrieve behavior. Deliberately does NOT reject sub-queries for
    being short or "generic" by word count -- that conflates length with
    specificity (see Step 7 note above) and would drop useful short
    identifiers (e.g. "RQ1", "SOC2") along with genuinely vague output.

    When `capture` is provided, `capture["raw"]` is set to the provider's
    raw response string (or `None` if `generation_provider.generate` itself
    raised) regardless of whether validation later rejects it -- callers
    that want to debug *why* decomposition fell back can inspect exactly
    what the LLM returned.
    """
    prompt = _DECOMPOSITION_PROMPT_TEMPLATE.format(question=question, max_subqueries=max_subqueries)
    raw = None
    try:
        raw = generation_provider.generate(prompt)
        parsed = json.loads(raw)
    except Exception:
        if capture is not None:
            capture["raw"] = raw
        return [question]

    if capture is not None:
        capture["raw"] = raw

    if not isinstance(parsed, list) or not parsed:
        return [question]

    subqueries = [str(item) for item in parsed if str(item).strip()]
    if not subqueries:
        return [question]

    if len(subqueries) == 1 and _normalize(subqueries[0]) == _normalize(question):
        return [question]

    return subqueries[:max_subqueries]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_query_decomposer.py -v`
Expected: all PASS.

- [ ] **Step 9: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures.

- [ ] **Step 10: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/query_decomposer.py tests/rag_pipeline/test_query_decomposer.py
git commit -m "$(cat <<'EOF'
feat: add comparative query detection and LLM-based decomposition

is_comparative_query() flags questions like "compare X and Y across
A, B, C" via an expanded keyword regex (comparison, relational,
superiority, and cost/benefit phrasing). decompose_query() asks the
generation provider to split them into independent sub-queries,
rejecting only a single sub-query that echoes the question back
verbatim (word-count/genericity heuristics were deliberately rejected
-- they'd drop good short identifiers like "RQ1"/"SOC2" along with bad
output) and falling back to the original question on any failure.
EOF
)"
```

---

### Task 2: Multi-query retrieval merge in RagPipeline

**Files:**
- Modify: `rag_pipeline/rag_pipeline.py`
- Modify: `rag_hybrid_search/trace.py`
- Test: `tests/rag_pipeline/test_rag_pipeline.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `is_comparative_query`, `decompose_query` (Task 1, including its `capture` param); `HybridRetriever.retrieve(query, dev_trace=None) -> tuple[list[RetrievedChunk], RetrievalTrace]` (existing, unchanged).
- Produces: `_merge_multi_query_results(results_per_query: list[list[RetrievedChunk]]) -> list[RetrievedChunk]` (module-level helper in `rag_pipeline.py`); `RequestTrace.log_query_decomposition(self, is_comparative: bool, subqueries: list[str], raw_llm_output: str | None, concepts_retrieved: int) -> None`.

- [ ] **Step 1: Write failing test for the merge helper**

Add to `tests/rag_pipeline/test_rag_pipeline.py` (uses `make_retrieved_chunk` already defined in this file):

```python
from rag_pipeline.rag_pipeline import _merge_multi_query_results


def test_merge_multi_query_results_dedupes_by_chunk_id_keeping_best_score():
    """Same chunk_id retrieved by two different sub-queries: keep the
    higher rerank_score (as the chunk's displayed score), drop the
    duplicate, and re-rank the merged list."""
    q1_results = [
        make_retrieved_chunk("c1", "text1", rerank_score=2.0, final_rank=1),
        make_retrieved_chunk("c2", "text2", rerank_score=1.0, final_rank=2),
    ]
    q2_results = [
        make_retrieved_chunk("c1", "text1", rerank_score=3.0, final_rank=1),
        make_retrieved_chunk("c3", "text3", rerank_score=0.5, final_rank=2),
    ]

    merged = _merge_multi_query_results([q1_results, q2_results])

    assert [r.chunk.chunk_id for r in merged] == ["c1", "c2", "c3"]
    assert merged[0].rerank_score == 3.0
    assert [r.final_rank for r in merged] == [1, 2, 3]


def test_merge_multi_query_results_boosts_chunks_appearing_in_multiple_subqueries():
    """A chunk that surfaces in every sub-query's retrieval is a stronger
    relevance signal than a slightly higher-scoring chunk that only one
    sub-query found. The log2-scaled frequency bonus must be able to flip
    the ranking: chunk A appears 4x at score 2.0 (2.0 + 0.15*log2(4) = 2.3),
    chunk B appears once at score 2.1 -- A ranks first."""
    a = make_retrieved_chunk("a", "text-a", rerank_score=2.0, final_rank=1)
    b = make_retrieved_chunk("b", "text-b", rerank_score=2.1, final_rank=1)
    results_per_query = [[a], [a], [a], [a, b]]

    merged = _merge_multi_query_results(results_per_query)

    assert [r.chunk.chunk_id for r in merged] == ["a", "b"]


def test_merge_multi_query_results_single_appearance_uses_raw_score():
    """No frequency bonus when a chunk appears in only one sub-query's
    results -- ranking degrades to plain best-rerank-score, same as before
    frequency weighting was added (log2(1) == 0, no bonus)."""
    a = make_retrieved_chunk("a", "text-a", rerank_score=2.0, final_rank=1)
    b = make_retrieved_chunk("b", "text-b", rerank_score=2.1, final_rank=1)

    merged = _merge_multi_query_results([[a], [b]])

    assert [r.chunk.chunk_id for r in merged] == ["b", "a"]


def test_merge_multi_query_results_frequency_bonus_does_not_overwhelm_a_much_higher_score():
    """Regression guard for a linear (non-diminishing) bonus over-promoting
    a frequently-occurring but clearly weaker chunk: A scores 1.5 and
    appears 8x (1.5 + 0.15*log2(8) = 1.5 + 0.45 = 1.95), B scores 2.2 and
    appears once (bonus 0). A linear bonus of 0.1/appearance would have
    given A 1.5 + 0.7 = 2.2, tying or beating B -- log2 scaling keeps A
    below B here, as it should given the score gap."""
    a = make_retrieved_chunk("a", "text-a", rerank_score=1.5, final_rank=1)
    b = make_retrieved_chunk("b", "text-b", rerank_score=2.2, final_rank=1)
    results_per_query = [[a]] * 7 + [[a, b]]

    merged = _merge_multi_query_results(results_per_query)

    assert [r.chunk.chunk_id for r in merged] == ["b", "a"]


def test_merge_multi_query_results_handles_missing_rerank_score():
    """PassthroughReranker never sets rerank_score -- merge must not crash,
    just fall back to original retrieval order (no score to sort by)."""
    q1_results = [make_retrieved_chunk("c1", "text1", rerank_score=None, final_rank=1)]
    q2_results = [make_retrieved_chunk("c2", "text2", rerank_score=None, final_rank=1)]

    merged = _merge_multi_query_results([q1_results, q2_results])

    assert {r.chunk.chunk_id for r in merged} == {"c1", "c2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py -k merge_multi_query -v`
Expected: FAIL with `ImportError: cannot import name '_merge_multi_query_results'`.

- [ ] **Step 3: Implement `_merge_multi_query_results` with a frequency-weighted rank**

Add `import math` to the existing top-of-file import block in `rag_pipeline/rag_pipeline.py` (alongside `import json`, `import re`, etc.). Then add the following to `rag_pipeline/rag_pipeline.py`, near the other module-level helpers (after `_inline_citation_drift`):

```python
_FREQUENCY_BONUS_SCALE = 0.15


def _merge_multi_query_results(results_per_query: list[list]) -> list:
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
    """
    best_by_id: dict[str, object] = {}
    appearances: dict[str, int] = {}
    for results in results_per_query:
        for r in results:
            chunk_id = r.chunk.chunk_id
            appearances[chunk_id] = appearances.get(chunk_id, 0) + 1
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
    return [r.model_copy(update={"final_rank": i}) for i, r in enumerate(merged, start=1)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py -k merge_multi_query -v`
Expected: PASS.

- [ ] **Step 5: Write failing pipeline-level tests**

Add `import threading` to the existing top-of-file import block in `tests/rag_pipeline/test_rag_pipeline.py`. Then add to `tests/rag_pipeline/test_rag_pipeline.py`:

```python
class MultiQueryFakeRetriever:
    """Records every query it was asked to retrieve for, and returns a
    different chunk per distinct query string so tests can verify
    decomposition actually reached the retriever.

    Multiple sub-query retrievals now run concurrently (Task 2, Step 7),
    so `queries_seen` may be appended to from several threads -- a lock
    keeps the list itself consistent, though the *order* of entries across
    different sub-queries is no longer deterministic (tests assert set
    membership, not list order, wherever more than one sub-query is
    involved)."""

    def __init__(self, chunks_by_query: dict[str, list]):
        self._chunks_by_query = chunks_by_query
        self.queries_seen: list[str] = []
        self._lock = threading.Lock()

    def retrieve(self, query, dev_trace=None):
        with self._lock:
            self.queries_seen.append(query)
        return self._chunks_by_query.get(query, []), RetrievalTrace()


def test_comparative_question_retrieves_once_per_subquery():
    rq1_chunk = make_retrieved_chunk("rq1", "RQ1 finding: granularity matters.", rerank_score=2.0)
    rq2_chunk = make_retrieved_chunk("rq2", "RQ2 finding: class-level differs.", rerank_score=1.5, final_rank=2)
    retriever = MultiQueryFakeRetriever({
        "RQ1 findings": [rq1_chunk],
        "RQ2 findings": [rq2_chunk],
    })
    decompose_canned = json.dumps(["RQ1 findings", "RQ2 findings"])
    answer_canned = json.dumps({
        "answer": "RQ1 shows granularity matters [d1] and RQ2 shows class-level differs [d2].",
        "claims": [
            {"text": "Granularity matters.", "citation_ids": ["d1"]},
            {"text": "Class-level differs.", "citation_ids": ["d2"]},
        ],
    })
    provider = MockProvider(canned_json=answer_canned)
    # Decomposition and generation share one provider in RagPipeline; swap
    # canned output between the two calls via a small wrapper.
    calls = {"n": 0}
    original_generate = provider.generate

    def generate(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return decompose_canned
        return original_generate(prompt, **kwargs)

    provider.generate = generate

    pipeline = RagPipeline(retriever, provider)
    result = pipeline.answer("How do function-level and class-level detection patterns differ across RQ1 and RQ2?")

    # Concurrent execution (Step 7) means call order across sub-queries
    # isn't guaranteed -- assert membership, not sequence.
    assert set(retriever.queries_seen) == {"RQ1 findings", "RQ2 findings"}
    assert result.error is None
    assert {c for c in result.citations} == {"d1", "d2"}


def test_non_comparative_question_retrieves_once_with_original_question():
    chunk = make_retrieved_chunk("c1", "Employees get 20 days of paid leave.")
    retriever = MultiQueryFakeRetriever({"How many days of paid leave?": [chunk]})
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{"text": "Employees get 20 days of paid leave.", "citation_ids": ["d1"]}],
    })
    pipeline = RagPipeline(retriever, MockProvider(canned_json=canned))

    result = pipeline.answer("How many days of paid leave?")

    assert retriever.queries_seen == ["How many days of paid leave?"]
    assert result.error is None


def test_one_failing_subquery_does_not_abort_the_others():
    """Regression test for concurrent retrieval: RQ2's retrieve() raising
    must not lose RQ1 and RQ3's results or blow up the whole request --
    the failing sub-query just contributes no chunks."""

    class FlakyMultiQueryRetriever:
        def retrieve(self, query, dev_trace=None):
            if query == "RQ2 findings":
                raise RuntimeError("simulated retrieval timeout")
            chunks = {
                "RQ1 findings": [make_retrieved_chunk("rq1", "RQ1 finding.", rerank_score=2.0)],
                "RQ3 findings": [make_retrieved_chunk("rq3", "RQ3 finding.", rerank_score=1.0)],
            }
            return chunks.get(query, []), RetrievalTrace()

    decompose_canned = json.dumps(["RQ1 findings", "RQ2 findings", "RQ3 findings"])
    answer_canned = json.dumps({
        "answer": "RQ1 shows one finding [d1] and RQ3 shows another [d2].",
        "claims": [
            {"text": "RQ1 finding.", "citation_ids": ["d1"]},
            {"text": "RQ3 finding.", "citation_ids": ["d2"]},
        ],
    })
    provider = MockProvider(canned_json=answer_canned)
    calls = {"n": 0}
    original_generate = provider.generate

    def generate(prompt, **kwargs):
        calls["n"] += 1
        return decompose_canned if calls["n"] == 1 else original_generate(prompt, **kwargs)

    provider.generate = generate

    pipeline = RagPipeline(FlakyMultiQueryRetriever(), provider)
    result = pipeline.answer("Compare RQ1, RQ2, and RQ3 findings.")

    assert result.error is None
    assert {c for c in result.citations} == {"d1", "d2"}
```

- [ ] **Step 5b: Write failing unit tests for the concurrency helper directly**

Add to `tests/rag_pipeline/test_rag_pipeline.py`:

```python
from rag_pipeline.rag_pipeline import _MAX_CONCURRENT_RETRIEVAL_WORKERS, _retrieve_subqueries_concurrently


def test_retrieve_subqueries_concurrently_caps_worker_count(monkeypatch):
    """Worker count must never exceed _MAX_CONCURRENT_RETRIEVAL_WORKERS,
    even if called with many more sub-queries than that -- thread creation
    shouldn't scale 1:1 with an eventually-configurable max_subqueries."""
    seen_max_workers = {}
    real_executor_init = ThreadPoolExecutor.__init__

    def recording_init(self, max_workers=None, *args, **kwargs):
        seen_max_workers["value"] = max_workers
        return real_executor_init(self, max_workers=max_workers, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", recording_init)

    many_subqueries = [f"q{i}" for i in range(_MAX_CONCURRENT_RETRIEVAL_WORKERS + 6)]
    _retrieve_subqueries_concurrently(many_subqueries, lambda q, trace: ([], None))

    assert seen_max_workers["value"] == _MAX_CONCURRENT_RETRIEVAL_WORKERS


def test_retrieve_subqueries_concurrently_isolates_one_failure():
    def flaky_retrieve(q, trace):
        if q == "bad":
            raise RuntimeError("boom")
        return [q], None

    results = _retrieve_subqueries_concurrently(["good1", "bad", "good2"], flaky_retrieve)

    assert results == [["good1"], [], ["good2"]]
```

Add `from concurrent.futures import ThreadPoolExecutor` to the top of `tests/rag_pipeline/test_rag_pipeline.py` if not already present (needed to monkeypatch it in the worker-cap test).

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py -k "comparative or non_comparative or concurrently or failing_subquery" -v`
Expected: FAIL -- `RagPipeline.answer` currently calls `self._retriever.retrieve(question, ...)` exactly once, unconditionally, so `queries_seen` won't match; `_retrieve_subqueries_concurrently` doesn't exist yet.

- [ ] **Step 7: Wire decomposition + multi-query retrieval into `RagPipeline.answer`**

In `rag_pipeline/rag_pipeline.py`, add the import:

```python
from rag_pipeline.query_decomposer import decompose_query, is_comparative_query
```

Replace the retrieval block in `answer()` (currently):

```python
        if self._chunk_store is not None:
            retrieved_chunks, _trace = route_query(question, self._chunk_store, self._retriever, dev_trace=dev_trace)
        else:
            retrieved_chunks, _trace = self._retriever.retrieve(question, dev_trace=dev_trace)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]
```

with:

```python
        comparative = is_comparative_query(question)
        decompose_capture: dict = {}
        subqueries = (
            decompose_query(question, self._generation_provider, capture=decompose_capture)
            if comparative else [question]
        )

        def _retrieve_one(q: str, trace_for_call):
            if self._chunk_store is not None:
                return route_query(q, self._chunk_store, self._retriever, dev_trace=trace_for_call)[0]
            return self._retriever.retrieve(q, dev_trace=trace_for_call)[0]

        if len(subqueries) == 1:
            results_per_query = [_retrieve_one(subqueries[0], dev_trace)]
        else:
            results_per_query = _retrieve_subqueries_concurrently(subqueries, _retrieve_one)

        concepts_retrieved = sum(1 for results in results_per_query if results)
        dev_trace.log_query_decomposition(
            comparative, subqueries, decompose_capture.get("raw"), concepts_retrieved,
        )
        retrieved_chunks = _merge_multi_query_results(results_per_query)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]
```

Add the import at the top of `rag_pipeline/rag_pipeline.py` (alongside the other stdlib imports):

```python
import logging
from concurrent.futures import ThreadPoolExecutor
```

If `rag_pipeline.py` doesn't already have a module-level logger, add one right after the imports:

```python
logger = logging.getLogger(__name__)
```

Add the concurrency helper as a module-level function (near `_merge_multi_query_results`):

```python
_MAX_CONCURRENT_RETRIEVAL_WORKERS = 4


def _retrieve_subqueries_concurrently(subqueries: list[str], retrieve_one) -> list[list]:
    """Run `retrieve_one(q, None)` for every sub-query in parallel.

    Worker count is capped at `_MAX_CONCURRENT_RETRIEVAL_WORKERS` regardless
    of how many sub-queries there are, so if `max_subqueries` is ever raised
    well beyond today's default of 4, thread creation doesn't scale
    1:1 with it.

    A single sub-query's retrieve() call failing (provider timeout,
    connection error, etc.) must not abort the other sub-queries or the
    whole request -- each future is resolved individually, a failure is
    logged and contributes an empty result list for that sub-query (it
    simply doesn't show up in the merged context and drags down
    `concepts_retrieved`/coverage), and every other sub-query's results
    are still used. dev_trace is intentionally not threaded through here
    (see the call site) -- concurrent writers aren't safe against the
    shared RequestTrace state.
    """
    max_workers = min(len(subqueries), _MAX_CONCURRENT_RETRIEVAL_WORKERS)
    results: list[list] = [[] for _ in subqueries]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(retrieve_one, q, None): i for i, q in enumerate(subqueries)
        }
        for future in future_to_index:
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception:
                logger.exception("retrieve() failed for sub-query %r -- treating as empty", subqueries[i])
                results[i] = []
    return results
```

Apply the identical replacement to the matching block in `answer_stream()`. Note `log_query_decomposition` is now called after retrieval (not before, as in the original draft) because `concepts_retrieved` needs the per-subquery results -- this only changes trace ordering, not retrieval behavior. `results` is pre-sized and written by index (`future_to_index[future]`), so `results_per_query` always lines up positionally with `subqueries` regardless of completion order or per-future failure -- but the *order in which the retriever itself is called* is still not guaranteed across threads (see the test fix below).

- [ ] **Step 8: Add `log_query_decomposition` to the trace module**

In `rag_hybrid_search/trace.py`, add a new method near `log_pruning`:

```python
    def log_query_decomposition(
        self, is_comparative: bool, subqueries: list[str],
        raw_llm_output: str | None, concepts_retrieved: int,
    ) -> None:
        coverage = concepts_retrieved / len(subqueries) if subqueries else 0.0
        self._data["query_decomposition"] = {
            "comparative": is_comparative, "subqueries": subqueries,
            "raw_llm_output": raw_llm_output,
            "concepts_requested": len(subqueries), "concepts_retrieved": concepts_retrieved,
            "coverage": coverage,
        }
        if not self.enabled:
            return
        _section("STEP 1b -- QUERY DECOMPOSITION")
        _kv(Comparative=is_comparative, **{
            "Concepts requested": len(subqueries),
            "Concepts retrieved": concepts_retrieved,
            "Coverage": f"{coverage * 100:.0f}%",
        })
        for i, q in enumerate(subqueries, 1):
            print(f"  {i}. {q!r}")
        if raw_llm_output is not None:
            print(f"\n  Raw decomposition output: {raw_llm_output!r}")
```

- [ ] **Step 9: Write failing trace test**

Add to `tests/test_trace.py`:

```python
def test_log_query_decomposition_records_subqueries_raw_output_and_coverage(monkeypatch, capsys):
    monkeypatch.setenv("TRACE_RAG", "true")
    trace = RequestTrace("question", {})

    trace.log_query_decomposition(
        True, ["RQ1 findings", "RQ2 findings", "RQ3 findings"],
        raw_llm_output='["RQ1 findings", "RQ2 findings", "RQ3 findings"]',
        concepts_retrieved=2,
    )

    data = trace._data["query_decomposition"]
    assert data["subqueries"] == ["RQ1 findings", "RQ2 findings", "RQ3 findings"]
    assert data["concepts_requested"] == 3
    assert data["concepts_retrieved"] == 2
    assert data["coverage"] == pytest.approx(2 / 3)
    out = capsys.readouterr().out
    assert "QUERY DECOMPOSITION" in out
    assert "RQ1 findings" in out
    assert "Coverage" in out
    assert "67%" in out
    assert "Raw decomposition output" in out


def test_log_query_decomposition_coverage_zero_when_no_subqueries():
    trace = RequestTrace("question", {})

    trace.log_query_decomposition(False, [], raw_llm_output=None, concepts_retrieved=0)

    assert trace._data["query_decomposition"]["coverage"] == 0.0
```

- [ ] **Step 10: Run trace test to verify it fails, then passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_trace.py -v`
Expected: fails first (`AttributeError: 'RequestTrace' object has no attribute 'log_query_decomposition'`) before Step 8, passes after.

- [ ] **Step 11: Run the pipeline tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py tests/test_trace.py -v`
Expected: all PASS.

- [ ] **Step 12: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures. (Existing single-question tests must keep passing unchanged — `is_comparative_query` returns `False` for all of them, so they take the one-subquery path with `subqueries == [question]`, identical to today's single `retrieve()` call.)

- [ ] **Step 13: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/rag_pipeline.py rag_hybrid_search/trace.py tests/rag_pipeline/test_rag_pipeline.py tests/test_trace.py
git commit -m "$(cat <<'EOF'
feat: retrieve once per decomposed sub-query for comparative questions

Comparative questions ("compare X and Y across A, B, C") no longer
collapse to one dominant-embedding retrieval. Each decomposed
sub-query runs the existing retrieve() pipeline independently and
concurrently (worker count capped at 4 regardless of sub-query count;
one sub-query's retrieve() failing is isolated and logged, not fatal
to the request); results are merged by chunk_id with a log2-scaled
frequency bonus (a chunk surfaced by multiple sub-queries outranks a
single-query chunk with a marginally higher raw rerank_score, but
diminishing returns prevent a weak chunk with many appearances from
out-ranking a much stronger one). Non-comparative questions are
unaffected -- decompose_query returns [question] unchanged, so they
still retrieve exactly once, sequentially. Trace gains per-request
concepts requested/retrieved/coverage plus the raw LLM decomposition
output for debugging incomplete comparative answers.
EOF
)"
```

---

### Task 3: Adaptive pruning floor for comparative questions

**Files:**
- Modify: `rag_pipeline/context_pruning.py`
- Modify: `rag_pipeline/rag_pipeline.py`
- Test: `tests/rag_pipeline/test_context_pruning.py`

**Interfaces:**
- Consumes: `is_comparative_query` (Task 1, already imported in `rag_pipeline.py` from Task 2).
- Produces: `prune_by_score_margin(chunks, margin, min_keep: int = 1) -> list[RetrievedChunk]` (signature gains an optional, defaulted 3rd param — existing call sites without it are unaffected).

- [ ] **Step 1: Write failing test for the min_keep floor**

Add to `tests/rag_pipeline/test_context_pruning.py`:

```python
def test_min_keep_prevents_over_pruning_below_the_floor():
    """A tight margin would normally prune down to 1 chunk (the top
    scorer dominates the range) -- min_keep=3 must override that and keep
    the top 3 regardless of how wide the score gap is."""
    chunks = [
        make_chunk_with_score("c1", 5.0),
        make_chunk_with_score("c2", 0.1),
        make_chunk_with_score("c3", 0.05),
        make_chunk_with_score("c4", 0.01),
    ]

    result = prune_by_score_margin(chunks, margin=0.1, min_keep=3)

    assert len(result) == 3
    assert [c.chunk.chunk_id for c in result] == ["c1", "c2", "c3"]


def test_min_keep_is_noop_when_margin_already_keeps_more():
    chunks = [
        make_chunk_with_score("c1", 1.0),
        make_chunk_with_score("c2", 0.95),
        make_chunk_with_score("c3", 0.9),
    ]

    result = prune_by_score_margin(chunks, margin=0.5, min_keep=1)

    assert len(result) == 3


def test_min_keep_default_preserves_existing_behavior():
    """Existing call sites that don't pass min_keep must behave exactly as
    before -- default min_keep=1 means "no floor beyond the margin rule"."""
    chunks = [
        make_chunk_with_score("c1", 5.0),
        make_chunk_with_score("c2", 0.1),
    ]

    result = prune_by_score_margin(chunks, margin=0.1)

    assert len(result) == 1
    assert result[0].chunk.chunk_id == "c1"
```

Check whether `make_chunk_with_score` already exists in this test file before adding it — if it's missing, add it near the top:

```python
def make_chunk_with_score(chunk_id: str, rerank_score: float):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text="text",
        strategy_version="fixed-v1", heading=None, page=None, char_count=4,
    )
    return RetrievedChunk(
        chunk=chunk, dense_score=0.5, bm25_score=0.5, rrf_score=0.5,
        rerank_score=rerank_score, final_rank=1,
    )
```

(Import `Chunk, RetrievedChunk` from `rag_hybrid_search.models` at the top of the file if not already imported.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_context_pruning.py -v`
Expected: the three new tests FAIL with `TypeError: prune_by_score_margin() got an unexpected keyword argument 'min_keep'`.

- [ ] **Step 3: Implement `min_keep`**

In `rag_pipeline/context_pruning.py`, replace the function signature and body:

```python
def prune_by_score_margin(chunks: list[RetrievedChunk], margin: float, min_keep: int = 1) -> list[RetrievedChunk]:
    """Drop chunks whose rerank_score falls more than `margin` of the
    top-to-bottom score range below the top chunk, but never prune below
    `min_keep` chunks -- comparative questions need multiple supporting
    chunks even when the reranker is confident about a single top result.

    No-op (returns chunks unchanged) when: fewer than 2 chunks, any chunk is
    missing rerank_score (PassthroughReranker never scores candidates -- no
    ground truth to prune by), or all chunks score identically (no basis to
    discriminate). Chunks are assumed already sorted best-first.
    """
    if len(chunks) < 2:
        return chunks
    scores = [c.rerank_score for c in chunks]
    if any(s is None for s in scores):
        return chunks

    top_score = scores[0]
    score_range = top_score - min(scores)
    if score_range <= 0:
        return chunks

    threshold = top_score - margin * score_range
    pruned = [c for c in chunks if c.rerank_score >= threshold]
    if len(pruned) < min_keep:
        return chunks[:min_keep]
    return pruned
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_context_pruning.py -v`
Expected: all PASS, including pre-existing tests in this file (default `min_keep=1` never raises the floor above what the margin rule already keeps for non-comparative cases).

- [ ] **Step 5: Wire `min_keep=3` for comparative questions in `RagPipeline`**

In `rag_pipeline/rag_pipeline.py`, in `answer()`, replace:

```python
        pruned_chunks = prune_by_score_margin(retrieved_chunks, self._context_prune_margin)
```

with:

```python
        pruned_chunks = prune_by_score_margin(
            retrieved_chunks, self._context_prune_margin, min_keep=3 if comparative else 1,
        )
```

Apply the identical change in `answer_stream()`.

- [ ] **Step 6: Write failing pipeline-level regression test for the original bug**

Add to `tests/rag_pipeline/test_rag_pipeline.py`:

```python
def test_comparative_question_keeps_multiple_chunks_after_pruning():
    """Regression test for the production bug: a comparative question
    whose reranker gave one chunk a dominant score used to get pruned down
    to 1 chunk, starving the generator of evidence for the other concepts.
    min_keep=3 for comparative questions prevents that collapse."""
    chunks_by_query = {
        "RQ1 findings": [make_retrieved_chunk("rq1", "RQ1: granularity matters.", rerank_score=5.0)],
        "RQ2 findings": [make_retrieved_chunk("rq2", "RQ2: class-level differs.", rerank_score=0.2, final_rank=2)],
        "RQ3 findings": [make_retrieved_chunk("rq3", "RQ3: features overlap.", rerank_score=0.1, final_rank=3)],
    }
    retriever = MultiQueryFakeRetriever(chunks_by_query)
    decompose_canned = json.dumps(["RQ1 findings", "RQ2 findings", "RQ3 findings"])
    answer_canned = json.dumps({
        "answer": "RQ1 shows granularity matters [d1], RQ2 shows class-level differs [d2], RQ3 shows features overlap [d3].",
        "claims": [
            {"text": "Granularity matters.", "citation_ids": ["d1"]},
            {"text": "Class-level differs.", "citation_ids": ["d2"]},
            {"text": "Features overlap.", "citation_ids": ["d3"]},
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
    result = pipeline.answer("How do RQ1, RQ2, and RQ3 findings differ?")

    assert result.error is None
    assert {c for c in result.citations} == {"d1", "d2", "d3"}
```

- [ ] **Step 7: Run test to verify it fails, then passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py -k comparative_question_keeps_multiple -v`
Expected: fails before Step 5 (margin rule alone prunes rq2/rq3 out given the 5.0 vs 0.2/0.1 score spread with the default `context_prune_margin=0.3`), passes after.

- [ ] **Step 8: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures.

- [ ] **Step 9: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/context_pruning.py rag_pipeline/rag_pipeline.py tests/rag_pipeline/test_context_pruning.py tests/rag_pipeline/test_rag_pipeline.py
git commit -m "$(cat <<'EOF'
fix: keep at least 3 chunks after pruning for comparative questions

prune_by_score_margin gains an optional min_keep floor (default 1,
preserving existing single-question behavior). RagPipeline passes
min_keep=3 when is_comparative_query() is true, so a dominant top
chunk's score margin can no longer collapse a multi-concept question's
context down to a single chunk before generation.
EOF
)"
```

---

## Deferred (not in this plan)

- True query-graph decomposition (nested sub-questions, e.g. "A because B, compared to C") -- this plan handles flat lists of concepts only.
- Caching/reusing decomposition results across repeated similar questions.
- Exposing `max_subqueries` (hardcoded 4), `min_keep` (hardcoded 3), and `_FREQUENCY_BONUS_SCALE` (hardcoded 0.15) via `Settings` -- left as constants until real usage data suggests specific values worth tuning per-deployment.
- Cost/latency accounting for the extra decomposition LLM call and the N concurrent retrieve() calls (N = number of sub-queries) in `ConfidenceScores` or the trace timing summary.
- Richer per-subquery coverage: this plan's `coverage` field is binary per sub-query (did it return >=1 chunk?), so "1 weak chunk" and "20 excellent chunks" both count as full coverage for that concept. A future version could weight coverage by average rerank score and/or chunks retained after pruning, not just presence/absence.
- Aggregate/cross-request retrieval coverage reporting (a dashboard or `/metrics`-style endpoint tracking coverage trends over time, low-coverage question logs, etc.) -- this plan only adds the per-request `concepts_requested`/`concepts_retrieved`/`coverage` fields to the dev trace (Task 2, Step 8), which is the raw data an aggregate view would be built on top of.
