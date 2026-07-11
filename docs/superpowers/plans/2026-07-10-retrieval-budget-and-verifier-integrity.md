# Retrieval Budget + Verifier Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut reranker cost by truncating fused candidates before reranking, remove all silent citation-evidence mutation from the verifier and pipeline, tighten the prompt to one claim per factual assertion, and surface both in the developer trace.

**Architecture:** Six independent, sequentially-applied changes to the existing hybrid-retrieval + citation-verification pipeline. No new services, no schema migrations — all changes are additive fields/params with safe defaults, or behavior changes gated by existing config.

**Tech Stack:** Python 3.11, Pydantic (Settings + models), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-10-retrieval-budget-and-verifier-integrity-design.md`
- `rerank_top_n <= rerank_fused_top_n <= dense_k + sparse_k` (validated in `Settings`)
- Default `rerank_fused_top_n = 8` — must not change existing default pipeline behavior for any deployment that doesn't set `RAG_FUSED_TOP_N`.
- No `.model_copy(update={"answer": ...})` or `claim.citation_ids = [...]` mutation of model-generated content anywhere in this change — verification only validates, never repairs.
- Every new/changed Pydantic field needs a safe default so existing call sites/tests that don't pass it keep working.

---

### Task 1: Rerank candidate budget (config + retriever truncation)

**Files:**
- Modify: `rag_hybrid_search/config.py`
- Modify: `rag_hybrid_search/retrieval/retriever.py`
- Modify: `api/dependencies.py:189-198`
- Modify: `scripts/benchmark.py:63-68`
- Modify: `scripts/debug_retrieval.py:132-138`
- Test: `tests/test_config.py`
- Test: `tests/retrieval/test_retriever.py`

**Interfaces:**
- Produces: `Settings.rerank_fused_top_n: int` (env `RAG_FUSED_TOP_N`, default 8); `HybridRetriever.__init__(..., rerank_fused_top_n: int)`; `HybridRetriever.rerank_fused_top_n` property. Later tasks (2) consume this same param.

- [ ] **Step 1: Write failing config tests**

Add to `tests/test_config.py`:

```python
def test_defaults():
    settings = Settings()
    assert settings.provider == "gemini"
    assert settings.chunking_strategy == "recursive"
    assert settings.rrf_dense_weight == 0.7
    assert settings.rrf_sparse_weight == 0.3
    assert settings.rerank_fused_top_n == 8


def test_rerank_fused_top_n_cannot_exceed_k_sum():
    with pytest.raises(ValidationError):
        Settings(dense_k=2, sparse_k=2, rerank_fused_top_n=10)


def test_rerank_top_n_cannot_exceed_fused_top_n():
    with pytest.raises(ValidationError):
        Settings(rerank_top_n=9, rerank_fused_top_n=5)
```

(Replace the existing `test_defaults` function in place — it's the same function with one more assertion appended. The other two are new functions added anywhere in the file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_config.py -v`
Expected: `test_defaults` FAILs with `AttributeError` or `AssertionError` on `rerank_fused_top_n`; the two new tests FAIL because no `ValidationError` is raised (field doesn't exist yet, so passing it is simply ignored/rejected by pydantic-settings differently — confirm they fail, not that they fail for the *right* reason yet).

- [ ] **Step 3: Add the setting and validation**

In `rag_hybrid_search/config.py`, add the field right after `rerank_top_n: int = 5` (line 38):

```python
    rerank_top_n: int = 5
    # Fused RRF output is truncated to this many top-scored candidates
    # before being sent to the reranker -- the reranker is the dominant
    # latency cost, and most fused candidates never survive rerank_top_n
    # anyway. dense_k/sparse_k stay wide for RRF diversity; only what
    # reaches the expensive reranker call is trimmed.
    rerank_fused_top_n: int = 8
```

Replace the validator body (lines 55-67) with:

```python
    @model_validator(mode="after")
    def _validate_weights_and_k(self) -> "Settings":
        if not (0.0 <= self.rrf_dense_weight <= 1.0):
            raise ValueError("rrf_dense_weight must be in [0, 1]")
        if not (0.0 <= self.rrf_sparse_weight <= 1.0):
            raise ValueError("rrf_sparse_weight must be in [0, 1]")
        if abs(self.rrf_dense_weight + self.rrf_sparse_weight - 1.0) > 1e-6:
            raise ValueError(
                "rrf_dense_weight + rrf_sparse_weight must sum to 1.0"
            )
        if self.rerank_top_n > self.dense_k + self.sparse_k:
            raise ValueError("rerank_top_n cannot exceed dense_k + sparse_k")
        if self.rerank_fused_top_n > self.dense_k + self.sparse_k:
            raise ValueError("rerank_fused_top_n cannot exceed dense_k + sparse_k")
        if self.rerank_top_n > self.rerank_fused_top_n:
            raise ValueError("rerank_top_n cannot exceed rerank_fused_top_n")
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Write failing retriever test for candidate truncation**

Add to `tests/retrieval/test_retriever.py`, in the `defaults` dict inside `build_retriever` (lines 87-94), add the new key so every existing call to `build_retriever` keeps working without truncation:

```python
    defaults = dict(
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60,
        dense_k=10,
        sparse_k=10,
        rerank_top_n=10,
        rerank_fused_top_n=20,
    )
```

In the `hybrid_retriever` fixture (lines 137-147), add the new kwarg so it also keeps working:

```python
    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        rerank_provider=reranker,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        dense_k=10,
        sparse_k=10,
        rerank_top_n=2,
        rerank_fused_top_n=10,
    )
```

Then add a new test at the end of the file:

```python
def test_rerank_fused_top_n_caps_candidates_sent_to_reranker(tmp_path):
    """rerank_fused_top_n must truncate the fused candidate list before it
    reaches the reranker, independent of rerank_top_n (which caps the
    reranker's own *output*, not its input)."""
    docs = [make_chunk(f"c{i}", f"document number {i} about search") for i in range(6)]
    reranker = RecordingRerankProvider()
    retriever = build_retriever(
        tmp_path, docs, reranker,
        dense_weight=0.5, sparse_weight=0.5,
        dense_k=6, sparse_k=6, rerank_top_n=1, rerank_fused_top_n=3,
    )

    results, _ = retriever.retrieve("document about search")

    assert reranker.received_candidates is not None
    assert len(reranker.received_candidates) == 3
    assert len(results) == 1
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/retrieval/test_retriever.py -v`
Expected: existing tests FAIL with `TypeError: __init__() missing 1 required positional argument: 'rerank_fused_top_n'` (or similar) since `HybridRetriever` doesn't accept the kwarg yet. The new test fails the same way.

- [ ] **Step 7: Implement truncation in `HybridRetriever`**

In `rag_hybrid_search/retrieval/retriever.py`, update `__init__` (lines 14-34):

```python
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rerank_provider: RerankProvider,
        dense_weight: float,
        sparse_weight: float,
        rrf_k: int,
        dense_k: int,
        sparse_k: int,
        rerank_top_n: int,
        rerank_fused_top_n: int,
    ):
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._rerank_provider = rerank_provider
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._rrf_k = rrf_k
        self._dense_k = dense_k
        self._sparse_k = sparse_k
        self._rerank_top_n = rerank_top_n
        self._rerank_fused_top_n = rerank_fused_top_n
```

Add a property near the other properties (after `rrf_k`, around line 62):

```python
    @property
    def rerank_fused_top_n(self) -> int:
        return self._rerank_fused_top_n
```

Update the tail of `retrieve()` (lines 110-122) to truncate before calling the reranker:

```python
        start = time.perf_counter()
        budgeted = fused[: self._rerank_fused_top_n]
        reranked = self._rerank_provider.rerank(query, budgeted, top_n=self._rerank_top_n)
        trace.rerank_latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "retrieve: rerank (top_n=%d, fused_budget=%d) via provider=%s returned %d results latency_ms=%.1f",
            self._rerank_top_n, self._rerank_fused_top_n, type(self._rerank_provider).__name__,
            len(reranked), trace.rerank_latency_ms,
        )
        logger.debug("retrieve: reranked results %s", [(r.chunk.chunk_id, r.rerank_score, r.final_rank) for r in reranked])
        logger.info("retrieve: done total_latency_ms=%.1f", trace.total_latency_ms)
        if dev_trace is not None:
            dev_trace.log_rerank(type(self._rerank_provider).__name__, budgeted, reranked, trace.rerank_latency_ms)

        return reranked, trace
```

(`dev_trace.log_rerank` gets a new required `budget_applied` argument in Task 2 — leave the call as 4 positional args here; Task 2 updates it to 5.)

- [ ] **Step 8: Update the three call sites that construct `HybridRetriever`**

In `api/dependencies.py`, in the `HybridRetriever(...)` call (around line 189), add:

```python
    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(embedding_provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25_index),
        rerank_provider=_select_rerank_provider(settings),
        dense_weight=settings.rrf_dense_weight,
        sparse_weight=settings.rrf_sparse_weight,
        rrf_k=settings.rrf_k,
        dense_k=settings.dense_k,
        sparse_k=settings.sparse_k,
        rerank_top_n=settings.rerank_top_n,
        rerank_fused_top_n=settings.rerank_fused_top_n,
    )
```

In `scripts/benchmark.py`, in the `HybridRetriever(...)` call (around line 63), add `rerank_fused_top_n=10` (matches `dense_k=5, sparse_k=5` sum, so no truncation vs. today's behavior):

```python
    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(embedding_provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25_index),
        rerank_provider=CrossEncoderReranker(),
        dense_weight=0.7, sparse_weight=0.3, rrf_k=60,
        dense_k=5, sparse_k=5, rerank_top_n=3, rerank_fused_top_n=10,
    )
```

In `scripts/debug_retrieval.py`, in the `HybridRetriever(...)` call (around line 132), add:

```python
        retriever = HybridRetriever(
            dense_retriever=dense_retriever, sparse_retriever=sparse_retriever,
            rerank_provider=PassthroughReranker(),
            dense_weight=settings.rrf_dense_weight, sparse_weight=settings.rrf_sparse_weight,
            rrf_k=settings.rrf_k, dense_k=settings.dense_k, sparse_k=settings.sparse_k,
            rerank_top_n=settings.rerank_top_n, rerank_fused_top_n=settings.rerank_fused_top_n,
        )
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/retrieval/test_retriever.py tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 10: Run the full test suite to check for regressions**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures (any existing failures unrelated to this change should already exist on `main`; if the suite was clean before, it must stay clean now).

- [ ] **Step 11: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_hybrid_search/config.py rag_hybrid_search/retrieval/retriever.py api/dependencies.py scripts/benchmark.py scripts/debug_retrieval.py tests/test_config.py tests/retrieval/test_retriever.py
git commit -m "$(cat <<'EOF'
feat: add rerank candidate budget to truncate fused results before reranking

RAG_FUSED_TOP_N (default 8) trims RRF fusion output before it reaches
the reranker, cutting reranker evaluations without narrowing dense_k/
sparse_k recall. rerank_top_n still caps the reranker's own output.
EOF
)"
```

---

### Task 2: Rerank candidate budget trace logging

**Files:**
- Modify: `rag_hybrid_search/models.py`
- Modify: `rag_hybrid_search/trace.py`
- Modify: `rag_hybrid_search/retrieval/retriever.py`
- Test: `tests/test_models.py`
- Create: `tests/test_trace.py`
- Test: `tests/retrieval/test_retriever.py`

**Interfaces:**
- Consumes: `HybridRetriever.rerank_fused_top_n` (Task 1).
- Produces: `RetrievalTrace.fusion_candidates/budget_applied/sent_to_reranker/returned: int`; `RequestTrace.log_rerank(self, provider_name, fused, reranked, latency_ms, budget_applied)` (signature change — new required 5th positional/keyword arg).

- [ ] **Step 1: Write failing model test**

Add to `tests/test_models.py`, right after `test_retrieval_trace_total_latency`:

```python
def test_retrieval_trace_budget_fields_default_to_zero():
    trace = RetrievalTrace()
    assert trace.fusion_candidates == 0
    assert trace.budget_applied == 0
    assert trace.sent_to_reranker == 0
    assert trace.returned == 0


def test_retrieval_trace_budget_fields_roundtrip():
    trace = RetrievalTrace(fusion_candidates=17, budget_applied=8, sent_to_reranker=8, returned=5)
    assert trace.fusion_candidates == 17
    assert trace.budget_applied == 8
    assert trace.sent_to_reranker == 8
    assert trace.returned == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_models.py -v`
Expected: FAIL with `AttributeError` (fields don't exist) or `ValidationError` (unexpected kwargs).

- [ ] **Step 3: Add fields to `RetrievalTrace`**

In `rag_hybrid_search/models.py`, replace the `RetrievalTrace` class (lines 63-76):

```python
class RetrievalTrace(BaseModel):
    dense_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    fusion_candidates: int = 0
    budget_applied: int = 0
    sent_to_reranker: int = 0
    returned: int = 0

    @property
    def total_latency_ms(self) -> float:
        return (
            self.dense_latency_ms
            + self.bm25_latency_ms
            + self.fusion_latency_ms
            + self.rerank_latency_ms
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Write failing trace-printing tests**

Create `tests/test_trace.py`:

```python
import pytest

from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_hybrid_search.trace import RequestTrace


def make_result(chunk_id, rrf_score=0.5, rerank_score=None, final_rank=1):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text="text",
        strategy_version="fixed-v1", heading=None, page=None, char_count=4,
    )
    return RetrievedChunk(
        chunk=chunk, dense_score=0.5, bm25_score=0.5, rrf_score=rrf_score,
        rerank_score=rerank_score, final_rank=final_rank,
    )


def test_log_rerank_records_budget_counts(monkeypatch):
    monkeypatch.delenv("TRACE_RAG", raising=False)
    trace = RequestTrace("question", {})
    trace._data["dense"] = [{}] * 5
    trace._data["bm25"] = [{}] * 5
    trace._data["fusion"] = [{}] * 8

    budgeted = [make_result(f"c{i}") for i in range(4)]
    reranked = [budgeted[0].model_copy(update={"final_rank": 1, "rerank_score": 0.9})]

    trace.log_rerank("nvidia", budgeted, reranked, latency_ms=12.0, budget_applied=4)

    rerank_data = trace._data["rerank"]
    assert rerank_data["fusion_candidates"] == 8
    assert rerank_data["sent_to_reranker"] == 4
    assert rerank_data["returned"] == 1
    assert rerank_data["budget_applied"] == 4


def test_log_rerank_prints_retrieval_budget_block_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("TRACE_RAG", "true")
    trace = RequestTrace("question", {})
    trace._data["dense"] = [{}] * 10
    trace._data["bm25"] = [{}] * 10
    trace._data["fusion"] = [{}] * 17

    budgeted = [make_result(f"c{i}") for i in range(8)]
    reranked = [budgeted[i].model_copy(update={"final_rank": i + 1, "rerank_score": 0.9}) for i in range(5)]

    trace.log_rerank("nvidia", budgeted, reranked, latency_ms=12.0, budget_applied=8)

    out = capsys.readouterr().out
    assert "RETRIEVAL BUDGET" in out
    assert "Dense candidates" in out
    assert "Saved" in out
    assert "9 reranker evaluations" in out
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_trace.py -v`
Expected: FAIL with `TypeError: log_rerank() got an unexpected keyword argument 'budget_applied'`.

- [ ] **Step 7: Implement the budget block in `log_rerank`**

In `rag_hybrid_search/trace.py`, replace `log_rerank` (lines 161-178):

```python
    def log_rerank(self, provider_name: str, fused: list, reranked: list, latency_ms: float, budget_applied: int) -> None:
        selected_ids = {r.chunk.chunk_id for r in reranked}
        dropped = [c.chunk.chunk_id for c in fused if c.chunk.chunk_id not in selected_ids]
        fusion_candidates = len(self._data.get("fusion", []))
        self._data["rerank"] = {
            "provider": provider_name,
            "selected": [{"chunk_id": r.chunk.chunk_id, "score": r.rerank_score, "final_rank": r.final_rank} for r in reranked],
            "dropped": dropped,
            "budget_applied": budget_applied,
            "fusion_candidates": fusion_candidates,
            "sent_to_reranker": len(fused),
            "returned": len(reranked),
        }
        self.mark("rerank", latency_ms)
        if not self.enabled:
            return
        _section("STEP 5 -- CROSS-ENCODER / RERANKER")
        _kv(Provider=provider_name, Candidates=len(fused), Selected=len(reranked), Latency=f"{latency_ms:.1f} ms")
        for r in reranked:
            score = "n/a" if r.rerank_score is None else f"{r.rerank_score:.4f}"
            print(f"  SELECTED  chunk={r.chunk.chunk_id[:12]}  rank={r.final_rank}  score={score}")
        for cid in dropped:
            print(f"  DROPPED   chunk={cid[:12]}  reason=not in reranker top-{len(reranked)}")

        dense_count = len(self._data.get("dense", []))
        bm25_count = len(self._data.get("bm25", []))
        saved = fusion_candidates - len(fused)
        _section("RETRIEVAL BUDGET")
        _kv(**{
            "Dense candidates": dense_count,
            "BM25 candidates": bm25_count,
            "Unique fused candidates": fusion_candidates,
            "Configured budget": budget_applied,
            "Sent to reranker": len(fused),
            "Returned": len(reranked),
            "Saved": f"{saved} reranker evaluations",
        })
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_trace.py -v`
Expected: PASS.

- [ ] **Step 9: Wire the retriever to populate `RetrievalTrace` fields and pass `budget_applied`**

In `rag_hybrid_search/retrieval/retriever.py`, update the fusion block (around line 85-98) to record `trace.fusion_candidates` and `trace.budget_applied` right after computing `fused`:

```python
        start = time.perf_counter()
        fused = weighted_rrf(
            dense_results,
            sparse_results,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
            k=self._rrf_k,
        )
        trace.fusion_latency_ms = (time.perf_counter() - start) * 1000
        trace.fusion_candidates = len(fused)
        trace.budget_applied = self._rerank_fused_top_n
        logger.info(
            "retrieve: fusion (rrf_k=%d, dense_weight=%.2f, sparse_weight=%.2f) produced %d candidates latency_ms=%.1f",
            self._rrf_k, self._dense_weight, self._sparse_weight, len(fused), trace.fusion_latency_ms,
        )
        logger.debug("retrieve: fused candidates %s", [(r.chunk.chunk_id, r.rrf_score) for r in fused])
        if dev_trace is not None:
            dev_trace.log_fusion(
                fused,
                dense_ids_ranked=[r.chunk.chunk_id for r in dense_results],
                bm25_ids_ranked=[r.chunk.chunk_id for r in sparse_results],
                rrf_k=self._rrf_k,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
                latency_ms=trace.fusion_latency_ms,
            )
```

Then update the rerank block (the one from Task 1 Step 7) to record `trace.sent_to_reranker`/`trace.returned` and pass `budget_applied` to `log_rerank`:

```python
        start = time.perf_counter()
        budgeted = fused[: self._rerank_fused_top_n]
        reranked = self._rerank_provider.rerank(query, budgeted, top_n=self._rerank_top_n)
        trace.rerank_latency_ms = (time.perf_counter() - start) * 1000
        trace.sent_to_reranker = len(budgeted)
        trace.returned = len(reranked)
        logger.info(
            "retrieve: rerank (top_n=%d, fused_budget=%d) via provider=%s returned %d results latency_ms=%.1f",
            self._rerank_top_n, self._rerank_fused_top_n, type(self._rerank_provider).__name__,
            len(reranked), trace.rerank_latency_ms,
        )
        logger.debug("retrieve: reranked results %s", [(r.chunk.chunk_id, r.rerank_score, r.final_rank) for r in reranked])
        logger.info("retrieve: done total_latency_ms=%.1f", trace.total_latency_ms)
        if dev_trace is not None:
            dev_trace.log_rerank(
                type(self._rerank_provider).__name__, budgeted, reranked,
                trace.rerank_latency_ms, self._rerank_fused_top_n,
            )

        return reranked, trace
```

- [ ] **Step 10: Write failing retriever test for `RetrievalTrace` budget fields**

Add to `tests/retrieval/test_retriever.py`:

```python
def test_retrieve_records_budget_trace_fields(hybrid_retriever):
    results, trace = hybrid_retriever.retrieve("ERROR_CODE_0x834")

    assert trace.fusion_candidates == 3
    assert trace.budget_applied == 10
    assert trace.sent_to_reranker == 3
    assert trace.returned == len(results)


def test_retrieve_records_smaller_budget_than_fusion_candidates(tmp_path):
    docs = [make_chunk(f"c{i}", f"document number {i} about search") for i in range(6)]
    reranker = RecordingRerankProvider()
    retriever = build_retriever(
        tmp_path, docs, reranker,
        dense_weight=0.5, sparse_weight=0.5,
        dense_k=6, sparse_k=6, rerank_top_n=1, rerank_fused_top_n=3,
    )

    _, trace = retriever.retrieve("document about search")

    assert trace.fusion_candidates == 6
    assert trace.budget_applied == 3
    assert trace.sent_to_reranker == 3
    assert trace.returned == 1
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/retrieval/test_retriever.py tests/test_trace.py tests/test_models.py -v`
Expected: all PASS.

- [ ] **Step 12: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures.

- [ ] **Step 13: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_hybrid_search/models.py rag_hybrid_search/trace.py rag_hybrid_search/retrieval/retriever.py tests/test_models.py tests/test_trace.py tests/retrieval/test_retriever.py
git commit -m "$(cat <<'EOF'
feat: log rerank candidate budget counts in RetrievalTrace and dev trace

RetrievalTrace gains fusion_candidates/budget_applied/sent_to_reranker/
returned counts; TRACE_RAG output prints a RETRIEVAL BUDGET block
showing exactly how many reranker evaluations the budget saved.
EOF
)"
```

---

### Task 3: Verifier — no silent citation mutation

**Files:**
- Modify: `rag_pipeline/citation_verifier.py`
- Test: `tests/rag_pipeline/test_citation_verifier.py`

**Interfaces:**
- Produces: `ClaimResult.failure_reason` gains a new possible value `"citation_reattribution_candidate"`. `Claim.citation_ids` is never mutated by `verify_citations` (previously mutated at lines 66-67 of the old file).

- [ ] **Step 1: Write the failing test (replace the old reattribution test)**

In `tests/rag_pipeline/test_citation_verifier.py`, replace `test_misattributed_citation_id_is_corrected_when_quote_matches_another_doc` (lines 75-96) with:

```python
def test_misattributed_citation_id_is_flagged_not_silently_corrected():
    """The model sometimes copies a verbatim quote correctly but tags it
    with the wrong doc id. The verifier detects a better-matching doc but
    must never rewrite citation_ids -- it validates model output, it
    doesn't repair it. The claim fails with a distinct reason so the
    caller can decide what to do, and citation_ids stay exactly what the
    model wrote."""
    context = PromptContext(
        text=(
            "[d1]\nThe cafeteria serves lunch from 12pm to 2pm daily.\n\n"
            "[d2]\nEmployees get 20 days of paid annual leave per year."
        ),
        doc_id_map={"d1": "chunk-1", "d2": "chunk-2"},
    )
    claim = Claim(
        text="Employees get 20 days leave",
        citation_ids=["d1"],
        supporting_quote="20 days of paid annual leave",
    )
    report = verify_citations(make_draft([claim]), context)
    assert report.verified_claims == 0
    assert report.failed_claims == 1
    assert report.claim_results[0].passed is False
    assert report.claim_results[0].failure_reason == "citation_reattribution_candidate"
    assert report.claim_results[0].claim.citation_ids == ["d1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_citation_verifier.py::test_misattributed_citation_id_is_flagged_not_silently_corrected -v`
Expected: FAIL — current code sets `passed is True` and rewrites `citation_ids` to `["d2"]`.

- [ ] **Step 3: Remove the mutation, add the new failure reason**

In `rag_pipeline/citation_verifier.py`, replace lines 45-104 (from `best_quote_score = 0.0` through the end of the `claim_results.append(...)` call inside the loop) with:

```python
        best_quote_score = 0.0
        if doc_ids_valid:
            for citation_id in claim.citation_ids:
                chunk_text = chunk_text_for_doc_id(context, citation_id)
                score = _quote_containment_score(claim.supporting_quote, chunk_text)
                best_quote_score = max(best_quote_score, score)

        # The model occasionally copies a supporting_quote verbatim but
        # tags it with the wrong doc id. A better-matching doc might exist
        # in context -- but the verifier never repairs evidence, only
        # validates it. Detecting this case and still failing the claim
        # (with a distinct, actionable reason) is intentional: it favors
        # evidence integrity over answer recovery. Rewriting citation_ids
        # here would mean the caller sees a citation the model never
        # actually wrote.
        reattribution_doc_id = None
        if doc_ids_valid and best_quote_score < QUOTE_MATCH_THRESHOLD:
            best_doc_id, best_doc_score = None, best_quote_score
            for doc_id in context.doc_id_map:
                if doc_id in claim.citation_ids:
                    continue
                chunk_text = chunk_text_for_doc_id(context, doc_id)
                score = _quote_containment_score(claim.supporting_quote, chunk_text)
                if score > best_doc_score:
                    best_doc_id, best_doc_score = doc_id, score
            if best_doc_id is not None and best_doc_score >= QUOTE_MATCH_THRESHOLD:
                reattribution_doc_id = best_doc_id

        passed = (
            doc_ids_valid
            and best_quote_score >= QUOTE_MATCH_THRESHOLD
            and reattribution_doc_id is None
        )

        failure_reason = None
        if not doc_ids_valid:
            failure_reason = "hallucinated_citation_id"
        elif reattribution_doc_id is not None:
            missing_quotes.append(claim.supporting_quote)
            failure_reason = "citation_reattribution_candidate"
        elif not passed:
            missing_quotes.append(claim.supporting_quote)
            # A quote that scores low against its own cited chunk but scores
            # high against all cited-context chunks concatenated (markers
            # stripped, chunks joined with a plain space) is the signature
            # of a quote that spans multiple chunks: it exists in the
            # document as a whole, just split across a chunk boundary.
            # Using context.text directly would miss this -- the literal
            # "[dN]" marker between chunks breaks contiguous matching right
            # at the boundary, making a genuinely cross-chunk quote score
            # just as low against the whole raw context as a true
            # hallucination would.
            all_chunks_concat = " ".join(
                chunk_text_for_doc_id(context, doc_id) for doc_id in context.doc_id_map
            )
            whole_context_score = _quote_containment_score(claim.supporting_quote, all_chunks_concat)
            failure_reason = (
                "quote_spans_multiple_chunks"
                if whole_context_score >= QUOTE_MATCH_THRESHOLD
                else "quote_not_found"
            )

        claim_results.append(
            ClaimResult(
                claim=claim,
                doc_ids_valid=doc_ids_valid,
                quote_match_score=best_quote_score,
                passed=passed,
                failure_reason=failure_reason,
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_citation_verifier.py -v`
Expected: all PASS, including all pre-existing tests in this file (they don't touch the reattribution path, so their behavior is unchanged).

- [ ] **Step 5: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures. (`test_multi_citation_claim_from_model_is_narrowed_and_verified_safely` in `tests/rag_pipeline/test_rag_pipeline.py` should still pass — its quote is backend-extracted directly from the cited chunk, so it never enters the reattribution branch.)

- [ ] **Step 6: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/citation_verifier.py tests/rag_pipeline/test_citation_verifier.py
git commit -m "$(cat <<'EOF'
fix: verifier no longer silently rewrites citation_ids on reattribution

When a claim's quote matches a different doc than the one it cites, the
verifier now fails the claim with failure_reason=citation_reattribution_
candidate instead of rewriting citation_ids and passing it. The
verifier validates model output; it must never repair it.
EOF
)"
```

---

### Task 4: Kill silent inline-citation rewrite; add `CitationStatus`

**Files:**
- Modify: `rag_pipeline/models.py`
- Modify: `rag_pipeline/rag_pipeline.py`
- Modify: `rag_hybrid_search/trace.py`
- Test: `tests/rag_pipeline/test_models.py`
- Test: `tests/rag_pipeline/test_rag_pipeline.py`

**Interfaces:**
- Consumes: `VerificationReport.claim_results[].passed` (existing).
- Produces: `CitationStatus` enum (`OK`, `INLINE_DRIFT`, `VERIFICATION_FAILED`) in `rag_pipeline.models`; `RagAnswer.citation_status: CitationStatus = CitationStatus.OK`. `RequestTrace.log_citation_check(self, inline_ids, structured_ids, status: str)` (signature change — third param is now a status string, not a bool).

- [ ] **Step 1: Write failing model test**

Add to `tests/rag_pipeline/test_models.py`:

```python
from rag_pipeline.models import CitationStatus  # add to the existing import block


def test_citation_status_values():
    assert CitationStatus.OK == "ok"
    assert CitationStatus.INLINE_DRIFT == "inline_drift"
    assert CitationStatus.VERIFICATION_FAILED == "verification_failed"


def test_rag_answer_defaults_to_ok_citation_status():
    report = VerificationReport(
        total_claims=0, verified_claims=0, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
    )
    scores = ConfidenceScores(retrieval=0.0, citations=0.0, coverage=0.0, overall=0.0)
    answer = RagAnswer(answer=None, citations=[], confidence=scores, verification=report)
    assert answer.citation_status == CitationStatus.OK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'CitationStatus'`.

- [ ] **Step 3: Add the enum and field**

In `rag_pipeline/models.py`, add the `Enum` import (line 1-4 area):

```python
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from rag_hybrid_search.compliance.regulation_models import Citation
```

Add the enum right before `class RagAnswer(BaseModel):` (currently line 52):

```python
class CitationStatus(str, Enum):
    OK = "ok"
    INLINE_DRIFT = "inline_drift"
    VERIFICATION_FAILED = "verification_failed"


class RagAnswer(BaseModel):
    answer: Optional[str]
    citations: list[str]
    structured_citations: list[Citation] = []
    confidence: ConfidenceScores
    verification: VerificationReport
    citation_status: CitationStatus = CitationStatus.OK
    error: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Write failing pipeline tests**

Add to `tests/rag_pipeline/test_rag_pipeline.py`. First, add the import:

```python
from rag_pipeline.models import CitationStatus
```

Then add two new tests at the end of the file:

```python
def test_inline_citation_drift_is_flagged_not_rewritten():
    """When inline [dN] markers in the answer prose disagree with the
    structured claims' citation_ids, the pipeline must not rewrite the
    answer text -- it only flags the drift via citation_status."""
    chunks = [
        make_retrieved_chunk("c1", "The cafeteria serves lunch from 12pm to 2pm."),
        make_retrieved_chunk("c2", "Employees get 20 days of paid annual leave.", final_rank=2),
    ]
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{
            "text": "Employees get 20 days of paid leave.",
            "citation_ids": ["d2"],
            "supporting_quote": "20 days of paid annual leave",
        }],
    })
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("How many days of paid leave?")

    assert result.error is None
    assert result.answer == "Employees get 20 days of paid leave [d1]."
    assert result.citation_status == CitationStatus.INLINE_DRIFT


def test_verification_failure_takes_precedence_over_inline_drift():
    chunks = [make_retrieved_chunk("c1", "Employees get 20 days of paid annual leave.")]
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{
            "text": "Employees get 20 days of paid leave.",
            "citation_ids": ["d99"],
        }],
    })
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("question")

    assert result.verification.failed_claims == 1
    assert result.citation_status == CitationStatus.VERIFICATION_FAILED
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py -v`
Expected: `test_inline_citation_drift_is_flagged_not_rewritten` FAILs — current code rewrites the answer to `"Employees get 20 days of paid leave [d2]."` and has no `citation_status` attribute. `test_verification_failure_takes_precedence_over_inline_drift` FAILs the same way (no `citation_status`).

- [ ] **Step 7: Replace `_reconcile_inline_citations` and its call sites**

In `rag_pipeline/rag_pipeline.py`, replace `_reconcile_inline_citations` (lines 107-129) with:

```python
def _inline_citation_drift(answer: str, structured_ids: set[str]) -> tuple[list[str], bool]:
    """Detect drift between inline ``[dN]`` refs in the free-text answer and
    the citation_ids the model put in its own structured claims.

    The model writes citations twice: inline in ``answer`` prose and again
    in ``claims[].citation_ids``. Nothing enforces they agree. This only
    detects and reports drift -- it never rewrites the answer text. The
    pipeline validates what the model produced; it does not repair it.
    """
    inline_ids = _INLINE_CITATION_RE.findall(answer)
    inline_set = set(inline_ids)
    ok = inline_set <= structured_ids
    return sorted(inline_set, key=lambda x: (len(x), x)), ok
```

Add the `CitationStatus` import to the existing `from rag_pipeline.models import (...)` block (lines 18-25):

```python
from rag_pipeline.models import (
    Claim,
    CitationStatus,
    ConfidenceScores,
    GenerationMetadata,
    RagAnswer,
    RagAnswerDraft,
    VerificationReport,
)
```

In `answer()` (around lines 240-254), replace:

```python
        citations = sorted({cid for c in draft.claims for cid in c.citation_ids})
        fixed_answer, inline_ids, citations_ok = _reconcile_inline_citations(draft.answer, set(citations))
        dev_trace.log_citation_check(inline_ids, citations, citations_ok)
        if not citations_ok:
            draft = draft.model_copy(update={"answer": fixed_answer})

        structured_citations = build_citations(retrieved_chunks, self._filename_by_doc_id())
        documents_used = len({r.chunk.document_id for r in retrieved_chunks})
        dev_trace.log_summary(draft.answer, chunks_used=len(retrieved_chunks), documents_used=documents_used)
        dev_trace.finish()

        return RagAnswer(
            answer=draft.answer, citations=citations, structured_citations=structured_citations,
            confidence=confidence, verification=verification, error=parse_error,
        )
```

with:

```python
        citations = sorted({cid for c in draft.claims for cid in c.citation_ids})
        inline_ids, citations_ok = _inline_citation_drift(draft.answer, set(citations))

        citation_status = CitationStatus.OK
        if any(not cr.passed for cr in verification.claim_results):
            citation_status = CitationStatus.VERIFICATION_FAILED
        elif not citations_ok:
            citation_status = CitationStatus.INLINE_DRIFT
        dev_trace.log_citation_check(inline_ids, citations, citation_status.value)

        structured_citations = build_citations(retrieved_chunks, self._filename_by_doc_id())
        documents_used = len({r.chunk.document_id for r in retrieved_chunks})
        dev_trace.log_summary(draft.answer, chunks_used=len(retrieved_chunks), documents_used=documents_used)
        dev_trace.finish()

        return RagAnswer(
            answer=draft.answer, citations=citations, structured_citations=structured_citations,
            confidence=confidence, verification=verification, citation_status=citation_status,
            error=parse_error,
        )
```

Apply the identical change to the matching block in `answer_stream()` (currently lines 334-350), including the corresponding `yield ("final", RagAnswer(...))` call gaining `citation_status=citation_status`.

- [ ] **Step 8: Update `log_citation_check` in the trace module**

In `rag_hybrid_search/trace.py`, replace `log_citation_check` (lines 275-287):

```python
    def log_citation_check(self, inline_ids: list[str], structured_ids: list[str], status: str) -> None:
        self._data["citation_check"] = {
            "inline": inline_ids, "structured": structured_ids, "status": status,
        }
        if not self.enabled:
            return
        _section("CITATION STATUS")
        _kv(Status=status, Inline=inline_ids or "[]", Structured=structured_ids or "[]")
        if status != "ok":
            print("Action            : no mutation performed")
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py tests/rag_pipeline/test_models.py -v`
Expected: all PASS.

- [ ] **Step 10: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures.

- [ ] **Step 11: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/models.py rag_pipeline/rag_pipeline.py rag_hybrid_search/trace.py tests/rag_pipeline/test_models.py tests/rag_pipeline/test_rag_pipeline.py
git commit -m "$(cat <<'EOF'
fix: stop rewriting answer text on inline/structured citation drift

_reconcile_inline_citations no longer mutates the model's answer text.
RagAnswer gains citation_status (OK/INLINE_DRIFT/VERIFICATION_FAILED)
so callers can see evidence problems instead of having them silently
papered over. Verification failures take precedence over inline drift.
EOF
)"
```

---

### Task 5: One claim per factual assertion (prompt v2)

**Files:**
- Modify: `rag_pipeline/prompt_builder.py`
- Test: `tests/rag_pipeline/test_prompt_builder.py`

**Interfaces:** None (prompt text only, no code interface changes).

- [ ] **Step 1: Write failing test**

Add to `tests/rag_pipeline/test_prompt_builder.py`:

```python
def test_prompt_v2_instructs_one_claim_per_assertion():
    context = PromptContext(text="[d1]\nsome fact", doc_id_map={"d1": "c1"})
    prompt = build_prompt("What is the fact?", context, prompt_version="v2")
    assert "MUST produce exactly one claim object" in prompt
    assert "even when they cite the same source" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_prompt_builder.py::test_prompt_v2_instructs_one_claim_per_assertion -v`
Expected: FAIL — text not present yet.

- [ ] **Step 3: Add the rule and worked example to `_PROMPT_V2`**

In `rag_pipeline/prompt_builder.py`, insert a new rule bullet into `_PROMPT_V2` right after the existing "Never combine or concatenate..." bullet (after line 103, before the "Do NOT include a supporting_quote field" bullet on line 104):

```python
- Never combine or concatenate wording from two different [dN] blocks
  into a single claim. Each claim's "text" must be fully supported by
  its ONE cited block alone. If a full statement genuinely requires two
  sources, express it as two separate claims, each citing its own source.
- Every independently verifiable factual assertion MUST produce exactly
  one claim object. If a single sentence contains multiple factual
  assertions (e.g. "A because B."), split them into separate claims,
  each with its own citation_ids, even when they cite the same source.
- Do NOT include a supporting_quote field. The backend extracts the
  supporting quote itself from the cited block -- you only provide the
  claim text and its single citation id.
```

Then insert a second worked example into `_PROMPT_V2`, right after the existing "WRONG (claim combines wording from two different sources...)" block and before the "WRONG (refusing to answer...)" block (i.e. between the current lines 128 and 130):

```python
WRONG (claim combines wording from two different sources into one citation):
{{"answer": "Personal information may only be retained as long as necessary, and can be erased on request [d1].", "claims": [{{"text": "Personal information may only be retained as long as necessary and can be erased on request.", "citation_ids": ["d1"]}}]}}

Example -- one claim per factual assertion, even from the same source:
<context>
[d1]
Granularity dominates detection accuracy because architecture-specific
features overfit to training distribution shifts.
</context>

<question>
Why does granularity dominate detection accuracy?
</question>

CORRECT (two claims, one per assertion):
{{"answer": "Granularity dominates detection accuracy [d1] because architecture-specific features overfit to training distribution shifts [d1].", "claims": [{{"text": "Granularity dominates detection accuracy.", "citation_ids": ["d1"]}}, {{"text": "Architecture-specific features overfit to training distribution shifts.", "citation_ids": ["d1"]}}]}}

WRONG (one claim collapsing two assertions):
{{"answer": "Granularity dominates detection accuracy because architecture-specific features overfit to training distribution shifts [d1].", "claims": [{{"text": "Granularity dominates detection accuracy because architecture-specific features overfit to training distribution shifts.", "citation_ids": ["d1"]}}]}}

WRONG (refusing to answer even though the passage supports it):
{{"answer": "I don't know.", "claims": []}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_prompt_builder.py -v`
Expected: all PASS, including the pre-existing `test_prompt_v2_instructs_one_citation_per_claim` and `test_prompt_v2_schema_omits_supporting_quote` (unaffected — this task only adds text, doesn't remove any).

- [ ] **Step 5: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/prompt_builder.py tests/rag_pipeline/test_prompt_builder.py
git commit -m "$(cat <<'EOF'
feat: instruct prompt v2 to split compound answers into one claim per assertion

Adds an explicit rule and worked example so the model splits sentences
like "A because B." into separate claim objects (even when both cite
the same source), improving verification granularity. Prompt-only;
not enforced in code -- a model that ignores it still yields a valid,
if coarser, verification result.
EOF
)"
```

---

### Task 6: Verification summary in trace

**Files:**
- Modify: `rag_hybrid_search/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:** None (internal trace formatting only — `log_verification`'s signature is unchanged, it still takes a single `verification` object).

**Note:** the spec's mocked "Citation Status" line inside the verification summary block is not literally reproduced here — `citation_status` isn't computed until after `log_verification` runs (it depends on the later inline-drift check from Task 4). Both pieces of information still land in the same per-request trace output, just in two adjacent sections (`STEP 9 -- CITATION VERIFICATION` / `VERIFICATION SUMMARY`, then later `CITATION STATUS`), in pipeline order.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_trace.py`:

```python
def test_log_verification_computes_ratio_and_prints_summary(monkeypatch, capsys):
    monkeypatch.setenv("TRACE_RAG", "true")
    trace = RequestTrace("question", {})

    class FakeClaim:
        text = "claim text"
        citation_ids = ["d1"]

    class FakeClaimResult:
        def __init__(self, passed):
            self.claim = FakeClaim()
            self.doc_ids_valid = True
            self.quote_match_score = 1.0
            self.passed = passed
            self.failure_reason = None if passed else "quote_not_found"

    class FakeVerification:
        total_claims = 4
        verified_claims = 3
        failed_claims = 1
        claim_results = [FakeClaimResult(True), FakeClaimResult(True), FakeClaimResult(True), FakeClaimResult(False)]

    trace.log_verification(FakeVerification())

    assert trace._data["verification"]["verification_ratio"] == pytest.approx(0.75)
    out = capsys.readouterr().out
    assert "VERIFICATION SUMMARY" in out
    assert "75%" in out


def test_log_verification_ratio_zero_when_no_claims():
    trace = RequestTrace("question", {})

    class FakeVerification:
        total_claims = 0
        verified_claims = 0
        failed_claims = 0
        claim_results = []

    trace.log_verification(FakeVerification())
    assert trace._data["verification"]["verification_ratio"] == 0.0
```

(`pytest` is already imported at the top of `tests/test_trace.py` from Task 2.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_trace.py -v`
Expected: `test_log_verification_computes_ratio_and_prints_summary` FAILs with `KeyError: 'verification_ratio'`.

- [ ] **Step 3: Add the ratio and summary block**

In `rag_hybrid_search/trace.py`, replace `log_verification` (lines 225-243):

```python
    def log_verification(self, verification) -> None:
        rows = [
            {"text": cr.claim.text, "citation_ids": cr.claim.citation_ids, "doc_ids_valid": cr.doc_ids_valid,
             "quote_match_score": cr.quote_match_score, "passed": cr.passed, "failure_reason": cr.failure_reason}
            for cr in verification.claim_results
        ]
        ratio = (
            verification.verified_claims / verification.total_claims
            if verification.total_claims else 0.0
        )
        self._data["verification"] = {
            "total": verification.total_claims, "verified": verification.verified_claims,
            "failed": verification.failed_claims, "verification_ratio": ratio, "claims": rows,
        }
        if not self.enabled:
            return
        _section("STEP 9 -- CITATION VERIFICATION")
        _kv(**{"Total Claims": verification.total_claims, "Verified": verification.verified_claims, "Failed": verification.failed_claims})
        for i, row in enumerate(rows, 1):
            status = "PASS" if row["passed"] else "FAIL"
            reason = f"  reason={row['failure_reason']}" if row["failure_reason"] else ""
            print(f"\n  Claim {i} [{status}]  citations={row['citation_ids']}  quote_match={row['quote_match_score']:.3f}{reason}")
            print(f"    {row['text'][:160]!r}")
        _section("VERIFICATION SUMMARY")
        _kv(**{
            "Claims generated": verification.total_claims,
            "Claims verified": verification.verified_claims,
            "Claims failed": verification.failed_claims,
            "Verification Ratio": f"{ratio * 100:.0f}%",
        })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_trace.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_hybrid_search/trace.py tests/test_trace.py
git commit -m "$(cat <<'EOF'
feat: add verification ratio and summary block to dev trace

TRACE_RAG output now prints claims generated/verified/failed and an
objective verification_ratio, instead of only a per-claim pass/fail
list -- gives a single-screen number for how much of an answer was
actually checked.
EOF
)"
```

---

## Deferred (not in this plan)

- Comparative-query retrieval (RQ1/RQ3 gap) — separate design (query decomposition / multi-query retrieval).
- Wiring `verification_ratio` into `score_confidence`.
- Confidence scoring redesign more broadly.
- Automatic regeneration/retry after a `VERIFICATION_FAILED` result.
