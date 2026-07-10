# Retrieval Budget + Verifier Integrity — Design

## Context

RAG pipeline currently: dense_k=10 + sparse_k=10 → RRF fusion (~13-20 unique
candidates) → all candidates sent to NVIDIA reranker → rerank_top_n=5 →
dynamic context pruning → generation. Reranker latency is the dominant cost;
it evaluates every fused candidate even though only 5 survive.

Separately, the citation/verification pipeline has two places that silently
mutate model output instead of failing explicitly:

1. `citation_verifier.py` re-attributes a claim's `citation_ids` to a
   different doc when the quote doesn't match the cited doc but matches
   another one — and marks the claim `passed=True` under the new id.
2. `rag_pipeline.py::_reconcile_inline_citations` rewrites the answer's
   inline `[dN]` markers to match structured `citation_ids` when they drift,
   silently editing the returned prose.

Both hide evidence problems from the caller. Verification also currently
lets multiple factual assertions hide inside one claim object, and the
confidence report gives no objective verified/failed ratio.

Out of scope for this change: fixing retrieval for comparative questions
(e.g. "relationship between RQ1 and RQ3") — that requires query
decomposition / multi-query retrieval and is its own design.

## Changes

### 1. Rerank candidate budget (fused-candidate truncation)

Named "rerank candidate budget," not "retrieval budget" — the budget is
applied after retrieval (dense + BM25 + fusion), only trimming what
reaches the reranker.

`rag_hybrid_search/config.py`: new setting `rerank_fused_top_n: int = 8`
(env `RAG_FUSED_TOP_N`). Validation extended:
`rerank_top_n <= rerank_fused_top_n <= dense_k + sparse_k`.

`rag_hybrid_search/retrieval/retriever.py`: in `HybridRetriever.retrieve`,
after `weighted_rrf` fusion and before calling `rerank_provider.rerank(...)`,
slice `fused` (already sorted by `rrf_score`) to the top
`self._rerank_fused_top_n`. `HybridRetriever.__init__` takes the new param.

### 2. Retrieval budget trace logging

`RetrievalTrace` (models.py) gains fields: `fusion_candidates: int`,
`budget_applied: int`, `sent_to_reranker: int`, `returned: int`.
`dev_trace.log_fusion(...)` call site updated to pass fusion_candidates
(pre-truncation count) and budget_applied (the configured `fused_top_n`).
`dev_trace.log_rerank(...)` already has `sent_to_reranker` (len of
truncated fused list) and `returned` (len of reranked output) available
from existing args — surface them in the printed trace block:

```
──────── RETRIEVAL BUDGET ────────
Dense candidates          : 10
BM25 candidates           : 10
Unique fused candidates   : 17
Configured budget         : 8
Sent to reranker          : 8
Returned                  : 5
Saved                     : 9 reranker evaluations
```

`Saved = fusion_candidates - sent_to_reranker`.

**Expected impact:** fewer reranker evaluations, smaller prompt, lower
generation latency, identical retrieval recall (dense_k/sparse_k
unchanged), no change to the ranking algorithm itself.

**Backward compatibility:** existing deployments unaffected unless
`RAG_FUSED_TOP_N` is explicitly set. Default (8) preserves current
rerank_top_n=5 behavior with no observable change beyond the latency win.

**Risks:** a very small `fused_top_n` can cut recall (dropping a
candidate the reranker would have promoted). Recommended default:
`dense_k=10, sparse_k=10, fused_top_n=8, rerank_top_n=5`. Don't go below
`rerank_top_n + ~2` without re-checking recall on eval questions.

### 3. Verifier: no silent citation mutation

`citation_verifier.py`, reattribution block (currently lines 56-67):
stop mutating `claim.citation_ids`. When a better-matching doc is found,
do not reassign — instead fall through to the existing failure path with
a new `failure_reason="citation_reattribution_candidate"`. `passed=False`
unconditionally in this branch. No `ClaimResult`/`Claim` schema change
(reuses existing `failure_reason: str | None` field).

### 4. No silent inline-citation rewrite; `citation_status` enum

`rag_pipeline/models.py`: new enum

```python
class CitationStatus(str, Enum):
    OK = "ok"
    INLINE_DRIFT = "inline_drift"
    VERIFICATION_FAILED = "verification_failed"
```

`RagAnswer` gains `citation_status: CitationStatus` (new field — there was
no prior bool). Derivation in `rag_pipeline.py` after verification and
citation-check, precedence:

1. Any `claim_results[].passed == False` → `VERIFICATION_FAILED`
   (detail already in `verification.claim_results[].failure_reason`,
   which now includes `citation_reattribution_candidate` from change #3).
2. Else inline `[dN]` set != structured `citation_ids` set (drift
   detected by `_reconcile_inline_citations`) → `INLINE_DRIFT`.
3. Else `OK`.

`_reconcile_inline_citations` no longer rewrites `draft.answer`. It still
computes and returns the inline/structured id sets for drift detection and
trace logging, but the pipeline no longer calls `draft.model_copy(update=
{"answer": fixed_answer})`. `RagAnswer.answer` is always exactly what the
model generated.

`dev_trace.log_citation_check(...)` output extended:

```
Citation Status
Status : INLINE_DRIFT
Inline:     [d3]
Structured: [d1]
Action: no mutation performed
```

`is_verified = citation_status == CitationStatus.OK` is a derived
convenience, not a stored field (callers/UI compute it inline).

### 5. One claim per factual assertion (prompt)

`prompt_builder.py`, prompt v2 instructions: replace the vague "each
factual assertion" guidance with an explicit rule:

> Every sentence containing an independently verifiable fact MUST produce
> exactly one claim object. If your answer contains multiple factual
> assertions (e.g. "A because B."), split them into separate claims, each
> with its own citation and supporting_quote.

Include a worked example in the prompt template showing a compound answer
("A because B.") split into 2 claim objects.

### 6. Verification summary in trace

`dev_trace.log_verification(...)` (or a new call site) prints:

```
Verification Summary
Claims generated : 4
Claims verified   : 3
Claims failed     : 1
Verification Ratio: 75%
```

Computed from existing `VerificationReport.total_claims` /
`verified_claims` / `failed_claims` — no new fields needed on that model,
just new trace formatting. `confidence_scorer.py` is NOT changed in this
PR — it keeps its current heuristic. Wiring `verification_ratio` directly
into `score_confidence` is a follow-up, not bundled here.

## Testing

- `retrieval/retriever` tests: fused-candidate truncation applied before
  rerank call; budget validation rejects `fused_top_n` outside
  `[rerank_top_n, dense_k+sparse_k]`; trace fields populated correctly.
- `citation_verifier` tests: reattribution case now asserts
  `passed=False`, `failure_reason="citation_reattribution_candidate"`,
  and `claim.citation_ids` unchanged (currently asserts the opposite).
- `rag_pipeline` tests: `_reconcile_inline_citations` drift case asserts
  `RagAnswer.answer` unchanged (no rewrite) and
  `citation_status=CitationStatus.INLINE_DRIFT`; add a case for
  `VERIFICATION_FAILED` precedence over `INLINE_DRIFT` when both
  conditions are present.
- `prompt_builder` test: new instruction text and worked example present
  in v2 template output.

**Note:** one claim per assertion is a prompt-level instruction, not an
enforced invariant — it improves verification granularity but can't
guarantee the model decomposes every complex answer perfectly. No code
enforces claim count; a model that ignores the instruction still
produces a valid (if coarser) verification result.
- One end-to-end regression: a drifted-citation answer flows through
  the full pipeline and comes out with `citation_status=INLINE_DRIFT`,
  unmodified answer text, and the trace shows "no mutation performed".

## Deferred (not in this change)

- Comparative-query retrieval (RQ1/RQ3 gap) — needs query decomposition /
  multi-query retrieval, separate design.
- Wiring `verification_ratio` into `score_confidence` — noted as a
  natural follow-up in change #6, not bundled here.
