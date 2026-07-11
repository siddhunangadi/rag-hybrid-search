# Storage Migration: Chroma/SQLite/local-BM25 → Pinecone

**Date:** 2026-07-11
**Status:** Frozen — approved design
**Problem:** Render's free-tier web service has no persistent disk. `data_dir`
(Chroma's on-disk index, the SQLite chunk store, and the BM25 pickle) all live on
ephemeral container storage — every redeploy/restart wipes the whole index silently.
**Non-goal:** this is a storage-backend swap, not a retrieval-architecture change.
Hybrid retrieval (dense + sparse + weighted RRF + reranking) is fully preserved —
removing it would be a separate, deliberate, later decision, not a side effect of
fixing deployment.

## Architecture (end state)

```
PDF/doc → Loader → Chunker → NVIDIA embeddings
                                    │
                              ┌─────┴─────┐
                              │  Pinecone │
                              │  (dense + sparse vectors,
                              │   chunk/document metadata)
                              └─────┬─────┘
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  DenseRetriever         SparseRetriever
                         │                     │
                         └─────────┬───────────┘
                                   ▼
                            weighted_rrf (unchanged)
                                   ▼
                              Reranker (unchanged)
                                   ▼
                    [rest of the pipeline: unchanged]
```

Everything from the reranker up through generation, citation verification,
confidence scoring, comparative decomposition, grouped context, and the evaluation
harness is **untouched** — none of it depends on storage internals.

## What changes vs. what doesn't

**New:**
- `PineconeStore` — **one class implementing both** the `VectorStore` ABC
  (`upsert`, `query`, `delete`) and the `ChunkStore` ABC (`get`,
  `get_by_document`, `get_document_hash`, `put`, `delete_by_document`, `all`),
  backed by a single Pinecone index/client. Vector and metadata are already the
  same Pinecone record — a separate `PineconeChunkStore` class would just wrap
  the same underlying data a second time and hold a second client instance for
  no capability gain. One class, one client, both ABCs satisfied. Metadata
  payload: chunk text, heading, page, `legal_metadata`, `document_id` per
  vector (well under Pinecone's 40KB per-vector metadata limit for typical
  chunk sizes).
- `PineconeSparseIndex` — new class matching `BM25Index`'s `search(query, k) ->
  list[tuple[chunk_id, score]]` return shape, so `SparseRetriever`'s dependency
  swaps with no interface change. Backed by `pinecone-text`'s `BM25Encoder`,
  fit on this project's own corpus (not a generic pretrained corpus) for term
  weighting that matches today's locally-fit `BM25Index` quality.

**Unchanged (interfaces and logic both):**
- `DenseRetriever`, `SparseRetriever`, `HybridRetriever` (name, constructor,
  `search` signature — all preserved exactly).
- `weighted_rrf` fusion, reranker (`rerank.py`/`passthrough_rerank.py`).
- `RetrievedChunk` model — `bm25_score`, `rrf_score` fields kept as-is. Not
  removed until Phase 3 validation confirms the migration is solid; useful for
  side-by-side comparison against the old backend while both exist.
- Everything in `rag_pipeline/` (generation, verification, confidence,
  comparative retrieval, grouped context), `rag_hybrid_search/compliance/`,
  the evaluation harness.

**Removed, only after Phase 3 validation passes (not part of this plan's tasks):**
`ChromaVectorStore`, SQLite `ChunkStore` impl, local `BM25Index`, `data_dir`
config for these three, `chroma/`, `chunks.db`, `bm25.pkl` on disk.

## Query semantics: two calls, not Pinecone's native alpha-hybrid

Pinecone's built-in hybrid search combines dense+sparse into one weighted-dot-product
score per query, which would bypass `weighted_rrf` entirely and change fusion
behavior. To preserve the existing application-level `weighted_rrf` fusion logic
and retrieval flow, `DenseRetriever` and `SparseRetriever` issue **two
independent Pinecone queries** — one with only `vector` (dense) set, one with
only `sparse_vector` set — each returning its own ranked `(chunk_id, score)`
list, fused the same algorithm as today. This is not a promise of
bit-for-bit identical retrieval results: Pinecone's internal tie-breaking,
floating-point precision, and ANN index behavior can differ from Chroma's even
with identical fusion logic on top — which is exactly what Phase 3's evaluation
gate exists to catch if the difference matters.

## Sparse encoder: no persistence at all

Encoder state is configuration, not retrieval data — it doesn't belong stored
inside the search index (a sentinel vector per namespace doesn't scale cleanly
to multiple namespaces, e.g. `legal`/`finance`/`research`, and mixes concerns).
For this project's corpus size (~449 chunks today), refitting `BM25Encoder` is
trivial, so there is no persistence step at all: `PineconeSparseIndex` refits
in-process by reading the current corpus from `PineconeStore.all()` — once at
app startup, and again after every `add_document`/`remove_document` operation
(mirrors today's `rebuild_bm25_index()` call pattern exactly, just held in
memory instead of pickled to disk). A Render redeploy wiping in-process memory
is fine — startup already re-reads the corpus from Pinecone and refits.
Not persisting is explicitly the simplification here, not a gap: if the corpus
ever grows large enough that refit-on-startup becomes slow, that's the trigger
to revisit (Deferred).

## Backend selection: one flag, not a hard cutover

New config: **`RAG_STORAGE_BACKEND=local|pinecone`** (default `local`, unchanged
behavior for anyone not opting in) — a single selector, not three independent
flags. `local` and `pinecone` are the only two combinations this project will
ever run (Chroma+SQLite+local-BM25 always travel together; Pinecone bundles
vector+chunk+sparse together) — a `vector=pinecone, chunk=sqlite,
sparse=pinecone` mixed config is never a real deployment target, so exposing
three flags would only invite invalid combinations for no benefit. Plus new
secrets `RAG_PINECONE_API_KEY`, `RAG_PINECONE_INDEX_NAME`,
`RAG_PINECONE_ENVIRONMENT` (following the existing `RAG_`-prefixed convention).
`api/dependencies.py`'s container construction branches once on
`RAG_STORAGE_BACKEND` to wire either `(ChromaVectorStore, SQLite ChunkStore,
BM25Index)` or `(PineconeStore, PineconeStore, PineconeSparseIndex)` behind the
same ABCs — `IndexManager` and every retriever above it are unaware which
backend is active.

**Rollback:** if Pinecone-backed retrieval fails the Phase 3 evaluation gate or
hits production issues after cutover, setting `RAG_STORAGE_BACKEND=local`
restores the previous Chroma/SQLite/BM25 implementation with no code changes —
this is the whole reason the flag exists as a phased toggle rather than a
one-way rewrite.

`IndexManager.index()` gets a conditional path: the Pinecone backend folds
`chunk_store.put()` + `vector_store.upsert()` into one `PineconeStore.upsert()`
call storing the vector and its metadata together (Pinecone's native model —
vector and metadata are the same record), plus an in-process
`PineconeSparseIndex` refit instead of BM25's pickle-to-disk save.
The Chroma/SQLite/local-BM25 path is untouched.

## Phases

**Phase 1 — Dense + metadata migration.** `PineconeStore` (vector + chunk
methods), `IndexManager` conditional path for vector+chunk. This is an
**implementation phase, not a deployable configuration**: during Phase 1,
Pinecone vectors/metadata coexist with sparse retrieval still on local
`BM25Index` in the codebase, but this transitional state is never exposed as a
supported value of `RAG_STORAGE_BACKEND` — it exists only while Phase 2 is
being built, behind a build-time branch, not a runtime flag anyone would set.
The public `RAG_STORAGE_BACKEND=pinecone` mode is introduced only once Phase 2
completes, so the flag's two values (`local`, `pinecone`) never contradict the
single-flag design — there's never a moment where the flag itself offers a
mixed local/Pinecone configuration.

**Phase 2 — Sparse migration.** `PineconeSparseIndex`, `BM25Encoder`
in-process refit-on-startup/refit-on-write, `IndexManager` refit call added to
its Pinecone path. Once this lands, `RAG_STORAGE_BACKEND=pinecone` fully
replaces all three local stores with no partial state.

**Phase 3 — Validation.** Run `scripts/run_eval.py --compare-baseline` with a
Pinecone-backed baseline vs. the existing Chroma/SQLite/BM25 baseline. Required
before any default flips or old code is removed:
- Retrieval quality (citation precision/recall/F1) not regressed
- Verification pass rate not decreased
- Hallucination rate not increased
- Latency not regressed beyond an acceptable margin (reuses Phase 2 eval
  infrastructure's existing latency gate pattern)
- Manual spot-check that sparse relevance (corpus-fit `BM25Encoder`) is
  comparable to today's local BM25 on a few keyword-heavy queries

**Phase 4 (separate, deferred, not part of this plan) — Cleanup.** Remove
Chroma/SQLite/local-BM25 code, flip defaults, drop the feature flags, delete
local persistence config — only after Phase 3 passes in production.

## Success criterion

**Migration complete** when: the application no longer depends on local
persistent storage for indexing or retrieval. A full Render redeploy, followed
by application startup with `RAG_STORAGE_BACKEND=pinecone`, requires no manual
re-indexing before `/answer` queries succeed against the previously-indexed
corpus. This is the direct, verifiable statement of the deployment problem
this spec exists to fix — everything else in this document is how, not why.

## Testing

- `PineconeStore`/`PineconeSparseIndex`: unit tests against a mocked Pinecone
  client (no live network calls in the suite), verifying `PineconeStore`
  satisfies both the `VectorStore` and `ChunkStore` ABC contracts identically
  to `ChromaVectorStore`/SQLite `ChunkStore`'s existing test coverage, and
  `PineconeSparseIndex` matches `BM25Index`'s.
- `IndexManager`'s conditional path: parametrized over both backends, same
  assertions for `index()`/`remove_document()`/`rebuild_all()` behavior.
- Integration: `DenseRetriever`/`SparseRetriever`/`HybridRetriever` against
  `PineconeStore`/`PineconeSparseIndex` — no changes needed to these
  retrievers' own tests beyond swapping which store fixture they're
  constructed with (same interface).
- Live-provider test (skipped by default, matching the project's existing
  `tests/rag_pipeline/test_live_providers.py` pattern) exercising a real
  Pinecone index for one end-to-end round trip.

## Deferred (explicitly out of scope for this spec)

- Removing hybrid retrieval / BM25 / RRF — a separate, deliberate product
  decision, not a consequence of this migration.
- Renaming `HybridRetriever` or any other class.
- Removing `bm25_score`/`rrf_score` fields or fusion trace logging.
- Pinecone's native alpha-weighted hybrid search (bypassing app-level RRF).
- Scaling `BM25Encoder` refit beyond full-corpus-recompute-on-every-write.
- Phase 4 cleanup itself (tracked as a future, separate plan once Phase 3 passes).
