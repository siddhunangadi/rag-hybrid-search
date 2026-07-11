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
- `PineconeVectorStore` — implements the existing `VectorStore` ABC (`upsert`,
  `query`, `delete`), backed by Pinecone dense vectors.
- `PineconeChunkStore` — implements the existing `ChunkStore` ABC (`get`,
  `get_by_document`, `get_document_hash`, `put`, `delete_by_document`, `all`),
  backed by Pinecone metadata payloads (chunk text, heading, page,
  `legal_metadata`, `document_id` stored per-vector; well under Pinecone's 40KB
  per-vector metadata limit for typical chunk sizes).
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
behavior. To keep `weighted_rrf`'s existing rank-position-based fusion logic
byte-identical, `DenseRetriever` and `SparseRetriever` issue **two independent
Pinecone queries** — one with only `vector` (dense) set, one with only
`sparse_vector` set — each returning its own ranked `(chunk_id, score)` list,
fused the same way as today.

## Sparse encoder persistence

`BM25Encoder`'s fitted state (corpus IDF statistics) is small (KBs, not the
multi-MB-at-scale problem a raw BM25 pickle can become) — persisted as a single
JSON blob stored inside Pinecone itself, under a reserved sentinel vector ID in a
dedicated namespace (not on local disk, so it survives redeploys same as
everything else). Refit triggered on every `add_document`/`remove_document`
operation, recomputing IDF over the full corpus. Acceptable for this project's
corpus sizes; refit-on-every-write is explicitly noted as not scaling to a huge
corpus (Deferred).

## Backend selection: feature flag, not a hard cutover

New config: `RAG_VECTOR_STORE=chroma|pinecone` (default `chroma`, unchanged
behavior for anyone not opting in), `RAG_CHUNK_STORE=sqlite|pinecone`,
`RAG_SPARSE_INDEX=local|pinecone`, plus `RAG_PINECONE_API_KEY`,
`RAG_PINECONE_INDEX_NAME`, `RAG_PINECONE_ENVIRONMENT` (new secrets, following the
existing `RAG_`-prefixed convention). `api/dependencies.py`'s container
construction branches on these flags to wire the chosen implementations behind
the same `VectorStore`/`ChunkStore` ABCs — `IndexManager` and every retriever
above it are unaware which backend is active.

`IndexManager.index()` gets a conditional path: the Pinecone backend folds
`chunk_store.put()` + `vector_store.upsert()` into `PineconeVectorStore.upsert()`
storing both the vector and its metadata in one call (Pinecone's native model —
vector and metadata are the same record), plus an incremental
`PineconeSparseIndex` upsert instead of BM25's full-corpus rebuild-and-save.
The Chroma/SQLite/local-BM25 path is untouched.

## Phases

**Phase 1 — Dense + metadata migration.** `PineconeVectorStore`,
`PineconeChunkStore`, feature flag wiring, `IndexManager` conditional path for
these two. Sparse retrieval keeps using local `BM25Index` regardless of the
`RAG_VECTOR_STORE` flag (still ephemeral-disk-exposed, but isolated to Phase 2).

**Phase 2 — Sparse migration.** `PineconeSparseIndex`, `BM25Encoder`
integration and encoder-state persistence, `IndexManager` incremental-upsert
path for sparse, `RAG_SPARSE_INDEX` flag wired through.

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

## Testing

- `PineconeVectorStore`/`PineconeChunkStore`/`PineconeSparseIndex`: unit tests
  against a mocked Pinecone client (no live network calls in the suite),
  verifying each satisfies its ABC's contract identically to
  `ChromaVectorStore`/SQLite `ChunkStore`/`BM25Index`'s existing test coverage.
- `IndexManager`'s conditional path: parametrized over both backends, same
  assertions for `index()`/`remove_document()`/`rebuild_all()` behavior.
- Integration: `DenseRetriever`/`SparseRetriever`/`HybridRetriever` against
  `PineconeVectorStore`/`PineconeChunkStore`/`PineconeSparseIndex` — no changes
  needed to these retrievers' own tests beyond swapping which store fixture
  they're constructed with (same interface).
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
