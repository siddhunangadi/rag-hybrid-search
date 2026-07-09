# Retrieval Benchmark

Reproducible retrieval-quality check for the hybrid retrieval stack (BM25 + dense + RRF fusion + CrossEncoder rerank), run over the sample doc corpus in `tests/fixtures/sample_docs/` (3 short markdown docs: deployment, onboarding, setup) against 6 fixed queries in `tests/fixtures/benchmark_queries.py` (2 per doc).

## Honest scope

This is a small, fixed toy corpus (3 documents), not a large-scale IR benchmark — treat the numbers as a regression guard and a sanity check, not a claim of production-scale retrieval quality. It also uses `FakeEmbeddingProvider` (a deterministic trigram-hash embedding, not a real embedding model) so the numbers are reproducible without an API key; a real embedding model (NVIDIA or a local sentence-transformer) would need its own benchmark run to validate dense-retrieval quality specifically.

## Metrics

- **Recall@3** — fraction of queries where the expected source document appears in the top 3 reranked results.
- **MRR** (mean reciprocal rank) — average of `1/rank` of the first correct hit per query.

## Results (current, `uv run python -m scripts.benchmark`)

| Metric | Value |
|---|---|
| Queries | 6 |
| Recall@3 | 1.00 |
| MRR | 1.00 |

All 6 queries hit their expected document at rank 1. `tests/test_benchmark.py` asserts `recall_at_3 == 1.0` and `mrr >= 0.9` as a regression guard — a retrieval change that drops these numbers will fail CI.

## Running it yourself

```bash
uv run python -m scripts.benchmark
```

Prints the full per-query breakdown as JSON.

## Citation verification — what it catches

`verify_citations` (`rag_pipeline/citation_verifier.py`) checks two independent things per claim: the cited chunk ID actually exists in the retrieved context, and the claimed `supporting_quote` is a genuine (near-)verbatim match against that chunk's text — not just plausible-sounding. Three real cases from `tests/rag_pipeline/test_citation_verifier.py`, against the fixed context `"[d1]\nEmployees get 20 days of paid annual leave per year."`:

| Claim | citation_ids | supporting_quote | Result |
|---|---|---|---|
| "Employees get 20 days leave" | `["d1"]` | "20 days of paid annual leave" | ✅ verified — quote genuinely present in `d1` |
| "Employees get unlimited leave" | `["d99"]` | "unlimited leave" | ❌ failed — `d99` doesn't exist; flagged in `hallucinated_doc_ids` |
| "Employees get free lunch" | `["d1"]` | "completely unrelated text about lunch" | ❌ failed — `d1` is real, but the quote isn't actually in it (`quote_match_score` below threshold) |

The second and third cases are the two distinct hallucination shapes this catches: citing a source that doesn't exist at all, versus citing a real source for a claim it doesn't actually support.

## Confidence scoring — what moves the number

`score_confidence` combines three independent signals into `overall` — it's arithmetic over the verification report and retrieval scores, not another model call, so a given input always produces the same score:

- **retrieval** — mean rerank score of the chunks actually used
- **citations** — fraction of claims that passed verification (1.0 if all pass, 0.0 if all fail, proportional in between)
- **coverage** — fraction of retrieved context that ended up cited by at least one claim

A previously-fixed bug is worth noting here directly: with `PassthroughReranker` (the default, no model, no network call), `rerank_score` used to come back `None` for every chunk, which silently zeroed out the `retrieval` component of every confidence score regardless of how good the actual retrieval was. Fixed by falling back to RRF rank when no reranker populated a score; regression-tested in `tests/rag_pipeline/test_confidence_scorer.py`.

## Known failure modes (honest, not hidden)

- **Toy corpus, not IR-benchmark scale.** 3 documents, 6 queries — a retrieval regression guard, not evidence of production-scale recall.
- **`FakeEmbeddingProvider` in the benchmark.** A deterministic trigram-hash, not a real embedding model — validates the retrieval *pipeline*, not embedding-model quality. Swap in `NvidiaProvider` and re-run for that.
- **PDF table extraction is text-layer only, not OCR.** pdfplumber preserves table structure when a text layer exists; a scanned image with no text layer still yields nothing (see `rag_hybrid_search/ingestion/loaders/pdf.py` docstring).
- **No multi-tenant isolation.** Every query searches the full corpus regardless of who uploaded what — acceptable for a single-tenant deployment, not for storing multiple users' confidential documents.
