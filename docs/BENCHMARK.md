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
