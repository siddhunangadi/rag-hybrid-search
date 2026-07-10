# Evaluation Framework — Design Spec (Phase 1: Static Benchmark)

Date: 2026-07-10

## Problem

There is no automated way to answer "did this change actually help?" for
retrieval, prompting, verification, or routing changes to the RAG pipeline.
Every optimization so far has been judged by manual inspection. This is the
first of a four-part roadmap (evaluation framework → verification-aware
retry → context organization → adaptive model routing); it is a prerequisite
because the other three need a way to measure whether they helped.

Phase 2 (regression comparison against a saved baseline, CI wiring) is
explicitly out of scope for this spec and will be its own follow-on design.

## Goals

- Run a fixed set of hand-authored questions through the real `RagPipeline`
  and produce a report with objective and judge-derived metrics, split
  cleanly so neither category masks the other.
- Persist a complete record per question (not just scores) so a broken
  parser, a bad judge call, or a surprising regression can be root-caused
  from the report alone, without re-running the pipeline.
- Make every report self-describing: pipeline configuration, model
  identities, and dataset version travel with the metrics so two reports
  can be compared without tribal knowledge of what changed between them.

## Non-goals

- No baseline/regression comparison, no CI gating (Phase 2).
- No single weighted "overall score" — every metric is reported
  independently; combining them is a policy decision left to the reader,
  not baked into the harness.
- No new LLM provider integrations — the judge provider is a config seam,
  defaulting to the existing generation provider.
- No cost/token accounting yet (deferred, same as the plan that shipped
  comparative-query retrieval deferred it for the pipeline itself).

## Architecture

```
question
     │
     ▼
 RagPipeline
     │
     ▼
 RagAnswer + Trace
     │
     ├────────► Objective Metrics
     │
     ├────────► Retrieval Record
     │
     └────────► Judge Provider
                  │
                  ▼
          Evaluation Record (one per question)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 Metadata  Per-Question  Summary
              │
       report.json
              │
       report.html
```

**New package:** `rag_pipeline/eval/`
- `questions.py` — loads and validates `eval/questions.yaml`
- `judge.py` — calls the judge provider with a structured prompt, returns
  verdict + reasoning + raw response
- `metrics.py` — computes objective metrics from `RagAnswer` +
  `RequestTrace`; combines with judge output into one `EvaluationRecord`
- `report.py` — writes `report.json` and renders `report.html`

**Driver:** `scripts/run_eval.py` — builds the pipeline via
`build_container()` (same wiring as the API server), loads
`eval/questions.yaml`, runs every question sequentially, writes the report.
No new runtime dependencies; consistent with how `scripts/debug_retrieval.py`
already works in this repo.

## Data model

### `eval/questions.yaml`

```yaml
dataset:
  name: benchmark-v1
  version: "1.0.0"

questions:
  - id: q001
    question: "..."
    category: comparative   # factual | comparative | multi-hop | summarization | definition
    expected:
      answer: "..."
      citation_doc_ids: ["d1", "d3"]
      # reserved, unused in Phase 1 — documented so the schema doesn't
      # need to change to add them later:
      # acceptable_answers: [...]
      # minimum_claims: N
      # difficulty: easy | medium | hard
```

30–50 hand-authored questions against the real ingested corpus, spanning
all five categories, for this first version. The category taxonomy also
seeds the later adaptive-routing work but is not consumed by anything but
per-category reporting here.

### Judge verdicts

`CORRECT | PARTIAL | INCORRECT | UNSUPPORTED`, scored `1.0 | 0.5 | 0.0 | 0.0`
for the aggregate accuracy figure, but `UNSUPPORTED` is also reported as its
own rate (hallucination rate) — an answer that invents an unsupported claim
is a qualitatively different failure than one that's merely wrong, and this
system's whole verification philosophy is about catching exactly that
distinction.

### Evaluation record (per question, full contents of one entry in `report.json`'s `results` list)

```json
{
  "id": "q001",
  "question": "...",
  "category": "comparative",
  "expected": { "answer": "...", "citation_doc_ids": ["d1", "d3"] },

  "model_answer": "...",
  "citations": ["d1", "d3"],
  "verification": { "...": "VerificationReport, as-is" },
  "confidence": { "...": "ConfidenceScores, as-is" },
  "status": "success",
  "error_type": null,
  "error_message": null,

  "objective_metrics": {
    "latency_ms": 2830,
    "citation_precision": 1.0,
    "citation_recall": 1.0,
    "citation_f1": 1.0,
    "verification_pass": true,
    "coverage": 1.0
  },

  "judge": {
    "verdict": "PARTIAL",
    "reasoning": "...",
    "prompt": "...",
    "raw_response": "..."
  },

  "retrieval": {
    "retrieved_chunk_ids": ["..."],
    "reranked_chunk_ids": ["..."],
    "chunks_used_ids": ["..."],
    "document_ids_used": ["..."],
    "context_size_chars": 4210,
    "retrieval_latency_ms": 340,
    "rerank_latency_ms": 120,
    "generation_latency_ms": 2200
  }
}
```

`retrieval` is populated straight from the `RequestTrace`/`RetrievalTrace`
object already built during `answer()` — no new instrumentation, recorded
but not scored. When something regresses, this tells you whether retrieval
or generation changed.

`status` is `"success"` or `"error"`; on `"error"`, `error_type` is a short
machine-checkable code (e.g. `"generation_timeout"`, `"parse_error"`,
`"provider_error"`) and `error_message` carries the human-readable detail.
This replaces a bare `error: <string|null>` field so aggregation can group
by `error_type` instead of string-matching messages.

### Report metadata (top of `report.json`, also rendered at the top of `report.html`)

```json
{
  "report_version": "1",
  "metadata": {
    "timestamp": "2026-07-10T16:40:00+05:30",
    "git_commit": "32e359a",
    "package_version": "0.9.0",
    "generation_model": "...",
    "judge_model": "...",
    "prompt_version": "v2",
    "judge_prompt_version": "v1",
    "settings": { "...": "sanitized Settings.model_dump(), secrets excluded" },
    "corpus_version": "...",
    "dataset": { "name": "benchmark-v1", "version": "1.0.0" }
  }
}
```

`settings` is the full `Settings.model_dump()` from the container the
pipeline was built from, with secret fields (`nvidia_api_key`,
`gemini_api_key`, `debug_token`) stripped — this replaces hand-picking
individual config fields (`dense_k`, `rerank_top_n`, ...) into metadata, so
every future config addition is captured automatically without a report
schema change. `report_version` is bumped whenever the report.json shape
itself changes, independent of `dataset.version` (question set changes) and
`package_version` (pip-installable version string, useful when running
outside a git checkout — `git_commit` remains the primary identifier when
available). `judge_prompt_version` tracks the judge prompt the same way
`prompt_version` already tracks the generation prompt, since judge prompts
evolve too.

Pulled from the same `Settings`/container the pipeline is built from, plus
`git rev-parse HEAD` and the dataset's own `dataset:` block. This is what
makes two reports comparable months apart without guesswork.

### Summary (aggregate + per-category, also in `report.json`)

Two independent groups, never merged into one score:

- **Objective:** mean latency, citation precision/recall/F1, verification
  pass rate, mean coverage, error rate — each also broken out per category.
- **Subjective (judge-derived):** accuracy (weighted verdict mean),
  hallucination rate (`UNSUPPORTED` fraction) — each also broken out per
  category.

## Judge provider

```python
@dataclass
class EvalConfig:
    generation_provider: GenerationProvider   # from build_container()
    judge_provider: GenerationProvider         # defaults to generation_provider
```

`scripts/run_eval.py` accepts `--judge-provider` / `EVAL_JUDGE_PROVIDER`;
unset means the judge reuses the generation provider (Phase 1 default is
"judge = generation," but the seam exists so a stronger/independent judge
model can be swapped in later without a refactor). No new provider
implementations are added in this spec.

## Report rendering

`report.html` is a single static page, generated by a plain string/template
(no JS framework, no server) — metadata block at top, objective and
subjective summary tables (aggregate + per-category), then one row per
question linking to its full evaluation record (judge reasoning, raw
response, retrieval ids) for drill-down.

## Testing

- Unit tests for `metrics.py` (citation precision/recall/F1 math, verdict
  scoring, per-category aggregation) using synthetic `RagAnswer`/
  `RequestTrace` fixtures — no real pipeline or LLM calls.
- Unit tests for `judge.py`'s response parsing (valid verdict, malformed
  JSON, missing fields) using a fake provider, mirroring the existing
  `MockProvider` pattern used elsewhere in this codebase.
- `questions.py` schema validation tests (missing required fields, unknown
  category, reserved-but-absent fields tolerated).
- One integration-style test running `scripts/run_eval.py` end-to-end
  against a tiny (2-3 question) in-memory fixture pipeline, asserting
  `report.json` has the expected shape — not a correctness assertion on
  real corpus content.

## Deferred (not in this spec)

- Phase 2: baseline storage + regression comparison, CI gating.
- Expanding the dataset beyond 30–50 questions.
- Cost/token accounting per run.
- A second, independently-configured judge model (the seam exists; wiring
  a second real provider is deferred until it's actually needed).
- Using the category taxonomy for adaptive routing (separate spec).
- Multi-run variance per question (running each question N times to detect
  answer instability) — valuable once model routing is in play, not needed
  for a single-model Phase 1 baseline.
- Any weighted/combined "overall score" — deliberately excluded; metrics
  stay independent to avoid disputes over weighting.
