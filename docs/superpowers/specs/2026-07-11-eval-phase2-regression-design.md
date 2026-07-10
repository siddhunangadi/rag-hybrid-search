# Evaluation Framework Phase 2: Baseline Storage, Regression Comparison, CI Gating

**Date:** 2026-07-11
**Status:** Approved design (pending final spec review)
**Depends on:** Phase 1 static benchmark (`rag_pipeline/eval/`, `scripts/run_eval.py`)

## Problem

One eval run measures quality *today*. It cannot tell you whether a code change made
quality *worse*. Continuous regression detection needs three pieces:

1. A frozen reference point — the **baseline**
2. Diff logic with tolerance tiers — the **comparison**
3. Automated enforcement — the **CI gate**

This is snapshot testing applied to ML quality metrics.

## Decisions (settled with user)

| Decision | Choice | Rejected alternatives |
|---|---|---|
| CI judge | Objective metrics only; judge metrics compared only when both sides have them | Full judge in CI (cost, flakiness, secrets); nightly judge workflow (deferred) |
| Threshold model | Two-tier warn/fail, absolute deltas | Single threshold (noisy); exact-match (flaky) |
| Baseline updates | Explicit `--update-baseline`, committed to git, reviewable in PR diff | CI auto-update (silent quality ratchet); artifact/experiment stores like MLflow/W&B (overkill at this scale) |
| CI scope | Compare-only: unit tests + fixture-pipeline smoke + baseline schema validation. Real eval runs stay local | Full eval in CI (needs corpus + keys checked in) |
| Threshold source | `eval/thresholds.yaml`, hardcoded defaults as fallback | Hardcoded only (per-project tolerance differs) |
| Baseline naming | `--baseline-name <name>` → `eval/baselines/<name>.json`, default `main` | Single fixed file (blocks strategy comparison later) |
| Delta semantics | Absolute deltas on 0–1 ratio metrics | Relative % (explodes on small bases: 0.02→0.04 reads as "+100%") |

## Architecture

```
scripts/run_eval.py  (CLI flags, table rendering, exit codes)
        │
        ├── rag_pipeline/eval/baseline.py     (save/load/validate baseline JSON)
        ├── rag_pipeline/eval/thresholds.py   (load thresholds.yaml, defaults)
        └── rag_pipeline/eval/comparison.py   (pure compare logic, no I/O)
```

`comparison.py` never reads files or prints — takes dicts, returns a result object.
All I/O lives at the edges (`baseline.py`, `run_eval.py`).

## Baseline schema (`eval/baselines/<name>.json`)

```json
{
  "baseline_version": 1,
  "created_at": "2026-07-11T01:30:00Z",
  "git_commit": "abc1234",
  "package_version": "0.x.y",
  "question_set_hash": "sha256 of canonical questions.yaml content",
  "summary": { "...": "same shape as report.json summary block" },
  "per_question": {
    "q001": {"objective_metrics": {"citation_precision": 1.0, "...": "..."}, "status": "success"}
  }
}
```

- `baseline_version` enables painless future schema migrations.
- `question_set_hash`: comparing runs over different question sets is meaningless.
  Compare aborts with a clear error on hash mismatch, forcing a deliberate baseline refresh.
- `per_question` stores each question's `objective_metrics` and `status` keyed by id.

## Thresholds (`eval/thresholds.yaml`)

```yaml
evaluation:
  thresholds:
    citation_precision: {warn: 0.02, fail: 0.05}
    citation_recall:    {warn: 0.02, fail: 0.05}
    citation_f1:        {warn: 0.02, fail: 0.05}
    coverage:           {warn: 0.02, fail: 0.05}
    judge_score:        {warn: 0.05, fail: 0.10}
  error_count: {warn: 0, fail: 1}     # warn on any new error, fail on >1 new
  per_question_fail: 0.5              # per-question metric drop that flags regardless of aggregates
```

Missing file → hardcoded defaults identical to the above. Partial file → merge over defaults.

## Comparison semantics

`compare_to_baseline(current_summary, current_per_question, baseline, thresholds) -> ComparisonResult`

- Per aggregate metric: delta = current − baseline. Drop ≥ fail → FAIL; ≥ warn → WARN; else OK.
  Improvements reported as info, never gate.
- Error count: increase over baseline compared against `error_count` tiers.
- Judge metrics: compared only when present on **both** sides; skipped silently otherwise
  (objective-only CI stays clean).
- **Per-question tier:** a question whose objective metric drops by ≥ `per_question_fail`
  vs its own baseline entry is a FAIL finding, even when aggregates pass — averages hide
  localized catastrophes. Question add/remove is impossible mid-compare because hash
  mismatch already aborts.
- `ComparisonResult`: list of findings `(metric, scope, baseline, current, delta, status)`;
  overall status = worst finding.

## CLI

- `run_eval.py --update-baseline [--baseline-name main]` — write baseline from this run.
- `run_eval.py --compare-baseline [--baseline-name main]` — run, compare, render table,
  exit 1 on FAIL, 0 otherwise (warnings printed).
- `scripts/check_baseline.py --report <report.json> [--baseline-name main]` — compare an
  existing report without re-running eval (quick local checks; CI-compatible).

Table output:

```
Metric                Baseline  Current   Δ       Status
citation_precision    0.87      0.82      -0.05   FAIL
citation_recall       0.76      0.73      -0.03   WARN
citation_f1 (q017)    1.00      0.20      -0.80   FAIL (per-question)
```

Missing baseline → clear message: run `--update-baseline` first (exit 1 for compare).
Unknown `baseline_version` or corrupt JSON → clear error, exit 1.

## CI (`.github/workflows/ci.yml`)

Single workflow, on push/PR:

1. Install deps (uv), run `pytest`
2. `run_eval.py --fixture-pipeline` smoke (no network, no corpus)
3. Validate `eval/baselines/*.json` against schema (version, required keys, hash format)

No API keys, no corpus in CI. The real regression loop is local:
run eval → compare → (if intentional) update baseline → commit; reviewer sees baseline
diff in PR.

## Testing (TDD, matching Phase 1 style)

Unit (`tests/eval/test_baseline.py`, `test_thresholds.py`, `test_comparison.py`):
- baseline save/load round-trip; corrupt JSON and unknown version rejected clearly
- question-set hash mismatch aborts compare
- thresholds: defaults, full file, partial merge
- comparison: OK/WARN/FAIL per tier, improvement not gating, judge skip when absent,
  error-count tiers, per-question catastrophic drop flags while aggregates pass
- empty/None aggregates (Phase 1 returns None for empty sets) handled without crash

Integration (`tests/eval/test_run_eval_baseline.py`):
- `--fixture-pipeline --update-baseline` then `--compare-baseline` exits 0
- degraded fixture metrics → compare exits 1

## Deferred (Phase 2.5, not built now)

- HTML trend history / graphs over multiple runs
- Nightly judge workflow
- Baseline auto-update automation
