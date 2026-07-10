# Evaluation Phase 2: Regression Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Baseline storage, two-tier regression comparison (aggregate + per-question), and a compare-only CI gate for the Phase 1 evaluation harness.

**Architecture:** Typed Pydantic models (`schema.py`) shared by an I/O layer (`baseline.py`, `thresholds.py`) and a pure comparison engine (`comparison.py`). `scripts/run_eval.py` gains `--update-baseline` / `--compare-baseline` flags; `scripts/check_baseline.py` compares an existing report without re-running eval. GitHub Actions runs tests + fixture smoke + baseline schema validation only (no corpus, no API keys).

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML (already deps), pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-07-11-eval-phase2-regression-design.md`

## Global Constraints

- All deltas are absolute (current − baseline) on 0–1 ratio metrics.
- Judge/subjective metrics compared only when present on both sides; skipped silently otherwise.
- Question-set hash mismatch aborts comparison (exit 4). Pipeline-config mismatch only warns.
- Exit codes: 0 success/warn, 1 regression FAIL, 2 baseline missing, 3 baseline corrupt/unknown version, 4 hash mismatch, 5 CLI usage.
- Improvements never gate (reported as `info`).
- Comparison layer performs no file or console I/O.
- Tests live in `tests/rag_pipeline/eval/` (existing Phase 1 location).
- Every task: TDD — failing test first, then minimal implementation, run affected tests, commit.

---

### Task 1: Typed models (`schema.py`)

**Files:**
- Create: `rag_pipeline/eval/schema.py`
- Test: `tests/rag_pipeline/eval/test_schema.py`

**Interfaces:**
- Produces: `BASELINE_VERSION: int = 1`; models `MetricThreshold(warn: float, fail: float)`, `Thresholds(metrics: dict[str, MetricThreshold], error_count: MetricThreshold, per_question_fail: float)`, `QuestionMetrics(status: str, objective_metrics: dict[str, float | bool | None])`, `SnapshotSummary(objective: dict[str, float] | None, subjective: dict[str, float] | None, error_count: int, total_questions: int)`, `EvaluationSnapshot(question_set_hash: str, summary: SnapshotSummary, per_question: dict[str, QuestionMetrics], pipeline_config: dict)`, `Baseline(baseline_version: int, created_at: str, git_commit: str, package_version: str, branch: str | None, notes: str | None, question_set_hash: str, pipeline_config: dict, summary: SnapshotSummary, per_question: dict[str, QuestionMetrics])` with method `Baseline.snapshot() -> EvaluationSnapshot`; `Finding(metric: str, scope: str, baseline: float | None, current: float | None, delta: float | None, status: str)`, `ComparisonResult(findings: list[Finding])` with property `overall -> str` ("ok"/"warn"/"fail" = worst finding; `info` never worsens).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag_pipeline/eval/test_schema.py
import pytest
from pydantic import ValidationError

from rag_pipeline.eval.schema import (
    BASELINE_VERSION,
    Baseline,
    ComparisonResult,
    EvaluationSnapshot,
    Finding,
    MetricThreshold,
    QuestionMetrics,
    SnapshotSummary,
    Thresholds,
)


def _summary():
    return SnapshotSummary(
        objective={"citation_precision": 0.9, "citation_recall": 0.8},
        subjective=None,
        error_count=0,
        total_questions=2,
    )


def _baseline_payload():
    return {
        "baseline_version": BASELINE_VERSION,
        "created_at": "2026-07-11T00:00:00+00:00",
        "git_commit": "abc1234",
        "package_version": "0.1.0",
        "branch": "main",
        "notes": None,
        "question_set_hash": "deadbeef",
        "pipeline_config": {"chunk_size": 500},
        "summary": _summary().model_dump(),
        "per_question": {
            "q001": {"status": "success", "objective_metrics": {"citation_f1": 1.0}},
        },
    }


def test_baseline_round_trips_and_exposes_snapshot():
    baseline = Baseline.model_validate(_baseline_payload())
    snap = baseline.snapshot()
    assert isinstance(snap, EvaluationSnapshot)
    assert snap.question_set_hash == "deadbeef"
    assert snap.summary.objective["citation_precision"] == 0.9
    assert snap.per_question["q001"].objective_metrics["citation_f1"] == 1.0
    assert snap.pipeline_config == {"chunk_size": 500}


def test_baseline_rejects_missing_required_fields():
    payload = _baseline_payload()
    del payload["question_set_hash"]
    with pytest.raises(ValidationError):
        Baseline.model_validate(payload)


def test_metric_threshold_and_thresholds_validate():
    t = Thresholds(
        metrics={"citation_f1": MetricThreshold(warn=0.02, fail=0.05)},
        error_count=MetricThreshold(warn=0, fail=1),
        per_question_fail=0.5,
    )
    assert t.metrics["citation_f1"].fail == 0.05


def test_comparison_result_overall_is_worst_finding():
    ok = Finding(metric="a", scope="aggregate", baseline=1.0, current=1.0, delta=0.0, status="ok")
    warn = Finding(metric="b", scope="aggregate", baseline=1.0, current=0.97, delta=-0.03, status="warn")
    fail = Finding(metric="c", scope="aggregate", baseline=1.0, current=0.5, delta=-0.5, status="fail")
    info = Finding(metric="d", scope="aggregate", baseline=0.5, current=0.9, delta=0.4, status="info")
    assert ComparisonResult(findings=[ok]).overall == "ok"
    assert ComparisonResult(findings=[ok, warn]).overall == "warn"
    assert ComparisonResult(findings=[ok, warn, fail]).overall == "fail"
    assert ComparisonResult(findings=[ok, info]).overall == "ok"
    assert ComparisonResult(findings=[]).overall == "ok"


def test_question_metrics_allows_bool_and_none_values():
    qm = QuestionMetrics(status="success", objective_metrics={"verification_pass": True, "coverage": None})
    assert qm.objective_metrics["verification_pass"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag_pipeline/eval/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_pipeline.eval.schema'`

- [ ] **Step 3: Write minimal implementation**

```python
# rag_pipeline/eval/schema.py
"""Typed models for baseline storage and regression comparison (eval Phase 2)."""
from typing import Literal, Optional

from pydantic import BaseModel

BASELINE_VERSION = 1

FindingStatus = Literal["ok", "warn", "fail", "info"]


class MetricThreshold(BaseModel):
    warn: float
    fail: float


class Thresholds(BaseModel):
    metrics: dict[str, MetricThreshold]
    error_count: MetricThreshold
    per_question_fail: float = 0.5


class QuestionMetrics(BaseModel):
    status: str
    objective_metrics: dict[str, Optional[float | bool]] = {}


class SnapshotSummary(BaseModel):
    objective: Optional[dict[str, float]] = None
    subjective: Optional[dict[str, float]] = None
    error_count: int
    total_questions: int


class EvaluationSnapshot(BaseModel):
    question_set_hash: str
    summary: SnapshotSummary
    per_question: dict[str, QuestionMetrics]
    pipeline_config: dict = {}


class Baseline(BaseModel):
    baseline_version: int
    created_at: str
    git_commit: str = "unknown"
    package_version: str = "unknown"
    branch: Optional[str] = None
    notes: Optional[str] = None
    question_set_hash: str
    pipeline_config: dict = {}
    summary: SnapshotSummary
    per_question: dict[str, QuestionMetrics]

    def snapshot(self) -> EvaluationSnapshot:
        return EvaluationSnapshot(
            question_set_hash=self.question_set_hash,
            summary=self.summary,
            per_question=self.per_question,
            pipeline_config=self.pipeline_config,
        )


class Finding(BaseModel):
    metric: str
    scope: str
    baseline: Optional[float] = None
    current: Optional[float] = None
    delta: Optional[float] = None
    status: FindingStatus


_SEVERITY = {"ok": 0, "info": 0, "warn": 1, "fail": 2}
_BY_SEVERITY = {0: "ok", 1: "warn", 2: "fail"}


class ComparisonResult(BaseModel):
    findings: list[Finding]

    @property
    def overall(self) -> str:
        worst = max((_SEVERITY[f.status] for f in self.findings), default=0)
        return _BY_SEVERITY[worst]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/rag_pipeline/eval/test_schema.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline/eval/schema.py tests/rag_pipeline/eval/test_schema.py
git commit -m "feat: add typed eval baseline/comparison models"
```

---

### Task 2: Threshold config loader (`thresholds.py`)

**Files:**
- Create: `rag_pipeline/eval/thresholds.py`
- Test: `tests/rag_pipeline/eval/test_thresholds.py`

**Interfaces:**
- Consumes: `Thresholds`, `MetricThreshold` from Task 1.
- Produces: `DEFAULT_THRESHOLDS: Thresholds` (module constant) and `load_thresholds(path: str | Path | None) -> Thresholds`. Missing/None path → defaults. Partial YAML → deep-merge over defaults. Malformed YAML or wrong types → `ValueError` with filename in message.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag_pipeline/eval/test_thresholds.py
import pytest

from rag_pipeline.eval.thresholds import DEFAULT_THRESHOLDS, load_thresholds


def test_missing_path_returns_defaults(tmp_path):
    t = load_thresholds(tmp_path / "nope.yaml")
    assert t == DEFAULT_THRESHOLDS
    assert t.metrics["citation_precision"].fail == 0.05
    assert t.metrics["judge_score"].warn == 0.05
    assert t.error_count.warn == 0
    assert t.error_count.fail == 1
    assert t.per_question_fail == 0.5


def test_none_path_returns_defaults():
    assert load_thresholds(None) == DEFAULT_THRESHOLDS


def test_partial_file_merges_over_defaults(tmp_path):
    p = tmp_path / "thresholds.yaml"
    p.write_text(
        """
evaluation:
  thresholds:
    citation_precision: {warn: 0.01, fail: 0.03}
  per_question_fail: 0.4
"""
    )
    t = load_thresholds(p)
    assert t.metrics["citation_precision"].warn == 0.01
    assert t.metrics["citation_precision"].fail == 0.03
    # untouched metric keeps default
    assert t.metrics["citation_recall"].fail == 0.05
    assert t.per_question_fail == 0.4
    assert t.error_count.fail == 1


def test_malformed_yaml_raises_value_error(tmp_path):
    p = tmp_path / "thresholds.yaml"
    p.write_text("evaluation: [unclosed")
    with pytest.raises(ValueError, match="thresholds.yaml"):
        load_thresholds(p)


def test_wrong_types_raise_value_error(tmp_path):
    p = tmp_path / "thresholds.yaml"
    p.write_text("evaluation:\n  thresholds:\n    citation_f1: {warn: banana, fail: 0.05}\n")
    with pytest.raises(ValueError, match="thresholds.yaml"):
        load_thresholds(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag_pipeline/eval/test_thresholds.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# rag_pipeline/eval/thresholds.py
"""Load regression thresholds from eval/thresholds.yaml, merged over defaults."""
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from rag_pipeline.eval.schema import MetricThreshold, Thresholds

DEFAULT_THRESHOLDS = Thresholds(
    metrics={
        "citation_precision": MetricThreshold(warn=0.02, fail=0.05),
        "citation_recall": MetricThreshold(warn=0.02, fail=0.05),
        "citation_f1": MetricThreshold(warn=0.02, fail=0.05),
        "coverage": MetricThreshold(warn=0.02, fail=0.05),
        "judge_score": MetricThreshold(warn=0.05, fail=0.10),
    },
    error_count=MetricThreshold(warn=0, fail=1),
    per_question_fail=0.5,
)

DEFAULT_THRESHOLDS_PATH = Path("eval/thresholds.yaml")


def load_thresholds(path: Optional[str | Path]) -> Thresholds:
    if path is None:
        return DEFAULT_THRESHOLDS
    path = Path(path)
    if not path.exists():
        return DEFAULT_THRESHOLDS
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        section = (raw.get("evaluation") or {})
        merged_metrics = {name: t.model_copy() for name, t in DEFAULT_THRESHOLDS.metrics.items()}
        for name, tiers in (section.get("thresholds") or {}).items():
            merged_metrics[name] = MetricThreshold.model_validate(tiers)
        error_count = (
            MetricThreshold.model_validate(section["error_count"])
            if "error_count" in section
            else DEFAULT_THRESHOLDS.error_count
        )
        return Thresholds(
            metrics=merged_metrics,
            error_count=error_count,
            per_question_fail=section.get("per_question_fail", DEFAULT_THRESHOLDS.per_question_fail),
        )
    except (yaml.YAMLError, ValidationError, TypeError, AttributeError) as e:
        raise ValueError(f"Invalid thresholds file {path.name}: {e}") from e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/rag_pipeline/eval/test_thresholds.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline/eval/thresholds.py tests/rag_pipeline/eval/test_thresholds.py
git commit -m "feat: add configurable regression thresholds with defaults"
```

---

### Task 3: Pure comparison engine (`comparison.py`)

**Files:**
- Create: `rag_pipeline/eval/comparison.py`
- Test: `tests/rag_pipeline/eval/test_comparison.py`

**Interfaces:**
- Consumes: all Task 1 models; thresholds from Task 2 (in tests).
- Produces: `QuestionSetMismatchError(ValueError)`; `compare(current: EvaluationSnapshot, baseline: EvaluationSnapshot, thresholds: Thresholds) -> ComparisonResult`.

**Semantics (from spec):**
- Hash mismatch → raise `QuestionSetMismatchError` before any comparison.
- Aggregate objective metrics: for each name in `thresholds.metrics` (except `judge_score`), if present on both sides: `delta = current − baseline`; `delta <= -fail` → fail, `<= -warn` → warn, `> 0` → info, else ok.
- `judge_score`: same tiers, using `summary.subjective["judge_score"]`, only when subjective present on both sides. Silent skip otherwise (no finding).
- Error count: `delta = current − baseline`; `delta > fail` → fail, `delta > warn` → warn, `delta < 0` → info, else ok. (Defaults warn=0 fail=1: one new error warns, two+ fail.)
- Per-question: for each qid in baseline, for each metric name in `thresholds.metrics` (except `judge_score`) present as a number/bool on both sides (bools cast to 1.0/0.0, None skipped): drop ≥ `per_question_fail` → fail finding with scope `per_question:<qid>`. No warn tier, no info per question.
- Pipeline config: if both non-empty and unequal → single warn finding, metric `pipeline_config`, scope `config`, baseline/current/delta all None.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag_pipeline/eval/test_comparison.py
import pytest

from rag_pipeline.eval.comparison import QuestionSetMismatchError, compare
from rag_pipeline.eval.schema import (
    EvaluationSnapshot,
    QuestionMetrics,
    SnapshotSummary,
)
from rag_pipeline.eval.thresholds import DEFAULT_THRESHOLDS


def _snap(objective=None, subjective=None, error_count=0, per_question=None,
          question_set_hash="h1", pipeline_config=None):
    return EvaluationSnapshot(
        question_set_hash=question_set_hash,
        summary=SnapshotSummary(
            objective=objective,
            subjective=subjective,
            error_count=error_count,
            total_questions=len(per_question or {}) or 1,
        ),
        per_question=per_question or {},
        pipeline_config=pipeline_config or {},
    )


def _findings_by(result, metric):
    return [f for f in result.findings if f.metric == metric]


def test_hash_mismatch_aborts():
    with pytest.raises(QuestionSetMismatchError):
        compare(_snap(question_set_hash="a"), _snap(question_set_hash="b"), DEFAULT_THRESHOLDS)


def test_aggregate_ok_warn_fail_tiers():
    base = _snap(objective={"citation_precision": 0.90, "citation_recall": 0.90, "citation_f1": 0.90})
    cur = _snap(objective={"citation_precision": 0.895, "citation_recall": 0.87, "citation_f1": 0.80})
    result = compare(cur, base, DEFAULT_THRESHOLDS)
    assert _findings_by(result, "citation_precision")[0].status == "ok"      # -0.005
    assert _findings_by(result, "citation_recall")[0].status == "warn"       # -0.03
    assert _findings_by(result, "citation_f1")[0].status == "fail"           # -0.10
    assert result.overall == "fail"


def test_improvement_is_info_and_never_gates():
    base = _snap(objective={"citation_f1": 0.5})
    cur = _snap(objective={"citation_f1": 0.9})
    result = compare(cur, base, DEFAULT_THRESHOLDS)
    assert _findings_by(result, "citation_f1")[0].status == "info"
    assert result.overall == "ok"


def test_judge_skipped_when_absent_on_either_side():
    base = _snap(objective={"citation_f1": 0.9}, subjective={"judge_score": 0.9})
    cur = _snap(objective={"citation_f1": 0.9}, subjective=None)
    result = compare(cur, base, DEFAULT_THRESHOLDS)
    assert _findings_by(result, "judge_score") == []


def test_judge_compared_when_present_both_sides():
    base = _snap(objective={"citation_f1": 0.9}, subjective={"judge_score": 0.9})
    cur = _snap(objective={"citation_f1": 0.9}, subjective={"judge_score": 0.75})
    result = compare(cur, base, DEFAULT_THRESHOLDS)
    assert _findings_by(result, "judge_score")[0].status == "fail"           # -0.15


def test_error_count_tiers():
    base = _snap(error_count=0)
    one_new = compare(_snap(error_count=1), base, DEFAULT_THRESHOLDS)
    two_new = compare(_snap(error_count=2), base, DEFAULT_THRESHOLDS)
    fewer = compare(_snap(error_count=0), _snap(error_count=2), DEFAULT_THRESHOLDS)
    assert _findings_by(one_new, "error_count")[0].status == "warn"
    assert _findings_by(two_new, "error_count")[0].status == "fail"
    assert _findings_by(fewer, "error_count")[0].status == "info"


def test_per_question_catastrophe_fails_while_aggregates_pass():
    base_pq = {
        "q001": QuestionMetrics(status="success", objective_metrics={"citation_f1": 1.0}),
        "q002": QuestionMetrics(status="success", objective_metrics={"citation_f1": 0.6}),
    }
    cur_pq = {
        "q001": QuestionMetrics(status="success", objective_metrics={"citation_f1": 0.2}),
        "q002": QuestionMetrics(status="success", objective_metrics={"citation_f1": 0.6}),
    }
    # aggregates identical -> no aggregate finding fails
    base = _snap(objective={"citation_f1": 0.8}, per_question=base_pq)
    cur = _snap(objective={"citation_f1": 0.8}, per_question=cur_pq)
    result = compare(cur, base, DEFAULT_THRESHOLDS)
    pq = [f for f in result.findings if f.scope == "per_question:q001"]
    assert pq and pq[0].status == "fail"
    assert result.overall == "fail"


def test_per_question_bool_metrics_cast_and_none_skipped():
    base_pq = {"q001": QuestionMetrics(status="success",
                                       objective_metrics={"coverage": None, "citation_f1": True})}
    cur_pq = {"q001": QuestionMetrics(status="success",
                                      objective_metrics={"coverage": 0.9, "citation_f1": False})}
    result = compare(_snap(per_question=cur_pq), _snap(per_question=base_pq), DEFAULT_THRESHOLDS)
    f1 = [f for f in result.findings if f.scope == "per_question:q001" and f.metric == "citation_f1"]
    cov = [f for f in result.findings if f.metric == "coverage" and f.scope.startswith("per_question")]
    assert f1 and f1[0].status == "fail"   # True->False = drop of 1.0
    assert cov == []                       # None on baseline side -> skipped


def test_empty_aggregates_do_not_crash():
    result = compare(_snap(objective=None), _snap(objective=None), DEFAULT_THRESHOLDS)
    assert result.overall == "ok"


def test_pipeline_config_mismatch_warns_not_aborts():
    base = _snap(pipeline_config={"chunk_size": 500})
    cur = _snap(pipeline_config={"chunk_size": 800})
    result = compare(cur, base, DEFAULT_THRESHOLDS)
    cfg = [f for f in result.findings if f.metric == "pipeline_config"]
    assert cfg and cfg[0].status == "warn" and cfg[0].scope == "config"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag_pipeline/eval/test_comparison.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# rag_pipeline/eval/comparison.py
"""Pure regression comparison between two evaluation snapshots. No I/O here."""
from typing import Optional

from rag_pipeline.eval.schema import (
    ComparisonResult,
    EvaluationSnapshot,
    Finding,
    Thresholds,
)


class QuestionSetMismatchError(ValueError):
    """Snapshots were produced from different question sets; metrics not comparable."""


def _as_float(value: Optional[float | bool]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _tier(delta: float, warn: float, fail: float) -> str:
    if delta <= -fail:
        return "fail"
    if delta <= -warn:
        return "warn"
    if delta > 0:
        return "info"
    return "ok"


def compare(
    current: EvaluationSnapshot, baseline: EvaluationSnapshot, thresholds: Thresholds
) -> ComparisonResult:
    if current.question_set_hash != baseline.question_set_hash:
        raise QuestionSetMismatchError(
            f"Question set hash mismatch (baseline {baseline.question_set_hash!r}, "
            f"current {current.question_set_hash!r}). Re-create the baseline with "
            "--update-baseline after changing questions."
        )

    findings: list[Finding] = []

    # Aggregate objective + judge metrics.
    for name, tiers in thresholds.metrics.items():
        if name == "judge_score":
            base_side, cur_side = baseline.summary.subjective, current.summary.subjective
        else:
            base_side, cur_side = baseline.summary.objective, current.summary.objective
        if not base_side or not cur_side:
            continue
        base_val, cur_val = base_side.get(name), cur_side.get(name)
        if base_val is None or cur_val is None:
            continue
        delta = cur_val - base_val
        findings.append(Finding(
            metric=name, scope="aggregate", baseline=base_val, current=cur_val,
            delta=delta, status=_tier(delta, tiers.warn, tiers.fail),
        ))

    # Error count: increases are bad.
    err_delta = current.summary.error_count - baseline.summary.error_count
    if err_delta > thresholds.error_count.fail:
        err_status = "fail"
    elif err_delta > thresholds.error_count.warn:
        err_status = "warn"
    elif err_delta < 0:
        err_status = "info"
    else:
        err_status = "ok"
    findings.append(Finding(
        metric="error_count", scope="aggregate",
        baseline=float(baseline.summary.error_count),
        current=float(current.summary.error_count),
        delta=float(err_delta), status=err_status,
    ))

    # Per-question catastrophic drops (aggregates can hide these).
    for qid, base_q in baseline.per_question.items():
        cur_q = current.per_question.get(qid)
        if cur_q is None:
            continue  # hash equality means sets match; defensive only
        for name in thresholds.metrics:
            if name == "judge_score":
                continue
            base_val = _as_float(base_q.objective_metrics.get(name))
            cur_val = _as_float(cur_q.objective_metrics.get(name))
            if base_val is None or cur_val is None:
                continue
            delta = cur_val - base_val
            if delta <= -thresholds.per_question_fail:
                findings.append(Finding(
                    metric=name, scope=f"per_question:{qid}",
                    baseline=base_val, current=cur_val, delta=delta, status="fail",
                ))

    # Pipeline config drift: warn, never abort (A/B comparison can be deliberate).
    if baseline.pipeline_config and current.pipeline_config \
            and baseline.pipeline_config != current.pipeline_config:
        findings.append(Finding(
            metric="pipeline_config", scope="config",
            baseline=None, current=None, delta=None, status="warn",
        ))

    return ComparisonResult(findings=findings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/rag_pipeline/eval/test_comparison.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline/eval/comparison.py tests/rag_pipeline/eval/test_comparison.py
git commit -m "feat: add pure regression comparison engine with per-question tier"
```

---

### Task 4: Baseline I/O (`baseline.py`)

**Files:**
- Create: `rag_pipeline/eval/baseline.py`
- Test: `tests/rag_pipeline/eval/test_baseline.py`

**Interfaces:**
- Consumes: `Baseline`, `BASELINE_VERSION`, `EvaluationSnapshot` from Task 1.
- Produces: `BaselineMissingError(FileNotFoundError)`, `BaselineCorruptError(ValueError)`; `question_set_hash(questions_path: str | Path) -> str` (sha256 hex of file bytes); `baseline_path(name: str, base_dir: str | Path = "eval/baselines") -> Path`; `save_baseline(baseline: Baseline, name: str, base_dir=...) -> Path`; `load_baseline(name: str, base_dir=...) -> Baseline` (raises the two errors above; unknown `baseline_version` → `BaselineCorruptError`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag_pipeline/eval/test_baseline.py
import pytest

from rag_pipeline.eval.baseline import (
    BaselineCorruptError,
    BaselineMissingError,
    baseline_path,
    load_baseline,
    question_set_hash,
    save_baseline,
)
from rag_pipeline.eval.schema import BASELINE_VERSION, Baseline, QuestionMetrics, SnapshotSummary


def _baseline():
    return Baseline(
        baseline_version=BASELINE_VERSION,
        created_at="2026-07-11T00:00:00+00:00",
        git_commit="abc1234",
        package_version="0.1.0",
        branch="main",
        notes="initial",
        question_set_hash="deadbeef",
        pipeline_config={"chunk_size": 500},
        summary=SnapshotSummary(objective={"citation_f1": 0.9}, subjective=None,
                                error_count=0, total_questions=1),
        per_question={"q001": QuestionMetrics(status="success",
                                              objective_metrics={"citation_f1": 0.9})},
    )


def test_save_load_round_trip(tmp_path):
    path = save_baseline(_baseline(), "main", base_dir=tmp_path)
    assert path == tmp_path / "main.json"
    loaded = load_baseline("main", base_dir=tmp_path)
    assert loaded == _baseline()


def test_missing_baseline_raises_with_hint(tmp_path):
    with pytest.raises(BaselineMissingError, match="--update-baseline"):
        load_baseline("main", base_dir=tmp_path)


def test_corrupt_json_raises(tmp_path):
    (tmp_path / "main.json").write_text("{not json")
    with pytest.raises(BaselineCorruptError):
        load_baseline("main", base_dir=tmp_path)


def test_unknown_version_raises(tmp_path):
    b = _baseline().model_copy(update={"baseline_version": 999})
    (tmp_path / "main.json").write_text(b.model_dump_json())
    with pytest.raises(BaselineCorruptError, match="999"):
        load_baseline("main", base_dir=tmp_path)


def test_schema_violation_raises(tmp_path):
    (tmp_path / "main.json").write_text('{"baseline_version": 1}')
    with pytest.raises(BaselineCorruptError):
        load_baseline("main", base_dir=tmp_path)


def test_question_set_hash_is_stable_and_content_sensitive(tmp_path):
    q = tmp_path / "questions.yaml"
    q.write_text("questions: [a]")
    h1 = question_set_hash(q)
    assert h1 == question_set_hash(q)
    q.write_text("questions: [a, b]")
    assert question_set_hash(q) != h1


def test_baseline_path_maps_name(tmp_path):
    assert baseline_path("bm25", base_dir=tmp_path) == tmp_path / "bm25.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/rag_pipeline/eval/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# rag_pipeline/eval/baseline.py
"""Baseline persistence: save/load eval/baselines/<name>.json. Validation via schema models."""
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from rag_pipeline.eval.schema import BASELINE_VERSION, Baseline

DEFAULT_BASE_DIR = Path("eval/baselines")


class BaselineMissingError(FileNotFoundError):
    pass


class BaselineCorruptError(ValueError):
    pass


def question_set_hash(questions_path: str | Path) -> str:
    return hashlib.sha256(Path(questions_path).read_bytes()).hexdigest()


def baseline_path(name: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> Path:
    return Path(base_dir) / f"{name}.json"


def save_baseline(baseline: Baseline, name: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> Path:
    path = baseline_path(name, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(baseline.model_dump_json(indent=2))
    return path


def load_baseline(name: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> Baseline:
    path = baseline_path(name, base_dir)
    if not path.exists():
        raise BaselineMissingError(
            f"No baseline at {path}. Create one with: python scripts/run_eval.py --update-baseline"
        )
    try:
        payload = json.loads(path.read_text())
        baseline = Baseline.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as e:
        raise BaselineCorruptError(f"Corrupt baseline {path}: {e}") from e
    if baseline.baseline_version != BASELINE_VERSION:
        raise BaselineCorruptError(
            f"Unsupported baseline_version {baseline.baseline_version} in {path} "
            f"(supported: {BASELINE_VERSION})"
        )
    return baseline
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/rag_pipeline/eval/test_baseline.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add rag_pipeline/eval/baseline.py tests/rag_pipeline/eval/test_baseline.py
git commit -m "feat: add baseline save/load with versioning and question-set hashing"
```

---

### Task 5: Snapshot assembly + CLI integration in `run_eval.py`

**Files:**
- Create: `rag_pipeline/eval/snapshot.py`
- Modify: `scripts/run_eval.py` (add flags, comparison rendering, exit codes; current `main()` is at the bottom of the file)
- Test: `tests/rag_pipeline/eval/test_snapshot.py`, extend `tests/test_run_eval.py`

**Interfaces:**
- Consumes: Tasks 1–4 (`compare`, `load_baseline`, `save_baseline`, `question_set_hash`, `load_thresholds`, all models).
- Produces:
  - `rag_pipeline/eval/snapshot.py`: `snapshot_from_records(records: list[dict], summary: dict, question_set_hash: str, pipeline_config: dict) -> EvaluationSnapshot` and `snapshot_summary_from_report_summary(summary: dict) -> SnapshotSummary`. Phase 1 `build_summary` shape: `summary["objective"]["aggregate"]` is a metric→float dict or `None`; same for `subjective`; plus `error_count`, `total_questions`.
  - `run_eval.py` new flags: `--update-baseline`, `--compare-baseline`, `--baseline-name` (default `main`), `--baseline-dir` (default `eval/baselines`, for tests), `--notes` (default None), `--thresholds` (default `eval/thresholds.yaml`).
  - `render_comparison(result: ComparisonResult) -> str` in `run_eval.py` — fixed-width table `Metric / Scope / Baseline / Current / Δ / Status`.
  - Exit codes per Global Constraints. `report.json` metadata gains `"question_set_hash"` and `"pipeline_config"`.
  - `_pipeline_config(settings) -> dict` selecting comparable fields: `provider, generation_model, chunking_strategy, chunk_size, chunk_overlap, dense_k, sparse_k, rerank_top_n, rerank_backend` (via `getattr(settings, f, None)`).

- [ ] **Step 1: Write the failing snapshot tests**

```python
# tests/rag_pipeline/eval/test_snapshot.py
from rag_pipeline.eval.snapshot import snapshot_from_records


def _record(qid, status="success", metrics=None):
    return {
        "id": qid,
        "category": "factual",
        "status": status,
        "objective_metrics": metrics or {},
    }


def test_snapshot_built_from_phase1_report_shapes():
    records = [
        _record("q001", metrics={"citation_f1": 1.0, "verification_pass": True,
                                 "coverage": 0.8, "latency_ms": 120.0}),
        _record("q002", status="error"),
    ]
    summary = {
        "objective": {"aggregate": {"citation_f1": 1.0}, "by_category": {}},
        "subjective": {"aggregate": None, "by_category": {}},
        "error_count": 1,
        "total_questions": 2,
    }
    snap = snapshot_from_records(records, summary, "hash123", {"chunk_size": 500})
    assert snap.question_set_hash == "hash123"
    assert snap.summary.objective == {"citation_f1": 1.0}
    assert snap.summary.subjective is None
    assert snap.summary.error_count == 1
    assert snap.per_question["q001"].objective_metrics["verification_pass"] is True
    assert snap.per_question["q002"].status == "error"
    assert snap.per_question["q002"].objective_metrics == {}


def test_snapshot_handles_none_objective_aggregate():
    summary = {
        "objective": {"aggregate": None, "by_category": {}},
        "subjective": {"aggregate": None, "by_category": {}},
        "error_count": 0,
        "total_questions": 0,
    }
    snap = snapshot_from_records([], summary, "h", {})
    assert snap.summary.objective is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/rag_pipeline/eval/test_snapshot.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `snapshot.py`**

```python
# rag_pipeline/eval/snapshot.py
"""Build EvaluationSnapshot from Phase 1 report shapes (records + build_summary output)."""
from rag_pipeline.eval.schema import EvaluationSnapshot, QuestionMetrics, SnapshotSummary


def _numeric_only(aggregate):
    if aggregate is None:
        return None
    return {k: float(v) for k, v in aggregate.items() if isinstance(v, (int, float))}


def snapshot_summary_from_report_summary(summary: dict) -> SnapshotSummary:
    return SnapshotSummary(
        objective=_numeric_only((summary.get("objective") or {}).get("aggregate")),
        subjective=_numeric_only((summary.get("subjective") or {}).get("aggregate")),
        error_count=summary["error_count"],
        total_questions=summary["total_questions"],
    )


def snapshot_from_records(
    records: list[dict], summary: dict, question_set_hash: str, pipeline_config: dict
) -> EvaluationSnapshot:
    per_question = {
        r["id"]: QuestionMetrics(
            status=r["status"],
            objective_metrics={
                k: v for k, v in (r.get("objective_metrics") or {}).items()
                if v is None or isinstance(v, (int, float, bool))
            },
        )
        for r in records
    }
    return EvaluationSnapshot(
        question_set_hash=question_set_hash,
        summary=snapshot_summary_from_report_summary(summary),
        per_question=per_question,
        pipeline_config=pipeline_config,
    )
```

- [ ] **Step 4: Run snapshot tests**

Run: `python -m pytest tests/rag_pipeline/eval/test_snapshot.py -v`
Expected: 2 passed

- [ ] **Step 5: Write failing CLI integration tests**

Append to `tests/test_run_eval.py` (the existing smoke test at the top of that file shows the questions-YAML pattern; `json`, `subprocess`, `sys`, `Path` are already imported there):

```python
QUESTIONS_YAML = """
dataset:
  name: smoke-test
  version: "0.0.1"

questions:
  - id: q001
    question: "What is X?"
    category: factual
    expected:
      answer: "X is a thing."
      citation_doc_ids: ["d1"]
"""


def _run_eval(tmp_path, *extra_args):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(QUESTIONS_YAML)
    out_dir = tmp_path / "report"
    repo_root = Path(__file__).resolve().parent.parent
    return subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--fixture-pipeline",
         "--questions", str(questions_path), "--out-dir", str(out_dir),
         "--baseline-dir", str(tmp_path / "baselines"), *extra_args],
        cwd=repo_root, capture_output=True, text=True,
    )


def test_update_then_compare_baseline_exits_zero(tmp_path):
    create = _run_eval(tmp_path, "--update-baseline", "--notes", "initial")
    assert create.returncode == 0, create.stderr
    baseline_file = tmp_path / "baselines" / "main.json"
    assert baseline_file.exists()
    payload = json.loads(baseline_file.read_text())
    assert payload["baseline_version"] == 1
    assert payload["notes"] == "initial"
    assert payload["question_set_hash"]

    comparison = _run_eval(tmp_path, "--compare-baseline")
    assert comparison.returncode == 0, comparison.stderr
    assert "Status" in comparison.stdout


def test_compare_without_baseline_exits_2(tmp_path):
    result = _run_eval(tmp_path, "--compare-baseline")
    assert result.returncode == 2
    assert "--update-baseline" in result.stderr


def test_compare_with_corrupt_baseline_exits_3(tmp_path):
    (tmp_path / "baselines").mkdir()
    (tmp_path / "baselines" / "main.json").write_text("{broken")
    result = _run_eval(tmp_path, "--compare-baseline")
    assert result.returncode == 3


def test_update_and_compare_together_is_usage_error(tmp_path):
    result = _run_eval(tmp_path, "--update-baseline", "--compare-baseline")
    assert result.returncode == 5


def test_compare_with_changed_questions_exits_4(tmp_path):
    create = _run_eval(tmp_path, "--update-baseline")
    assert create.returncode == 0, create.stderr
    # mutate the questions file after baseline creation
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text(QUESTIONS_YAML.replace("What is X?", "What is Y?"))
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--fixture-pipeline",
         "--questions", str(questions_path), "--out-dir", str(tmp_path / "r2"),
         "--baseline-dir", str(tmp_path / "baselines"), "--compare-baseline"],
        cwd=repo_root, capture_output=True, text=True,
    )
    assert result.returncode == 4
```

- [ ] **Step 6: Run to verify failure**

Run: `python -m pytest tests/test_run_eval.py -v`
Expected: new tests FAIL (`unrecognized arguments: --baseline-dir`); the existing smoke test still passes.

- [ ] **Step 7: Extend `scripts/run_eval.py`**

Replace the existing `from rag_pipeline.eval.report import write_report  # noqa: E402` line with:

```python
from rag_pipeline.eval.report import build_summary, write_report  # noqa: E402
```

Add after the existing `rag_pipeline` imports:

```python
from rag_pipeline.eval.baseline import (  # noqa: E402
    BaselineCorruptError,
    BaselineMissingError,
    load_baseline,
    question_set_hash,
    save_baseline,
)
from rag_pipeline.eval.comparison import QuestionSetMismatchError, compare  # noqa: E402
from rag_pipeline.eval.schema import BASELINE_VERSION, Baseline, ComparisonResult  # noqa: E402
from rag_pipeline.eval.snapshot import snapshot_from_records  # noqa: E402
from rag_pipeline.eval.thresholds import load_thresholds  # noqa: E402
```

Add helpers before `main()`:

```python
_PIPELINE_CONFIG_FIELDS = [
    "provider", "generation_model", "chunking_strategy", "chunk_size",
    "chunk_overlap", "dense_k", "sparse_k", "rerank_top_n", "rerank_backend",
]

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_BASELINE_MISSING = 2
EXIT_BASELINE_CORRUPT = 3
EXIT_HASH_MISMATCH = 4
EXIT_USAGE = 5


def _pipeline_config(settings) -> dict:
    return {f: getattr(settings, f, None) for f in _PIPELINE_CONFIG_FIELDS}


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _format_value(v) -> str:
    if v is None:
        return "-"
    return f"{v:+.3f}" if v < 0 else f"{v:.3f}"


def render_comparison(result: ComparisonResult) -> str:
    header = f"{'Metric':<22}{'Scope':<22}{'Baseline':>10}{'Current':>10}{'Δ':>9}  Status"
    lines = [header, "-" * len(header)]
    for f in result.findings:
        lines.append(
            f"{f.metric:<22}{f.scope:<22}{_format_value(f.baseline):>10}"
            f"{_format_value(f.current):>10}{_format_value(f.delta):>9}  {f.status.upper()}"
        )
    lines.append(f"\nOverall: {result.overall.upper()}")
    return "\n".join(lines)
```

In `main()`, add arguments after the existing ones, then the mutual-exclusion check right after `parse_args()`:

```python
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--baseline-name", default="main")
    parser.add_argument("--baseline-dir", default="eval/baselines")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--thresholds", default="eval/thresholds.yaml")
    args = parser.parse_args()

    if args.update_baseline and args.compare_baseline:
        print("Use either --update-baseline or --compare-baseline, not both.", file=sys.stderr)
        sys.exit(EXIT_USAGE)
```

Right after `load_questions(...)`, compute the hash:

```python
    q_hash = question_set_hash(args.questions)
```

Add to the existing `metadata` dict:

```python
        "question_set_hash": q_hash,
        "pipeline_config": _pipeline_config(settings),
```

After the existing `write_report(...)` prints, append:

```python
    if args.update_baseline or args.compare_baseline:
        snapshot = snapshot_from_records(
            records, build_summary(records), q_hash, _pipeline_config(settings)
        )

    if args.update_baseline:
        baseline = Baseline(
            baseline_version=BASELINE_VERSION,
            created_at=metadata["timestamp"],
            git_commit=metadata["git_commit"],
            package_version=metadata["package_version"],
            branch=_git_branch(),
            notes=args.notes,
            question_set_hash=q_hash,
            pipeline_config=snapshot.pipeline_config,
            summary=snapshot.summary,
            per_question=snapshot.per_question,
        )
        path = save_baseline(baseline, args.baseline_name, base_dir=args.baseline_dir)
        print(f"Wrote baseline {path}")

    if args.compare_baseline:
        try:
            baseline = load_baseline(args.baseline_name, base_dir=args.baseline_dir)
            thresholds = load_thresholds(args.thresholds)
            result = compare(snapshot, baseline.snapshot(), thresholds)
        except BaselineMissingError as e:
            print(str(e), file=sys.stderr)
            sys.exit(EXIT_BASELINE_MISSING)
        except BaselineCorruptError as e:
            print(str(e), file=sys.stderr)
            sys.exit(EXIT_BASELINE_CORRUPT)
        except QuestionSetMismatchError as e:
            print(str(e), file=sys.stderr)
            sys.exit(EXIT_HASH_MISMATCH)
        print(render_comparison(result))
        if result.overall == "fail":
            sys.exit(EXIT_REGRESSION)
```

- [ ] **Step 8: Run the full run_eval test file**

Run: `python -m pytest tests/test_run_eval.py -v`
Expected: all pass (existing smoke + 5 new)

- [ ] **Step 9: Run whole suite**

Run: `python -m pytest -q`
Expected: all pass, no regressions

- [ ] **Step 10: Commit**

```bash
git add rag_pipeline/eval/snapshot.py tests/rag_pipeline/eval/test_snapshot.py scripts/run_eval.py tests/test_run_eval.py
git commit -m "feat: wire baseline update/compare into run_eval with typed snapshots and exit codes"
```

---

### Task 6: Standalone compare of existing report (`check_baseline.py`)

**Files:**
- Create: `scripts/check_baseline.py`
- Test: extend `tests/test_run_eval.py`

**Interfaces:**
- Consumes: Task 5 `snapshot_from_records` and `render_comparison`/exit-code constants (imported from `scripts.run_eval`), Task 4 loaders, Task 3 `compare`, Task 2 `load_thresholds`.
- Produces: CLI `python scripts/check_baseline.py --report <report.json> [--baseline-name main] [--baseline-dir eval/baselines] [--thresholds eval/thresholds.yaml]`. Reads `report.json` (needs `metadata.question_set_hash`, `summary`, `results`), builds a snapshot, compares, prints table, exits with the same codes as Task 5. Reports produced before Phase 2 (no `question_set_hash` in metadata) → exit 5 with message naming the missing field.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_eval.py`:

```python
def test_check_baseline_compares_existing_report(tmp_path):
    create = _run_eval(tmp_path, "--update-baseline")
    assert create.returncode == 0, create.stderr
    report_path = tmp_path / "report" / "report.json"
    assert report_path.exists()
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/check_baseline.py",
         "--report", str(report_path),
         "--baseline-dir", str(tmp_path / "baselines")],
        cwd=repo_root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Overall: OK" in result.stdout


def test_check_baseline_rejects_pre_phase2_report(tmp_path):
    report_path = tmp_path / "old_report.json"
    report_path.write_text(json.dumps({
        "metadata": {}, "summary": {"error_count": 0, "total_questions": 0,
                                    "objective": {"aggregate": None},
                                    "subjective": {"aggregate": None}},
        "results": [],
    }))
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/check_baseline.py", "--report", str(report_path),
         "--baseline-dir", str(tmp_path)],
        cwd=repo_root, capture_output=True, text=True,
    )
    assert result.returncode == 5
    assert "question_set_hash" in result.stderr
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_run_eval.py -k check_baseline -v`
Expected: FAIL — script does not exist

- [ ] **Step 3: Implement the script**

```python
#!/usr/bin/env python3
# scripts/check_baseline.py
"""Compare an existing report.json against a stored baseline without re-running eval."""
import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag_pipeline.eval.baseline import (  # noqa: E402
    BaselineCorruptError,
    BaselineMissingError,
    load_baseline,
)
from rag_pipeline.eval.comparison import QuestionSetMismatchError, compare  # noqa: E402
from rag_pipeline.eval.snapshot import snapshot_from_records  # noqa: E402
from rag_pipeline.eval.thresholds import load_thresholds  # noqa: E402
from scripts.run_eval import (  # noqa: E402
    EXIT_BASELINE_CORRUPT,
    EXIT_BASELINE_MISSING,
    EXIT_HASH_MISMATCH,
    EXIT_REGRESSION,
    EXIT_USAGE,
    render_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--baseline-name", default="main")
    parser.add_argument("--baseline-dir", default="eval/baselines")
    parser.add_argument("--thresholds", default="eval/thresholds.yaml")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    q_hash = report.get("metadata", {}).get("question_set_hash")
    if not q_hash:
        print(
            "Report has no metadata.question_set_hash (pre-Phase 2 report?). "
            "Re-run scripts/run_eval.py to produce a comparable report.",
            file=sys.stderr,
        )
        sys.exit(EXIT_USAGE)

    snapshot = snapshot_from_records(
        report["results"], report["summary"], q_hash,
        report.get("metadata", {}).get("pipeline_config", {}),
    )
    try:
        baseline = load_baseline(args.baseline_name, base_dir=args.baseline_dir)
        result = compare(snapshot, baseline.snapshot(), load_thresholds(args.thresholds))
    except BaselineMissingError as e:
        print(str(e), file=sys.stderr)
        sys.exit(EXIT_BASELINE_MISSING)
    except BaselineCorruptError as e:
        print(str(e), file=sys.stderr)
        sys.exit(EXIT_BASELINE_CORRUPT)
    except QuestionSetMismatchError as e:
        print(str(e), file=sys.stderr)
        sys.exit(EXIT_HASH_MISMATCH)

    print(render_comparison(result))
    if result.overall == "fail":
        sys.exit(EXIT_REGRESSION)


if __name__ == "__main__":
    main()
```

Note: importing `scripts.run_eval` executes only its module-level imports (`main()` is guarded); `scripts/__init__.py` already exists, so the package import works.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_run_eval.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add scripts/check_baseline.py tests/test_run_eval.py
git commit -m "feat: add check_baseline.py for comparing existing reports"
```

---

### Task 7: Default thresholds file, CI workflow, docs

**Files:**
- Create: `eval/thresholds.yaml`, `.github/workflows/ci.yml`
- Modify: `README.md` (evaluation section — add regression workflow)

**Interfaces:**
- Consumes: everything prior; CI invokes `pytest`, `run_eval.py --fixture-pipeline`, and validates any committed baselines by loading them through `load_baseline`.

- [ ] **Step 1: Write `eval/thresholds.yaml`** (matches `DEFAULT_THRESHOLDS` exactly; exists as the visible, editable copy)

```yaml
evaluation:
  thresholds:
    citation_precision: {warn: 0.02, fail: 0.05}
    citation_recall:    {warn: 0.02, fail: 0.05}
    citation_f1:        {warn: 0.02, fail: 0.05}
    coverage:           {warn: 0.02, fail: 0.05}
    judge_score:        {warn: 0.05, fail: 0.10}
  error_count: {warn: 0, fail: 1}
  per_question_fail: 0.5
```

- [ ] **Step 2: Verify defaults equivalence**

Run: `python -c "from rag_pipeline.eval.thresholds import load_thresholds, DEFAULT_THRESHOLDS; assert load_thresholds('eval/thresholds.yaml') == DEFAULT_THRESHOLDS; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Install dependencies
        run: uv sync
      - name: Run test suite
        run: uv run python -m pytest -q
      - name: Eval harness smoke (fixture pipeline, no network)
        run: uv run python scripts/run_eval.py --fixture-pipeline --out-dir /tmp/eval-smoke
      - name: Validate committed baselines
        run: |
          uv run python - <<'EOF'
          from pathlib import Path
          from rag_pipeline.eval.baseline import load_baseline
          base_dir = Path("eval/baselines")
          files = sorted(base_dir.glob("*.json")) if base_dir.exists() else []
          for f in files:
              load_baseline(f.stem, base_dir=base_dir)
              print(f"OK {f}")
          print(f"Validated {len(files)} baseline(s)")
          EOF
```

- [ ] **Step 4: Update README evaluation section**

Add under the existing evaluation docs (adapt heading level to surrounding content; if no evaluation section exists, add one at the end):

````markdown
### Regression detection

```bash
# Create/refresh the baseline (commit the result)
python scripts/run_eval.py --update-baseline --notes "why the baseline moved"

# Compare a fresh run against the baseline (exit 1 on regression)
python scripts/run_eval.py --compare-baseline

# Compare an existing report without re-running
python scripts/check_baseline.py --report eval/reports/<ts>/report.json
```

Thresholds live in `eval/thresholds.yaml` (two tiers: warn prints, fail gates).
Baselines are named: `--baseline-name bm25` → `eval/baselines/bm25.json`.
Exit codes: 0 ok/warn, 1 regression, 2 baseline missing, 3 corrupt, 4 question-set changed, 5 usage.
````

- [ ] **Step 5: Full suite + lint**

Run: `python -m pytest -q && ruff check .`
Expected: all pass, no lint errors

- [ ] **Step 6: Commit**

```bash
git add eval/thresholds.yaml .github/workflows/ci.yml README.md
git commit -m "feat: add thresholds config, CI workflow with eval smoke and baseline validation"
```

---

## Final verification

- [ ] `python -m pytest -q` — full suite green
- [ ] `ruff check .` — clean
- [ ] Manual: `python scripts/run_eval.py --fixture-pipeline --update-baseline --baseline-dir /tmp/bl && python scripts/run_eval.py --fixture-pipeline --compare-baseline --baseline-dir /tmp/bl` → table printed, exit 0
