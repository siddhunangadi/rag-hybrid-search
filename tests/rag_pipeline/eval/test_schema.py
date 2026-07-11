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


def test_models_are_frozen():
    baseline = Baseline.model_validate(_baseline_payload())
    with pytest.raises(ValidationError):
        baseline.git_commit = "mutated"
    snap = baseline.snapshot()
    with pytest.raises(ValidationError):
        snap.question_set_hash = "mutated"
