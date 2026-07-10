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
