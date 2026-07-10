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
