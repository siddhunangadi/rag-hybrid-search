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
