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
