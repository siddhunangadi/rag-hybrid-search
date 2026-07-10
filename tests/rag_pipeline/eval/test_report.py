import json

from rag_pipeline.eval.report import build_summary, write_report


def _record(category, verdict, precision=1.0, recall=1.0, f1=1.0, verification_pass=True, latency_ms=100.0, coverage=1.0):
    return {
        "id": "q", "category": category, "status": "success", "error_type": None,
        "objective_metrics": {
            "latency_ms": latency_ms, "citation_precision": precision, "citation_recall": recall,
            "citation_f1": f1, "verification_pass": verification_pass, "coverage": coverage,
        },
        "judge": {"verdict": verdict, "reasoning": "...", "prompt": "...", "raw_response": "..."},
    }


def test_build_summary_aggregates_objective_and_subjective_separately():
    records = [
        _record("factual", "CORRECT"),
        _record("factual", "PARTIAL", precision=0.5, recall=0.5, f1=0.5, verification_pass=False),
        _record("comparative", "UNSUPPORTED"),
    ]

    summary = build_summary(records)

    assert summary["objective"]["aggregate"]["citation_precision"] == (1.0 + 0.5 + 1.0) / 3
    assert summary["objective"]["aggregate"]["verification_pass_rate"] == 2 / 3
    assert summary["subjective"]["aggregate"]["accuracy"] == (1.0 + 0.5 + 0.0) / 3
    assert summary["subjective"]["aggregate"]["hallucination_rate"] == 1 / 3

    assert summary["objective"]["by_category"]["factual"]["citation_precision"] == (1.0 + 0.5) / 2
    assert summary["subjective"]["by_category"]["comparative"]["hallucination_rate"] == 1.0
    assert "overall_score" not in summary
    assert "combined_score" not in summary


def test_build_summary_excludes_error_records_from_metric_aggregation():
    records = [
        _record("factual", "CORRECT"),
        {"id": "q2", "category": "factual", "status": "error", "error_type": "generation_timeout",
         "objective_metrics": None, "judge": None},
    ]

    summary = build_summary(records)

    assert summary["objective"]["aggregate"]["citation_precision"] == 1.0
    assert summary["error_count"] == 1


def test_write_report_produces_json_and_html(tmp_path):
    records = [_record("factual", "CORRECT")]
    metadata = {
        "report_version": "1", "timestamp": "2026-07-10T00:00:00Z", "git_commit": "abc123",
        "package_version": "0.1.0", "generation_model": "mock", "judge_model": "mock",
        "prompt_version": "v2", "judge_prompt_version": "v1",
        "settings": {"dense_k": 10}, "corpus_version": "unknown",
        "dataset": {"name": "benchmark-v1", "version": "1.0.0"},
    }

    json_path, html_path = write_report(records, metadata, tmp_path)

    assert json_path.exists() and html_path.exists()
    written = json.loads(json_path.read_text())
    assert written["report_version"] == "1"
    assert written["metadata"]["git_commit"] == "abc123"
    assert written["results"] == records
    assert "summary" in written

    html = html_path.read_text()
    assert "benchmark-v1" in html
    assert "citation_precision" in html.lower() or "citation precision" in html.lower()
