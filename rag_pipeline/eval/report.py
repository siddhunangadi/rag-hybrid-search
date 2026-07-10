import json
from pathlib import Path

_OBJECTIVE_FIELDS = ("citation_precision", "citation_recall", "citation_f1", "coverage")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _objective_aggregate(records: list[dict]) -> dict:
    metrics = [r["objective_metrics"] for r in records if r["status"] == "success"]
    result = {field: _mean([m[field] for m in metrics]) for field in _OBJECTIVE_FIELDS}
    result["latency_ms"] = _mean([m["latency_ms"] for m in metrics])
    result["verification_pass_rate"] = _mean([1.0 if m["verification_pass"] else 0.0 for m in metrics])
    return result


def _subjective_aggregate(records: list[dict]) -> dict:
    verdicts = [r["judge"]["verdict"] for r in records if r["status"] == "success"]
    scores = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0, "UNSUPPORTED": 0.0}
    return {
        "accuracy": _mean([scores[v] for v in verdicts]),
        "hallucination_rate": _mean([1.0 if v == "UNSUPPORTED" else 0.0 for v in verdicts]),
    }


def build_summary(records: list[dict]) -> dict:
    categories = sorted({r["category"] for r in records})
    error_count = sum(1 for r in records if r["status"] == "error")

    return {
        "objective": {
            "aggregate": _objective_aggregate(records),
            "by_category": {
                cat: _objective_aggregate([r for r in records if r["category"] == cat])
                for cat in categories
            },
        },
        "subjective": {
            "aggregate": _subjective_aggregate(records),
            "by_category": {
                cat: _subjective_aggregate([r for r in records if r["category"] == cat])
                for cat in categories
            },
        },
        "error_count": error_count,
        "total_questions": len(records),
    }


def _render_metric_table(title: str, aggregate: dict, by_category: dict) -> str:
    rows = "".join(
        f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>"
        for metric, value in aggregate.items()
    )
    category_rows = "".join(
        f"<tr><td>{cat}</td>" + "".join(f"<td>{value:.3f}</td>" for value in metrics.values()) + "</tr>"
        for cat, metrics in by_category.items()
    )
    header_cells = "".join(f"<th>{m}</th>" for m in aggregate)
    return f"""
    <h2>{title}</h2>
    <table border="1"><tr><th>Metric</th><th>Aggregate</th></tr>{rows}</table>
    <table border="1"><tr><th>Category</th>{header_cells}</tr>{category_rows}</table>
    """


def _render_html(report: dict) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    metadata_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metadata.items() if k != "settings")

    question_rows = "".join(
        f"<tr><td>{r['id']}</td><td>{r['category']}</td><td>{r['status']}</td>"
        f"<td>{(r.get('judge') or {}).get('verdict', '')}</td></tr>"
        for r in report["results"]
    )

    return f"""<!doctype html>
<html><head><title>Eval Report - {metadata['dataset']['name']}</title></head>
<body>
<h1>Evaluation Report: {metadata['dataset']['name']} v{metadata['dataset']['version']}</h1>
<table border="1">{metadata_rows}</table>
{_render_metric_table("Objective Metrics", summary["objective"]["aggregate"], summary["objective"]["by_category"])}
{_render_metric_table("Subjective (Judge) Metrics", summary["subjective"]["aggregate"], summary["subjective"]["by_category"])}
<h2>Per-Question Results</h2>
<table border="1"><tr><th>ID</th><th>Category</th><th>Status</th><th>Verdict</th></tr>{question_rows}</table>
</body></html>
"""


def write_report(records: list[dict], metadata: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report_version": metadata["report_version"],
        "metadata": metadata,
        "summary": build_summary(records),
        "results": records,
    }

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    html_path = out_dir / "report.html"
    html_path.write_text(_render_html(report))

    return json_path, html_path
