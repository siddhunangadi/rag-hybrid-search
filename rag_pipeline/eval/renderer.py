"""Console rendering of comparison results. Presentation only — no logic, no I/O."""
from rag_pipeline.eval.schema import ComparisonResult


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
