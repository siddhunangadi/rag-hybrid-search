from rag_pipeline.eval.renderer import render_comparison
from rag_pipeline.eval.schema import ComparisonResult, Finding


def test_render_comparison_table():
    result = ComparisonResult(findings=[
        Finding(metric="citation_f1", scope="aggregate",
                baseline=0.9, current=0.8, delta=-0.1, status="fail"),
        Finding(metric="pipeline_config", scope="config",
                baseline=None, current=None, delta=None, status="warn"),
    ])
    out = render_comparison(result)
    assert "Status" in out
    assert "citation_f1" in out and "FAIL" in out
    assert "-0.100" in out
    assert "-" in out  # None rendered as dash
    assert out.strip().endswith("Overall: FAIL")
