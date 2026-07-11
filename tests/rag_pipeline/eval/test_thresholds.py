import pytest

from rag_pipeline.eval.thresholds import DEFAULT_THRESHOLDS, load_thresholds


def test_missing_path_returns_defaults(tmp_path):
    t = load_thresholds(tmp_path / "nope.yaml")
    assert t == DEFAULT_THRESHOLDS
    assert t.metrics["citation_precision"].fail == 0.05
    assert t.metrics["accuracy"].warn == 0.05
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
