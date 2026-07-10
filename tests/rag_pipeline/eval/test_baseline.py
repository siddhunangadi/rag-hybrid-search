import pytest

from rag_pipeline.eval.baseline import (
    BaselineCorruptError,
    BaselineMissingError,
    baseline_path,
    load_baseline,
    question_set_hash,
    save_baseline,
)
from rag_pipeline.eval.schema import BASELINE_VERSION, Baseline, QuestionMetrics, SnapshotSummary


def _baseline():
    return Baseline(
        baseline_version=BASELINE_VERSION,
        created_at="2026-07-11T00:00:00+00:00",
        git_commit="abc1234",
        package_version="0.1.0",
        branch="main",
        notes="initial",
        question_set_hash="deadbeef",
        pipeline_config={"chunk_size": 500},
        summary=SnapshotSummary(objective={"citation_f1": 0.9}, subjective=None,
                                error_count=0, total_questions=1),
        per_question={"q001": QuestionMetrics(status="success",
                                              objective_metrics={"citation_f1": 0.9})},
    )


def test_save_load_round_trip(tmp_path):
    path = save_baseline(_baseline(), "main", base_dir=tmp_path)
    assert path == tmp_path / "main.json"
    loaded = load_baseline("main", base_dir=tmp_path)
    assert loaded == _baseline()
    # atomic write leaves no temp file behind
    assert list(tmp_path.glob("*.tmp")) == []


def test_missing_baseline_raises_with_hint(tmp_path):
    with pytest.raises(BaselineMissingError, match="--update-baseline"):
        load_baseline("main", base_dir=tmp_path)


def test_corrupt_json_raises(tmp_path):
    (tmp_path / "main.json").write_text("{not json")
    with pytest.raises(BaselineCorruptError):
        load_baseline("main", base_dir=tmp_path)


def test_unknown_version_raises(tmp_path):
    b = _baseline().model_copy(update={"baseline_version": 999})
    (tmp_path / "main.json").write_text(b.model_dump_json())
    with pytest.raises(BaselineCorruptError, match="999"):
        load_baseline("main", base_dir=tmp_path)


def test_schema_violation_raises(tmp_path):
    (tmp_path / "main.json").write_text('{"baseline_version": 1}')
    with pytest.raises(BaselineCorruptError):
        load_baseline("main", base_dir=tmp_path)


def test_question_set_hash_is_stable_and_content_sensitive(tmp_path):
    q = tmp_path / "questions.yaml"
    q.write_text("questions: [a]")
    h1 = question_set_hash(q)
    assert h1 == question_set_hash(q)
    q.write_text("questions: [a, b]")
    assert question_set_hash(q) != h1


def test_baseline_path_maps_name(tmp_path):
    assert baseline_path("bm25", base_dir=tmp_path) == tmp_path / "bm25.json"
