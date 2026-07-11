import pytest

from rag_pipeline.eval.questions import load_questions


VALID_YAML = """
dataset:
  name: benchmark-v1
  version: "1.0.0"

questions:
  - id: q001
    question: "What is X?"
    category: factual
    expected:
      answer: "X is a thing."
      citation_doc_ids: ["d1"]
  - id: q002
    question: "How do A and B differ?"
    category: comparative
    expected:
      answer: "A differs from B in..."
      citation_doc_ids: ["d1", "d2"]
      acceptable_answers: ["A differs from B because..."]
      difficulty: medium
"""


def test_load_questions_parses_dataset_and_questions(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text(VALID_YAML)

    dataset, questions = load_questions(path)

    assert dataset.name == "benchmark-v1"
    assert dataset.version == "1.0.0"
    assert len(questions) == 2
    assert questions[0].id == "q001"
    assert questions[0].category == "factual"
    assert questions[0].expected.answer == "X is a thing."
    assert questions[0].expected.citation_doc_ids == ["d1"]


def test_load_questions_tolerates_reserved_unused_fields(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text(VALID_YAML)

    _, questions = load_questions(path)

    # q002 sets acceptable_answers/difficulty -- Phase 1 ignores them but
    # must not error on their presence.
    assert questions[1].id == "q002"


def test_load_questions_rejects_unknown_category(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("""
dataset:
  name: benchmark-v1
  version: "1.0.0"
questions:
  - id: q001
    question: "What is X?"
    category: not-a-real-category
    expected:
      answer: "X."
      citation_doc_ids: ["d1"]
""")

    with pytest.raises(ValueError, match="not-a-real-category"):
        load_questions(path)


def test_load_questions_rejects_missing_required_field(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("""
dataset:
  name: benchmark-v1
  version: "1.0.0"
questions:
  - id: q001
    category: factual
    expected:
      answer: "X."
      citation_doc_ids: ["d1"]
""")

    with pytest.raises(ValueError, match="question"):
        load_questions(path)


def test_load_questions_rejects_empty_yaml(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("")

    with pytest.raises(ValueError, match="empty or not a valid YAML mapping"):
        load_questions(path)


def test_load_questions_rejects_non_dict_yaml(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("- item1\n- item2\n")

    with pytest.raises(ValueError, match="empty or not a valid YAML mapping"):
        load_questions(path)


def test_load_questions_rejects_missing_dataset_key(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("""
questions:
  - id: q001
    question: "What is X?"
    category: factual
    expected:
      answer: "X."
      citation_doc_ids: ["d1"]
""")

    with pytest.raises(ValueError, match="dataset"):
        load_questions(path)


def test_load_questions_rejects_missing_questions_key(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("""
dataset:
  name: benchmark-v1
  version: "1.0.0"
""")

    with pytest.raises(ValueError, match="questions"):
        load_questions(path)


def test_load_questions_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("""
dataset:
  name: benchmark-v1
  version: "1.0.0"
questions:
  - id: q001
    question: "What is X?"
    category: factual
    expected:
      answer: "X."
      citation_doc_ids: ["d1"]
  - id: q001
    question: "What is Y?"
    category: comparative
    expected:
      answer: "Y."
      citation_doc_ids: ["d2"]
""")

    with pytest.raises(ValueError, match="duplicate.*q001"):
        load_questions(path)
