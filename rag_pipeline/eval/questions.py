from dataclasses import dataclass
from pathlib import Path

import yaml

_VALID_CATEGORIES = {"factual", "comparative", "multi-hop", "summarization", "definition"}
_REQUIRED_QUESTION_FIELDS = ("id", "question", "category", "expected")
_REQUIRED_EXPECTED_FIELDS = ("answer", "citation_doc_ids")


@dataclass
class ExpectedAnswer:
    answer: str
    citation_doc_ids: list[str]


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: str
    expected: ExpectedAnswer


@dataclass
class Dataset:
    name: str
    version: str


def load_questions(path: str | Path) -> tuple[Dataset, list[EvalQuestion]]:
    raw = yaml.safe_load(Path(path).read_text())

    dataset_raw = raw["dataset"]
    dataset = Dataset(name=dataset_raw["name"], version=dataset_raw["version"])

    questions = [_parse_question(q) for q in raw["questions"]]
    return dataset, questions


def _parse_question(raw: dict) -> EvalQuestion:
    for field in _REQUIRED_QUESTION_FIELDS:
        if field not in raw:
            raise ValueError(f"question entry missing required field: {field!r} (entry: {raw})")

    category = raw["category"]
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"unknown category {category!r}; must be one of {sorted(_VALID_CATEGORIES)}")

    expected_raw = raw["expected"]
    for field in _REQUIRED_EXPECTED_FIELDS:
        if field not in expected_raw:
            raise ValueError(f"expected block missing required field: {field!r} (question id: {raw['id']})")

    expected = ExpectedAnswer(
        answer=expected_raw["answer"],
        citation_doc_ids=expected_raw["citation_doc_ids"],
    )
    return EvalQuestion(id=raw["id"], question=raw["question"], category=category, expected=expected)
