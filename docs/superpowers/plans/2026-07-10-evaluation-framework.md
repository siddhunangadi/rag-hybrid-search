# Evaluation Framework (Phase 1: Static Benchmark) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static evaluation harness that runs a fixed question set through the real `RagPipeline`, scores objective and judge-derived metrics separately, and writes a self-describing `report.json` + `report.html`.

**Architecture:** A new `rag_pipeline/eval/` package (question loading, judge, metrics, report) driven by a standalone script `scripts/run_eval.py` that builds the real pipeline via `api/dependencies.build_container()` and runs questions in-process — no server, no new runtime dependencies.

**Tech Stack:** Python, pytest, existing `GenerationProvider`/`RagPipeline`/`RequestTrace` abstractions, PyYAML (already a dependency via existing config loading — verify in Task 2).

## Global Constraints

- No single weighted "overall score" anywhere in the report — every metric (objective and subjective) stays independent.
- The judge provider defaults to the generation provider but is a separate, swappable seam (`EvalConfig.judge_provider`).
- Report metadata's `settings` field is the full `Settings.model_dump()` with secrets stripped: `nvidia_api_key`, `gemini_api_key`, `debug_token` must never appear in `report.json`.
- Per-question failures are structured as `status` (`"success"|"error"`) + `error_type` + `error_message` — never a bare `error: <string|null>` field.
- `report_version` is a top-level field bumped only when `report.json`'s shape changes; `dataset.version` (question set) and `package_version` (installed package) are separate fields and must not be conflated.
- Retrieval data (chunk ids, latencies) is recorded per question but never scored — it has no pass/fail metric attached.
- Judge verdicts are exactly `CORRECT | PARTIAL | INCORRECT | UNSUPPORTED`, scored `1.0 | 0.5 | 0.0 | 0.0` for aggregate accuracy, with `UNSUPPORTED` also reported as its own hallucination-rate metric.

---

## File Structure

- `rag_hybrid_search/trace.py` — modify: expose `RequestTrace.data` property (read-only view of the finalized trace dict).
- `rag_pipeline/rag_pipeline.py` — modify: `RagPipeline.answer()` accepts an optional `dev_trace` param so callers can inject and later read a trace.
- `rag_pipeline/eval/__init__.py` — new, empty (package marker).
- `rag_pipeline/eval/questions.py` — new: loads and validates `eval/questions.yaml` into typed objects.
- `rag_pipeline/eval/judge.py` — new: calls the judge provider, parses its verdict.
- `rag_pipeline/eval/metrics.py` — new: objective metrics, retrieval record extraction, evaluation record assembly.
- `rag_pipeline/eval/report.py` — new: aggregate/per-category summary, `report.json` + `report.html` writers.
- `scripts/run_eval.py` — new: driver script wiring everything together.
- `eval/questions.yaml` — new: starter question set (grounded in the one document currently ingested in this repo's `data/` corpus).
- Tests: `tests/rag_pipeline/eval/test_questions.py`, `test_judge.py`, `test_metrics.py`, `test_report.py`, `tests/rag_pipeline/test_rag_pipeline.py` (trace injection), `tests/test_run_eval.py` (end-to-end).

---

### Task 1: Inject an optional `dev_trace` into `RagPipeline.answer()` and expose trace data

**Files:**
- Modify: `rag_hybrid_search/trace.py`
- Modify: `rag_pipeline/rag_pipeline.py:265-270` (the `answer()` signature and its `dev_trace = RequestTrace(...)` line)
- Test: `tests/rag_pipeline/test_rag_pipeline.py`

**Interfaces:**
- Produces: `RequestTrace.data -> dict` (property); `RagPipeline.answer(question, max_chunks=5, verify=True, dev_trace: RequestTrace | None = None) -> RagAnswer`.
- Consumes (Task 4/6): callers construct their own `RequestTrace(question, {...})`, pass it in, call `.answer(...)`, then read `dev_trace.data` afterward — `finish()` is still called internally by `answer()`, so the dict is fully populated (including `timings_ms`) by the time `answer()` returns.

This is the only way to get the retrieval record (chunk ids, per-stage latencies) into the eval harness without adding new instrumentation — `RequestTrace` already collects everything the design's `retrieval` block needs; it just isn't reachable from outside `answer()` today.

- [ ] **Step 1: Write failing test for `RequestTrace.data`**

Add to `tests/rag_pipeline/test_rag_pipeline.py`:

```python
def test_answer_accepts_injected_dev_trace_and_exposes_its_data():
    from rag_hybrid_search.trace import RequestTrace

    chunks = [make_retrieved_chunk("c1", "Some evidence text.")]
    retriever = FakeRetriever(chunks)
    provider = MockProvider()
    pipeline = RagPipeline(retriever, provider)

    trace = RequestTrace("What is X?", {"Generation": "MockProvider"})
    result = pipeline.answer("What is X?", dev_trace=trace)

    assert result.error is None
    assert trace.data["question"] == "What is X?"
    assert "timings_ms" in trace.data
    assert trace.data["summary"]["chunks_used"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py::test_answer_accepts_injected_dev_trace_and_exposes_its_data -v`
Expected: FAIL with `TypeError: answer() got an unexpected keyword argument 'dev_trace'`.

- [ ] **Step 3: Add the `data` property to `RequestTrace`**

In `rag_hybrid_search/trace.py`, add this property anywhere inside the `RequestTrace` class body (alongside the other methods, e.g. right after `__init__`):

```python
    @property
    def data(self) -> dict:
        """Read-only view of the trace's collected data. Populated
        incrementally by each ``log_*`` call and finalized (timings,
        runtime info) once ``finish()`` runs."""
        return self._data
```

- [ ] **Step 4: Accept an injected `dev_trace` in `answer()`**

In `rag_pipeline/rag_pipeline.py`, change:

```python
    def answer(self, question: str, max_chunks: int = 5, verify: bool = True) -> RagAnswer:
        dev_trace = RequestTrace(question, {
            "Generation": type(self._generation_provider).__name__,
            "Max Chunks": max_chunks,
            "Verify": verify,
            "Prompt Version": self._prompt_version,
        })
```

to:

```python
    def answer(
        self, question: str, max_chunks: int = 5, verify: bool = True,
        dev_trace: RequestTrace | None = None,
    ) -> RagAnswer:
        dev_trace = dev_trace or RequestTrace(question, {
            "Generation": type(self._generation_provider).__name__,
            "Max Chunks": max_chunks,
            "Verify": verify,
            "Prompt Version": self._prompt_version,
        })
```

(`RequestTrace` is already imported in this file — no new import needed.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/test_rag_pipeline.py::test_answer_accepts_injected_dev_trace_and_exposes_its_data -v`
Expected: PASS

- [ ] **Step 6: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures (this is a purely additive, default-preserving change — every existing call site omits `dev_trace` and behaves exactly as before).

- [ ] **Step 7: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_hybrid_search/trace.py rag_pipeline/rag_pipeline.py tests/rag_pipeline/test_rag_pipeline.py
git commit -m "$(cat <<'EOF'
feat: allow injecting a dev_trace into RagPipeline.answer()

The evaluation harness needs the full RequestTrace (chunk ids,
per-stage latencies) for its retrieval record without adding new
instrumentation -- RequestTrace already collects everything, it just
wasn't reachable from outside answer(). dev_trace defaults to None
(unchanged behavior for every existing call site); RequestTrace.data
exposes the finalized dict once answer() returns.
EOF
)"
```

---

### Task 2: `eval/questions.yaml` loader and schema

**Files:**
- Create: `rag_pipeline/eval/__init__.py`
- Create: `rag_pipeline/eval/questions.py`
- Test: `tests/rag_pipeline/eval/__init__.py`, `tests/rag_pipeline/eval/test_questions.py`

**Interfaces:**
- Produces: `ExpectedAnswer` (dataclass: `answer: str`, `citation_doc_ids: list[str]`), `EvalQuestion` (dataclass: `id: str`, `question: str`, `category: str`, `expected: ExpectedAnswer`), `Dataset` (dataclass: `name: str`, `version: str`), `load_questions(path: str | Path) -> tuple[Dataset, list[EvalQuestion]]`.
- Consumes (Task 6): `scripts/run_eval.py` calls `load_questions("eval/questions.yaml")`.

Valid `category` values: `factual`, `comparative`, `multi-hop`, `summarization`, `definition`. The `expected` block only requires `answer` and `citation_doc_ids` in Phase 1; `acceptable_answers`, `minimum_claims`, `difficulty` are reserved for later and must be silently ignored if present (never an error) so the schema doesn't need to change when they're adopted.

- [ ] **Step 1: Create the package marker files**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
touch rag_pipeline/eval/__init__.py
mkdir -p tests/rag_pipeline/eval
touch tests/rag_pipeline/eval/__init__.py
```

- [ ] **Step 2: Write failing tests for `load_questions`**

Create `tests/rag_pipeline/eval/test_questions.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/eval/test_questions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.eval.questions'`.

- [ ] **Step 4: Implement `rag_pipeline/eval/questions.py`**

```python
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
```

- [ ] **Step 5: Confirm PyYAML is available**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run python -c "import yaml; print(yaml.__version__)"`
Expected: prints a version number (PyYAML is already pulled in transitively by this project's dependencies). If this fails with `ModuleNotFoundError`, add it explicitly: `uv add pyyaml` before continuing.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/eval/test_questions.py -v`
Expected: all 4 PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/eval/__init__.py rag_pipeline/eval/questions.py tests/rag_pipeline/eval/__init__.py tests/rag_pipeline/eval/test_questions.py
git commit -m "$(cat <<'EOF'
feat: add eval/questions.yaml loader and schema validation

load_questions() parses the dataset block (name/version) and each
question's id/question/category/expected.{answer,citation_doc_ids},
rejecting unknown categories and missing required fields. Reserved
fields (acceptable_answers, minimum_claims, difficulty) are tolerated
but unused, per the design spec's forward-compatible schema.
EOF
)"
```

---

### Task 3: LLM judge

**Files:**
- Create: `rag_pipeline/eval/judge.py`
- Test: `tests/rag_pipeline/eval/test_judge.py`

**Interfaces:**
- Consumes: `GenerationProvider.generate(prompt: str, **kwargs) -> str` (existing protocol in `rag_pipeline/generation_provider.py`).
- Produces: `JudgeVerdict` (dataclass: `verdict: str`, `reasoning: str`, `prompt: str`, `raw_response: str`); `judge_answer(question: str, expected_answer: str, model_answer: str, judge_provider: GenerationProvider, prompt_version: str = "v1") -> JudgeVerdict`.
- Consumes (Task 4): `metrics.py` calls `judge_answer(...)` and reads `.verdict` for scoring, embeds the whole `JudgeVerdict` in the evaluation record's `judge` block.

Valid `verdict` values: `CORRECT`, `PARTIAL`, `INCORRECT`, `UNSUPPORTED`. On any parse failure (malformed JSON, missing `verdict` key, or an unrecognized verdict string), return `verdict="INCORRECT"` with `reasoning` explaining the parse failure — never raise, since one bad judge call must not abort the whole eval run (mirrors the existing "never raise out of the pipeline" philosophy in `query_decomposer.py`).

- [ ] **Step 1: Create the test file and write failing tests**

Create `tests/rag_pipeline/eval/test_judge.py`:

```python
import json

from rag_pipeline.eval.judge import judge_answer


class CannedJudgeProvider:
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt, **kwargs):
        return self._response


def test_judge_answer_parses_valid_verdict():
    response = json.dumps({"verdict": "CORRECT", "reasoning": "Matches the gold answer."})
    provider = CannedJudgeProvider(response)

    result = judge_answer("What is X?", "X is a thing.", "X is a thing.", provider)

    assert result.verdict == "CORRECT"
    assert result.reasoning == "Matches the gold answer."
    assert result.raw_response == response
    assert "What is X?" in result.prompt


def test_judge_answer_accepts_all_valid_verdicts():
    for verdict in ("CORRECT", "PARTIAL", "INCORRECT", "UNSUPPORTED"):
        response = json.dumps({"verdict": verdict, "reasoning": "..."})
        result = judge_answer("Q", "gold", "model", CannedJudgeProvider(response))
        assert result.verdict == verdict


def test_judge_answer_falls_back_to_incorrect_on_malformed_json():
    provider = CannedJudgeProvider("not valid json at all")

    result = judge_answer("What is X?", "X is a thing.", "garbage", provider)

    assert result.verdict == "INCORRECT"
    assert "parse" in result.reasoning.lower()
    assert result.raw_response == "not valid json at all"


def test_judge_answer_falls_back_to_incorrect_on_unrecognized_verdict():
    response = json.dumps({"verdict": "MAYBE", "reasoning": "..."})
    provider = CannedJudgeProvider(response)

    result = judge_answer("Q", "gold", "model", provider)

    assert result.verdict == "INCORRECT"
    assert "MAYBE" in result.reasoning


def test_judge_answer_falls_back_to_incorrect_on_missing_verdict_key():
    response = json.dumps({"reasoning": "no verdict field here"})
    provider = CannedJudgeProvider(response)

    result = judge_answer("Q", "gold", "model", provider)

    assert result.verdict == "INCORRECT"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/eval/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.eval.judge'`.

- [ ] **Step 3: Implement `rag_pipeline/eval/judge.py`**

```python
import json
from dataclasses import dataclass

from rag_pipeline.generation_provider import GenerationProvider

_VALID_VERDICTS = {"CORRECT", "PARTIAL", "INCORRECT", "UNSUPPORTED"}

_JUDGE_PROMPT_TEMPLATE = """You are grading a RAG system's answer against a gold reference answer.

Question: {question}

Gold reference answer: {expected_answer}

System's answer: {model_answer}

Score the system's answer as one of:
- CORRECT: matches the gold answer's meaning (paraphrasing is fine).
- PARTIAL: partially correct, missing some aspect of the gold answer.
- INCORRECT: wrong, contradicts the gold answer.
- UNSUPPORTED: the system's answer contains a claim not grounded in the
  gold reference or the question (e.g. a hallucinated fact), regardless
  of whether it happens to be plausible-sounding.

Respond with ONLY a JSON object: {{"verdict": "<one of the four above>", "reasoning": "<one sentence>"}}
"""


@dataclass
class JudgeVerdict:
    verdict: str
    reasoning: str
    prompt: str
    raw_response: str


def judge_answer(
    question: str, expected_answer: str, model_answer: str,
    judge_provider: GenerationProvider, prompt_version: str = "v1",
) -> JudgeVerdict:
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question, expected_answer=expected_answer, model_answer=model_answer,
    )
    raw_response = judge_provider.generate(prompt)

    try:
        parsed = json.loads(raw_response)
        verdict = parsed["verdict"]
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, TypeError):
        return JudgeVerdict(
            verdict="INCORRECT", reasoning="Judge response failed to parse as JSON.",
            prompt=prompt, raw_response=raw_response,
        )

    if verdict not in _VALID_VERDICTS:
        return JudgeVerdict(
            verdict="INCORRECT",
            reasoning=f"Judge returned an unrecognized verdict {verdict!r}.",
            prompt=prompt, raw_response=raw_response,
        )

    return JudgeVerdict(verdict=verdict, reasoning=reasoning, prompt=prompt, raw_response=raw_response)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/eval/test_judge.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/eval/judge.py tests/rag_pipeline/eval/test_judge.py
git commit -m "$(cat <<'EOF'
feat: add LLM judge for evaluation harness

judge_answer() scores a candidate answer against a gold reference as
CORRECT/PARTIAL/INCORRECT/UNSUPPORTED, never raising -- any parse
failure or unrecognized verdict degrades to INCORRECT with an
explanatory reasoning string, so one bad judge call can't abort an
eval run. Full prompt and raw response are always retained.
EOF
)"
```

---

### Task 4: Objective metrics, retrieval record, and evaluation record assembly

**Files:**
- Create: `rag_pipeline/eval/metrics.py`
- Test: `tests/rag_pipeline/eval/test_metrics.py`

**Interfaces:**
- Consumes: `RagAnswer` (`rag_pipeline/models.py`), `RequestTrace.data` (Task 1), `EvalQuestion`/`ExpectedAnswer` (Task 2), `JudgeVerdict`/`judge_answer` (Task 3).
- Produces:
  - `citation_precision_recall_f1(predicted: list[str], expected: list[str]) -> tuple[float, float, float]`
  - `verdict_score(verdict: str) -> float`
  - `build_retrieval_record(trace_data: dict) -> dict`
  - `evaluate_question(question: EvalQuestion, rag_answer: RagAnswer, trace_data: dict, latency_ms: float, judge_provider, judge_prompt_version: str = "v1") -> dict` — returns one fully-assembled evaluation record dict (matching the design spec's per-question JSON shape, including `status`/`error_type`/`error_message`).
- Consumes (Task 6): `scripts/run_eval.py` calls `evaluate_question(...)` once per question inside a try/except that catches pipeline-level exceptions and produces an `"error"`-status record instead.

- [ ] **Step 1: Write failing tests for citation precision/recall/F1**

Create `tests/rag_pipeline/eval/test_metrics.py`:

```python
import pytest

from rag_pipeline.eval.metrics import (
    build_retrieval_record,
    citation_precision_recall_f1,
    evaluate_question,
    verdict_score,
)
from rag_pipeline.eval.questions import EvalQuestion, ExpectedAnswer
from rag_pipeline.models import CitationStatus, ClaimResult, Claim, ConfidenceScores, RagAnswer, VerificationReport


def test_citation_precision_recall_f1_perfect_match():
    p, r, f1 = citation_precision_recall_f1(["d1", "d2"], ["d1", "d2"])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_citation_precision_recall_f1_partial_overlap():
    # predicted d1,d3; expected d1,d2 -- 1 true positive, 1 false positive, 1 false negative
    p, r, f1 = citation_precision_recall_f1(["d1", "d3"], ["d1", "d2"])
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_citation_precision_recall_f1_no_overlap():
    p, r, f1 = citation_precision_recall_f1(["d3"], ["d1"])
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_citation_precision_recall_f1_empty_expected_and_predicted():
    p, r, f1 = citation_precision_recall_f1([], [])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("verdict,expected_score", [
    ("CORRECT", 1.0), ("PARTIAL", 0.5), ("INCORRECT", 0.0), ("UNSUPPORTED", 0.0),
])
def test_verdict_score(verdict, expected_score):
    assert verdict_score(verdict) == expected_score


def test_build_retrieval_record_extracts_expected_fields():
    trace_data = {
        "dense": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        "rerank": {"selected": [{"chunk_id": "c1", "score": 0.9, "final_rank": 1}]},
        "pruning": {"before": 2, "after": 1, "dropped": ["c2"]},
        "prompt": {"chars": 500, "approx_tokens": 125},
        "summary": {"chunks_used": 1, "documents_used": 1},
        "timings_ms": {"dense_search": 10.0, "rerank": 20.0, "generation": 200.0, "total": 250.0},
    }

    record = build_retrieval_record(trace_data)

    assert record["retrieved_chunk_ids"] == ["c1", "c2"]
    assert record["reranked_chunk_ids"] == ["c1"]
    assert record["document_ids_used"] == 1
    assert record["context_size_chars"] == 500
    assert record["retrieval_latency_ms"] == 10.0
    assert record["rerank_latency_ms"] == 20.0
    assert record["generation_latency_ms"] == 200.0


def _make_rag_answer(citations, verified=True):
    verification = VerificationReport(
        total_claims=1, verified_claims=1 if verified else 0, failed_claims=0 if verified else 1,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[ClaimResult(
            claim=Claim(text="claim", citation_ids=citations), doc_ids_valid=True,
            quote_match_score=1.0, passed=verified,
        )],
    )
    return RagAnswer(
        answer="model answer", citations=citations,
        confidence=ConfidenceScores(retrieval=1.0, citations=1.0, coverage=1.0, overall=1.0),
        verification=verification, citation_status=CitationStatus.OK, error=None,
    )


class CannedJudgeProvider:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt, **kwargs):
        return self._response


def test_evaluate_question_assembles_full_record():
    import json

    question = EvalQuestion(
        id="q001", question="What is X?", category="factual",
        expected=ExpectedAnswer(answer="X is a thing.", citation_doc_ids=["d1"]),
    )
    rag_answer = _make_rag_answer(citations=["d1"])
    trace_data = {
        "dense": [{"chunk_id": "c1"}], "rerank": {"selected": [{"chunk_id": "c1"}]},
        "pruning": {"before": 1, "after": 1, "dropped": []},
        "prompt": {"chars": 100, "approx_tokens": 25},
        "summary": {"chunks_used": 1, "documents_used": 1},
        "timings_ms": {"dense_search": 5.0, "rerank": 5.0, "generation": 100.0, "total": 120.0},
    }
    judge_provider = CannedJudgeProvider(json.dumps({"verdict": "CORRECT", "reasoning": "matches"}))

    record = evaluate_question(question, rag_answer, trace_data, latency_ms=120.0, judge_provider=judge_provider)

    assert record["id"] == "q001"
    assert record["status"] == "success"
    assert record["error_type"] is None
    assert record["objective_metrics"]["citation_precision"] == 1.0
    assert record["objective_metrics"]["citation_recall"] == 1.0
    assert record["objective_metrics"]["verification_pass"] is True
    assert record["judge"]["verdict"] == "CORRECT"
    assert record["retrieval"]["retrieved_chunk_ids"] == ["c1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.eval.metrics'`.

- [ ] **Step 3: Implement `rag_pipeline/eval/metrics.py`**

```python
from rag_pipeline.eval.judge import judge_answer
from rag_pipeline.eval.questions import EvalQuestion
from rag_pipeline.models import RagAnswer

_VERDICT_SCORES = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0, "UNSUPPORTED": 0.0}


def verdict_score(verdict: str) -> float:
    return _VERDICT_SCORES[verdict]


def citation_precision_recall_f1(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    predicted_set, expected_set = set(predicted), set(expected)
    true_positives = len(predicted_set & expected_set)

    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def build_retrieval_record(trace_data: dict) -> dict:
    timings = trace_data.get("timings_ms", {})
    rerank = trace_data.get("rerank", {})
    pruning = trace_data.get("pruning", {})
    prompt = trace_data.get("prompt", {})
    summary = trace_data.get("summary", {})

    return {
        "retrieved_chunk_ids": [r["chunk_id"] for r in trace_data.get("dense", [])],
        "reranked_chunk_ids": [r["chunk_id"] for r in rerank.get("selected", [])],
        "chunks_used": summary.get("chunks_used"),
        "document_ids_used": summary.get("documents_used"),
        "context_size_chars": prompt.get("chars"),
        "pruning": pruning,
        "retrieval_latency_ms": timings.get("dense_search"),
        "rerank_latency_ms": timings.get("rerank"),
        "generation_latency_ms": timings.get("generation"),
    }


def evaluate_question(
    question: EvalQuestion, rag_answer: RagAnswer, trace_data: dict, latency_ms: float,
    judge_provider, judge_prompt_version: str = "v1",
) -> dict:
    predicted_citations = rag_answer.citations
    expected_citations = question.expected.citation_doc_ids
    precision, recall, f1 = citation_precision_recall_f1(predicted_citations, expected_citations)

    verification_pass = all(cr.passed for cr in rag_answer.verification.claim_results)

    judge_result = judge_answer(
        question.question, question.expected.answer, rag_answer.answer or "",
        judge_provider, prompt_version=judge_prompt_version,
    )

    return {
        "id": question.id,
        "question": question.question,
        "category": question.category,
        "expected": {"answer": question.expected.answer, "citation_doc_ids": expected_citations},
        "model_answer": rag_answer.answer,
        "citations": predicted_citations,
        "verification": rag_answer.verification.model_dump(),
        "confidence": rag_answer.confidence.model_dump(),
        "status": "success",
        "error_type": None,
        "error_message": None,
        "objective_metrics": {
            "latency_ms": latency_ms,
            "citation_precision": precision,
            "citation_recall": recall,
            "citation_f1": f1,
            "verification_pass": verification_pass,
            "coverage": rag_answer.confidence.coverage,
        },
        "judge": {
            "verdict": judge_result.verdict,
            "reasoning": judge_result.reasoning,
            "prompt": judge_result.prompt,
            "raw_response": judge_result.raw_response,
        },
        "retrieval": build_retrieval_record(trace_data),
    }


def error_record(question: EvalQuestion, error_type: str, error_message: str) -> dict:
    """Built by the driver (Task 6) when the pipeline call itself raises,
    so one question's failure doesn't abort the whole eval run."""
    return {
        "id": question.id,
        "question": question.question,
        "category": question.category,
        "expected": {"answer": question.expected.answer, "citation_doc_ids": question.expected.citation_doc_ids},
        "model_answer": None,
        "citations": [],
        "verification": None,
        "confidence": None,
        "status": "error",
        "error_type": error_type,
        "error_message": error_message,
        "objective_metrics": None,
        "judge": None,
        "retrieval": None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/eval/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/eval/metrics.py tests/rag_pipeline/eval/test_metrics.py
git commit -m "$(cat <<'EOF'
feat: add objective metrics and evaluation record assembly

citation_precision_recall_f1() and verdict_score() are pure functions
with no LLM dependency; build_retrieval_record() extracts chunk ids
and per-stage latencies straight from RequestTrace.data with no new
instrumentation. evaluate_question() assembles the full per-question
record the design spec calls for, including structured status/error_type
fields; error_record() covers the pipeline-exception path.
EOF
)"
```

---

### Task 5: Summary aggregation and report writers

**Files:**
- Create: `rag_pipeline/eval/report.py`
- Test: `tests/rag_pipeline/eval/test_report.py`

**Interfaces:**
- Consumes: evaluation record dicts from Task 4's `evaluate_question`/`error_record`.
- Produces: `build_summary(records: list[dict]) -> dict` (aggregate + per-category, `objective` and `subjective` kept separate); `write_report(records: list[dict], metadata: dict, out_dir: str | Path) -> tuple[Path, Path]` (writes `report.json` and `report.html`, returns their paths).
- Consumes (Task 6): `scripts/run_eval.py` calls `write_report(records, metadata, "eval/reports/<timestamp>/")`.

- [ ] **Step 1: Write failing tests for `build_summary`**

Create `tests/rag_pipeline/eval/test_report.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/eval/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.eval.report'`.

- [ ] **Step 3: Implement `rag_pipeline/eval/report.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/rag_pipeline/eval/test_report.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add rag_pipeline/eval/report.py tests/rag_pipeline/eval/test_report.py
git commit -m "$(cat <<'EOF'
feat: add eval summary aggregation and report.json/report.html writers

build_summary() computes objective and subjective metrics as two
permanently separate trees (aggregate + per-category), with no combined
or weighted overall score. write_report() persists the full record set
plus self-describing metadata to report.json and a plain static
report.html for human review.
EOF
)"
```

---

### Task 6: Driver script, starter question set, and end-to-end wiring

**Files:**
- Create: `scripts/run_eval.py`
- Create: `eval/questions.yaml`
- Test: `tests/test_run_eval.py`

**Interfaces:**
- Consumes: `load_questions` (Task 2), `evaluate_question`/`error_record` (Task 4), `write_report` (Task 5), `RequestTrace`/`RequestTrace.data` (Task 1), `api.dependencies.build_container` (existing), `RagPipeline.answer(..., dev_trace=...)` (Task 1).
- Produces: a runnable script; no other task depends on this one.

- [ ] **Step 1: Author the starter question set**

Inspect what's actually ingested in this repo's local corpus so the questions and `citation_doc_ids` are grounded in real content, not invented:

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && sqlite3 data/chunks.db "select distinct document_id from chunks;"`

This repo's `data/` currently has one ingested document (a comparative-detection-methods paper, document_id printed by the command above). Create `eval/questions.yaml` with the document_id from that command substituted in wherever `<DOCUMENT_ID>` appears below, and the `expected.answer` text adjusted to match what the corpus actually says if it differs (spot-check with `sqlite3 data/chunks.db "select text from chunks where document_id = '<DOCUMENT_ID>' limit 5;"` before finalizing each answer):

```yaml
dataset:
  name: benchmark-v1
  version: "1.0.0"

questions:
  - id: q001
    question: "What granularities does this paper study for LLM-generated code detection?"
    category: factual
    expected:
      answer: "The paper studies both function-level and class-level code detection."
      citation_doc_ids: ["<DOCUMENT_ID>"]

  - id: q002
    question: "How do function-level and class-level detection approaches differ in this study?"
    category: comparative
    expected:
      answer: "Function-level and class-level detection rely on largely disjoint structural signatures, with limited feature overlap between the two granularities."
      citation_doc_ids: ["<DOCUMENT_ID>"]

  - id: q003
    question: "What is the main goal of this paper?"
    category: summarization
    expected:
      answer: "It presents a comparative study of automatic detection of LLM-generated code across multiple contemporary models, examining both function-level and class-level granularities."
      citation_doc_ids: ["<DOCUMENT_ID>"]

  - id: q004
    question: "Define what is meant by 'function-level' detection in this paper's context."
    category: definition
    expected:
      answer: "Detection performed at the granularity of an individual function, rather than an entire class."
      citation_doc_ids: ["<DOCUMENT_ID>"]

  - id: q005
    question: "What is the capital of Mars?"
    category: factual
    expected:
      answer: "This question has no answer grounded in the ingested corpus; a correct system should decline to answer rather than invent one."
      citation_doc_ids: []
```

(`q005` is deliberately unanswerable from the corpus — it's the harness's canary for the `UNSUPPORTED` verdict path.) This is a starter set (5 questions, below the 30-50 target) sized to validate the harness end-to-end; expanding it to the full 30-50 spanning all five categories with multi-hop examples is explicitly a fast-follow, not part of this task.

- [ ] **Step 2: Write a failing end-to-end test using a fixture pipeline**

Create `tests/test_run_eval.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


def test_run_eval_end_to_end_produces_report(tmp_path):
    questions_path = tmp_path / "questions.yaml"
    questions_path.write_text("""
dataset:
  name: smoke-test
  version: "0.0.1"

questions:
  - id: q001
    question: "What is X?"
    category: factual
    expected:
      answer: "X is a thing."
      citation_doc_ids: ["d1"]
""")
    out_dir = tmp_path / "reports"

    result = subprocess.run(
        [sys.executable, "scripts/run_eval.py",
         "--questions", str(questions_path), "--out-dir", str(out_dir),
         "--fixture-pipeline"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["report_version"] == "1"
    assert report["metadata"]["dataset"]["name"] == "smoke-test"
    assert "settings" in report["metadata"]
    assert "nvidia_api_key" not in json.dumps(report["metadata"]["settings"])
    assert len(report["results"]) == 1
    assert report["results"][0]["id"] == "q001"
    assert (out_dir / "report.html").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_run_eval.py -v`
Expected: FAIL (script doesn't exist yet — `FileNotFoundError` or non-zero exit).

- [ ] **Step 4: Implement `scripts/run_eval.py`**

```python
#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rag_hybrid_search.trace import RequestTrace
from rag_pipeline.eval.metrics import error_record, evaluate_question
from rag_pipeline.eval.questions import load_questions
from rag_pipeline.eval.report import write_report
from rag_pipeline.generation_provider import MockProvider

_SECRET_SETTINGS_FIELDS = {"nvidia_api_key", "gemini_api_key", "debug_token"}


def _package_version() -> str:
    try:
        return version("rag-hybrid-search")
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sanitized_settings(settings) -> dict:
    dumped = settings.model_dump()
    return {k: v for k, v in dumped.items() if k not in _SECRET_SETTINGS_FIELDS}


def _build_pipeline(use_fixture: bool):
    """Real pipeline via the app's container, or an in-memory fixture
    pipeline for the smoke test (--fixture-pipeline avoids requiring a
    populated data/ dir and a real or mock-provider network round trip
    in CI)."""
    if use_fixture:
        from rag_hybrid_search.config import Settings
        from rag_hybrid_search.models import Chunk, RetrievalTrace, RetrievedChunk
        from rag_pipeline.rag_pipeline import RagPipeline

        class _FixtureRetriever:
            def retrieve(self, query, dev_trace=None):
                chunk = Chunk(
                    chunk_id="c1", document_id="d1", chunk_index=0, text="X is a thing.",
                    strategy_version="fixed-v1", heading=None, page=None, char_count=13,
                )
                return [RetrievedChunk(
                    chunk=chunk, dense_score=0.9, bm25_score=0.9, rrf_score=0.9,
                    rerank_score=0.9, final_rank=1,
                )], RetrievalTrace()

        provider = MockProvider(canned_json=json.dumps({
            "answer": "X is a thing [d1].",
            "claims": [{"text": "X is a thing.", "citation_ids": ["d1"], "supporting_quote": "X is a thing."}],
        }))
        pipeline = RagPipeline(_FixtureRetriever(), provider)
        return pipeline, provider, Settings()

    from api.dependencies import build_container
    container = build_container()
    return container.rag_pipeline, container.generation_provider, container.settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fixture-pipeline", action="store_true", help="Use an in-memory fixture pipeline instead of building the real one (for smoke-testing this script).")
    args = parser.parse_args()

    dataset, questions = load_questions(args.questions)
    pipeline, generation_provider, settings = _build_pipeline(args.fixture_pipeline)
    judge_provider = generation_provider  # Phase 1 default: judge = generation

    records = []
    for question in questions:
        trace = RequestTrace(question.question, {"Generation": type(generation_provider).__name__})
        started = time.perf_counter()
        try:
            rag_answer = pipeline.answer(question.question, dev_trace=trace)
            latency_ms = (time.perf_counter() - started) * 1000
            records.append(evaluate_question(question, rag_answer, trace.data, latency_ms, judge_provider))
        except Exception as e:
            records.append(error_record(question, error_type=type(e).__name__, error_message=str(e)))

    metadata = {
        "report_version": "1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "package_version": _package_version(),
        "generation_model": type(generation_provider).__name__,
        "judge_model": type(judge_provider).__name__,
        "prompt_version": getattr(pipeline, "prompt_version", "unknown"),
        "judge_prompt_version": "v1",
        "settings": _sanitized_settings(settings),
        "corpus_version": "unknown",
        "dataset": {"name": dataset.name, "version": dataset.version},
    }

    out_dir = args.out_dir or f"eval/reports/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}"
    json_path, html_path = write_report(records, metadata, out_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest tests/test_run_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Run the real script against the starter question set**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run python scripts/run_eval.py --questions eval/questions.yaml`
Expected: exits 0, prints two `Wrote ...` lines, and `eval/reports/<timestamp>/report.json` / `report.html` exist. Open `report.json` and confirm `metadata.settings` has no `nvidia_api_key`/`gemini_api_key`/`debug_token` keys, and that `q005` (the unanswerable question) scored `UNSUPPORTED` or `INCORRECT` rather than `CORRECT` — if the real pipeline instead fabricates a confident wrong answer scored `CORRECT`, note this in your task report as a real-world observation (not a defect to fix in this task; it would inform the prompt-tightening or retry work later in the roadmap).

- [ ] **Step 7: Run the full test suite**

Run: `cd /Users/siddhunangadi/Projects/rag-hybrid-search && uv run pytest -q`
Expected: no new failures.

- [ ] **Step 8: Commit**

```bash
cd /Users/siddhunangadi/Projects/rag-hybrid-search
git add scripts/run_eval.py eval/questions.yaml tests/test_run_eval.py
git commit -m "$(cat <<'EOF'
feat: add scripts/run_eval.py driver and starter question set

Wires questions.yaml loading, the real RagPipeline (via
api.dependencies.build_container), the LLM judge, and report writing
into one runnable script. --fixture-pipeline lets the smoke test and CI
exercise the full flow without a populated corpus or network calls.
Starter eval/questions.yaml has 5 questions across factual/comparative/
summarization/definition categories plus one deliberately unanswerable
question exercising the UNSUPPORTED verdict path; expanding to the full
30-50 question target spanning multi-hop examples is a fast-follow.
EOF
)"
```

---

## Deferred (not in this plan, per the design spec)

- Phase 2: baseline storage + regression comparison, CI gating.
- Expanding `eval/questions.yaml` beyond this plan's 5-question starter set to the full 30-50.
- Cost/token accounting per run.
- A second, independently-configured judge model (the `judge_provider` seam exists; wiring a second real provider is deferred).
- Using the category taxonomy for adaptive routing (separate plan).
- Multi-run variance per question.
