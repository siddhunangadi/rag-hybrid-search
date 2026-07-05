# Grounded Generation (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the grounded generation pipeline (context building, prompting, generation, citation verification, confidence scoring) for `rag-hybrid-search`, exactly as specified in `docs/superpowers/specs/2026-07-05-grounded-generation-design.md`.

**Architecture:** New `rag_pipeline/` package alongside the existing `rag_hybrid_search/` package (same repo, same `.venv`). Layered: `models.py` (data) at the base; `context_builder.py` and `prompt_builder.py` (pure transforms) next; `generation_provider.py` (protocol + Mock, reusing the existing `NvidiaProvider`) alongside; `citation_verifier.py` and `confidence_scorer.py` (pure functions consuming generation output); `rag_pipeline.py` (orchestrator) on top, depending only on `HybridRetriever` (existing) and `GenerationProvider` (interface) — never on concrete provider classes.

**Tech Stack:** Python 3.11+, pydantic v2 (matching existing `rag_hybrid_search/models.py` convention), stdlib `difflib`/`json`/`datetime`, pytest, pytest-mock (already a dev dependency).

## Global Constraints

- Python >= 3.11.
- New package `rag_pipeline/` lives at repo root alongside `rag_hybrid_search/` (not nested inside it) — separate top-level package, added to `pyproject.toml`'s package discovery.
- `GenerationProvider` is a `Protocol` (structural typing) — `rag_hybrid_search.providers.nvidia.NvidiaProvider` already implements a compatible `generate(prompt: str, **kwargs) -> str` method and satisfies it without modification or explicit inheritance.
- `ContextBuilder`'s token budget is a **character-count approximation** (`len(text) // 4`), explicitly documented as such — no tokenizer dependency.
- Citation quote matching uses `difflib.SequenceMatcher(None, quote, chunk_text).ratio()` with `QUOTE_MATCH_THRESHOLD = 0.80` (module-level constant).
- The LLM's output is a structured JSON contract (see spec's "Generation Contract" section) — the pipeline parses JSON, it does not extract claims from free-form prose.
- `RagPipeline.answer()` never raises: generation-provider exceptions and JSON-parse failures are both caught and converted into a `RagAnswer` with `error` set.
- Every new module gets a corresponding test module under `tests/rag_pipeline/` mirroring its path (e.g. `rag_pipeline/context_builder.py` → `tests/rag_pipeline/test_context_builder.py`).
- Every task ends green: `pytest` passes for the whole suite (existing 85 tests + new) before committing.
- Commit after every task with a `feat:`/`test:`/`chore:` prefix matching the change.

---

### Task 1: `rag_pipeline` package scaffolding and data models

**Files:**
- Create: `rag_pipeline/__init__.py`
- Create: `rag_pipeline/models.py`
- Modify: `pyproject.toml` (add `rag_pipeline*` to package discovery)
- Create: `tests/rag_pipeline/__init__.py`
- Test: `tests/rag_pipeline/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Claim`, `GenerationMetadata`, `RagAnswerDraft`, `ClaimResult`, `VerificationReport`, `ConfidenceScores`, `RagAnswer`, `PromptContext` (all pydantic `BaseModel`), used by every later task in this plan.

- [ ] **Step 1: Update `pyproject.toml` package discovery**

Find the `[tool.setuptools.packages.find]` section (currently `include = ["rag_hybrid_search*"]`) and update it:

```toml
[tool.setuptools.packages.find]
include = ["rag_hybrid_search*", "rag_pipeline*"]
```

- [ ] **Step 2: Create package markers**

`rag_pipeline/__init__.py`:
```python
```

`tests/rag_pipeline/__init__.py`:
```python
```

- [ ] **Step 3: Write failing tests for models**

`tests/rag_pipeline/test_models.py`:
```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag_pipeline.models import (
    Claim,
    ClaimResult,
    ConfidenceScores,
    GenerationMetadata,
    PromptContext,
    RagAnswer,
    RagAnswerDraft,
    VerificationReport,
)


def test_claim_roundtrip():
    claim = Claim(
        text="Employees get 20 days of paid leave.",
        citation_ids=["d1"],
        supporting_quote="20 days of paid annual leave",
    )
    assert claim.citation_ids == ["d1"]


def test_generation_metadata_roundtrip():
    metadata = GenerationMetadata(
        provider="mock",
        model="mock-v1",
        prompt_version="v1",
        generated_at=datetime.now(timezone.utc),
    )
    assert metadata.prompt_version == "v1"


def test_rag_answer_draft_roundtrip():
    metadata = GenerationMetadata(
        provider="mock",
        model="mock-v1",
        prompt_version="v1",
        generated_at=datetime.now(timezone.utc),
    )
    claim = Claim(text="x", citation_ids=["d1"], supporting_quote="x")
    draft = RagAnswerDraft(answer="Answer [d1].", claims=[claim], metadata=metadata)
    assert draft.claims[0].citation_ids == ["d1"]


def test_claim_result_and_verification_report():
    claim = Claim(text="x", citation_ids=["d1"], supporting_quote="x")
    result = ClaimResult(
        claim=claim, doc_ids_valid=True, quote_match_score=0.95, passed=True
    )
    report = VerificationReport(
        total_claims=1,
        verified_claims=1,
        failed_claims=0,
        hallucinated_doc_ids=[],
        missing_quotes=[],
        claim_results=[result],
    )
    assert report.verified_claims == 1
    assert report.claim_results[0].passed is True


def test_confidence_scores_roundtrip():
    scores = ConfidenceScores(retrieval=0.9, citations=1.0, coverage=0.8, overall=0.92)
    assert scores.overall == 0.92


def test_rag_answer_roundtrip():
    report = VerificationReport(
        total_claims=0,
        verified_claims=0,
        failed_claims=0,
        hallucinated_doc_ids=[],
        missing_quotes=[],
        claim_results=[],
    )
    scores = ConfidenceScores(retrieval=0.0, citations=0.0, coverage=0.0, overall=0.0)
    answer = RagAnswer(
        answer=None,
        citations=[],
        confidence=scores,
        verification=report,
        error="provider unavailable",
    )
    assert answer.error == "provider unavailable"


def test_prompt_context_roundtrip():
    context = PromptContext(text="[d1] some text", doc_id_map={"d1": "chunk-uuid-1"})
    assert context.doc_id_map["d1"] == "chunk-uuid-1"


def test_claim_requires_citation_ids_list():
    with pytest.raises(ValidationError):
        Claim(text="x", citation_ids="not-a-list", supporting_quote="x")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/rag_pipeline/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.models'`

- [ ] **Step 5: Implement `rag_pipeline/models.py`**

```python
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Claim(BaseModel):
    text: str
    citation_ids: list[str]
    supporting_quote: str


class GenerationMetadata(BaseModel):
    provider: str
    model: str
    prompt_version: str
    generated_at: datetime


class RagAnswerDraft(BaseModel):
    answer: str
    claims: list[Claim]
    metadata: GenerationMetadata


class ClaimResult(BaseModel):
    claim: Claim
    doc_ids_valid: bool
    quote_match_score: float
    passed: bool


class VerificationReport(BaseModel):
    total_claims: int
    verified_claims: int
    failed_claims: int
    hallucinated_doc_ids: list[str]
    missing_quotes: list[str]
    claim_results: list[ClaimResult]


class ConfidenceScores(BaseModel):
    retrieval: float
    citations: float
    coverage: float
    overall: float


class RagAnswer(BaseModel):
    answer: Optional[str]
    citations: list[str]
    confidence: ConfidenceScores
    verification: VerificationReport
    error: Optional[str] = None


class PromptContext(BaseModel):
    text: str
    doc_id_map: dict[str, str]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/rag_pipeline/test_models.py -v`
Expected: PASS (8 tests)

- [ ] **Step 7: Reinstall package (picks up new `rag_pipeline` package) and run full suite**

Run:
```bash
pip install -e ".[dev]"
pytest -q
```
Expected: all prior tests (85) plus 8 new pass, 93 total.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml rag_pipeline/__init__.py rag_pipeline/models.py tests/rag_pipeline/__init__.py tests/rag_pipeline/test_models.py
git commit -m "feat: add rag_pipeline package and generation data models"
```

---

### Task 2: `ContextBuilder`

**Files:**
- Create: `rag_pipeline/context_builder.py`
- Test: `tests/rag_pipeline/test_context_builder.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from `rag_hybrid_search.models`, `PromptContext` from `rag_pipeline.models` (Task 1).
- Produces: `build_context(chunks: list[RetrievedChunk], approx_token_budget: int = 2000) -> PromptContext`, used by `rag_pipeline.py` (Task 7) and `prompt_builder.py` tests (Task 3).

- [ ] **Step 1: Write failing tests**

`tests/rag_pipeline/test_context_builder.py`:
```python
from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_pipeline.context_builder import build_context


def make_retrieved_chunk(chunk_id, text, final_rank):
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="d1",
        chunk_index=0,
        text=text,
        strategy_version="fixed-v1",
        heading=None,
        page=None,
        char_count=len(text),
    )
    return RetrievedChunk(
        chunk=chunk,
        dense_score=0.9,
        bm25_score=0.9,
        rrf_score=0.5,
        rerank_score=0.8,
        final_rank=final_rank,
    )


def test_empty_context():
    context = build_context([])
    assert context.text == ""
    assert context.doc_id_map == {}


def test_numbers_chunks_in_rank_order():
    chunks = [
        make_retrieved_chunk("c1", "first chunk text", final_rank=1),
        make_retrieved_chunk("c2", "second chunk text", final_rank=2),
    ]
    context = build_context(chunks)
    assert "[d1]" in context.text
    assert "[d2]" in context.text
    assert context.text.index("[d1]") < context.text.index("[d2]")
    assert context.doc_id_map == {"d1": "c1", "d2": "c2"}


def test_deduplicates_by_chunk_id():
    chunk = make_retrieved_chunk("c1", "same chunk", final_rank=1)
    context = build_context([chunk, chunk])
    assert len(context.doc_id_map) == 1


def test_truncates_lowest_ranked_chunks_first_without_splitting_text():
    big_text = "word " * 400  # ~2000 chars, ~500 approx tokens
    chunks = [
        make_retrieved_chunk("c1", big_text, final_rank=1),
        make_retrieved_chunk("c2", big_text, final_rank=2),
        make_retrieved_chunk("c3", big_text, final_rank=3),
    ]
    # Budget only large enough for ~1 chunk (500 tokens * 4 chars/token = 2000 chars)
    context = build_context(chunks, approx_token_budget=500)
    assert "[d1]" in context.text
    assert "[d3]" not in context.text
    assert big_text.strip() in context.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag_pipeline/test_context_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.context_builder'`

- [ ] **Step 3: Implement `rag_pipeline/context_builder.py`**

```python
from rag_hybrid_search.models import RetrievedChunk
from rag_pipeline.models import PromptContext

_CHARS_PER_TOKEN_ESTIMATE = 4


def build_context(
    chunks: list[RetrievedChunk], approx_token_budget: int = 2000
) -> PromptContext:
    """Builds a numbered prompt context from ranked, deduplicated chunks.

    approx_token_budget is estimated from character count
    (len(text) // CHARS_PER_TOKEN_ESTIMATE) -- an approximation, not an
    exact tokenizer count. If the budget would be exceeded, the
    lowest-ranked chunks are dropped whole (never truncated mid-text) so
    every included chunk stays intact and citable.
    """
    char_budget = approx_token_budget * _CHARS_PER_TOKEN_ESTIMATE

    seen_chunk_ids: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for retrieved in sorted(chunks, key=lambda r: r.final_rank):
        if retrieved.chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(retrieved.chunk.chunk_id)
        deduped.append(retrieved)

    included: list[RetrievedChunk] = []
    used_chars = 0
    for retrieved in deduped:
        chunk_chars = len(retrieved.chunk.text)
        if included and used_chars + chunk_chars > char_budget:
            break
        included.append(retrieved)
        used_chars += chunk_chars

    doc_id_map: dict[str, str] = {}
    lines: list[str] = []
    for i, retrieved in enumerate(included, start=1):
        doc_id = f"d{i}"
        doc_id_map[doc_id] = retrieved.chunk.chunk_id
        lines.append(f"[{doc_id}]\n{retrieved.chunk.text.strip()}")

    return PromptContext(text="\n\n".join(lines), doc_id_map=doc_id_map)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag_pipeline/test_context_builder.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full suite and commit**

```bash
pytest -q
git add rag_pipeline/context_builder.py tests/rag_pipeline/test_context_builder.py
git commit -m "feat: add ContextBuilder for ranked, budgeted prompt context"
```

---

### Task 3: `PromptBuilder`

**Files:**
- Create: `rag_pipeline/prompt_builder.py`
- Test: `tests/rag_pipeline/test_prompt_builder.py`

**Interfaces:**
- Consumes: `PromptContext` from `rag_pipeline.models` (Task 1).
- Produces: `build_prompt(question: str, context: PromptContext, prompt_version: str = "v1") -> str`, used by `rag_pipeline.py` (Task 7).

- [ ] **Step 1: Write failing tests, including a prompt snapshot test**

`tests/rag_pipeline/test_prompt_builder.py`:
```python
from rag_pipeline.models import PromptContext
from rag_pipeline.prompt_builder import build_prompt

_EXPECTED_V1_PROMPT = """You are a retrieval assistant. Only answer using the CONTEXT below.

Rules:
- Cite every factual claim inline using its bracketed id, e.g. [d1].
- Never invent a citation id that is not present in the CONTEXT.
- If the CONTEXT does not support an answer, say you don't know.
- Respond ONLY with JSON matching this shape, no prose wrapper, no markdown fences:
  {"answer": "...", "claims": [{"text": "...", "citation_ids": ["d1"], "supporting_quote": "..."}]}

CONTEXT:
[d1]
Employees get 20 days of paid leave.

QUESTION:
How many days of paid leave do employees get?"""


def test_prompt_contains_system_instructions():
    context = PromptContext(text="[d1]\nsome fact", doc_id_map={"d1": "c1"})
    prompt = build_prompt("What is the fact?", context)
    assert "Only answer using the CONTEXT" in prompt
    assert "Never invent a citation id" in prompt


def test_prompt_contains_numbered_context():
    context = PromptContext(text="[d1]\nsome fact", doc_id_map={"d1": "c1"})
    prompt = build_prompt("What is the fact?", context)
    assert "[d1]" in prompt
    assert "some fact" in prompt


def test_prompt_handles_empty_context():
    context = PromptContext(text="", doc_id_map={})
    prompt = build_prompt("What is the fact?", context)
    assert "QUESTION:" in prompt
    assert "What is the fact?" in prompt


def test_prompt_snapshot_v1():
    context = PromptContext(
        text="[d1]\nEmployees get 20 days of paid leave.", doc_id_map={"d1": "c1"}
    )
    prompt = build_prompt(
        "How many days of paid leave do employees get?", context, prompt_version="v1"
    )
    assert prompt == _EXPECTED_V1_PROMPT


def test_unknown_prompt_version_raises():
    context = PromptContext(text="", doc_id_map={})
    try:
        build_prompt("q", context, prompt_version="v99")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "v99" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag_pipeline/test_prompt_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.prompt_builder'`

- [ ] **Step 3: Implement `rag_pipeline/prompt_builder.py`**

```python
from rag_pipeline.models import PromptContext

_PROMPT_V1 = """You are a retrieval assistant. Only answer using the CONTEXT below.

Rules:
- Cite every factual claim inline using its bracketed id, e.g. [d1].
- Never invent a citation id that is not present in the CONTEXT.
- If the CONTEXT does not support an answer, say you don't know.
- Respond ONLY with JSON matching this shape, no prose wrapper, no markdown fences:
  {{"answer": "...", "claims": [{{"text": "...", "citation_ids": ["d1"], "supporting_quote": "..."}}]}}

CONTEXT:
{context}

QUESTION:
{question}"""

_PROMPT_TEMPLATES = {"v1": _PROMPT_V1}


def build_prompt(
    question: str, context: PromptContext, prompt_version: str = "v1"
) -> str:
    if prompt_version not in _PROMPT_TEMPLATES:
        raise ValueError(f"unknown prompt_version: {prompt_version}")
    template = _PROMPT_TEMPLATES[prompt_version]
    return template.format(context=context.text, question=question)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag_pipeline/test_prompt_builder.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run full suite and commit**

```bash
pytest -q
git add rag_pipeline/prompt_builder.py tests/rag_pipeline/test_prompt_builder.py
git commit -m "feat: add versioned PromptBuilder with snapshot test"
```

---

### Task 4: `GenerationProvider` protocol, `MockProvider`, NVIDIA reuse confirmation

**Files:**
- Create: `rag_pipeline/generation_provider.py`
- Test: `tests/rag_pipeline/test_generation_provider.py`

**Interfaces:**
- Consumes: nothing new (structurally checks `rag_hybrid_search.providers.nvidia.NvidiaProvider`).
- Produces: `GenerationProvider` (`Protocol`), `MockProvider(canned_json: str | None = None)`, used by `rag_pipeline.py` (Task 7) and its tests.

- [ ] **Step 1: Write failing tests**

`tests/rag_pipeline/test_generation_provider.py`:
```python
from rag_hybrid_search.providers.nvidia import NvidiaProvider
from rag_pipeline.generation_provider import MockProvider, GenerationProvider

_DEFAULT_CANNED_JSON = (
    '{"answer": "mock answer", "claims": '
    '[{"text": "mock claim", "citation_ids": ["d1"], "supporting_quote": "mock quote"}]}'
)


def test_mock_provider_returns_default_canned_json():
    provider = MockProvider()
    result = provider.generate("any prompt")
    assert result == _DEFAULT_CANNED_JSON


def test_mock_provider_returns_custom_canned_json():
    custom = '{"answer": "custom", "claims": []}'
    provider = MockProvider(canned_json=custom)
    assert provider.generate("any prompt") == custom


def test_nvidia_provider_satisfies_generation_provider_protocol():
    provider = NvidiaProvider(api_key="test-key")
    assert isinstance(provider, GenerationProvider)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag_pipeline/test_generation_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.generation_provider'`

- [ ] **Step 3: Implement `rag_pipeline/generation_provider.py`**

```python
from typing import Protocol, runtime_checkable

_DEFAULT_CANNED_JSON = (
    '{"answer": "mock answer", "claims": '
    '[{"text": "mock claim", "citation_ids": ["d1"], "supporting_quote": "mock quote"}]}'
)


@runtime_checkable
class GenerationProvider(Protocol):
    def generate(self, prompt: str, **kwargs) -> str: ...


class MockProvider:
    def __init__(self, canned_json: str | None = None):
        self._canned_json = canned_json or _DEFAULT_CANNED_JSON

    def generate(self, prompt: str, **kwargs) -> str:
        return self._canned_json
```

Note: `GenerationProvider` is `@runtime_checkable` specifically so
`isinstance(nvidia_provider, GenerationProvider)` works in the test above —
this is the mechanism that proves `NvidiaProvider` satisfies the protocol
without any inheritance or modification to `rag_hybrid_search/providers/nvidia.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag_pipeline/test_generation_provider.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run full suite and commit**

```bash
pytest -q
git add rag_pipeline/generation_provider.py tests/rag_pipeline/test_generation_provider.py
git commit -m "feat: add GenerationProvider protocol and MockProvider"
```

---

### Task 5: `CitationVerifier`

**Files:**
- Create: `rag_pipeline/citation_verifier.py`
- Test: `tests/rag_pipeline/test_citation_verifier.py`

**Interfaces:**
- Consumes: `RagAnswerDraft`, `Claim`, `ClaimResult`, `VerificationReport` from `rag_pipeline.models`, `PromptContext` from `rag_pipeline.models` (Task 1).
- Produces: `verify_citations(draft: RagAnswerDraft, context: PromptContext) -> VerificationReport`, `QUOTE_MATCH_THRESHOLD` constant, used by `rag_pipeline.py` (Task 7) and `confidence_scorer.py` tests (Task 6).

- [ ] **Step 1: Write failing tests**

`tests/rag_pipeline/test_citation_verifier.py`:
```python
from datetime import datetime, timezone

from rag_pipeline.citation_verifier import QUOTE_MATCH_THRESHOLD, verify_citations
from rag_pipeline.models import Claim, GenerationMetadata, PromptContext, RagAnswerDraft

_METADATA = GenerationMetadata(
    provider="mock", model="mock-v1", prompt_version="v1",
    generated_at=datetime.now(timezone.utc),
)

_CONTEXT = PromptContext(
    text="[d1]\nEmployees get 20 days of paid annual leave per year.",
    doc_id_map={"d1": "chunk-1"},
)


def make_draft(claims):
    return RagAnswerDraft(answer="answer", claims=claims, metadata=_METADATA)


def test_valid_citation_and_matching_quote_passes():
    claim = Claim(
        text="Employees get 20 days leave",
        citation_ids=["d1"],
        supporting_quote="20 days of paid annual leave",
    )
    report = verify_citations(make_draft([claim]), _CONTEXT)
    assert report.total_claims == 1
    assert report.verified_claims == 1
    assert report.failed_claims == 0
    assert report.claim_results[0].passed is True
    assert report.claim_results[0].doc_ids_valid is True


def test_hallucinated_doc_id_fails():
    claim = Claim(
        text="Employees get unlimited leave",
        citation_ids=["d99"],
        supporting_quote="unlimited leave",
    )
    report = verify_citations(make_draft([claim]), _CONTEXT)
    assert report.verified_claims == 0
    assert report.failed_claims == 1
    assert "d99" in report.hallucinated_doc_ids
    assert report.claim_results[0].doc_ids_valid is False


def test_missing_quote_fails_even_with_valid_doc_id():
    claim = Claim(
        text="Employees get free lunch",
        citation_ids=["d1"],
        supporting_quote="completely unrelated text about lunch",
    )
    report = verify_citations(make_draft([claim]), _CONTEXT)
    assert report.verified_claims == 0
    assert report.failed_claims == 1
    assert len(report.missing_quotes) == 1
    assert report.claim_results[0].quote_match_score < QUOTE_MATCH_THRESHOLD


def test_multiple_claims_mixed_pass_fail():
    valid_claim = Claim(
        text="20 days leave", citation_ids=["d1"],
        supporting_quote="20 days of paid annual leave",
    )
    invalid_claim = Claim(
        text="unlimited leave", citation_ids=["d99"], supporting_quote="unlimited",
    )
    report = verify_citations(make_draft([valid_claim, invalid_claim]), _CONTEXT)
    assert report.total_claims == 2
    assert report.verified_claims == 1
    assert report.failed_claims == 1


def test_zero_claims_produces_empty_report():
    report = verify_citations(make_draft([]), _CONTEXT)
    assert report.total_claims == 0
    assert report.verified_claims == 0
    assert report.failed_claims == 0
    assert report.claim_results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag_pipeline/test_citation_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.citation_verifier'`

- [ ] **Step 3: Implement `rag_pipeline/citation_verifier.py`**

```python
from difflib import SequenceMatcher

from rag_pipeline.models import (
    ClaimResult,
    PromptContext,
    RagAnswerDraft,
    VerificationReport,
)

QUOTE_MATCH_THRESHOLD = 0.80


def verify_citations(
    draft: RagAnswerDraft, context: PromptContext
) -> VerificationReport:
    claim_results: list[ClaimResult] = []
    hallucinated_doc_ids: list[str] = []
    missing_quotes: list[str] = []

    for claim in draft.claims:
        doc_ids_valid = all(
            citation_id in context.doc_id_map for citation_id in claim.citation_ids
        )
        if not doc_ids_valid:
            for citation_id in claim.citation_ids:
                if citation_id not in context.doc_id_map:
                    hallucinated_doc_ids.append(citation_id)

        best_quote_score = 0.0
        if doc_ids_valid:
            for citation_id in claim.citation_ids:
                chunk_text = _chunk_text_for_doc_id(context, citation_id)
                score = SequenceMatcher(
                    None, claim.supporting_quote, chunk_text
                ).ratio()
                best_quote_score = max(best_quote_score, score)

        passed = doc_ids_valid and best_quote_score >= QUOTE_MATCH_THRESHOLD
        if doc_ids_valid and best_quote_score < QUOTE_MATCH_THRESHOLD:
            missing_quotes.append(claim.supporting_quote)

        claim_results.append(
            ClaimResult(
                claim=claim,
                doc_ids_valid=doc_ids_valid,
                quote_match_score=best_quote_score,
                passed=passed,
            )
        )

    verified = sum(1 for r in claim_results if r.passed)
    return VerificationReport(
        total_claims=len(claim_results),
        verified_claims=verified,
        failed_claims=len(claim_results) - verified,
        hallucinated_doc_ids=hallucinated_doc_ids,
        missing_quotes=missing_quotes,
        claim_results=claim_results,
    )


def _chunk_text_for_doc_id(context: PromptContext, doc_id: str) -> str:
    marker = f"[{doc_id}]"
    if marker not in context.text:
        return ""
    after_marker = context.text.split(marker, 1)[1]
    return after_marker.split("\n\n", 1)[0].strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag_pipeline/test_citation_verifier.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run full suite and commit**

```bash
pytest -q
git add rag_pipeline/citation_verifier.py tests/rag_pipeline/test_citation_verifier.py
git commit -m "feat: add two-stage CitationVerifier (doc-id + fuzzy quote match)"
```

---

### Task 6: `ConfidenceScorer`

**Files:**
- Create: `rag_pipeline/confidence_scorer.py`
- Test: `tests/rag_pipeline/test_confidence_scorer.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from `rag_hybrid_search.models`, `VerificationReport` from `rag_pipeline.models` (Task 1), `PromptContext` from `rag_pipeline.models`.
- Produces: `score_confidence(retrieved_chunks: list[RetrievedChunk], verification: VerificationReport, context: PromptContext) -> ConfidenceScores`, used by `rag_pipeline.py` (Task 7).

- [ ] **Step 1: Write failing tests**

`tests/rag_pipeline/test_confidence_scorer.py`:
```python
from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_pipeline.confidence_scorer import score_confidence
from rag_pipeline.models import Claim, ClaimResult, PromptContext, VerificationReport


def make_retrieved_chunk(chunk_id, rerank_score, final_rank):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text="text",
        strategy_version="fixed-v1", heading=None, page=None, char_count=4,
    )
    return RetrievedChunk(
        chunk=chunk, dense_score=0.5, bm25_score=0.5, rrf_score=0.5,
        rerank_score=rerank_score, final_rank=final_rank,
    )


def make_claim_result(citation_ids, passed):
    claim = Claim(text="x", citation_ids=citation_ids, supporting_quote="x")
    return ClaimResult(
        claim=claim, doc_ids_valid=passed, quote_match_score=1.0 if passed else 0.0,
        passed=passed,
    )


def test_all_citations_pass_gives_high_citation_score():
    chunks = [make_retrieved_chunk("c1", rerank_score=0.9, final_rank=1)]
    context = PromptContext(text="[d1]\ntext", doc_id_map={"d1": "c1"})
    report = VerificationReport(
        total_claims=1, verified_claims=1, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[make_claim_result(["d1"], passed=True)],
    )
    scores = score_confidence(chunks, report, context)
    assert scores.citations == 1.0


def test_half_citations_fail_gives_half_citation_score():
    chunks = [make_retrieved_chunk("c1", rerank_score=0.9, final_rank=1)]
    context = PromptContext(text="[d1]\ntext", doc_id_map={"d1": "c1"})
    report = VerificationReport(
        total_claims=2, verified_claims=1, failed_claims=1,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[
            make_claim_result(["d1"], passed=True),
            make_claim_result(["d1"], passed=False),
        ],
    )
    scores = score_confidence(chunks, report, context)
    assert scores.citations == 0.5


def test_zero_claims_gives_full_citation_score_no_false_penalty():
    chunks = [make_retrieved_chunk("c1", rerank_score=0.9, final_rank=1)]
    context = PromptContext(text="[d1]\ntext", doc_id_map={"d1": "c1"})
    report = VerificationReport(
        total_claims=0, verified_claims=0, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
    )
    scores = score_confidence(chunks, report, context)
    assert scores.citations == 1.0


def test_coverage_reflects_fraction_of_chunks_cited():
    chunks = [
        make_retrieved_chunk("c1", rerank_score=0.9, final_rank=1),
        make_retrieved_chunk("c2", rerank_score=0.8, final_rank=2),
    ]
    context = PromptContext(text="[d1]\ntext\n\n[d2]\ntext", doc_id_map={"d1": "c1", "d2": "c2"})
    report = VerificationReport(
        total_claims=1, verified_claims=1, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[make_claim_result(["d1"], passed=True)],
    )
    scores = score_confidence(chunks, report, context)
    assert scores.coverage == 0.5


def test_overall_is_weighted_combination():
    chunks = [make_retrieved_chunk("c1", rerank_score=1.0, final_rank=1)]
    context = PromptContext(text="[d1]\ntext", doc_id_map={"d1": "c1"})
    report = VerificationReport(
        total_claims=1, verified_claims=1, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[make_claim_result(["d1"], passed=True)],
    )
    scores = score_confidence(chunks, report, context)
    # retrieval=1.0 (normalized top rerank score), citations=1.0, coverage=1.0
    assert scores.overall == 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 1.0


def test_empty_retrieved_chunks_gives_zero_retrieval_score():
    context = PromptContext(text="", doc_id_map={})
    report = VerificationReport(
        total_claims=0, verified_claims=0, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
    )
    scores = score_confidence([], report, context)
    assert scores.retrieval == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag_pipeline/test_confidence_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.confidence_scorer'`

- [ ] **Step 3: Implement `rag_pipeline/confidence_scorer.py`**

```python
from rag_hybrid_search.models import RetrievedChunk
from rag_pipeline.models import ConfidenceScores, PromptContext, VerificationReport

RETRIEVAL_WEIGHT = 0.4
CITATION_WEIGHT = 0.4
COVERAGE_WEIGHT = 0.2


def score_confidence(
    retrieved_chunks: list[RetrievedChunk],
    verification: VerificationReport,
    context: PromptContext,
) -> ConfidenceScores:
    retrieval = _retrieval_score(retrieved_chunks)
    citations = _citation_score(verification)
    coverage = _coverage_score(verification, context)
    overall = (
        RETRIEVAL_WEIGHT * retrieval
        + CITATION_WEIGHT * citations
        + COVERAGE_WEIGHT * coverage
    )
    return ConfidenceScores(
        retrieval=retrieval, citations=citations, coverage=coverage, overall=overall
    )


def _retrieval_score(retrieved_chunks: list[RetrievedChunk]) -> float:
    if not retrieved_chunks:
        return 0.0
    top = min(retrieved_chunks, key=lambda r: r.final_rank)
    score = top.rerank_score if top.rerank_score is not None else top.rrf_score
    return max(0.0, min(1.0, score))


def _citation_score(verification: VerificationReport) -> float:
    if verification.total_claims == 0:
        return 1.0
    return verification.verified_claims / verification.total_claims


def _coverage_score(
    verification: VerificationReport, context: PromptContext
) -> float:
    if not context.doc_id_map:
        return 0.0
    cited_doc_ids: set[str] = set()
    for result in verification.claim_results:
        if result.doc_ids_valid:
            cited_doc_ids.update(result.claim.citation_ids)
    return len(cited_doc_ids) / len(context.doc_id_map)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag_pipeline/test_confidence_scorer.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run full suite and commit**

```bash
pytest -q
git add rag_pipeline/confidence_scorer.py tests/rag_pipeline/test_confidence_scorer.py
git commit -m "feat: add ConfidenceScorer with retrieval/citation/coverage dimensions"
```

---

### Task 7: `RagPipeline` orchestrator

**Files:**
- Create: `rag_pipeline/rag_pipeline.py`
- Test: `tests/rag_pipeline/test_rag_pipeline.py`

**Interfaces:**
- Consumes: `HybridRetriever` from `rag_hybrid_search.retrieval.retriever` (existing), `GenerationProvider`/`MockProvider` from `rag_pipeline.generation_provider` (Task 4), `build_context` (Task 2), `build_prompt` (Task 3), `verify_citations` (Task 5), `score_confidence` (Task 6), `RagAnswer`/`RagAnswerDraft`/`GenerationMetadata`/`VerificationReport`/`ConfidenceScores`/`Claim` (Task 1).
- Produces: `RagPipeline(retriever, generation_provider, prompt_version="v1")` with `.answer(question, max_chunks=5, verify=True) -> RagAnswer`, the public API of this package.

- [ ] **Step 1: Write failing tests**

`tests/rag_pipeline/test_rag_pipeline.py`:
```python
import json

from rag_hybrid_search.models import Chunk, RetrievalTrace, RetrievedChunk
from rag_pipeline.generation_provider import MockProvider
from rag_pipeline.rag_pipeline import RagPipeline


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query):
        return self._chunks, RetrievalTrace()


class RaisingGenerationProvider:
    def generate(self, prompt, **kwargs):
        raise RuntimeError("network down")


def make_retrieved_chunk(chunk_id, text, rerank_score=0.9, final_rank=1):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text=text,
        strategy_version="fixed-v1", heading=None, page=None, char_count=len(text),
    )
    return RetrievedChunk(
        chunk=chunk, dense_score=0.5, bm25_score=0.5, rrf_score=0.5,
        rerank_score=rerank_score, final_rank=final_rank,
    )


def test_answer_end_to_end_with_mock_provider():
    chunks = [make_retrieved_chunk("c1", "Employees get 20 days of paid annual leave.")]
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{
            "text": "Employees get 20 days of paid leave.",
            "citation_ids": ["d1"],
            "supporting_quote": "20 days of paid annual leave",
        }],
    })
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("How many days of paid leave?")

    assert result.answer == "Employees get 20 days of paid leave [d1]."
    assert result.citations == ["d1"]
    assert result.error is None
    assert result.verification.verified_claims == 1
    assert result.confidence.overall > 0.0


def test_answer_with_verify_false_skips_verification():
    chunks = [make_retrieved_chunk("c1", "Employees get 20 days of paid leave.")]
    canned = json.dumps({"answer": "Answer.", "claims": []})
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("question", verify=False)

    assert result.verification.total_claims == 0
    assert result.confidence.citations == 0.0
    assert result.confidence.coverage == 0.0


def test_generation_provider_exception_is_caught_not_raised():
    chunks = [make_retrieved_chunk("c1", "some text")]
    pipeline = RagPipeline(FakeRetriever(chunks), RaisingGenerationProvider())

    result = pipeline.answer("question")

    assert result.answer is None
    assert result.error is not None
    assert "network down" in result.error
    assert result.confidence.overall == 0.0


def test_malformed_json_from_provider_degrades_gracefully():
    chunks = [make_retrieved_chunk("c1", "some text")]
    pipeline = RagPipeline(
        FakeRetriever(chunks), MockProvider(canned_json="not valid json at all")
    )

    result = pipeline.answer("question")

    assert result.answer == "not valid json at all"
    assert result.error is not None
    assert result.verification.total_claims == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag_pipeline/test_rag_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_pipeline.rag_pipeline'`

- [ ] **Step 3: Implement `rag_pipeline/rag_pipeline.py`**

```python
import json
from datetime import datetime, timezone

from pydantic import ValidationError

from rag_pipeline.confidence_scorer import score_confidence
from rag_pipeline.context_builder import build_context
from rag_pipeline.generation_provider import GenerationProvider
from rag_pipeline.citation_verifier import verify_citations
from rag_pipeline.models import (
    Claim,
    ConfidenceScores,
    GenerationMetadata,
    RagAnswer,
    RagAnswerDraft,
    VerificationReport,
)
from rag_pipeline.prompt_builder import build_prompt

_EMPTY_VERIFICATION = VerificationReport(
    total_claims=0, verified_claims=0, failed_claims=0,
    hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
)
_ZERO_CONFIDENCE = ConfidenceScores(retrieval=0.0, citations=0.0, coverage=0.0, overall=0.0)


class RagPipeline:
    def __init__(self, retriever, generation_provider: GenerationProvider, prompt_version: str = "v1"):
        self._retriever = retriever
        self._generation_provider = generation_provider
        self._prompt_version = prompt_version

    def answer(self, question: str, max_chunks: int = 5, verify: bool = True) -> RagAnswer:
        retrieved_chunks, _trace = self._retriever.retrieve(question)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]

        context = build_context(retrieved_chunks)
        prompt = build_prompt(question, context, prompt_version=self._prompt_version)

        try:
            raw_output = self._generation_provider.generate(prompt)
        except Exception as e:
            return RagAnswer(
                answer=None, citations=[], confidence=_ZERO_CONFIDENCE,
                verification=_EMPTY_VERIFICATION, error=str(e),
            )

        draft, parse_error = self._parse_draft(raw_output)

        if verify:
            verification = verify_citations(draft, context)
            confidence = score_confidence(retrieved_chunks, verification, context)
        else:
            verification = _EMPTY_VERIFICATION
            confidence = ConfidenceScores(
                retrieval=score_confidence(retrieved_chunks, _EMPTY_VERIFICATION, context).retrieval,
                citations=0.0, coverage=0.0, overall=0.0,
            )

        citations = sorted({cid for c in draft.claims for cid in c.citation_ids})

        return RagAnswer(
            answer=draft.answer, citations=citations, confidence=confidence,
            verification=verification, error=parse_error,
        )

    def _parse_draft(self, raw_output: str) -> tuple[RagAnswerDraft, str | None]:
        metadata = GenerationMetadata(
            provider=type(self._generation_provider).__name__,
            model="unknown",
            prompt_version=self._prompt_version,
            generated_at=datetime.now(timezone.utc),
        )
        try:
            parsed = json.loads(raw_output)
            claims = [Claim(**c) for c in parsed.get("claims", [])]
            draft = RagAnswerDraft(answer=parsed["answer"], claims=claims, metadata=metadata)
            return draft, None
        except (json.JSONDecodeError, KeyError, ValidationError, TypeError) as e:
            degraded = RagAnswerDraft(answer=raw_output, claims=[], metadata=metadata)
            return degraded, f"failed to parse structured generation output: {e}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag_pipeline/test_rag_pipeline.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full suite and commit**

```bash
pytest -q
git add rag_pipeline/rag_pipeline.py tests/rag_pipeline/test_rag_pipeline.py
git commit -m "feat: add RagPipeline orchestrator (retrieve->context->prompt->generate->verify->score)"
```

---

### Task 8: End-to-end integration test with real `HybridRetriever`

**Files:**
- Test: `tests/rag_pipeline/test_end_to_end.py`

**Interfaces:**
- Consumes: everything from Tasks 1-7, plus the existing `rag_hybrid_search` ingestion/retrieval stack (`IngestionPipeline`, `HybridRetriever`, `SqliteChunkStore`, `ChromaVectorStore`, `BM25Index`, `IndexManager`, `DenseRetriever`, `SparseRetriever`, `CrossEncoderReranker`, `FakeEmbeddingProvider` from `tests/fakes.py`).
- Produces: nothing new for later tasks — this is the final acceptance test proving the whole Phase 1+2+3 stack works together.

- [ ] **Step 1: Write the end-to-end test**

`tests/rag_pipeline/test_end_to_end.py`:
```python
import json

from rag_hybrid_search.ingestion.pipeline import IngestionPipeline
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.rerank import CrossEncoderReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager
from rag_pipeline.generation_provider import MockProvider
from rag_pipeline.rag_pipeline import RagPipeline

from tests.fakes import FakeEmbeddingProvider


def build_pipeline_and_retriever(tmp_path):
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    bm25_index = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index_manager = IndexManager(chunk_store, vector_store, bm25_index)
    embedding_provider = FakeEmbeddingProvider()

    ingestion = IngestionPipeline(
        chunk_store=chunk_store,
        embedding_provider=embedding_provider,
        index_manager=index_manager,
        chunker=_SimpleWholeDocChunker(),
        dedup_cosine_threshold=0.95,
        dedup_text_threshold=0.9,
    )

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(embedding_provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(bm25_index, chunk_store),
        rerank_provider=CrossEncoderReranker(),
        dense_weight=0.7, sparse_weight=0.3, rrf_k=60,
        dense_k=5, sparse_k=5, rerank_top_n=3,
    )
    return ingestion, retriever


class _SimpleWholeDocChunker:
    """Test-local chunker: one chunk per document, for a tiny fixture corpus."""

    def chunk(self, document):
        from rag_hybrid_search.models import Chunk
        return [Chunk(
            chunk_id=f"{document.document_id[:8]}-0",
            document_id=document.document_id, chunk_index=0,
            text=document.content, strategy_version="whole-doc-v1",
            heading=None, page=None, char_count=len(document.content),
        )]


def test_end_to_end_grounded_answer_with_correct_citation(tmp_path):
    fixtures_dir = tmp_path / "docs"
    fixtures_dir.mkdir()
    doc_path = fixtures_dir / "leave-policy.md"
    doc_path.write_text("Employees get 20 days of paid annual leave per year.")

    ingestion, retriever = build_pipeline_and_retriever(tmp_path)
    ingestion.ingest(str(doc_path))

    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave per year [d1].",
        "claims": [{
            "text": "Employees get 20 days of paid leave per year.",
            "citation_ids": ["d1"],
            "supporting_quote": "20 days of paid annual leave per year",
        }],
    })
    pipeline = RagPipeline(retriever, MockProvider(canned_json=canned))

    result = pipeline.answer("How many days of paid leave do employees get?")

    assert "20 days" in result.answer
    assert result.citations == ["d1"]
    assert result.verification.verified_claims == 1
    assert result.confidence.overall > 0.5
    assert result.error is None
```

- [ ] **Step 2: Run to verify it needs the real loader's markdown path**

The existing `MarkdownLoader` (Task 13, Phase 1+2) expects a real markdown
file. Confirm the ingestion side reuses it correctly by checking
`rag_hybrid_search/ingestion/pipeline.py`'s constructor signature (from
Task 16) before wiring `IngestionPipeline` above — if the constructor
differs from what's assumed here (e.g. expects a `loader_registry` instead
of implicit format-detection), adjust the `build_pipeline_and_retriever`
helper to match the actual signature, not the other way around.

Run: `pytest tests/rag_pipeline/test_end_to_end.py -v`
Expected: initially FAILs or errors while wiring is corrected to match the
real `IngestionPipeline`/`HybridRetriever` constructors; iterate until it
passes for the right reason (a genuine assertion failure on missing
citations/answer text, not a constructor mismatch).

- [ ] **Step 3: Fix wiring until the test passes**

Adjust `build_pipeline_and_retriever` to match the actual constructor
signatures already committed in Tasks 6, 16, 17, 19, 20 (read those files
if any parameter name here doesn't match). Do not change any production
code in `rag_hybrid_search/` or `rag_pipeline/` to make this test pass —
only the test's own wiring helper.

Run: `pytest tests/rag_pipeline/test_end_to_end.py -v`
Expected: PASS (1 test)

- [ ] **Step 4: Run full suite and commit**

```bash
pytest -q
git add tests/rag_pipeline/test_end_to_end.py
git commit -m "test: add end-to-end grounded generation test over real retrieval stack"
```

---

## Self-Review Notes

- **Spec coverage:** ContextBuilder (Task 2), PromptBuilder + versioning + snapshot test (Task 3), GenerationProvider protocol + MockProvider + NVIDIA reuse (Task 4), CitationVerifier two-layer check (Task 5), ConfidenceScorer three dimensions (Task 6), RagPipeline orchestrator with error handling (Task 7), end-to-end proof (Task 8). All spec sections have a corresponding task.
- **Deferred items** (real tokenizer, additional providers, additional prompt versions, `benchmark/generation.py`) are intentionally not tasked here, matching the spec's "Deferred" section — Phase 4 territory.
- **Type consistency:** `PromptContext`, `RagAnswerDraft`, `Claim`, `ClaimResult`, `VerificationReport`, `ConfidenceScores`, `RagAnswer` are defined once in Task 1 and referenced identically (same field names) in every later task.
