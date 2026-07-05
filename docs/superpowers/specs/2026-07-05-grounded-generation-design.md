# Phase 3: Grounded Generation — Design

Date: 2026-07-05
Status: Approved

## Purpose

Extend the Phase 1+2 hybrid retrieval core (tagged `v1.0.0-retrieval`) with a
grounded generation stage: retrieve → build context → build prompt → generate
→ verify citations → score confidence → return a structured, citation-checked
answer. Evaluation framework, persistent config wiring, and API/dashboard are
explicitly out of scope for this sub-spec.

## Non-goals (this sub-spec)

- Evaluation framework / golden dataset / benchmark runner (Phase 4).
- Composition root wiring `Settings` into constructed components (deferred
  until a real entrypoint — CLI/API/benchmark — needs it).
- Embedding-reuse optimization for ingestion dedup (tracked as Phase 1+2 tech
  debt, unrelated to generation).
- Free-form claim extraction from arbitrary prose — the LLM is required to
  emit structured output (see Generation Contract below), not parsed post-hoc
  from markdown/prose.

## Architecture

```
Question
     │
     ▼
HybridRetriever (existing, v1.0.0-retrieval)
     │
     ▼
ContextBuilder — dedup chunks, preserve rank, number [d1]..[dn],
                 cap at approx_token_budget (truncate lowest-ranked first)
     │
     ▼
PromptBuilder — versioned system prompt (prompt_version="v1") + question + context
     │
     ▼
GenerationProvider (protocol) — MockProvider | NvidiaProvider
     │
     ▼
RagAnswerDraft { answer, claims[], metadata }
     │
     ▼
CitationVerifier
     │
     ▼
VerificationReport { total_claims, verified_claims, failed_claims,
                      hallucinated_doc_ids, missing_quotes, claim_results[] }
     │
     ▼
ConfidenceScorer
     │
     ▼
RagAnswer { answer, citations, confidence: ConfidenceScores, verification, error }
```

## Components

### `rag_pipeline/context_builder.py`

- `build_context(chunks: list[RetrievedChunk], approx_token_budget: int = 2000) -> PromptContext`
- Dedupes by `chunk_id`, preserves `final_rank` order, numbers as `[d1]..[dn]`.
- Token budget is **approximated from character count** (no tokenizer
  dependency introduced this phase) — the parameter and its docstring make
  this explicit: `approx_token_budget` estimates tokens as
  `len(text) // 4` (a standard rough English-text heuristic), documented as
  an approximation, not an exact count. If a real tokenizer is added later,
  only this function's internals change.
- When the budget is exceeded, truncates by dropping the **lowest-ranked**
  chunks first (not truncating mid-chunk text), so every included chunk
  stays intact and citable.
- Returns `PromptContext { text: str, doc_id_map: dict[str, str] }` —
  `doc_id_map` maps `"d1"` → chunk_id, needed by `CitationVerifier` to
  resolve `[d1]` markers back to real chunks.

### `rag_pipeline/prompt_builder.py`

- `build_prompt(question: str, context: PromptContext, prompt_version: str = "v1") -> str`
- Prompt templates are versioned functions/constants (`_PROMPT_V1`), selected
  by `prompt_version` string — adding `_PROMPT_V2` later requires no call-site
  changes, just a new version string.
- The `v1` system prompt instructs the model to: only answer from supplied
  context; cite `[dN]` inline for every factual claim; say it doesn't know if
  the context doesn't support an answer; never invent citations; **respond
  only with JSON matching the `RagAnswerDraft` schema** (see Generation
  Contract below) — no prose wrapper, no markdown fences.

### Generation Contract (structured output, not prose parsing)

The model is required to return JSON matching this shape (validated by
`RagAnswerDraft`, see Data Model below):

```json
{
  "answer": "Employees get 20 days of paid leave [d1].",
  "claims": [
    {
      "text": "Employees get 20 days of paid leave.",
      "citation_ids": ["d1"],
      "supporting_quote": "20 days of paid annual leave"
    }
  ]
}
```

`RagPipeline` parses this JSON directly (`json.loads` + pydantic validation).
If parsing fails (malformed JSON, schema mismatch), it is **not** a crash:
the pipeline treats it as a zero-claim draft with the raw text as `answer`,
so `CitationVerifier`/`ConfidenceScorer` still run (reporting 0 verified
citations), and `RagAnswer.error` notes the parse failure for visibility.

### `rag_pipeline/generation_provider.py`

- `GenerationProvider` — a `Protocol`, generic by design: `generate(prompt: str) -> str`.
  Nothing in the interface or its consumers implies a specific backend.
- `MockProvider(canned_json: str | None = None)` — returns fixed/templated
  JSON matching the Generation Contract; no network; used for all
  `rag_pipeline` orchestration tests.
- `NvidiaProvider` — **reuses** the existing `rag_hybrid_search.providers.nvidia.NvidiaProvider`
  (already implements `GenerationProvider`-compatible `generate()` from
  Phase 1+2, Task 11) rather than introducing a new class. No other
  concrete provider is implemented this phase, but the protocol imposes no
  NVIDIA-specific assumption — an `OllamaProvider`-style addition later
  requires zero changes to `rag_pipeline` internals.

### `rag_pipeline/models.py` (pydantic, matching existing `models.py` convention)

```python
class Claim(BaseModel):
    text: str
    citation_ids: list[str]
    supporting_quote: str

class RagAnswerDraft(BaseModel):
    answer: str
    claims: list[Claim]
    metadata: GenerationMetadata

class GenerationMetadata(BaseModel):
    provider: str
    model: str
    prompt_version: str
    generated_at: datetime

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
    answer: str | None
    citations: list[str]
    confidence: ConfidenceScores
    verification: VerificationReport
    error: str | None
```

### `rag_pipeline/citation_verifier.py`

- `verify_citations(draft: RagAnswerDraft, context: PromptContext) -> VerificationReport`
- Per claim, two layers:
  1. **Doc-id check**: every `citation_id` in `claim.citation_ids` must exist
     in `context.doc_id_map`. Missing → `doc_ids_valid=False`, contributes to
     `hallucinated_doc_ids`.
  2. **Quote check**: `claim.supporting_quote` fuzzy-matched via
     `difflib.SequenceMatcher(None, quote, chunk_text).ratio()` against the
     text of each cited chunk — consistent with the existing dedup.py
     text-similarity approach from Phase 1+2.
     - `QUOTE_MATCH_THRESHOLD = 0.80` (module-level constant, not 0.9 —
       tuned down from the original proposal since paraphrased-but-faithful
       quotes commonly score 0.75-0.85 against source text; configurable via
       an optional parameter for later benchmarking).
     - Below threshold → contributes to `missing_quotes`.
- `ClaimResult.passed = doc_ids_valid and quote_match_score >= QUOTE_MATCH_THRESHOLD`.
- `VerificationReport` aggregates counts/lists directly from `claim_results`,
  so `ConfidenceScorer` consumes summary fields without recomputing them.

### `rag_pipeline/confidence_scorer.py`

- `score_confidence(retrieved_chunks: list[RetrievedChunk], verification: VerificationReport) -> ConfidenceScores`
- Three independent dimensions, each documented with its own meaning:
  - **retrieval**: how strong were the retrieved chunks (normalized top
    fused/rerank score from `retrieved_chunks`).
  - **citations**: `verified_claims / total_claims` (1.0 if `total_claims == 0`
    and `answer` is a "don't know" — no false confidence penalty for
    correctly declining to answer).
  - **coverage**: fraction of `retrieved_chunks` actually cited by at least
    one claim (did the answer use the evidence it was given, or ignore most
    of it).
  - **overall**: `0.4*retrieval + 0.4*citations + 0.2*coverage` (weights as
    configurable module-level constants, not magic numbers inline).
- Pure function, fully deterministic, no LLM self-rating involved.

### `rag_pipeline/rag_pipeline.py`

- `RagPipeline(retriever: HybridRetriever, generation_provider: GenerationProvider, prompt_version: str = "v1")`
- `.answer(question: str, max_chunks: int = 5, verify: bool = True) -> RagAnswer`
- Orchestrates: retrieve(max_chunks) → build_context → build_prompt →
  generate → parse-or-degrade → (if `verify`) verify_citations → score_confidence → RagAnswer.
- `verify=False` skips citation verification (e.g. for fast/cheap draft
  mode) — `VerificationReport` is still returned but zeroed/empty, and
  `ConfidenceScores.citations`/`coverage` are `0.0` in that case, clearly
  signaling "not checked" rather than "checked and failed."
- `GenerationProvider.generate()` exceptions (network/model errors) are
  caught here — never propagate to the caller. Returns `RagAnswer` with
  `answer=None`, `error=<message>`, `confidence` all-zero,
  `verification` empty.

## Data flow / schemas

Covered inline in the Data Model section above (`Claim`, `RagAnswerDraft`,
`GenerationMetadata`, `ClaimResult`, `VerificationReport`, `ConfidenceScores`,
`RagAnswer`).

## Error handling

- Generation provider failure → caught in `RagPipeline.answer`, returns a
  populated-but-empty `RagAnswer` with `error` set. Never raises.
- Malformed/non-JSON generation output → treated as a zero-claim draft
  (raw text preserved as `answer`), `error` notes the parse failure,
  verification/scoring still run and correctly report zero verified claims.
- `CitationVerifier` never raises on a missing doc-id or quote — it records
  the failure in `VerificationReport`, which downstream scoring consumes.

## Testing

Target ~40-50 new tests, per the approved estimate:

- **`context_builder`**: empty context; duplicate chunks deduped; rank order
  preserved; `approx_token_budget` truncation drops lowest-ranked chunks
  first without splitting any retained chunk's text; `doc_id_map` correctness.
- **`prompt_builder`**: prompt contains system instructions; prompt contains
  numbered context; no-context case (empty `PromptContext`) still produces a
  valid prompt; **prompt snapshot test** — assert the full rendered `v1`
  prompt string matches a stored expected value, so an accidental edit to
  the template fails a test rather than silently changing behavior; version
  string round-trips into the prompt's own metadata comment (so it's
  traceable which version generated a given answer).
- **`generation_provider`**: `MockProvider` returns configured JSON;
  `NvidiaProvider` reuse test verifies it's called correctly (mocked HTTP,
  pytest-mock, matching the existing Task 11 pattern — no real network call).
- **`citation_verifier`**: valid doc ids; hallucinated doc ids; exact quote
  match; fuzzy quote match above/below `QUOTE_MATCH_THRESHOLD`; missing
  quote; multiple claims mixing pass/fail; `VerificationReport` summary
  fields computed correctly from `claim_results`.
- **`confidence_scorer`**: all citations pass; half fail; zero claims with
  a "don't know" answer (no false penalty); multiple sources cited vs. one;
  weighted `overall` arithmetic asserted exactly, not just ordering.
- **`rag_pipeline`**: end-to-end with `MockProvider` only, fully
  deterministic, no network — covers the full retrieve→context→prompt→
  generate→verify→score chain; `verify=False` path; generation-provider
  exception path (caught, not raised); malformed-JSON-from-provider path.

## Deferred (explicit extension points, not built now)

- **Real tokenizer** for `ContextBuilder`'s budget (currently character-count
  approximation) — swap-in point is `build_context`'s internals only.
- **Additional `GenerationProvider` implementations** (Ollama, OpenRouter,
  etc.) — protocol imposes no NVIDIA-specific assumption, purely additive.
- **Additional prompt versions** (`v2`, `v3`) for benchmark comparison —
  `prompt_version` parameter and versioned-template pattern already support
  this without call-site changes.
- **`benchmark/generation.py`** alongside the existing-from-Phase-1+2-plan
  `benchmark/retrieval.py` placeholder — directory structure anticipated,
  not implemented this phase (Phase 4).
