# Compliance Core (v2.0) — Design Spec

## Goal

Transform rag-hybrid-search's general-purpose retrieval core into a compliance/legal-aware engine, without breaking its general-purpose use. Deliverable: a system that understands legal document structure (Article/Section/Clause hierarchy), retrieves by structured legal reference or semantic query, and returns citations mapped to exact clauses.

This is Spec 1 of a multi-spec roadmap. Later specs (deferred, not in scope here):
- v2.1 Version Intelligence (clause diff engine, regulation versioning)
- v2.2 Regulatory Monitoring (RSS/gov feeds, alert engine)
- v2.3 Compliance Intelligence (business impact analysis, risk scoring)
- v2.4 Compliance Portal (dashboard UI)
- v2.5 Evaluation & Benchmarking (compliance QA test set)

## Non-goals (this spec)

- No regulation versioning or diffing
- No external feed ingestion or alerting
- No business impact analysis
- No dashboard UI beyond existing frontend's existing views
- No LLM-based clause parsing (deferred fallback, see below)

## Scope

New `compliance/` package inside `rag_hybrid_search/`, additive to existing `ingestion/`, `retrieval/`, `storage/`, `api/`. Existing general-purpose RAG path is unchanged for non-legal documents.

```
rag_hybrid_search/
  compliance/
    __init__.py
    clause_parser.py       # detects Article/Section/Clause boundaries
    clause_chunker.py      # produces clause-scoped chunks with legal metadata
    regulation_models.py   # pydantic models: LegalMetadata, ClauseParseResult, Citation
    query_router.py        # classifies query as structured/semantic/mixed
    citation_mapper.py     # builds structured Citation objects from retrieved chunks
```

## Components

### 1. `regulation_models.py`

```python
class LegalMetadata(BaseModel):
    regulation: str | None         # e.g. "GDPR"
    version: str | None            # e.g. "2025", "2022" — regulation edition, not our schema version
    jurisdiction: str | None       # e.g. "EU"
    article: str | None
    section: str | None
    clause: str | None             # e.g. "5.2(a)"
    effective_date: date | None
    document_id: str
    document_title: str
    document_type: Literal["regulation", "policy", "contract", "standard", "guideline"] | None
    page: int | None

class ClauseParseResult(BaseModel):
    clauses: list[ClauseSpan]      # text span + LegalMetadata per clause
    confidence: float              # 0.0-0.4 poor, 0.4-0.7 acceptable, 0.7-1.0 high confidence
    parser: Literal["regex", "gemini"]   # only "regex" produced in v1; "gemini" reserved for future fallback
    fallback_used: bool = False    # always False in v1; reserved for future LLM fallback

class Citation(BaseModel):
    citation_id: str               # stable ID for frontend expand/highlight/bookmark, independent of chunk_id
    regulation: str | None
    version: str | None
    jurisdiction: str | None
    article: str | None
    section: str | None
    clause: str | None
    page: int | None
    document_id: str
    document_title: str
    document_type: Literal["regulation", "policy", "contract", "standard", "guideline"] | None
    effective_date: date | None
    chunk_id: str
    confidence: float
    display: str   # e.g. "GDPR Art. 17(3)(b), p.42"
```

`LegalMetadata` fields are all optional — a non-legal document ingested through the same pipeline just gets all-null legal metadata and behaves exactly as it does today.

`version` tracks the regulation's own edition/revision (e.g. "GDPR 2025", "ISO27001:2022"), not this spec's schema version — needed now so v2.1's clause diff engine doesn't require a schema migration later.

`LegalMetadata` fields (particularly `regulation`, `jurisdiction`, `article`, `section`, `clause`, `document_type`) must be added as indexed Chroma metadata fields, not just stored as opaque payload — otherwise structured/metadata queries degrade to a full collection scan instead of a metadata lookup.

### 2. `clause_parser.py`

Rule-based only in v1: regex + heading-pattern detection over extracted text (and PDF formatting hints — font size/bold/indentation where the loader exposes them) for patterns like `Article 5`, `ARTICLE 5`, `Art. 5`, `Section 2`, `5.1`, `5.2(a)`, `Chapter III`, `Annex A`, `Appendix 1`. Runs after the existing loaders (PDF/DOCX/HTML — all three already supported by rag-hybrid-search's ingestion loaders) produce extracted text, before chunking.

Returns a `ClauseParseResult` with a confidence score (heuristic: proportion of document covered by matched numbering patterns, consistency of hierarchy depth).

**Future enhancement (not built now):** if confidence < configurable threshold, fall back to an LLM-based parser (Gemini) to infer structure. Field `fallback_used` and `parser: Literal["regex", "gemini"]` are already modeled so this slots in later without a schema change.

### 3. `clause_chunker.py`

Given a `ClauseParseResult`, produces chunks scoped to clause boundaries (falling back to paragraph-level splitting within a clause if a clause is very long) instead of the existing generic chunker's fixed-size windows. Each chunk carries a `LegalMetadata` object in its metadata dict alongside whatever the existing chunk metadata schema already stores (document_id, chunk_id, etc.) — additive, not a replacement field.

Wiring: `ingestion/pipeline.py` gets a mode flag (e.g. detected from document metadata, or an explicit `document_type="regulation"` passed at ingest time) that routes to `clause_chunker` instead of the default chunker. Default behavior for undeclared documents is unchanged.

### 4. `query_router.py`

Classifies incoming questions into:
- **Structured**: matches a legal reference pattern directly (`"Show Article 17"`, `"clause 5.2(a)"`) → filter chunk store by `LegalMetadata` fields (exact match on regulation/article/section/clause/jurisdiction), no embedding search.
- **Metadata**: no clause-level reference, but a scope/filter constraint (`"Show only HIPAA"`, `"only EU regulations"`, `"search only policies"`) → filter chunk store by `LegalMetadata` fields (regulation/jurisdiction/document_type), then run existing hybrid retrieval + rerank pipeline scoped to that filtered subset.
- **Semantic**: no structured reference or metadata filter detected → existing dense+sparse+rerank pipeline, completely unchanged.
- **Mixed**: contains both a structured clause reference and additional intent (`"Explain Article 17"`, `"What changed in the data retention clause of GDPR?"`) → filter candidate set by metadata first, then run existing hybrid retrieval + rerank pipeline scoped to that filtered subset only.

This sits in front of `retrieval/retriever.py` as a routing layer; it doesn't modify `dense.py`/`sparse.py`/`fusion.py`/`rerank.py` internals.

### 5. `citation_mapper.py`

Takes retrieved chunks (post-rerank) and their `LegalMetadata`, produces `Citation` objects. `display` string built via template: `"{regulation} Art. {article}({clause}), p.{page}"` with graceful omission of missing fields (e.g. non-legal doc citation just falls back to existing filename/chunk-id format). API response (`api/schemas.py` `AnswerResponse` or equivalent) gains a `citations: list[Citation]` field alongside whatever citation representation exists today — additive.

## Data flow

```
PDF / DOCX / HTML
        │
        ▼
 Existing Loader (unchanged)
        │
        ▼
  Clause Parser (regex, confidence score)
        │
        ▼
  Clause Chunker (clause-scoped chunks + LegalMetadata)
        │
        ▼
 Existing Storage (chroma_store + bm25_index,
 metadata now includes indexed LegalMetadata)


Query
        │
        ▼
   Query Router
        │
   ┌────┼─────────┬──────────┐
   ▼    ▼         ▼          ▼
structured  metadata     semantic     mixed
   │          │             │           │
   │          │             │           │
metadata   metadata      existing    metadata filter
filter     filter  →     hybrid      → existing hybrid
only       existing      pipeline    pipeline on subset
           hybrid on     (unchanged)
           subset
   │          │             │           │
   └────┴─────┴─────────────┘
              │
              ▼
       Citation Mapper → Citation objects (with citation_id)
              │
              ▼
 Existing answer generation, citation verification
 (unchanged, now receives structured Citations)
```

## Error handling

- Clause parser finds no recognizable structure at all → confidence 0.0, `clauses` falls back to whole-document-as-one-clause; chunker falls back to existing default chunking behavior for that document. Document is never dropped or ingestion-failed due to parse failure.
- Structured query references a clause/article that doesn't exist in the store → return empty result set with a clear "no matching clause found" message, not a silent fallback to semantic search (avoids misleading answers).
- Malformed/partial legal metadata (e.g. article present, clause missing) → still stored and filterable on whatever fields are present.

## Testing

- Unit tests for `clause_parser`: fixtures covering GDPR-style, HIPAA-style, numbered-list-only, and unstructured/no-numbering documents; assert correct clause boundaries and confidence scores.
- Unit tests for `clause_chunker`: assert clause-to-chunk mapping, metadata attachment, fallback path for unparseable docs.
- Unit tests for `query_router`: table of queries → expected classification (structured/metadata/semantic/mixed).
- Unit tests for `citation_mapper`: assert `Citation` object construction (including unique `citation_id` generation, e.g. `uuid4` str) and `display` string rendering, including graceful degradation for missing fields.
- Integration test: ingest a synthetic multi-article regulation fixture end-to-end, ask a structured query ("Show Article 5"), a metadata-filter query ("only HIPAA"), a semantic query, and a mixed query, assert correct chunks/citations returned.
- Regression check: existing non-legal ingestion/retrieval test suite (142 tests per last cleanup) must stay green — this spec is additive only.

## Backward compatibility guarantee

The `compliance/` package SHALL NOT modify any existing ingestion, retrieval, storage, generation, or API logic. It extends the existing architecture solely through well-defined extension points (chunker selection in `ingestion/pipeline.py`, a routing layer in front of `retrieval/retriever.py`, additive fields on API response schemas). General-purpose RAG behavior remains 100% backward compatible — existing tests must stay green with zero modification.

## Open questions / explicitly deferred

- LLM-fallback parsing threshold and prompt — deferred to when real low-confidence documents are observed.
- Cross-jurisdiction clause equivalence (e.g. GDPR Art 17 vs CCPA equivalent) — out of scope, no such mapping in v2.0.
- Multi-language regulations — out of scope, assume English source text for v1.
