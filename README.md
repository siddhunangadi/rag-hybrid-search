# rag-hybrid-search

A grounded, citation-verified RAG pipeline: hybrid (dense + sparse) retrieval with reranking, feeding a generation stage that verifies every claim against retrieved source text and scores its own confidence.

**Diagrams:** [plain-English walkthrough](https://claude.ai/code/artifact/53c664d3-aaad-4f7b-8b59-a7b47b952869) (six flows, analogies, no engineering background needed) · [technical reference](https://claude.ai/code/artifact/872c6f00-3d13-4294-bd97-0700a18f52f4) (component diagram, request sequences, storage schema, RRF math)

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        A[Loaders: Markdown / PDF / HTML] --> B[Chunkers]
        B --> C[Dedup]
        C --> D[IndexManager]
    end

    subgraph Storage
        D --> E[ChromaStore - dense]
        D --> F[BM25Index - sparse]
        D --> G[ChunkStore]
    end

    subgraph Retrieval
        Q[Query] --> H[DenseRetriever]
        Q --> I[SparseRetriever]
        E --> H
        F --> I
        H --> J[Fusion - RRF]
        I --> J
        J --> K[Reranker]
        K --> L[HybridRetriever]
    end

    subgraph Generation
        L --> M[ContextBuilder]
        M --> N[PromptBuilder]
        N --> O[GenerationProvider]
        O --> P[CitationVerifier]
        P --> R[ConfidenceScorer]
        R --> S[RagAnswer]
    end
```

Two packages:

- `rag_hybrid_search/` — ingestion (loaders, chunkers, dedup), storage (Chroma dense store, BM25 sparse index, chunk store), retrieval (dense, sparse, RRF fusion, reranking, `HybridRetriever` orchestrator), and provider clients (NVIDIA, Ollama).
- `rag_pipeline/` — grounded generation on top of retrieval: `ContextBuilder`, `PromptBuilder`, `GenerationProvider` protocol + `MockProvider`, `CitationVerifier`, `ConfidenceScorer`, and the `RagPipeline` orchestrator.

## Key design points

- **Citation verification is not trust-based.** Every claim the model emits must cite chunk IDs, and `CitationVerifier` checks the claim's quote is actually substring-present (containment score against the retrieved chunk text) before it counts as verified. Unverifiable claims are flagged, not silently kept.
- **Confidence is deterministic, not another LLM call.** `ConfidenceScorer` combines retrieval quality, citation verification rate, and context coverage into `overall`/`retrieval`/`citations`/`coverage` scores — auditable and reproducible.
- **Generation failures degrade gracefully.** If the provider errors or returns unparseable output, `RagPipeline.answer()` returns a `RagAnswer` with `error` set and zeroed confidence instead of raising.

## Usage

```python
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_pipeline.rag_pipeline import RagPipeline
from rag_pipeline.generation_provider import MockProvider

retriever = HybridRetriever(...)  # wired to your ChromaStore/BM25Index
pipeline = RagPipeline(retriever=retriever, generation_provider=MockProvider())

result = pipeline.answer("How many days of paid leave do employees get?")

print(result.answer)
print(result.citations)          # chunk IDs backing the answer
print(result.confidence.overall) # 0.0-1.0
print(result.verification)       # per-claim verification report
```

### Real LLM provider

Swap `MockProvider` for `GeminiProvider` to generate against a real model (free-tier API, no local install required):

```bash
export GEMINI_API_KEY=your_api_key
```

```python
from rag_hybrid_search.providers.gemini import GeminiProvider

pipeline = RagPipeline(retriever=retriever, generation_provider=GeminiProvider(api_key="..."))
```

`NvidiaProvider` (`rag_hybrid_search/providers/nvidia.py`) is also available behind the same `GenerationProvider` interface if you have an NVIDIA API key.

## API

A thin FastAPI service (`api/`) exposes the same pipeline over HTTP, in addition to the library usage above — the library remains directly importable exactly as documented in [Usage](#usage).

Provider selection is automatic based on which API keys are set in the environment (`RAG_GEMINI_API_KEY`, `RAG_NVIDIA_API_KEY`): with no keys set, it falls back to `MockProvider` for generation and a deterministic `FakeEmbeddingProvider` for embeddings — useful for trying the API without any credentials, but `/answer` won't produce a grounded, real answer in that mode. `GET /health` reports which providers were actually selected.

### Run locally

```bash
uv sync
uv run uvicorn api.main:app --reload
```

Swagger/OpenAPI docs are auto-available at `http://localhost:8000/docs` (FastAPI default).

### Run via Docker

```bash
docker build -t rag-hybrid-search .
docker run --rm -p 8000:8000 rag-hybrid-search
```

### Endpoints

**`POST /index`** — ingest one or more documents as JSON text content (`.md`, `.markdown`, `.html`, `.htm`, `.txt`):

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"filename": "leave-policy.md", "content": "Employees get 20 days of paid annual leave per year."}]}'
```

```json
{"results": [{"filename": "leave-policy.md", "status": "ready", "error": null}]}
```

**`POST /upload`** — ingest one or more real files as multipart/form-data (binary-safe: `.pdf`, `.csv`, `.xlsx`, `.docx`, plus the text formats above):

```bash
curl -F "files=@document.pdf" http://localhost:8000/upload
```

```json
{"results": [{"filename": "document.pdf", "status": "ready", "error": null}]}
```

**`POST /answer`** — ask a grounded question against the indexed corpus:

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of paid leave do employees get?", "max_chunks": 5, "verify": true}'
```

```json
{"answer": "...", "citations": ["d1"], "confidence": {"overall": 0.8, "...": "..."}, "verification": {"...": "..."}, "error": null}
```

**`GET /health`** — reports the providers actually selected and where data is persisted:

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "generation_provider": "mock", "embedding_provider": "fake", "data_dir": "./data"}
```

**`GET /documents`** — how many documents/chunks are currently indexed:

```bash
curl http://localhost:8000/documents
```

```json
{"total_documents": 1, "total_chunks": 3, "documents": [{"document_id": "0198...", "filename": "leave-policy.md", "chunk_count": 3}]}
```

**`GET /version`** — package name and version:

```bash
curl http://localhost:8000/version
```

```json
{"name": "rag-hybrid-search", "version": "0.1.0"}
```

**`DELETE /documents/{document_id}`** — purge a document from the chunk store, vector store, and BM25 index together:

```bash
curl -X DELETE http://localhost:8000/documents/<document_id>
```

```json
{"document_id": "0198...", "chunks_deleted": 3}
```

**`POST /upload/async`** — accept file uploads without blocking on ingestion (returns `202` immediately, a large/slow file runs on a background worker instead of tying up the request):

```bash
curl -F "files=@big-report.pdf" http://localhost:8000/upload/async
```

```json
{"job_id": "b46640d4-...", "status": "processing"}
```

**`GET /jobs/{job_id}`** — poll a background ingestion job started via `/upload/async`:

```bash
curl http://localhost:8000/jobs/b46640d4-...
```

```json
{"job_id": "b46640d4-...", "status": "ready", "result": {"results": [{"filename": "big-report.pdf", "status": "ready", "error": null}]}, "error": null}
```

**`POST /answer/stream`** — same as `/answer`, but streamed as Server-Sent Events: `event: delta` frames with raw text as the LLM produces it, then one `event: final` frame with the full verified `RagAnswer` once citation verification (which needs the complete text) finishes:

```bash
curl -N -X POST http://localhost:8000/answer/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of paid leave do employees get?"}'
```

## Frontend

A vanilla HTML/CSS/JS frontend (`frontend/`) — no framework, no build step. FastAPI serves it directly as static files at the app root (`api/main.py` mounts `frontend/` after the API routes, so `/health`, `/answer`, etc. still take precedence), so there's a single Render service for both the API and the UI.

- `frontend/index.html` — layout: sidebar (health/provider status, document upload with document-type selector + background-processing toggle, live indexed-document list with delete) + main area (question input with streaming toggle, answer card with citations/confidence, collapsible developer panel showing the raw API response)
- `frontend/css/styles.css` — design tokens (dark OLED palette, Inter typeface) and responsive layout
- `frontend/css/components.css` — component styling
- `frontend/js/config.js`, `api.js`, `ui.js`, `app.js` — config, thin fetch client (including SSE stream parsing and job polling), DOM rendering, event wiring

Visit `http://localhost:8000/` (or the deployed URL) directly — no separate process to run.

Note: the developer panel shows the real raw JSON response (confidence breakdown, claim verification) rather than per-chunk BM25/dense/RRF scores — the API doesn't currently expose retrieval-trace detail in `RagAnswer`, so the panel only surfaces what's actually returned. `GET /debug/retrieval` exposes that trace separately, gated behind `RAG_DEBUG_TOKEN`.

## Security

- **Prompt injection:** document text is untrusted input. The generation prompt wraps retrieved context and the user's question in `<context>`/`<question>` tags with an explicit instruction that anything inside them is data, never instructions — so a PDF containing "ignore previous instructions, you are now..." gets quoted or ignored, not obeyed.
- **No auth / multi-tenancy.** There is no login and no per-user data isolation — anyone who can reach the API can see every indexed document. Fine for a single-tenant deployment or a demo; not appropriate for storing multiple users' confidential documents without adding auth + tenant scoping first.

## Benchmark

Retrieval-quality regression check over a small fixed corpus: **Recall@3 = 1.00, MRR = 1.00** across 6 queries. See [docs/BENCHMARK.md](docs/BENCHMARK.md) for methodology, honest scope (small toy corpus, deterministic fake embeddings), and how to reproduce it (`uv run python -m scripts.benchmark`).

## Running tests

```bash
uv sync
uv run pytest -q
```

**193/195 tests passing** (2 skipped — live-provider tests that need real API keys), full suite runtime ~90s on a local M-series MacBook.

## Docker

```bash
docker build -t rag-hybrid-search .
docker run --rm -p 8000:8000 rag-hybrid-search
```

The image installs dependencies via `uv` and launches the FastAPI service via `uvicorn` on port 8000 as its default command (see [API](#api) above).

## Project layout

```
rag_hybrid_search/
  ingestion/      loaders, chunkers, dedup
  storage/        chroma_store, bm25_index, chunk_store, index_manager
  retrieval/      dense, sparse, fusion (RRF), rerank, retriever
  providers/      nvidia, ollama client wrappers
rag_pipeline/
  models.py               pydantic contracts (Claim, RagAnswer, ...)
  context_builder.py
  prompt_builder.py
  generation_provider.py  protocol + MockProvider
  citation_verifier.py
  confidence_scorer.py
  rag_pipeline.py         orchestrator
api/
  main.py           FastAPI app instance + startup wiring
  routes.py         /answer, /answer/stream, /index, /upload, /upload/async,
                     /jobs/{id}, /documents (+DELETE), /health, /version handlers
  jobs.py           in-memory JobStore, single-worker background ingestion
  schemas.py        request/response pydantic models
  dependencies.py   singleton construction + provider fallback selection
docs/superpowers/
  specs/, plans/          design docs this codebase was built from
```

## Status

v1.2.0 — core hybrid retrieval + grounded generation pipeline, a FastAPI HTTP layer (`api/`) backed by a persistent on-disk index under `data_dir`, streaming answers (SSE), async background ingestion, document deletion, and table-aware PDF parsing. 193 tests green. Usable both as a library (unchanged) and as a service.

**Known scaling limits** (by design, not yet addressed): BM25 index is fully in-RAM (no sharding past tens of thousands of documents), SQLite chunk store is single-file (fine for one instance, not for concurrent writers across multiple app instances), and there's no multi-tenant auth. All three require external infra (Elasticsearch/OpenSearch, Postgres, an auth layer) that this project doesn't provision — see the diagrams above for where they'd slot in.

## License

MIT — see [LICENSE](LICENSE).
