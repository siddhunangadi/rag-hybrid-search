# RAG Hybrid Search — Phase 1+2 Design: Ingestion & Hybrid Retrieval Core

Date: 2026-07-05
Status: Approved for planning

## Purpose

Build the ingestion and hybrid retrieval core of a production-style RAG system
over internal documentation. This sub-spec covers document ingestion,
configurable chunking, deduplication, index management, and hybrid
(dense + sparse) retrieval with fusion and reranking. Generation, citations,
confidence scoring, the eval framework, and the API/dashboard are explicitly
out of scope and will be separate sub-specs.

## Goals

- Ingest markdown, HTML, plain text, and PDF documents into a normalized,
  typed representation with source metadata.
- Support three interchangeable chunking strategies (fixed, recursive,
  semantic) behind one interface, each explicitly versioned.
- Avoid re-processing unchanged documents; re-index automatically when a
  document's content changes.
- Avoid storing near-duplicate chunks via a two-stage check.
- Retrieve relevant chunks for a query using both dense (embedding) and
  sparse (BM25) search, fused via a weighted Reciprocal Rank Fusion variant,
  then reranked with a cross-encoder.
- Keep the vector store, sparse index, LLM/embedding provider, and reranker
  swappable behind explicit interfaces so no other module imports a concrete
  implementation directly.
- Track index build/sync status explicitly rather than assuming indexing
  always succeeds.
- Deploy for free: NVIDIA NIM as the primary embedding/LLM provider (with an
  Ollama-backed local implementation of the same interface for offline dev),
  ChromaDB embedded (file-based, no separate service) for the vector store,
  and a local sentence-transformers cross-encoder for reranking (no per-query
  API cost or rate-limit exposure at request time).

## Non-goals (this sub-spec)

- Answer generation, citation formatting/verification, confidence scoring.
- Evaluation harness / golden dataset.
- FastAPI service, dashboard, Docker Compose, auth, full observability
  (metrics export/dashboards), caching, rate limiting, background job queues.
- This sub-spec adds latency *telemetry hooks* only (in-process timing,
  exposed as return values/logs) — wiring those into an external metrics
  system is a later sub-spec.

## Architecture

```
                  Documents
                      │
                      ▼
               IngestionPipeline
                      │
             Normalize / Chunk
                      │
                      ▼
              EmbeddingProvider (batched)
                      │
                      ▼
                 Deduplication
                      │
                      ▼
                 ChunkStore (SQLite, canonical)
                      │
                      ▼
                 IndexManager
                ┌────────────┐
                ▼            ▼
          VectorStore     BM25Index
                │            │
                └──────┬─────┘
                       ▼
               HybridRetriever
                       ▼
                Weighted RRF
                       ▼
              RerankProvider (CrossEncoder today)
                       ▼
              RetrievedChunk[]
```

```
rag-hybrid-search/
  ingestion/
    loaders/
      base.py        # Loader ABC: load(path) -> Document
      markdown.py
      html.py
      text.py
      pdf.py
    chunkers/
      base.py         # Chunker ABC: chunk(document) -> list[Chunk]; exposes version
      fixed.py         # fixed-size with overlap (version "fixed-v1")
      recursive.py      # structure-aware, splits on headers/sections ("recursive-v1")
      semantic.py        # splits on embedding-similarity topic boundaries ("semantic-v1")
    dedup.py             # two-stage duplicate detection
    pipeline.py           # IngestionPipeline orchestrator
  retrieval/
    retriever.py           # HybridRetriever orchestrator
    dense.py                 # DenseRetriever
    sparse.py                 # SparseRetriever (BM25)
    fusion.py                  # weighted_rrf()
    rerank.py                   # RerankProvider ABC + CrossEncoderReranker impl
  providers/
    base.py                     # EmbeddingProvider ABC, GenerationProvider ABC, RerankProvider ABC
    nvidia.py                    # NIM implementation of embedding+generation
    ollama.py                     # local implementation of embedding+generation
  storage/
    base.py                       # VectorStore ABC, ChunkStore ABC
    chroma_store.py                 # VectorStore impl (ChromaDB), stores embedding + version metadata
    chunk_store.py                    # canonical chunk store (SQLite)
    bm25_index.py                      # BM25 index, built/synced via IndexManager
    index_manager.py                    # rebuild/verify/sync orchestration + status tracking
  models.py                            # Document, Chunk, EmbeddingRecord, RetrievedChunk (pydantic)
  config.py                             # Settings (pydantic-settings)
  tests/
    ingestion/
    retrieval/
    storage/
```

### Data model (`models.py`)

```python
class Document(BaseModel):
    document_id: str        # sha256(content)
    source_path: str
    content: str             # normalized plaintext
    format: Literal["markdown", "html", "text", "pdf"]

class Chunk(BaseModel):
    chunk_id: str             # uuid7
    document_id: str
    chunk_index: int
    text: str
    strategy_version: str      # e.g. "recursive-v1" (chunker name + version combined)
    heading: str | None
    page: int | None
    char_count: int

class EmbeddingRecord(BaseModel):
    chunk_id: str
    embedding: list[float]
    embedding_model: str        # e.g. "nvidia/nv-embedqa-e5-v5"
    embedding_dimension: int
    provider: str                 # e.g. "nvidia"
    created_at: datetime

class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None
    bm25_score: float | None
    rrf_score: float
    rerank_score: float | None
    final_rank: int

class IndexStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
```

Embeddings are **not** stored on `Chunk` — they live in `EmbeddingRecord`,
associated by `chunk_id`, inside the vector store's metadata. This means
switching embedding models later does not require touching `Chunk` records;
only re-embedding and re-upserting `EmbeddingRecord`s.

Chunk IDs are UUIDv7 (time-ordered, globally unique), not derived
sequentially, so they remain stable if chunking is re-run or distributed
across workers in the future. `document_id` stays a content hash
(`sha256(content)`) so identical uploads are detected cheaply.

### Ingestion flow

1. `Loader.load(path) -> Document`: reads file, strips to clean plaintext,
   computes `document_id = sha256(content)`, attaches `source_path`/`format`.
2. `IngestionPipeline.ingest(path)`:
   - Load document.
   - Look up `document_id` in `ChunkStore`:
     - **Same hash already indexed** → skip entirely (idempotent re-upload).
     - **A document at this `source_path` exists with a different hash**
       (i.e. the file was edited) → delete its old chunks (and their
       vector/BM25 entries via `IndexManager`), then proceed to re-index
       under the new `document_id`.
     - **Not present** → proceed to index.
   - Run configured `Chunker.chunk(document) -> list[Chunk]`, tagging each
     chunk with `strategy_version` (e.g. `"recursive-v1"`). Bumping a
     chunker's internal logic bumps its version string, so old and new
     chunks from the same strategy family are distinguishable.
   - Set index status `PENDING` → `INDEXING` for this document's chunk batch.
   - Embed all chunks in **one batched call**: `EmbeddingProvider.embed(list[str])
     -> list[list[float]]` (never per-chunk calls) — cheaper and faster on
     every provider.
   - Run `dedup.py` two-stage check against existing chunk embeddings:
     1. Cosine similarity > 0.95 against any existing chunk → candidate.
     2. Confirm with normalized text similarity (difflib
        `SequenceMatcher` ratio > 0.9) on the candidate pair. Only skip
        the new chunk if both stages agree; otherwise keep it (avoids
        false positives like two structurally similar but distinct code
        snippets).
   - Persist surviving chunks to `ChunkStore` (canonical source of truth:
     id, text, metadata — no embeddings).
   - Hand off to `IndexManager.index(chunks, embedding_records)`:
     - Upsert `EmbeddingRecord`s into `VectorStore`, keyed by `chunk_id`.
     - Rebuild/update `BM25Index` from `ChunkStore` contents for the
       affected documents, keyed by the same `chunk_id`s, and persist it
       (`bm25.pkl`).
     - Mark status `READY` on success, `FAILED` (with error detail) if
       either index write fails — a failed batch does not leave the
       document silently half-indexed; it can be retried.

Because `ChunkStore` is canonical and both `VectorStore` and `BM25Index` are
built from it and keyed by the same IDs, there is no independent chunk list
to drift out of sync.

### IndexManager (`storage/index_manager.py`)

Centralizes all operations that touch both indexes together, so
`IngestionPipeline` never talks to `VectorStore`/`BM25Index` directly:

```python
class IndexManager:
    def index(self, chunks: list[Chunk], embeddings: list[EmbeddingRecord]) -> IndexStatus: ...
    def remove_document(self, document_id: str) -> None: ...
    def rebuild_vector_index(self) -> None: ...
    def rebuild_bm25_index(self) -> None: ...
    def rebuild_all(self) -> None: ...
    def verify_sync(self) -> list[str]: ...  # returns list of mismatched chunk_ids, if any
```

This is the seam where future operational commands (rebuild, verify,
optimize) attach without touching ingestion or retrieval code.

### Retrieval flow

`HybridRetriever.retrieve(query: str, k: int = 5) -> list[RetrievedChunk]`:

1. `DenseRetriever(embedding_provider, vector_store).search(query, k=10)`
   — embeds the query, queries `VectorStore` for top-10 by cosine similarity.
   Records `dense_latency_ms`.
2. `SparseRetriever(chunk_store, bm25_index).search(query, k=10)` — BM25
   top-10 over the corpus. Records `bm25_latency_ms`.
3. `fusion.weighted_rrf(dense_results, sparse_results, dense_weight=0.7,
   sparse_weight=0.3, k=60) -> list[RetrievedChunk]` — a **weighted variant**
   of Reciprocal Rank Fusion (standard RRF sums `1/(k+rank)` per list;
   here each list's contribution is scaled by its configured weight before
   summing). Named and documented as such, not presented as vanilla RRF.
   Produces top-20 fused candidates. Records `fusion_latency_ms`.
4. `RerankProvider.rerank(query, candidates, top_n=5)` — scores each
   candidate against the query, sets `final_rank`. Records
   `rerank_latency_ms`.
5. `HybridRetriever` sums all four into `total_latency_ms` and returns it
   alongside the ranked `RetrievedChunk[]` (as a small `RetrievalTrace`
   companion object) — a telemetry hook, not a metrics pipeline; later
   phases can forward this to real monitoring without changing this
   function's internals.

`DenseRetriever` depends only on the `EmbeddingProvider` interface (never
imports `providers.nvidia` or `providers.ollama` directly) — provider choice
is injected via `config.py` at wiring time. Likewise `HybridRetriever`
depends on the `RerankProvider` interface, not a concrete cross-encoder
class.

### Providers (`providers/`)

```python
class EmbeddingProvider(ABC):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def model_name(self) -> str: ...
    @property
    def dimension(self) -> int: ...

class GenerationProvider(ABC):
    def generate(self, prompt: str, **kwargs) -> str: ...

class RerankProvider(ABC):
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...
```

- `nvidia.py`: implements `EmbeddingProvider` + `GenerationProvider` against
  the NVIDIA NIM API (OpenAI-compatible client), using an embedding model
  (e.g. `nvidia/nv-embedqa-e5-v5`) and an LLM (e.g.
  `meta/llama-3.1-70b-instruct`). Primary provider, used in the deployed
  environment. Requires `NVIDIA_API_KEY` env var. `embed()` calls the API
  once with the full batch of texts.
- `ollama.py`: implements the same two interfaces against a local Ollama
  instance (e.g. `nomic-embed-text`, `llama3.1`). Used for offline dev;
  selected via `config.py` provider setting.
- `rerank.py` (`retrieval/`): implements `RerankProvider` today via a local
  sentence-transformers `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`). The
  interface leaves room for a future NVIDIA/Cohere/Jina/BAAI-hosted reranker
  implementation with zero changes to `HybridRetriever`.
- Provider selection is a `Settings.provider: Literal["nvidia", "ollama"]`
  field; a factory function in `providers/__init__.py` returns the
  configured implementation. No other module hardcodes a provider.

### Storage (`storage/`)

```python
class VectorStore(ABC):
    def upsert(self, chunk_id: str, embedding_record: EmbeddingRecord) -> None: ...
    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]: ...
    def delete(self, chunk_ids: list[str]) -> None: ...

class ChunkStore(ABC):
    def get(self, chunk_id: str) -> Chunk | None: ...
    def get_by_document(self, document_id: str) -> list[Chunk]: ...
    def get_document_hash(self, source_path: str) -> str | None: ...
    def put(self, chunk: Chunk) -> None: ...
    def delete_by_document(self, document_id: str) -> None: ...
    def all(self) -> Iterator[Chunk]: ...
```

- `chroma_store.py`: `VectorStore` impl using ChromaDB's persistent client;
  stores `embedding_model`, `embedding_dimension`, `provider`, `created_at`
  as metadata alongside each vector. Data directory mounted as a Docker
  volume / persistent disk in deployment.
- `chunk_store.py`: **SQLite-backed** `ChunkStore` (chosen directly over
  JSONL — ACID transactions, indexing, concurrent reads, SQL filtering by
  metadata, and straightforward incremental updates/deletes). Single file,
  easy to mount as a persistent volume alongside the Chroma data dir.
- `bm25_index.py`: builds a `rank_bm25.BM25Okapi` index from all chunks in
  `ChunkStore`, exposes `search(query, k) -> list[tuple[chunk_id, score]]`,
  and persists/reloads itself via pickle. Rebuilds are driven by
  `IndexManager`, not on every process restart.
- `index_manager.py`: see IndexManager section above.

### Configuration (`config.py`)

```python
class Settings(BaseSettings):
    provider: Literal["nvidia", "ollama"] = "nvidia"
    nvidia_api_key: str | None = None
    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50
    dense_k: int = 10
    sparse_k: int = 10
    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    rrf_k: int = 60
    rerank_top_n: int = 5
    dedup_cosine_threshold: float = 0.95
    dedup_text_similarity_threshold: float = 0.9
    data_dir: str = "./data"   # holds chroma/, chunks.db, bm25.pkl

    class Config:
        env_prefix = "RAG_"
```

Pydantic gives validation and IDE support over raw env var reads.

## Testing

- `tests/ingestion/`: each loader produces expected `Document` for a sample
  file; each chunker produces expected chunk boundaries/counts on a fixed
  input and tags the correct `strategy_version`; dedup correctly skips a
  true near-duplicate and correctly *keeps* a high-cosine-but-different-text
  pair (e.g. two similar code snippets); re-ingesting an unchanged document
  is a no-op; re-ingesting an edited document (same `source_path`, new hash)
  replaces old chunks rather than duplicating them.
- `tests/retrieval/`: `weighted_rrf` produces correct scores against
  hand-computed rank lists; a BM25-favorable query (exact function name)
  surfaces the right chunk even when dense search ranks it low; hybrid
  retrieval end-to-end returns 5 results ordered by `final_rank`; latency
  fields are populated and `total_latency_ms` is at least the sum of the
  per-stage timings.
- `tests/storage/`: `ChunkStore` and `VectorStore`/`BM25Index` stay in sync
  after an ingest (same chunk IDs present in all three); `IndexManager.verify_sync()`
  returns empty on a healthy index and reports mismatches after a simulated
  partial failure; BM25 index persists and reloads correctly;
  `IndexManager.remove_document` clears a document from both indexes.

## Deployment notes

- ChromaDB persistent client + SQLite `ChunkStore` + `bm25.pkl` all live
  under `data_dir`, which is the single directory to mount as a persistent
  disk (e.g. Render persistent disk, matching the existing
  `llm-cost-autopilot` deployment pattern).
- Cross-encoder reranker model is bundled into the Docker image at build
  time (small, ~80MB) so no network call is needed per request.
- NVIDIA NIM is called over HTTPS at ingest time (embeddings, batched) and
  query time (embeddings + generation in later phases); requires
  `NVIDIA_API_KEY` set as a deployment secret.

## Sample corpus

A small public markdown documentation set will be used to validate the
pipeline end-to-end (ingestion → chunking → dedup → dense+sparse retrieval →
fusion → rerank) during implementation and testing.
