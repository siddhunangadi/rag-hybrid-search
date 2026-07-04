# RAG Hybrid Search — Phase 1+2 Design: Ingestion & Hybrid Retrieval Core

Date: 2026-07-05
Status: Approved for planning

## Purpose

Build the ingestion and hybrid retrieval core of a production-style RAG system
over internal documentation. This sub-spec covers document ingestion,
configurable chunking, deduplication, and hybrid (dense + sparse) retrieval
with fusion and reranking. Generation, citations, confidence scoring, the eval
framework, and the API/dashboard are explicitly out of scope and will be
separate sub-specs.

## Goals

- Ingest markdown, HTML, plain text, and PDF documents into a normalized,
  typed representation with source metadata.
- Support three interchangeable chunking strategies (fixed, recursive,
  semantic) behind one interface.
- Avoid re-processing identical documents and avoid storing near-duplicate
  chunks.
- Retrieve relevant chunks for a query using both dense (embedding) and
  sparse (BM25) search, fused via a weighted Reciprocal Rank Fusion variant,
  then reranked with a cross-encoder.
- Keep the vector store, sparse index, and LLM/embedding provider swappable
  behind explicit interfaces so no other module imports a concrete
  implementation directly.
- Deploy for free: NVIDIA NIM as the primary embedding/LLM provider (with an
  Ollama-backed local implementation of the same interface for offline dev),
  ChromaDB embedded (file-based, no separate service) for the vector store,
  and a local sentence-transformers cross-encoder for reranking (no per-query
  API cost or rate-limit exposure at request time).

## Non-goals (this sub-spec)

- Answer generation, citation formatting/verification, confidence scoring.
- Evaluation harness / golden dataset.
- FastAPI service, dashboard, Docker Compose, auth, observability, caching,
  rate limiting, background job queues.

## Architecture

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
      base.py         # Chunker ABC: chunk(document) -> list[Chunk]
      fixed.py         # fixed-size with overlap
      recursive.py      # structure-aware, splits on headers/sections
      semantic.py        # splits on embedding-similarity topic boundaries
    dedup.py             # two-stage duplicate detection
    pipeline.py           # IngestionPipeline orchestrator
  retrieval/
    retriever.py           # HybridRetriever orchestrator
    dense.py                 # DenseRetriever
    sparse.py                 # SparseRetriever (BM25)
    fusion.py                  # weighted_rrf()
    rerank.py                   # CrossEncoderReranker
  providers/
    base.py                     # EmbeddingProvider ABC, GenerationProvider ABC
    nvidia.py                    # NIM implementation of both
    ollama.py                     # local implementation of both
  storage/
    base.py                       # VectorStore ABC, ChunkStore ABC
    chroma_store.py                 # VectorStore impl (ChromaDB)
    chunk_store.py                    # canonical chunk store (SQLite)
    bm25_index.py                      # BM25 index built/synced from ChunkStore
  models.py                            # Document, Chunk, RetrievedChunk (pydantic)
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
    chunk_id: str             # sha256(document_id + chunk_index)
    document_id: str
    text: str
    chunk_index: int
    strategy: Literal["fixed", "recursive", "semantic"]
    heading: str | None
    page: int | None
    char_count: int

class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None
    sparse_score: float | None
    fusion_score: float
    rerank_score: float | None
```

### Ingestion flow

1. `Loader.load(path) -> Document`: reads file, strips to clean plaintext,
   computes `document_id = sha256(content)`, attaches `source_path`/`format`.
2. `IngestionPipeline.ingest(path)`:
   - Load document.
   - Check `ChunkStore` for existing chunks with this `document_id`; if
     present, skip re-ingestion entirely (idempotent re-upload).
   - Run configured `Chunker.chunk(document) -> list[Chunk]`.
   - Embed each chunk via `EmbeddingProvider`.
   - Run `dedup.py` two-stage check against existing chunk embeddings:
     1. Cosine similarity > 0.95 against any existing chunk → candidate.
     2. Confirm with normalized text similarity (difflib
        `SequenceMatcher` ratio > 0.9) on the candidate pair. Only skip
        the new chunk if both stages agree; otherwise keep it (avoids
        false positives like two structurally similar but distinct code
        snippets).
   - Persist surviving chunks to `ChunkStore` (canonical source of truth:
     id, text, metadata — no embeddings stored here).
   - Upsert embeddings into `VectorStore` (ChromaStore), keyed by
     `chunk_id`.
   - Rebuild/update `BM25Index` from `ChunkStore` contents, keyed by the
     same `chunk_id`s, and persist it (`bm25.pkl`).

Because `ChunkStore` is canonical and both `VectorStore` and `BM25Index` are
built from it and keyed by the same IDs, there is no independent chunk list
to drift out of sync.

### Retrieval flow

`HybridRetriever.retrieve(query: str, k: int = 5) -> list[RetrievedChunk]`:

1. `DenseRetriever(embedding_provider, vector_store).search(query, k=10)`
   — embeds the query, queries `VectorStore` for top-10 by cosine similarity.
2. `SparseRetriever(chunk_store, bm25_index).search(query, k=10)` — BM25
   top-10 over the corpus.
3. `fusion.weighted_rrf(dense_results, sparse_results, dense_weight=0.7,
   sparse_weight=0.3, k=60) -> list[RetrievedChunk]` — a **weighted variant**
   of Reciprocal Rank Fusion (standard RRF sums `1/(k+rank)` per list;
   here each list's contribution is scaled by its configured weight before
   summing). Named and documented as such, not presented as vanilla RRF.
   Produces top-20 fused candidates.
4. `CrossEncoderReranker.rerank(query, candidates, top_n=5)` — local
   sentence-transformers cross-encoder (`ms-marco-MiniLM-L-6-v2` or similar)
   scores each candidate against the query text directly; returns top 5.

`DenseRetriever` depends only on the `EmbeddingProvider` interface (never
imports `providers.nvidia` or `providers.ollama` directly) — provider choice
is injected via `config.py` at wiring time.

### Providers (`providers/`)

```python
class EmbeddingProvider(ABC):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class GenerationProvider(ABC):
    def generate(self, prompt: str, **kwargs) -> str: ...
```

- `nvidia.py`: implements both against the NVIDIA NIM API (OpenAI-compatible
  client), using an embedding model (e.g. `nvidia/nv-embedqa-e5-v5`) and an
  LLM (e.g. `meta/llama-3.1-70b-instruct`). Primary provider, used in the
  deployed environment. Requires `NVIDIA_API_KEY` env var.
- `ollama.py`: implements both against a local Ollama instance (e.g.
  `nomic-embed-text`, `llama3.1`). Used for offline dev; selected via
  `config.py` provider setting.
- Provider selection is a `Settings.provider: Literal["nvidia", "ollama"]`
  field; a factory function in `providers/__init__.py` returns the
  configured implementation. No other module hardcodes a provider.

### Storage (`storage/`)

```python
class VectorStore(ABC):
    def upsert(self, chunk_id: str, embedding: list[float], metadata: dict): ...
    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]: ...

class ChunkStore(ABC):
    def get(self, chunk_id: str) -> Chunk | None: ...
    def get_by_document(self, document_id: str) -> list[Chunk]: ...
    def put(self, chunk: Chunk) -> None: ...
    def all(self) -> Iterator[Chunk]: ...
```

- `chroma_store.py`: `VectorStore` impl using ChromaDB's persistent client,
  data directory mounted as a Docker volume / persistent disk in deployment.
- `chunk_store.py`: SQLite-backed `ChunkStore` (single file, easy to mount
  as a persistent volume alongside the Chroma data dir).
- `bm25_index.py`: builds a `rank_bm25.BM25Okapi` index from all chunks in
  `ChunkStore`, exposes `search(query, k) -> list[tuple[chunk_id, score]]`,
  and persists/reloads itself via pickle so it doesn't rebuild from scratch
  on every process restart unless the underlying chunk store has changed
  (tracked via a stored content hash / row count check).

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
  input; dedup correctly skips a true near-duplicate and correctly *keeps*
  a high-cosine-but-different-text pair (e.g. two similar code snippets);
  re-ingesting the same document is a no-op.
- `tests/retrieval/`: `weighted_rrf` produces correct scores against
  hand-computed rank lists; a BM25-favorable query (exact function name)
  surfaces the right chunk even when dense search ranks it low; hybrid
  retrieval end-to-end returns 5 results ordered by rerank score.
- `tests/storage/`: `ChunkStore` and `VectorStore`/`BM25Index` stay in sync
  after an ingest (same chunk IDs present in all three); BM25 index persists
  and reloads correctly.

## Deployment notes

- ChromaDB persistent client + SQLite `ChunkStore` + `bm25.pkl` all live
  under `data_dir`, which is the single directory to mount as a persistent
  disk (e.g. Render persistent disk, matching the existing
  `llm-cost-autopilot` deployment pattern).
- Cross-encoder reranker model is bundled into the Docker image at build
  time (small, ~80MB) so no network call is needed per request.
- NVIDIA NIM is called over HTTPS at ingest time (embeddings) and query time
  (embeddings + generation in later phases); requires `NVIDIA_API_KEY` set
  as a deployment secret.

## Sample corpus

A small public markdown documentation set will be used to validate the
pipeline end-to-end (ingestion → chunking → dedup → dense+sparse retrieval →
fusion → rerank) during implementation and testing.
