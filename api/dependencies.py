"""Builds and holds the app-lifetime singletons used by the API routes.

Provider selection (kept intentionally simple, no config DSL):

Generation provider:
    1. ``settings.gemini_api_key`` set -> ``GeminiProvider``
    2. ``settings.nvidia_api_key`` set -> ``NvidiaProvider``
    3. otherwise -> ``MockProvider`` (dev/demo fallback: ``/answer`` will not
       produce a grounded, real answer without a configured API key; it just
       echoes a canned response so the pipeline plumbing can be exercised).

Embedding provider:
    1. ``settings.nvidia_api_key`` set -> ``NvidiaProvider`` (the same
       instance is reused if generation also picked Nvidia)
    2. otherwise -> ``FakeEmbeddingProvider`` (dev/demo fallback: a
       deterministic trigram-hash embedding that is NOT semantically
       meaningful, useful only for exercising retrieval plumbing without a
       real embedding model configured).

Rerank provider: chosen via ``settings.rerank_backend`` (not auto-selected by
key presence, unlike above — this trades off memory/latency/accuracy and
should be an explicit choice):
    - ``"passthrough"`` (default) -> ``PassthroughReranker``: no model, no
      network call, just truncates to ``rerank_top_n`` by existing RRF score.
      Safe for memory-constrained deployments (e.g. a 512Mi free-tier
      instance, where loading a cross-encoder at startup causes an OOM crash
      before the app can bind a port).
    - ``"cross_encoder"`` -> ``CrossEncoderReranker``: local
      sentence-transformers/torch model, most accurate, heaviest (needs
      real memory headroom).
    - ``"nvidia"`` -> ``NvidiaRerankProvider``: NVIDIA's hosted reranking API,
      no local model. Requires ``settings.nvidia_api_key``. NOTE: this
      integration's request/response contract is unverified against a live
      call (documented in ``nvidia_rerank.py``) — smoke-test with a real key
      before relying on it in production.
"""

from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from api.jobs import JobStore
from rag_hybrid_search.config import Settings
from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.ingestion.chunkers.recursive import RecursiveChunker
from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.ingestion.pipeline import IngestionPipeline
from rag_hybrid_search.providers.base import EmbeddingProvider, GenerationProvider
from rag_hybrid_search.providers.gemini import GeminiProvider
from rag_hybrid_search.providers.nvidia import NvidiaProvider
from rag_hybrid_search.providers.base import RerankProvider
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.passthrough_rerank import PassthroughReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager
from rag_pipeline.generation_provider import MockProvider
from rag_pipeline.rag_pipeline import RagPipeline
from tests.fakes import FakeEmbeddingProvider

_UPLOADS_DIRNAME = "uploads"
_CHUNK_DB_FILENAME = "chunks.db"
_CHROMA_DIRNAME = "chroma"
_BM25_INDEX_FILENAME = "bm25.pkl"


@dataclass
class Container:
    """Holds the app-lifetime singletons wired for a given ``Settings``."""

    settings: Settings
    embedding_provider: EmbeddingProvider
    generation_provider: GenerationProvider
    embedding_provider_name: str
    generation_provider_name: str
    chunk_store: SqliteChunkStore
    index_manager: IndexManager
    chunker: Chunker
    rag_pipeline: RagPipeline
    uploads_dir: Path
    job_store: JobStore

    def build_ingestion_pipeline(self, loader: Loader, chunker: Chunker | None = None) -> IngestionPipeline:
        """Build an ``IngestionPipeline`` for a specific loader, reusing shared singletons.

        ``IngestionPipeline.ingest`` needs a loader matched to the document's
        format (markdown/html/text/pdf), so routes.py picks one per uploaded
        file and calls this to get a pipeline wired against the shared
        chunk store, index manager, and embedding provider.

        ``chunker`` overrides the container's default chunker for this one
        pipeline — routes.py passes a ``ClauseChunker`` here when a document
        is ingested with ``document_type="regulation"``, leaving the shared
        default chunker untouched for every other document.
        """
        return IngestionPipeline(
            loader=loader,
            chunker=chunker or self.chunker,
            embedding_provider=self.embedding_provider,
            chunk_store=self.chunk_store,
            index_manager=self.index_manager,
            dedup_cosine_threshold=self.settings.dedup_cosine_threshold,
            dedup_text_threshold=self.settings.dedup_text_similarity_threshold,
        )


def _select_generation_provider(
    settings: Settings, nvidia_provider: NvidiaProvider | None
) -> tuple[GenerationProvider, str]:
    """Pick a generation provider per the fallback order documented above."""
    if settings.gemini_api_key:
        return GeminiProvider(api_key=settings.gemini_api_key), "gemini"
    if settings.nvidia_api_key:
        return nvidia_provider or NvidiaProvider(api_key=settings.nvidia_api_key), "nvidia"
    return MockProvider(), "mock"


def _select_embedding_provider(settings: Settings) -> tuple[EmbeddingProvider, str, NvidiaProvider | None]:
    """Pick an embedding provider per the fallback order documented above."""
    if settings.nvidia_api_key:
        provider = NvidiaProvider(api_key=settings.nvidia_api_key)
        return provider, "nvidia", provider
    return FakeEmbeddingProvider(), "fake", None


def _select_rerank_provider(settings: Settings) -> RerankProvider:
    """Pick a reranker per ``settings.rerank_backend``.

    Defaults to ``PassthroughReranker`` (no torch import, no model load) so the
    API stays usable on memory-constrained deployments (e.g. a 512Mi free-tier
    instance, where loading ``sentence-transformers``/torch at startup for
    ``CrossEncoderReranker`` causes an out-of-memory crash before the app can
    bind a port). Set ``RAG_RERANK_BACKEND=cross_encoder`` for real cross-encoder
    reranking when running with enough memory headroom.
    """
    if settings.rerank_backend == "nvidia":
        if not settings.nvidia_api_key:
            raise ValueError(
                "RAG_RERANK_BACKEND=nvidia requires RAG_NVIDIA_API_KEY to be set"
            )
        from rag_hybrid_search.providers.nvidia_rerank import NvidiaRerankProvider

        return NvidiaRerankProvider(api_key=settings.nvidia_api_key)
    if settings.rerank_backend == "cross_encoder":
        from rag_hybrid_search.retrieval.rerank import CrossEncoderReranker

        return CrossEncoderReranker()
    return PassthroughReranker()


def build_container(settings: Settings | None = None) -> Container:
    """Construct all app-lifetime singletons for the given (or env-derived) settings."""
    settings = settings or Settings()

    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = data_dir / _UPLOADS_DIRNAME
    uploads_dir.mkdir(parents=True, exist_ok=True)

    embedding_provider, embedding_provider_name, nvidia_provider = _select_embedding_provider(settings)
    generation_provider, generation_provider_name = _select_generation_provider(settings, nvidia_provider)

    chunk_store = SqliteChunkStore(db_path=str(data_dir / _CHUNK_DB_FILENAME))
    vector_store = ChromaVectorStore(data_dir=str(data_dir / _CHROMA_DIRNAME))
    bm25_index = BM25Index(index_path=str(data_dir / _BM25_INDEX_FILENAME))
    # BM25Index.__init__ starts empty (no disk read) -- without this, every
    # process restart silently wipes sparse/keyword retrieval to zero
    # results until the next document upload rebuilds it, even though
    # bm25.pkl on disk still has the full corpus indexed.
    bm25_index.load()
    index_manager = IndexManager(chunk_store, vector_store, bm25_index)

    chunker = RecursiveChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(embedding_provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25_index),
        rerank_provider=_select_rerank_provider(settings),
        dense_weight=settings.rrf_dense_weight,
        sparse_weight=settings.rrf_sparse_weight,
        rrf_k=settings.rrf_k,
        dense_k=settings.dense_k,
        sparse_k=settings.sparse_k,
        rerank_top_n=settings.rerank_top_n,
    )
    rag_pipeline = RagPipeline(retriever, generation_provider, chunk_store=chunk_store)

    return Container(
        settings=settings,
        embedding_provider=embedding_provider,
        generation_provider=generation_provider,
        embedding_provider_name=embedding_provider_name,
        generation_provider_name=generation_provider_name,
        chunk_store=chunk_store,
        index_manager=index_manager,
        chunker=chunker,
        rag_pipeline=rag_pipeline,
        uploads_dir=uploads_dir,
        job_store=JobStore(),
    )


def get_container(request: Request) -> Container:
    """FastAPI dependency returning the app-lifetime ``Container`` built at startup."""
    return request.app.state.container
