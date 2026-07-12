import pytest

from rag_hybrid_search.ingestion.chunkers.recursive import RecursiveChunker
from rag_hybrid_search.ingestion.loaders.markdown import MarkdownLoader
from rag_hybrid_search.ingestion.pipeline import IngestionPipeline
from rag_hybrid_search.models import IndexStatus
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.rerank import CrossEncoderReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.index_manager import IndexManager
from tests.fakes import FakeEmbeddingProvider, fake_pinecone_stores

SAMPLE_DOCS = [
    "tests/fixtures/sample_docs/setup.md",
    "tests/fixtures/sample_docs/deployment.md",
    "tests/fixtures/sample_docs/onboarding.md",
]


@pytest.fixture
def system(tmp_path):
    provider = FakeEmbeddingProvider()
    chunk_store, vector_store = fake_pinecone_stores(embedding_dimension=provider.dimension)
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index_manager = IndexManager(chunk_store, vector_store, bm25)

    pipeline = IngestionPipeline(
        loader=MarkdownLoader(),
        chunker=RecursiveChunker(chunk_size=300, chunk_overlap=30),
        embedding_provider=provider,
        chunk_store=chunk_store,
        index_manager=index_manager,
        dedup_cosine_threshold=0.95,
        dedup_text_threshold=0.9,
    )

    for path in SAMPLE_DOCS:
        status = pipeline.ingest(path)
        assert status == IndexStatus.READY

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25),
        rerank_provider=CrossEncoderReranker(),
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        dense_k=10,
        sparse_k=10,
        rerank_top_n=3,
        rerank_fused_top_n=20,
    )
    return retriever


def test_keyword_query_surfaces_deployment_error_doc(system):
    results, trace = system.retrieve("ERROR_CODE_0x834")

    assert any("ERROR_CODE_0x834" in r.chunk.text for r in results)
    assert trace.total_latency_ms > 0


def test_conceptual_query_surfaces_onboarding_doc(system):
    results, _trace = system.retrieve("What should new engineers do in their first week?")

    assert any(
        "onboarding" in r.chunk.text.lower() or "first week" in r.chunk.text.lower()
        for r in results
    )


def test_all_results_have_final_rank_set(system):
    results, _trace = system.retrieve("How do I configure the provider?")

    ranks = [r.final_rank for r in results]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1
