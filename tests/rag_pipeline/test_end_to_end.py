import json

from rag_hybrid_search.ingestion.loaders.markdown import MarkdownLoader
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
        loader=MarkdownLoader(),
        chunker=_SimpleWholeDocChunker(),
        embedding_provider=embedding_provider,
        chunk_store=chunk_store,
        index_manager=index_manager,
        dedup_cosine_threshold=0.95,
        dedup_text_threshold=0.9,
    )

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(embedding_provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25_index),
        rerank_provider=CrossEncoderReranker(),
        dense_weight=0.7, sparse_weight=0.3, rrf_k=60,
        dense_k=5, sparse_k=5, rerank_top_n=3, rerank_fused_top_n=10,
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
