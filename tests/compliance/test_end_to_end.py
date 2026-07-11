import tempfile
from datetime import datetime, timezone

from rag_hybrid_search.compliance.citation_mapper import build_citations
from rag_hybrid_search.compliance.clause_chunker import ClauseChunker
from rag_hybrid_search.compliance.query_router import route_query
from rag_hybrid_search.models import Document, EmbeddingRecord
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.passthrough_rerank import PassthroughReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager
from tests.fakes import FakeEmbeddingProvider

_GDPR_TEXT = """Article 5

1. Personal data shall be processed lawfully, fairly and in a transparent manner.

Article 17

1. The data subject shall have the right to obtain from the controller the erasure of personal data.

3. Paragraph 1 shall not apply to the extent that processing is necessary for compliance with a legal obligation.
"""


def _build_pipeline_components(
    tmp_dir: str,
    regulation: str | None = None,
    jurisdiction: str | None = None,
):
    chunk_store = SqliteChunkStore(db_path=f"{tmp_dir}/chunks.db")
    vector_store = ChromaVectorStore(data_dir=f"{tmp_dir}/chroma")
    bm25_index = BM25Index(index_path=f"{tmp_dir}/bm25.pkl")
    index_manager = IndexManager(chunk_store, vector_store, bm25_index)
    embedding_provider = FakeEmbeddingProvider()

    document = Document(document_id="doc-gdpr", source_path="/tmp/gdpr.txt", content=_GDPR_TEXT, format="text")
    chunker = ClauseChunker(
        document_title="GDPR Consolidated Text",
        regulation=regulation,
        jurisdiction=jurisdiction,
    )
    chunks = chunker.chunk(document)

    embeddings = embedding_provider.embed([c.text for c in chunks])
    records = [
        EmbeddingRecord(
            chunk_id=c.chunk_id, embedding=e, embedding_model=embedding_provider.model_name,
            embedding_dimension=embedding_provider.dimension, provider="FakeEmbeddingProvider",
            created_at=datetime.now(timezone.utc),
        )
        for c, e in zip(chunks, embeddings)
    ]
    for chunk in chunks:
        chunk_store.put(chunk, source_path=document.source_path)
    index_manager.index(chunks, records)

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(embedding_provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25_index),
        rerank_provider=PassthroughReranker(),
        dense_weight=0.7, sparse_weight=0.3, rrf_k=60,
        dense_k=10, sparse_k=10, rerank_top_n=5, rerank_fused_top_n=20,
    )
    return chunk_store, retriever


def test_structured_query_returns_exact_article():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp)
        results, _trace = route_query("Show Article 17", chunk_store, retriever)
        assert len(results) >= 1
        assert all(r.chunk.legal_metadata.article == "17" for r in results)


def test_semantic_query_returns_results_via_hybrid_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp)
        results, _trace = route_query("What does the regulation say about data processing?", chunk_store, retriever)
        assert len(results) >= 1


def test_mixed_query_filters_to_matching_article():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp)
        results, _trace = route_query("Explain Article 5", chunk_store, retriever)
        assert len(results) >= 1
        assert all(r.chunk.legal_metadata.article == "5" for r in results)


def test_metadata_query_returns_chunks_scoped_to_regulation():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp, regulation="GDPR", jurisdiction="EU")
        results, _trace = route_query("Show only GDPR documents", chunk_store, retriever)
        assert len(results) >= 1
        assert all(r.chunk.legal_metadata.regulation == "GDPR" for r in results)


def test_citations_built_from_structured_query_results():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store, retriever = _build_pipeline_components(tmp)
        results, _trace = route_query("Show Article 17", chunk_store, retriever)
        citations = build_citations(results)
        assert len(citations) >= 1
        assert citations[0].article == "17"
        assert "Art. 17" in citations[0].display
