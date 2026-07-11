"""Retrieval benchmark over the sample doc corpus.

Runs the real ingestion + hybrid retrieval stack (BM25 + dense + RRF fusion +
CrossEncoder rerank) against a small fixed query set and reports recall@k and
mean reciprocal rank. Uses a deterministic fake embedding provider so results
are reproducible without an API key.

Run directly: `uv run python -m scripts.benchmark`
"""

import json
import shutil
import sys
from pathlib import Path

from rag_hybrid_search.ingestion.chunkers.recursive import RecursiveChunker
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

from tests.fakes import FakeEmbeddingProvider
from tests.fixtures.benchmark_queries import BENCHMARK_QUERIES

_SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "tests/fixtures/sample_docs"


class BenchmarkCorpus:
    def __init__(self, retriever: HybridRetriever, doc_id_by_filename: dict[str, str]):
        self.retriever = retriever
        self.doc_id_by_filename = doc_id_by_filename


def build_benchmark_corpus(tmp_path) -> BenchmarkCorpus:
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    bm25_index = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index_manager = IndexManager(chunk_store, vector_store, bm25_index)
    embedding_provider = FakeEmbeddingProvider()
    loader = MarkdownLoader()

    ingestion = IngestionPipeline(
        loader=loader,
        chunker=RecursiveChunker(chunk_size=500, chunk_overlap=50),
        embedding_provider=embedding_provider,
        chunk_store=chunk_store,
        index_manager=index_manager,
        dedup_cosine_threshold=0.95,
        dedup_text_threshold=0.9,
    )

    doc_id_by_filename = {}
    for doc_path in sorted(_SAMPLE_DOCS_DIR.glob("*.md")):
        ingestion.ingest(str(doc_path))
        doc_id_by_filename[doc_path.name] = loader.load(str(doc_path)).document_id

    retriever = HybridRetriever(
        dense_retriever=DenseRetriever(embedding_provider, vector_store, chunk_store),
        sparse_retriever=SparseRetriever(chunk_store, bm25_index),
        rerank_provider=CrossEncoderReranker(),
        dense_weight=0.7, sparse_weight=0.3, rrf_k=60,
        dense_k=5, sparse_k=5, rerank_top_n=3, rerank_fused_top_n=10,
    )
    return BenchmarkCorpus(retriever, doc_id_by_filename)


def run_benchmark(corpus: BenchmarkCorpus, top_k: int = 3) -> dict:
    hits_at_k = 0
    reciprocal_ranks = []
    per_query = []

    for case in BENCHMARK_QUERIES:
        expected_doc_id = corpus.doc_id_by_filename[case["expected_doc"]]
        retrieved, _trace = corpus.retriever.retrieve(case["query"])
        ranked = sorted(retrieved, key=lambda r: r.final_rank)[:top_k]

        rank_of_hit = None
        for i, chunk in enumerate(ranked, start=1):
            if chunk.chunk.document_id == expected_doc_id:
                rank_of_hit = i
                break

        hit = rank_of_hit is not None
        hits_at_k += int(hit)
        reciprocal_ranks.append(1.0 / rank_of_hit if hit else 0.0)
        per_query.append({
            "query": case["query"],
            "expected_doc": case["expected_doc"],
            "hit": hit,
            "rank": rank_of_hit,
        })

    num_queries = len(BENCHMARK_QUERIES)
    return {
        "num_queries": num_queries,
        "recall_at_3": hits_at_k / num_queries,
        "mrr": sum(reciprocal_ranks) / num_queries,
        "per_query": per_query,
    }


def main() -> None:
    tmp_dir = Path(__file__).resolve().parent.parent / ".benchmark_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    try:
        corpus = build_benchmark_corpus(tmp_dir)
        results = run_benchmark(corpus, top_k=3)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
