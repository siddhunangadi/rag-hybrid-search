"""Developer-only retrieval debugger.

Two modes:

local mode -- indexes one document into a scratch data dir (never touches
your real ``./data``), runs a query through every pipeline stage, and prints
each stage's output: raw PDF text extraction, chunk boundaries, BM25 top-k,
dense top-k, RRF fusion top-k, the exact prompt, and the raw generation
output before citation verification. Uses real providers from env
(RAG_NVIDIA_API_KEY / RAG_GEMINI_API_KEY).

live mode -- calls GET /debug/retrieval on an already-deployed instance,
tracing the same stages against data already indexed there. Requires
RAG_DEBUG_TOKEN to be set on that deployment and passed here.

Run:
  uv run python -m scripts.debug_retrieval local <path/to/doc.pdf> "<query>"
  uv run python -m scripts.debug_retrieval live <base_url> "<query>" <debug_token>
"""

import shutil
import sys
from pathlib import Path

import httpx
from pypdf import PdfReader

from rag_hybrid_search.config import Settings
from rag_hybrid_search.ingestion.chunkers.recursive import RecursiveChunker
from rag_hybrid_search.ingestion.loaders.pdf import PdfLoader
from rag_hybrid_search.ingestion.pipeline import IngestionPipeline
from rag_hybrid_search.retrieval.dense import DenseRetriever
from rag_hybrid_search.retrieval.passthrough_rerank import PassthroughReranker
from rag_hybrid_search.retrieval.retriever import HybridRetriever
from rag_hybrid_search.retrieval.sparse import SparseRetriever
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.index_manager import IndexManager
from rag_hybrid_search.models import ChunkProvenance, ContextChunk
from rag_pipeline.context_builder import ContextLayout, build_context
from rag_pipeline.prompt_builder import build_prompt
from tests.fakes import fake_pinecone_stores

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.dependencies import _select_embedding_provider, _select_generation_provider  # noqa: E402

_PREVIEW_PAGES = 3
_PREVIEW_CHARS_PER_PAGE = 1500
_TOP_K = 10


def _print_header(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def _print_raw_pdf_preview(path: str) -> None:
    _print_header("RAW PDF TEXT (first pages, pre-chunking)")
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages[:_PREVIEW_PAGES]):
        text = page.extract_text() or ""
        print(f"\n--- page {i + 1} ---")
        print(text[:_PREVIEW_CHARS_PER_PAGE])


def _print_chunk_boundaries(chunks) -> None:
    _print_header("CHUNK BOUNDARIES")
    for c in chunks[:20]:
        preview = c.text[:120].replace("\n", " ")
        print(f"chunk_index={c.chunk_index:>4}  chars={c.char_count:>5}  {preview!r}")
    if len(chunks) > 20:
        print(f"... ({len(chunks) - 20} more chunks)")


def _print_retrieved(title: str, results, score_attr: str) -> None:
    _print_header(title)
    ranked = sorted(results, key=lambda r: getattr(r, score_attr) or 0.0, reverse=True)[:_TOP_K]
    for r in ranked:
        score = getattr(r, score_attr)
        preview = r.chunk.text[:200].replace("\n", " ")
        print(f"\nchunk_id={r.chunk.chunk_id}  chunk_index={r.chunk.chunk_index}  {score_attr}={score}")
        print(f"text: {preview!r}")


def _print_retrieved_dicts(title: str, results: list[dict]) -> None:
    _print_header(title)
    for r in results[:_TOP_K]:
        preview = r["text"][:200].replace("\n", " ")
        print(f"\nchunk_id={r['chunk_id']}  chunk_index={r['chunk_index']}  score={r['score']}")
        print(f"text: {preview!r}")


def run_local(doc_path: str, query: str) -> None:
    _print_raw_pdf_preview(doc_path)

    settings = Settings()
    embedding_provider, embedding_name, nvidia_provider = _select_embedding_provider(settings)
    generation_provider, generation_name = _select_generation_provider(settings, nvidia_provider)
    print(f"\nembedding provider: {embedding_name}   generation provider: {generation_name}")

    tmp_dir = Path(__file__).resolve().parent.parent / ".debug_retrieval_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    try:
        chunk_store, vector_store = fake_pinecone_stores(embedding_dimension=embedding_provider.dimension)
        bm25_index = BM25Index(index_path=str(tmp_dir / "bm25.pkl"))
        index_manager = IndexManager(chunk_store, vector_store, bm25_index)

        loader = PdfLoader()
        chunker = RecursiveChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        ingestion = IngestionPipeline(
            loader=loader, chunker=chunker, embedding_provider=embedding_provider,
            chunk_store=chunk_store, index_manager=index_manager,
            dedup_cosine_threshold=settings.dedup_cosine_threshold,
            dedup_text_threshold=settings.dedup_text_similarity_threshold,
        )
        ingestion.ingest(doc_path)

        document = loader.load(doc_path)
        chunks = chunker.chunk(document)
        _print_chunk_boundaries(chunks)

        dense_retriever = DenseRetriever(embedding_provider, vector_store, chunk_store)
        sparse_retriever = SparseRetriever(chunk_store, bm25_index)

        dense_results = dense_retriever.search(query, k=settings.dense_k)
        _print_retrieved("DENSE TOP-K", dense_results, "dense_score")

        sparse_results = sparse_retriever.search(query, k=settings.sparse_k)
        _print_retrieved("BM25 TOP-K", sparse_results, "bm25_score")

        retriever = HybridRetriever(
            dense_retriever=dense_retriever, sparse_retriever=sparse_retriever,
            rerank_provider=PassthroughReranker(),
            dense_weight=settings.rrf_dense_weight, sparse_weight=settings.rrf_sparse_weight,
            rrf_k=settings.rrf_k, dense_k=settings.dense_k, sparse_k=settings.sparse_k,
            rerank_top_n=settings.rerank_top_n, rerank_fused_top_n=settings.rerank_fused_top_n,
        )
        reranked, _trace = retriever.retrieve(query)
        _print_retrieved("RRF FUSION TOP-K (post-rerank)", reranked, "rrf_score")

        top_chunks = [
            ContextChunk(
                chunk=r,
                provenance=ChunkProvenance(primary_subquery=0, all_subqueries=[0]),
            )
            for r in sorted(reranked, key=lambda r: r.final_rank)[:5]
        ]
        context = build_context(top_chunks, subqueries=[], layout=ContextLayout.FLAT)
        prompt = build_prompt(query, context)
        _print_header("EXACT PROMPT SENT TO GENERATION PROVIDER")
        print(prompt)

        raw_output = generation_provider.generate(prompt)
        _print_header("RAW GENERATION OUTPUT (pre citation-verification)")
        print(raw_output)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_live(base_url: str, query: str, debug_token: str) -> None:
    response = httpx.get(
        f"{base_url.rstrip('/')}/debug/retrieval",
        params={"query": query},
        headers={"X-Debug-Token": debug_token},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    _print_retrieved_dicts("DENSE TOP-K (live)", data["dense_results"])
    _print_retrieved_dicts("BM25 TOP-K (live)", data["bm25_results"])
    _print_retrieved_dicts("RRF FUSION TOP-K (live)", data["rrf_results"])
    _print_header("EXACT PROMPT SENT TO GENERATION PROVIDER (live)")
    print(data["prompt"])
    _print_header("RAW GENERATION OUTPUT (live, pre citation-verification)")
    print(data["raw_generation"])


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("local", "live"):
        print(
            "usage:\n"
            '  python -m scripts.debug_retrieval local <path/to/doc> "<query>"\n'
            '  python -m scripts.debug_retrieval live <base_url> "<query>" <debug_token>'
        )
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "local":
        if len(sys.argv) != 4:
            print('usage: python -m scripts.debug_retrieval local <path/to/doc> "<query>"')
            sys.exit(1)
        run_local(sys.argv[2], sys.argv[3])
    else:
        if len(sys.argv) != 5:
            print('usage: python -m scripts.debug_retrieval live <base_url> "<query>" <debug_token>')
            sys.exit(1)
        run_live(sys.argv[2], sys.argv[3], sys.argv[4])


if __name__ == "__main__":
    main()
