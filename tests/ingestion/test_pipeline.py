import pytest

from rag_hybrid_search.ingestion.chunkers.fixed import FixedChunker
from rag_hybrid_search.ingestion.loaders.text import TextLoader
from rag_hybrid_search.ingestion.pipeline import IngestionPipeline
from rag_hybrid_search.models import IndexStatus
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.chroma_store import ChromaVectorStore
from rag_hybrid_search.storage.chunk_store import SqliteChunkStore
from rag_hybrid_search.storage.index_manager import IndexManager
from tests.fakes import FakeEmbeddingProvider


@pytest.fixture
def pipeline(tmp_path):
    chunk_store = SqliteChunkStore(db_path=str(tmp_path / "chunks.db"))
    vector_store = ChromaVectorStore(data_dir=str(tmp_path / "chroma"))
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index_manager = IndexManager(chunk_store, vector_store, bm25)
    return IngestionPipeline(
        loader=TextLoader(),
        chunker=FixedChunker(chunk_size=100, chunk_overlap=0),
        embedding_provider=FakeEmbeddingProvider(),
        chunk_store=chunk_store,
        index_manager=index_manager,
        dedup_cosine_threshold=0.95,
        dedup_text_threshold=0.9,
    )


def test_ingest_produces_ready_status_and_chunks(pipeline, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Some content about hybrid retrieval systems.")

    status = pipeline.ingest(str(path))

    assert status == IndexStatus.READY
    chunks = list(pipeline.chunk_store.all())
    assert len(chunks) >= 1


def test_reingesting_unchanged_document_is_noop(pipeline, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Stable content that never changes.")

    pipeline.ingest(str(path))
    first_count = len(list(pipeline.chunk_store.all()))

    pipeline.ingest(str(path))
    second_count = len(list(pipeline.chunk_store.all()))

    assert first_count == second_count


def test_reingesting_edited_document_replaces_old_chunks(pipeline, tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Original content version one.")
    pipeline.ingest(str(path))
    original_ids = {c.chunk_id for c in pipeline.chunk_store.all()}

    path.write_text("Completely different content version two, much longer than before.")
    pipeline.ingest(str(path))
    new_ids = {c.chunk_id for c in pipeline.chunk_store.all()}

    assert original_ids.isdisjoint(new_ids)
    assert len(new_ids) >= 1


def test_dedup_skips_near_duplicate_chunk_across_documents(pipeline, tmp_path):
    path_a = tmp_path / "a.txt"
    path_a.write_text("The quick brown fox jumps over the lazy dog repeatedly.")
    path_b = tmp_path / "b.txt"
    path_b.write_text("The quick brown fox jumps over the lazy dog repeatedly.")

    pipeline.ingest(str(path_a))
    count_after_first = len(list(pipeline.chunk_store.all()))

    pipeline.ingest(str(path_b))
    count_after_second = len(list(pipeline.chunk_store.all()))

    assert count_after_second == count_after_first
