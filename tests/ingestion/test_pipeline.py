import pytest

from rag_hybrid_search.ingestion.chunkers.fixed import FixedChunker
from rag_hybrid_search.ingestion.loaders.text import TextLoader
from rag_hybrid_search.ingestion.pipeline import IngestionPipeline
from rag_hybrid_search.models import IndexStatus
from rag_hybrid_search.storage.bm25_index import BM25Index
from rag_hybrid_search.storage.index_manager import IndexManager
from tests.fakes import FakeEmbeddingProvider, fake_pinecone_stores


class _CountingEmbeddingProvider(FakeEmbeddingProvider):
    """Counts embed() calls and how many texts were embedded in total, to
    prove ingestion doesn't re-embed the existing corpus on every ingest()."""

    def __init__(self):
        self.call_count = 0
        self.total_texts_embedded = 0

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        self.call_count += 1
        self.total_texts_embedded += len(texts)
        return super().embed(texts, input_type)


@pytest.fixture
def pipeline(tmp_path):
    chunk_store, vector_store = fake_pinecone_stores()
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


def test_ingesting_new_document_does_not_re_embed_existing_chunks(tmp_path):
    """Regression test for the dedup re-embedding fix: ingesting a second,
    unrelated document must embed only the NEW document's texts, not every
    previously-ingested chunk's text again."""
    chunk_store, vector_store = fake_pinecone_stores()
    bm25 = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    index_manager = IndexManager(chunk_store, vector_store, bm25)
    provider = _CountingEmbeddingProvider()
    pipeline = IngestionPipeline(
        loader=TextLoader(),
        chunker=FixedChunker(chunk_size=100, chunk_overlap=0),
        embedding_provider=provider,
        chunk_store=chunk_store,
        index_manager=index_manager,
        dedup_cosine_threshold=0.95,
        dedup_text_threshold=0.9,
    )

    path_a = tmp_path / "a.txt"
    path_a.write_text("First document about hybrid retrieval systems.")
    pipeline.ingest(str(path_a))
    texts_embedded_after_first = provider.total_texts_embedded
    assert texts_embedded_after_first > 0

    path_b = tmp_path / "b.txt"
    path_b.write_text("Second, completely unrelated document about baking bread.")
    pipeline.ingest(str(path_b))
    texts_embedded_for_second_ingest = provider.total_texts_embedded - texts_embedded_after_first

    b_chunks = FixedChunker(chunk_size=100, chunk_overlap=0).chunk(TextLoader().load(str(path_b)))
    assert texts_embedded_for_second_ingest == len(b_chunks)
