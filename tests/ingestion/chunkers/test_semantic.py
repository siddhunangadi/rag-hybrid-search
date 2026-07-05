from rag_hybrid_search.ingestion.chunkers.semantic import SemanticChunker
from rag_hybrid_search.models import Document
from tests.fakes import FakeEmbeddingProvider


def make_document(content):
    return Document(
        document_id="c" * 64, source_path="/docs/c.txt", content=content, format="text"
    )


def test_splits_on_topic_boundary():
    content = (
        "The soup needs more salt. Add pepper to taste. "
        "The moon orbits the earth. Stars burn hydrogen for fuel."
    )
    chunker = SemanticChunker(
        embedding_provider=FakeEmbeddingProvider(), similarity_threshold=0.3
    )

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) >= 2
    assert all(c.strategy_version == "semantic-v1" for c in chunks)


def test_single_sentence_document_produces_one_chunk():
    chunker = SemanticChunker(
        embedding_provider=FakeEmbeddingProvider(), similarity_threshold=0.3
    )
    chunks = chunker.chunk(make_document("Just one sentence here."))

    assert len(chunks) == 1
