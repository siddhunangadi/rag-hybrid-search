from rag_hybrid_search.ingestion.chunkers.semantic import SemanticChunker
from rag_hybrid_search.models import Document
from rag_hybrid_search.providers.base import EmbeddingProvider
from tests.fakes import FakeEmbeddingProvider


def make_document(content):
    return Document(
        document_id="c" * 64, source_path="/docs/c.txt", content=content, format="text"
    )


class StubEmbeddingProvider(EmbeddingProvider):
    """Test-local embedding stub that returns hand-crafted, discriminative
    vectors so adjacent-sentence cosine similarity can be controlled exactly.

    Unlike the shared `FakeEmbeddingProvider` (trigram-hash based, low-dim),
    which produces uniformly high absolute cosine similarity (0.79-0.93) for
    any short English sentences, this stub maps each expected input text to
    an explicit orthogonal-ish vector so tests can exercise a real absolute
    similarity threshold rather than incidental variance.
    """

    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self._vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors_by_text[text] for text in texts]

    @property
    def model_name(self) -> str:
        return "stub-embedding-v1"

    @property
    def dimension(self) -> int:
        return 4


def test_splits_on_topic_boundary():
    content = (
        "The soup needs more salt. Add pepper to taste. "
        "The moon orbits the earth. Stars burn hydrogen for fuel."
    )
    # Cooking sentences cluster tightly on one axis; astronomy sentences
    # cluster tightly on an orthogonal axis, so the adjacent pair straddling
    # the topic boundary has ~0 cosine similarity while within-topic pairs
    # have high similarity.
    vectors = {
        "The soup needs more salt.": [1.0, 0.0, 0.0, 0.0],
        "Add pepper to taste.": [0.9, 0.1, 0.0, 0.0],
        "The moon orbits the earth.": [0.0, 0.0, 1.0, 0.0],
        "Stars burn hydrogen for fuel.": [0.0, 0.0, 0.9, 0.1],
    }
    chunker = SemanticChunker(
        embedding_provider=StubEmbeddingProvider(vectors), similarity_threshold=0.5
    )

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) == 2
    assert chunks[0].text == "The soup needs more salt. Add pepper to taste."
    assert chunks[1].text == "The moon orbits the earth. Stars burn hydrogen for fuel."
    assert all(c.strategy_version == "semantic-v1" for c in chunks)


def test_does_not_split_when_all_sentences_are_similar():
    content = "The soup needs more salt. Add pepper to taste. Season with herbs too."
    vectors = {
        "The soup needs more salt.": [1.0, 0.0, 0.0, 0.0],
        "Add pepper to taste.": [0.95, 0.05, 0.0, 0.0],
        "Season with herbs too.": [0.9, 0.1, 0.0, 0.0],
    }
    chunker = SemanticChunker(
        embedding_provider=StubEmbeddingProvider(vectors), similarity_threshold=0.5
    )

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) == 1


def test_single_sentence_document_produces_one_chunk():
    chunker = SemanticChunker(
        embedding_provider=FakeEmbeddingProvider(), similarity_threshold=0.3
    )
    chunks = chunker.chunk(make_document("Just one sentence here."))

    assert len(chunks) == 1
