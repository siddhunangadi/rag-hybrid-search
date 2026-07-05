import re

from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.models import Chunk, Document
from rag_hybrid_search.providers.base import EmbeddingProvider
from rag_hybrid_search.uuid7 import uuid7

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(Chunker):
    version = "semantic-v1"

    def __init__(self, embedding_provider: EmbeddingProvider, similarity_threshold: float = 0.5):
        self._embedding_provider = embedding_provider
        self._similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = [s.strip() for s in _SENTENCE_RE.split(document.content) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._make_chunk(document, 0, sentences[0])]

        embeddings = self._embedding_provider.embed(sentences)

        similarities = [
            _cosine(embeddings[i - 1], embeddings[i]) for i in range(1, len(sentences))
        ]
        lo, hi = min(similarities), max(similarities)
        spread = hi - lo
        # Normalize similarities relative to the document's own range so that
        # boundary detection works regardless of the embedding provider's
        # absolute similarity scale (some providers cluster all scores high).
        normalized = [
            1.0 if spread == 0 else (s - lo) / spread for s in similarities
        ]

        groups: list[list[str]] = [[sentences[0]]]
        for i, norm_similarity in enumerate(normalized, start=1):
            if norm_similarity >= self._similarity_threshold:
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        return [
            self._make_chunk(document, idx, " ".join(group))
            for idx, group in enumerate(groups)
        ]

    def _make_chunk(self, document: Document, index: int, text: str) -> Chunk:
        return Chunk(
            chunk_id=uuid7(),
            document_id=document.document_id,
            chunk_index=index,
            text=text,
            strategy_version=self.version,
            heading=None,
            page=None,
            char_count=len(text),
        )
