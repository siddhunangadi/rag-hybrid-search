import hashlib

from rag_hybrid_search.providers.base import EmbeddingProvider, GenerationProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding stand-in for tests.

    Produces an 8-dim vector derived from character trigram hashes so that
    textually similar strings land close together in cosine space, which is
    enough to exercise dense retrieval and dedup logic without a real model.
    """

    _DIM = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._DIM
        normalized = text.lower()
        for i in range(len(normalized) - 2):
            trigram = normalized[i : i + 3]
            digest = hashlib.sha256(trigram.encode()).digest()
            bucket = digest[0] % self._DIM
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    @property
    def model_name(self) -> str:
        return "fake-embedding-v1"

    @property
    def dimension(self) -> int:
        return self._DIM


class FakeGenerationProvider(GenerationProvider):
    def __init__(self, fixed_response: str = "fake response"):
        self._fixed_response = fixed_response

    def generate(self, prompt: str, **kwargs) -> str:
        return self._fixed_response
