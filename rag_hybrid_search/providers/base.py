from abc import ABC, abstractmethod

from rag_hybrid_search.models import RetrievedChunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        """Embed texts. ``input_type`` distinguishes query vs. passage embedding

        for asymmetric models (e.g. NVIDIA's e5-v5 QA embedder), where a query
        embedded as a passage lands in the wrong vector space and degrades
        dense retrieval quality.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class GenerationProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...


class RerankProvider(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        ...
