from abc import ABC, abstractmethod

from rag_hybrid_search.models import RetrievedChunk


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
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
