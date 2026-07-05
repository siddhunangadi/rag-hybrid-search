from abc import ABC, abstractmethod

from rag_hybrid_search.models import Chunk, Document


class Chunker(ABC):
    version: str

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...
