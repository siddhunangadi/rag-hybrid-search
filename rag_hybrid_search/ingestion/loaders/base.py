import hashlib
from abc import ABC, abstractmethod
from typing import Literal

from rag_hybrid_search.models import Document


class Loader(ABC):
    format: Literal["markdown", "html", "text", "pdf", "csv", "xlsx", "docx"]

    @abstractmethod
    def load(self, path: str) -> Document:
        ...

    def _build_document(self, path: str, content: str) -> Document:
        document_id = hashlib.sha256(content.encode()).hexdigest()
        return Document(
            document_id=document_id,
            source_path=path,
            content=content,
            format=self.format,
        )
