from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class MarkdownLoader(Loader):
    format = "markdown"

    def load(self, path: str) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self._build_document(path, content)
