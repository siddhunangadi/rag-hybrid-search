from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class TextLoader(Loader):
    format = "text"

    def load(self, path: str) -> Document:
        content = self._read_text_file(path)
        return self._build_document(path, content)
