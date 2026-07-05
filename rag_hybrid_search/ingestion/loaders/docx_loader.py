import docx

from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class DocxLoader(Loader):
    format = "docx"

    def load(self, path: str) -> Document:
        document = docx.Document(path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        content = "\n".join(paragraphs)
        return self._build_document(path, content)
