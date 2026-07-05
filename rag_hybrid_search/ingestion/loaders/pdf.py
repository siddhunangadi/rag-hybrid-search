from pypdf import PdfReader

from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class PdfLoader(Loader):
    format = "pdf"

    def load(self, path: str) -> Document:
        reader = PdfReader(path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        content = "\n".join(pages_text)
        return self._build_document(path, content)
