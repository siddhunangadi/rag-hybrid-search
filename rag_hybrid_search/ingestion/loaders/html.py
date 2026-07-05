from bs4 import BeautifulSoup

from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class HtmlLoader(Loader):
    format = "html"

    def load(self, path: str) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            raw_html = f.read()
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        content = soup.get_text(separator="\n", strip=True)
        return self._build_document(path, content)
