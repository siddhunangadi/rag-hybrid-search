import csv

from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class CsvLoader(Loader):
    format = "csv"

    def load(self, path: str) -> Document:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            lines = [
                ", ".join(f"{key}: {value}" for key, value in row.items())
                for row in reader
            ]
        content = "\n".join(lines)
        return self._build_document(path, content)
