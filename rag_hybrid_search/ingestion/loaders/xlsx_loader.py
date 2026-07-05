import openpyxl

from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.models import Document


class XlsxLoader(Loader):
    format = "xlsx"

    def load(self, path: str) -> Document:
        workbook = openpyxl.load_workbook(path, data_only=True)
        sections = []
        for sheet in workbook.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            headers = [str(h) if h is not None else "" for h in rows[0]]
            lines = [f"Sheet: {sheet.title}"]
            for row in rows[1:]:
                lines.append(
                    ", ".join(
                        f"{header}: {value}"
                        for header, value in zip(headers, row)
                    )
                )
            sections.append("\n".join(lines))
        content = "\n\n".join(sections)
        return self._build_document(path, content)
