from openpyxl import Workbook

from rag_hybrid_search.ingestion.loaders.xlsx_loader import XlsxLoader


def test_load_renders_sheets_and_rows(tmp_path):
    path = tmp_path / "data.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["name", "age"])
    ws.append(["Alice", 30])
    ws.append(["Bob", 25])
    wb.save(str(path))

    doc = XlsxLoader().load(str(path))

    assert doc.format == "xlsx"
    assert "Sheet1" in doc.content
    assert "name: Alice, age: 30" in doc.content
    assert "name: Bob, age: 25" in doc.content
