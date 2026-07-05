from rag_hybrid_search.ingestion.loaders.csv_loader import CsvLoader


def test_load_renders_rows_with_headers(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,age\nAlice,30\nBob,25\n")

    doc = CsvLoader().load(str(path))

    assert doc.format == "csv"
    assert "name: Alice, age: 30" in doc.content
    assert "name: Bob, age: 25" in doc.content
