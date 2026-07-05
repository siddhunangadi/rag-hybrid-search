from rag_hybrid_search.ingestion.loaders.pdf import PdfLoader


def test_load_extracts_text_from_fixture_pdf():
    fixture_path = "tests/ingestion/loaders/fixtures/sample.pdf"

    doc = PdfLoader().load(fixture_path)

    assert doc.format == "pdf"
    assert "Sample PDF content for testing" in doc.content
