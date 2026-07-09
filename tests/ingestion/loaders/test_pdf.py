from rag_hybrid_search.ingestion.loaders.pdf import PdfLoader


def test_load_extracts_text_from_fixture_pdf():
    fixture_path = "tests/ingestion/loaders/fixtures/sample.pdf"

    doc = PdfLoader().load(fixture_path)

    assert doc.format == "pdf"
    assert "Sample PDF content for testing" in doc.content


def test_load_preserves_word_spacing_on_tight_kerning_pdf():
    """pdfplumber's default x_tolerance merges adjacent words on some PDFs'
    tightly-kerned fonts (no explicit space glyphs), producing run-on text
    like 'MusfiqurRahman'. This must not happen -- word boundaries in the
    source PDF must survive extraction."""
    fixture_path = "tests/ingestion/loaders/fixtures/tight_kerning.pdf"

    doc = PdfLoader().load(fixture_path)

    assert "Musfiqur Rahman" in doc.content
    assert "MusfiqurRahman" not in doc.content
