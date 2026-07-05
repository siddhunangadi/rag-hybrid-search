from rag_hybrid_search.ingestion.loaders.text import TextLoader


def test_load_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("just plain text\nwith two lines")

    doc = TextLoader().load(str(path))

    assert doc.format == "text"
    assert "just plain text" in doc.content
