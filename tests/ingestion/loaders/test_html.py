from rag_hybrid_search.ingestion.loaders.html import HtmlLoader


def test_load_strips_tags_and_scripts(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><script>evil()</script></head>"
        "<body><h1>Title</h1><p>Body text.</p></body></html>"
    )

    doc = HtmlLoader().load(str(path))

    assert doc.format == "html"
    assert "evil()" not in doc.content
    assert "Title" in doc.content
    assert "Body text." in doc.content
