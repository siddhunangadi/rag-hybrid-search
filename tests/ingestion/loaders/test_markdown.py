import hashlib

from rag_hybrid_search.ingestion.loaders.markdown import MarkdownLoader


def test_load_normalizes_content_and_computes_hash(tmp_path):
    path = tmp_path / "readme.md"
    content = "# Title\n\nSome body text.\n"
    path.write_text(content)

    doc = MarkdownLoader().load(str(path))

    assert doc.format == "markdown"
    assert doc.source_path == str(path)
    assert doc.content == content
    assert doc.document_id == hashlib.sha256(content.encode()).hexdigest()


def test_same_content_produces_same_document_id(tmp_path):
    path_a = tmp_path / "a.md"
    path_b = tmp_path / "b.md"
    path_a.write_text("identical content")
    path_b.write_text("identical content")

    doc_a = MarkdownLoader().load(str(path_a))
    doc_b = MarkdownLoader().load(str(path_b))

    assert doc_a.document_id == doc_b.document_id
