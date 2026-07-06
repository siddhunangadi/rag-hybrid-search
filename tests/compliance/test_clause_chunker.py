from rag_hybrid_search.compliance.clause_chunker import ClauseChunker
from rag_hybrid_search.models import Document

_GDPR_TEXT = """Article 5

1. Personal data shall be processed lawfully, fairly and in a transparent manner.

2. The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1.

Article 17

1. The data subject shall have the right to obtain from the controller the erasure of personal data.
"""

_UNSTRUCTURED_TEXT = "Just a plain memo with no legal structure of any kind whatsoever."


def _doc(text: str, document_id: str = "doc-1") -> Document:
    return Document(document_id=document_id, source_path="/tmp/x.txt", content=text, format="text")


def test_chunks_have_legal_metadata_per_clause():
    chunker = ClauseChunker(document_title="GDPR")
    chunks = chunker.chunk(_doc(_GDPR_TEXT))
    assert len(chunks) >= 2
    articles = {c.legal_metadata.article for c in chunks}
    assert "5" in articles
    assert "17" in articles
    for c in chunks:
        assert c.strategy_version == "clause-v1"
        assert c.document_id == "doc-1"


def test_chunk_ids_are_unique():
    chunker = ClauseChunker(document_title="GDPR")
    chunks = chunker.chunk(_doc(_GDPR_TEXT))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_falls_back_to_single_chunk_for_unstructured_document():
    chunker = ClauseChunker(document_title="Memo")
    chunks = chunker.chunk(_doc(_UNSTRUCTURED_TEXT, document_id="doc-2"))
    assert len(chunks) == 1
    assert chunks[0].legal_metadata.article is None
    assert chunks[0].text == _UNSTRUCTURED_TEXT


def test_empty_document_returns_no_chunks():
    chunker = ClauseChunker(document_title="Empty")
    chunks = chunker.chunk(_doc("", document_id="doc-3"))
    assert chunks == []
