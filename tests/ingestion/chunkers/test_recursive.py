from rag_hybrid_search.ingestion.chunkers.recursive import RecursiveChunker
from rag_hybrid_search.models import Document


def make_document(content):
    return Document(
        document_id="b" * 64, source_path="/docs/b.md", content=content, format="markdown"
    )


def test_splits_on_markdown_headers_and_tags_heading():
    content = (
        "# Intro\n\nSome intro text.\n\n"
        "## Setup\n\nSetup instructions here.\n\n"
        "## Usage\n\nUsage instructions here."
    )
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)

    chunks = chunker.chunk(make_document(content))

    headings = [c.heading for c in chunks]
    assert "Intro" in headings
    assert "Setup" in headings
    assert "Usage" in headings
    assert all(c.strategy_version == "recursive-v1" for c in chunks)


def test_splits_large_section_further_by_chunk_size():
    content = "# Section\n\n" + ("word " * 400)
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) > 1
    assert all(c.heading == "Section" for c in chunks)
