from rag_hybrid_search.ingestion.chunkers.fixed import FixedChunker
from rag_hybrid_search.models import Document


def make_document(content):
    return Document(
        document_id="a" * 64, source_path="/docs/a.txt", content=content, format="text"
    )


def test_chunk_respects_size_and_overlap():
    content = "x" * 1000
    chunker = FixedChunker(chunk_size=300, chunk_overlap=50)

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) == 4
    assert chunks[0].char_count == 300
    assert chunks[0].text[-50:] == chunks[1].text[:50]
    assert all(c.strategy_version == "fixed-v1" for c in chunks)


def test_chunk_indexes_are_sequential():
    chunker = FixedChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk(make_document("y" * 250))

    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_short_document_produces_one_chunk():
    chunker = FixedChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(make_document("short text"))

    assert len(chunks) == 1
    assert chunks[0].text == "short text"
