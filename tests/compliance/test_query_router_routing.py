from unittest.mock import MagicMock

from rag_hybrid_search.compliance.query_router import route_query
from rag_hybrid_search.models import Chunk, RetrievalTrace, RetrievedChunk


def _retrieved(chunk_id: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id, document_id="doc-1", chunk_index=0, text="t",
        strategy_version="clause-v1", char_count=1,
    )
    return RetrievedChunk(chunk=chunk, rrf_score=1.0, final_rank=0)


def test_structured_query_uses_metadata_filter_only_no_retriever_call():
    chunk_store = MagicMock()
    chunk_store.get_by_legal_metadata.return_value = [
        Chunk(chunk_id="c1", document_id="doc-1", chunk_index=0, text="Article 17 text",
              strategy_version="clause-v1", char_count=10)
    ]
    retriever = MagicMock()

    results, trace = route_query("Show Article 17", chunk_store, retriever)

    chunk_store.get_by_legal_metadata.assert_called_once_with({"article": "17"})
    retriever.retrieve.assert_not_called()
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"


def test_semantic_query_delegates_to_retriever_unchanged():
    chunk_store = MagicMock()
    retriever = MagicMock()
    retriever.retrieve.return_value = ([_retrieved("c2")], RetrievalTrace())

    results, trace = route_query("What is data minimization?", chunk_store, retriever)

    chunk_store.get_by_legal_metadata.assert_not_called()
    retriever.retrieve.assert_called_once_with("What is data minimization?", dev_trace=None)
    assert results[0].chunk.chunk_id == "c2"


def test_mixed_query_filters_then_retrieves():
    chunk_store = MagicMock()
    chunk_store.get_by_legal_metadata.return_value = [
        Chunk(chunk_id="c3", document_id="doc-1", chunk_index=0, text="Article 17 text",
              strategy_version="clause-v1", char_count=10)
    ]
    retriever = MagicMock()
    retriever.retrieve.return_value = ([_retrieved("c3")], RetrievalTrace())

    results, trace = route_query("Explain Article 17", chunk_store, retriever)

    chunk_store.get_by_legal_metadata.assert_called_once_with({"article": "17"})
    retriever.retrieve.assert_called_once()
    assert results[0].chunk.chunk_id == "c3"


def test_metadata_query_with_no_matching_chunks_returns_empty_not_unfiltered():
    chunk_store = MagicMock()
    chunk_store.get_by_legal_metadata.return_value = []
    retriever = MagicMock()
    retriever.retrieve.return_value = ([_retrieved("c9"), _retrieved("c10")], RetrievalTrace())

    results, trace = route_query("Show only HIPAA documents", chunk_store, retriever)

    chunk_store.get_by_legal_metadata.assert_called_once_with({"regulation": "HIPAA"})
    retriever.retrieve.assert_called_once_with("Show only HIPAA documents", dev_trace=None)
    assert results == []


def test_mixed_query_with_no_matching_chunks_returns_empty_not_unfiltered():
    chunk_store = MagicMock()
    chunk_store.get_by_legal_metadata.return_value = []
    retriever = MagicMock()
    retriever.retrieve.return_value = ([_retrieved("c11"), _retrieved("c12")], RetrievalTrace())

    results, trace = route_query("Explain Article 17 in plain terms", chunk_store, retriever)

    chunk_store.get_by_legal_metadata.assert_called_once_with({"article": "17"})
    retriever.retrieve.assert_called_once_with("Explain Article 17 in plain terms", dev_trace=None)
    assert results == []
