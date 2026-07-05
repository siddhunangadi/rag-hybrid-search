import httpx
import pytest

from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_hybrid_search.providers.nvidia_rerank import NvidiaRerankProvider


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text=text,
        strategy_version="v1", heading=None, page=None, char_count=len(text),
    )


@pytest.fixture
def provider():
    return NvidiaRerankProvider(api_key="test-key")


@pytest.fixture
def candidates():
    return [
        RetrievedChunk(chunk=_chunk("a", "irrelevant passage"), rrf_score=0.5, final_rank=0),
        RetrievedChunk(chunk=_chunk("b", "relevant passage"), rrf_score=0.4, final_rank=0),
    ]


def test_rerank_calls_expected_endpoint_with_query_and_passages(provider, candidates, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={"rankings": [{"index": 1, "logit": 0.9}, {"index": 0, "logit": 0.1}]},
        request=httpx.Request(
            "POST", "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        ),
    )
    mock_post = mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    provider.rerank("a query", candidates, top_n=2)

    called_url = mock_post.call_args[0][0]
    assert called_url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["query"] == {"text": "a query"}
    assert called_json["passages"] == [
        {"text": "irrelevant passage"}, {"text": "relevant passage"}
    ]


def test_rerank_orders_by_returned_logit_and_truncates(provider, candidates, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={"rankings": [{"index": 1, "logit": 0.9}, {"index": 0, "logit": 0.1}]},
        request=httpx.Request(
            "POST", "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        ),
    )
    mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    result = provider.rerank("a query", candidates, top_n=1)

    assert len(result) == 1
    assert result[0].chunk.chunk_id == "b"
    assert result[0].rerank_score == 0.9
    assert result[0].final_rank == 1


def test_rerank_handles_empty_candidates(provider):
    assert provider.rerank("q", [], top_n=3) == []
