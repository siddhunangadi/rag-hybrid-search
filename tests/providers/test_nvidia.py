import httpx
import pytest

from rag_hybrid_search.providers.nvidia import NvidiaProvider


@pytest.fixture
def provider():
    return NvidiaProvider(api_key="test-key")


def test_embed_calls_expected_endpoint_and_parses_response(provider, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        },
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/embeddings"),
    )
    mock_post = mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    result = provider.embed(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    called_url = mock_post.call_args[0][0]
    assert called_url == "https://integrate.api.nvidia.com/v1/embeddings"
    called_json = mock_post.call_args.kwargs["json"]
    assert called_json["input"] == ["hello", "world"]


def test_generate_calls_chat_completions_and_returns_content(provider, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "an answer"}}]},
        request=httpx.Request(
            "POST", "https://integrate.api.nvidia.com/v1/chat/completions"
        ),
    )
    mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    result = provider.generate("What is RAG?")

    assert result == "an answer"


def test_model_name_and_dimension_reflect_configured_model(provider):
    assert provider.model_name == "nvidia/nv-embedqa-e5-v5"
    assert provider.dimension == 1024
