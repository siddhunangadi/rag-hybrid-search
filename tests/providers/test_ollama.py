import httpx
import pytest

from rag_hybrid_search.providers.ollama import OllamaProvider


@pytest.fixture
def provider():
    return OllamaProvider()


def test_embed_calls_per_text_and_collects_in_order(provider, mocker):
    def fake_post(url, json):
        text = json["prompt"]
        value = 0.1 if text == "hello" else 0.2
        return httpx.Response(
            status_code=200,
            json={"embedding": [value, value, value]},
            request=httpx.Request("POST", url),
        )

    mocker.patch.object(httpx.Client, "post", side_effect=fake_post)

    result = provider.embed(["hello", "world"])

    assert result == [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]


def test_generate_calls_generate_endpoint(provider, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={"response": "an answer"},
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )
    mock_post = mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    result = provider.generate("What is RAG?")

    assert result == "an answer"
    assert mock_post.call_args[0][0] == "http://localhost:11434/api/generate"


def test_model_name_and_dimension(provider):
    assert provider.model_name == "nomic-embed-text"
    assert provider.dimension == 768
