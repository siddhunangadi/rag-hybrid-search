import httpx
import pytest

from rag_hybrid_search.providers.gemini import GeminiProvider


@pytest.fixture
def provider():
    return GeminiProvider(api_key="test-key")


def test_generate_calls_generate_content_endpoint(provider, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={"candidates": [{"content": {"parts": [{"text": "an answer"}]}}]},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"),
    )
    mock_post = mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    result = provider.generate("What is RAG?")

    assert result == "an answer"
    called_url = mock_post.call_args[0][0]
    assert "gemini-1.5-flash:generateContent" in called_url
    assert mock_post.call_args.kwargs["params"] == {"key": "test-key"}


def test_generate_sends_prompt_as_content(provider, mocker):
    mock_response = httpx.Response(
        status_code=200,
        json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]},
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"),
    )
    mock_post = mocker.patch.object(httpx.Client, "post", return_value=mock_response)

    provider.generate("hello world")

    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["contents"][0]["parts"][0]["text"] == "hello world"


def test_model_name_defaults_to_gemini_1_5_flash():
    provider = GeminiProvider(api_key="test-key")
    assert provider.model_name == "gemini-1.5-flash"
