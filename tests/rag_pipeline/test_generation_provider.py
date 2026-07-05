from rag_hybrid_search.providers.nvidia import NvidiaProvider
from rag_pipeline.generation_provider import MockProvider, GenerationProvider

_DEFAULT_CANNED_JSON = (
    '{"answer": "mock answer", "claims": '
    '[{"text": "mock claim", "citation_ids": ["d1"], "supporting_quote": "mock quote"}]}'
)


def test_mock_provider_returns_default_canned_json():
    provider = MockProvider()
    result = provider.generate("any prompt")
    assert result == _DEFAULT_CANNED_JSON


def test_mock_provider_returns_custom_canned_json():
    custom = '{"answer": "custom", "claims": []}'
    provider = MockProvider(canned_json=custom)
    assert provider.generate("any prompt") == custom


def test_nvidia_provider_satisfies_generation_provider_protocol():
    provider = NvidiaProvider(api_key="test-key")
    assert isinstance(provider, GenerationProvider)
