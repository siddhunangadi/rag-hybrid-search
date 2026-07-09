from typing import Protocol, runtime_checkable

_DEFAULT_CANNED_JSON = (
    '{"answer": "mock answer", "claims": '
    '[{"text": "mock claim", "citation_ids": ["d1"], "supporting_quote": "mock quote"}]}'
)


@runtime_checkable
class GenerationProvider(Protocol):
    def generate(self, prompt: str, **kwargs) -> str: ...


class MockProvider:
    def __init__(self, canned_json: str | None = None):
        self._canned_json = canned_json or _DEFAULT_CANNED_JSON

    def generate(self, prompt: str, **kwargs) -> str:
        return self._canned_json

    def generate_stream(self, prompt: str, **kwargs):
        """Dev/demo fallback streaming: yields the canned JSON in a few chunks
        so the SSE code path is exercisable without a real API key."""
        text = self._canned_json
        chunk_size = max(1, len(text) // 5)
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
