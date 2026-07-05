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
