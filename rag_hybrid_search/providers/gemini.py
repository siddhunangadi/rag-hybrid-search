import httpx

from rag_hybrid_search.providers.base import GenerationProvider

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(GenerationProvider):
    def __init__(
        self,
        api_key: str,
        generation_model: str = "gemini-2.5-flash",
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._generation_model = generation_model
        self._client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.post(
            f"{_BASE_URL}/{self._generation_model}:generateContent",
            params={"key": self._api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    @property
    def model_name(self) -> str:
        return self._generation_model
