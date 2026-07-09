import json

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

    def generate_stream(self, prompt: str, **kwargs):
        """Stream text deltas via Gemini's ``streamGenerateContent`` SSE endpoint.

        Each SSE frame is a full JSON chunk with the same shape as the
        non-streaming response; only the ``text`` piece from each chunk's
        first candidate is yielded, so callers get incremental text without
        needing to know the Gemini response envelope.
        """
        with self._client.stream(
            "POST",
            f"{_BASE_URL}/{self._generation_model}:streamGenerateContent",
            params={"key": self._api_key, "alt": "sse"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                **kwargs,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: "):])
                candidates = payload.get("candidates") or []
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    text = part.get("text")
                    if text:
                        yield text

    @property
    def model_name(self) -> str:
        return self._generation_model
