import httpx

from rag_hybrid_search.providers.base import EmbeddingProvider, GenerationProvider

_BASE_URL = "https://integrate.api.nvidia.com/v1"

_MODEL_DIMENSIONS = {
    "nvidia/nv-embedqa-e5-v5": 1024,
    "nvidia/nv-embed-v2": 4096,
}


class NvidiaProvider(EmbeddingProvider, GenerationProvider):
    def __init__(
        self,
        api_key: str,
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        generation_model: str = "meta/llama-3.1-70b-instruct",
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._embedding_model = embedding_model
        self._generation_model = generation_model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            f"{_BASE_URL}/embeddings",
            json={
                "input": texts,
                "model": self._embedding_model,
                "input_type": "passage",
            },
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.post(
            f"{_BASE_URL}/chat/completions",
            json={
                "model": self._generation_model,
                "messages": [{"role": "user", "content": prompt}],
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def model_name(self) -> str:
        return self._embedding_model

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS.get(self._embedding_model, 1024)
