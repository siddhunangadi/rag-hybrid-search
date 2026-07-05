import httpx

from rag_hybrid_search.providers.base import EmbeddingProvider, GenerationProvider

_MODEL_DIMENSIONS = {
    "nomic-embed-text": 768,
}


class OllamaProvider(EmbeddingProvider, GenerationProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        generation_model: str = "llama3.1",
        timeout: float = 30.0,
    ):
        self._base_url = base_url
        self._embedding_model = embedding_model
        self._generation_model = generation_model
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._embedding_model, "prompt": text},
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._generation_model,
                "prompt": prompt,
                "stream": False,
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()["response"]

    @property
    def model_name(self) -> str:
        return self._embedding_model

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS.get(self._embedding_model, 768)
