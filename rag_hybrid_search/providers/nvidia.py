import logging
import resource
import sys

import httpx

from rag_hybrid_search.providers.base import EmbeddingProvider, GenerationProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _rss_mb() -> float:
    """Temporary instrumentation (see rag_hybrid_search/ingestion/pipeline.py's
    matching helper) -- embed() is a black box from the caller's log lines,
    so if a crash happens inside it (e.g. during response.json() parsing),
    the caller never sees a next log line at all. This pinpoints which
    sub-step inside embed() it actually is."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024 if sys.platform == "linux" else raw / 1024 / 1024

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
        timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._embedding_model = embedding_model
        self._generation_model = generation_model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )

    def embed(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        logger.info("embed: sending request for %d texts rss_mb=%.1f", len(texts), _rss_mb())
        response = self._client.post(
            f"{_BASE_URL}/embeddings",
            json={
                "input": texts,
                "model": self._embedding_model,
                "input_type": input_type,
            },
        )
        logger.info(
            "embed: response received status=%d content_length=%s rss_mb=%.1f",
            response.status_code, response.headers.get("content-length"), _rss_mb(),
        )
        response.raise_for_status()
        parsed = response.json()
        logger.info("embed: response.json() parsed rss_mb=%.1f", _rss_mb())
        data = parsed["data"]
        result = [item["embedding"] for item in data]
        logger.info("embed: built %d embedding lists rss_mb=%.1f", len(result), _rss_mb())
        return result

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": self._generation_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
        }
        payload.update(kwargs)
        response = self._client.post(f"{_BASE_URL}/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def model_name(self) -> str:
        return self._embedding_model

    @property
    def dimension(self) -> int:
        return _MODEL_DIMENSIONS.get(self._embedding_model, 1024)
