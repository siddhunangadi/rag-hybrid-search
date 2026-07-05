import pytest

from rag_hybrid_search.providers.base import (
    EmbeddingProvider,
    GenerationProvider,
    RerankProvider,
)


def test_embedding_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_generation_provider_is_abstract():
    with pytest.raises(TypeError):
        GenerationProvider()


def test_rerank_provider_is_abstract():
    with pytest.raises(TypeError):
        RerankProvider()
