import pytest
from pydantic import ValidationError

from rag_hybrid_search.config import Settings


def test_defaults():
    settings = Settings()
    assert settings.provider == "gemini"
    assert settings.chunking_strategy == "recursive"
    assert settings.rrf_dense_weight == 0.7
    assert settings.rrf_sparse_weight == 0.3


def test_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        Settings(rrf_dense_weight=0.9, rrf_sparse_weight=0.3)


def test_weight_out_of_range():
    with pytest.raises(ValidationError):
        Settings(rrf_dense_weight=1.5, rrf_sparse_weight=-0.5)


def test_rerank_top_n_cannot_exceed_k_sum():
    with pytest.raises(ValidationError):
        Settings(dense_k=2, sparse_k=2, rerank_top_n=10)


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "1000")
    settings = Settings()
    assert settings.chunk_size == 1000
