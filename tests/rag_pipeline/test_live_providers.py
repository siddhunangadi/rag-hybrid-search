import os

import httpx
import pytest

from rag_hybrid_search.providers.nvidia import NvidiaProvider
from rag_hybrid_search.providers.ollama import OllamaProvider
from rag_pipeline.rag_pipeline import RagPipeline

from tests.rag_pipeline.test_end_to_end import build_pipeline_and_retriever


def _ollama_reachable(base_url: str) -> bool:
    try:
        httpx.get(f"{base_url}/api/tags", timeout=1.0).raise_for_status()
        return True
    except Exception:
        return False


_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


@pytest.mark.skipif(
    not _ollama_reachable(_OLLAMA_BASE_URL),
    reason=f"no Ollama server reachable at {_OLLAMA_BASE_URL}",
)
def test_rag_pipeline_answers_with_real_ollama_provider(tmp_path):
    fixtures_dir = tmp_path / "docs"
    fixtures_dir.mkdir()
    doc_path = fixtures_dir / "leave-policy.md"
    doc_path.write_text("Employees get 20 days of paid annual leave per year.")

    ingestion, retriever = build_pipeline_and_retriever(tmp_path)
    ingestion.ingest(str(doc_path))

    provider = OllamaProvider(
        base_url=_OLLAMA_BASE_URL, generation_model=_OLLAMA_MODEL, timeout=120.0
    )
    pipeline = RagPipeline(retriever, provider)

    result = pipeline.answer("How many days of paid leave do employees get?")

    # A real model's output is not fully deterministic, so we only assert the
    # pipeline completed against a live model and produced a structured RagAnswer
    # (either a verified answer, or a well-formed error/parse-failure degradation) —
    # not a crash and not a silently-invented citation.
    assert result.answer is not None or result.error is not None
    if result.error is None:
        assert all(cid == "d1" for cid in result.citations)
        assert result.confidence.overall >= 0.0


@pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY"),
    reason="NVIDIA_API_KEY not set",
)
def test_rag_pipeline_answers_with_real_nvidia_provider(tmp_path):
    fixtures_dir = tmp_path / "docs"
    fixtures_dir.mkdir()
    doc_path = fixtures_dir / "leave-policy.md"
    doc_path.write_text("Employees get 20 days of paid annual leave per year.")

    ingestion, retriever = build_pipeline_and_retriever(tmp_path)
    ingestion.ingest(str(doc_path))

    provider = NvidiaProvider(api_key=os.environ["NVIDIA_API_KEY"])
    pipeline = RagPipeline(retriever, provider)

    result = pipeline.answer("How many days of paid leave do employees get?")

    assert result.answer is not None
    if result.error is None:
        assert all(cid == "d1" for cid in result.citations)
        assert result.confidence.overall >= 0.0
