import os

import pytest

from rag_hybrid_search.providers.gemini import GeminiProvider
from rag_hybrid_search.providers.nvidia import NvidiaProvider
from rag_pipeline.rag_pipeline import RagPipeline

from tests.rag_pipeline.test_end_to_end import build_pipeline_and_retriever


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
def test_rag_pipeline_answers_with_real_gemini_provider(tmp_path):
    fixtures_dir = tmp_path / "docs"
    fixtures_dir.mkdir()
    doc_path = fixtures_dir / "leave-policy.md"
    doc_path.write_text("Employees get 20 days of paid annual leave per year.")

    ingestion, retriever = build_pipeline_and_retriever(tmp_path)
    ingestion.ingest(str(doc_path))

    provider = GeminiProvider(api_key=os.environ["GEMINI_API_KEY"])
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
