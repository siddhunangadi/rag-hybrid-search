import pytest

from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_hybrid_search.trace import RequestTrace


def make_result(chunk_id, rrf_score=0.5, rerank_score=None, final_rank=1):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text="text",
        strategy_version="fixed-v1", heading=None, page=None, char_count=4,
    )
    return RetrievedChunk(
        chunk=chunk, dense_score=0.5, bm25_score=0.5, rrf_score=rrf_score,
        rerank_score=rerank_score, final_rank=final_rank,
    )


def test_log_rerank_records_budget_counts(monkeypatch):
    monkeypatch.delenv("TRACE_RAG", raising=False)
    trace = RequestTrace("question", {})
    trace._data["dense"] = [{}] * 5
    trace._data["bm25"] = [{}] * 5
    trace._data["fusion"] = [{}] * 8

    budgeted = [make_result(f"c{i}") for i in range(4)]
    reranked = [budgeted[0].model_copy(update={"final_rank": 1, "rerank_score": 0.9})]

    trace.log_rerank("nvidia", budgeted, reranked, latency_ms=12.0, budget_applied=4)

    rerank_data = trace._data["rerank"]
    assert rerank_data["fusion_candidates"] == 8
    assert rerank_data["sent_to_reranker"] == 4
    assert rerank_data["returned"] == 1
    assert rerank_data["budget_applied"] == 4


def test_log_rerank_prints_retrieval_budget_block_when_enabled(monkeypatch, capsys):
    monkeypatch.setenv("TRACE_RAG", "true")
    trace = RequestTrace("question", {})
    trace._data["dense"] = [{}] * 10
    trace._data["bm25"] = [{}] * 10
    trace._data["fusion"] = [{}] * 17

    budgeted = [make_result(f"c{i}") for i in range(8)]
    reranked = [budgeted[i].model_copy(update={"final_rank": i + 1, "rerank_score": 0.9}) for i in range(5)]

    trace.log_rerank("nvidia", budgeted, reranked, latency_ms=12.0, budget_applied=8)

    out = capsys.readouterr().out
    assert "RETRIEVAL BUDGET" in out
    assert "Dense candidates" in out
    assert "Saved" in out
    assert "9 reranker evaluations" in out
