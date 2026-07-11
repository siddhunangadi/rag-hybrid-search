import pytest

from rag_hybrid_search.models import Chunk, ChunkProvenance, ContextChunk, RetrievedChunk
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


def test_log_verification_computes_ratio_and_prints_summary(monkeypatch, capsys):
    monkeypatch.setenv("TRACE_RAG", "true")
    trace = RequestTrace("question", {})

    class FakeClaim:
        text = "claim text"
        citation_ids = ["d1"]

    class FakeClaimResult:
        def __init__(self, passed):
            self.claim = FakeClaim()
            self.doc_ids_valid = True
            self.quote_match_score = 1.0
            self.passed = passed
            self.failure_reason = None if passed else "quote_not_found"

    class FakeVerification:
        total_claims = 4
        verified_claims = 3
        failed_claims = 1
        claim_results = [FakeClaimResult(True), FakeClaimResult(True), FakeClaimResult(True), FakeClaimResult(False)]

    trace.log_verification(FakeVerification())

    assert trace._data["verification"]["verification_ratio"] == pytest.approx(0.75)
    out = capsys.readouterr().out
    assert "VERIFICATION SUMMARY" in out
    assert "75%" in out


def test_log_verification_ratio_zero_when_no_claims():
    trace = RequestTrace("question", {})

    class FakeVerification:
        total_claims = 0
        verified_claims = 0
        failed_claims = 0
        claim_results = []

    trace.log_verification(FakeVerification())
    assert trace._data["verification"]["verification_ratio"] == 0.0


def test_log_query_decomposition_records_subqueries_raw_output_and_coverage(monkeypatch, capsys):
    monkeypatch.setenv("TRACE_RAG", "true")
    trace = RequestTrace("question", {})

    trace.log_query_decomposition(
        True, ["RQ1 findings", "RQ2 findings", "RQ3 findings"],
        raw_llm_output='["RQ1 findings", "RQ2 findings", "RQ3 findings"]',
        concepts_retrieved=2,
    )

    data = trace._data["query_decomposition"]
    assert data["subqueries"] == ["RQ1 findings", "RQ2 findings", "RQ3 findings"]
    assert data["concepts_requested"] == 3
    assert data["concepts_retrieved"] == 2
    assert data["coverage"] == pytest.approx(2 / 3)
    out = capsys.readouterr().out
    assert "QUERY DECOMPOSITION" in out
    assert "RQ1 findings" in out
    assert "Coverage" in out
    assert "67%" in out
    assert "Raw decomposition output" in out


def test_log_query_decomposition_coverage_zero_when_no_subqueries():
    trace = RequestTrace("question", {})

    trace.log_query_decomposition(False, [], raw_llm_output=None, concepts_retrieved=0)

    assert trace._data["query_decomposition"]["coverage"] == 0.0


def test_log_provenance_records_primary_and_all_subqueries():
    trace = RequestTrace("question", {})
    chunk = Chunk(
        chunk_id="c1", document_id="d1", chunk_index=0, text="hello",
        strategy_version="fixed-v1", heading=None, page=None, char_count=5,
    )
    retrieved = RetrievedChunk(
        chunk=chunk, dense_score=0.9, bm25_score=0.9, rrf_score=0.5,
        rerank_score=0.8, final_rank=1,
    )
    context_chunk = ContextChunk(
        chunk=retrieved,
        provenance=ChunkProvenance(primary_subquery=0, all_subqueries=[0, 1]),
    )

    trace.log_provenance([context_chunk])

    assert trace._data["provenance"] == {
        "c1": {"primary_subquery": 0, "all_subqueries": [0, 1]},
    }
