import pytest

from rag_pipeline.eval.metrics import (
    build_retrieval_record,
    citation_precision_recall_f1,
    evaluate_question,
    verdict_score,
)
from rag_pipeline.eval.questions import EvalQuestion, ExpectedAnswer
from rag_pipeline.models import CitationStatus, ClaimResult, Claim, ConfidenceScores, RagAnswer, VerificationReport


def test_citation_precision_recall_f1_perfect_match():
    p, r, f1 = citation_precision_recall_f1(["d1", "d2"], ["d1", "d2"])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_citation_precision_recall_f1_partial_overlap():
    # predicted d1,d3; expected d1,d2 -- 1 true positive, 1 false positive, 1 false negative
    p, r, f1 = citation_precision_recall_f1(["d1", "d3"], ["d1", "d2"])
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_citation_precision_recall_f1_no_overlap():
    p, r, f1 = citation_precision_recall_f1(["d3"], ["d1"])
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_citation_precision_recall_f1_empty_expected_and_predicted():
    p, r, f1 = citation_precision_recall_f1([], [])
    assert (p, r, f1) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("verdict,expected_score", [
    ("CORRECT", 1.0), ("PARTIAL", 0.5), ("INCORRECT", 0.0), ("UNSUPPORTED", 0.0),
])
def test_verdict_score(verdict, expected_score):
    assert verdict_score(verdict) == expected_score


def test_build_retrieval_record_extracts_expected_fields():
    trace_data = {
        "dense": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        "rerank": {"selected": [{"chunk_id": "c1", "score": 0.9, "final_rank": 1}]},
        "pruning": {"before": 2, "after": 1, "dropped": ["c2"]},
        "prompt": {"chars": 500, "approx_tokens": 125},
        "summary": {"chunks_used": 1, "documents_used": 1},
        "timings_ms": {"dense_search": 10.0, "rerank": 20.0, "generation": 200.0, "total": 250.0},
    }

    record = build_retrieval_record(trace_data)

    assert record["retrieved_chunk_ids"] == ["c1", "c2"]
    assert record["reranked_chunk_ids"] == ["c1"]
    assert record["document_ids_used"] == 1
    assert record["context_size_chars"] == 500
    assert record["retrieval_latency_ms"] == 10.0
    assert record["rerank_latency_ms"] == 20.0
    assert record["generation_latency_ms"] == 200.0


def _make_rag_answer(citations, verified=True):
    verification = VerificationReport(
        total_claims=1, verified_claims=1 if verified else 0, failed_claims=0 if verified else 1,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[ClaimResult(
            claim=Claim(text="claim", citation_ids=citations), doc_ids_valid=True,
            quote_match_score=1.0, passed=verified,
        )],
    )
    return RagAnswer(
        answer="model answer", citations=citations,
        confidence=ConfidenceScores(retrieval=1.0, citations=1.0, coverage=1.0, overall=1.0),
        verification=verification, citation_status=CitationStatus.OK, error=None,
    )


class CannedJudgeProvider:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt, **kwargs):
        return self._response


def test_evaluate_question_assembles_full_record():
    import json

    question = EvalQuestion(
        id="q001", question="What is X?", category="factual",
        expected=ExpectedAnswer(answer="X is a thing.", citation_doc_ids=["d1"]),
    )
    rag_answer = _make_rag_answer(citations=["d1"])
    trace_data = {
        "dense": [{"chunk_id": "c1"}], "rerank": {"selected": [{"chunk_id": "c1"}]},
        "pruning": {"before": 1, "after": 1, "dropped": []},
        "prompt": {"chars": 100, "approx_tokens": 25},
        "summary": {"chunks_used": 1, "documents_used": 1},
        "timings_ms": {"dense_search": 5.0, "rerank": 5.0, "generation": 100.0, "total": 120.0},
    }
    judge_provider = CannedJudgeProvider(json.dumps({"verdict": "CORRECT", "reasoning": "matches"}))

    record = evaluate_question(question, rag_answer, trace_data, latency_ms=120.0, judge_provider=judge_provider)

    assert record["id"] == "q001"
    assert record["status"] == "success"
    assert record["error_type"] is None
    assert record["objective_metrics"]["citation_precision"] == 1.0
    assert record["objective_metrics"]["citation_recall"] == 1.0
    assert record["objective_metrics"]["verification_pass"] is True
    assert record["judge"]["verdict"] == "CORRECT"
    assert record["retrieval"]["retrieved_chunk_ids"] == ["c1"]
