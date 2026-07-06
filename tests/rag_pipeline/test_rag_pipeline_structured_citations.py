from rag_pipeline.models import RagAnswer, ConfidenceScores, VerificationReport


def test_rag_answer_structured_citations_defaults_to_empty_list():
    answer = RagAnswer(
        answer="ok",
        citations=[],
        confidence=ConfidenceScores(retrieval=1.0, citations=1.0, coverage=1.0, overall=1.0),
        verification=VerificationReport(
            total_claims=0, verified_claims=0, failed_claims=0,
            hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
        ),
    )
    assert answer.structured_citations == []
