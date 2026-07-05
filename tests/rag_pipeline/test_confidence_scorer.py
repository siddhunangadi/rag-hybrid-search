from rag_hybrid_search.models import Chunk, RetrievedChunk
from rag_pipeline.confidence_scorer import score_confidence
from rag_pipeline.models import Claim, ClaimResult, PromptContext, VerificationReport


def make_retrieved_chunk(chunk_id, rerank_score, final_rank):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text="text",
        strategy_version="fixed-v1", heading=None, page=None, char_count=4,
    )
    return RetrievedChunk(
        chunk=chunk, dense_score=0.5, bm25_score=0.5, rrf_score=0.5,
        rerank_score=rerank_score, final_rank=final_rank,
    )


def make_claim_result(citation_ids, passed):
    claim = Claim(text="x", citation_ids=citation_ids, supporting_quote="x")
    return ClaimResult(
        claim=claim, doc_ids_valid=passed, quote_match_score=1.0 if passed else 0.0,
        passed=passed,
    )


def test_all_citations_pass_gives_high_citation_score():
    chunks = [make_retrieved_chunk("c1", rerank_score=0.9, final_rank=1)]
    context = PromptContext(text="[d1]\ntext", doc_id_map={"d1": "c1"})
    report = VerificationReport(
        total_claims=1, verified_claims=1, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[make_claim_result(["d1"], passed=True)],
    )
    scores = score_confidence(chunks, report, context)
    assert scores.citations == 1.0


def test_half_citations_fail_gives_half_citation_score():
    chunks = [make_retrieved_chunk("c1", rerank_score=0.9, final_rank=1)]
    context = PromptContext(text="[d1]\ntext", doc_id_map={"d1": "c1"})
    report = VerificationReport(
        total_claims=2, verified_claims=1, failed_claims=1,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[
            make_claim_result(["d1"], passed=True),
            make_claim_result(["d1"], passed=False),
        ],
    )
    scores = score_confidence(chunks, report, context)
    assert scores.citations == 0.5


def test_zero_claims_gives_full_citation_score_no_false_penalty():
    chunks = [make_retrieved_chunk("c1", rerank_score=0.9, final_rank=1)]
    context = PromptContext(text="[d1]\ntext", doc_id_map={"d1": "c1"})
    report = VerificationReport(
        total_claims=0, verified_claims=0, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
    )
    scores = score_confidence(chunks, report, context)
    assert scores.citations == 1.0


def test_coverage_reflects_fraction_of_chunks_cited():
    chunks = [
        make_retrieved_chunk("c1", rerank_score=0.9, final_rank=1),
        make_retrieved_chunk("c2", rerank_score=0.8, final_rank=2),
    ]
    context = PromptContext(text="[d1]\ntext\n\n[d2]\ntext", doc_id_map={"d1": "c1", "d2": "c2"})
    report = VerificationReport(
        total_claims=1, verified_claims=1, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[make_claim_result(["d1"], passed=True)],
    )
    scores = score_confidence(chunks, report, context)
    assert scores.coverage == 0.5


def test_overall_is_weighted_combination():
    chunks = [make_retrieved_chunk("c1", rerank_score=1.0, final_rank=1)]
    context = PromptContext(text="[d1]\ntext", doc_id_map={"d1": "c1"})
    report = VerificationReport(
        total_claims=1, verified_claims=1, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[],
        claim_results=[make_claim_result(["d1"], passed=True)],
    )
    scores = score_confidence(chunks, report, context)
    # retrieval=1.0 (normalized top rerank score), citations=1.0, coverage=1.0
    assert scores.overall == 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 1.0


def test_empty_retrieved_chunks_gives_zero_retrieval_score():
    context = PromptContext(text="", doc_id_map={})
    report = VerificationReport(
        total_claims=0, verified_claims=0, failed_claims=0,
        hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
    )
    scores = score_confidence([], report, context)
    assert scores.retrieval == 0.0
