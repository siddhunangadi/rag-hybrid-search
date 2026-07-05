from rag_hybrid_search.models import RetrievedChunk
from rag_pipeline.models import ConfidenceScores, PromptContext, VerificationReport

RETRIEVAL_WEIGHT = 0.4
CITATION_WEIGHT = 0.4
COVERAGE_WEIGHT = 0.2


def score_confidence(
    retrieved_chunks: list[RetrievedChunk],
    verification: VerificationReport,
    context: PromptContext,
) -> ConfidenceScores:
    retrieval = _retrieval_score(retrieved_chunks)
    citations = _citation_score(verification)
    coverage = _coverage_score(verification, context)
    overall = (
        RETRIEVAL_WEIGHT * retrieval
        + CITATION_WEIGHT * citations
        + COVERAGE_WEIGHT * coverage
    )
    return ConfidenceScores(
        retrieval=retrieval, citations=citations, coverage=coverage, overall=overall
    )


def _retrieval_score(retrieved_chunks: list[RetrievedChunk]) -> float:
    if not retrieved_chunks:
        return 0.0
    top = min(retrieved_chunks, key=lambda r: r.final_rank)
    score = top.rerank_score if top.rerank_score is not None else top.rrf_score
    return max(0.0, min(1.0, score))


def _citation_score(verification: VerificationReport) -> float:
    if verification.total_claims == 0:
        return 1.0
    return verification.verified_claims / verification.total_claims


def _coverage_score(
    verification: VerificationReport, context: PromptContext
) -> float:
    if not context.doc_id_map:
        return 0.0
    cited_doc_ids: set[str] = set()
    for result in verification.claim_results:
        if result.doc_ids_valid:
            cited_doc_ids.update(result.claim.citation_ids)
    return len(cited_doc_ids) / len(context.doc_id_map)
