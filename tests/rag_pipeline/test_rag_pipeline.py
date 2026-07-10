import json

from rag_hybrid_search.models import Chunk, RetrievalTrace, RetrievedChunk
from rag_pipeline.generation_provider import MockProvider
from rag_pipeline.models import CitationStatus
from rag_pipeline.rag_pipeline import RagPipeline


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, dev_trace=None):
        return self._chunks, RetrievalTrace()


class RaisingGenerationProvider:
    def generate(self, prompt, **kwargs):
        raise RuntimeError("network down")


def make_retrieved_chunk(chunk_id, text, rerank_score=0.9, final_rank=1):
    chunk = Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0, text=text,
        strategy_version="fixed-v1", heading=None, page=None, char_count=len(text),
    )
    return RetrievedChunk(
        chunk=chunk, dense_score=0.5, bm25_score=0.5, rrf_score=0.5,
        rerank_score=rerank_score, final_rank=final_rank,
    )


def test_answer_end_to_end_with_mock_provider():
    chunks = [make_retrieved_chunk("c1", "Employees get 20 days of paid annual leave.")]
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{
            "text": "Employees get 20 days of paid leave.",
            "citation_ids": ["d1"],
            "supporting_quote": "20 days of paid annual leave",
        }],
    })
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("How many days of paid leave?")

    assert result.answer == "Employees get 20 days of paid leave [d1]."
    assert result.citations == ["d1"]
    assert result.error is None
    assert result.verification.verified_claims == 1
    assert result.confidence.overall > 0.0


def test_multi_citation_claim_from_model_is_narrowed_and_verified_safely():
    """Regression test for the production bug: even if the model ignores
    the v2 prompt's "one citation per claim" instruction and emits a claim
    citing two chunks, the backend narrows it to one citation and extracts
    the supporting_quote from that single chunk only -- so the final,
    verified answer can never carry a quote spanning both chunks, no
    matter what the model returns."""
    chunks = [
        make_retrieved_chunk("c1", "The most critical insight is that granularity dominates."),
        make_retrieved_chunk("c2", "While RQ2 establishes that structural detection is effective.", final_rank=2),
    ]
    canned = json.dumps({
        "answer": "Granularity dominates and RQ2 establishes detection is effective [d1].",
        "claims": [{
            "text": "Granularity dominates model effects, and RQ2 establishes structural detection is effective.",
            "citation_ids": ["d1", "d2"],
        }],
    })
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("How do RQ1 and RQ2 relate?")

    assert result.error is None
    assert result.verification.total_claims == 1
    # Narrowed to exactly one citation -- never both.
    assert result.verification.claim_results[0].claim.citation_ids == ["d1"]
    quote = result.verification.claim_results[0].claim.supporting_quote
    assert "granularity dominates" in quote.lower()
    assert "RQ2" not in quote
    # Backend-extracted from one real chunk, so it always verifies.
    assert result.verification.claim_results[0].passed is True


def test_answer_with_verify_false_skips_verification():
    chunks = [make_retrieved_chunk("c1", "Employees get 20 days of paid leave.")]
    canned = json.dumps({"answer": "Answer.", "claims": []})
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("question", verify=False)

    assert result.verification.total_claims == 0
    assert result.confidence.citations == 0.0
    assert result.confidence.coverage == 0.0


def test_generation_provider_exception_is_caught_not_raised():
    chunks = [make_retrieved_chunk("c1", "some text")]
    pipeline = RagPipeline(FakeRetriever(chunks), RaisingGenerationProvider())

    result = pipeline.answer("question")

    assert result.answer is None
    assert result.error is not None
    assert "network down" in result.error
    assert result.confidence.overall == 0.0


def test_unescaped_inner_quotes_in_answer_are_repaired():
    """The model occasionally copies a quoted phrase from the source text
    into the "answer" field (e.g. the paper says the "GPT-3.5 trap")
    without JSON-escaping the inner quote marks, producing invalid JSON.
    This must be repaired rather than degrading the whole answer to a
    raw-text fallback. (supporting_quote is backend-extracted now, not
    model-provided -- see test_quote_extractor.py for that coverage.)"""
    chunks = [make_retrieved_chunk("c1", "some text")]
    broken_json = (
        '{"answer": "Explained by the "GPT-3.5 trap" [d1].", '
        '"claims": [{"text": "GPT-3.5 detectability is an anomaly.", '
        '"citation_ids": ["d1"]}]}'
    )
    pipeline = RagPipeline(
        FakeRetriever(chunks), MockProvider(canned_json=broken_json)
    )

    result = pipeline.answer("question")

    assert result.error is None
    assert result.verification.total_claims == 1
    assert "GPT-3.5 trap" in result.answer


def test_malformed_json_from_provider_degrades_gracefully():
    chunks = [make_retrieved_chunk("c1", "some text")]
    pipeline = RagPipeline(
        FakeRetriever(chunks), MockProvider(canned_json="not valid json at all")
    )

    result = pipeline.answer("question")

    assert result.answer == "not valid json at all"
    assert result.error is not None
    assert result.verification.total_claims == 0


def test_inline_citation_drift_is_flagged_not_rewritten():
    """When inline [dN] markers in the answer prose disagree with the
    structured claims' citation_ids, the pipeline must not rewrite the
    answer text -- it only flags the drift via citation_status."""
    chunks = [
        make_retrieved_chunk("c1", "The cafeteria serves lunch from 12pm to 2pm."),
        make_retrieved_chunk("c2", "Employees get 20 days of paid annual leave.", final_rank=2),
    ]
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{
            "text": "Employees get 20 days of paid leave.",
            "citation_ids": ["d2"],
            "supporting_quote": "20 days of paid annual leave",
        }],
    })
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("How many days of paid leave?")

    assert result.error is None
    assert result.answer == "Employees get 20 days of paid leave [d1]."
    assert result.citation_status == CitationStatus.INLINE_DRIFT


def test_verification_failure_takes_precedence_over_inline_drift():
    chunks = [make_retrieved_chunk("c1", "Employees get 20 days of paid annual leave.")]
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{
            "text": "Employees get 20 days of paid leave.",
            "citation_ids": ["d99"],
        }],
    })
    pipeline = RagPipeline(FakeRetriever(chunks), MockProvider(canned_json=canned))

    result = pipeline.answer("question")

    assert result.verification.failed_claims == 1
    assert result.citation_status == CitationStatus.VERIFICATION_FAILED
