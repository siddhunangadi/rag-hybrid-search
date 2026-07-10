import json

from rag_hybrid_search.models import Chunk, RetrievalTrace, RetrievedChunk
from rag_pipeline.generation_provider import MockProvider
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


def test_supporting_quote_with_unescaped_inner_quotes_is_repaired():
    """The model is instructed to copy supporting_quote verbatim from the
    source text. When that source text itself contains a quoted phrase
    (e.g. the paper says the "GPT-3.5 trap"), the model sometimes copies
    the inner quote marks literally without JSON-escaping them, producing
    invalid JSON. This must be repaired rather than degrading the whole
    answer to a raw-text fallback."""
    chunks = [make_retrieved_chunk("c1", "some text")]
    broken_json = (
        '{"answer": "Explained by the "GPT-3.5 trap" [d1].", '
        '"claims": [{"text": "GPT-3.5 detectability is an anomaly.", '
        '"citation_ids": ["d1"], '
        '"supporting_quote": "explains the "GPT-3.5 trap": detectors overfit."}]}'
    )
    pipeline = RagPipeline(
        FakeRetriever(chunks), MockProvider(canned_json=broken_json)
    )

    result = pipeline.answer("question")

    assert result.error is None
    assert result.verification.total_claims == 1
    assert "GPT-3.5 trap" in result.verification.claim_results[0].claim.supporting_quote


def test_malformed_json_from_provider_degrades_gracefully():
    chunks = [make_retrieved_chunk("c1", "some text")]
    pipeline = RagPipeline(
        FakeRetriever(chunks), MockProvider(canned_json="not valid json at all")
    )

    result = pipeline.answer("question")

    assert result.answer == "not valid json at all"
    assert result.error is not None
    assert result.verification.total_claims == 0
