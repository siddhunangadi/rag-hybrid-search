import json
import threading
from concurrent.futures import ThreadPoolExecutor

from rag_hybrid_search.models import Chunk, RetrievalTrace, RetrievedChunk
from rag_pipeline.generation_provider import MockProvider
from rag_pipeline.models import CitationStatus
from rag_pipeline.rag_pipeline import (
    RagPipeline,
    _MAX_CONCURRENT_RETRIEVAL_WORKERS,
    _merge_multi_query_results,
    _retrieve_subqueries_concurrently,
)


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


def test_merge_multi_query_results_dedupes_by_chunk_id_keeping_best_score():
    """Same chunk_id retrieved by two different sub-queries: keep the
    higher rerank_score (as the chunk's displayed score), drop the
    duplicate, and re-rank the merged list."""
    q1_results = [
        make_retrieved_chunk("c1", "text1", rerank_score=2.0, final_rank=1),
        make_retrieved_chunk("c2", "text2", rerank_score=1.0, final_rank=2),
    ]
    q2_results = [
        make_retrieved_chunk("c1", "text1", rerank_score=3.0, final_rank=1),
        make_retrieved_chunk("c3", "text3", rerank_score=0.5, final_rank=2),
    ]

    merged = _merge_multi_query_results([q1_results, q2_results])

    assert [r.chunk.chunk_id for r in merged] == ["c1", "c2", "c3"]
    assert merged[0].rerank_score == 3.0
    assert [r.final_rank for r in merged] == [1, 2, 3]


def test_merge_multi_query_results_boosts_chunks_appearing_in_multiple_subqueries():
    """A chunk that surfaces in every sub-query's retrieval is a stronger
    relevance signal than a slightly higher-scoring chunk that only one
    sub-query found. The log2-scaled frequency bonus must be able to flip
    the ranking: chunk A appears 4x at score 2.0 (2.0 + 0.15*log2(4) = 2.3),
    chunk B appears once at score 2.1 -- A ranks first."""
    a = make_retrieved_chunk("a", "text-a", rerank_score=2.0, final_rank=1)
    b = make_retrieved_chunk("b", "text-b", rerank_score=2.1, final_rank=1)
    results_per_query = [[a], [a], [a], [a, b]]

    merged = _merge_multi_query_results(results_per_query)

    assert [r.chunk.chunk_id for r in merged] == ["a", "b"]


def test_merge_multi_query_results_single_appearance_uses_raw_score():
    """No frequency bonus when a chunk appears in only one sub-query's
    results -- ranking degrades to plain best-rerank-score, same as before
    frequency weighting was added (log2(1) == 0, no bonus)."""
    a = make_retrieved_chunk("a", "text-a", rerank_score=2.0, final_rank=1)
    b = make_retrieved_chunk("b", "text-b", rerank_score=2.1, final_rank=1)

    merged = _merge_multi_query_results([[a], [b]])

    assert [r.chunk.chunk_id for r in merged] == ["b", "a"]


def test_merge_multi_query_results_frequency_bonus_does_not_overwhelm_a_much_higher_score():
    """Regression guard for a linear (non-diminishing) bonus over-promoting
    a frequently-occurring but clearly weaker chunk: A scores 1.5 and
    appears 8x (1.5 + 0.15*log2(8) = 1.5 + 0.45 = 1.95), B scores 2.2 and
    appears once (bonus 0). A linear bonus of 0.1/appearance would have
    given A 1.5 + 0.7 = 2.2, tying or beating B -- log2 scaling keeps A
    below B here, as it should given the score gap."""
    a = make_retrieved_chunk("a", "text-a", rerank_score=1.5, final_rank=1)
    b = make_retrieved_chunk("b", "text-b", rerank_score=2.2, final_rank=1)
    results_per_query = [[a]] * 7 + [[a, b]]

    merged = _merge_multi_query_results(results_per_query)

    assert [r.chunk.chunk_id for r in merged] == ["b", "a"]


def test_merge_multi_query_results_handles_missing_rerank_score():
    """PassthroughReranker never sets rerank_score -- merge must not crash,
    just fall back to original retrieval order (no score to sort by)."""
    q1_results = [make_retrieved_chunk("c1", "text1", rerank_score=None, final_rank=1)]
    q2_results = [make_retrieved_chunk("c2", "text2", rerank_score=None, final_rank=1)]

    merged = _merge_multi_query_results([q1_results, q2_results])

    assert {r.chunk.chunk_id for r in merged} == {"c1", "c2"}


def test_merge_returns_provenance_side_map():
    chunk_a = make_retrieved_chunk("a", "text a", final_rank=1)
    chunk_b = make_retrieved_chunk("b", "text b", final_rank=1)
    chunk_c = make_retrieved_chunk("c", "text c", final_rank=1)

    # subquery 0 retrieves a, b; subquery 1 retrieves b, c
    results_per_query = [[chunk_a, chunk_b], [chunk_b, chunk_c]]

    merged, provenance = _merge_multi_query_results(results_per_query)

    assert provenance["a"].primary_subquery == 0
    assert provenance["a"].all_subqueries == [0]
    assert provenance["b"].primary_subquery == 0  # first seen under subquery 0
    assert provenance["b"].all_subqueries == [0, 1]
    assert provenance["c"].primary_subquery == 1
    assert provenance["c"].all_subqueries == [1]
    assert {r.chunk.chunk_id for r in merged} == {"a", "b", "c"}


def test_merge_provenance_single_subquery():
    chunk_a = make_retrieved_chunk("a", "text a", final_rank=1)
    merged, provenance = _merge_multi_query_results([[chunk_a]])
    assert provenance["a"].primary_subquery == 0
    assert provenance["a"].all_subqueries == [0]


class MultiQueryFakeRetriever:
    """Records every query it was asked to retrieve for, and returns a
    different chunk per distinct query string so tests can verify
    decomposition actually reached the retriever.

    Multiple sub-query retrievals now run concurrently (Task 2, Step 7),
    so `queries_seen` may be appended to from several threads -- a lock
    keeps the list itself consistent, though the *order* of entries across
    different sub-queries is no longer deterministic (tests assert set
    membership, not list order, wherever more than one sub-query is
    involved)."""

    def __init__(self, chunks_by_query: dict[str, list]):
        self._chunks_by_query = chunks_by_query
        self.queries_seen: list[str] = []
        self._lock = threading.Lock()

    def retrieve(self, query, dev_trace=None):
        with self._lock:
            self.queries_seen.append(query)
        return self._chunks_by_query.get(query, []), RetrievalTrace()


def test_comparative_question_retrieves_once_per_subquery():
    rq1_chunk = make_retrieved_chunk("rq1", "RQ1 finding: granularity matters.", rerank_score=2.0)
    rq2_chunk = make_retrieved_chunk("rq2", "RQ2 finding: class-level differs.", rerank_score=1.5, final_rank=2)
    retriever = MultiQueryFakeRetriever({
        "RQ1 findings": [rq1_chunk],
        "RQ2 findings": [rq2_chunk],
    })
    decompose_canned = json.dumps(["RQ1 findings", "RQ2 findings"])
    answer_canned = json.dumps({
        "answer": "RQ1 shows granularity matters [d1] and RQ2 shows class-level differs [d2].",
        "claims": [
            {"text": "Granularity matters.", "citation_ids": ["d1"]},
            {"text": "Class-level differs.", "citation_ids": ["d2"]},
        ],
    })
    provider = MockProvider(canned_json=answer_canned)
    # Decomposition and generation share one provider in RagPipeline; swap
    # canned output between the two calls via a small wrapper.
    calls = {"n": 0}
    original_generate = provider.generate

    def generate(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return decompose_canned
        return original_generate(prompt, **kwargs)

    provider.generate = generate

    pipeline = RagPipeline(retriever, provider)
    result = pipeline.answer("How do function-level and class-level detection patterns differ across RQ1 and RQ2?")

    # Concurrent execution (Step 7) means call order across sub-queries
    # isn't guaranteed -- assert membership, not sequence.
    assert set(retriever.queries_seen) == {"RQ1 findings", "RQ2 findings"}
    assert result.error is None
    assert {c for c in result.citations} == {"d1", "d2"}


def test_non_comparative_question_retrieves_once_with_original_question():
    chunk = make_retrieved_chunk("c1", "Employees get 20 days of paid leave.")
    retriever = MultiQueryFakeRetriever({"How many days of paid leave?": [chunk]})
    canned = json.dumps({
        "answer": "Employees get 20 days of paid leave [d1].",
        "claims": [{"text": "Employees get 20 days of paid leave.", "citation_ids": ["d1"]}],
    })
    pipeline = RagPipeline(retriever, MockProvider(canned_json=canned))

    result = pipeline.answer("How many days of paid leave?")

    assert retriever.queries_seen == ["How many days of paid leave?"]
    assert result.error is None


def test_one_failing_subquery_does_not_abort_the_others():
    """Regression test for concurrent retrieval: RQ2's retrieve() raising
    must not lose RQ1 and RQ3's results or blow up the whole request --
    the failing sub-query just contributes no chunks."""

    class FlakyMultiQueryRetriever:
        def retrieve(self, query, dev_trace=None):
            if query == "RQ2 findings":
                raise RuntimeError("simulated retrieval timeout")
            chunks = {
                "RQ1 findings": [make_retrieved_chunk("rq1", "RQ1 finding.", rerank_score=2.0)],
                "RQ3 findings": [make_retrieved_chunk("rq3", "RQ3 finding.", rerank_score=1.0)],
            }
            return chunks.get(query, []), RetrievalTrace()

    decompose_canned = json.dumps(["RQ1 findings", "RQ2 findings", "RQ3 findings"])
    answer_canned = json.dumps({
        "answer": "RQ1 shows one finding [d1] and RQ3 shows another [d2].",
        "claims": [
            {"text": "RQ1 finding.", "citation_ids": ["d1"]},
            {"text": "RQ3 finding.", "citation_ids": ["d2"]},
        ],
    })
    provider = MockProvider(canned_json=answer_canned)
    calls = {"n": 0}
    original_generate = provider.generate

    def generate(prompt, **kwargs):
        calls["n"] += 1
        return decompose_canned if calls["n"] == 1 else original_generate(prompt, **kwargs)

    provider.generate = generate

    pipeline = RagPipeline(FlakyMultiQueryRetriever(), provider)
    result = pipeline.answer("Compare RQ1, RQ2, and RQ3 findings.")

    assert result.error is None
    assert {c for c in result.citations} == {"d1", "d2"}


def test_retrieve_subqueries_concurrently_caps_worker_count(monkeypatch):
    """Worker count must never exceed _MAX_CONCURRENT_RETRIEVAL_WORKERS,
    even if called with many more sub-queries than that -- thread creation
    shouldn't scale 1:1 with an eventually-configurable max_subqueries."""
    seen_max_workers = {}
    real_executor_init = ThreadPoolExecutor.__init__

    def recording_init(self, max_workers=None, *args, **kwargs):
        seen_max_workers["value"] = max_workers
        return real_executor_init(self, max_workers=max_workers, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", recording_init)

    many_subqueries = [f"q{i}" for i in range(_MAX_CONCURRENT_RETRIEVAL_WORKERS + 6)]
    _retrieve_subqueries_concurrently(many_subqueries, lambda q, trace: ([], None))

    assert seen_max_workers["value"] == _MAX_CONCURRENT_RETRIEVAL_WORKERS


def test_retrieve_subqueries_concurrently_isolates_one_failure():
    def flaky_retrieve(q, trace):
        if q == "bad":
            raise RuntimeError("boom")
        return [q]

    results = _retrieve_subqueries_concurrently(["good1", "bad", "good2"], flaky_retrieve)

    assert results == [["good1"], [], ["good2"]]


def test_comparative_question_keeps_multiple_chunks_after_pruning():
    """Regression test for the production bug: a comparative question
    whose reranker gave one chunk a dominant score used to get pruned down
    to 1 chunk, starving the generator of evidence for the other concepts.
    min_keep=3 for comparative questions prevents that collapse."""
    chunks_by_query = {
        "RQ1 findings": [make_retrieved_chunk("rq1", "RQ1: granularity matters.", rerank_score=5.0)],
        "RQ2 findings": [make_retrieved_chunk("rq2", "RQ2: class-level differs.", rerank_score=0.2, final_rank=2)],
        "RQ3 findings": [make_retrieved_chunk("rq3", "RQ3: features overlap.", rerank_score=0.1, final_rank=3)],
    }
    retriever = MultiQueryFakeRetriever(chunks_by_query)
    decompose_canned = json.dumps(["RQ1 findings", "RQ2 findings", "RQ3 findings"])
    answer_canned = json.dumps({
        "answer": "RQ1 shows granularity matters [d1], RQ2 shows class-level differs [d2], RQ3 shows features overlap [d3].",
        "claims": [
            {"text": "Granularity matters.", "citation_ids": ["d1"]},
            {"text": "Class-level differs.", "citation_ids": ["d2"]},
            {"text": "Features overlap.", "citation_ids": ["d3"]},
        ],
    })
    provider = MockProvider(canned_json=answer_canned)
    calls = {"n": 0}
    original_generate = provider.generate

    def generate(prompt, **kwargs):
        calls["n"] += 1
        return decompose_canned if calls["n"] == 1 else original_generate(prompt, **kwargs)

    provider.generate = generate

    pipeline = RagPipeline(retriever, provider)
    result = pipeline.answer("How do RQ1, RQ2, and RQ3 findings differ?")

    assert result.error is None
    assert {c for c in result.citations} == {"d1", "d2", "d3"}


def test_answer_accepts_injected_dev_trace_and_exposes_its_data():
    from rag_hybrid_search.trace import RequestTrace

    chunks = [make_retrieved_chunk("c1", "Some evidence text.")]
    retriever = FakeRetriever(chunks)
    provider = MockProvider()
    pipeline = RagPipeline(retriever, provider)

    trace = RequestTrace("What is X?", {"Generation": "MockProvider"})
    result = pipeline.answer("What is X?", dev_trace=trace)

    assert result.error is None
    assert trace.data["question"] == "What is X?"
    assert "timings_ms" in trace.data
    assert trace.data["summary"]["chunks_used"] == 1
