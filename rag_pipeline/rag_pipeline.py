import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from rag_hybrid_search.compliance.citation_mapper import build_citations
from rag_hybrid_search.compliance.query_router import route_query
from rag_hybrid_search.trace import RequestTrace
from rag_pipeline.citation_utils import chunk_text_for_doc_id
from rag_pipeline.confidence_scorer import score_confidence
from rag_pipeline.context_pruning import prune_by_score_margin
from rag_pipeline.context_builder import build_context
from rag_pipeline.generation_provider import GenerationProvider
from rag_pipeline.citation_verifier import verify_citations
from rag_pipeline.models import (
    Claim,
    CitationStatus,
    ConfidenceScores,
    GenerationMetadata,
    RagAnswer,
    RagAnswerDraft,
    VerificationReport,
)
from rag_pipeline.prompt_builder import build_prompt
from rag_pipeline.quote_extractor import extract_supporting_quotes
from rag_pipeline.query_decomposer import decompose_query, is_comparative_query

logger = logging.getLogger(__name__)

_EMPTY_VERIFICATION = VerificationReport(
    total_claims=0, verified_claims=0, failed_claims=0,
    hallucinated_doc_ids=[], missing_quotes=[], claim_results=[],
)
_ZERO_CONFIDENCE = ConfidenceScores(retrieval=0.0, citations=0.0, coverage=0.0, overall=0.0)

_KEY_TERMINATORS = (":",)
_VALUE_TERMINATORS = (",", "}", "]")


def _repair_unescaped_quotes(raw: str) -> str:
    """Escape literal '"' characters that appear inside a JSON string value
    but aren't the string's real delimiter.

    Verbatim-quote instructions mean the model often copies a source
    sentence that itself contains quote marks (e.g. the paper says the
    "GPT-3.5 trap") without escaping them, breaking JSON parsing. A quote
    only opens/closes a real string if the surrounding tokens match what's
    structurally expected: a string starting right after '{' or ',' is an
    object KEY and must close before ':'; a string starting after ':' or
    '[' is a VALUE and must close before ',', '}', or ']'. Distinguishing
    key vs. value context (not just "any structural char") matters because
    a bare '"' followed by ':' inside a value (e.g. a quoted term followed
    by a colon in ordinary English) would otherwise be misread as a key
    terminator. Any quote that doesn't fit its expected terminator is a
    literal character inside the string and gets escaped.
    """
    out: list[str] = []
    in_string = False
    expect_terminators: tuple[str, ...] = ()
    containers: list[str] = []  # '{' or '[' for each open container
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            out.append(raw[i:i + 2])
            i += 2
            continue
        if not in_string and ch in "{[":
            containers.append(ch)
        elif not in_string and ch in "}]":
            if containers:
                containers.pop()
        if ch == '"':
            if not in_string:
                prev = next((c for c in reversed(out) if not c.isspace()), None)
                if prev == "{" or (prev == "," and containers and containers[-1] == "{"):
                    in_string = True
                    expect_terminators = _KEY_TERMINATORS
                    out.append(ch)
                elif prev is None or prev in (":", "[") or (
                    prev == "," and containers and containers[-1] == "["
                ):
                    in_string = True
                    expect_terminators = _VALUE_TERMINATORS
                    out.append(ch)
                else:
                    out.append('\\"')
            else:
                j = i + 1
                while j < n and raw[j].isspace():
                    j += 1
                nxt = raw[j] if j < n else None
                if nxt is None or nxt in expect_terminators:
                    in_string = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


_INLINE_CITATION_RE = re.compile(r"\[(d\d+)\]")


def _inline_citation_drift(answer: str, structured_ids: set[str]) -> tuple[list[str], bool]:
    """Detect drift between inline ``[dN]`` refs in the free-text answer and
    the citation_ids the model put in its own structured claims.

    The model writes citations twice: inline in ``answer`` prose and again
    in ``claims[].citation_ids``. Nothing enforces they agree. This only
    detects and reports drift -- it never rewrites the answer text. The
    pipeline validates what the model produced; it does not repair it.
    """
    inline_ids = _INLINE_CITATION_RE.findall(answer)
    inline_set = set(inline_ids)
    ok = inline_set <= structured_ids
    return sorted(inline_set, key=lambda x: (len(x), x)), ok


_FREQUENCY_BONUS_SCALE = 0.15


def _merge_multi_query_results(results_per_query: list[list]) -> list:
    """Merge reranked result lists from multiple sub-query retrievals into
    one ranked list, deduping by chunk_id.

    Each sub-query already went through the full retrieve() pipeline
    (dense+sparse+fuse+rerank) independently. For ranking the merged list,
    a chunk that surfaced in N independent sub-query retrievals is a
    stronger relevance signal than its single best rerank_score alone
    would suggest -- appearing under multiple distinct queries means
    multiple lines of evidence point to it, not just one lucky embedding
    match. combined_score = best_rerank_score + 0.15 * log2(appearances),
    so a single appearance is unaffected (log2(1) == 0, matches
    pre-frequency-weighting behavior) and each additional appearance adds
    a *diminishing* bonus rather than a linear one -- a chunk with a
    clearly weaker score shouldn't out-rank a much stronger one just by
    showing up many times (log2(8) is only 2x log2(4), not 8x). The chunk
    object itself keeps its original (best) rerank_score unchanged --
    combined_score is sort-only, not stored on the model. Chunks with
    rerank_score=None (e.g. PassthroughReranker) sort last but are never
    dropped.
    """
    best_by_id: dict[str, object] = {}
    appearances: dict[str, int] = {}
    for results in results_per_query:
        for r in results:
            chunk_id = r.chunk.chunk_id
            appearances[chunk_id] = appearances.get(chunk_id, 0) + 1
            existing = best_by_id.get(chunk_id)
            if existing is None:
                best_by_id[chunk_id] = r
                continue
            existing_score = existing.rerank_score
            new_score = r.rerank_score
            if new_score is not None and (existing_score is None or new_score > existing_score):
                best_by_id[chunk_id] = r

    def combined_score(r) -> float:
        base = r.rerank_score if r.rerank_score is not None else 0.0
        bonus = _FREQUENCY_BONUS_SCALE * math.log2(appearances[r.chunk.chunk_id])
        return base + bonus

    merged = sorted(
        best_by_id.values(),
        key=lambda r: (r.rerank_score is None, -combined_score(r)),
    )
    return [r.model_copy(update={"final_rank": i}) for i, r in enumerate(merged, start=1)]


_MAX_CONCURRENT_RETRIEVAL_WORKERS = 4


def _retrieve_subqueries_concurrently(subqueries: list[str], retrieve_one) -> list[list]:
    """Run `retrieve_one(q, None)` for every sub-query in parallel.

    Worker count is capped at `_MAX_CONCURRENT_RETRIEVAL_WORKERS` regardless
    of how many sub-queries there are, so if `max_subqueries` is ever raised
    well beyond today's default of 4, thread creation doesn't scale
    1:1 with it.

    A single sub-query's retrieve() call failing (provider timeout,
    connection error, etc.) must not abort the other sub-queries or the
    whole request -- each future is resolved individually, a failure is
    logged and contributes an empty result list for that sub-query (it
    simply doesn't show up in the merged context and drags down
    `concepts_retrieved`/coverage), and every other sub-query's results
    are still used. dev_trace is intentionally not threaded through here
    (see the call site) -- concurrent writers aren't safe against the
    shared RequestTrace state.
    """
    max_workers = min(len(subqueries), _MAX_CONCURRENT_RETRIEVAL_WORKERS)
    results: list[list] = [[] for _ in subqueries]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(retrieve_one, q, None): i for i, q in enumerate(subqueries)
        }
        for future in future_to_index:
            i = future_to_index[future]
            try:
                results[i] = future.result()
            except Exception:
                logger.exception("retrieve() failed for sub-query %r -- treating as empty", subqueries[i])
                results[i] = []
    return results


def _build_claim_diagnostics(verification: VerificationReport, context) -> list[dict]:
    """Per-claim citation/quote diagnostics for TRACE_RAG, computed once here
    (not inside rag_hybrid_search.trace, which must stay pipeline-agnostic).
    """
    rows = []
    for i, cr in enumerate(verification.claim_results, start=1):
        citation_id = cr.claim.citation_ids[0] if cr.claim.citation_ids else None
        chunk_id = context.doc_id_map.get(citation_id) if citation_id else None
        quote = cr.claim.supporting_quote or ""
        chunk_text = chunk_text_for_doc_id(context, citation_id) if citation_id else ""
        start = chunk_text.find(quote) if quote else -1
        end = start + len(quote) if start != -1 else -1
        rows.append({
            "claim_index": i,
            "citation_id": citation_id,
            "chunk_id": chunk_id,
            "quote_length": len(quote),
            "quote_found": start != -1,
            "quote_start_offset": start,
            "quote_end_offset": end,
            "crossed_boundary": cr.failure_reason == "quote_spans_multiple_chunks",
            "failure_reason": cr.failure_reason,
        })
    return rows


class RagPipeline:
    def __init__(
        self, retriever, generation_provider: GenerationProvider, chunk_store=None,
        prompt_version: str = "v2", context_prune_margin: float = 0.3,
    ):
        self._retriever = retriever
        self._generation_provider = generation_provider
        self._chunk_store = chunk_store
        self._prompt_version = prompt_version
        self._context_prune_margin = context_prune_margin

    @property
    def retriever(self):
        return self._retriever

    @property
    def generation_provider(self) -> GenerationProvider:
        return self._generation_provider

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def answer(self, question: str, max_chunks: int = 5, verify: bool = True) -> RagAnswer:
        dev_trace = RequestTrace(question, {
            "Generation": type(self._generation_provider).__name__,
            "Max Chunks": max_chunks,
            "Verify": verify,
            "Prompt Version": self._prompt_version,
        })

        comparative = is_comparative_query(question)
        decompose_capture: dict = {}
        subqueries = (
            decompose_query(question, self._generation_provider, capture=decompose_capture)
            if comparative else [question]
        )

        def _retrieve_one(q: str, trace_for_call):
            if self._chunk_store is not None:
                return route_query(q, self._chunk_store, self._retriever, dev_trace=trace_for_call)[0]
            return self._retriever.retrieve(q, dev_trace=trace_for_call)[0]

        if len(subqueries) == 1:
            results_per_query = [_retrieve_one(subqueries[0], dev_trace)]
        else:
            results_per_query = _retrieve_subqueries_concurrently(subqueries, _retrieve_one)

        concepts_retrieved = sum(1 for results in results_per_query if results)
        dev_trace.log_query_decomposition(
            comparative, subqueries, decompose_capture.get("raw"), concepts_retrieved,
        )
        retrieved_chunks = _merge_multi_query_results(results_per_query)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]
        pruned_chunks = prune_by_score_margin(retrieved_chunks, self._context_prune_margin)
        dev_trace.log_pruning(retrieved_chunks, pruned_chunks)
        retrieved_chunks = pruned_chunks

        context = build_context(retrieved_chunks)
        prompt = build_prompt(question, context, prompt_version=self._prompt_version)
        dev_trace.log_prompt(prompt)

        try:
            gen_start = time.perf_counter()
            raw_output = self._generation_provider.generate(prompt)
            gen_latency_ms = (time.perf_counter() - gen_start) * 1000
            dev_trace.log_generation(
                type(self._generation_provider).__name__,
                getattr(self._generation_provider, "model_name", "unknown"),
                raw_output, gen_latency_ms,
            )
        except Exception as e:
            dev_trace.finish()
            return RagAnswer(
                answer=None, citations=[], confidence=_ZERO_CONFIDENCE,
                verification=_EMPTY_VERIFICATION, error=str(e),
            )

        draft, parse_error = self._parse_draft(raw_output)
        draft = draft.model_copy(update={"claims": extract_supporting_quotes(draft.claims, context)})
        dev_trace.log_parse(
            success=parse_error is None,
            claims_count=len(draft.claims),
            quotes_count=sum(1 for c in draft.claims if c.supporting_quote),
            error=parse_error,
        )

        if verify:
            verification = verify_citations(draft, context)
            confidence = score_confidence(retrieved_chunks, verification, context)
        else:
            verification = _EMPTY_VERIFICATION
            confidence = ConfidenceScores(
                retrieval=score_confidence(retrieved_chunks, _EMPTY_VERIFICATION, context).retrieval,
                citations=0.0, coverage=0.0, overall=0.0,
            )
        dev_trace.log_verification(verification)
        dev_trace.log_claim_diagnostics(_build_claim_diagnostics(verification, context))
        dev_trace.log_confidence(confidence)

        citations = sorted({cid for c in draft.claims for cid in c.citation_ids})
        inline_ids, citations_ok = _inline_citation_drift(draft.answer, set(citations))

        citation_status = CitationStatus.OK
        if any(not cr.passed for cr in verification.claim_results):
            citation_status = CitationStatus.VERIFICATION_FAILED
        elif not citations_ok:
            citation_status = CitationStatus.INLINE_DRIFT
        dev_trace.log_citation_check(inline_ids, citations, citation_status.value)

        structured_citations = build_citations(retrieved_chunks, self._filename_by_doc_id())
        documents_used = len({r.chunk.document_id for r in retrieved_chunks})
        dev_trace.log_summary(draft.answer, chunks_used=len(retrieved_chunks), documents_used=documents_used)
        dev_trace.finish()

        return RagAnswer(
            answer=draft.answer, citations=citations, structured_citations=structured_citations,
            confidence=confidence, verification=verification, citation_status=citation_status,
            error=parse_error,
        )

    def answer_stream(self, question: str, max_chunks: int = 5, verify: bool = True):
        """Stream the raw generation as text deltas, then yield the final verified RagAnswer.

        Citation verification and confidence scoring need the complete
        generation text (they check whether each claim's supporting_quote
        appears verbatim in the retrieved context), so they can't run
        incrementally -- streaming here only removes the "frozen spinner"
        wait for the LLM call itself. Yields ``("delta", str)`` tuples while
        text arrives, then exactly one ``("final", RagAnswer)`` tuple once
        parsing/verification/confidence scoring complete.
        """
        dev_trace = RequestTrace(question, {
            "Generation": type(self._generation_provider).__name__,
            "Max Chunks": max_chunks,
            "Verify": verify,
            "Prompt Version": self._prompt_version,
            "Streaming": True,
        })

        comparative = is_comparative_query(question)
        decompose_capture: dict = {}
        subqueries = (
            decompose_query(question, self._generation_provider, capture=decompose_capture)
            if comparative else [question]
        )

        def _retrieve_one(q: str, trace_for_call):
            if self._chunk_store is not None:
                return route_query(q, self._chunk_store, self._retriever, dev_trace=trace_for_call)[0]
            return self._retriever.retrieve(q, dev_trace=trace_for_call)[0]

        if len(subqueries) == 1:
            results_per_query = [_retrieve_one(subqueries[0], dev_trace)]
        else:
            results_per_query = _retrieve_subqueries_concurrently(subqueries, _retrieve_one)

        concepts_retrieved = sum(1 for results in results_per_query if results)
        dev_trace.log_query_decomposition(
            comparative, subqueries, decompose_capture.get("raw"), concepts_retrieved,
        )
        retrieved_chunks = _merge_multi_query_results(results_per_query)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]
        pruned_chunks = prune_by_score_margin(retrieved_chunks, self._context_prune_margin)
        dev_trace.log_pruning(retrieved_chunks, pruned_chunks)
        retrieved_chunks = pruned_chunks

        context = build_context(retrieved_chunks)
        prompt = build_prompt(question, context, prompt_version=self._prompt_version)
        dev_trace.log_prompt(prompt)

        chunks: list[str] = []
        try:
            gen_start = time.perf_counter()
            for delta in self._generation_provider.generate_stream(prompt):
                chunks.append(delta)
                yield ("delta", delta)
            gen_latency_ms = (time.perf_counter() - gen_start) * 1000
        except Exception as e:
            dev_trace.finish()
            yield (
                "final",
                RagAnswer(
                    answer=None, citations=[], confidence=_ZERO_CONFIDENCE,
                    verification=_EMPTY_VERIFICATION, error=str(e),
                ),
            )
            return

        raw_output = "".join(chunks)
        dev_trace.log_generation(
            type(self._generation_provider).__name__,
            getattr(self._generation_provider, "model_name", "unknown"),
            raw_output, gen_latency_ms,
        )
        draft, parse_error = self._parse_draft(raw_output)
        draft = draft.model_copy(update={"claims": extract_supporting_quotes(draft.claims, context)})
        dev_trace.log_parse(
            success=parse_error is None,
            claims_count=len(draft.claims),
            quotes_count=sum(1 for c in draft.claims if c.supporting_quote),
            error=parse_error,
        )

        if verify:
            verification = verify_citations(draft, context)
            confidence = score_confidence(retrieved_chunks, verification, context)
        else:
            verification = _EMPTY_VERIFICATION
            confidence = ConfidenceScores(
                retrieval=score_confidence(retrieved_chunks, _EMPTY_VERIFICATION, context).retrieval,
                citations=0.0, coverage=0.0, overall=0.0,
            )
        dev_trace.log_verification(verification)
        dev_trace.log_claim_diagnostics(_build_claim_diagnostics(verification, context))
        dev_trace.log_confidence(confidence)

        citations = sorted({cid for c in draft.claims for cid in c.citation_ids})
        inline_ids, citations_ok = _inline_citation_drift(draft.answer, set(citations))

        citation_status = CitationStatus.OK
        if any(not cr.passed for cr in verification.claim_results):
            citation_status = CitationStatus.VERIFICATION_FAILED
        elif not citations_ok:
            citation_status = CitationStatus.INLINE_DRIFT
        dev_trace.log_citation_check(inline_ids, citations, citation_status.value)

        structured_citations = build_citations(retrieved_chunks, self._filename_by_doc_id())
        documents_used = len({r.chunk.document_id for r in retrieved_chunks})
        dev_trace.log_summary(draft.answer, chunks_used=len(retrieved_chunks), documents_used=documents_used)
        dev_trace.finish()

        yield (
            "final",
            RagAnswer(
                answer=draft.answer, citations=citations, structured_citations=structured_citations,
                confidence=confidence, verification=verification, citation_status=citation_status,
                error=parse_error,
            ),
        )

    def _filename_by_doc_id(self) -> dict[str, str]:
        if self._chunk_store is None:
            return {}
        return {
            s["document_id"]: Path(s["source_path"]).name
            for s in self._chunk_store.get_document_summaries()
            if s["source_path"]
        }

    def _parse_draft(self, raw_output: str) -> tuple[RagAnswerDraft, str | None]:
        metadata = GenerationMetadata(
            provider=type(self._generation_provider).__name__,
            model="unknown",
            prompt_version=self._prompt_version,
            generated_at=datetime.now(timezone.utc),
        )
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_repair_unescaped_quotes(raw_output))
            except json.JSONDecodeError as e:
                degraded = RagAnswerDraft(answer=raw_output, claims=[], metadata=metadata)
                return degraded, f"failed to parse structured generation output: {e}"

        try:
            claims = [Claim(**c) for c in parsed.get("claims", [])]
            draft = RagAnswerDraft(answer=parsed["answer"], claims=claims, metadata=metadata)
            return draft, None
        except (KeyError, ValidationError, TypeError) as e:
            degraded = RagAnswerDraft(answer=raw_output, claims=[], metadata=metadata)
            return degraded, f"failed to parse structured generation output: {e}"
