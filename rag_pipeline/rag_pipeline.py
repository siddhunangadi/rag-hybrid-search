import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from rag_hybrid_search.compliance.citation_mapper import build_citations
from rag_hybrid_search.compliance.query_router import route_query
from rag_hybrid_search.trace import RequestTrace
from rag_pipeline.confidence_scorer import score_confidence
from rag_pipeline.context_builder import build_context
from rag_pipeline.generation_provider import GenerationProvider
from rag_pipeline.citation_verifier import verify_citations
from rag_pipeline.models import (
    Claim,
    ConfidenceScores,
    GenerationMetadata,
    RagAnswer,
    RagAnswerDraft,
    VerificationReport,
)
from rag_pipeline.prompt_builder import build_prompt

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


class RagPipeline:
    def __init__(self, retriever, generation_provider: GenerationProvider, chunk_store=None, prompt_version: str = "v1"):
        self._retriever = retriever
        self._generation_provider = generation_provider
        self._chunk_store = chunk_store
        self._prompt_version = prompt_version

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

        if self._chunk_store is not None:
            retrieved_chunks, _trace = route_query(question, self._chunk_store, self._retriever, dev_trace=dev_trace)
        else:
            retrieved_chunks, _trace = self._retriever.retrieve(question, dev_trace=dev_trace)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]

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
        dev_trace.log_confidence(confidence)

        citations = sorted({cid for c in draft.claims for cid in c.citation_ids})
        structured_citations = build_citations(retrieved_chunks, self._filename_by_doc_id())
        documents_used = len({r.chunk.document_id for r in retrieved_chunks})
        dev_trace.log_summary(draft.answer, chunks_used=len(retrieved_chunks), documents_used=documents_used)
        dev_trace.finish()

        return RagAnswer(
            answer=draft.answer, citations=citations, structured_citations=structured_citations,
            confidence=confidence, verification=verification, error=parse_error,
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
        if self._chunk_store is not None:
            retrieved_chunks, _trace = route_query(question, self._chunk_store, self._retriever)
        else:
            retrieved_chunks, _trace = self._retriever.retrieve(question)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]

        context = build_context(retrieved_chunks)
        prompt = build_prompt(question, context, prompt_version=self._prompt_version)

        chunks: list[str] = []
        try:
            for delta in self._generation_provider.generate_stream(prompt):
                chunks.append(delta)
                yield ("delta", delta)
        except Exception as e:
            yield (
                "final",
                RagAnswer(
                    answer=None, citations=[], confidence=_ZERO_CONFIDENCE,
                    verification=_EMPTY_VERIFICATION, error=str(e),
                ),
            )
            return

        raw_output = "".join(chunks)
        draft, parse_error = self._parse_draft(raw_output)

        if verify:
            verification = verify_citations(draft, context)
            confidence = score_confidence(retrieved_chunks, verification, context)
        else:
            verification = _EMPTY_VERIFICATION
            confidence = ConfidenceScores(
                retrieval=score_confidence(retrieved_chunks, _EMPTY_VERIFICATION, context).retrieval,
                citations=0.0, coverage=0.0, overall=0.0,
            )

        citations = sorted({cid for c in draft.claims for cid in c.citation_ids})
        structured_citations = build_citations(retrieved_chunks, self._filename_by_doc_id())

        yield (
            "final",
            RagAnswer(
                answer=draft.answer, citations=citations, structured_citations=structured_citations,
                confidence=confidence, verification=verification, error=parse_error,
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
