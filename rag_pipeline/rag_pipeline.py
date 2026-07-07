import json
from datetime import datetime, timezone

from pydantic import ValidationError

from rag_hybrid_search.compliance.citation_mapper import build_citations
from rag_hybrid_search.compliance.query_router import route_query
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
        if self._chunk_store is not None:
            retrieved_chunks, _trace = route_query(question, self._chunk_store, self._retriever)
        else:
            retrieved_chunks, _trace = self._retriever.retrieve(question)
        retrieved_chunks = sorted(retrieved_chunks, key=lambda r: r.final_rank)[:max_chunks]

        context = build_context(retrieved_chunks)
        prompt = build_prompt(question, context, prompt_version=self._prompt_version)

        try:
            raw_output = self._generation_provider.generate(prompt)
        except Exception as e:
            return RagAnswer(
                answer=None, citations=[], confidence=_ZERO_CONFIDENCE,
                verification=_EMPTY_VERIFICATION, error=str(e),
            )

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
        structured_citations = build_citations(retrieved_chunks)

        return RagAnswer(
            answer=draft.answer, citations=citations, structured_citations=structured_citations,
            confidence=confidence, verification=verification, error=parse_error,
        )

    def _parse_draft(self, raw_output: str) -> tuple[RagAnswerDraft, str | None]:
        metadata = GenerationMetadata(
            provider=type(self._generation_provider).__name__,
            model="unknown",
            prompt_version=self._prompt_version,
            generated_at=datetime.now(timezone.utc),
        )
        try:
            parsed = json.loads(raw_output)
            claims = [Claim(**c) for c in parsed.get("claims", [])]
            draft = RagAnswerDraft(answer=parsed["answer"], claims=claims, metadata=metadata)
            return draft, None
        except (json.JSONDecodeError, KeyError, ValidationError, TypeError) as e:
            degraded = RagAnswerDraft(answer=raw_output, claims=[], metadata=metadata)
            return degraded, f"failed to parse structured generation output: {e}"
