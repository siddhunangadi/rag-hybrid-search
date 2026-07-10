import re
from dataclasses import dataclass
from typing import Literal

from rag_hybrid_search.models import RetrievalTrace, RetrievedChunk
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.retrieval.retriever import HybridRetriever

QueryKind = Literal["structured", "metadata", "semantic", "mixed"]

_CLAUSE_REF_RE = re.compile(r"\bclause\s+([\d.]+(?:\([a-z]\))?)", re.IGNORECASE)
_ARTICLE_REF_RE = re.compile(r"\barticle\s+(\d+[A-Za-z]?)", re.IGNORECASE)
_SECTION_REF_RE = re.compile(r"\bsection\s+([\d.]+)", re.IGNORECASE)

_KNOWN_REGULATIONS = {"GDPR", "HIPAA", "SOC2", "PCI-DSS", "ISO27001", "CCPA"}
_KNOWN_JURISDICTIONS = {"EU", "US", "UK", "INDIA"}

# Words that signal the user wants more than a bare lookup — pushes a
# "structured" match into "mixed" instead.
_ELABORATION_WORDS = {"explain", "what", "why", "how", "summarize", "does", "say", "mean"}


@dataclass
class QueryIntent:
    kind: QueryKind
    filters: dict[str, str]


def classify_query(question: str) -> QueryIntent:
    """Classify a question into structured/metadata/semantic/mixed intent.

    - structured: a bare clause-level reference with no elaboration words.
    - mixed: a clause-level reference plus additional intent/elaboration.
    - metadata: a regulation/jurisdiction scope filter with no clause reference.
    - semantic: none of the above — full existing hybrid pipeline, unchanged.
    """
    filters: dict[str, str] = {}

    clause_match = _CLAUSE_REF_RE.search(question)
    article_match = _ARTICLE_REF_RE.search(question)
    section_match = _SECTION_REF_RE.search(question)

    if clause_match:
        filters["clause"] = clause_match.group(1)
    elif article_match:
        filters["article"] = article_match.group(1)
    elif section_match:
        filters["section"] = section_match.group(1)

    if filters:
        has_elaboration = any(
            re.search(rf"\b{word}\b", question, re.IGNORECASE) for word in _ELABORATION_WORDS
        )
        return QueryIntent(kind="mixed" if has_elaboration else "structured", filters=filters)

    upper_question = question.upper()
    for regulation in _KNOWN_REGULATIONS:
        if regulation in upper_question:
            return QueryIntent(kind="metadata", filters={"regulation": regulation})
    for jurisdiction in _KNOWN_JURISDICTIONS:
        if re.search(rf"\b{jurisdiction}\b", upper_question):
            return QueryIntent(kind="metadata", filters={"jurisdiction": jurisdiction})

    return QueryIntent(kind="semantic", filters={})


def route_query(
    question: str, chunk_store: ChunkStore, retriever: HybridRetriever, dev_trace=None
) -> tuple[list[RetrievedChunk], RetrievalTrace]:
    """Route a question to the retrieval path matching its classified intent.

    structured -> metadata filter only, no retriever call.
    metadata   -> metadata filter, then existing hybrid retrieval unchanged
                  (v1 simplification: metadata-only intent still runs the
                  full retriever; scoping the retriever's candidate set to
                  the filtered chunks is deferred, see spec open questions).
    semantic   -> existing hybrid retrieval, completely unchanged.
    mixed      -> metadata filter narrows candidates conceptually; v1 runs
                  the existing retriever and filters its output down to
                  matching chunk_ids, since HybridRetriever has no
                  candidate-subset parameter yet.
    """
    intent = classify_query(question)
    trace = RetrievalTrace()

    if intent.kind == "structured":
        matched = chunk_store.get_by_legal_metadata(intent.filters)
        results = [
            RetrievedChunk(chunk=chunk, rrf_score=1.0, final_rank=i)
            for i, chunk in enumerate(matched)
        ]
        return results, trace

    if intent.kind in ("metadata", "mixed"):
        matched_ids = {c.chunk_id for c in chunk_store.get_by_legal_metadata(intent.filters)}
        results, trace = retriever.retrieve(question, dev_trace=dev_trace)
        results = [r for r in results if r.chunk.chunk_id in matched_ids]
        return results, trace

    return retriever.retrieve(question, dev_trace=dev_trace)
