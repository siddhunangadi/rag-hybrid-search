from rag_hybrid_search.compliance.citation_mapper import build_citations
from rag_hybrid_search.compliance.regulation_models import LegalMetadata
from rag_hybrid_search.models import Chunk, RetrievedChunk


def _retrieved(chunk_id: str, legal_metadata=None, rerank_score=0.9) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id, document_id="doc-1", chunk_index=0,
        text="Personal data shall be processed lawfully.",
        strategy_version="clause-v1", char_count=42, page=42,
        legal_metadata=legal_metadata,
    )
    return RetrievedChunk(chunk=chunk, rrf_score=1.0, rerank_score=rerank_score, final_rank=0)


def test_build_citation_for_legal_chunk():
    legal_metadata = LegalMetadata(
        document_id="doc-1", document_title="GDPR Consolidated Text",
        regulation="GDPR", article="17", clause="17.3(b)", page=42,
    )
    citations = build_citations([_retrieved("c1", legal_metadata=legal_metadata)])
    assert len(citations) == 1
    citation = citations[0]
    assert citation.regulation == "GDPR"
    assert citation.article == "17"
    assert citation.display == "GDPR Art. 17(3)(b), p.42"
    assert citation.confidence == 0.9
    assert citation.chunk_id == "c1"


def test_build_citation_for_non_legal_chunk_degrades_gracefully():
    citations = build_citations([_retrieved("c2", legal_metadata=None)])
    assert len(citations) == 1
    citation = citations[0]
    assert citation.regulation is None
    assert "doc-1" in citation.display or "c2" in citation.display


def test_citation_ids_are_unique_across_calls():
    legal_metadata = LegalMetadata(document_id="doc-1", document_title="GDPR", article="5")
    citations_1 = build_citations([_retrieved("c3", legal_metadata=legal_metadata)])
    citations_2 = build_citations([_retrieved("c3", legal_metadata=legal_metadata)])
    assert citations_1[0].citation_id != citations_2[0].citation_id


def test_missing_rerank_score_defaults_confidence_to_zero():
    citations = build_citations([_retrieved("c4", rerank_score=None)])
    assert citations[0].confidence == 0.0
