from datetime import date

from rag_hybrid_search.compliance.regulation_models import (
    Citation,
    ClauseParseResult,
    ClauseSpan,
    LegalMetadata,
)


def test_legal_metadata_all_fields_optional():
    meta = LegalMetadata(document_id="doc-1", document_title="GDPR")
    assert meta.regulation is None
    assert meta.version is None
    assert meta.jurisdiction is None
    assert meta.article is None
    assert meta.section is None
    assert meta.clause is None
    assert meta.effective_date is None
    assert meta.document_type is None
    assert meta.page is None
    assert meta.document_id == "doc-1"
    assert meta.document_title == "GDPR"


def test_legal_metadata_typed_effective_date():
    meta = LegalMetadata(
        document_id="doc-1",
        document_title="GDPR",
        effective_date=date(2018, 5, 25),
    )
    assert meta.effective_date == date(2018, 5, 25)


def test_clause_parse_result_defaults_no_fallback():
    result = ClauseParseResult(
        clauses=[
            ClauseSpan(
                text="Personal data shall be processed lawfully.",
                metadata=LegalMetadata(document_id="doc-1", document_title="GDPR", article="5"),
            )
        ],
        confidence=0.92,
        parser="regex",
    )
    assert result.fallback_used is False
    assert result.parser == "regex"
    assert len(result.clauses) == 1


def test_citation_construction_and_display():
    citation = Citation(
        citation_id="0198c1a2-0000-7000-8000-000000000001",
        regulation="GDPR",
        version="2018",
        jurisdiction="EU",
        article="17",
        section="3",
        clause="17.3(b)",
        page=42,
        document_id="doc-1",
        document_title="GDPR Consolidated Text",
        document_type="regulation",
        effective_date=date(2018, 5, 25),
        chunk_id="chunk-1",
        confidence=0.9,
        display="GDPR Art. 17(3)(b), p.42",
    )
    assert citation.citation_id == "0198c1a2-0000-7000-8000-000000000001"
    assert citation.display == "GDPR Art. 17(3)(b), p.42"


def test_citation_missing_fields_are_optional():
    citation = Citation(
        citation_id="0198c1a2-0000-7000-8000-000000000002",
        document_id="doc-2",
        document_title="internal_notes.txt",
        chunk_id="chunk-2",
        confidence=0.5,
        display="internal_notes.txt, chunk 2",
    )
    assert citation.regulation is None
    assert citation.article is None
