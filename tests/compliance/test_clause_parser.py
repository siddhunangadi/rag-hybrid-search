from pathlib import Path

from rag_hybrid_search.compliance.clause_parser import parse_clauses

_FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def test_gdpr_style_detects_articles():
    result = parse_clauses(_read("gdpr_style.txt"), document_id="doc-1", document_title="GDPR")
    articles = {c.metadata.article for c in result.clauses}
    assert "5" in articles
    assert "17" in articles
    assert result.parser == "regex"
    assert result.fallback_used is False


def test_gdpr_style_high_confidence():
    result = parse_clauses(_read("gdpr_style.txt"), document_id="doc-1", document_title="GDPR")
    assert result.confidence >= 0.7


def test_hipaa_style_detects_sections():
    result = parse_clauses(_read("hipaa_style.txt"), document_id="doc-2", document_title="HIPAA")
    sections = {c.metadata.section for c in result.clauses}
    assert "164.502" in sections
    assert "164.508" in sections


def test_unstructured_document_low_confidence_single_clause():
    result = parse_clauses(_read("unstructured.txt"), document_id="doc-3", document_title="Memo")
    assert result.confidence < 0.4
    assert len(result.clauses) == 1
    assert result.clauses[0].metadata.article is None


def test_empty_text_returns_empty_result():
    result = parse_clauses("", document_id="doc-4", document_title="Empty")
    assert result.clauses == []
    assert result.confidence == 0.0
