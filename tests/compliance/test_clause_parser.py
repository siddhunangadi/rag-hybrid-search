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


def test_unstructured_document_sets_fallback_used():
    result = parse_clauses(_read("unstructured.txt"), document_id="doc-3", document_title="Memo")
    assert result.fallback_used is True


def test_hipaa_style_splits_lettered_subclauses():
    result = parse_clauses(_read("hipaa_style.txt"), document_id="doc-2", document_title="HIPAA")
    section_502_clauses = [c for c in result.clauses if c.metadata.section == "164.502"]
    assert len(section_502_clauses) == 2
    clause_ids = {c.metadata.clause for c in section_502_clauses}
    assert len(clause_ids) == 2
    assert "Standard" in section_502_clauses[0].text
    assert "Implementation specification" in section_502_clauses[1].text


def test_gdpr_style_splits_lettered_subclauses_nested_under_numbered_clause():
    result = parse_clauses(_read("gdpr_style.txt"), document_id="doc-1", document_title="GDPR")
    article_17_clauses = [c for c in result.clauses if c.metadata.article == "17"]
    lettered = [c for c in article_17_clauses if c.metadata.clause and "(" in c.metadata.clause]
    assert len(lettered) == 2
    clause_ids = {c.metadata.clause for c in lettered}
    assert len(clause_ids) == 2
    for cid in clause_ids:
        assert cid.startswith("17.3")


def test_heading_type_switch_resets_stale_state():
    text = (
        "Article 5\n\n"
        "1. Some article-level clause text.\n\n"
        "Section 164.502\n\n"
        "1. Some section-level clause text.\n"
    )
    result = parse_clauses(text, document_id="doc-5", document_title="Mixed")
    section_clauses = [c for c in result.clauses if c.metadata.section == "164.502"]
    assert section_clauses
    assert all(c.metadata.article is None for c in section_clauses)

    article_clauses = [c for c in result.clauses if c.metadata.article == "5"]
    assert article_clauses
    assert all(c.metadata.section is None for c in article_clauses)
