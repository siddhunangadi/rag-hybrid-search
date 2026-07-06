from rag_hybrid_search.compliance.query_router import QueryIntent, classify_query


def test_classifies_structured_article_reference():
    intent = classify_query("Show Article 17")
    assert intent.kind == "structured"
    assert intent.filters == {"article": "17"}


def test_classifies_structured_clause_reference():
    intent = classify_query("What does clause 5.2(a) say?")
    assert intent.kind == "mixed"
    assert intent.filters == {"clause": "5.2(a)"}


def test_classifies_metadata_only_scope():
    intent = classify_query("Show only HIPAA documents")
    assert intent.kind == "metadata"
    assert intent.filters == {"regulation": "HIPAA"}


def test_classifies_jurisdiction_metadata_scope():
    intent = classify_query("only EU regulations")
    assert intent.kind == "metadata"
    assert intent.filters == {"jurisdiction": "EU"}


def test_classifies_pure_semantic_query():
    intent = classify_query("What is the purpose of data minimization?")
    assert intent.kind == "semantic"
    assert intent.filters == {}


def test_classifies_mixed_query_with_intent_beyond_lookup():
    intent = classify_query("Explain Article 17 in plain terms")
    assert intent.kind == "mixed"
    assert intent.filters == {"article": "17"}
